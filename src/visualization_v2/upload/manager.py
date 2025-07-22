"""Upload manager that orchestrates Pinata and Cloudflare KV operations."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import structlog
from pydantic import BaseModel

from models.config import UploadConfig
from models.sensor import SensorType
from models.visualization import Interval, VisualizationResult
from .pinata import PinataUploader
from .cloudflare_kv import CloudflareKVClient

logger = structlog.get_logger()


class UploadResult(BaseModel):
    """Result of a complete upload operation (Pinata + Cloudflare KV)."""
    success: bool
    file_path: Path
    sensor_type: SensorType
    interval: Interval
    cid: Optional[str] = None
    ipfs_url: Optional[str] = None
    kv_key: Optional[str] = None
    local_file_deleted: bool = False
    error_message: Optional[str] = None
    pinata_error: Optional[str] = None
    cloudflare_error: Optional[str] = None


class UploadManager:
    """Manager for uploading visualization images to external services."""
    
    def __init__(self, config: UploadConfig):
        """Initialize the upload manager."""
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate the upload configuration."""
        if not self.config.enabled:
            logger.info("Upload functionality is disabled")
            return
            
        if not self.config.pinata:
            raise ValueError("Pinata configuration is required when uploads are enabled")
            
        if not self.config.cloudflare_kv:
            raise ValueError("Cloudflare KV configuration is required when uploads are enabled")
            
        if not self.config.pinata.jwt:
            raise ValueError("Pinata JWT is required")
            
        if not self.config.cloudflare_kv.api_token:
            raise ValueError("Cloudflare API token is required")
            
        if not self.config.cloudflare_kv.namespace_id:
            raise ValueError("Cloudflare KV namespace ID is required")
    
    async def upload_visualization(
        self,
        result: VisualizationResult
    ) -> UploadResult:
        """Upload a visualization result to external services.
        
        Args:
            result: The visualization result to upload
            
        Returns:
            UploadResult with details of the upload operation
        """
        if not self.config.enabled:
            return UploadResult(
                success=False,
                file_path=result.output_path,
                sensor_type=result.request.sensor_type,
                interval=result.request.interval,
                error_message="Upload functionality is disabled"
            )
        
        if not result.success or not result.output_path:
            return UploadResult(
                success=False,
                file_path=result.output_path or Path("unknown"),
                sensor_type=result.request.sensor_type,
                interval=result.request.interval,
                error_message="Visualization generation was not successful"
            )
        
        file_path = result.output_path
        sensor_type = result.request.sensor_type
        interval = result.request.interval
        
        logger.info(
            "Starting upload process",
            file_path=str(file_path),
            sensor_type=sensor_type.value,
            interval=interval.value
        )
        
        upload_result = UploadResult(
            success=False,
            file_path=file_path,
            sensor_type=sensor_type,
            interval=interval
        )
        
        try:
            # Step 1: Upload to Pinata IPFS
            async with PinataUploader(self.config.pinata) as pinata:
                # Create a clean filename: sensor-imagetype-date.png (avoiding duplication)
                date_str = result.request.start_time.strftime('%Y-%m-%d')
                
                # Add hour for hourly visualizations for uniqueness
                if interval == Interval.HOURLY:
                    hour_str = result.request.start_time.strftime('%H')
                    filename = f"{sensor_type.value}-{interval.value}-{date_str}-{hour_str}.png"
                elif interval == Interval.CUMULATIVE:
                    # For cumulative, add start and end time for uniqueness
                    start_time_str = result.request.start_time.strftime('%H-%M')
                    end_time_str = result.request.end_time.strftime('%H-%M')
                    filename = f"{sensor_type.value}-{interval.value}-{date_str}-{start_time_str}-to-{end_time_str}.png"
                else:  # DAILY
                    filename = f"{sensor_type.value}-{interval.value}-{date_str}.png"
                
                pinata_result = await pinata.upload_file(file_path, filename=filename)
                
                if not pinata_result.success:
                    upload_result.pinata_error = pinata_result.error_message
                    upload_result.error_message = f"Pinata upload failed: {pinata_result.error_message}"
                    return upload_result
                
                upload_result.cid = pinata_result.cid
                upload_result.ipfs_url = pinata_result.ipfs_url
            
            # Step 2: Store CID in Cloudflare KV
            async with CloudflareKVClient(self.config.cloudflare_kv) as kv_client:
                metadata = {
                    'filename': file_path.name,
                    'sensor_type': sensor_type.value,
                    'interval': interval.value,
                    'data_points': result.data_points,
                    'processing_time': result.processing_time_seconds,
                    'start_time': result.request.start_time.isoformat(),
                    'end_time': result.request.end_time.isoformat(),
                }
                
                kv_result = await kv_client.set_latest_image(
                    sensor_type, 
                    interval,
                    upload_result.ipfs_url,
                    metadata
                )
                
                if not kv_result.success:
                    upload_result.cloudflare_error = kv_result.error_message
                    upload_result.error_message = f"Cloudflare KV update failed: {kv_result.error_message}"
                    return upload_result
                
                upload_result.kv_key = kv_result.key
            
            # Step 3: Delete local file if configured
            if self.config.delete_local:
                try:
                    file_path.unlink()
                    upload_result.local_file_deleted = True
                    logger.info(
                        "Local file deleted after successful upload",
                        file_path=str(file_path)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to delete local file after upload",
                        file_path=str(file_path),
                        error=str(e)
                    )
                    # Don't fail the entire operation for this
            
            upload_result.success = True
            
            logger.info(
                "Upload process completed successfully",
                file_path=str(file_path),
                cid=upload_result.cid,
                kv_key=upload_result.kv_key,
                local_deleted=upload_result.local_file_deleted
            )
            
            return upload_result
            
        except Exception as e:
            upload_result.error_message = f"Unexpected error during upload: {str(e)}"
            logger.error(
                "Unexpected error during upload",
                file_path=str(file_path),
                error=str(e)
            )
            return upload_result
    
    async def upload_multiple(
        self,
        results: List[VisualizationResult],
        max_concurrent: int = 3
    ) -> List[UploadResult]:
        """Upload multiple visualization results with concurrency control.
        
        Args:
            results: List of visualization results to upload
            max_concurrent: Maximum number of concurrent uploads
            
        Returns:
            List of upload results in the same order as input
        """
        if not self.config.enabled:
            logger.info("Upload functionality is disabled, skipping batch upload")
            return [
                UploadResult(
                    success=False,
                    file_path=result.output_path or Path("unknown"),
                    sensor_type=result.request.sensor_type,
                    interval=result.request.interval,
                    error_message="Upload functionality is disabled"
                )
                for result in results
            ]
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def upload_with_semaphore(result: VisualizationResult) -> UploadResult:
            async with semaphore:
                return await self.upload_visualization(result)
        
        logger.info(
            "Starting batch upload",
            count=len(results),
            max_concurrent=max_concurrent
        )
        
        tasks = [upload_with_semaphore(result) for result in results]
        upload_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert any exceptions to failed UploadResults
        final_results = []
        for i, upload_result in enumerate(upload_results):
            if isinstance(upload_result, Exception):
                result = results[i]
                final_results.append(UploadResult(
                    success=False,
                    file_path=result.output_path or Path("unknown"),
                    sensor_type=result.request.sensor_type,
                    interval=result.request.interval,
                    error_message=f"Exception during upload: {str(upload_result)}"
                ))
            else:
                final_results.append(upload_result)
        
        # Log summary
        successful = sum(1 for r in final_results if r.success)
        failed = len(final_results) - successful
        
        logger.info(
            "Batch upload completed",
            total=len(results),
            successful=successful,
            failed=failed
        )
        
        return final_results
    
    def is_enabled(self) -> bool:
        """Check if upload functionality is enabled."""
        return self.config.enabled 