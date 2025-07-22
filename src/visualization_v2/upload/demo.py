#!/usr/bin/env python3
"""Demo script to test the upload functionality."""

import asyncio
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
from PIL import Image
import structlog

# Setup basic logging for demo
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

from models.config import UploadConfig, PinataConfig, CloudflareKVConfig
from models.sensor import SensorType
from models.visualization import Interval, VisualizationRequest, VisualizationResult
from upload.manager import UploadManager

logger = structlog.get_logger()


async def create_test_image() -> Path:
    """Create a test PNG image for upload."""
    # Create a simple test image
    image = Image.new('RGB', (100, 100), color='red')
    
    # Save to temporary file
    temp_file = Path(tempfile.mktemp(suffix='.png'))
    image.save(temp_file, 'PNG')
    
    logger.info(f"Created test image: {temp_file}")
    return temp_file


async def test_upload_functionality():
    """Test the complete upload functionality."""
    logger.info("Starting upload functionality demo")
    
    # Check environment variables
    required_vars = [
        "PINATA_JWT",
        "CLOUDFLARE_API_TOKEN", 
        "CLOUDFLARE_ACCOUNT_ID",
        "KV_NAMESPACE_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(
            "Missing required environment variables",
            missing=missing_vars,
            help="Please set these variables in your .env file"
        )
        return
    
    # Create upload configuration
    upload_config = UploadConfig.from_env()
    upload_config.delete_local = False  # Keep files for demo
    
    logger.info(
        "Upload configuration loaded",
        enabled=upload_config.enabled,
        pinata_configured=upload_config.pinata is not None,
        cloudflare_configured=upload_config.cloudflare_kv is not None
    )
    
    if not upload_config.enabled:
        logger.error("Upload functionality is not enabled")
        return
    
    try:
        # Create test image
        test_image_path = await create_test_image()
        
        # Create mock visualization result
        mock_request = VisualizationRequest(
            sensor_type=SensorType.TEMPERATURE,
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            interval=Interval.DAILY,
            color_scheme="redblue"
        )
        
        mock_result = VisualizationResult(
            request=mock_request,
            output_path=test_image_path,
            success=True,
            data_points=1440,  # 24 hours * 60 minutes
            processing_time_seconds=1.5
        )
        
        # Initialize upload manager
        upload_manager = UploadManager(upload_config)
        
        # Perform upload
        logger.info("Starting upload test")
        upload_result = await upload_manager.upload_visualization(mock_result)
        
        if upload_result.success:
            logger.info(
                "Upload test completed successfully! 🎉",
                cid=upload_result.cid,
                ipfs_url=upload_result.ipfs_url,
                kv_key=upload_result.kv_key,
                local_deleted=upload_result.local_file_deleted
            )
            
            # Verify we can retrieve from Cloudflare KV
            from upload.cloudflare_kv import CloudflareKVClient
            async with CloudflareKVClient(upload_config.cloudflare_kv) as kv_client:
                get_result = await kv_client.get_latest_image(
                    SensorType.TEMPERATURE,
                    Interval.DAILY
                )
                
                if get_result.success:
                    import json
                    data = json.loads(get_result.value)
                    logger.info(
                        "Successfully retrieved from Cloudflare KV",
                        ipfs_url=data.get('ipfs_url'),
                        updated_at=data.get('updated_at')
                    )
                else:
                    logger.warning(
                        "Failed to retrieve from Cloudflare KV",
                        error=get_result.error_message
                    )
        else:
            logger.error(
                "Upload test failed",
                error=upload_result.error_message,
                pinata_error=upload_result.pinata_error,
                cloudflare_error=upload_result.cloudflare_error
            )
        
        # Clean up test file
        if test_image_path.exists():
            test_image_path.unlink()
            logger.info("Cleaned up test file")
            
    except Exception as e:
        logger.error("Unexpected error during demo", error=str(e))


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the demo
    asyncio.run(test_upload_functionality()) 