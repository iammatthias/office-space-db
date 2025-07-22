"""Pinata IPFS uploader service."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
import httpx
from pydantic import BaseModel

from models.config import PinataConfig

logger = structlog.get_logger()


class PinataUploadResult(BaseModel):
    """Result of a Pinata upload operation."""
    success: bool
    cid: Optional[str] = None
    ipfs_url: Optional[str] = None
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


class PinataUploader:
    """Client for uploading files to Pinata IPFS."""
    
    def __init__(self, config: PinataConfig):
        """Initialize the Pinata client."""
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
        
    async def upload_file(
        self,
        file_path: Path,
        filename: Optional[str] = None
    ) -> PinataUploadResult:
        """Upload a file to Pinata IPFS.
        
        Args:
            file_path: Path to the file to upload
            filename: Optional custom filename (defaults to file_path.name)
            
        Returns:
            PinataUploadResult with CID and IPFS URL if successful
        """
        if not file_path.exists():
            return PinataUploadResult(
                success=False,
                error_message=f"File does not exist: {file_path}"
            )
            
        filename = filename or file_path.name
        
        logger.debug(
            "Uploading file to Pinata",
            file_path=str(file_path),
            filename=filename,
            file_size=file_path.stat().st_size
        )
        
        try:
            # Prepare the multipart form data
            with open(file_path, 'rb') as f:
                files = {
                    'file': (filename, f, 'application/octet-stream')
                }
                
                # Add required fields for Pinata API
                files['name'] = (None, filename)  # File name
                files['network'] = (None, 'public')  # Network type - public since file and group are public
                
                # Add group_id as form field if specified
                if self.config.group_id:
                    files['group_id'] = (None, self.config.group_id)
                
                headers = {
                    'Authorization': f'Bearer {self.config.jwt}'
                }
                
                response = await self.client.post(
                    self.config.base_url,
                    headers=headers,
                    files=files
                )
                
            response.raise_for_status()
            response_json = response.json()
            
            # Extract CID from response
            cid = response_json.get('data', {}).get('cid')
            if not cid:
                return PinataUploadResult(
                    success=False,
                    error_message="No CID returned from Pinata",
                    response_data=response_json
                )
            
            ipfs_url = f"ipfs://{cid}"
            
            logger.info(
                "File uploaded to Pinata successfully",
                filename=filename,
                cid=cid,
                ipfs_url=ipfs_url
            )
            
            return PinataUploadResult(
                success=True,
                cid=cid,
                ipfs_url=ipfs_url,
                response_data=response_json
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(
                "HTTP error uploading to Pinata",
                status_code=e.response.status_code,
                response_text=e.response.text,
                filename=filename
            )
            return PinataUploadResult(
                success=False,
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"Error uploading to Pinata: {str(e)}"
            logger.error(
                "Unexpected error uploading to Pinata",
                error=str(e),
                filename=filename
            )
            return PinataUploadResult(
                success=False,
                error_message=error_msg
            )
    
    async def upload_multiple(
        self,
        file_paths: list[Path],
        max_concurrent: int = 3
    ) -> list[PinataUploadResult]:
        """Upload multiple files with concurrency control.
        
        Args:
            file_paths: List of file paths to upload
            max_concurrent: Maximum number of concurrent uploads
            
        Returns:
            List of upload results in the same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def upload_with_semaphore(file_path: Path) -> PinataUploadResult:
            async with semaphore:
                return await self.upload_file(file_path)
        
        tasks = [upload_with_semaphore(path) for path in file_paths]
        return await asyncio.gather(*tasks) 