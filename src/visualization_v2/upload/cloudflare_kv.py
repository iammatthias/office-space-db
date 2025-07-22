"""Cloudflare KV client service."""

from typing import Optional, Dict, Any, Union
import structlog
import httpx
from pydantic import BaseModel

from models.config import CloudflareKVConfig
from models.sensor import SensorType
from models.visualization import Interval

logger = structlog.get_logger()


class CloudflareKVResult(BaseModel):
    """Result of a Cloudflare KV operation."""
    success: bool
    key: Optional[str] = None
    value: Optional[str] = None
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


class CloudflareKVClient:
    """Client for interacting with Cloudflare KV storage."""
    
    def __init__(self, config: CloudflareKVConfig):
        """Initialize the Cloudflare KV client."""
        self.config = config
        self.client = httpx.AsyncClient(timeout=10.0)
        self.base_url = f"{config.base_url}/accounts/{self._get_account_id()}/storage/kv/namespaces/{config.namespace_id}"
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
        
    def _get_account_id(self) -> str:
        """Extract account ID from API token.
        
        For simplicity, we'll use a placeholder here.
        In production, you might want to get this from an env var or API call.
        """
        # This should be obtained from environment or API call
        import os
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID environment variable is required")
        return account_id
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            'Authorization': f'Bearer {self.config.api_token}',
            'Content-Type': 'application/json'
        }
    
    def _create_key(self, sensor_type: SensorType, interval: Interval) -> str:
        """Create a KV key for sensor type and interval.
        
        Args:
            sensor_type: Type of sensor
            interval: Visualization interval
            
        Returns:
            KV key string like "temperature:daily:latest"
        """
        return f"{sensor_type.value}:{interval.value}:latest"
    
    async def set_latest_image(
        self,
        sensor_type: SensorType,
        interval: Interval,
        ipfs_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CloudflareKVResult:
        """Set the latest image IPFS URL for a sensor type and interval.
        
        Args:
            sensor_type: Type of sensor
            interval: Visualization interval
            ipfs_url: IPFS URL (e.g., "ipfs://QmHash...")
            metadata: Optional metadata to store with the URL
            
        Returns:
            CloudflareKVResult indicating success or failure
        """
        key = self._create_key(sensor_type, interval)
        
        # Create value object with IPFS URL and optional metadata
        value_data = {
            'ipfs_url': ipfs_url,
            'updated_at': str(int(__import__('time').time())),  # Unix timestamp
        }
        
        if metadata:
            value_data['metadata'] = metadata
            
        value = __import__('json').dumps(value_data)
        
        logger.debug(
            "Setting latest image in Cloudflare KV",
            key=key,
            ipfs_url=ipfs_url,
            metadata=metadata
        )
        
        try:
            response = await self.client.put(
                f"{self.base_url}/values/{key}",
                headers=self._get_headers(),
                content=value
            )
            
            response.raise_for_status()
            response_data = response.json() if response.content else {}
            
            logger.info(
                "Successfully set latest image in Cloudflare KV",
                key=key,
                ipfs_url=ipfs_url,
                sensor_type=sensor_type.value,
                interval=interval.value
            )
            
            return CloudflareKVResult(
                success=True,
                key=key,
                value=value,
                response_data=response_data
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(
                "HTTP error setting value in Cloudflare KV",
                status_code=e.response.status_code,
                response_text=e.response.text,
                key=key
            )
            return CloudflareKVResult(
                success=False,
                key=key,
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"Error setting value in Cloudflare KV: {str(e)}"
            logger.error(
                "Unexpected error setting value in Cloudflare KV",
                error=str(e),
                key=key
            )
            return CloudflareKVResult(
                success=False,
                key=key,
                error_message=error_msg
            )
    
    async def get_latest_image(
        self,
        sensor_type: SensorType,
        interval: Interval
    ) -> CloudflareKVResult:
        """Get the latest image IPFS URL for a sensor type and interval.
        
        Args:
            sensor_type: Type of sensor
            interval: Visualization interval
            
        Returns:
            CloudflareKVResult with the stored value
        """
        key = self._create_key(sensor_type, interval)
        
        logger.debug(
            "Getting latest image from Cloudflare KV",
            key=key,
            sensor_type=sensor_type.value,
            interval=interval.value
        )
        
        try:
            response = await self.client.get(
                f"{self.base_url}/values/{key}",
                headers=self._get_headers()
            )
            
            if response.status_code == 404:
                logger.debug("Key not found in Cloudflare KV", key=key)
                return CloudflareKVResult(
                    success=False,
                    key=key,
                    error_message="Key not found"
                )
                
            response.raise_for_status()
            value = response.text
            
            logger.debug(
                "Retrieved value from Cloudflare KV",
                key=key,
                value_length=len(value)
            )
            
            return CloudflareKVResult(
                success=True,
                key=key,
                value=value
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(
                "HTTP error getting value from Cloudflare KV",
                status_code=e.response.status_code,
                response_text=e.response.text,
                key=key
            )
            return CloudflareKVResult(
                success=False,
                key=key,
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"Error getting value from Cloudflare KV: {str(e)}"
            logger.error(
                "Unexpected error getting value from Cloudflare KV",
                error=str(e),
                key=key
            )
            return CloudflareKVResult(
                success=False,
                key=key,
                error_message=error_msg
            )
    
    async def delete_key(self, key: str) -> CloudflareKVResult:
        """Delete a key from Cloudflare KV.
        
        Args:
            key: The key to delete
            
        Returns:
            CloudflareKVResult indicating success or failure
        """
        logger.debug("Deleting key from Cloudflare KV", key=key)
        
        try:
            response = await self.client.delete(
                f"{self.base_url}/values/{key}",
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            
            logger.info("Successfully deleted key from Cloudflare KV", key=key)
            
            return CloudflareKVResult(
                success=True,
                key=key
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(
                "HTTP error deleting key from Cloudflare KV",
                status_code=e.response.status_code,
                response_text=e.response.text,
                key=key
            )
            return CloudflareKVResult(
                success=False,
                key=key,
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"Error deleting key from Cloudflare KV: {str(e)}"
            logger.error(
                "Unexpected error deleting key from Cloudflare KV",
                error=str(e),
                key=key
            )
            return CloudflareKVResult(
                success=False,
                key=key,
                error_message=error_msg
            ) 