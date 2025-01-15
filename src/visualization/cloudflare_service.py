import boto3
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Union
from config.config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_ACCESS_KEY_ID,
    CLOUDFLARE_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME,
    CLOUDFLARE_API_TOKEN,
    KV_NAMESPACE_ID
)

logger = logging.getLogger(__name__)

class CloudflareService:
    """Service for interacting with Cloudflare R2 and KV."""
    
    def __init__(self):
        """Initialize the Cloudflare service."""
        # Initialize R2 client
        self.r2 = boto3.client(
            service_name='s3',
            endpoint_url=f'https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com',
            aws_access_key_id=CLOUDFLARE_ACCESS_KEY_ID,
            aws_secret_access_key=CLOUDFLARE_SECRET_ACCESS_KEY
        )
        self.bucket = R2_BUCKET_NAME
        
        # KV API base URL for updates
        self.kv_base_url = (
            f'https://api.cloudflare.com/client/v4/accounts/'
            f'{CLOUDFLARE_ACCOUNT_ID}/storage/kv/namespaces/{KV_NAMESPACE_ID}'
        )
        # Image API endpoint for reads
        self.image_api_url = 'https://image-api.office.pure---internet.com/'
        self.headers = {
            'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}',
            'Content-Type': 'application/json'
        }
    
    def upload_image(self, image_bytes: bytes, key: str) -> str:
        """
        Upload an image to R2 bucket.
        Returns the URL of the uploaded image.
        """
        try:
            self.r2.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=image_bytes,
                ContentType='image/png'
            )
            return f'https://{self.bucket}/{key}'
        except Exception as e:
            logger.error(f"Error uploading image to R2: {str(e)}")
            raise
    
    def update_kv_record(self, key: str, value: Union[str, dict]) -> bool:
        """
        Update a KV record with the given key and value.
        Value can be either a string or a dictionary.
        Returns True if successful, False otherwise.
        """
        try:
            url = f'{self.kv_base_url}/values/{key}'
            response = requests.put(
                url,
                headers=self.headers,
                json={'value': value}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error updating KV record: {str(e)}")
            return False
    
    def get_kv_record(self, key: str) -> Optional[str]:
        """
        Get a KV record value by key.
        Returns None if the key doesn't exist.
        
        This method fetches data from the image API for reads, while updates still go directly to KV.
        The API returns data in a structure like:
            {
              "latest_sensor_interval": {
                "value": {
                  "path": "...",
                  "last_processed": "...",
                  "status": "success",
                  ...
                }
              }
            }
        
        We return a JSON string of the sub-object for 'key', or None if not found.
        """
        try:
            logger.info(f"Fetching data from image API for key: {key}")
            response = requests.get(self.image_api_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Image API response for {key}: {json.dumps(data.get(key, {}), indent=2)}")
            
            if key in data:
                # data[key] is typically a dict, e.g. {"value": {...}}
                # Return as a JSON string so that the caller can parse it as needed
                return json.dumps(data[key])  
            
            logger.debug(f"No value found for key {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting record from image API: {str(e)}")
            return None
