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
        
        # KV API base URL
        self.kv_base_url = f'https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/storage/kv/namespaces/{KV_NAMESPACE_ID}'
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
        """
        try:
            url = f'{self.kv_base_url}/values/{key}'
            response = requests.get(url, headers=self.headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error getting KV record: {str(e)}")
            return None 