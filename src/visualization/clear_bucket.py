import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')

session = boto3.Session(
    aws_access_key_id=os.getenv('CLOUDFLARE_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('CLOUDFLARE_SECRET_ACCESS_KEY'),
)
s3 = session.resource(
    's3',
    endpoint_url=f"https://{os.getenv('CLOUDFLARE_ACCOUNT_ID')}.r2.cloudflarestorage.com"
)

bucket = s3.Bucket(os.getenv('R2_BUCKET_NAME'))
bucket.objects.all().delete()