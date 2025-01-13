from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 60))  # Default to 60 if not set

# Cloudflare R2 Configuration
CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')
CLOUDFLARE_ACCESS_KEY_ID = os.getenv('CLOUDFLARE_ACCESS_KEY_ID')
CLOUDFLARE_SECRET_ACCESS_KEY = os.getenv('CLOUDFLARE_SECRET_ACCESS_KEY')
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')

# Cloudflare KV Configuration
CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')
KV_NAMESPACE_ID = os.getenv('KV_NAMESPACE_ID')
