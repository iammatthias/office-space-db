from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 60))  # Default to 60 seconds if not set
