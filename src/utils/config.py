import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database location
DB_PATH = os.getenv("DB_PATH", "database/sentiment_monitor.db")

# Input CSV path
INPUT_CSV = os.getenv("INPUT_CSV", "data/raw/social_media_posts.csv")

# Brand keyword filter
BRAND_KEYWORD = os.getenv("BRAND_KEYWORD", "Netflix")