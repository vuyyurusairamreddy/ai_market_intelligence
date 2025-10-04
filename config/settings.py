import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
    RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
    RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST', 'app-store-scraper.p.rapidapi.com')
    
    # Perplexity API Configuration
    PERPLEXITY_MODEL = "sonar-pro"
    PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
    
    # Data paths
    RAW_DATA_PATH = "data/raw/"
    PROCESSED_DATA_PATH = "data/processed/"
    UNIFIED_DATA_PATH = "data/unified/"
    #UNIFIED_DATA_PATH =  data/ "unified"  # Directory only
    OUTPUTS_PATH = "outputs/"
    
    # File names
    GOOGLE_PLAY_FILE = "googleplaystore.csv"
    GOOGLE_PLAY_REVIEWS_FILE = "googleplaystore_user_reviews.csv"
    CLEANED_DATA_FILE = "cleaned_data.csv"
    UNIFIED_DATA_FILE = "combined_dataset.json"
    INSIGHTS_FILE = "insights.json"
    REPORT_FILE = "executive_report.md"
    
    # API Rate limiting
    RAPIDAPI_DELAY = 1  # seconds between requests
    MAX_RETRIES = 3
    
    # Data processing
    MIN_CONFIDENCE_SCORE = 0.6
    MAX_APPS_TO_SCRAPE = 100  # Limit for demo purposes
    
    # D2C Dataset settings
    D2C_DATASET_FILE = "D2C_Synthetic_Dataset.csv"
    D2C_INSIGHTS_FILE = "d2c_insights_and_creative.json"
    D2C_PROCESSED_FILE = "d2c_processed_data.csv"

# D2C Analysis parameters
    MIN_ROAS_THRESHOLD = 2.0
    MIN_RETENTION_RATE = 0.25
    HIGH_OPPORTUNITY_PERCENTILE = 0.7
