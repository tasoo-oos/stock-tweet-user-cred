"""
Central configuration and constants for the Stock Tweet User Credibility project.
"""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATASET_DIR = PROJECT_ROOT / "dataset"
CACHE_DIR = PROJECT_ROOT / "cache"
PRICE_DATA_DIR = DATASET_DIR / "price" / "raw"
TWEET_DATA_DIR = DATASET_DIR / "tweet" / "raw"
BATCH_OUTPUT_DIR = CACHE_DIR / "batch_output"
CUSTOM_BENCHMARK_DIR = CACHE_DIR / "custom_benchmark_dataset"

# Cache file paths
COMBINED_JSONL_PATH = CACHE_DIR / "combined.jsonl"
FLATTENED_TWITTER_CSV_PATH = CACHE_DIR / "flattened_twitter_data.csv"
FLATTENED_TWITTER_PKL_PATH = CACHE_DIR / "flattened_twitter_data.pkl"
OUTPUT_TABLE_STOCK_PATH = CACHE_DIR / "output_table_with_stock_data.csv"
OUTPUT_TABLE_USER_STOCK_PATH = CACHE_DIR / "output_table_user_and_stock.csv"
USER_CREDIBILITY_PATH = CACHE_DIR / "user_credibility.csv"
COLUMN_INFO_PATH = CACHE_DIR / "column_info.txt"

# Dataset configuration (acl, cikm, bigdata)
FLARE_DATASET_SPLITS = {
    'acl_train': 'data/train-00000-of-00001-24d52140a30ef03c.parquet',
    'acl_test': 'data/test-00000-of-00001-9e63b9de85b2453a.parquet',
    'acl_valid': 'data/valid-00000-of-00001-7ec206eb036ab81e.parquet',
}
FLARE_DATASET_URL_PREFIX = "hf://datasets/TheFinAI/flare-sm-acl/"
FLARE_EDITED_TEST_PATH = CUSTOM_BENCHMARK_DIR / "flare_edited_test_.parquet"

# Stock mapping
STOCK_TABLE_PATH = DATASET_DIR / "StockTable"

# Model configuration
DEFAULT_GPT_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_GPT_SYSTEM_INSTRUCTION = """
You are an expert financial analyst. Your SOLE task is to predict stock price movement based on the user's query.
You MUST provide your response ONLY in the format of a valid JSON object, and nothing else. Do not include explanations or any text outside the JSON structure.

The JSON object must strictly follow this schema:
{
  "answer": "string",
  "confidence": "float"
}

The value for "answer" MUST be one of "Rise" or "Fall".
The value for "confidence" MUST be a number between 0.0 and 1.0.

Begin your response immediately with "{"
""".strip()

# Sentiment levels
SENTIMENT_LEVELS = ["negative", "neutral", "positive"]

# Stock movement evaluation
STOCK_MOVEMENT_LABELS = ["rise", "fall"]
STOCK_MOVEMENT_CHOICE_MAPPING = {
    "rise": ["yes", "positive"],
    "fall": ["no", "negative", "neutral"],
}
STOCK_MOVEMENT_DEFAULT = "error"

# Date patterns
DATE_PATTERN = r'\d{4}-\d{2}-\d{2}'
DATE_IN_FOLLOWING_PATTERN = r'\n\d{4}-\d{2}-\d{2}: '

# Text patterns
TICKER_PATTERN = r'\$[A-Za-z]{1,7}(?:-[A-Za-z]{1})?'
USER_PATTERN = r'@[A-Za-z0-9_]{1,15}:'
URL_PATTERN = r'https?://[^\s]+'

# Performance thresholds
PRICE_RISE_THRESHOLD = 3.0  # 3% for rise
PRICE_FALL_THRESHOLD = -3.0  # -3% for fall

# Batch API configuration
BATCH_CHECK_INTERVAL = 10  # seconds
MAX_BATCH_WAIT_TIME = 3600  # 1 hour
BATCH_API_MAX_RETRIES = 3
BATCH_API_SLEEP_TIME = 5

# Business days for stock analysis
BUSINESS_DAYS_FORWARD = 10  # 2 weeks

# Wilson confidence interval
WILSON_CONFIDENCE = 0.95