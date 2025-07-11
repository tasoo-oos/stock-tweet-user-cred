"""
Central configuration and constants for the Stock Tweet User Credibility project.

[Index]
- Path
- Dataset configuration
- Model & Prompt & Batch configuration
- Stock analysis configuration
- Date and Text Patterns

"""

# ========== Path ==========

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Datasets directories
DATASETS_DIR = PROJECT_ROOT / "datasets"
ACL18_DATASET_DIR = DATASETS_DIR / "acl18_dataset"
ACL18_PRICE_DATA_DIR = ACL18_DATASET_DIR / "price" / "raw"
ACL18_TWEET_DATA_DIR = ACL18_DATASET_DIR / "tweet" / "raw"

# Data directories
CACHE_DIR = PROJECT_ROOT / "cache"
BATCH_OUTPUT_DIR = CACHE_DIR / "batch_benchmark_result" # Batch API로 benchmark를 돌린(주가 예측을 한) 결과를 저장하는 경로
CUSTOM_BENCHMARK_DIR = CACHE_DIR / "custom_benchmark_datasets" # benchmark 실행을 위한 커스텀 데이터셋(flare_edited~.parquet 등)을 저장하는 경로
TWEET_SENTIMENT_RESULTS_DIR = CACHE_DIR / "tweet_sentiment_results" # tweet별로 감성 분석을 실행한 결과를 저장하는 경로 (~_combined.jsonl이 저장됨)
FLATTENED_TWITTER_DIR = CACHE_DIR / "flattened_twitter"
JOIN_TWEET_AND_STOCK_DIR = CACHE_DIR / "join_tweet_and_stock"
TEMP_DIR = CACHE_DIR / "temp"  # 임시 파일 저장 경로

# Cache file paths
ACL18_COMBINED_JSONL_PATH = CACHE_DIR / TWEET_SENTIMENT_RESULTS_DIR / "acl18_combined.jsonl"
ACL18_FLATTENED_TWITTER_CSV_PATH = CACHE_DIR / FLATTENED_TWITTER_DIR / "acl18_flattened_twitter_data.csv"
ACL18_FLATTENED_TWITTER_PKL_PATH = CACHE_DIR / FLATTENED_TWITTER_DIR / "acl18_flattened_twitter_data.pkl"
ACL18_COLUMN_INFO_PATH = CACHE_DIR / FLATTENED_TWITTER_DIR / "acl18_column_info.txt"

KAGGLE1_COMBINED_JSONL_PATH = CACHE_DIR / TWEET_SENTIMENT_RESULTS_DIR / "acl18_combined.jsonl"
KAGGLE1_FLATTENED_TWITTER_CSV_PATH = CACHE_DIR / FLATTENED_TWITTER_DIR / "acl18_flattened_twitter_data.csv"
KAGGLE1_FLATTENED_TWITTER_PKL_PATH = CACHE_DIR / FLATTENED_TWITTER_DIR / "acl18_flattened_twitter_data.pkl"
KAGGLE1_COLUMN_INFO_PATH = CACHE_DIR / FLATTENED_TWITTER_DIR / "acl18_column_info.txt"

OUTPUT_TABLE_STOCK_PATH = CACHE_DIR / JOIN_TWEET_AND_STOCK_DIR / "output_table_with_stock_data.csv"
OUTPUT_TABLE_USER_STOCK_PATH = CACHE_DIR / JOIN_TWEET_AND_STOCK_DIR / "output_table_user_and_stock.csv"
BATCH_REQUEST_TEMP_SAVE_PATH = TEMP_DIR / "batch_request_temp_save.jsonl"  # Batch API 요청을 임시로 저장하는 경로


# ========== Dataset configuration ==========

# Dataset configuration (acl, )
FLARE_DATASET_SPLITS = {
    'acl_train': 'data/train-00000-of-00001-24d52140a30ef03c.parquet',
    'acl_test': 'data/test-00000-of-00001-9e63b9de85b2453a.parquet',
    'acl_valid': 'data/valid-00000-of-00001-7ec206eb036ab81e.parquet',
}
FLARE_DATASET_URL_PREFIX = "hf://datasets/TheFinAI/flare-sm-acl/"
FLARE_EDITED_PREFIX = "flare_edited_"
FLARE_EDITED_TEST_PATH = CUSTOM_BENCHMARK_DIR / (FLARE_EDITED_PREFIX + "acl_test.parquet")

# Stock mapping
STOCK_TABLE_PATH = ACL18_DATASET_DIR / "StockTable"



# ========== Model & Prompt & Batch configuration =========

# Model configuration
DEFAULT_GPT_MODEL = "gpt-4.1-mini-2025-04-14"

from .constants_prompt import *
SENTIMENT_ANALYSIS_SYSTEM_INSTRUCTION = SENTIMENT_ANALYSIS_SYSTEM_INSTRUCTION
SENTIMENT_ANALYSIS_QUERY_INSTRUCTION = SENTIMENT_ANALYSIS_QUERY_INSTRUCTION

DEFAULT_GPT_SYSTEM_INSTRUCTION = DEFAULT_GPT_SYSTEM_INSTRUCTION_COT_version1
QUERY_INSTRUCTION = QUERY_INSTRUCTION

# Prompt configuration
PREFIX_FOR_TWEET_LIST = '' # 기본값
# PREFIX_FOR_TWEET_LIST = '\nTweet List (Credible User):'
# PREFIX_FOR_TWEET_LIST = '\n# Tweet Window : {date1} ～ {date2} (14 calendar days)'
ANSWER_SUFFIX = '\nAnswer:'

# Batch API configuration
BATCH_CHECK_INTERVAL = 10  # seconds
MAX_BATCH_WAIT_TIME = 14400  # 4 hour
BATCH_API_MAX_RETRIES = 3
BATCH_API_SLEEP_TIME = 5

# Batch IDs for evaluation (주요 배치 ID 저장)
BATCH_ID_MATCH = {
    'basic':'batch_685cd0be015881908f09b9430bde0430', # flare 원본 데이터셋 조금 수정 -> 모든 트윗 포함 + 형식 약간 바꿈
    'non_tweets':'batch_685d71b160ec8190a2dde5ad64f2a63f', # 트윗 없이 주가 변동만 넣었을 때의 결과 (트위터 정보가 유의미한지 파악)

    'non_neutral':'batch_685ce8241c648190bf57f433f69ac8a4', # non_neutral + 임계값 이하의 신뢰도를 가진 유저를 제거
    'exclude_low':'batch_685cf93779388190a12643dba2978214', # non_neutral + 임계값 이하의 신뢰도를 가진 유저를 제거
    'include_cred':'batch_685d076df2a08190ac5aacb9b12ae75d', # exclude_low + 유저 신뢰도를 프롬프트에 추가

    'exclude_low+0.5s':'batch_685d343e7b108190a6f116a6d3523b2a', # exclude_low에서 threshold를 평균+0.5표준편차로 설정 (상위 30%만 남김)
    'exclude_low-0.5s':'batch_685d4bd0b4e08190b70f7accdfdf9f7a', # exclude_low에서 threshold를 평균-0.5표준편차로 설정 (상위 70%만 남김)
}

# ========== Stock analysis configuration ==========

# Business days for stock analysis
BUSINESS_DAYS_FORWARD = 10  # 2 weeks

# Sentiment levels
SENTIMENT_LEVELS = ["negative", "neutral", "positive"]

# Performance thresholds
PRICE_RISE_THRESHOLD = 3.0  # 3% for rise
PRICE_FALL_THRESHOLD = -3.0  # -3% for fall

# Stock movement evaluation
STOCK_MOVEMENT_LABELS = ["rise", "fall"]
STOCK_MOVEMENT_CHOICE_MAPPING = {
    "rise": ["yes", "positive"],
    "fall": ["no", "negative", "neutral"],
}
STOCK_MOVEMENT_DEFAULT = "error"

# Wilson confidence interval
WILSON_CONFIDENCE = 0.95



# ========== Date and Text Patterns ==========

# Date patterns
DATE_PATTERN = r'\d{4}-\d{2}-\d{2}'
DATE_IN_FOLLOWING_PATTERN = r'\n\d{4}-\d{2}-\d{2}: '

# Text patterns
TICKER_PATTERN = r'\$[A-Za-z]{1,7}(?:-[A-Za-z]{1})?'
USER_PATTERN = r'@[A-Za-z0-9_]{1,15}:'
URL_PATTERN = r'https?://[^\s]+'
