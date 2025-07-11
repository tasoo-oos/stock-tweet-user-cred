"""

"""
import pandas as pd
from src.constants import (
    FLATTENED_TWITTER_DIR,

    KAGGLE1_TWEET_DATA_FILE, # Tweets.csv
    KAGGLE1_COMPANY_TWEET_FILE, # Company_Tweet.csv
    KAGGLE1_COMPANY_INFO, # Company.csv

    KAGGLE1_COMBINED_JSONL_PATH,
    KAGGLE1_FLATTENED_TWITTER_CSV_PATH,
    KAGGLE1_FLATTENED_TWITTER_PKL_PATH,
    KAGGLE1_COLUMN_INFO_PATH
)
from .utils import setup_logging

logger = setup_logging(__name__)

class TweetsParser:
    """

    Required Output CSV Columns:
    - stock_ticker
    - created_at
    - id (tweet_id)
    - text
    - user_id
    - user_name
    """

    def __init__(self, dataset_choice):
        self.dataset_choice = dataset_choice

    def _parse_acl18_tweets(self):
        """
        Parse ACL18 tweets data.
        """
        # Implement ACL18 parsing logic here
        pass

    def _parse_kaggle1_tweets(self):
        """
        Parse Kaggle1 tweets data.
        """
        tweets_df = pd.read_csv(KAGGLE1_TWEET_DATA_FILE, encoding='utf-8')
        company_tweet_df = pd.read_csv(KAGGLE1_COMPANY_TWEET_FILE, encoding='utf-8')
        company_info_df = pd.read_csv(KAGGLE1_COMPANY_INFO, encoding='utf-8')

        logger.info(f'tweets_df shape: {tweets_df.shape} / num of tweet_id: {tweets_df["tweet_id"].nunique()}')
        logger.info(f'company_tweet_df shape: {company_tweet_df.shape} / num of tweet_id: {company_tweet_df["tweet_id"].nunique()}')
        logger.info(f'company_info_df shape: {company_info_df.shape} / num of ticker_symbol: {company_info_df["ticker_symbol"].nunique()}')

        tweets_df = pd.merge(tweets_df, company_tweet_df, on='tweet_id', how='left')
        tweets_df = pd.merge(tweets_df, company_info_df, on='ticker_symbol', how='left')

        tweets_df = tweets_df.rename(columns={
            'tweet_id': 'id',
            'writer': 'user_name',
            'post_date': 'created_at',
            'body': 'text',
            'ticker_symbol': 'stock_ticker',
        })

        tweets_df['user_id'] = pd.factorize(tweets_df['user_name'])[0] + 1  # Ensure user_id starts from 1

        tweets_df.to_csv(KAGGLE1_FLATTENED_TWITTER_CSV_PATH, index=False, encoding='utf-8')
        tweets_df.to_pickle(KAGGLE1_FLATTENED_TWITTER_PKL_PATH)

        logger.info(f"Kaggle1 tweets data parsed and saved to {KAGGLE1_FLATTENED_TWITTER_CSV_PATH}")

    def parse_tweets_data(self):
        """

        Returns:

        """
        if self.dataset_choice == 'acl18':
            return self._parse_acl18_tweets()
        elif self.dataset_choice == 'kaggle1':
            return self._parse_kaggle1_tweets()
