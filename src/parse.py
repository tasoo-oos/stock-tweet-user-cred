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
import time
import random

random.seed(42)

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

    def _drop_duplicate_tweets(self, df):
        """
        Drop duplicate tweets based on user_id, text, and stock_ticker.
        """
        initial_shape = df.shape
        df = df.drop_duplicates(['created_at', 'text', 'stock_ticker'])
        final_shape = df.shape
        logger.info(f"Dropped {initial_shape[0] - final_shape[0]} duplicate tweets.")
        return df

    def _drop_old_tweets(self, df):
        """
        Drop tweets older than 2 years from the current date.
        """
        import datetime

        # 2017년 12월 31일 23시 59분 기준으로 설정 (2018년부터의 데이터만 사용)
        dt = datetime.datetime(2018, 12, 31, 23, 59, 0)

        # 타임스탬프 변환 (로컬 시간 기준)
        timestamp = int(dt.timestamp())

        df = df[df['created_at'].astype(int) > timestamp]

        logger.info(f"Dropped tweets older than {dt.strftime('%Y-%m-%d %H:%M:%S')}. Remaining tweets: {df.shape[0]}")

        return df

    def _random_sample_by_user(self, df, num_user):
        _unique_users = df['user_id'].unique()
        is_selected_users = random.sample(list(_unique_users), num_user)
        df = df[df['user_id'].isin(is_selected_users)]
        logger.info(f"Randomly selected {num_user} users from {len(_unique_users)} unique users.")
        return df

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

        tweets_df.dropna(subset=['user_name'], inplace=True)  # Drop rows where user_name is NaN
        tweets_df['user_id'] = pd.factorize(tweets_df['user_name'])[0] + 1  # Ensure user_id starts from 1

        # 중복 제거
        tweets_df = self._drop_duplicate_tweets(tweets_df)

        # 최근 2년만 남기기
        tweets_df = self._drop_old_tweets(tweets_df)

        # user 수 줄이기
        num_user = 10000
        tweets_df = self._random_sample_by_user(tweets_df, num_user)

        tweets_df.to_csv(KAGGLE1_FLATTENED_TWITTER_CSV_PATH, index=False, encoding='utf-8')
        tweets_df.to_pickle(KAGGLE1_FLATTENED_TWITTER_PKL_PATH)

        logger.info(f"row count: {tweets_df.shape[0]}")
        logger.info(f"Kaggle1 tweets data parsed and saved to {KAGGLE1_FLATTENED_TWITTER_CSV_PATH}")

    def parse_tweets_data(self):
        """

        Returns:

        """
        if self.dataset_choice == 'acl18':
            return self._parse_acl18_tweets()
        elif self.dataset_choice == 'kaggle1':
            return self._parse_kaggle1_tweets()
