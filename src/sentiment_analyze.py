"""

"""
from .openai_client import OpenAIClient
from .constants import (
    SENTIMENT_ANALYSIS_SYSTEM_INSTRUCTION,
    SENTIMENT_ANALYSIS_QUERY_INSTRUCTION,
    DEFAULT_GPT_MODEL,
    ACL18_FLATTENED_TWITTER_CSV_PATH,
    KAGGLE1_FLATTENED_TWITTER_CSV_PATH
)
import pandas as pd
from datetime import datetime
import time
from .utils import setup_logging

logger = setup_logging(__name__)

class SentimentAnalyze:
    def __init__(self, dataset_choice, flatten_dataset_path=None, model=None):

        self.sentiment_levels = [
            "negative",
            "neutral",
            "positive"
        ]

        self.dataset_choice = dataset_choice.lower()
        if not flatten_dataset_path:
            if dataset_choice == 'acl18':
                self.flatten_dataset_path = ACL18_FLATTENED_TWITTER_CSV_PATH
            elif dataset_choice == 'kaggle1':
                self.flatten_dataset_path = KAGGLE1_FLATTENED_TWITTER_CSV_PATH

        if not model:
            model = DEFAULT_GPT_MODEL
        self.client = OpenAIClient(model=model)

        logger.info(f"SentimentAnalyze initialized with dataset: {self.dataset_choice}, model: {model}")
        logger.info(f"Flattened dataset path: {self.flatten_dataset_path}")

    def _format_timestamp(self, timestamp):
        if isinstance(timestamp, str):
            try:
                # 이미 형식이 맞는 경우엔 그대로 반환
                datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                return timestamp
            except ValueError:
                pass

        if self.dataset_choice == 'kaggle1':
            formatted_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            return formatted_timestamp
        else:
            logger.info('WARNING: acl18 등의 데이터셋의 경우, 아직 미구현')

    def _load_dataset(self):
        """
        Load the flattened dataset from the specified path.
        The dataset can be in CSV or Parquet format.

        Returns:
            pd.DataFrame: The loaded dataset as a DataFrame.
        """
        if str(self.flatten_dataset_path.suffix).endswith('.csv'):
            df = pd.read_csv(self.flatten_dataset_path, encoding='utf-8')
        else:
            df = pd.read_parquet(self.flatten_dataset_path)
        return df

    def create_prompt(self, stock_ticker, company_name, tweet_text) -> str:
        """
        Create a prompt for sentiment analysis based on the stock ticker, company name, and tweet text.
        """

        prompt = SENTIMENT_ANALYSIS_QUERY_INSTRUCTION.format(
            stock_ticker=stock_ticker,
            company_name=f'({company_name})' if company_name else '',
            tweet_text=tweet_text
        )

        return prompt


    def generate_batch(
        self,
        temperature=0.0,
        max_tokens=20,
        max_batch_wait_time=14400,  # 4 hours
    ):
        """
        """
        prompts = []
        custom_ids = []

        df = self._load_dataset()

        for _, row in df.iterrows():
            stock_ticker = row['stock_ticker'].split('$')[-1]
            try:
                company_name = row['company_name']
            except KeyError:
                company_name = None

            tweet_text = row['text']
            timestamp = self._format_timestamp(row['created_at'])
            tweet_id = row['id']

            prompt = self.create_prompt(stock_ticker, company_name, tweet_text)
            prompts.append(prompt)


            custom_id = f"tweet-sentiment-{stock_ticker}-{timestamp}-{tweet_id}"
            custom_ids.append(custom_id)

        self.client.generate_batch(
            prompts=prompts,
            custom_ids=custom_ids,
            system_instruction=SENTIMENT_ANALYSIS_SYSTEM_INSTRUCTION,
            temperature=temperature,
            max_tokens=max_tokens,
            max_wait_time=max_batch_wait_time
        )
