"""

"""
from .openai_client import OpenAIClient
from .constants import (
    SENTIMENT_ANALYSIS_SYSTEM_INSTRUCTION,
    SENTIMENT_ANALYSIS_QUERY_INSTRUCTION,
    DEFAULT_GPT_MODEL,
    ACL18_FLATTENED_TWITTER_CSV_PATH,
    KAGGLE1_FLATTENED_TWITTER_CSV_PATH,
    BATCH_OUTPUT_DIR,
)
import pandas as pd
from datetime import datetime
import time
from .utils import setup_logging
from .benchmark_cli import list_batch_id
import json

logger = setup_logging(__name__)

class SentimentAnalyze:
    def __init__(self, dataset_choice, num_samples=0, flattened_dataset_path=None, model=None):

        self.sentiment_levels = [
            "negative",
            "neutral",
            "positive"
        ]

        self.dataset_choice = dataset_choice.lower()
        if not flattened_dataset_path:
            if dataset_choice == 'acl18':
                self.flattened_dataset_path = ACL18_FLATTENED_TWITTER_CSV_PATH
            elif dataset_choice == 'kaggle1':
                self.flattened_dataset_path = KAGGLE1_FLATTENED_TWITTER_CSV_PATH
        else:
            self.flattened_dataset_path = flattened_dataset_path

        self.num_samples = num_samples

        if not model:
            model = DEFAULT_GPT_MODEL
        self.client = OpenAIClient(model=model)

        logger.info(f"SentimentAnalyze initialized with dataset: {self.dataset_choice}, model: {model}")
        logger.info(f"Flattened dataset path: {self.flattened_dataset_path}")

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

    def _load_dataset(self, batch_split=1, batch_index=0):
        """
        Load the flattened dataset from the specified path.
        The dataset can be in CSV or Parquet format.

        Returns:
            pd.DataFrame: The loaded dataset as a DataFrame.
        """
        if str(self.flattened_dataset_path.suffix).endswith('.csv'):
            df = pd.read_csv(self.flattened_dataset_path, encoding='utf-8')
        else:
            df = pd.read_parquet(self.flattened_dataset_path)

        if self.num_samples > 0:
            df = df.sample(n=self.num_samples, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {self.num_samples} rows from the dataset.")
        elif batch_split > 1:
            length = len(df)
            batch_size = length // batch_split
            if batch_index == batch_split-1:
                df = df.iloc[batch_index * batch_size:]
            else:
                df = df.iloc[batch_index * batch_size:(batch_index + 1) * batch_size]
            percentage = (batch_index + 1) * 100 / batch_split
            logger.info(f"Loaded batch {batch_index + 1}/{batch_split} ({percentage:.2f}%) with {len(df)} rows.")

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
        batch_split=1,
        batch_index=0,
        temperature=0.0,
        max_tokens=20,
        max_batch_wait_time=14400,  # 4 hours
    ):
        """
        """
        prompts = []
        custom_ids = []

        df = self._load_dataset(batch_split, batch_index)

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

    def check_batch(self, batch_id: str) -> None:
        """Check batch status and results."""

        if batch_id == 'recent':
            # 가장 최근에 호출한 batch ID로 대체
            logger.info('No batch ID provided, using the most recent batch ID.')
            recent_batch = list_batch_id(1)[0]
            batch_id = recent_batch.id
            logger.info(f'Batch ID: {batch_id}')

        try:
            int(batch_id)
        except:
            pass
        else:
            if int(batch_id) < 0:
                batch_job = list_batch_id(abs(int(batch_id)))[-1]
                batch_id = batch_job.id
                logger.info(f'Batch ID: {batch_id}')
            else:
                # error
                logger.error(
                    'Invalid batch ID format. Please provide a valid batch ID or use "recent" or a negative integer.')
                return

        output_jsonl_path = BATCH_OUTPUT_DIR / "jsonl" / f"{batch_id}.jsonl"
        output_csv_path = BATCH_OUTPUT_DIR / "csv" / f"{batch_id}.csv"

        try:
            client = OpenAIClient()
            batch_job = client.client.batches.retrieve(batch_id)
        except Exception:
            logger.error('Invalid batch ID')
            return

        # Wait for completion
        while True:
            batch_job = client.client.batches.retrieve(batch_id)
            logger.info(f'Current status: {batch_job.status}')

            if batch_job.status == 'completed':
                logger.info('Batch API completed')
                break
            elif batch_job.status in ['failed', 'cancelled']:
                logger.error(f'Batch API {batch_job.status}')
                return

            time.sleep(15)

        # Process results
        if batch_job.output_file_id:
            output_file_content = client.client.files.content(batch_job.output_file_id).read()
            output_file_content = output_file_content.decode('utf-8')

            # Save JSONL
            output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl_path.open('w', encoding='utf-8') as f:
                for line in output_file_content.strip().split('\n'):
                    f.write(json.dumps(json.loads(line), ensure_ascii=False) + '\n')

            logger.info(f'Output saved to: {output_jsonl_path}')

