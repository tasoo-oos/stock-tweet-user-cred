import pandas as pd
import re
import json
from pathlib import Path

# --- 상수 정의 ---
DATA_SPLITS = {
    'train': 'data/train-00000-of-00001-24d52140a30ef03c.parquet',
    'test': 'data/test-00000-of-00001-9e63b9de85b2453a.parquet',
    'valid': 'data/valid-00000-of-00001-7ec206eb036ab81e.parquet'
}
DATASET_URL_PREFIX = "hf://datasets/TheFinAI/flare-sm-acl/"
TWEET_RAW_DATA_DIR = Path('dataset/tweet/raw/')
OUTPUT_CSV_FILE = Path('cache/flare_edited.csv')  # 베이스 파일 이름으로 사용

# 정규 표현식
TICKER_PATTERN = r'\$[A-Za-z]{1,7}(?:-[A-Za-z]{1})?'
DATE_IN_FOLLOWING_PATTERN = r'\n\d{4}-\d{2}-\d{2}: '
USER_PATTERN = r'@[A-Za-z0-9_]{1,15}:'
URL_PATTERN = r'https?://[^\s]+'


# --- 도우미 함수 ---

def load_dataframe(split_name: str) -> pd.DataFrame:
    """지정된 split의 데이터프레임을 로드합니다."""
    file_path = DATA_SPLITS.get(split_name)
    if not file_path:
        raise ValueError(f"Invalid split name: {split_name}. Available: {list(DATA_SPLITS.keys())}")
    return pd.read_parquet(DATASET_URL_PREFIX + file_path)


def extract_query_parts(query_text: str) -> tuple[str, str, str]:
    """쿼리 텍스트에서 티커, prefix, following 텍스트를 추출합니다."""
    prefix, following_text = "", ""
    try:
        prefix, following_text = query_text.split('\n\n', 1)
    except ValueError:
        print(f"Warning: Query could not be split by '\\n\\n'. Assuming entire text is prefix: {query_text[:100]}...")
        prefix = query_text

    ticker_match = re.search(TICKER_PATTERN, prefix)
    if not ticker_match:
        raise ValueError(f"Ticker not found in prefix: {prefix[:100]}...")

    ticker = ticker_match.group(0)[1:]
    return ticker, prefix, following_text


def format_tweet_text(text: str) -> str:
    """트윗 텍스트를 정제합니다."""
    processed_text = text.replace('\n', ' ').replace('"', "'")
    processed_text = re.sub(USER_PATTERN, 'AT_USER', processed_text.lower())
    processed_text = re.sub(URL_PATTERN, '', processed_text)
    return processed_text


def process_tweet_file(ticker: str, date_str: str) -> tuple[list[str], list[str]]:
    """주어진 티커와 날짜에 해당하는 트윗 파일을 읽고 처리합니다."""
    file_path = TWEET_RAW_DATA_DIR / ticker.upper() / date_str
    tweets_v1, tweets_v2 = [], []

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return tweets_v1, tweets_v2

    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    tweet_data = json.loads(line.strip())
                    text = tweet_data.get('text', '')
                    formatted_text = format_tweet_text(text)
                    tweets_v1.append(f'- "{formatted_text}"')

                    user_info = tweet_data.get('user', {})
                    tweets_v2.extend([
                        f'- "{formatted_text}"',
                        f'- user_id: {user_info.get("id_str", "")}',
                        f'- user_followers: {user_info.get("followers_count", "")}'
                    ])
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error processing line in {file_path}: {e}, line: {line.strip()}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return tweets_v1, tweets_v2


def process_single_query(query_text: str) -> tuple[str, str]:
    """단일 쿼리를 처리하여 두 가지 버전의 새로운 쿼리 문자열을 생성합니다."""
    try:
        ticker, prefix, following_text = extract_query_parts(query_text)
    except ValueError as e:
        print(f"Skipping query due to error during extraction: {e}. Query: {query_text[:150]}...")
        error_message_suffix = f" [Error: Could not process query - {e}]"
        final_instruction = "\n\n---\n\nBased on the information, will the stock price rise or fall? Answer with only a single word: 'Rise' or 'Fall'."
        return (query_text.strip() + error_message_suffix + final_instruction,
                query_text.strip() + error_message_suffix + final_instruction)

    matched_dates_with_format = re.findall(DATE_IN_FOLLOWING_PATTERN, '\n' + following_text)
    dates_to_process = [d.strip('\n :') for d in matched_dates_with_format]

    all_tweets_for_query_v1 = [prefix, ""]
    all_tweets_for_query_v2 = [prefix, ""]

    for date_str in dates_to_process:
        tweets_v1_for_date, tweets_v2_for_date = process_tweet_file(ticker, date_str)
        if tweets_v1_for_date:
            all_tweets_for_query_v1.append(f"{date_str}:")
            all_tweets_for_query_v1.extend(tweets_v1_for_date)
        if tweets_v2_for_date:
            all_tweets_for_query_v2.append(f"{date_str}:")
            all_tweets_for_query_v2.extend(tweets_v2_for_date)

    # *** 핵심 변경점: 최종 프롬프트 생성 ***
    # 1. 수집된 모든 정보(주가, 트윗 등)를 하나의 문자열(컨텍스트)로 결합합니다.
    context_v1 = "\n".join(all_tweets_for_query_v1)
    context_v2 = "\n".join(all_tweets_for_query_v2)

    # 2. 모델에게 내릴 강력한 지시문을 정의합니다.
    final_instruction = "\n\n---\n\nBased on the provided stock data and tweets, predict the stock price movement. Respond with a single word only: 'Rise' or 'Fall'."

    # 3. 컨텍스트와 지시문을 합쳐 최종 프롬프트를 완성합니다.
    new_query_v1 = context_v1 + final_instruction
    new_query_v2 = context_v2 + final_instruction

    return new_query_v1, new_query_v2


# --- 메인 로직 ---
def main():
    """메인 실행 함수"""
    for split_name in DATA_SPLITS.keys():
        print(f"\n--- Processing data split: {split_name} ---")
        try:
            df = load_dataframe(split_name)
        except Exception as e:
            print(f"Error loading dataframe for split {split_name}: {e}")
            continue

        if df.empty:
            print(f"DataFrame for split '{split_name}' is empty. Skipping.")
            continue

        processed_queries_v1 = []
        processed_queries_v2 = []

        total_queries = len(df['query'])
        print(f"Found {total_queries} queries to process for {split_name}.")
        for idx, query_text in enumerate(df['query']):
            if (idx + 1) % 100 == 0:
                print(f"Processing query {idx + 1}/{total_queries} for split {split_name}...")
            new_q1, new_q2 = process_single_query(query_text)
            processed_queries_v1.append(new_q1)
            processed_queries_v2.append(new_q2)

        df['new_query1'] = processed_queries_v1
        df['new_query2'] = processed_queries_v2

        df.rename(columns={'query': 'old_query'}, inplace=True)
        df.rename(columns={'new_query1': 'query'}, inplace=True)

        output_path = OUTPUT_CSV_FILE.parent / f"{OUTPUT_CSV_FILE.stem}_{split_name}{OUTPUT_CSV_FILE.suffix}"
        output_path_parquet = output_path.with_suffix('.parquet')

        df.to_parquet(output_path_parquet, index=False)
        print(f"Successfully processed {split_name} and saved to {output_path_parquet}")

        if not df.empty:
            print(f"\n--- Example of new 'query' for {split_name} (first entry) ---")
            print(df['query'].iloc[0])


if __name__ == '__main__':
    main()