"""
Generate prompts for the FLARE dataset based on user queries and tweets.
"""
import pandas as pd
import re
import json
import numpy as np

from .constants import (
    FLARE_DATASET_SPLITS,
    FLARE_DATASET_URL_PREFIX,
    FLARE_EDITED_PREFIX,
    TICKER_PATTERN,
    DATE_PATTERN,
    USER_PATTERN,
    URL_PATTERN,
    ACL18_TWEET_DATA_DIR,
    DATE_IN_FOLLOWING_PATTERN,
    ANSWER_SUFFIX,
    OUTPUT_TABLE_USER_STOCK_PATH,
    CUSTOM_BENCHMARK_DIR,
    BATCH_ID_MATCH,
    QUERY_INSTRUCTION,
    COLUMN_INFO,
    PREFIX_FOR_TWEET_LIST,
    TEMP_DIR,
)
import math
import scipy.stats as st
from transformers import LlamaTokenizer
import random

random.seed(42)


def wilson_lower_bound(pos, n, confidence=0.95):
    """
    윌슨 신뢰 구간의 하한값을 계산합니다.

    Parameters:
    - pos (int): 성공 횟수 (positive outcomes)
    - n (int): 전체 시도 횟수 (total trials)
    - confidence (float): 신뢰 수준 (e.g., 0.95 for 95%)

    Returns:
    - float: 윌슨 신뢰 구간의 하한값 (신뢰도 점수)
    """
    if n == 0:
        return 0

    # z-score 계산 (예: 95% 신뢰 수준의 z-score는 약 1.96)
    # 양측 검정을 기준으로 하므로, (1-confidence)/2 만큼을 양쪽 꼬리에서 제외
    z = st.norm.ppf(1 - (1 - confidence) / 2)

    # 관측된 성공 비율
    p_hat = pos / n

    # 윌슨 스코어 공식 적용
    numerator = p_hat + z ** 2 / (2 * n) - z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)
    denominator = 1 + z ** 2 / n

    return numerator / denominator

class GeneratePrompts():
    def __init__(self, dataset, split):
        self.tokenizer = LlamaTokenizer.from_pretrained("TheFinAI/finma-7b-full")
        self.tweet_max_tokens = 33

        self.dataset = dataset
        self.split = split
        self.ANSWER_SUFFIX = ANSWER_SUFFIX
        self.SAVE_FILE_PREFIX = FLARE_EDITED_PREFIX
        self.SAVE_FILE_TYPE = 'parquet' # 'csv' or 'parquet'

        self.query_types = list(BATCH_ID_MATCH.keys())

        if dataset == 'flare-acl':
            if split == 'all':
                self.flare_splits = list(FLARE_DATASET_SPLITS.keys())
            else:
                self.flare_splits = ['acl_'+split]

            self.load_sentiment_dataframe(OUTPUT_TABLE_USER_STOCK_PATH)
            self.create_user_cred_df()

            self.count_non_exist = 0  # 트윗이 sentiment_df에 없는 경우의 카운트
            self.count_exist = 0  # 트윗이 sentiment_df에 있는 경우의 카운트
        elif dataset == 'kaggle-1':
            1
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    def load_flare_dataframe(self, split_name: str) -> pd.DataFrame:
        """지정된 split의 데이터프레임을 로드합니다."""
        print(f"\n--- Processing data split: {split_name} ---")
        try:
            file_path = FLARE_DATASET_SPLITS.get(split_name)
            if not file_path:
                raise ValueError(f"Invalid split name: {split_name}. Available: {list(FLARE_DATASET_SPLITS.keys())}")
            return pd.read_parquet(FLARE_DATASET_URL_PREFIX + file_path)
        except ValueError as e:
            print(f"Error loading dataframe for split {split_name}: {e}")
            return pd.DataFrame()
        except Exception as e:  # Catch other potential errors from pd.read_parquet
            print(f"An unexpected error occurred while loading dataframe for split {split_name}: {e}")
            return pd.DataFrame()

    def load_sentiment_dataframe(self, dataset_path) -> pd.DataFrame:
        """트윗별 감성 분석 및 실제 주가 변동률이 기록된 데이터프레임을 로드합니다."""
        try:
            # sentiment_full_df 불러오기 (트윗별로 감성 분석 및 실제 주가 변동률이 기록된 db)
            self.sentiment_full_df = pd.read_csv(dataset_path)

            # sentiment_df 생성
            custom_id_series = self.sentiment_full_df.apply(lambda x: int(x['Custom ID'].split('-')[-1]),
                                                       axis=1)  # custom id에서 트윗 id를 정수로 추출
            self.sentiment_df = pd.DataFrame()
            self.sentiment_df['tweet_id'] = custom_id_series
            self.sentiment_df['sentiment'] = self.sentiment_full_df['Sentiment']

        except FileNotFoundError:
            raise FileNotFoundError(f"Sentiment DataFrame not found at {OUTPUT_TABLE_USER_STOCK_PATH}")

    def create_user_cred_df(self):
        """user_cred_df 생성"""

        # user별 신뢰도 산정을 위해선 neutral을 일단 제거해야 함.
        non_neutral_df = self.sentiment_full_df[self.sentiment_full_df['Sentiment'] != 'neutral']
        self.user_cred_df = non_neutral_df.groupby('user_id').apply(self.process_user_accuracy)

        # 연속적인 정확도 계산
        self.user_cred_df['con_acc_log1p'] = np.log1p(self.user_cred_df['continuous_accuracy'])

        # 연속적인 정확도 + 베이즈 평균
        C = self.user_cred_df['continuous_accuracy'].mean() # 3.865 (acl18)
        m = self.user_cred_df['pred_counts'].quantile(0.75) # 16 (acl18)
        print(f"Bayesian Average Parameters: Global Avg. Accuracy (C) = {C:.4f}, Prior Weight (m) = {m:.2f}")

        def bayesian_average_score(row):
            R = row['continuous_accuracy']
            v = row['pred_counts']
            return (v / (v + m)) * R + (m / (v + m)) * C

        self.user_cred_df['bayesian_score'] = self.user_cred_df.apply(bayesian_average_score, axis=1)
        self.user_cred_df['bayesian_pct'] = self.user_cred_df['bayesian_score'].rank(pct=True)

        # 이산적인 정확도 계산
        self.user_cred_df['wilson_score'] = self.user_cred_df.apply(
            lambda row: wilson_lower_bound(row['success_counts'], row['pred_counts']),
            axis=1
        )
        self.user_cred_df['wilson_minmax'] = (self.user_cred_df['wilson_score'] - self.user_cred_df['wilson_score'].min()) / (self.user_cred_df['wilson_score'].max() - self.user_cred_df['wilson_score'].min())

        # 임시 저장
        self.user_cred_df.to_csv(TEMP_DIR / "user_cred.csv", encoding='utf-8')

        # 출력
        print(self.user_cred_df.head())

    def extract_query_parts(self, query_text: str) -> tuple[str, str, str]:  # ticker가 None일 수 없으므로 타입 힌트 변경
        """
        쿼리 텍스트에서 티커, prefix, following 텍스트를 추출합니다.
        티커를 찾지 못하면 ValueError를 발생시킵니다.
        """
        prefix = ""  # 초기화
        following_text = ""  # 초기화
        try:
            # flare 데이터셋의 경우, 'instruction+price_table'과 'tweet_list'가 \n\n으로 구분되어 있음.
            prefix, following_text = query_text.split('\n\n', 1)
        except ValueError:
            # \n\n으로 분리되지 않는 경우, 전체 텍스트를 prefix로 간주하고 following_text는 비워둡니다.
            # 이 경우, 아래 티커 검색 로직에서 티커가 없으면 ValueError가 발생합니다.
            print(
                f"Warning: Query could not be split by '\\n\\n'. Assuming entire text is prefix for ticker search: {query_text[:100]}...")
            prefix = query_text
            # following_text는 "" (빈 문자열)로 유지

        ticker_match = re.search(TICKER_PATTERN, prefix)
        if not ticker_match:
            # 티커를 찾지 못한 경우 ValueError 발생
            raise ValueError(f"Ticker not found in prefix: {prefix[:100]}...")

        ticker = ticker_match.group(0)[1:]
        return ticker, prefix, following_text

    def format_tweet_text(self, text: str) -> str:
        """트윗 텍스트를 정제합니다 (사용자명, URL 제거 등)."""
        processed_text = text.replace('\n', ' ').replace('"', "'")
        processed_text = re.sub(USER_PATTERN, 'AT_USER', processed_text.lower())
        processed_text = re.sub(URL_PATTERN, '', processed_text)
        return processed_text

    def process_tweet_file(self, ticker: str, date_str: str, use_bayesian=True) -> dict[str, list[str]]:
        """
        주어진 티커와 날짜에 해당하는 트윗 파일을 읽고 처리 -> query_type별 트윗 리스트(\n만 join하면 됨)를 반환합니다.
        현재는 n가지 버전의 트윗 목록을 반환합니다:
        1. basic
        2. non_neutral
        3. exclude_low
        4. include_cred
        5. exclude_low+0.5s
        6. exclude_low-0.5s
        ...
        """

        if use_bayesian:
            criterion = 'bayesian_score'
        else:
            criterion = 'con_acc_log1p'
            
        file_path = ACL18_TWEET_DATA_DIR / ticker.upper() / date_str
        cred_mean = self.user_cred_df[criterion].mean()
        cred_std = self.user_cred_df[criterion].std()
        # high / middle / low 구분은 상하위 20%로
        cred_high_cut = self.user_cred_df[criterion].quantile(0.8) # bayesian: 4.174
        cred_low_cut = self.user_cred_df[criterion].quantile(0.2) # 0.2일 땐 bayesian: 3.437

        cred_threshold = cred_mean  # 기본 임계값은 평균으로 설정.
        credible_user_df = self.user_cred_df[self.user_cred_df[criterion] >= cred_threshold]
        credible_user_df_plus_half_std = self.user_cred_df[self.user_cred_df[criterion] >= cred_mean + 0.5 * cred_std]
        credible_user_df_minus_half_std = self.user_cred_df[self.user_cred_df[criterion] >= cred_mean - 0.5 * cred_std]

        max_tweets_per_day = 10

        tweets_dic = {}
        for query in self.query_types:
            tweets_dic[query] = []

        finma_basic_tweets = []
        finma_cred_users = []

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return tweets_dic

        try:
            with file_path.open('r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        tweet_data = json.loads(line.strip())

                        text = tweet_data.get('text', '')
                        tweet_id = int(tweet_data.get('id', 0))
                        formatted_text = self.format_tweet_text(text)

                        token_ids = self.tokenizer.encode(formatted_text)
                        truncated_ids = token_ids[:self.tweet_max_tokens]
                        truncated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)

                        # 변인 통제를 위해, sentiment_df에 있는 tweet_id만 사용
                        if self.sentiment_df[self.sentiment_df['tweet_id'] == tweet_id].empty:
                            self.count_non_exist += 1
                            continue
                        self.count_exist += 1

                        tweets_dic['basic'].append(f'- "{formatted_text}"')

                        # finma basic용 쿼리 추가
                        finma_basic_tweets.append(truncated_text)

                        # neutral 거르기
                        if self.sentiment_df[self.sentiment_df['tweet_id'] == tweet_id].iloc[0]['sentiment'] == 'neutral':
                            is_neutral = True
                        else:
                            is_neutral = False

                        if not is_neutral:
                            tweets_dic['non_neutral'].append(f'- "{formatted_text}"')

                        # user 정보 불러오기
                        user_info = tweet_data.get('user', {})
                        uid = int(user_info.get('id', 0))

                        if uid in self.user_cred_df.index:
                            cred_score = self.user_cred_df.loc[uid, criterion]
                            cred_pct = self.user_cred_df['bayesian_pct'].loc[uid] if use_bayesian else self.user_cred_df['con_acc_log1p'].loc[uid]

                            include_cred_text = (
                                f'- "{formatted_text}"'
                                + '\n' +
                                f'- user_credibility: {cred_pct:.2f}'
                            )

                            if cred_score >= cred_high_cut:
                                cred_level = 'high'
                            elif cred_score >= cred_low_cut:
                                cred_level = 'medium'
                            else:
                                cred_level = 'low'

                            include_cred_level_text = (
                                f'- "{formatted_text}"'
                                + '\n' +
                                f'- user_credibility: {cred_level}'
                            )

                            tweets_dic['include_all_cred'].append(include_cred_text)
                            tweets_dic['include_cred_level'].append(include_cred_level_text)

                        # threshold보다 낮은 신뢰도를 가진 유저 거르기
                        if uid in credible_user_df.index:
                            user_cred_row = credible_user_df.loc[uid]

                            tweets_dic['exclude_low'].append(f'- "{formatted_text}"')
                            if not is_neutral:
                                tweets_dic['nn_exclude_low'].append(f'- "{formatted_text}"')

                            finma_cred_users.append((user_cred_row[criterion], truncated_text))

                            # 메타 데이터 포함
                            tweets_dic['include_cred'].append(include_cred_text)
                            if not is_neutral:
                                tweets_dic['nn_include_cred'].append(include_cred_text)

                        if uid in credible_user_df_plus_half_std.index:
                            tweets_dic['exclude_low+0.5s'].append(f'- "{formatted_text}"')

                        if uid in credible_user_df_minus_half_std.index:
                            tweets_dic['exclude_low-0.5s'].append(f'- "{formatted_text}"')

                    except json.JSONDecodeError as e_json:
                        print(f"Error decoding JSON in file {file_path}, line: {line.strip()}: {e_json}")
                    except AttributeError as e_attr:  # js.get('user').get('id_str') 같은 경우 대비
                        print(f"Attribute error processing tweet in {file_path}: {e_attr}, data: {line.strip()}")

                for query_type in tweets_dic:
                    if 'finma' in query_type: continue

                    if len(tweets_dic[query_type]) > max_tweets_per_day:
                        tweets_dic[query_type] = random.sample(tweets_dic[query_type], max_tweets_per_day)

                # finma_basic 생성
                if len(finma_basic_tweets) > 3:
                    finma_basic_text = ' |'.join(random.sample(finma_basic_tweets, 3))
                    tweets_dic['finma_basic'] = [finma_basic_text]
                elif finma_basic_tweets:
                    finma_basic_text = ' |'.join(finma_basic_tweets)
                    tweets_dic['finma_basic'] = [finma_basic_text]

                # finma_exclude_low 생성
                finma_cred_users.sort(key=lambda x: x[0], reverse=True)  # 신뢰도 높은 순으로 정렬
                finma_exclude_low_text = ' |'.join(list(map(lambda x: x[-1], finma_cred_users[:3])))
                if finma_exclude_low_text:
                    tweets_dic['finma_exclude_low'].append(finma_exclude_low_text)

        except FileNotFoundError:  # 이중 확인 (Path.exists() 이후에도 발생 가능성 있음, race condition 등)
            print(f"File not found (double check): {file_path}")
        except Exception as e:  # 그 외 일반적인 파일 처리 오류
            print(f"Error processing file {file_path}: {e}")
            raise
        return tweets_dic

    def make_prefix(self, prefix: str, instruction_type: str) -> str:
        """
        주어진 prefix와 instruction_type에 따라 prefix(Instruction + Price Table)을 생성합니다.
        """
        if instruction_type == 'default':
            return prefix

        ticker_match = re.search(TICKER_PATTERN, prefix)
        date_match = re.search(DATE_PATTERN, prefix)
        ticker = ticker_match.group(0)[1:] # $ 제외
        date = date_match.group(0)

        if instruction_type == 'basic':
            """
            Query_instruction을 하나로 고정 (10개 중 제일 성능이 좋은 'Analyze ~' 사용)
            """
            query_instruction = QUERY_INSTRUCTION.format(ticker=ticker, date=date)
            spliter = '\nContext: '
            price_table = prefix.split(spliter)[1]
            return query_instruction + spliter + price_table

        if instruction_type == 'with_column_info':
            query_instruction = QUERY_INSTRUCTION.format(ticker=ticker, date=date)
            spliter = '\nContext: '
            price_table = prefix.split(spliter)[1]
            return query_instruction + f'\n\n{COLUMN_INFO}\n' + spliter + price_table

        if instruction_type == 'with_column_info+cred':
            query_instruction = QUERY_INSTRUCTION.format(ticker=ticker, date=date)
            spliter = '\nContext: '
            price_table = prefix.split(spliter)[1]
            return query_instruction + f'\n\n{COLUMN_INFO}\n' + spliter + price_table + '\n\nUser Credibility:'

    def process_single_query(self, query_text: str) -> dict:
        """
        단일 쿼리를 처리하여 n 가지 버전의 새로운 쿼리 문자열을 생성합니다.

        Input:
        - query_text: str, 원본 쿼리 문자열 (예: "Query for $AAPL on 2023-01-01")

        Output:
        - query_results_dic: dict, 각 쿼리 타입에 대한 처리된 쿼리 문자열을 포함하는 딕셔너리
          {
              'basic': '...',
              'non_neutral': '...',
              'exclude_low': '...',
              'include_cred': '...',
              'exclude_low+0.5s': '...',
              'exclude_low-0.5s': '...'
          }

        """

        # --- 기존 쿼리에서 필요한 정보 추출 ---

        # ticker나 prefix 추출 안되는 경우는 없다고 가정
        # prefix: 'Query instruction + Price Table' / following_text: 'Tweet List'
        ticker, prefix, following_text = self.extract_query_parts(query_text)

        matched_dates_with_format = re.findall(DATE_IN_FOLLOWING_PATTERN, '\n' + following_text)
        dates_to_process = [d.strip('\n :') for d in
                            matched_dates_with_format]  # 날짜 부분만 추출 (예: '\n2015-12-17: ' -> '2015-12-17') # 원래 코드: dt=dt[1:-2]

        # --- 쿼리 문자열 생성 단계 ---

        # 'default'면 prefix를 그대로 사용
        new_prefix = self.make_prefix(prefix, 'basic') #SOTA는 with_column_info로 설정

        query_list_dic = {}
        for query in self.query_types:
            if '{date' in  PREFIX_FOR_TWEET_LIST:
                matched_dates_from_table = re.findall(DATE_PATTERN, prefix.split('Context:')[-1])
                prefix_for_tweet_list = PREFIX_FOR_TWEET_LIST.format(date1=matched_dates_from_table[0], date2=matched_dates_from_table[-1])
            else:
                prefix_for_tweet_list = PREFIX_FOR_TWEET_LIST
            query_list_dic[query] = [new_prefix, prefix_for_tweet_list]  # prefix와 빈 줄(or PREFIX_FOR_TWEET_LIST)로 시작 (나중에 \n\n으로 join)

        for date_str in dates_to_process:
            # 해당 날짜의 트윗 내용 생성
            current_date_header = f"{date_str}:"

            preprocessed_tweets_for_date = self.process_tweet_file(ticker, date_str)

            for query in preprocessed_tweets_for_date:
                tweets = preprocessed_tweets_for_date[query]
                if tweets:  # 트윗이 있는 경우에만 헤더와 함께 추가
                    if 'finma' in query:
                        if len(tweets) > 1:
                            print('ERROR'*30)
                            raise
                        query_list_dic[query].append(f"{current_date_header} {tweets[0]}")
                    else:
                        query_list_dic[query].append(current_date_header)
                        query_list_dic[query].extend(tweets)

        # 최종 쿼리 문자열 생성
        # 각 부분을 개행 문자로 연결하고, 마지막에 Answer 접미사 추가
        # prefix와 첫번째 날짜 그룹 사이에 두번의 개행이 들어가도록 함.
        query_results_dic = {}
        for query in query_list_dic:
            queries = query_list_dic[query]
            query_results_dic[query] = "\n".join(queries) + '\n' + ANSWER_SUFFIX

        return query_results_dic

    def process_user_accuracy(self, df):
        def correcting(x):
            if x['Sentiment'] == 'positive' and x['High Change from Tweet (%)'] > 3: return True
            if x['Sentiment'] == 'negative' and x['Low Change from Tweet (%)'] < -3: return True
            return False

        df['pred_correct'] = df.apply(correcting, axis=1)

        d = {}
        d['success_counts'] = df['pred_correct'].sum()
        d['pred_counts'] = len(df)

        d['simple_accuracy'] = len(df[df['pred_correct']]) / len(df)
        column_match = {'positive': 'High Change from Tweet (%)', 'negative': 'Low Change from Tweet (%)'}
        df['correspond_change'] = df.apply(lambda x: abs(x[column_match[x['Sentiment']]]), axis=1)
        d['continuous_accuracy'] = df['correspond_change'].mean()
        return pd.Series(d)

    def process_and_save(self):
        """메인 실행 함수"""

        if self.dataset == 'flare-acl':

            for split_name in self.flare_splits:
                self.processed_queries_dic = {key: [] for key in self.query_types}

                flare_df = self.load_flare_dataframe(split_name)
                if flare_df.empty:
                    print(f"DataFrame for split '{split_name}' is empty or failed to load. Skipping.")
                    continue

                # DataFrame의 'query' 열을 순회하며 처리
                total_queries = len(flare_df['query'])
                print(f"Found {total_queries} queries to process for {split_name}.")
                for idx, query_text in enumerate(flare_df['query']):

                    if (idx + 1) % 100 == 0:  # 100개마다 진행 상황 표시 (큰 데이터셋의 경우 유용)
                        print(f"Processing query {idx + 1}/{total_queries} for split {split_name}...")

                    new_queries = self.process_single_query(query_text)
                    for query in new_queries:
                        self.processed_queries_dic[query].append(new_queries[query])

                # --- DataFrame 형성 및 저장 ---

                # DataFrame에 열 삽입
                for query in self.processed_queries_dic:
                    flare_df[query + '_query'] = self.processed_queries_dic[query]
                flare_df.rename(columns={'query': 'old_query'}, inplace=True)

                # 각 스플릿에 대한 별도의 출력 파일 이름 생성
                output_file_name = f"{self.SAVE_FILE_PREFIX}{split_name}.csv"
                output_path = CUSTOM_BENCHMARK_DIR / output_file_name  # Path 객체로 경로 조합

                # 지정된 타입에 따라서 파일 저장
                if self.SAVE_FILE_TYPE == 'csv':
                    flare_df.to_csv(output_path, index=False)
                    print(f"Successfully processed {split_name} and saved to {output_path}")
                elif self.SAVE_FILE_TYPE == 'parquet':
                    output_path_parquet = output_path.with_suffix('.parquet')
                    flare_df.to_parquet(output_path_parquet, index=False)
                    print(f"Successfully processed {split_name} and saved to {output_path_parquet}")

                # --- 예시 출력 --- (필요한 경우, 각 스플릿의 첫 번째 항목)

            self.show_example_queries('from-file', number_of_examples=1)

            print('non_exist:', self.count_non_exist)
            print('exist:', self.count_exist)

    def show_example_queries(self, example_type, number_of_examples=1, query_types=[]):
        """각 쿼리 타입에 대한 예시 출력"""
        if example_type == 'from-code':
            if self.dataset == 'flare-acl':
                processed_queries_df = pd.DataFrame(columns=self.query_types)

                # 데이터셋 로드
                flare_df = self.load_flare_dataframe(self.flare_splits[0]) # split 중에 하나만 선택
                if flare_df.empty:
                    print("No data available to show examples.")
                    return

                # 첫번째 쿼리 추출 -> 쿼리 처리
                sample_query_texts = flare_df['query'].iloc[:number_of_examples]
                for idx, one_query_text in enumerate(sample_query_texts):
                    processed_queries_df.loc[idx] = self.process_single_query(one_query_text)
            else:
                print("Example queries are not available for this dataset.")
                return
        elif example_type == 'from-file':
            # 파일에서 예시 쿼리 로드
            file_path = CUSTOM_BENCHMARK_DIR / f"{self.SAVE_FILE_PREFIX}acl_test.{self.SAVE_FILE_TYPE}"
            if self.SAVE_FILE_TYPE == 'csv':
                processed_queries_df = pd.read_csv(file_path).iloc[:number_of_examples]
            elif self.SAVE_FILE_TYPE == 'parquet':
                processed_queries_df = pd.read_parquet(file_path).iloc[:number_of_examples]
            else:
                raise ValueError(f"Unsupported file type: {self.SAVE_FILE_TYPE}")

            column_change = {}
            for query_type in self.query_types:
                column_change[query_type + '_query'] = query_type
            processed_queries_df.rename(columns=column_change, inplace=True)

        for query_type in self.query_types:
            if query_types and query_type not in query_types:
                continue
            print(f"\n========== Example of \"{query_type}\" query ==========")
            for idx, query in enumerate(processed_queries_df[query_type]):
                print(f'[{idx}]')
                print(query)
                print('\n')


if __name__ == "__main__":
    # 예시 실행
    generator = GeneratePrompts(dataset='flare-acl', split='all')
    generator.show_example_queries()
    # 다른 split이나 dataset을 원하면, 인스턴스를 새로 생성하여 호출
    # generator = GeneratePrompts(dataset='flare-acl', split='acl_test')
    # generator.process_and_save()