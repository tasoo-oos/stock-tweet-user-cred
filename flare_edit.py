import pandas as pd
import re
import json
from pathlib import Path  # os 모듈 대신 pathlib 사용

# --- 상수 정의 ---
DATA_SPLITS = {
    'train': 'data/train-00000-of-00001-24d52140a30ef03c.parquet',
    'test': 'data/test-00000-of-00001-9e63b9de85b2453a.parquet',
    'valid': 'data/valid-00000-of-00001-7ec206eb036ab81e.parquet'
}
DATASET_URL_PREFIX = "hf://datasets/TheFinAI/flare-sm-acl/"
TWEET_RAW_DATA_DIR = Path('dataset/tweet/raw/')
OUTPUT_CSV_FILE = Path('cache/flare_edited.csv')
ANSWER_SUFFIX = 'Answer:'

# 정규 표현식
TICKER_PATTERN = r'\$[A-Za-z]{1,7}(?:-[A-Za-z]{1})?'
DATE_PATTERN = r'\d{4}-\d{2}-\d{2}'  # prefix에서 날짜 찾기 (사용 안 함, 주석 처리)
# DATE_IN_FOLLOWING_PATTERN = r'\n(\d{4}-\d{2}-\d{2}): ' # following text에서 날짜 추출 (캡처 그룹 사용)
DATE_IN_FOLLOWING_PATTERN = r'\n\d{4}-\d{2}-\d{2}: '  # 원본 유지, 대신 strip으로 처리
USER_PATTERN = r'@[A-Za-z0-9_]{1,15}:'
URL_PATTERN = r'https?://[^\s]+'


# --- 도우미 함수 ---

def load_dataframe(split_name: str) -> pd.DataFrame:
    """지정된 split의 데이터프레임을 로드합니다."""
    file_path = DATA_SPLITS.get(split_name)
    if not file_path:
        raise ValueError(f"Invalid split name: {split_name}. Available: {list(DATA_SPLITS.keys())}")
    return pd.read_parquet(DATASET_URL_PREFIX + file_path)


def extract_query_parts(query_text: str) -> tuple[str, str, str]:  # ticker가 None일 수 없으므로 타입 힌트 변경
    """
    쿼리 텍스트에서 티커, prefix, following 텍스트를 추출합니다.
    티커를 찾지 못하면 ValueError를 발생시킵니다.
    """
    prefix = ""  # 초기화
    following_text = ""  # 초기화
    try:
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


def format_tweet_text(text: str) -> str:
    """트윗 텍스트를 정제합니다 (사용자명, URL 제거 등)."""
    processed_text = text.replace('\n', ' ').replace('"', "'")
    processed_text = re.sub(USER_PATTERN, 'AT_USER', processed_text.lower())
    processed_text = re.sub(URL_PATTERN, '', processed_text)
    return processed_text


def process_tweet_file(ticker: str, date_str: str) -> tuple[list[str], list[str]]:
    """
    주어진 티커와 날짜에 해당하는 트윗 파일을 읽고 처리합니다.
    두 가지 버전의 트윗 목록을 반환합니다:
    1. 정제된 트윗 텍스트만 포함
    2. 정제된 트윗 텍스트와 메타데이터 포함
    """
    file_path = TWEET_RAW_DATA_DIR / ticker.upper() / date_str

    tweets_v1 = []  # 텍스트만
    tweets_v2 = []  # 텍스트 + 메타데이터

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

                    # 메타데이터 포함 버전
                    user_info = tweet_data.get('user', {})
                    uid = user_info.get('id_str', '')
                    followers_count = user_info.get('followers_count', '')
                    retweet_count = tweet_data.get('retweet_count', '')
                    favorite_count = tweet_data.get('favorite_count', '')

                    tweet_v2_lines = [
                        f'- "{formatted_text}"',
                        f'- user_id: {uid}',
                        f'- user_followers: {followers_count}'
                    ]
                    tweets_v2.extend(tweet_v2_lines)

                except json.JSONDecodeError as e_json:
                    print(f"Error decoding JSON in file {file_path}, line: {line.strip()}: {e_json}")
                except AttributeError as e_attr:  # js.get('user').get('id_str') 같은 경우 대비
                    print(f"Attribute error processing tweet in {file_path}: {e_attr}, data: {line.strip()}")

    except FileNotFoundError:  # 이중 확인 (Path.exists() 이후에도 발생 가능성 있음, race condition 등)
        print(f"File not found (double check): {file_path}")
    except Exception as e:  # 그 외 일반적인 파일 처리 오류
        print(f"Error processing file {file_path}: {e}")

    return tweets_v1, tweets_v2


def process_single_query(query_text: str) -> tuple[str, str]:
    """단일 쿼리를 처리하여 두 가지 버전의 새로운 쿼리 문자열을 생성합니다."""
    try:
        ticker, prefix, following_text = extract_query_parts(query_text)
    except ValueError as e:
        # extract_query_parts에서 티커를 못찾거나 분리 실패 시 ValueError 발생
        print(f"Skipping query due to error during extraction: {e}. Query: {query_text[:150]}...")
        # 오류 발생 시 처리 방식: 원본 쿼리에 에러 메시지와 함께 Answer 접미사 추가
        # 또는 (None, None)을 반환하여 main 로직에서 해당 쿼리 건너뛰기 등의 다른 처리도 가능
        error_message_suffix = f" [Error: Could not process query - {e}]"
        return (query_text.strip() + error_message_suffix + '\n' + ANSWER_SUFFIX,
                query_text.strip() + error_message_suffix + '\n' + ANSWER_SUFFIX)

    # DEBUG: print("\nProcessing Prefix: " + prefix) # ticker는 항상 존재한다고 가정

    matched_dates_with_format = re.findall(DATE_IN_FOLLOWING_PATTERN, '\n' + following_text)

    # 날짜 부분만 추출 (예: '\n2015-12-17: ' -> '2015-12-17')
    # 원본 코드: dt=dt[1:-2]
    dates_to_process = [d.strip('\n :') for d in matched_dates_with_format]

    all_tweets_for_query_v1 = [prefix, ""]  # prefix와 빈 줄로 시작 (나중에 \n\n으로 join)
    all_tweets_for_query_v2 = [prefix, ""]

    for date_str in dates_to_process:
        # 해당 날짜의 트윗 내용 생성
        current_date_header = f"{date_str}:"

        tweets_v1_for_date, tweets_v2_for_date = process_tweet_file(ticker, date_str)

        if tweets_v1_for_date:  # 트윗이 있는 경우에만 헤더와 함께 추가
            all_tweets_for_query_v1.append(current_date_header)
            all_tweets_for_query_v1.extend(tweets_v1_for_date)
        else:
            all_tweets_for_query_v1.append("")

        if tweets_v2_for_date:
            all_tweets_for_query_v2.append(current_date_header)
            all_tweets_for_query_v2.extend(tweets_v2_for_date)
        else:
            all_tweets_for_query_v2.append("")

    # 최종 쿼리 문자열 생성
    # 각 부분을 개행 문자로 연결하고, 마지막에 Answer 접미사 추가
    # prefix와 첫번째 날짜 그룹 사이에 두번의 개행이 들어가도록 함.
    # all_tweets_for_query_v1[1]이 "" 이므로, join 시 prefix + "\n\n" + date_header... 형태가 됨
    new_query_v1 = "\n".join(all_tweets_for_query_v1) + '\n' + ANSWER_SUFFIX
    new_query_v2 = "\n".join(all_tweets_for_query_v2) + '\n' + ANSWER_SUFFIX

    return new_query_v1, new_query_v2


# --- 메인 로직 ---
def main():
    """메인 실행 함수"""
    for split_name in DATA_SPLITS.keys():
        print(f"\n--- Processing data split: {split_name} ---")
        try:
            df = load_dataframe(split_name)
        except ValueError as e:
            print(f"Error loading dataframe for split {split_name}: {e}")
            continue
        except Exception as e: # Catch other potential errors from pd.read_parquet
            print(f"An unexpected error occurred while loading dataframe for split {split_name}: {e}")
            continue

        if df.empty:
            print(f"DataFrame for split '{split_name}' is empty or failed to load. Skipping.")
            continue

        processed_queries_v1 = []
        processed_queries_v2 = []

        # DataFrame의 'query' 열을 순회하며 처리
        total_queries = len(df['query'])
        print(f"Found {total_queries} queries to process for {split_name}.")
        for idx, query_text in enumerate(df['query']):
            if (idx + 1) % 100 == 0: # 100개마다 진행 상황 표시 (큰 데이터셋의 경우 유용)
                print(f"Processing query {idx+1}/{total_queries} for split {split_name}...")
            new_q1, new_q2 = process_single_query(query_text)
            processed_queries_v1.append(new_q1)
            processed_queries_v2.append(new_q2)

        df['new_query1'] = processed_queries_v1
        df['new_query2'] = processed_queries_v2

        # 각 스플릿에 대한 별도의 출력 파일 이름 생성
        # 예: flare_edited_train.csv, flare_edited_test.csv
        output_file_name = f"{OUTPUT_CSV_FILE.stem}_{split_name}{OUTPUT_CSV_FILE.suffix}"
        output_path = OUTPUT_CSV_FILE.parent / output_file_name # Path 객체로 경로 조합

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully processed {split_name} and saved to {output_path}")

        # 예시 출력 (필요한 경우, 각 스플릿의 첫 번째 항목)
        if not df.empty and 'new_query1' in df.columns and not df['new_query1'].empty:
            print(f"\n--- Example of new_query1 for {split_name} (first entry) ---")
            print(df['new_query1'].iloc[0])
        # if not df.empty and 'new_query2' in df.columns and not df['new_query2'].empty: # 필요시 new_query2 예시도 활성화
        #     print(f"\n--- Example of new_query2 for {split_name} (first entry) ---")
        #     print(df['new_query2'].iloc[0])


if __name__ == '__main__':
    main()

    # 제공된 예제 테스트 (주석 해제하여 사용)
    example1 = """By reviewing the data and tweets, can we predict if the closing price of $aapl will go upwards or downwards at 2015-12-31? Please indicate either Rise or Fall.
Context: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30
2015-12-16,-0.2,0.6,-2.3,0.8,0.8,1.3,3.3,4.1,4.6,4.3,4.9
2015-12-17,2.8,3.0,0.0,-2.1,-2.1,2.1,5.0,5.8,6.5,6.2,6.8
2015-12-18,2.7,3.3,-0.2,-2.7,-2.7,3.6,6.6,8.0,8.9,8.8,9.3
2015-12-21,-0.0,0.0,-1.6,1.2,1.2,1.4,4.3,6.0,7.0,7.3,7.5
2015-12-22,0.2,0.5,-0.7,-0.1,-0.1,0.9,3.4,5.4,6.6,7.2,7.2
2015-12-23,-1.2,0.2,-1.3,1.3,1.3,-0.9,1.4,3.6,4.8,5.6,5.6
2015-12-24,0.9,0.9,-0.1,-0.5,-0.5,-0.5,1.2,3.7,4.9,5.8,5.9
2015-12-28,0.7,0.8,-0.6,-1.1,-1.1,0.7,1.8,4.2,5.5,6.6,6.8
2015-12-29,-1.6,0.6,-1.7,1.8,1.8,-0.8,-0.3,1.7,3.2,4.3,4.8
2015-12-30,1.2,1.3,-0.1,-1.3,-1.3,0.5,0.7,2.4,4.1,5.3,6.0

2015-12-17: time to buy $aapl:  $gs|*** $mbly has been downgraded by deutsche bank with a $10 price target **** $scty $p $sune $fdx $aapl $xlf $dust
2015-12-19: rt AT_USER motif investing review   $dia $spy $qqq $aapl $gpro $iwm $opk $csiq $uvxy $dust $svxy $nugt |rt AT_USER track trending assets in 1 watchlist #greececrisis $aapl $djia $spx  |rt AT_USER it's
2015-12-20: aapl apple, inc. summary$aapl $c $ung $swn #aapl #investing #investing|rt AT_USER tvix velocityshares daily 2x vix short term etn market cap$tvix $aa $aapl $nxga #tvix #invest #stoc…|free video: under
2015-12-21: favorable royalty rates hike sends #pandora stock higher $p also $siri $aapl  |rt AT_USER .AT_USER says on AT_USER that new campus costs around $5b. rbc's daryanani: $aapl generates that in fcf every 
2015-12-22: shorter iphone upgrade cycle could fuel services growth for apple $aapl |rt AT_USER 2015 - the year in money: $aapl $nflx $jnk $pfe $amzn --   |rt AT_USER 2015 - the year in money: $aapl $nflx $jnk $p
2015-12-23: rt AT_USER bob peck's top 10 internet investor debate topics for 2016 $aapl $amzn $baba $fb $msft $yhoo $rax $pypl $sq |$spy 10:00 is the drop hour.  if we survive that, we'll be in relatively good sh
2015-12-24: block trade: $aapl 935,814 shares @ $108.03 [13:00:00]|$aapl shorts, you will lose your shirt on this stock.|benefits of apple enterprise deals flow one way  $ibm $msft $aapl $csco|AT_USER i have no $
2015-12-25: apple is ill-equipped in the tablet space  $intc $msft $aapl |how to day trade successfully:  $aapl $nke $twtr $strp $endp #trading #stocks … |it hardware and cloud computing have an inverse relations
2015-12-26: edc ishares msci emerging markets yield$edc $aapl $dgc.to $ge #edc #finance #invest|vz verizon communications company info$vz $aapl $unh $bib #vz #stock #stocks|$aapl - nosy apple watch users discover
2015-12-27: see why these assets are trending in 1 watchlist $aapl $eurusd $fb $tsla $usdcad  |looks like the h&s on $aapl is beating the kardashians this season. every1 and their grandma talking abt it. #buy
2015-12-28: celg celgene corp. dividend$celg $fnma $aapl $xle #celg #finance #nasdaq |free video: trading psychology series: hope  #trading #startups #stock $aapl #money #inves… |rt AT_USER #applepay merchant-loc
2015-12-29: aapl apple, inc. bid size$aapl $uso $mcd $ccl.in #aapl #stock #stock|rt AT_USER what was your favorite app of the year? top picks from apple and AT_USER  $aapl $googl |$aapl investor opinions updated 
2015-12-30: 2016 – a ‘make-or-break’ for apple inc. (nasdaq:aapl) $aapl #apple  |the slow death of old standards and facebook's role  #macworld $aapl|fully automated trading! sign up for our free trial! trade $go
Answer:"""

    example2 = """With the help of the data and tweets given, can you forecast whether the closing price of $brk-a will climb or drop at 2015-12-31? Please state either Rise or Fall.
Context: date,open,high,low,close,adj-close,inc-5,inc-10,inc-15,inc-20,inc-25,inc-30
2015-12-16,-0.2,0.1,-1.9,0.7,0.7,-2.1,-1.8,-1.4,-1.0,-1.2,-1.0
2015-12-17,1.3,1.5,-0.0,-1.4,-1.4,-0.4,-0.3,-0.0,0.3,0.2,0.3
2015-12-18,2.3,2.4,0.0,-3.1,-3.1,2.7,2.4,3.0,3.3,3.3,3.3
2015-12-21,-0.2,0.4,-1.2,1.1,1.1,1.5,1.0,1.7,2.0,2.2,2.1
2015-12-22,-0.9,0.1,-1.5,1.5,1.5,-0.2,-0.5,0.1,0.4,0.7,0.7
2015-12-23,0.0,1.8,-0.1,0.2,0.2,-0.8,-0.5,-0.2,0.2,0.6,0.5
2015-12-24,-0.1,0.2,-0.5,0.6,0.6,-1.3,-0.9,-0.7,-0.4,-0.1,-0.1
2015-12-28,0.2,0.2,-0.5,-0.6,-0.6,-0.2,-0.1,-0.2,0.2,0.5,0.5
2015-12-29,0.2,0.9,-0.3,0.1,0.1,0.0,-0.0,-0.4,0.1,0.3,0.5
2015-12-30,0.2,0.5,-0.2,0.0,0.0,0.1,-0.2,-0.4,-0.1,0.2,0.5


Answer:"""

    # print("\n--- Testing Example 1 ---")
    # # 임시로 트윗 파일 생성 (테스트를 위해)
    # TWEET_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # (TWEET_RAW_DATA_DIR / "AAPL").mkdir(exist_ok=True)
    # test_dates_aapl = ["2015-12-17", "2015-12-19", "2015-12-20", "2015-12-21", "2015-12-22", "2015-12-23", "2015-12-24", "2015-12-25", "2015-12-26", "2015-12-27", "2015-12-28", "2015-12-29", "2015-12-30"]
    # for date_str in test_dates_aapl:
    #     with (TWEET_RAW_DATA_DIR / "AAPL" / date_str).open('w', encoding='utf-8') as f:
    #         # example1의 해당 날짜 트윗 내용을 기반으로 JSON 라인 생성
    #         # 실제 데이터와 유사하게 구조화 필요
    #         if date_str == "2015-12-17":
    #             f.write(json.dumps({"text": "time to buy $aapl:  $gs|*** $mbly has been downgraded by deutsche bank with a $10 price target **** $scty $p $sune $fdx $aapl $xlf $dust", "user": {"id_str": "123", "followers_count": 100}, "retweet_count": 5, "favorite_count": 10}) + "\n")
    #         # ... (다른 날짜에 대한 트윗도 유사하게 추가)
    #         else: # 임시로 간단한 트윗 추가
    #             f.write(json.dumps({"text": f"Sample tweet for $AAPL on {date_str} @someuser http://example.com", "user": {"id_str": "456", "followers_count": 50}, "retweet_count": 2, "favorite_count": 3}) + "\n")

    # res1_ex1, res2_ex1 = process_single_query(example1.split(ANSWER_SUFFIX)[0].strip()) # Answer: 이전까지만 전달
    # print("--- Result new_query1 for Example 1 ---")
    # print(res1_ex1)
    # print("\n--- Result new_query2 for Example 1 ---")
    # print(res2_ex1)

    # print("\n--- Testing Example 2 (No Tweets Expected) ---")
    # (TWEET_RAW_DATA_DIR / "BRK-A").mkdir(exist_ok=True) # 폴더는 있지만 파일은 없음
    # res1_ex2, res2_ex2 = process_single_query(example2.split(ANSWER_SUFFIX)[0].strip())
    # print("--- Result new_query1 for Example 2 ---")
    # print(res1_ex2) # 원본 + Answer:
    # # 원본 예제와 비교하여 동일한지 확인 (트윗이 없는 경우)
    # # print(example2)
    # assert res1_ex2.strip() == example2.strip()
    # print("Example 2 output matches expected (no tweets added).")