import pandas as pd

# 파일 경로 설정
stock_data_file = 'cache/output_table_with_stock_data.csv'
twitter_data_file = 'cache/flattened_twitter_data.csv'
output_file = 'cache/output_table_user_and_stock.csv'

try:
    # 기존 주식 데이터 CSV 파일 읽기
    print("Loading stock data...")
    stock_df = pd.read_csv(stock_data_file, encoding='utf-8-sig')
    print(f"Stock data loaded: {len(stock_df)} records")

    # 트위터 데이터 CSV 파일 읽기
    print("Loading Twitter data...")
    twitter_df = pd.read_csv(twitter_data_file, encoding='utf-8-sig')
    print(f"Twitter data loaded: {len(twitter_df)} records")


    # 트윗 ID 추출 함수
    def extract_tweet_id(custom_id):
        """Custom ID에서 트윗 ID를 추출"""
        try:
            return int(custom_id.split('-')[-1])
        except (ValueError, AttributeError, IndexError):
            print(f"Warning: Could not extract tweet ID from: {custom_id}")
            return None


    # 주식 데이터에 트윗 ID 컬럼 추가
    print("Extracting tweet IDs...")
    stock_df['Tweet ID'] = stock_df['Custom ID'].apply(extract_tweet_id)

    # 유효한 트윗 ID가 있는 행의 개수 확인
    valid_tweet_ids = stock_df['Tweet ID'].notna().sum()
    print(f"Valid tweet IDs extracted: {valid_tweet_ids}")

    # 트위터 데이터에서 id와 user_id만 선택하고 중복 제거
    twitter_lookup = twitter_df[['id', 'user_id']].drop_duplicates(subset=['id'])
    print(f"Unique tweet-user pairs in Twitter data: {len(twitter_lookup)}")

    # 트윗 ID를 기준으로 user_id 조인
    print("Matching tweet IDs with user IDs...")
    merged_df = pd.merge(
        stock_df,
        twitter_lookup,
        left_on='Tweet ID',
        right_on='id',
        how='left'
    )

    # 불필요한 컬럼 제거 (id 컬럼은 Tweet ID와 중복)
    if 'id' in merged_df.columns:
        merged_df = merged_df.drop('id', axis=1)

    # 컬럼 순서 재정렬 (User ID를 네 번째 열로 이동)
    columns = list(merged_df.columns)

    # User ID를 네 번째 위치로 이동
    if 'user_id' in columns:
        columns.remove('user_id')
        columns.insert(3, 'user_id')  # 0-based index이므로 3이 네 번째 열
        merged_df = merged_df[columns]

    # 매칭 결과 통계
    matched_count = merged_df['user_id'].notna().sum()
    print(f"Successfully matched user IDs: {matched_count} out of {len(merged_df)}")
    print(f"Match rate: {(matched_count / len(merged_df) * 100):.1f}%")

    # 불필요한 컬럼 제거 (Tweet ID)
    if 'Tweet ID' in merged_df.columns:
        merged_df = merged_df.drop('Tweet ID', axis=1)

    # 결과를 새 CSV 파일로 저장
    print(f"Saving results to {output_file}...")
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 결과 미리보기
    print("\nFirst 5 rows of the result:")
    print(merged_df.head())

    # 컬럼 정보 출력
    print(f"\nColumn names (total: {len(merged_df.columns)}):")
    for i, col in enumerate(merged_df.columns, 1):
        print(f"{i:2d}. {col}")

    print(f"\nProcess completed successfully!")
    print(f"Output saved to: {output_file}")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    print("Please make sure the following files exist:")
    print(f"- {stock_data_file}")
    print(f"- {twitter_data_file}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback

    traceback.print_exc()