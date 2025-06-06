import json
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

file_path = 'cache/combined.jsonl'
price_path = 'dataset/price/raw' # AAPL.csv, ABB.csv, ...
sentiment_levels = [
            "negative",
            "neutral",
            "positive"
        ]

def get_business_days_later(date_str, business_days=10):
    """주어진 날짜로부터 영업일 기준으로 며칠 후의 날짜를 반환"""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    days_added = 0
    current_date = date

    while days_added < business_days:
        current_date += timedelta(days=1)
        # 주말(토요일=5, 일요일=6)이 아닌 경우만 카운트
        if current_date.weekday() < 5:
            days_added += 1

    return current_date.strftime('%Y-%m-%d')


def get_stock_data(ticker, tweet_date, price_path):
    """주식 데이터에서 필요한 정보를 추출"""
    csv_file = os.path.join(price_path, f"{ticker}.csv")

    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found")
        return None, None, None, None, None, None, None

    try:
        # CSV 파일 읽기
        stock_df = pd.read_csv(csv_file)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df = stock_df.sort_values('Date')

        tweet_datetime = pd.to_datetime(tweet_date)

        # 트윗 올린 시점의 종가 찾기 (해당 날짜 또는 그 이후 첫 거래일)
        tweet_day_data = stock_df[stock_df['Date'] >= tweet_datetime]
        if tweet_day_data.empty:
            print(f"Warning: No stock data found for {ticker} on or after {tweet_date}")
            return None, None, None, None, None, None, None

        tweet_close_price = tweet_day_data.iloc[0]['Close']
        actual_tweet_date = tweet_day_data.iloc[0]['Date']

        # 2주 후(10 영업일 후) 날짜 계산
        two_weeks_later = get_business_days_later(tweet_date, 10)
        two_weeks_datetime = pd.to_datetime(two_weeks_later)

        # 2주 후의 종가 찾기 (해당 날짜 또는 그 이후 첫 거래일)
        two_weeks_data = stock_df[stock_df['Date'] >= two_weeks_datetime]
        if two_weeks_data.empty:
            print(f"Warning: No stock data found for {ticker} on or after {two_weeks_later}")
            two_weeks_close_price = None
            price_change_pct = None
        else:
            two_weeks_close_price = two_weeks_data.iloc[0]['Close']
            # 2주간 종가 변화율 계산
            price_change_pct = ((two_weeks_close_price - tweet_close_price) / tweet_close_price) * 100

        # 2주간의 데이터 범위 설정
        period_data = stock_df[
            (stock_df['Date'] >= actual_tweet_date) &
            (stock_df['Date'] <= two_weeks_datetime)
            ]

        if not period_data.empty:
            # 2주간 최고가, 최저가
            period_high = period_data['High'].max()
            period_low = period_data['Low'].min()

            # 트윗 시점 종가 대비 최고가, 최저가 변화율
            high_change_pct = ((period_high - tweet_close_price) / tweet_close_price) * 100
            low_change_pct = ((period_low - tweet_close_price) / tweet_close_price) * 100
        else:
            period_high = None
            period_low = None
            high_change_pct = None
            low_change_pct = None

        return (tweet_close_price, two_weeks_close_price, price_change_pct,
                period_high, period_low, high_change_pct, low_change_pct)

    except Exception as e:
        print(f"Error processing stock data for {ticker}: {e}")
        return None, None, None, None, None, None, None

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

extracted_data = []

for line in lines:
    if not line.strip():  # 빈 줄은 건너뛰기
        continue
    try:
        data = json.loads(line)

        custom_id = data.get("custom_id")
        ticker = custom_id.split('-20')[0].split('tweet-sentiment-')[-1] # ex: "AAPL"
        date = '201'+custom_id.split('-201')[1].split(' ')[0] # ex: "2015-01-01" (트윗 올린 시점)
        tweet_id = custom_id.split('-')[-1] # ex: "564077457850916864"

        highest_sentiment = ''
        outlook_results = {}

        response_data = data.get("response")
        if response_data:
            body_data = response_data.get("body")
            if body_data:
                choices = body_data.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    choice_item = choices[0] # 첫 번째 choice 사용

                    # Sentiment 및 Log Prob을 top_logprobs 기준으로 추출하도록 변경
                    # 기존 message.content 기반 sentiment 추출 로직은 이 블록에서 대체됨
                    logprobs_info = choice_item.get("logprobs")
                    if logprobs_info:
                        content_logprobs = logprobs_info.get("content")
                        if content_logprobs and isinstance(content_logprobs, list) and len(content_logprobs) > 0:
                            # content_logprobs는 모델이 실제로 선택한 토큰(들)에 대한 정보 리스트.
                            # 감성분석의 경우 보통 단일 토큰이므로 content_logprobs[0] 사용.
                            # 이 token_data_for_actual_output 변수는 모델이 실제로 출력한 토큰에 대한 정보를 담고 있음.
                            token_data_for_actual_output = content_logprobs[0]
                            top_logprobs_list = token_data_for_actual_output.get("top_logprobs")
                            if top_logprobs_list and isinstance(top_logprobs_list, list) and len(top_logprobs_list) > 0:

                                for sentiment in sentiment_levels:
                                    outlook_results[sentiment] = 0.0

                                for prob in top_logprobs_list:
                                    sentiment = prob.get("token")
                                    if sentiment in sentiment_levels:
                                        outlook_results[sentiment] = np.exp(prob.get("logprob")).item()

                                # top_logprobs 리스트의 첫 번째 요소가 가장 확률이 높은 토큰.
                                highest_prob_token_info = top_logprobs_list[0]
                                highest_sentiment = highest_prob_token_info.get("token")

        # 주식 가격 데이터 추출
        (tweet_close, two_weeks_close, price_change_pct,
         period_high, period_low, high_change_pct, low_change_pct) = get_stock_data(ticker, date, price_path)

        extracted_data.append({
            "Custom ID": custom_id,
            "Ticker": ticker,
            "Tweet Date": date,
            "Sentiment": highest_sentiment,
            "Prob-positive": outlook_results['positive'],
            "Prob-neutral": outlook_results['neutral'],
            "Prob-negative": outlook_results['negative'],
            "Tweet Close Price": tweet_close,
            "Two Weeks Close Price": two_weeks_close,
            "Price Change (%)": round(price_change_pct, 2) if price_change_pct is not None else None,
            "Period High": period_high,
            "Period Low": period_low,
            "High Change from Tweet (%)": round(high_change_pct, 2) if high_change_pct is not None else None,
            "Low Change from Tweet (%)": round(low_change_pct, 2) if low_change_pct is not None else None
        })

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e} for line: {line[:100]}...")
    except Exception as e:
        print(f"An unexpected error occurred: {e} for line: {line[:100]}...")

# Pandas DataFrame으로 변환
df = pd.DataFrame(extracted_data)

# 결과 출력
print(f"Total records processed: {len(df)}")
print("\nSample data:")
print(df.head())

# 통계 정보 출력
print(f"\nStock price data found for: {df['Tweet Close Price'].notna().sum()} records")
print(f"Records with complete 2-week data: {df['Two Weeks Close Price'].notna().sum()}")

# CSV 파일로 저장
df.to_csv('cache/output_table_with_stock_data.csv', index=False, encoding='utf-8-sig')

# 기본 통계 정보
if df['Price Change (%)'].notna().sum() > 0:
    print(f"\nPrice Change Statistics:")
    print(f"Mean 2-week price change: {df['Price Change (%)'].mean():.2f}%")
    print(f"Median 2-week price change: {df['Price Change (%)'].median():.2f}%")
    print(f"Std deviation: {df['Price Change (%)'].std():.2f}%")