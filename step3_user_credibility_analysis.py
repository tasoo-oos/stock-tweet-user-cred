import pandas as pd, numpy as np, math
from scipy.stats import norm

# ---------- 0. 데이터 읽기 ----------
df = pd.read_csv("cache/output_table_user_and_stock.csv", parse_dates=["Tweet Date"])

# ---------- 1. 가격 레이블 ----------
def price_label(row):
    up   = row["High Change from Tweet (%)"] >= 3
    down = row["Low Change from Tweet (%)"]  <= -3
    if up and down:
        return "both"
    elif up:
        return "rise"
    elif down:
        return "fall"
    else:
        return "flat"

df["price_label"] = df.apply(price_label, axis=1)

# ---------- 2. Neutral 제거 ----------
df = df[df["Sentiment"] != "neutral"].copy()

# ---------- 3. 예측 적중 여부 ----------
def is_hit(s, p):
    if p == "both":                 # 어느 방향이든 정답
        return 1
    return int((s == "positive" and p == "rise") or
               (s == "negative" and p == "fall"))

df["hit"] = [is_hit(s, p) for s, p in zip(df["Sentiment"], df["price_label"])]

# ---------- 4. 가중치(모델 확신도) ----------
df["weight"] = np.exp(df["Log Prob"])

# ---------- 5. Wilson 하한값 함수 ----------
def wilson_lb(pos, n, conf=0.95):
    if n == 0: return 0
    z  = norm.ppf(1 - (1-conf)/2)
    ph = pos / n
    denom  = 1 + z*z/n
    centre = ph + z*z/(2*n)
    adj_sd = z * math.sqrt((ph*(1-ph) + z*z/(4*n)) / n)
    return (centre - adj_sd) / denom

# ---------- 6. 유저별 신뢰도 집계 ----------
def user_stats(g):
    hits      = (g["hit"] * g["weight"]).sum() # 틀렸으면 0, 맞았으면 0~1인데 괜찮겠지?
    w_sum     = g["weight"].sum()
    n         = len(g)
    wilson    = wilson_lb(g["hit"].sum(), n)
    # 피어슨 r: +1/-1 감성 점수 ↔ 실제 수익률
    sent_score  = g["Sentiment"].map({"positive": 1, "negative": -1})
    if sent_score.nunique() > 1:
        r = sent_score.corr(g["Price Change (%)"])
    else:
        r = np.nan
    return pd.Series({
        "tweets": n,
        "acc_plain": g["hit"].mean(),
        "acc_weighted": hits / w_sum if w_sum else 0,
        "wilson_lower": wilson,
        "pearson_corr": r
    })

# → apply 결과에서 user_id 컬럼이 중복 생성되는 걸 방지하기 위해 중간 변수(temp)로 받음
# include_groups=False 를 추가하여 DeprecationWarning을 해결하고, group key가 결과에 포함되지 않도록 명시
temp = df.groupby("user_id", group_keys=True).apply(user_stats, include_groups=False) # Pandas 1.5+ 권장
# 또는 이전 방식 유지 시
# temp = df.groupby("user_id").apply(user_stats)
# if "user_id" in temp.columns:
#    temp = temp.drop(columns="user_id")

user_cred = temp.reset_index()

# ---------- 7. 전체 감성 비율 ----------
overall_sentiment_ratio = (
    df["Sentiment"]
    .value_counts(normalize=True)
    .rename_axis("Sentiment")
    .reset_index(name="Ratio")
)

# ---------- 8. 유저별 감성 비율 ----------
# 중간 결과를 임시 변수에 저장
temp_user_sentiment_ratio = (
    df.groupby(["user_id", "Sentiment"])
      .size()
      .groupby(level=0) # SeriesGroupBy 에는 include_groups 파라미터가 없을 수 있음.
      .apply(lambda s: s/s.sum())     # 비율화
      .unstack(fill_value=0)
)

# unstack() 후 reset_index() 전에 'user_id' 컬럼이 이미 존재하면 삭제 (핵심 수정)
if "user_id" in temp_user_sentiment_ratio.columns:
    temp_user_sentiment_ratio = temp_user_sentiment_ratio.drop(columns="user_id")

user_sentiment_ratio = temp_user_sentiment_ratio.reset_index()


# ---------- 9. 결과 예시 출력 ----------
print("\n=== 상위 10명 신뢰도(Wilson 기준) ===")
print(user_cred.sort_values("wilson_lower", ascending=False).head(10))

print("\n=== 전체 감성 비율 ===")
print(overall_sentiment_ratio)

print("\n=== 유저별 감성 비율 + 평균 신뢰도 merge 예시 ===")
out = user_sentiment_ratio.merge(user_cred[["user_id","acc_plain"]], on="user_id")
print(out.head(10))

# ---------- 10. 결과 저장 ----------
user_cred.to_csv("user_credibility.csv", index=False)
overall_sentiment_ratio.to_csv("overall_sentiment_ratio.csv", index=False)
user_sentiment_ratio.to_csv("user_sentiment_ratio.csv", index=False)
out.to_csv("user_sentiment_and_accuracy.csv", index=False)

print("\n코드가 성공적으로 실행되었습니다.")