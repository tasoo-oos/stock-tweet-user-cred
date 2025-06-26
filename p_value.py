import pandas as pd
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

# --- 상수 ---

batch_output_path = Path('cache/batch_output/csv')
batch_id_match = {
    'basic':'batch_685cd0be015881908f09b9430bde0430',
    'non_neutral':'batch_685ce8241c648190bf57f433f69ac8a4',
    'exclude_low':'batch_685cf93779388190a12643dba2978214',
    'include_cred':'batch_685d076df2a08190ac5aacb9b12ae75d'
}

# --  함수 ---

def calculate_mcnemar(series1, series2):
    """

    Args:
        series1,2:  Boolean 속성의 Series(열)

    Returns:

    """
    both_series = (series1) & (series2)
    only_series1 = (series1) & (~series2)
    only_series2 = (~series1) & (series2)
    neither_series = (~series1) & (~series2)

    matrix22 = [[both_series.sum(), only_series1.sum()],
                [only_series2.sum(), neither_series.sum()]]

    result = mcnemar(matrix22, exact=False, correction=True)

    return result, matrix22

# --- 메인 로직 ---

# key_list 추출 (id 있는 것만)
query_types = []
for key in batch_id_match:
    if key: query_types.append(key)
print(f'id가 입력된 query_types: {query_types}\n')

# 각각 추출
tf_df = pd.DataFrame()
for query in query_types:
    batch_output_file_path = batch_output_path / (batch_id_match[query]+'.csv')
    result_df = pd.read_csv(batch_output_file_path)
    result_df['true_false'] = result_df['gold_label'] == result_df['pred_label']
    tf_df[query] = result_df['true_false'].copy()

# vs basic
for criterion_i, criterion in enumerate(query_types):
    for query_type in query_types[criterion_i+1:]:
        print(f'[{criterion} vs {query_type}]')

        # 결과 계산
        result, matrix22 = calculate_mcnemar(tf_df[criterion], tf_df[query_type])

        # matrix 출력
        matrix_df = pd.DataFrame(matrix22)
        matrix_df.index = [criterion+' - True', criterion+' - False']
        matrix_df.columns = [query_type+' - True', query_type+' - False']
        print(matrix_df)

        # 통계량 출력
        print(f'검정 통계량 (chi-square): {result.statistic:.4f}')
        print(f'p-value: {result.pvalue:.4f}')
        print()

    print('-'*50)
