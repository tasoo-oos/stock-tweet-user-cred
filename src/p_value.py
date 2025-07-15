"""
p_value.py
"""
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from .constants import (
    BATCH_ID_MATCH,
    BATCH_OUTPUT_DIR
)

class PValueCalculator:
    """
    P-Value Calculator for comparing query types using McNemar's test.

    This class reads the batch output CSV files, calculates the true/false
    predictions, and performs McNemar's test to compare different query types.
    """
    def __init__(self):
        self.batch_savefile_type = 'csv'
        self.batch_output_path = BATCH_OUTPUT_DIR / self.batch_savefile_type
        self.query_types = self._get_query_types() # BATCH_ID_MATCH 딕셔너리에서 value가 있는 key들만 추출

    # --- 메인 로직 ---

    def show_pvalue(self, batches, test_func='mcnemar'):

        self.create_tf_df(batches)
        print('유효한 batch ID/닉네임:', self.tf_df.columns)
        if len(self.tf_df.columns) < 2:
            print('비교할 batch가 2개 이상이어야 합니다.')
            return

        # 모든 조합에 대해 순서대로 비교
        for criterion_i, criterion in enumerate(self.tf_df.columns):
            for query_type in self.tf_df.iloc[:, criterion_i+1:]:
                print(f'[{criterion} vs {query_type}]')

                # 길이 미스매칭 시 건너뛰기
                len_criterion = len(self.tf_df[criterion])
                len_query_type = len(self.tf_df[query_type])
                if len_criterion != len_query_type:
                    print(f'- 길이 mismatch: {len_criterion} vs {len_query_type}\n')
                    continue

                # 결과 계산
                if test_func == 'mcnemar':
                    # McNemar's test
                    result, matrix22 = self.calculate_mcnemar(self.tf_df[criterion], self.tf_df[query_type])
                elif test_func == 'exact':
                    # Exact Binomial Test
                    result, matrix22 = self.calculate_mcnemar(self.tf_df[criterion], self.tf_df[query_type], exact=True)

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

    # -- 도구 함수 ---
    def _get_query_types(self):
        """
        Extracts query types from the BATCH_ID_MATCH dictionary where the value is not None.

        Returns:
            List of query types with IDs.
        """
        # key_list 추출 (id 있는 것만)
        query_types = []
        for key in BATCH_ID_MATCH:
            if BATCH_ID_MATCH[key]:
                query_types.append(key)
        return query_types

    def create_tf_df(self, batches):
        """
        Creates a DataFrame containing true/false predictions for specified batches.
        """
        # batches가 비어있으면, 모든 query_type을 사용
        if not batches:
            batches = self.query_types

        self.tf_df = pd.DataFrame()
        for batch in batches:
            if batch in self.query_types:
                batch_id = BATCH_ID_MATCH[batch]
            else:
                batch_id = batch
            batch_output_file_path = self.batch_output_path / f'{batch_id}.{self.batch_savefile_type}'
            result_df = pd.read_csv(batch_output_file_path)
            result_df['true_false'] = result_df['gold_label'] == result_df['pred_label']
            #if 'confidence' in result_df.columns:
                #result_df['confidence'] = result_df['confidence'].astype(float)
                #result_df['brier_score'] = (result_df['true_false'].astype(int)-result_df['confidence'])**2
            self.tf_df[batch] = result_df['true_false'].copy()

    def calculate_mcnemar(self, series1, series2, exact=False):
        """

        Args:
            series1,2:  Boolean 속성의 Series(열)

        Returns:
            result: McNemar's test result
            matrix22: 2x2 contingency matrix as a list of lists

        """
        both_series = (series1) & (series2)
        only_series1 = (series1) & (~series2)
        only_series2 = (~series1) & (series2)
        neither_series = (~series1) & (~series2)

        matrix22 = [[both_series.sum(), only_series1.sum()],
                    [only_series2.sum(), neither_series.sum()]]

        result = mcnemar(matrix22, exact=exact, correction=True)
        # exact=False가 기본값 (mcnemar)
        # exact=True는 정확 이항 검정 (Exact Binomial Test)에 해당함.

        return result, matrix22
