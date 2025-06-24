# 설명: FLARE의 주가 변동 예측 벤치마크를 독립적으로 실행하는 확장형 스크립트.
#      로컬 Hugging Face 모델(vLLM)과 GPT API를 모두 지원합니다.
#
# 실행 전 필수 설치 항목:
# pip install torch datasets scikit-learn tqdm vllm openai
#
# 사용법:
# 1. 환경변수 설정 (GPT 사용 시):
#    export OPENAI_API_KEY='your-api-key-here'
#    export OPENAI_ORG_ID='your-org-id-here'  # 선택사항
# 2. 명령줄 실행:
#    python run_benchmark.py --scenario gpt4
#    python run_benchmark.py --config config.json
# 3. 설정은 config.json 파일로 관리하거나 환경변수로 설정하세요.
#
import os
import time
import json
from abc import ABC, abstractmethod
# from pyexpat.errors import messages

from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import logging
from typing import List, Dict, Any, Optional
import sys
import copy
import datetime
import pandas as pd
from pathlib import Path

# 상수(프롬프트) 저장

# 수정된 SYSTEM_INSTRUCTION
SYSTEM_INSTRUCTION = """
You are an expert financial analyst. Your SOLE task is to predict stock price movement based on the user's query.
You MUST provide your response ONLY in the format of a valid JSON object, and nothing else. Do not include explanations or any text outside the JSON structure.

The JSON object must strictly follow this schema:
{
  "answer": "string",
  "confidence": "float"
}

The value for "answer" MUST be one of "Rise" or "Fall".
The value for "confidence" MUST be a number between 0.0 and 1.0.

Begin your response immediately with "{"
""".strip()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 모델 인터페이스 추상화 ---
class BaseLLM(ABC):
    """모든 언어 모델이 따라야 할 기본 인터페이스 (ABC: Abstract Base Class)"""

    @abstractmethod
    def __init__(self, model_path_or_name: str, **kwargs):
        pass

    @abstractmethod
    def generate(self, dataset, evaluator) -> List[str]:
        pass


class VLLMModel(BaseLLM):
    """vLLM을 사용하여 로컬 Hugging Face 모델을 실행하는 클래스"""

    def __init__(self, model_path_or_name: str, tensor_parallel_size: int = 1, trust_remote_code: bool = True,
                 **kwargs):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM을 사용하려면 'pip install vllm'을 실행해주세요.")

        logger.info(f"[VLLM] 모델 로딩 중: {model_path_or_name}")
        self.llm = LLM(
            model=model_path_or_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code
        )
        self.sampling_params = self._create_sampling_params(**kwargs)

    def _create_sampling_params(self, **kwargs):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 20)
        )

    def generate(self, dataset, evaluator) -> List[str]:
        prompts = [evaluator.create_prompt(doc) for doc in dataset]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text for out in outputs]


class GPTModel(BaseLLM):
    """OpenAI의 GPT API를 호출하는 클래스 (Batch API 지원)"""

    def __init__(self, model_path_or_name: str, **kwargs):
        try:
            import openai
        except ImportError:
            raise ImportError("GPT API를 사용하려면 'pip install openai'를 실행해주세요.")

        logger.info(f"[GPT] API 클라이언트 초기화 중: {model_path_or_name}")

        # API 키 확인
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "GPT를 사용하려면 환경 변수에 OPENAI_API_KEY를 설정해야 합니다.\n"
                "예: export OPENAI_API_KEY='your-api-key-here'"
            )

        # Organization ID 설정 (선택사항)
        if "OPENAI_ORG_ID" in os.environ:
            openai.organization = os.environ["OPENAI_ORG_ID"]
            logger.info("OpenAI Organization ID 지정됨")
        else:
            logger.info("OpenAI Organization ID 지정되지 않은 채로 계속")

        self.client = openai.OpenAI()
        self.model_name = model_path_or_name
        self.system_instruction = kwargs.get("system_instruction", None)
        self.request_timeout = kwargs.get("request_timeout", 30)
        self.max_retries = kwargs.get("max_retries", 3)
        self.sleep_time = kwargs.get("sleep_time", 5)
        self.use_batch_api = kwargs.get("use_batch_api", True)
        self.batch_check_interval = kwargs.get("batch_check_interval", 10)  # 10초마다 상태 확인
        self.max_batch_wait_time = kwargs.get("max_batch_wait_time", 3600)  # 최대 1시간 대기

    def generate(self, dataset, evaluator) -> List[str]:
        prompts = [evaluator.create_prompt(doc) for doc in dataset]
        """프롬프트 리스트를 받아 생성된 텍스트 리스트를 반환합니다."""
        if self.use_batch_api and len(prompts) > 1:
            return self._generate_batch(dataset, evaluator)
        else:
            return self._generate_sequential(prompts)

    def _generate_batch(self, dataset, evaluator) -> List[str]:
        prompts = [evaluator.create_prompt(doc) for doc in dataset]
        gold_labels = [evaluator.get_gold_label(doc) for doc in dataset]
        """Batch API를 사용하여 여러 프롬프트를 처리합니다."""
        logger.info(f"[GPT Batch] {len(prompts)}개 프롬프트를 Batch API로 처리 시작")

        # 1. Batch 요청 파일 생성
        batch_requests = []
        for i, prompt in enumerate(prompts):
            now_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_id = f"request-{now_datetime}-{i}-{gold_labels[i]}"

            if self.system_instruction:
                messages = [
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 20
                }
            }
            batch_requests.append(request)

        # 2. 임시 파일에 저장
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', encoding='utf-8', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            batch_file_path = f.name

        try:
            # 3. 파일 업로드
            logger.info("[GPT Batch] 배치 요청 파일 업로드 중...")
            with open(batch_file_path, 'rb') as file:
                batch_input_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )

            # 4. 배치 작업 생성
            logger.info("[GPT Batch] 배치 작업 생성 중...")
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            # 5. 배치 작업 완료 대기
            logger.info(f"[GPT Batch] 배치 작업 대기 중... (ID: {batch_job.id})")
            batch_job = self._wait_for_batch_completion(batch_job.id)

            # 6. 결과 다운로드 및 파싱
            logger.info("[GPT Batch] 결과 다운로드 및 파싱 중...")
            results = self._download_batch_results(batch_job.output_file_id, len(prompts))

            return results

        finally:
            # 임시 파일 정리
            import os
            if os.path.exists(batch_file_path):
                os.unlink(batch_file_path)

    def _wait_for_batch_completion(self, batch_id: str):
        """배치 작업이 완료될 때까지 대기합니다."""
        start_time = time.time()

        while True:
            batch_job = self.client.batches.retrieve(batch_id)
            status = batch_job.status

            logger.info(f"[GPT Batch] 현재 상태: {status}")

            if status == "completed":
                logger.info("[GPT Batch] 배치 작업이 완료되었습니다.")
                return batch_job
            elif status == "failed":
                raise Exception(f"배치 작업이 실패했습니다: {batch_job}")
            elif status == "expired":
                raise Exception("배치 작업이 만료되었습니다.")
            elif status == "cancelled":
                raise Exception("배치 작업이 취소되었습니다.")

            # 최대 대기 시간 확인
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_batch_wait_time:
                raise Exception(f"배치 작업 대기 시간 초과 ({self.max_batch_wait_time}초)")

            # 대기
            time.sleep(self.batch_check_interval)

    def _download_batch_results(self, output_file_id: str, expected_count: int) -> List[str]:
        """
        배치 결과를 다운로드하고 파싱합니다.
        ㄴ 오직 답변 텍스트(choices[0].messsage.content)만 추출
        """
        # 결과 파일 다운로드
        result_file_response = self.client.files.content(output_file_id)
        result_content = result_file_response.content.decode('utf-8')

        # 결과 파싱
        results = [""] * expected_count  # 순서 유지를 위해 초기화

        for line in result_content.strip().split('\n'):
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")

                # custom_id에서 인덱스 추출 (예: "request-일시-0" -> 0)
                try:
                    index = int(custom_id.split('-')[-2])

                    if result.get("response") and result["response"].get("body"):
                        response_body = result["response"]["body"]
                        if response_body.get("choices") and len(response_body["choices"]) > 0:
                            content = response_body["choices"][0]["message"]["content"]
                            results[index] = content
                        else:
                            results[index] = "BATCH_ERROR"
                    else:
                        results[index] = "BATCH_ERROR"

                except (ValueError, IndexError, KeyError) as e:
                    logger.warning(f"결과 파싱 오류: {e}")
                    continue

        # 빈 결과를 에러로 표시
        results = [r if r else "BATCH_ERROR" for r in results]

        logger.info(f"[GPT Batch] 총 {len(results)}개 결과 중 {sum(1 for r in results if r != 'BATCH_ERROR')}개 성공")

        return results

    def _generate_sequential(self, prompts: List[str]) -> List[str]:
        """순차적으로 API를 호출합니다 (기존 방식)."""
        logger.info(f"[GPT Sequential] {len(prompts)}개 프롬프트를 순차 처리 시작")
        results = []
        for prompt in tqdm(prompts, desc="[GPT] API 요청 중"):
            for attempt in range(self.max_retries):
                try:
                    if self.system_instruction:
                        messages = [
                            {"role": "system", "content": self.system_instruction},
                            {"role": "user", "content": prompt}
                        ]
                    else:
                        messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=20,
                        timeout=self.request_timeout
                    )
                    results.append(response.choices[0].message.content)
                    break  # 성공 시 루프 탈출
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit 오류 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.sleep_time * 2)  # Rate limit의 경우 더 오래 대기
                except openai.APIError as e:
                    logger.warning(f"API 오류 발생 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.sleep_time)
                except Exception as e:
                    logger.error(f"예상치 못한 오류 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.sleep_time)
                    else:
                        logger.error("최대 재시도 횟수 초과. 'API_ERROR'로 처리합니다.")
                        results.append("API_ERROR")
        return results


# --- 프롬프트 및 태스크 로직 ---


class StockMovementEvaluator:
    """주가 변동 예측 태스크의 로직 담당 (from src/tasks/flare.py)"""
    # 이 규칙은 flare.py의 StockMovement 클래스에서 정확히 가져온 것입니다.
    CHOICE_MAPPING = {
        "rise": ["yes", "positive"],
        "fall": ["no", "negative", "neutral"],
    }
    DEFAULT = "error" # PIXIU에서는 fall로 지정함.
    GOLD_LABELS = ["rise", "fall"]

    def create_prompt(self, doc: Dict[str, Any]) -> str:
        """
        아직까진 별다른 옵션 x. 그냥 그대로 출력.
        """
        return doc["query"]

    def process_single_result(self, generated_text: str) -> str:
        """모델 출력을 정규화된 레이블('rise' 또는 'fall')로 변환합니다."""
        text = generated_text.lower().strip() # 모델 출력 소문자화
        if "rise" in text or any(val in text for val in self.CHOICE_MAPPING["rise"]):
            return "rise"
        if "fall" in text or any(val in text for val in self.CHOICE_MAPPING["fall"]):
            return "fall"
        return self.DEFAULT

    def get_gold_label(self, doc: Dict[str, Any]) -> str:
        """데이터 샘플에서 정답 레이블을 추출하고 소문자로 변환합니다."""
        # .lower()를 추가하여 반환값을 항상 소문자로 만듭니다.
        return doc["choices"][doc["gold"]].lower()

# --- 성능 지표 출력 함수 ---
def show_metrics(gold_labels, predicted_labels, labels=['rise','fall'], error_label='error'):
    """
    Args:
        gold_labels: 정답 라벨 리스트
        predicted_labels: 예측 라벨 리스트
        labels: 라벨 종류 및 순서 지정
        error_label: 에러 키워드 (StockMovementEvaluator에서 self.DEFAULT로 지정한 값. 즉 모델이 의도와 다른 예측을 내놓았을 때 뭐라고 처리하는지)

    Print:
    - Confusion Matrix
    - Classification Report
    - MCC

    Returns: None (출력만 함)
    """
    # DataFrame화
    df = pd.DataFrame({
        'gold_label':gold_labels,
        'pred_label':predicted_labels
    })
    # error_label은 StockMovementEvaluator의 self.DEFAULT에서 가져오므로 'fall'과 같은 값일 수 있음. 이 경우에는 error_view는 empty가 됨.
    error_view = df[(df['pred_label']==error_label)&(~df['pred_label'].isin(labels))]
    logger.info(f'error 수 (모델의 예측이 라벨에 없는 값인 경우): {len(error_view)}')

    logger.info('[Confusion Matrix]')
    if error_view.empty: cm_labels=labels
    else: cm_labels=labels+[error_label]

    cm = confusion_matrix(df['gold_label'], df['pred_label'], labels=cm_labels)
    cm_df = pd.DataFrame(cm)

    cm_df.index = list(map(lambda x: 'true_'+x, cm_labels))
    cm_df.columns = list(map(lambda x: 'pred_'+x, cm_labels))
    if not error_view.empty:
        cm_df.drop(index='true_'+error_label, axis=0, inplace=True)

    cm_df['| sum'] = cm_df.sum(axis=1)
    cm_df.loc['-- sum --'] = cm_df.sum(axis=0)
    print(cm_df)

    logger.info('-'*50)
    logger.info('[Classification Report] - 오류 케이스 포함')
    logger.info(classification_report(df['gold_label'], df['pred_label'], zero_division=0))

    logger.info('-'*50)
    logger.info('[MCC] - 오류 케이스 제외')
    df_nonerror = df[df['pred_label']!=error_label]
    logger.info(f':{matthews_corrcoef(df_nonerror['gold_label'], df_nonerror['pred_label'])}')
    if not error_view.empty:
        logger.info('[MCC] - 오류 케이스 포함(틀린 것으로 간주)')
        df_erroriswrong = df.copy()
        for idx, row in df_erroriswrong.iterrows():
            if df_erroriswrong.loc[idx, 'pred_label'] == error_label:
                true_label = df_erroriswrong.loc[idx, 'gold_label']
                wrong_label = labels[labels.index(true_label)-1]
                df_erroriswrong.loc[idx, 'pred_label'] = wrong_label
        logger.info(f':{matthews_corrcoef(df_erroriswrong['gold_label'], df_erroriswrong['pred_label'])}')

# --- 메인 평가 함수 ---

def run_evaluation(config: Dict[str, Any], num_samples: int) -> Dict[str, Any]:
    """설정값을 바탕으로 평가를 수행하는 메인 함수"""
    logger.info("=" * 50)
    logger.info(f"벤치마크 평가를 시작합니다: {config.get('experiment_name', 'Untitled')}")
    logger.info(f"  - 모델 타입: {config['model_type']}")
    logger.info(f"  - 모델 이름/경로: {config['model_path_or_name']}")
    logger.info(f"  - 데이터셋 경로: {config['dataset_path']} ({config['dataset_split']} split)")
    if num_samples > 0:
        logger.info(f"  - 샘플 데이터 수: {num_samples}")
    else:
        logger.info(f"  - 일괄 호출 (전체 test셋 사용)")
    logger.info("=" * 50)

    # 설정 복사본 생성 (원본 변경 방지)
    config = copy.deepcopy(config)

    # 1. 모델 인스턴스 생성
    model_type = config.pop("model_type")
    model_path_or_name = config.pop("model_path_or_name")

    if model_type == 'vllm':
        model = VLLMModel(model_path_or_name, **config)
    elif model_type == 'gpt':
        model = GPTModel(model_path_or_name, **config)
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    # 2. 데이터셋 로드 및 프롬프트 생성
    logger.info("[1/4] 데이터셋 로드 및 프롬프트 생성...")
    try:
        # 데이터셋 경로가 로컬 파일(.parquet 등)인지 Hugging Face Hub 경로인지 확인
        is_local_file = os.path.isfile(config['dataset_path'])
        if is_local_file:
            # 로컬 파일인 경우, 데이터 타입을 명시적으로 지정
            logger.info(f"로컬 Parquet 파일 로드: {config['dataset_path']}")
            full_dataset = load_dataset(
                "parquet",
                data_files=config['dataset_path'],
                split=config['dataset_split']
            )
        else:
            # Hugging Face Hub 경로인 경우, 기존 방식대로 로드
            logger.info(f"Hugging Face Hub 데이터셋 로드: {config['dataset_path']}")
            full_dataset = load_dataset(config['dataset_path'], split=config['dataset_split'])

    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        raise

    # 샘플 추출 단계 (num_samples 존재 시에만)
    if num_samples > 0:
        dataset = full_dataset.select(range(min(num_samples, len(full_dataset))))
        logger.info(f"전체 {len(full_dataset)}개 데이터 중 {len(dataset)}개를 샘플링했습니다.")
    else:
        dataset = full_dataset

    evaluator = StockMovementEvaluator()

    prompts = [evaluator.create_prompt(doc) for doc in dataset] # 아직까진 별다른 프롬프트 설정 x. doc["query"]만 추출
    gold_labels = [evaluator.get_gold_label(doc) for doc in dataset]
    logger.info(f"총 {len(prompts)}개의 평가 데이터 준비 완료.")

    # 3. 모델 추론 실행
    logger.info("[2/4] 모델 추론 시작...")
    generated_texts = model.generate(dataset, evaluator)
    logger.info("모델 추론 완료.")

    # 4. 결과 처리 및 지표 계산
    logger.info("[3/4] 결과 처리 및 성능 지표 계산...")
    predicted_labels = [evaluator.process_single_result(text) for text in generated_texts]

    # 최종 결과 출력
    logger.info("[4/4] 최종 결과")
    logger.info("=" * 50)
    show_metrics(gold_labels, predicted_labels, labels=evaluator.GOLD_LABELS, error_label=evaluator.DEFAULT)
    logger.info("=" * 50)

    return



def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """기본 설정들을 반환합니다."""
    # 상수
    basic_gpt_model = "gpt-4.1-mini-2025-04-14"
    flare_edited_dataset_path = "./cache/flare_edited_test.parquet"

    # gpt 관련 config 정의
    gpt_batch_base_config = {
        "experiment_name": "[FULL] GPT-4.1 mini on Stock Movement",
        "model_type": "gpt",
        "model_path_or_name": basic_gpt_model,
        "system_instruction": SYSTEM_INSTRUCTION,
        # 데이터 파일의 실제 경로를 직접 지정합니다.
        "dataset_path": flare_edited_dataset_path,
        # 단일 파일을 로드할 때, datasets 라이브러리는 기본적으로 'train' 스플릿으로 인식합니다.
        "dataset_split": "train",
        "use_batch_api": True,
        "batch_check_interval": 15,
        "max_batch_wait_time": 7200,  # GPT-4는 더 오래 걸릴 수 있음
        "max_tokens": 20,
        "temperature": 0.0
    }

    gpt_batch_full_config = copy.deepcopy(gpt_batch_base_config)
    gpt_batch_full_config["batch_check_interval"] = 15
    gpt_batch_full_config["max_batch_wait_time"] = 7200  # GPT-4는 더 오래 걸릴 수 있음

    gpt_batch_sample_config = copy.deepcopy(gpt_batch_base_config)
    gpt_batch_sample_config["batch_check_interval"] = 10
    gpt_batch_sample_config["max_batch_wait_time"] = 600  # 샘플 테스트는 오래 기다릴 필요 없음

    gpt_batch_flare_original_config = {
        "experiment_name": "GPT 3.5 Turbo on Stock Movement (ACL18)",
        "model_type": "gpt",
        "model_path_or_name": "gpt-3.5-turbo",
        "system_instruction": SYSTEM_INSTRUCTION,
        "dataset_path": "TheFinAI/flare-sm-acl",
        "dataset_split": "test",
        "max_retries": 3,
        "sleep_time": 5,
        "use_batch_api": True,
        "batch_check_interval": 10,
        "max_batch_wait_time": 3600
    }

    finma_batch_config = {
        "experiment_name": "FinMA 7B Full on Stock Movement (ACL18)",
        "model_type": "vllm",
        "model_path_or_name": "TheFinAI/finma-7b-full",
        "system_instruction": None,
        "dataset_path": "TheFinAI/flare-sm-acl",
        "dataset_split": "test",
        "tensor_parallel_size": 1,
        "trust_remote_code": True
    }

    return {
        "gpt-batch-full": gpt_batch_full_config,
        "gpt-batch-sample": gpt_batch_sample_config,
        "gpt-batch-flare-original": gpt_batch_flare_original_config,
        "finma-batch": finma_batch_config
    }

def check_batch(batch_id):
    # 상수
    output_jsonl_path = Path(f'cache/batch_output/jsonl/{batch_id}.jsonl')
    output_csv_path = Path(f'cache/batch_output/csv/{batch_id}.csv')

    # gpt model 생성
    try:
        import openai
    except ImportError:
        raise ImportError("GPT API를 사용하려면 'pip install openai'를 실행해주세요.")

    # API 키 확인
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "GPT를 사용하려면 환경 변수에 OPENAI_API_KEY를 설정해야 합니다.\n"
            "예: export OPENAI_API_KEY='your-api-key-here'"
        )

    # Organization ID 설정 (선택사항)
    if "OPENAI_ORG_ID" in os.environ:
        openai.organization = os.environ["OPENAI_ORG_ID"]
        logger.info("OpenAI Organization ID 지정됨")
    else:
        logger.info("OpenAI Organization ID 지정되지 않은 채로 계속")

    client = openai.OpenAI()

    # batch_id 확인
    try:
        batch_job = client.batches.retrieve(batch_id)
    except:
        logger.info('잘못된 batch id입니다.')
        return

    # 15초마다 현황 확인
    while True:
        batch_job = client.batches.retrieve(batch_id)
        logger.info(f'현재 상황: {batch_job.status}')
        if batch_job.status == 'completed':
            logger.info('-' * 50)
            logger.info('배치 API 수행 완료')
            logger.info('-' * 50)
            break
        elif batch_job.status in ['failed', 'cancelled']:
            logger.info('-' * 50)
            logger.info('배치 API 수행 중 오류 발생 (failed or cancelled)')
            logger.info('-' * 50)
            exit()
        time.sleep(15)

    if batch_job.status == 'completed':
        output_file_id = batch_job.output_file_id
        error_file_id = batch_job.error_file_id
        logger.info(f'output_file_id: {output_file_id}')
        logger.info(f'error_file_id: {error_file_id}')

        if output_file_id:
            # 내용 추출
            output_file_content = client.files.content(output_file_id).read()
            output_file_content = output_file_content.decode('utf-8')

            # 파일로 저장 - jsonl
            output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

            with output_jsonl_path.open('w', encoding='utf-8') as f:
                for line in output_file_content.strip().split('\n'):
                    f.write(json.dumps(json.loads(line), ensure_ascii=False) + '\n')
            logger.info(f'output file 저장 : {str(output_jsonl_path)}')

            # evaluator 생성
            evaluator = StockMovementEvaluator()

            # gold 라벨 추출을 위한 데이터셋 로드
            dataset_path = "./cache/flare_edited_test.parquet"
            logger.info(f"로컬 Parquet 파일 로드: {dataset_path}")
            dataset_df = pd.read_parquet(dataset_path)

            prompts = [evaluator.create_prompt(doc) for _, doc in dataset_df.iterrows()]  # 아직까진 별다른 프롬프트 설정 x. doc["query"]만 추출

            # 파일로 저장 - csv
            d = {'custom_id': [],
                 'gold_label': [],
                 'pred_label':[],
                 'pred_text':[]}

            output_json_list = output_file_content.strip().split('\n')  # 빈 줄 제거
            for idx, line in enumerate(output_json_list):
                line = json.loads(line)
                custom_id = line.get('custom_id')
                df_idx = int(custom_id.split('-')[-2])
                response_body = line.get('response', {}).get('body', {})
                if response_body and 'choices' in response_body:
                    response_str = response_body.get('choices', [])[0].get('message', {}).get('content', '')
                    try:
                        response_json = json.loads(response_str)
                    except: # json이 아닐 경우
                        # 딕셔너리에 저장 (DataFrame 용)
                        d['custom_id'].append(custom_id)
                        d['gold_label'].append(custom_id.split('-')[-1])
                        d['pred_text'].append(response_str)
                        d['pred_label'].append(evaluator.process_single_result(response_str))
                    else: # json일 경우
                        d['custom_id'].append(custom_id)
                        d['gold_label'].append(custom_id.split('-')[-1])
                        d['pred_text'].append(response_json['answer'])
                        d['pred_label'].append(evaluator.process_single_result(response_json['answer']))

                    # 미리보기 출력 (앞 10개만)
                    if idx < 10:
                        logger.info('-' * 50)
                        logger.info(f'[PREVIEW-{idx}]')
                        logger.info(f'custom_id: {custom_id}')
                        logger.info(f'gold_label: {custom_id.split('-')[-1]}')
                        logger.info(f'pred_text: {response_str}')

                else:
                    logger.info(f'오류 발생 (response에 body 또는 choices 누락) | custom_id: {custom_id}')
                    logger.info('-' * 50)
                    logger.info(json.dumps(line))
                    logger.info('-' * 50)

                    # 딕셔너리에 저장 (DataFrame 용)
                    d['custom_id'].append(custom_id)
                    d['gold_label'].append(custom_id.split('-')[-1])
                    d['pred_text'].append('')
                    d['pred_label'].append(evaluator.DEFAULT)

            df = pd.DataFrame(d)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)
            logger.info('-' * 50 + '\n')
            show_metrics(d['gold_label'], d['pred_label'], labels=evaluator.GOLD_LABELS, error_label=evaluator.DEFAULT)

        if error_file_id:
            logger.info('-' * 50)
            logger.info('ERROR FILE:\n')
            error_file_content = client.files.content(error_file_id).read()
            logger.info(error_file_content.decode('utf-8'))
    return

def list_batch_id(list_len):
    # gpt model 생성
    try:
        import openai
    except ImportError:
        raise ImportError("GPT API를 사용하려면 'pip install openai'를 실행해주세요.")

    # API 키 확인
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "GPT를 사용하려면 환경 변수에 OPENAI_API_KEY를 설정해야 합니다.\n"
            "예: export OPENAI_API_KEY='your-api-key-here'"
        )

    # Organization ID 설정 (선택사항)
    if "OPENAI_ORG_ID" in os.environ:
        openai.organization = os.environ["OPENAI_ORG_ID"]
        logger.info("OpenAI Organization ID 지정됨")
    else:
        logger.info("OpenAI Organization ID 지정되지 않은 채로 계속")

    client = openai.OpenAI()

    # batch list 불러오기
    if list_len:
        batch_list = list(client.batches.list(limit=list_len))
    else:
        batch_list = list(client.batches.list())

    if not batch_list:
        print('-' * 30)
        print('BATCH API 호출 기록이 없습니다. 새로운 BATCH API를 호출해주세요.')
        print('-' * 30)
        batch_id = input('batch id를 입력해주세요.: ').strip()
    else:
        print('-' * 30, '(1번이 최신)')
        for idx, batch in enumerate(batch_list, start=1):
            try:
                request_counts = batch.request_counts.total
            except:
                request_counts = '?'
            print(f'{idx}. {batch.id} ({request_counts} requests)')
        print('-' * 30)


    return

def input_batch_id(client, limit=10):
    # batch list 불러오기
    batch_list = list(client.batches.list(limit=limit))

    if not batch_list:
        print('-' * 30)
        print('BATCH API 호출 기록이 없습니다. 새로운 BATCH API를 호출해주세요.')
        print('-' * 30)
        batch_id = input('batch id를 입력해주세요.: ').strip()
    else:
        print('-'*30, '(1번이 최신)')
        for idx, batch in enumerate(batch_list, start=1):
            try:
                request_counts = batch.request_counts.total
            except:
                request_counts = '?'
            print(f'{idx}. {batch.id} ({request_counts} requests)')
        print('-' * 30)
        batch_id_input = input('batch id 또는 번호를 입력해주세요.: ').strip()
        if batch_id_input in list(map(str,range(1,len(batch_list)+1))):
            batch_idx = int(batch_id_input)-1
            batch_id = batch_list[batch_idx].id
        else:
            batch_id = batch_id_input

    return batch_id

# --- 실험 실행 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch API로 벤치마크 실행")
    subparsers = parser.add_subparsers(dest="command", title="명령어 선택", required=True,
                                       help="다음 중 모드 선택 : call / check / list")
    call_parser = subparsers.add_parser("call", help="새로운 batch API 호출")
    check_parser = subparsers.add_parser("check", help="이미 호출한 batch API의 현황 또는 결과 출력 (batch ID 필요)")
    list_parser = subparsers.add_parser("list", help="과거 호출한 batch ID의 리스트 출력")

    # call: 새로운 batch API 작업 호출
    call_parser.add_argument("--config", type=str, help="JSON 설정 파일 경로")
    call_parser.add_argument("--num-samples", type=int, default=0, help="Batch API로 호출할 샘플 데이터 수 (미지정 시 일괄 실행)")
    call_parser.add_argument("--scenario", type=str, choices=list(get_default_configs().keys()), help="실행할 시나리오 선택")

    # check: batch ID를 입력하면, 해당 batch 작업의 현황 또는 결과 확인
    check_parser.add_argument("--batch-id", type=str, required=True, help="현황을 확인하려는 Batch ID 입력")

    # list: 최근 호출한 batch ID의 리스트 출력
    list_parser.add_argument("--list-len", type=int, help="몇 개의 최근 batch ID를 출력할 것인지 (0이면 모두 출력)")

    args = parser.parse_args()

    # config 설정
    if args.command == 'call':
        if args.scenario: # scenario 인자를 지정했을 경우
            config = get_default_configs()[args.scenario]
        elif args.num_samples > 0: # num_samples를 입력했을 경우
            config = get_default_configs()["gpt-batch-sample"]
        else: # num_samples가 0이면 전체 api 호출
            config = get_default_configs()["gpt-batch-full"]

        # 외부 설정 있으면 로드
        if args.config:
            try:
                with open(args.config, 'r', encoding='utf-8') as f:
                    config.update(json.load(f))
                logger.info(f"사용자 설정 파일 '{args.config}'을(를) 로드했습니다.")
            except FileNotFoundError:
                logger.error(f"설정 파일 '{args.config}'을(를) 찾을 수 없습니다. 기본 설정을 사용합니다.")
            except json.JSONDecodeError:
                logger.error(f"설정 파일 '{args.config}' 파싱 오류. 기본 설정을 사용합니다.")

        try:
            results = run_evaluation(config, args.num_samples) # 일단은 None
            logger.info("평가가 성공적으로 완료되었습니다.")
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {e}")
            sys.exit(1)

    elif args.command == 'check':
        try:
            results = check_batch(args.batch_id)
            logger.info("batch 체크가 성공적으로 완료되었습니다.")
        except Exception as e:
            logger.error(f'batch 체크 중 오류 발생: {e}')
            sys.exit(1)

    elif args.command == 'list':
        results = list_batch_id(args.list_len)

    else:
        logger.info('command가 잘못되었습니다. (call / check / list 중에 택 1)')