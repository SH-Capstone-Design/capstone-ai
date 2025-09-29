import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 방금 학습해서 저장한 모델의 기본 경로(필요하면 절대경로로 바꿔도 됨)
DEFAULT_FINETUNED_PATH = r"models/koelectra-small-emotion"
# 예) 절대경로로 확정하고 싶다면:
# DEFAULT_FINETUNED_PATH = r"C:\Users\Acer\AI\models\koelectra-small-emotion"

def load_model(device: str = "cpu", finetuned_path: str | None = None):
    """
    무조건 파인튜닝된 모델만 로드한다.
    우선순위: 함수 인자 > 환경변수 FINETUNED_MODEL_PATH > DEFAULT_FINETUNED_PATH.
    경로가 없으면 예외 발생.
    """
    ft_path = finetuned_path or os.getenv("FINETUNED_MODEL_PATH") or DEFAULT_FINETUNED_PATH

    if not (ft_path and os.path.exists(ft_path)):
        raise FileNotFoundError(
            f"파인튜닝 모델을 찾을 수 없습니다: {ft_path}\n"
            f"학습 시 저장된 경로를 전달하거나 FINETUNED_MODEL_PATH를 설정하세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(ft_path)
    model = AutoModelForSequenceClassification.from_pretrained(ft_path)

    torch.set_num_threads(1)   # CPU 환경 스레드 과다 방지
    model.to(device)
    model.eval()
    return tokenizer, model
