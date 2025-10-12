import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(device: str = None):
    """
    Hugging Face Hub 또는 로컬 모델을 로드함.
    """
    # 1️⃣ 환경 변수로 모델 이름 또는 경로 지정
    model_id = os.getenv("FINETUNED_MODEL_PATH", "JeongMin05/kcelectra-base-chat-emotion")

    # 2️⃣ 디바이스 설정
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")
    print(f"🔹 Loading model from: {model_id}")

    # 3️⃣ Hugging Face 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    model.to(device)
    model.eval()

    print("✅ Model and tokenizer loaded successfully.")
    return tokenizer, model
