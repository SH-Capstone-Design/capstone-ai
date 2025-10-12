from typing import List, Dict, Tuple
import os
import torch
from kc_ai_app.model.model_loader import load_model

LABELS = ["기쁨", "설렘", "애정", "편안함", "농담", "슬픔", "서운함", "실망", "후회", "미안함", "짜증", "화남", "질투", "불안", "의심", "중립"]

# 디바이스 선택 (환경변수 DEVICE가 있으면 사용, 기본은 cpu)
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# 모델/토크나이저는 프로세스 시작 시 1회 로드
# load_model은 이제 파인튜닝 모델만 로드하고, 경로가 없으면 예외를 던지도록 변경됨
tokenizer, model = load_model(device=DEVICE)

def _softmax_logits(logits) -> List[float]:
    # 배치=1 가정: dim=-1로 소프트맥스 후 1차원 리스트로 반환
    return torch.nn.functional.softmax(logits, dim=-1).squeeze(0).tolist()

def analyze_sentence(sentence: str) -> Tuple[Dict[str, float], str, float]:
    """
    문장 단일 분석(문맥 미사용). 필요 시 내부적으로 사용.
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # 모델 디바이스로 이동
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = _softmax_logits(outputs.logits)

    scores = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    label = LABELS[best_idx]
    return scores, label, float(probs[best_idx])

def _build_context_text(history: List[Dict[str, str]], current: Dict[str, str], context_size: int = 2) -> str:
    """
    직전 N개의 발화(context_size) + 현재 발화를 하나의 시퀀스로 연결.
    스피커 정보도 함께 넣어 문맥을 보존.
    history: [{"speaker": "BF", "text": "..."}, ...]
    current: {"speaker": "...", "text": "..."}
    """
    ctx = history[-context_size:] if context_size > 0 else []
    parts = []
    for utt in ctx:
        parts.append(f"[{utt['speaker']}] {utt['text']}")
    parts.append(f"[{current['speaker']}] {current['text']}")
    return " ".join(parts)

def analyze_sentence_with_context(history: List[Dict[str, str]], current: Dict[str, str], context_size: int = 2) -> Tuple[Dict[str, float], str, float]:
    """
    문맥/대화 흐름을 반영한 문장 분석.
    """
    text = _build_context_text(history, current, context_size=context_size)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = _softmax_logits(outputs.logits)

    scores = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    label = LABELS[best_idx]
    return scores, label, float(probs[best_idx])
