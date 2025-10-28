# kc_ai_app/src/inference.py
# -*- coding: utf-8 -*-
import os
import functools
from typing import List, Dict, Union, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== 라벨: 학습과 동일(8클래스) =====
LABELS: List[str] = ["기쁨", "설렘", "실망", "후회", "슬픔", "짜증", "불안", "중립"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

SPECIAL_TOKENS = {"additional_special_tokens": ["[GF]", "[BF]", "[CTX]"]}

def ensure_special_tokens(tokenizer, model) -> None:
    """
    토크나이저에 [GF]/[BF]/[CTX]가 없으면 추가하고,
    모델 임베딩 크기와 토크나이저 vocab 크기가 다르면 리사이즈한다.
    (훈련 때 이미 포함되어 있으면 아무 것도 하지 않음)
    """
    need_add = []
    unk_id = tokenizer.unk_token_id
    for t in SPECIAL_TOKENS["additional_special_tokens"]:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid == unk_id:
            need_add.append(t)
    if need_add:
        tokenizer.add_special_tokens({"additional_special_tokens": need_add})
    # 모델 임베딩 크기와 토크나이저 vocab 크기 동기화
    if hasattr(model, "resize_token_embeddings"):
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

def load_model_and_tokenizer(
    model_dir_or_id: str,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """
    HF 포맷(로컬 폴더/허깅페이스 ID)에서 모델/토크나이저 로드.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir_or_id,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    ensure_special_tokens(tokenizer, model)
    model.eval().to(device)
    return tokenizer, model, device

# ===== 모델 로딩 캐시(요청마다 재로딩 방지) =====
@functools.lru_cache(maxsize=1)
def _cached_load(model_id: str):
    return load_model_and_tokenizer(model_id)

def preprocess_texts(
    texts: List[str],
    speaker: Optional[str] = None,
    ctx: Optional[str] = None
) -> List[str]:
    """
    간단 전처리: (선택) 화자토큰/[CTX] 컨텍스트를 앞에 붙임.
    - speaker: "GF" | "BF" | None
    - ctx: 직전 대화 맥락 문자열(없으면 무시)
    """
    prefix_spk = f"[{speaker}] " if speaker in ("GF", "BF") else ""
    prefix_ctx = f"[CTX] {ctx} " if ctx else ""
    return [f"{prefix_ctx}{prefix_spk}{t}".strip() for t in texts]

@torch.inference_mode()
def predict_proba(
    tokenizer, model, device, texts: List[str],
    max_length: int = 256, padding: str = "longest", batch_size: int = 32,
    **_  # ← 호출부에서 넘어올 수 있는 labels 등 예기치 않은 키워드 무시
) -> List[Dict[str, float]]:
    """
    다중 문장 확률 예측.
    반환: [{라벨: 확률, ...}, ...]
    """
    softmax = torch.nn.Softmax(dim=-1)
    out: List[Dict[str, float]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            padding=padding,
            max_length=max_length,
        )
        # Electra 계열은 token_type_ids가 모두 0 → 전달 생략 가능(있어도 무방)
        enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids"}

        logits = model(**enc).logits  # [B, C]
        probs = softmax(logits).cpu().tolist()

        for p in probs:
            out.append({LABELS[j]: float(p[j]) for j in range(len(LABELS))})
    return out

def predict_label(
    tokenizer, model, device, text: str,
    speaker: Optional[str] = None, ctx: Optional[str] = None,
    **kw,
) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    단일 문장 예측(+선택적 화자/컨텍스트)
    """
    proc = preprocess_texts([text], speaker=speaker, ctx=ctx)
    probs = predict_proba(tokenizer, model, device, proc, **kw)[0]
    label = max(probs, key=probs.get)
    return {"label": label, "score": probs[label], "probs": probs}

# ---- compat shim: API가 기대하는 심볼 제공 ----
def analyze_sentence_with_context(
    history: Optional[List[Dict[str, str]]],
    current: Dict[str, str],
    context_size: int = 2,
    labels: Optional[List[str]] = None,
    **kwargs,
):
    """
    history: [{"speaker":"BF|GF", "text" 또는 "sentence": "..."} ...]
    current: {"speaker":"...", "text" 또는 "sentence":"..."}
    반환: (scores: Dict[str, float], label: str, confidence: float)
    """
    # 0) 모델 경로/ID 결정 (로컬이든 허브 ID든 문자열이면 됨)
    model_id = os.getenv("MODEL_DIR") or os.getenv("FINETUNED_MODEL_PATH")
    if not model_id:
        raise RuntimeError(
            "MODEL_DIR 또는 FINETUNED_MODEL_PATH 환경변수가 필요합니다."
        )

    # 1) 모델/토크나이저/디바이스 로드 (캐시 사용)
    tokenizer, model, device = _cached_load(model_id)

    # 2) 문맥 구성: 최근 N개(history) + 현재 발화
    def _t(x: Optional[Dict[str, str]]) -> str:
        if not x:
            return ""
        return x.get("text") or x.get("sentence") or ""
    hist = (history or [])[-int(context_size):]
    ctx_text = " ".join([_t(h) for h in hist if _t(h)])
    cur_text = _t(current)
    speaker = (current or {}).get("speaker")

    # 3) 후보 라벨 세트(없으면 기본 LABELS 사용) — 현재 모델 추론에는 직접 미사용
    _ = labels or LABELS  # 유지: 호출부 호환성(넘어와도 에러 안 나게)

    # 4) 실제 추론 (predict_label은 Dict 반환: {"label","score","probs"})
    out = predict_label(
        tokenizer=tokenizer,
        model=model,
        device=device,
        text=cur_text,
        speaker=speaker,
        ctx=ctx_text,
        labels=labels,  # 넘어올 수 있으므로 그대로 전달( predict_proba에서 **_로 무시 )
    )
    # api.py의 기대 형태에 맞게 변환
    scores = out["probs"]
    label = out["label"]
    confidence = out["score"]
    return scores, label, confidence
