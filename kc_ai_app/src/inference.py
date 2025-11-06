# kc_ai_app/src/inference.py
# -*- coding: utf-8 -*-
import os
import re
import functools
from typing import List, Dict, Union, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== 라벨: 학습과 동일(8클래스) =====
LABELS: List[str] = ["기쁨", "설렘", "실망", "후회", "슬픔", "짜증", "불안", "중립"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

SPECIAL_TOKENS = {"additional_special_tokens": ["[GF]", "[BF]", "[CTX]"]}

# ===== (추가) 휴리스틱 파라미터: .env 없으면 기본값 사용 =====
def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.getenv(key, default))
        return v
    except Exception:
        return default

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip() in ("1", "true", "True", "yes", "YES")

ENABLE_HEURISTIC: bool = _env_bool("ENABLE_HEURISTIC", True)  # 휴리스틱 켜기/끄기
KEYWORD_ALPHA: float   = _env_float("KEYWORD_ALPHA", 0.22)    # 키워드 가산 강도
NEGATION_BETA: float   = _env_float("NEGATION_BETA", 0.15)    # 부정 감쇠 강도
SHARP_GAMMA: float     = _env_float("SHARP_GAMMA", 0.80)      # 확률 샤프닝(γ<1 뾰족)
NEUTRAL_TH: float      = _env_float("NEUTRAL_TH", 0.45)       # 최고확률이 임계 미만이면 중립 복귀

# ===== (추가) 간단 키워드 사전 =====
# 필요 시 자유롭게 보강하세요.
LEXICON: Dict[str, List[str]] = {
    "기쁨": ["좋아", "좋다", "행복", "즐거", "완전 좋", "짱", "대박", "웃겨", "뿌듯", "사랑해", "좋지"],
    "설렘": ["설레", "두근", "기대", "가보자", "가자", "재밌겠", "좋을듯", "신나", "콜~", "오키", "좋겠다"],
    "실망": ["실망", "아쉽", "별로였", "기대 이하", "그닥", "실망스"],
    "후회": ["후회", "그럴걸", "그랬어야", "괜히", "다음엔", "다시는"],
    "슬픔": ["슬프", "우울", "눈물", "속상", "서럽", "힘들었", "괜찮지 않"],
    "짜증": ["짜증", "빡치", "열받", "개", "씨발", "젠장", "피곤하", "귀찮", "노답", "답답"],
    "불안": ["불안", "걱정", "괜찮을까", "무섭", "걱정되", "긴장", "찝찝", "불편하"],
    "중립": ["그래", "응", "알겠", "확인", "오케이", "그러", "그래서", "음", "웅", "넵"],
}

# ===== (추가) 부정/부정어 패턴 =====
NEG_PATTERNS: List[re.Pattern] = [
    re.compile(p) for p in [
        r"\b아니\b", r"\b아냐\b", r"\b아닌\b", r"\b아녔", r"\b싫", r"\b별로\b",
        r"\b안\s", r"\b못\s", r"\b없어\b", r"\b노노\b", r"\bㄴㄴ\b"
    ]
]

def _contains_negation(text: str) -> bool:
    for pat in NEG_PATTERNS:
        if pat.search(text):
            return True
    return False

def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def _normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
    # 음수를 방지하고, 합 1로 정규화
    safe = {k: max(0.0, float(v)) for k, v in dist.items()}
    s = sum(safe.values())
    if s <= 0:
        # 완전히 0이면 균등 분포(중립 가중)로 복귀
        n = len(LABELS)
        return {k: (1.0 / n) for k in LABELS}
    return {k: v / s for k, v in safe.items()}

def _sharpen(dist: Dict[str, float], gamma: float) -> Dict[str, float]:
    # 확률 샤프닝: p_i' = p_i^γ / sum_j p_j^γ  (γ<1 → 뾰족)
    if gamma == 1.0:
        return dist
    powered = {k: (v ** gamma) for k, v in dist.items()}
    return _normalize_dist(powered)

def _keyword_boost(text: str, dist: Dict[str, float], alpha: float) -> Dict[str, float]:
    if alpha <= 0:
        return dist
    t = text.lower()
    bumped = dist.copy()
    for label, keys in LEXICON.items():
        for kw in keys:
            if kw.lower() in t:
                bumped[label] = bumped.get(label, 0.0) + alpha
                break  # 한 라벨에 여러 키워드가 있어도 1회만 가산
    return _normalize_dist(bumped)

def _negation_adjust(text: str, dist: Dict[str, float], beta: float) -> Dict[str, float]:
    if beta <= 0 or not _contains_negation(text):
        return dist
    # 부정어가 있으면 긍정 계열(기쁨/설렘)을 감쇠하고, 부정 계열(짜증/실망/불안/슬픔)을 미세 가산
    adj = dist.copy()
    pos = ["기쁨", "설렘"]
    neg = ["짜증", "실망", "불안", "슬픔"]
    for p in pos:
        adj[p] = max(0.0, adj.get(p, 0.0) * (1.0 - beta))
    for n in neg:
        adj[n] = adj.get(n, 0.0) + (beta / len(neg))
    return _normalize_dist(adj)

def _neutral_fallback(dist: Dict[str, float], th: float) -> Dict[str, float]:
    # 최고 확률이 임계 미만이면 중립으로 소폭 복귀(너무 애매하면 중립 쪽으로)
    if th <= 0:
        return dist
    best_label = max(dist, key=dist.get)
    if dist[best_label] >= th:
        return dist
    nudged = dist.copy()
    # 중립에 작은 보너스, 그 외는 균등 감쇠
    bonus = 0.05
    nudged["중립"] = nudged.get("중립", 0.0) + bonus
    # 정규화로 자동 보정
    return _normalize_dist(nudged)

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

def _apply_heuristics(text: str, probs: Dict[str, float]) -> Dict[str, float]:
    """
    (추가) 체감 성능 향상을 위한 휴리스틱 파이프라인.
      1) 키워드 부스팅
      2) 부정어 감쇠/가산
      3) 샤프닝
      4) 중립 복귀(최고확률이 낮으면)
    """
    if not ENABLE_HEURISTIC:
        return probs

    dist = _normalize_dist(probs)
    # 1) 키워드 부스팅
    dist = _keyword_boost(text, dist, KEYWORD_ALPHA)
    # 2) 부정어 감쇠/가산
    dist = _negation_adjust(text, dist, NEGATION_BETA)
    # 3) 샤프닝
    dist = _sharpen(dist, SHARP_GAMMA)
    # 4) 중립 복귀
    dist = _neutral_fallback(dist, NEUTRAL_TH)
    return dist

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

    # (추가) 휴리스틱 적용 — 실제 입력 원문(text)을 기준으로 보정
    probs = _apply_heuristics(text, probs)

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
