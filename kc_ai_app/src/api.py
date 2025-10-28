# kc_ai_app/src/api.py
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from pathlib import Path
import dateutil.parser
from collections import defaultdict

# ==============================
# 환경 변수 / 인증 (먼저, 명시 경로로 로드!)
# ==============================
ROOT = Path(__file__).resolve().parents[1]            # .../AI
ENV_KC_APP = ROOT / "kc_ai_app" / ".env"              # 우선 후보
ENV_ROOT   = ROOT / ".env"                            # 대안 후보

if ENV_KC_APP.exists():
    load_dotenv(ENV_KC_APP)
elif ENV_ROOT.exists():
    load_dotenv(ENV_ROOT)
else:
    pass

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 API_KEY가 설정되지 않았습니다. (kc_ai_app/.env 또는 루트 .env 확인)")

if not os.getenv("FINETUNED_MODEL_PATH"):
    os.environ["FINETUNED_MODEL_PATH"] = str((ROOT / "models" / "kcelectra-base-emotion").resolve())

def get_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key

# ==============================
# 라벨: 8개 고정
# ==============================
LABELS8: List[str] = ["기쁨", "설렘", "실망", "후회", "슬픔", "짜증", "불안", "중립"]

def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

# ==============================
# 반올림 유틸(표시용) ★ 추가
# ==============================
ROUND_DIGITS = 2  # 소수 2자리 고정

def _round_scores(d: Dict[str, float], nd: int = ROUND_DIGITS) -> Dict[str, float]:
    return {k: round(float(v), nd) for k, v in d.items()}

def _round_nested(d: Dict[str, Dict[str, float]], nd: int = ROUND_DIGITS) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, sub in d.items():
        out[k] = _round_scores(sub, nd) if isinstance(sub, dict) else sub
    return out

# ==============================
# FastAPI 앱
# ==============================
app = FastAPI(title="Emotion Analysis API")

def to_dict(model: BaseModel) -> Dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

# ===== 요청 스키마 =====
class EmotionRequest(BaseModel):
    speaker: str
    sentence: str
    sent_at: Optional[str] = None
    emotion_label: Optional[str] = None
    confidence: Optional[float] = None
    analyzed_at: Optional[str] = None

class ChatSession(BaseModel):
    chat_session_id: str
    couple_id: int
    start_at: str
    end_at: str
    duration_minutes: int

class ChatRequest(BaseModel):
    chat_session: ChatSession
    emotions: List[EmotionRequest]

# ===== 응답 스키마 =====
class EmotionResponse(BaseModel):
    speaker: str
    sentence: str
    scores: Dict[str, float]      # 항상 8라벨 키로 반환
    emotion_label: str
    confidence: float
    analyzed_at: str
    sent_at: Optional[str] = None

# ===== 집계 유틸 =====
def _parse_iso(s: str):
    return dateutil.parser.isoparse(s)

def _to_utc_floor_min(dt):
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)

def _get(emo: Any, key: str):
    if hasattr(emo, key): return getattr(emo, key)
    if isinstance(emo, dict): return emo.get(key)
    return None

def _get_ts(emo: Any):
    return _get(emo, "sent_at")

def _get_scores(emo: Any) -> Dict[str, float]:
    s = _get(emo, "scores") or {}
    return s

def _get_speaker(emo: Any) -> str:
    return _get(emo, "speaker")

def _normalize_labels(scores: Dict[str, float]) -> Dict[str, float]:
    """
    8라벨 고정 분포로 정규화:
      - 8키 외의 키는 무시
      - 누락된 키는 0으로 채움
      - 합=1로 재정규화
    """
    merged = {k: float(scores.get(k, 0.0)) for k in LABELS8}
    s = sum(merged.values())
    if s > 0:
        merged = {k: _clamp01(v / s) for k, v in merged.items()}
    else:
        merged = {k: 0.0 for k in LABELS8}
    return merged

def _assert_inference_ready(emotions: List[Any]) -> None:
    missing = [i for i, e in enumerate(emotions) if not _get_scores(e)]
    if missing:
        raise ValueError(
            f"[aggregation] 추론 이후 데이터가 아님: scores 없는 레코드 {len(missing)}개 "
            f"(예: indices={missing[:10]}). 먼저 감정추론을 수행하세요."
        )

def calculate_aggregated(emotions: List[Any], start_at: str, end_at: str) -> Dict:
    _assert_inference_ready(emotions)

    start_dt = _parse_iso(start_at)
    end_dt   = _parse_iso(end_at)
    if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo   is None: end_dt   = end_dt.replace(tzinfo=timezone.utc)
    start_min = _to_utc_floor_min(start_dt)
    end_min   = _to_utc_floor_min(end_dt)
    total_minutes = int((end_min - start_min).total_seconds() // 60)
    if total_minutes <= 0:
        raise ValueError(f"[aggregation] 세션 시간이 0분입니다. start_at={start_at}, end_at={end_at}")

    labels = LABELS8

    minute_buckets: Dict[int, List[Any]] = defaultdict(list)
    for emo in emotions:
        ts = _get_ts(emo)
        if not ts:
            continue
        ts_min = _to_utc_floor_min(_parse_iso(ts))
        minute = int((ts_min - start_min).total_seconds() // 60)
        if 0 <= minute < total_minutes:
            minute_buckets[minute].append(emo)

    if not minute_buckets:
        return {
            "interval": "1min",
            "timeline": [],
            "overall": {
                "avg_scores": {l:0.0 for l in labels},
                "bf_avg_scores": {l:0.0 for l in labels},
                "gf_avg_scores": {l:0.0 for l in labels},
            }
        }

    timeline = []
    overall_sum: Dict[str, float] = defaultdict(float)
    overall_bf_sum: Dict[str, float] = defaultdict(float)
    overall_gf_sum: Dict[str, float] = defaultdict(float)
    overall_cnt = overall_bf_cnt = overall_gf_cnt = 0

    for minute in sorted(minute_buckets.keys()):
        emos = minute_buckets[minute]
        sum_all: Dict[str, float] = defaultdict(float)
        sum_bf: Dict[str, float] = defaultdict(float)
        sum_gf: Dict[str, float] = defaultdict(float)
        cnt_bf = cnt_gf = 0

        for emo in emos:
            spk = _get_speaker(emo)
            scores = _normalize_labels(_get_scores(emo))

            for lab, fv in scores.items():
                sum_all[lab] += fv
                overall_sum[lab] += fv; overall_cnt += 1
                if spk == "BF":
                    sum_bf[lab] += fv; cnt_bf += 1
                    overall_bf_sum[lab] += fv; overall_bf_cnt += 1
                elif spk == "GF":
                    sum_gf[lab] += fv; cnt_gf += 1
                    overall_gf_sum[lab] += fv; overall_gf_cnt += 1

        n = max(len(emos), 1)
        avg_scores = {lab: _clamp01(sum_all.get(lab, 0.0) / n) for lab in labels}
        bf_avg = ({lab: _clamp01(sum_bf.get(lab, 0.0) / cnt_bf) for lab in labels} if cnt_bf > 0 else {})
        gf_avg = ({lab: _clamp01(sum_gf.get(lab, 0.0) / cnt_gf) for lab in labels} if cnt_gf > 0 else {})

        # ★ 표시용 반올림 적용
        avg_scores = _round_scores(avg_scores)
        if bf_avg: bf_avg = _round_scores(bf_avg)
        if gf_avg: gf_avg = _round_scores(gf_avg)

        timeline.append({
            "minute": minute,
            "avg_scores": avg_scores,
            "bf_avg_scores": bf_avg,
            "gf_avg_scores": gf_avg
        })

    def _avg(sum_dict: Dict[str, float], denom: int) -> Dict[str, float]:
        return {lab: _clamp01(sum_dict.get(lab, 0.0) / denom) for lab in labels} if denom > 0 else {lab: 0.0 for lab in labels}

    overall = {
        "avg_scores": _avg(overall_sum, overall_cnt),
        "bf_avg_scores": _avg(overall_bf_sum, overall_bf_cnt),
        "gf_avg_scores": _avg(overall_gf_sum, overall_gf_cnt),
    }

    # ★ overall도 반올림
    overall = _round_nested(overall)

    return {"interval": "1min", "timeline": timeline, "overall": overall}

# ===== 헬스체크 =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 메인 엔드포인트 =====
@app.post("/analyze")
def analyze_chat(request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    문맥 반영: 직전 N개(history)+현재 발화로 분석 (기본 N=2)
    """
    # 지연 임포트
    from .inference import analyze_sentence_with_context

    analyzed_emotions: List[EmotionResponse] = []
    history: List[Dict[str, str]] = []

    for emo in request.emotions:
        current = {"speaker": emo.speaker, "text": emo.sentence}
        scores, label, confidence = analyze_sentence_with_context(history, current, context_size=2)

        # 8라벨 분포로 보정
        scores = _normalize_labels(scores)
        # ★ 개별 문장 score 반올림(표시용)
        scores = _round_scores(scores)

        analyzed_emotions.append(
            EmotionResponse(
                speaker=emo.speaker,
                sentence=emo.sentence,
                scores=scores,
                emotion_label=label,
                confidence=float(confidence),
                analyzed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                sent_at=emo.sent_at,
            )
        )
        history.append(current)

    aggregated = calculate_aggregated(
        [to_dict(e) for e in analyzed_emotions],
        request.chat_session.start_at,
        request.chat_session.end_at,
    )

    return {
        "chat_session": to_dict(request.chat_session),
        "emotions": [to_dict(e) for e in analyzed_emotions],
        "aggregated": aggregated
    }
