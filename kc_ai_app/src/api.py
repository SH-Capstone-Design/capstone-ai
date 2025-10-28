# kc_ai_app/src/api.py
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
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
# 반올림 유틸(표시용)
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
    """
    speaker: "BF" | "GF" | 실제 userId(문자열/UUID/숫자) 허용
    """
    speaker: str
    sentence: str
    sent_at: Optional[str] = None
    emotion_label: Optional[str] = None
    confidence: Optional[float] = None
    analyzed_at: Optional[str] = None

class ChatSession(BaseModel):
    chat_session_id: str
    couple_id: Union[int, str]
    start_at: str
    end_at: str
    duration_minutes: int

class ChatRequest(BaseModel):
    chat_session: ChatSession
    emotions: List[EmotionRequest]
    # 🔑 역할-사용자 매핑: {"BF":"<userId>", "GF":"<userId>"}
    role_binding: Dict[str, str] = Field(default_factory=dict)

# ===== 응답 스키마 =====
class EmotionResponse(BaseModel):
    # ✅ 규격: speaker = 실제 userId 문자열
    speaker: Optional[str] = None
    # 참고용(프론트/백엔드 디버깅에 유용)
    userId: Optional[str] = None
    alias: Optional[str] = None  # "BF"/"GF" 등 별칭(있으면)

    sentence: str
    scores: Dict[str, float]      # 항상 8라벨 키로 반환
    emotion_label: str
    confidence: float
    analyzed_at: str
    sent_at: Optional[str] = None

# ===== 공용 유틸 =====
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

# ===== speaker → userId 해석 =====
def _looks_like_uuid_or_id(s: str) -> bool:
    if not isinstance(s, str): 
        return False
    # 매우 느슨한 검사: UUID/숫자/혼합 문자열 허용
    return True if len(s) >= 1 else False

def _resolve_user_id(emo: EmotionRequest, role_binding: Dict[str, str]) -> Optional[str]:
    spk = (emo.speaker or "").strip()
    if spk in ("BF", "GF"):
        uid = role_binding.get(spk)
        return str(uid) if uid is not None else None
    # 이미 userId를 직접 넣어준 경우
    if _looks_like_uuid_or_id(spk):
        return spk
    return None

def _alias_of(uid: Optional[str], role_binding: Dict[str, str]) -> Optional[str]:
    if uid is None:
        return None
    for k, v in role_binding.items():
        if str(v) == str(uid):
            return k  # "BF" 또는 "GF"
    return None

# ===== 집계: userId 평면 키 {userId}_avg_scores =====
def calculate_aggregated_user_centric(
    emotions: List[EmotionResponse],
    start_at: str,
    end_at: str,
    role_binding: Dict[str, str]
) -> Dict[str, Any]:
    """
    출력:
      timeline[i] = {
        "minute": <int>,
        "avg_scores": {...},
        "<userId>_avg_scores": {...},   # 사용자 수 만큼 반복
        ...
      }
      overall = {
        "avg_scores": {...},            # 전체 평균
        "<userId>_avg_scores": {...},   # 사용자 수 만큼 반복
        ...
      }
    deprecated: 선택적으로 bf/gf 하위호환 제공
    """
    _assert_inference_ready([to_dict(e) for e in emotions])

    start_dt = _parse_iso(start_at)
    end_dt   = _parse_iso(end_at)
    if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo   is None: end_dt   = end_dt.replace(tzinfo=timezone.utc)
    start_min = _to_utc_floor_min(start_dt)
    end_min   = _to_utc_floor_min(end_dt)
    total_minutes = int((end_min - start_min).total_seconds() // 60)
    if total_minutes <= 0:
        raise ValueError(f"[aggregation] 세션 시간이 0분입니다. start_at={start_at}, end_at={end_at}")

    minute_buckets: Dict[int, List[EmotionResponse]] = defaultdict(list)
    for emo in emotions:
        if not emo.sent_at:
            continue
        ts_min = _to_utc_floor_min(_parse_iso(emo.sent_at))
        minute = int((ts_min - start_min).total_seconds() // 60)
        if 0 <= minute < total_minutes:
            minute_buckets[minute].append(emo)

    if not minute_buckets:
        return {
            "interval": "1min",
            "timeline": [],
            "overall": { "avg_scores": {l: 0.0 for l in LABELS8} },
            "deprecated": {}
        }

    def _avg(sum_dict: Dict[str, float], denom: int) -> Dict[str, float]:
        return {lab: _clamp01(sum_dict.get(lab, 0.0) / max(denom, 1)) for lab in LABELS8}

    # 전체 합산
    overall_sum: Dict[str, float] = defaultdict(float)
    overall_cnt = 0
    # per-user 합산
    per_user_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_user_cnt: Dict[str, int] = defaultdict(int)

    timeline: List[Dict[str, Any]] = []

    for minute in sorted(minute_buckets.keys()):
        emos = minute_buckets[minute]

        # 분 전체
        sum_all: Dict[str, float] = defaultdict(float)
        n_all = 0
        # 분 per-user
        sum_user: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        cnt_user: Dict[str, int] = defaultdict(int)

        for emo in emos:
            sc = emo.scores
            for lab, v in sc.items():
                sum_all[lab] += v
                overall_sum[lab] += v
            n_all += 1
            overall_cnt += 1

            if emo.userId is not None:
                uid = str(emo.userId)
                for lab, v in sc.items():
                    sum_user[uid][lab] += v
                    per_user_sum[uid][lab] += v
                cnt_user[uid] += 1
                per_user_cnt[uid] += 1

        minute_obj: Dict[str, Any] = {
            "minute": minute,
            "avg_scores": _round_scores(_avg(sum_all, n_all)),
        }
        # ✅ 각 사용자에 대해 "<userId>_avg_scores" 키 생성
        for uid, c in cnt_user.items():
            minute_obj[f"{uid}_avg_scores"] = _round_scores(_avg(sum_user[uid], c))

        timeline.append(minute_obj)

    # overall
    overall_obj: Dict[str, Any] = {
        "avg_scores": _round_scores(_avg(overall_sum, overall_cnt)),
    }
    # ✅ 사용자별 키
    for uid, c in per_user_cnt.items():
        overall_obj[f"{uid}_avg_scores"] = _round_scores(_avg(per_user_sum[uid], c))

    # (선택) 하위호환(bf/gf) 제공
    deprecated_obj: Dict[str, Any] = {}
    bf_id = role_binding.get("BF")
    gf_id = role_binding.get("GF")
    if bf_id is not None and f"{bf_id}_avg_scores" in overall_obj:
        deprecated_obj["bf_avg_scores"] = overall_obj[f"{bf_id}_avg_scores"]
    if gf_id is not None and f"{gf_id}_avg_scores" in overall_obj:
        deprecated_obj["gf_avg_scores"] = overall_obj[f"{gf_id}_avg_scores"]

    return {
        "interval": "1min",
        "timeline": timeline,
        "overall": overall_obj,
        "deprecated": deprecated_obj
    }

# ===== 헬스체크 =====
@app.get("/health")
def health():
    return {"status": "ok"}

# ===== 메인 엔드포인트 =====
@app.post("/analyze")
def analyze_chat(request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    문맥 반영: 직전 N개(history)+현재 발화로 분석 (기본 N=2)
    speaker는 "BF"/"GF" 또는 실제 userId 문자열 허용.
    응답에서는 반드시 speaker에 실제 userId를 넣어 반환.
    """
    # 지연 임포트
    from .inference import analyze_sentence_with_context

    analyzed_emotions: List[EmotionResponse] = []
    history: List[Dict[str, str]] = []

    for emo in request.emotions:
        # === 추론 ===
        current = {"speaker": emo.speaker, "text": emo.sentence}
        scores, label, confidence = analyze_sentence_with_context(history, current, context_size=2)

        # 8라벨 분포 보정 및 반올림(표시용)
        scores = _normalize_labels(scores)
        scores = _round_scores(scores)

        # === speaker → 실제 userId 변환 ===
        uid = _resolve_user_id(emo, request.role_binding)   # 문자열(UUID/ID)
        alias = _alias_of(uid, request.role_binding) if uid is not None else (emo.speaker or None)

        analyzed_emotions.append(
            EmotionResponse(
                speaker=(str(uid) if uid is not None else None),  # ✅ 규격: 실제 userId
                userId=(str(uid) if uid is not None else None),
                alias=alias,
                sentence=emo.sentence,
                scores=scores,
                emotion_label=label,
                confidence=float(confidence),
                analyzed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                sent_at=emo.sent_at,
            )
        )
        # 컨텍스트는 원본 발화자 문자열 유지해도 무방(모델 내부 문맥용)
        history.append(current)

    # === 집계 ===
    aggregated = calculate_aggregated_user_centric(
        emotions=analyzed_emotions,
        start_at=request.chat_session.start_at,
        end_at=request.chat_session.end_at,
        role_binding=request.role_binding,
    )

    return {
        "chat_session": to_dict(request.chat_session),
        "emotions": [to_dict(e) for e in analyzed_emotions],
        "aggregated": aggregated
    }
