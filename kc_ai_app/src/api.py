# api.py
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timezone
import dateutil.parser
import os
from dotenv import load_dotenv
from pathlib import Path
from kc_ai_app.src.label_map import LABEL_DISPLAY_MAP  # 사용자 표시 라벨 매핑

# ==============================
# 환경 변수 / 인증 (먼저, 명시 경로로 로드!)
# ==============================
ROOT = Path(__file__).resolve().parents[1]            # C:\Users\Acer\AI
ENV_KC_APP = ROOT / "kc_ai_app" / ".env"              # 우선 후보
ENV_ROOT   = ROOT / ".env"                            # 대안 후보

# 우선순위: kc_ai_app/.env -> 루트 .env
if ENV_KC_APP.exists():
    load_dotenv(ENV_KC_APP)
elif ENV_ROOT.exists():
    load_dotenv(ENV_ROOT)
else:
    # 둘 다 없으면 넘어가지만, 아래에서 필수값 검증함
    pass

# 필수: API_KEY
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 API_KEY가 설정되지 않았습니다. (kc_ai_app/.env 또는 루트 .env 확인)")

# 선택: FINETUNED_MODEL_PATH (없으면 로컬 기본값으로 세팅)
if not os.getenv("FINETUNED_MODEL_PATH"):
    os.environ["FINETUNED_MODEL_PATH"] = str((ROOT / "models" / "kcelectra-base-emotion").resolve())

def get_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key

# ==============================
# FastAPI 앱
# ==============================
app = FastAPI(title="Emotion Analysis API")

# ===== Pydantic v1/v2 호환 직렬화 헬퍼 =====
def to_dict(model: BaseModel) -> Dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

# ===== 요청 스키마 =====
class EmotionRequest(BaseModel):
    speaker: str
    sentence: str
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
    scores: Dict[str, float]
    emotion_label: str
    confidence: float
    analyzed_at: str

# ===== 집계(1분 단위) =====
def calculate_aggregated(emotions: List[EmotionResponse], start_at: str) -> Dict:
    start_time = dateutil.parser.isoparse(start_at)
    minute_buckets: Dict[int, List[EmotionResponse]] = {}

    # 1️⃣ 감정별 시간 버킷 분류
    for emo in emotions:
        analyzed_time = dateutil.parser.isoparse(emo.analyzed_at)
        minute = int((analyzed_time - start_time).total_seconds() // 60)
        minute_buckets.setdefault(minute, []).append(emo)

    timeline, overall_scores, overall_bf, overall_gf = [], {}, {}, {}
    count_bf, count_gf = 0, 0

    # 2️⃣ 각 분 단위 집계
    for minute in sorted(minute_buckets.keys()):
        emo_list = minute_buckets[minute]
        avg_scores, bf_scores, gf_scores = {}, {}, {}
        bf_count = gf_count = 0

        for emo in emo_list:
            # 세부 감정을 대표 감정으로 매핑
            for label, value in emo.scores.items():
                display_label = LABEL_DISPLAY_MAP.get(label, label)
                avg_scores[display_label] = avg_scores.get(display_label, 0.0) + value

            # BF / GF 별도 집계
            target_scores = bf_scores if emo.speaker == "BF" else gf_scores
            if emo.speaker == "BF": bf_count += 1
            if emo.speaker == "GF": gf_count += 1

            for label, value in emo.scores.items():
                display_label = LABEL_DISPLAY_MAP.get(label, label)
                target_scores[display_label] = target_scores.get(display_label, 0.0) + value

        # 평균 계산
        for label in avg_scores:
            avg_scores[label] /= max(len(emo_list), 1)
        for label in bf_scores:
            bf_scores[label] /= max(bf_count, 1)
        for label in gf_scores:
            gf_scores[label] /= max(gf_count, 1)

        timeline.append({
            "minute": minute,
            "avg_scores": avg_scores,
            "bf_avg_scores": bf_scores,
            "gf_avg_scores": gf_scores
        })

        # 전체 평균 계산용 누적
        for label, value in avg_scores.items():
            overall_scores[label] = overall_scores.get(label, 0.0) + value
        if bf_count > 0:
            count_bf += 1
            for label, value in bf_scores.items():
                overall_bf[label] = overall_bf.get(label, 0.0) + value
        if gf_count > 0:
            count_gf += 1
            for label, value in gf_scores.items():
                overall_gf[label] = overall_gf.get(label, 0.0) + value

    # 전체 평균 계산
    if len(timeline) > 0:
        for label in overall_scores:
            overall_scores[label] /= len(timeline)
    if count_bf > 0:
        for label in overall_bf:
            overall_bf[label] /= count_bf
    if count_gf > 0:
        for label in overall_gf:
            overall_gf[label] /= count_gf

    return {
        "interval": "1min",
        "timeline": timeline,
        "overall": {
            "avg_scores": overall_scores,
            "bf_avg_scores": overall_bf,
            "gf_avg_scores": overall_gf
        }
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
    """
    # 지연 임포트: 모델 경로 문제로 모듈 import 단계에서 죽는 것 방지
    from .inference import analyze_sentence_with_context

    analyzed_emotions: List[EmotionResponse] = []
    history: List[Dict[str, str]] = []

    for emo in request.emotions:
        current = {"speaker": emo.speaker, "text": emo.sentence}
        scores, label, confidence = analyze_sentence_with_context(history, current, context_size=2)
        display_label = LABEL_DISPLAY_MAP.get(label, label)
        analyzed_emotions.append(
            EmotionResponse(
                speaker=emo.speaker,
                sentence=emo.sentence,
                scores=scores,
                emotion_label=display_label,
                confidence=confidence,
                analyzed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
        )
        history.append(current)

    aggregated = calculate_aggregated(analyzed_emotions, request.chat_session.start_at)

    return {
        "chat_session": to_dict(request.chat_session),
        "emotions": [to_dict(e) for e in analyzed_emotions],
        "aggregated": aggregated
    }
