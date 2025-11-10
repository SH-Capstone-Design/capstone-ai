# ConnectBeat – 커플 대화 감정 분석 AI

- 대화(커플 챗) 문장을 입력하면 8개 감정 **(기쁨/설렘/실망/후회/슬픔/짜증/불안/중립)** 확률 분포를 예측합니다.
- `sent_at` 기반으로 1분 단위 타임라인/사용자별 집계를 제공합니다.
- 추론 품질 향상을 위해 키워드 부스팅 / 부정어 감쇠 / 확률 샤프닝 / 중립 복귀 휴리스틱을 선택적으로 적용할 수 있습니다.

## 주요 기능

- 문맥 반영 추론(직전 N개 발화 + 현재 발화)
- 8라벨 확률 분포 + 최종 라벨/신뢰도
- `sent_at` 기반 1분 단위 타임라인 집계
- 사용자 ID 중심 집계(`role_binding`으로 BF/GF → 실제 userId 매핑)
- 휴리스틱: 키워드 부스팅, 부정어 감쇠, 샤프닝, 중립 복귀 (ON/OFF 및 강도 조절)
- Pydantic(v1/v2) 호환, FastAPI 문서 자동화

## 기술 스택

- API 서버: FastAPI (Python 3.10+ 권장)
- 모델/추론: Hugging Face AutoModelForSequenceClassification + KcELECTRA 파인튜닝 가중치
- 토크나이저: Hugging Face AutoTokenizer (special tokens `[GF] [BF] [CTX]` 지원)
- DL 프레임워크: PyTorch
- 환경 변수: python-dotenv
- 고정 라벨(8): ["기쁨","설렘","실망","후회","슬픔","짜증","불안","중립"]

## 프로젝트 구조

- 아래 트리는 공백 정렬된 코드블록입니다. 그대로 복사해도 모양이 유지됩니다.

```text
kc_ai_app/
├─ src/
│  ├─ __init__.py
│  ├─ api.py               # FastAPI 엔드포인트 (/health, /analyze), 집계/반올림/검증
│  └─ inference.py         # 모델 로드/추론/문맥 처리, 휴리스틱(키워드/부정어/샤프닝/중립복귀)
├─ models/
│  └─ kcelectra-base-emotion/   # config.json, tokenizer.json, pytorch_model.bin ...
├─ app.py                   # uvicorn 엔트리포인트 (from src.api import app)
├─ .env                     # API_KEY, FINETUNED_MODEL_PATH 등
├─ requirements.txt
└─ README.md
```

## 환경 설정 (.env 예시)

```ini
# 필수
API_KEY=your-secret-key

# 모델 경로: 로컬 디렉토리 또는 허깅페이스 모델 ID
# 미설정 시 기본값: models/kcelectra-base-emotion
FINETUNED_MODEL_PATH=JeongMin05/kcelectra-base-chat-emotion

# (선택) 휴리스틱 ON/OFF 및 강도
HEURISTIC_KEYWORD_BOOST=true
BOOST_FACTOR=0.18

HEURISTIC_NEGATION_DAMPING=true
NEG_DAMP_FACTOR=0.12

HEURISTIC_SHARPENING=true
SHARPEN_GAMMA=1.25

HEURISTIC_NEUTRAL_FALLBACK=true
NEUTRAL_THRESHOLD=0.33
```

## 실행 방법

```bash
# 1) 가상환경
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

# 2) 의존성 설치
pip install -r kc_ai_app/requirements.txt

# 3) 서버 실행 (리포지토리 루트에서)
uvicorn kc_ai_app.app:app --host 0.0.0.0 --port 8081 --reload
```

## API

- 헬스체크
  - GET /health
  - Response
    ```json
    { "status": "ok" }
    ```

- 감정 분석
  - POST /analyze
  - Header
    - x-api-key: <API_KEY>
  - Request (요약)
    ```json
    {
      "chat_session": {
        "chat_session_id": "sess1",
        "couple_id": "c1",
        "start_at": "2025-09-26T12:00:00Z",
        "end_at": "2025-09-26T12:05:00Z",
        "duration_minutes": 5
      },
      "role_binding": { "BF": "u123", "GF": "u999" },
      "emotions": [
        { "speaker": "BF", "sentence": "내일 뭐할까?", "sent_at": "2025-09-26T12:00:05Z" }
      ]
    }
    ```
  - Response (요약)
    - emotions[]
      - speaker: 실제 userId 문자열(가능하면 role_binding로 매핑)
      - scores: 8라벨 확률(합=1), 표시용 반올림(기본 소수 2자리)
      - emotion_label: 최고 확률 라벨
      - confidence: 최고 확률 값(float, 원본)
      - sent_at: 입력 값 그대로 보존
    - aggregated (1분 단위, 대화가 있던 분만)
      - timeline[i].avg_scores: 해당 분 전체 발화 평균
      - timeline[i].<userId>_avg_scores: 해당 분 사용자별 평균(존재 시)
      - overall.*: 전체 구간 평균
      - deprecated.bf_avg_scores / gf_avg_scores: role_binding 제공 시 하위 호환

## 추론 파이프라인

- 문맥 반영: 직전 N개 발화를 `[CTX]` 프리픽스로, 현재 발화 앞에는 화자 토큰(`[BF]`/`[GF]`)을 부여하여 인퍼런스
- 정규화: 모델 출력 확률을 8라벨로 매핑 → 누락 라벨 0 채움 → 합=1 재정규화(부동소수 clamp)
- 표시용 반올림: 기본 소수 2자리(`api.py` 상수)
- 집계: `sent_at`의 “분” 단위로 버킷팅, 대화가 있었던 분만 타임라인 생성(Overall은 발화 단위 평균)

## 휴리스틱(선택)

- 키워드 부스팅: 감정별 핵심 단어 포함 시 해당 라벨 확률 가산
- 부정어 감쇠: “아니/별로/안~/못~” 등 부정 표현 등장 시 긍정계열 감쇠·부정계열 미세 가산
- 샤프닝: 분포에 γ-지수 적용(값이 클수록 뾰족)
- 중립 복귀: 최고 확률이 임계 미만이면 중립 가중
- 모든 단계 후 재정규화로 합=1 보장
- 설정은 `.env`에서 ON/OFF 및 강도 조절 가능

## 보안

- 모든 API 호출에 `x-api-key` 필수 (불일치 시 401 Unauthorized)
- 모델 경로는 환경변수로만 주입(.env / 시스템 환경변수)

## 백엔드 연동 가이드(권장)

- DB 스키마(예시)
  ```text
  emotion_events (원천 발화)
  - session_id, user_id, sentence, scores_raw(JSON), label, confidence, sent_at, analyzed_at

  emotion_agg_minutely (옵션, 배치 집계 보관)
  - session_id, minute, avg_scores(JSON), user_avg_scores(JSON), ...
  ```
- 프론트 표시
  - 차트/게이지 등 표시에는 반올림된 `scores` 사용
  - 툴팁/세부 패널에서는 `scores_raw`(원본) 사용 권장

## 라이선스 / 크레딧

- 모델: KcELECTRA 기반 파인튜닝(개인/사내 학습)
- 프레임워크: Hugging Face / PyTorch
- API: FastAPI
- 라벨 정책(8 고정)
  - 라벨 변경/확장은 모델 재학습 없이 지원하지 않음
  - 한글 표기 통일: ["기쁨","설렘","실망","후회","슬픔","짜증","불안","중립"]

## 트러블슈팅

- 401 Unauthorized: `x-api-key` 또는 `.env`의 `API_KEY` 확인
- CUDA 에러: GPU 미탑재 환경이면 자동으로 CPU 사용
- 모델 경로 에러: `FINETUNED_MODEL_PATH`(또는 `MODEL_DIR`) 폴더에 `config.json`, `tokenizer.json`, `pytorch_model.bin` 포함 확인
- 타임존 미포함 시간: `start_at`/`end_at` 타임존 없으면 UTC로 간주
- 빈 타임라인: 구간 내 `sent_at`이 없는 발화만 들어오면 타임라인은 빈 배열

## 엔트리포인트 팁 (`app.py`)

```python
# kc_ai_app/app.py
from src.api import app
```

```bash
uvicorn kc_ai_app.app:app --host 0.0.0.0 --port 8081 --reload
```
