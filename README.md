대화(커플 챗) 문장을 입력하면 8개 감정 라벨(기쁨/설렘/실망/후회/슬픔/짜증/불안/중립) 확률 분포를 예측하고, 1분 단위 타임라인/사용자별 집계를 제공합니다.
추론 품질 향상을 위해 간단 키워드 부스팅/부정어 감쇠/확률 샤프닝/중립 복귀 휴리스틱을 선택적으로 적용할 수 있습니다.


# ✨ 주요 기능

문맥 반영 추론(직전 N개 발화 + 현재 발화)

8라벨 확률 분포 + 최종 라벨/신뢰도

sent_at 기반 1분 단위 타임라인 집계

사용자 ID 중심 집계(role_binding으로 BF/GF → 실제 userId 매핑)

휴리스틱: 키워드 부스팅, 부정어 감쇠, 샤프닝, 중립 복귀 (ON/OFF 및 강도 조절 가능)

Pydantic(v1/v2) 호환, FastAPI 문서 자동화



## 1) 기술 스택

API 서버: FastAPI (Python 3.10+ 권장)

모델/추론: Hugging Face AutoModelForSequenceClassification + KcELECTRA 파인튜닝 가중치

토크나이저: Hugging Face AutoTokenizer (special tokens [GF] [BF] [CTX] 지원)

DL 프레임워크: PyTorch

환경 변수 관리: python-dotenv

고정 라벨(8): ["기쁨","설렘","실망","후회","슬픔","짜증","불안","중립"]

## 2) 프로젝트 구조
kc_ai_app/
├─ src/
│  ├─ __init__.py
│  ├─ api.py                 # FastAPI 엔드포인트 (/health, /analyze)
│  └─ inference.py           # KcELECTRA 로드/추론/문맥 처리
├─ .env                      # API_KEY, 모델 경로 등 환경변수
├─ requirements.txt          # 의존성
└─ README.md                 


## 3) 환경 설정 (.env)
필수
API_KEY

우선순위: kc_ai_app/.env → 저장소 루트 .env
FINETUNED_MODEL_PATH/MODEL_DIR를 지정하지 않으면 기본값: models/kcelectra-base-emotion

응답(요약)

emotions[]

speaker: 실제 userId 문자열로 정규화되어 반환(가능하면 role_binding로 매핑)

scores: 8라벨 확률 분포(합=1). 표시용 반올림 포함(기본 소수 2자리)

emotion_label: 최고 확률 라벨

confidence: 최고 확률 값 (float 원본)

sent_at: 입력 값 그대로 보존

aggregated (1분 단위, 대화가 있던 분만)

timeline[i].avg_scores: 1분 내 전체 발화 평균

timeline[i].<userId>_avg_scores: 발화자별 평균(존재 시)

overall.*: 전체 구간 평균

deprecated.bf_avg_scores / gf_avg_scores: role_binding 제공 시 하위 호환 필드


## 4) 추론 파이프라인

문맥 반영: 직전 N개 발화의 텍스트를 [CTX]로 프리픽스, 현재 발화 앞에 화자 토큰([BF]/[GF])을 부여하여 모델 인퍼런스

정규화: 모델 출력 확률을 8라벨로 맵핑 → 누락 라벨 0 채움 → 합=1로 재정규화(부동소수 오차 clamp 포함)

표시용 반올림: 기본 소수 2자리(api.py 내 상수). 필요 시 상수 값만 변경

집계: sent_at의 분 단위로 버킷팅, 대화가 있었던 분만 타임라인 생성. overall은 발화 단위 평균


## 5) 인증/보안

모든 API 호출에서 x-api-key 헤더 필수

잘못된 키 → 401 Unauthorized

모델 경로는 서버 환경변수로만 주입(.env / 시스템 환경변수)


## 6) 성능/안정화 팁

모델 캐싱: inference.py의 _cached_load()로 1회 로딩 후 재사용

GPU: CUDA 환경이면 자동 사용

메모리: 긴 대화는 context_size를 2~3으로 제한(현재 기본 2)

라벨 안정화: 8라벨 고정 정규화 + clamp로 합>1, 음수 등 모든 경우 방어


## 7) 백엔드 연동 가이드(권장)

DB 설계

emotion_events (원천 발화 단위): session_id, user_id, sentence, scores_raw(JSON), label, confidence, sent_at, analyzed_at

emotion_agg_minutely (옵션): 사후 배치 집계 시 분 단위 보관

프론트 표시

scores(반올림)로 차트/게이지 등 UI 표시

툴팁/세부 패널에서는 scores_raw(원본) 사용 가능

## 8) 라이선스/크레딧

모델: KcELECTRA 기반 파인튜닝(사내/개인 학습)

토크나이저/프레임워크: Hugging Face / PyTorch

API: FastAPI

부록 A. 라벨 정책(8 고정)

라벨 변경/확장은 모델 재학습 없이 지원하지 않음(정규화가 8개 가정)

한글 표기 통일: 프로젝트 전반 ["기쁨","설렘","실망","후회","슬픔","짜증","불안","중립"] 사용


# 🧩 모델/추론 로직 개요

## inference.py

Hugging Face 포맷 모델 로드(ELECTRA 계열)

문맥 전처리: [CTX] <최근발화 ...> [BF|GF] <현재문장>

휴리스틱 파이프라인(기본 ON)

키워드 부스팅: 감정별 핵심 단어 매칭 시 확률 가산

부정어 감쇠: “아니/별로/안~/못~” 등 등장 시 긍정 계열 감쇠·부정 계열 미세 가산

샤프닝: 확률 분포를 γ-지수로 뾰족하게

중립 복귀: 최고 확률이 임계 미만이면 중립 가중

모든 단계 후 정규화로 합=1 보장

## api.py

인증/검증/반올림(표시용)

sent_at 기반 타임라인 집계 및 userId별 집계

Pydantic 스키마 정의 및 응답 직렬화


# 🛠 트러블슈팅

401 Unauthorized: x-api-key 또는 .env의 API_KEY 확인

CUDA 관련 에러: GPU 미탑재 환경이면 자동으로 CPU로 전환됩니다

환경변수 미설정: API_KEY가 없으면 서버가 부팅 시 에러 발생

모델 경로 에러: MODEL_DIR/FINETUNED_MODEL_PATH가 가리키는 폴더에 config.json, pytorch_model.bin, tokenizer.json 등 필수 파일이 있는지 확인

타임존 없는 시간: start_at/end_at 타임존 없으면 UTC로 간주

빈 타임라인: 구간 내 sent_at이 없는 발화만 들어온 경우 타임라인은 빈 배열로 반환
