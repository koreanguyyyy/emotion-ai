# capstone-detection

감정 인식 멀티모달 Proactive AI 에이전트 — 로컬 감지 모듈

사용자의 표정, 음성, 텍스트를 실시간으로 감지하고 스트레스를 분석하여 AI 서버(`emotion-ai`)로 전달하는 로컬 PC 실행 모듈입니다.

---

## 전체 시스템 구조

```
capstone-detection (이 레포)
    DeepFace (표정 감지)
    Whisper STT (음성 → 텍스트)
    VAD 스트레스 계산 (로컬 연산)
        ↓ HTTP POST
capstone-ai
    LLM 스트레스 보정
    LangGraph 감정 에이전트
    추천 생성
        ↓
capstone-backend
    Spring Boot
    추천 이력 저장
```

---

## 프로젝트 구조

```
capstone-detection/
├── main.py                     # 진입점 — 스레드 관리 및 전체 흐름 제어
│
├── detection/                  # 멀티모달 감지 모듈
│   ├── __init__.py
│   ├── face_detector.py        # DeepFace 표정 감지 (웹캠)
│   └── stt_detector.py         # Whisper STT 음성 인식 (fixed / vad 모드)
│
├── ai/                         # 로컬 AI 처리
│   ├── __init__.py
│   ├── stress_calculator.py    # VAD 기반 스트레스 계산 (로컬 연산)
│   ├── llm_client.py           # capstone-ai 서버로 HTTP 요청
│   └── cooldown.py             # 개입 쿨다운 관리
│
├── state/                      # 공유 상태 관리
│   ├── __init__.py
│   └── emotion_state.py        # 스레드 간 공유 State + Lock
│
├── requirements.txt
└── .env.example
```

---

## 주요 기능

### 1. 표정 감지 (face_detector.py)
- DeepFace + MTCNN 백엔드로 웹캠 실시간 표정 분석
- 감지 감정: 행복 / 슬픔 / 분노 / 공포 / 혐오 / 놀람 / 무감정
- 10프레임마다 1회 분석으로 성능 최적화
- 감정 확률값 전체(`emotion_scores`)를 VAD 계산에 활용

---

### 2. 음성 인식 (stt_detector.py)
- Whisper medium 모델로 실시간 STT
- 한국어 특화 (`language="ko"`)
- 처리 후 임시 오디오 파일 즉시 삭제 (Privacy-by-Design)
- 30초 STT 버퍼 유지 → LLM 맥락 분석에 활용
- `.env`의 `STT_MODE`로 두 가지 방식 전환 가능

#### STT 모드 비교

| 모드 | 방식 | 장점 | 단점 |
|------|------|------|------|
| `fixed` | 고정 길이 녹음 | 구현 단순, 안정적 | 말하다 끊길 수 있음 |
| `vad` | 말소리 감지 후 녹음 | 자연스러운 끊김 없음 | 민감도 튜닝 필요 |

**fixed 모드** — `STT_DURATION`(초) 단위로 끊어서 녹음 후 STT 처리

**vad 모드** — `webrtcvad`로 말소리 감지 시 녹음 시작, `VAD_SILENCE_SEC`(초) 이상 무음이 지속되면 발화 종료로 판단 후 STT 처리

```
[fixed 모드]
0s ──────10s ──────20s ──────30s
  [녹음1]    [녹음2]    [녹음3]

[vad 모드]
0s    2s──────────7s    11s───13s
      [말소리감지→녹음]   [말소리감지→녹음]
```

---

### 3. VAD 스트레스 계산 (stress_calculator.py)
심리학 기반 VAD(Valence-Arousal-Dominance) 모델로 스트레스를 계산합니다.

> 여기서 VAD는 Voice Activity Detection이 아닌 심리학 용어입니다.

```
스트레스 = (1 - Valence) × 0.5 + Arousal × 0.3 + (1 - Dominance) × 0.2
```

| 감정 | Valence | Arousal | Dominance |
|------|---------|---------|-----------|
| happy | 0.9 | 0.7 | 0.8 |
| sad | 0.1 | 0.2 | 0.2 |
| angry | 0.1 | 0.9 | 0.7 |
| fear | 0.1 | 0.8 | 0.1 |
| disgust | 0.1 | 0.6 | 0.6 |
| surprise | 0.6 | 0.8 | 0.4 |
| neutral | 0.5 | 0.2 | 0.5 |

VAD 점수가 임계값(기본 6.0) 이상일 때만 AI 서버로 LLM 요청을 보내 불필요한 API 호출을 최소화합니다.

---

### 4. 쿨다운 관리 (cooldown.py)
개입 후 일정 시간 동안 재개입을 방지합니다.

| 스트레스 수준 | 쿨다운 |
|-------------|--------|
| 9 ~ 10 (긴급) | 10분 (고정) |
| 7 ~ 8 (높음) | 30분 (`COOLDOWN_MINUTES`로 조절 가능) |

---

### 5. AI 서버 통신 (llm_client.py)
- VAD 점수 + STT 텍스트 + 표정을 `capstone-ai` 서버로 POST
- 타임아웃(10초) 및 서버 오류 시 VAD 값으로 자동 fallback
- 서버 URL은 `.env`로 관리 → 로컬/배포 환경 전환 용이

---

## 데이터 흐름

```
[메인 스레드]          [Thread 1]              [Thread 2]
face_detector.py      stt_detector.py         run_stress_calculator()
      ↓                     ↓                        ↓
 emotion_scores         stt_text              VAD 계산 (DETECTION_INTERVAL마다)
      ↓                     ↓                        ↓
      └──────── emotion_state ────────────→   VAD >= 6?
                                                     ↓ Yes
                                               쿨다운 체크
                                                     ↓ 통과
                                             llm_client.py
                                             HTTP POST → capstone-ai
                                                     ↓
                                             stress_level + recommendation
```

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `AI_SERVER_URL` | `http://localhost:8000` | capstone-ai 서버 주소 |
| `DETECTION_INTERVAL` | `5` | 스트레스 계산 주기 (초) |
| `STT_MODE` | `fixed` | STT 방식 선택 (`fixed` / `vad`) |
| `STT_DURATION` | `10` | fixed 모드 녹음 시간 (초) |
| `VAD_SILENCE_SEC` | `1.5` | vad 모드 무음 감지 시간 (초) |
| `COOLDOWN_MINUTES` | `30` | 높음 수준 쿨다운 (분) / 긴급은 10분 고정 |

---

## 설치 및 실행

### 요구사항
- Python 3.10+
- conda 환경 권장
- 웹캠 및 마이크 연결 필요

### 1. 환경 설정

```bash
conda create -n capstone python=3.10
conda activate capstone
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일 수정:

```dotenv
AI_SERVER_URL=http://localhost:8000
DETECTION_INTERVAL=5
STT_MODE=fixed          # fixed: 고정 길이 / vad: 말소리 감지
STT_DURATION=10         # fixed 모드 녹음 시간 (초)
VAD_SILENCE_SEC=1.5     # vad 모드 무음 감지 시간 (초)
COOLDOWN_MINUTES=30
```

### 3. capstone-ai 서버 먼저 실행

```bash
# capstone-ai 레포에서
uvicorn server.main:app --reload --port 8000
```

### 4. 실행

```bash
python main.py
```

### 5. 종료
카메라 창에서 `q` 키를 누르면 모든 스레드가 종료됩니다.

---

## 카메라 인덱스 설정

카메라가 여러 개 연결된 경우 아래 스크립트로 사용 가능한 인덱스를 확인하세요:

```python
import cv2

for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"카메라 인덱스 {i} 사용 가능!")
        cap.release()
```

확인 후 `detection/face_detector.py`에서 인덱스를 변경하세요:

```python
cap = cv2.VideoCapture(1)  # 본인 환경에 맞게 수정
```

---

## VAD 모드 민감도 튜닝

`webrtcvad`의 민감도는 `stt_detector.py`에서 조절할 수 있습니다:

```python
vad = webrtcvad.Vad(2)  # 0~3 (높을수록 민감)
```

| 값 | 설명 |
|----|------|
| 0 | 가장 덜 민감 (조용한 환경 권장) |
| 1 | 낮은 민감도 |
| 2 | 기본값 (권장) |
| 3 | 가장 민감 (배경 소음 있는 환경) |

`VAD_SILENCE_SEC`도 환경에 따라 조절하세요:
- 빠른 대화: `1.0`
- 일반 대화: `1.5` (기본값)
- 느린 대화 / 긴 문장: `2.0`

---

## Privacy-by-Design

본 프로젝트는 사용자 프라이버시 보호를 핵심 원칙으로 합니다.

- 원본 오디오/비디오는 외부 서버로 전송하지 않습니다
- STT 처리 후 임시 오디오 파일을 즉시 삭제합니다
- AI 서버로는 텍스트 데이터(감정 분류 결과, STT 텍스트)만 전송합니다
- 개입 쿨다운으로 과도한 모니터링을 방지합니다

---

## 기술 스택

| 항목 | 기술 |
|------|------|
| 언어 | Python 3.10 |
| 표정 인식 | DeepFace + MTCNN |
| 음성 인식 | OpenAI Whisper medium |
| 말소리 감지 | webrtcvad |
| 스트레스 계산 | VAD 모델 (심리학 기반) |
| 멀티스레딩 | threading |
| 서버 통신 | requests (HTTP) |
| 오디오 처리 | sounddevice, soundfile |

---

## 관련 레포지토리

| 레포 | 설명 |
|------|------|
| [emotion-ai](../emotion-ai) | FastAPI + LangGraph 감정 에이전트 |
| [emotion-backend](../emotion-backend) | Spring Boot 백엔드 서버 |