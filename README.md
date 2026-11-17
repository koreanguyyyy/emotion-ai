# emotion-ai — 멀티모달 감정 인식 기반 선제적 개입 AI 에이전트

표정(MediaPipe 블렌드셰입)과 웹캠 심박(rPPG)으로 스트레스 지수를 계산하고,
지수가 높아지면 AI 에이전트가 먼저 말을 거는 시스템입니다.
기존에 모듈별로 나뉘어 있던 레포를 하나로 합쳤으며, 각 모듈의 커밋 히스토리는 보존했습니다.

## 구조

| 폴더 | 역할 | 언어 |
|---|---|---|
| `emotion-detection-fusion/` | 감지: 표정 + 웹캠 심박 → 스트레스 지수, 개입·대화 흐름 | Python |
| `emotion-ai-server/` | 에이전트 서버: FastAPI + LangGraph, 상담형 대화(`/chat`), 추천 | Python |
| `emotion-backend/` | 백엔드: 감정 로그·개입·추천·피드백 저장 | Java (Spring Boot, Gradle, MySQL) |
| `capstone-frontend/` | 웹 대시보드: 실시간 모니터, 대화 화면 | React (Vite, TypeScript) |
| `legacy/emotion-detection/` | 1학기 감지 코드 (참고용, fusion이 대체) | Python |

## 실행 순서

1. **백엔드**: `emotion-backend/` — DB 비밀번호는 환경변수로 넣습니다. `export DB_PASSWORD=...` 후 `./gradlew bootRun`
2. **에이전트 서버**: `emotion-ai-server/` — `.env.example`을 `.env`로 복사해 키를 채우고 실행
3. **감지**: `emotion-detection-fusion/` — `.env.example`을 `.env`로 복사한 뒤 `python main.py` (대화에는 `STT_MODE=vad` 권장)
4. **웹**: `capstone-frontend/` — `.env.example`을 `.env`로 복사한 뒤 `npm install && npm run dev`

각 폴더의 README와 `.env.example`에 자세한 설정이 있습니다.

## 주의

- `.env`, 로그(`logs/`), 음성 파일(`*.wav`)은 커밋하지 않습니다 (`.gitignore`로 제외).
- `emotion-detection-fusion/tools/demo_data.py`가 만드는 데이터는 화면 시연용 가짜 데이터이며 실제 측정이 아닙니다.
