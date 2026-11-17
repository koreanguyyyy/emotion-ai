# emotion-ai-server

FastAPI + LangGraph 기반 ReAct 추천 에이전트 서버

## 모듈 구성
| 모듈 | 설명 |
|------|------|
| `agent/` | LangGraph ReAct 에이전트 |
| `tools/` | 추천 Tool (Spotify, TMDB, Kakao, Aladin, Chroma) |
| `memory/` | 대화 메모리 |
| `tts/` | Clova Voice / pyttsx3 TTS |

## 실행 방법
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```


## 이어지는 대화 `/chat` (2026-11-03)
개입(`/analyze`) 뒤 사용자의 대답을 받아 대화를 이어 간다. 코드: `agent/chat.py`
```
POST /chat  {"session_id": "iv-42", "user_id": 1, "message": "다른 노래 없어?",
             "intervention_id": 42, "opening": "<첫 메시지>",
             "context": {"face_emotion": "슬픔", "stress_index": 7.2, "heart_rate": 95, "resting_heart_rate": 72}}
→ {"session_id": "iv-42", "reply": "...", "ended": false,
   "recommendations": [{"category": "MUSIC", "title": "...", "artist": "...", "url": "..."}]}
GET  /chat/{session_id}   대화 기록
```
- 세션별로 최근 20개 대화를 기억한다(메모리, 2시간 뒤 삭제). 지금 감지된 표정·스트레스·심박을 매 턴 알려 준다.
- 추천이 필요하면 기존 추천 도구(음악·책·장소)를 다시 부르고, 새 추천을 `intervention_id`의 추천으로 저장한다. 이미 추천한 항목은 피한다.
- 답은 음성으로 읽히므로 1~3문장, URL 없음. 링크는 `recommendations`로 따로 준다.
- 사용자가 마무리하면 `ended: true`. 자해를 비치는 말에는 109(자살예방상담전화) 안내를 하고 대화를 끝내지 않는다.
- 대화 기록: `logs/chat_sessions.jsonl`
- 시험: `python tests/test_chat.py` (가짜 LLM, 키 불필요)
