# emotion-ai-server

FastAPI + LangGraph 기반 ReAct 추천 에이전트 서버

## 모듈 구성
| 모듈 | 설명 |
|------|------|
| `agent/` | LangGraph ReAct 에이전트 |
| `tools/` | 추천 Tool (Spotify, TMDB, Kakao, Aladin, Chroma) |
| `memory/` | 대화 메모리 |
| `rag/` | Chroma 벡터 DB (도서 RAG) |
| `tts/` | Clova Voice / pyttsx3 TTS |

## 실행 방법
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
