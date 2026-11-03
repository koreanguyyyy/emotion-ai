# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import run_agent
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
from fastapi import HTTPException

app = FastAPI(title="capstone-ai", description="감정 에이전트 AI 서버")

class AnalyzeRequest(BaseModel):
    vad_stress: float # 0.0 ~ 10.0 스트레스 지수
    stt_text: str # 발화 내용
    face_emotion: str  # 얼굴 감정
    user_id: int # 사용자 ID
    emotion_log_id: int # 감정 로그 ID
    cooldown_minutes: int = 30  # 개입 쿨다운 시간 (분 단위, 기본값 30분))
    force: bool = False         # 시연용 '지금 말 걸기': 개입 기준(INTERVENE_THRESHOLD)을 건너뜀
    memory_key: str | None = None        # 지난 대화 기억 공간 (agent/memory.py)
    speech_history: list[dict] | None = None   # 최근 몇 분 발화 [{time, text}]

class AnalyzeResponse(BaseModel):
    stress_level: int   # LLM 판단 스트레스 수준
    recommendation: str | None  # LLM 추천 메시지
    intervention_id: int | None = None  # 개입 ID
    recommendation_id: int | None = None  # 추천 ID

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    result = run_agent(
        vad_stress=req.vad_stress,
        stt_text=req.stt_text,
        face_emotion=req.face_emotion,
        user_id=req.user_id,
        emotion_log_id=req.emotion_log_id,
        cooldown_minutes=req.cooldown_minutes,
        force=req.force,
        memory_key=req.memory_key,
        speech_history=req.speech_history,
    )
    return result

# ── 이어지는 대화 ─────────────────────────────────────
class ChatContext(BaseModel):
    face_emotion: str | None = None
    stress_index: float | None = None
    heart_rate: float | None = None
    resting_heart_rate: float | None = None

class ChatRequest(BaseModel):
    session_id: str               # 감지 프로그램이 정함 (개입 ID가 있으면 "iv-<id>")
    user_id: int
    message: str                  # 사용자 발화 (음성 인식 결과 또는 웹 채팅 입력)
    intervention_id: int | None = None   # 새 추천을 이 개입의 추천으로 저장
    opening: str | None = None    # 세션이 없을 때 첫 메시지로 넣을 에이전트의 말 (서버 재시작 대비)
    context: ChatContext | None = None   # 지금 감지된 표정·스트레스·심박
    memory_key: str | None = None        # 지난 대화 기억 공간 (agent/memory.py)
    speech_history: list[dict] | None = None   # 대화 전 최근 몇 분 발화 [{time, text}]

class ChatRecommendation(BaseModel):
    category: str
    title: str
    artist: str | None = None
    url: str | None = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    ended: bool
    recommendations: list[ChatRecommendation] = []

_chat_manager = None

def get_chat_manager():
    global _chat_manager
    if _chat_manager is None:
        from agent.chat import ChatManager
        from agent.nodes import RECOMMEND_ENABLED, llm_with_tools
        from llm.model import llm
        from tools import ALL_TOOLS
        from backend_client import _save_single_recommendation
        from agent.memory import get_memory
        _chat_manager = ChatManager(llm, llm_with_tools, ALL_TOOLS, save_rec=_save_single_recommendation,
                                    use_tools=RECOMMEND_ENABLED, memory=get_memory())
    return _chat_manager

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):   # LLM 호출이 오래 걸리므로 async가 아닌 일반 함수(스레드풀)로 둔다
    return get_chat_manager().chat(
        session_id=req.session_id,
        user_id=req.user_id,
        message=req.message,
        intervention_id=req.intervention_id,
        opening=req.opening,
        context=req.context.model_dump() if req.context else None,
        memory_key=req.memory_key,
        speech_history=req.speech_history,
    )

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """답을 조각으로 흘려보낸다 (한 줄에 JSON 하나, NDJSON).
    {"type":"delta","text":"..."} 여러 줄 → 마지막 {"type":"done","reply":...,"ended":...,"recommendations":[...]}"""
    gen = get_chat_manager().chat_stream(
        session_id=req.session_id,
        user_id=req.user_id,
        message=req.message,
        intervention_id=req.intervention_id,
        opening=req.opening,
        context=req.context.model_dump() if req.context else None,
        memory_key=req.memory_key,
        speech_history=req.speech_history,
    )
    return StreamingResponse((json.dumps(ev, ensure_ascii=False) + "\n" for ev in gen),
                             media_type="application/x-ndjson")

class SpeechMemoryRequest(BaseModel):
    memory_key: str
    items: list[dict]             # [{time:'HH:MM:SS', text}] 대화 밖에서 한 말

@app.post("/memory/speech")
def memory_speech(req: SpeechMemoryRequest):
    """평소 켜 두고 쓸 때: 감지 프로그램이 30분마다 보내는 대화 밖 발화를 요약해 기억에 쌓는다(백그라운드)."""
    import threading
    from agent.memory import get_memory, summarize_speech
    from llm.model import llm
    mem = get_memory()
    if mem is None or len(req.items) < 3:
        return {"accepted": False}
    sid = f"speech-{req.items[0].get('time', '')}-{len(req.items)}"
    def run():
        try:
            summary = summarize_speech(llm, req.items)
            if summary:
                mem.add(req.memory_key, summary, sid)
                print(f"[Memory] 혼잣말 요약 저장 ({req.memory_key}): {summary}")
        except Exception as e:
            print(f"[Memory] 혼잣말 요약 실패: {e}")
    threading.Thread(target=run, daemon=True).start()
    return {"accepted": True}

class ChatEndRequest(BaseModel):
    session_id: str

@app.post("/chat/end")
def chat_end(req: ChatEndRequest):
    """감지 프로그램이 대화가 끝났다고 알린다(웹 '끝내기'·무응답·새 관람객) → 요약해서 기억에 남긴다."""
    return {"summarizing": get_chat_manager().end_session(req.session_id)}

@app.get("/chat/{session_id}")
def chat_history(session_id: str):
    h = get_chat_manager().history(session_id)
    if h is None:
        raise HTTPException(status_code=404, detail="대화 세션이 없습니다")
    return h

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"stress_level" : 0, "recommendation": None}
    )