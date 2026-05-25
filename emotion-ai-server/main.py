# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import run_agent
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI(title="capstone-ai", description="감정 에이전트 AI 서버")

class AnalyzeRequest(BaseModel):
    vad_stress: float # 0.0 ~ 10.0 스트레스 지수
    stt_text: str # 발화 내용
    face_emotion: str  # 얼굴 감정
    user_id: int # 사용자 ID
    emotion_log_id: int # 감정 로그 ID
    cooldown_minutes: int = 30  # 개입 쿨다운 시간 (분 단위, 기본값 30분))

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
    )
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"stress_level" : 0, "recommendation": None}
    )