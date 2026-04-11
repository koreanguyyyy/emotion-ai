# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import run_agent

app = FastAPI(title="capstone-ai", description="감정 에이전트 AI 서버")

class AnalyzeRequest(BaseModel):
    vad_stress: float
    stt_text: str
    face_emotion: str

class AnalyzeResponse(BaseModel):
    stress_level: int
    recommendation: str | None

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    result = run_agent(
        vad_stress=req.vad_stress,
        stt_text=req.stt_text,
        face_emotion=req.face_emotion,
    )
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}
