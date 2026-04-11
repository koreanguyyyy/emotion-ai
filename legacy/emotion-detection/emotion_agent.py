# ai/emotion_agent.py
import time
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 쿨다운 설정 ────────────────────────────────────────
COOLDOWN_MAP = {
    "urgent": 10 * 60,   # 스트레스 9~10 → 10분
    "high":   30 * 60,   # 스트레스 7~8  → 30분
}

last_intervention_time = 0.0

def is_cooldown_active(stress_level: int) -> bool:
    cooldown = COOLDOWN_MAP["urgent"] if stress_level >= 9 else COOLDOWN_MAP["high"]
    return time.time() - last_intervention_time < cooldown

def update_intervention_time():
    global last_intervention_time
    last_intervention_time = time.time()

def get_remaining_cooldown(stress_level: int) -> int:
    cooldown = COOLDOWN_MAP["urgent"] if stress_level >= 9 else COOLDOWN_MAP["high"]
    remaining = cooldown - (time.time() - last_intervention_time)
    return max(0, int(remaining / 60))

# ── LLM 설정 ───────────────────────────────────────────
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 감정 케어 전문가입니다. 한국어로 답변하세요."),
    ("human", "사용자의 현재 감정은 '{emotion}'이고 스트레스 수준은 {stress_level}/10입니다. 위로와 함께 구체적인 활동을 추천해주세요.")
])

chain = prompt | llm | StrOutputParser()

# ── LangGraph State ────────────────────────────────────
class EmotionState(TypedDict):
    emotion: str
    stress_level: int
    recommendation: str

# ── 노드 함수 ──────────────────────────────────────────
def analyze_emotion(state: EmotionState) -> EmotionState:
    print(f"\n[분석 노드] 감정: {state['emotion']}, 스트레스: {state['stress_level']}/10")
    return state

def decide_intervention(state: EmotionState) -> EmotionState:
    if state["stress_level"] >= 7:
        state["recommendation"] = "개입 필요"
    else:
        state["recommendation"] = "개입 불필요"
    print(f"[판단 노드] {state['recommendation']}")
    return state

def recommend(state: EmotionState) -> EmotionState:
    print("[추천 노드] LLM 응답 생성 중...\n")
    response = chain.invoke({
        "emotion": state["emotion"],
        "stress_level": state["stress_level"]
    })
    state["recommendation"] = response
    return state

def should_recommend(state: EmotionState) -> str:
    if state["recommendation"] == "개입 불필요":
        return "skip"
    return "recommend"

# ── 그래프 구성 ────────────────────────────────────────
graph = StateGraph(EmotionState)
graph.add_node("analyze", analyze_emotion)
graph.add_node("decide", decide_intervention)
graph.add_node("recommend", recommend)
graph.set_entry_point("analyze")
graph.add_edge("analyze", "decide")
graph.add_conditional_edges(
    "decide",
    should_recommend,
    {"recommend": "recommend", "skip": END}
)
graph.add_edge("recommend", END)

app = graph.compile()

# ── 외부 호출 진입점 ───────────────────────────────────
def run_agent(emotion: str, stress_level: int) -> str | None:
    """
    multimodal_detector → main.py에서 호출
    쿨다운 중이면 None 반환
    """
    if is_cooldown_active(stress_level):
        remaining = get_remaining_cooldown(stress_level)
        print(f"[Agent] 쿨다운 중 → {remaining}분 남음, 스킵")
        return None

    result = app.invoke({
        "emotion": emotion,
        "stress_level": stress_level,
        "recommendation": ""
    })

    update_intervention_time()
    return result["recommendation"]
