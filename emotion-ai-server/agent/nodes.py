# agent/nodes.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm.model import llm
from agent.state import AgentState

# ── 스트레스 보정 프롬프트 ─────────────────────────────
stress_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 심리 분석 전문가입니다.
표정 기반 VAD 스트레스 점수와 최근 30초간 사용자 발화를 보고
최종 스트레스 수준을 0~10 사이 정수로 보정하세요.
반드시 숫자 하나만 반환하세요. 예) 7"""),
    ("human", """
VAD 기반 스트레스 점수: {vad_stress}/10
최근 30초 발화: "{stt_text}"

최종 스트레스 수준 (0~10):""")
])

# ── 추천 프롬프트 ──────────────────────────────────────
recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 감정 케어 전문가입니다. 한국어로 답변하세요."),
    ("human", "사용자의 현재 감정은 '{emotion}'이고 스트레스 수준은 {stress_level}/10입니다. 위로와 함께 구체적인 활동을 추천해주세요.")
])

stress_chain = stress_prompt | llm | StrOutputParser()
recommend_chain = recommend_prompt | llm | StrOutputParser()

# ── 노드 1: 스트레스 보정 ──────────────────────────────
def correct_stress(state: AgentState) -> AgentState:
    if not state["stt_text"]:
        state["stress_level"] = round(state["vad_stress"])
        print(f"[Node] 텍스트 없음 → VAD 값 사용: {state['stress_level']}/10")
        return state

    try:
        result = stress_chain.invoke({
            "vad_stress": state["vad_stress"],
            "stt_text": state["stt_text"]
        })
        stress = int("".join(filter(str.isdigit, result.strip()))[:1])
        state["stress_level"] = max(0, min(10, stress))
        print(f"[Node] LLM 보정: {state['vad_stress']} → {state['stress_level']}/10")
    except Exception as e:
        print(f"[Node] LLM 보정 실패: {e} → VAD 값 사용")
        state["stress_level"] = round(state["vad_stress"])

    return state

# ── 노드 2: 개입 판단 ──────────────────────────────────
def decide_intervention(state: AgentState) -> AgentState:
    if state["stress_level"] >= 7:
        state["recommendation"] = "개입 필요"
    else:
        state["recommendation"] = "개입 불필요"
    print(f"[Node] 판단: {state['recommendation']} (스트레스: {state['stress_level']}/10)")
    return state

# ── 노드 3: 추천 생성 ──────────────────────────────────
def recommend(state: AgentState) -> AgentState:
    print("[Node] 추천 생성 중...")
    response = recommend_chain.invoke({
        "emotion": state["face_emotion"],
        "stress_level": state["stress_level"]
    })
    state["recommendation"] = response
    print(f"[Node] 추천 완료")
    return state

# ── 조건부 엣지 ────────────────────────────────────────
def should_recommend(state: AgentState) -> str:
    if state["recommendation"] == "개입 불필요":
        return "skip"
    return "recommend"
