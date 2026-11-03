# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import correct_stress, decide_intervention, recommend, should_recommend
from backend_client import save_intervention, save_recommendations

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("correct_stress", correct_stress)
    graph.add_node("decide", decide_intervention)
    graph.add_node("recommend", recommend)
    graph.set_entry_point("correct_stress")
    graph.add_edge("correct_stress", "decide")
    graph.add_conditional_edges(
        "decide",
        should_recommend,
        {"recommend": "recommend", "skip": END}
    )
    graph.add_edge("recommend", END)
    return graph.compile()

app = build_graph()

def run_agent(
    vad_stress: float,
    stt_text: str,
    face_emotion: str,
    user_id: int,
    emotion_log_id: int,
    cooldown_minutes: int = 30,
    force: bool = False,
    memory_key: str | None = None,
    speech_history: list | None = None,
) -> dict:
    result = app.invoke({
        "force": force,
        "memory_key": memory_key,
        "speech_history": speech_history or [],
        "vad_stress": vad_stress,
        "stt_text": stt_text,
        "face_emotion": face_emotion,
        "stress_level": 0,
        "recommendation": "",
        "user_id": user_id,
        "emotion_log_id": emotion_log_id,
    })

    recommendation = result["recommendation"]
    stress_level = result["stress_level"]

    if recommendation == "개입 불필요" or not recommendation:
        return {
            "stress_level": stress_level,
            "recommendation": None,
            "intervention_id": None,
            "recommendation_id": None,
        }

    # 개입 저장
    intervention_id = save_intervention(
        user_id=user_id,
        emotion_log_id=emotion_log_id,
        agent_message=recommendation,
        cooldown_minutes=cooldown_minutes,
    )

    # 추천 항목 저장 → 첫 번째 recommendation_id 반환
    recommendation_id = None
    if intervention_id:
        recommendation_id = save_recommendations(
            intervention_id=intervention_id,
            recommendation_text=recommendation,
        )

    return {
        "stress_level": stress_level,
        "recommendation": recommendation,
        "intervention_id": intervention_id,
        "recommendation_id": recommendation_id,
    }