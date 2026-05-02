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

# 전역 앱 인스턴스
app = build_graph()

def run_agent(
    vad_stress: float,
    stt_text: str,
    face_emotion: str,
    user_id: int,
    emotion_log_id: int,
    cooldown_minutes: int = 30,
) -> dict:
    result = app.invoke({
        "vad_stress": vad_stress,
        "stt_text": stt_text,
        "face_emotion": face_emotion,
        "stress_level": 0,
        "recommendation": ""
    })

    recommendation = result["recommendation"]
    stress_level = result["stress_level"]

    # 개입 불필요면 DB 저장 안 함
    if recommendation == "개입 불필요" or not recommendation:
        return {
            "stress_level": stress_level,
            "recommendation": None
        }

    # 개입 저장
    intervention_id = save_intervention(
        user_id=user_id,
        emotion_log_id=emotion_log_id,
        agent_message=recommendation,
        cooldown_minutes=cooldown_minutes,
    )

    # 추천 항목 저장
    if intervention_id:
        save_recommendations(
            intervention_id=intervention_id,
            recommendation_text=recommendation,
        )

    return {
        "stress_level": stress_level,
        "recommendation": recommendation
    }