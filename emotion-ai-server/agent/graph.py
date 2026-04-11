# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import correct_stress, decide_intervention, recommend, should_recommend

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

def run_agent(vad_stress: float, stt_text: str, face_emotion: str) -> dict:
    result = app.invoke({
        "vad_stress": vad_stress,
        "stt_text": stt_text,
        "face_emotion": face_emotion,
        "stress_level": 0,
        "recommendation": ""
    })
    return {
        "stress_level": result["stress_level"],
        "recommendation": result["recommendation"]
        if result["recommendation"] != "개입 불필요" else None
    }
