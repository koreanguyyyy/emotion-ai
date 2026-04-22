# agent/nodes.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from llm.model import llm
from agent.state import AgentState
from tools import ALL_TOOLS

# ── LLM에 Tool 바인딩 ──────────────────────────────────
llm_with_tools = llm.bind_tools(ALL_TOOLS)

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

stress_chain = stress_prompt | llm | StrOutputParser()

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

# ── 노드 3: Tool Calling 추천 ──────────────────────────
def recommend(state: AgentState) -> AgentState:
    print("[Node] Tool Calling 추천 시작...")

    prompt = f"""
사용자의 현재 상태:
- 감정: {state['face_emotion']}
- 스트레스 수준: {state['stress_level']}/10
- 최근 발화: "{state['stt_text']}"

위 상태를 분석해서 사용자에게 가장 도움이 될 것 같은 추천을 해주세요.
반드시 아래 도구 중 하나 또는 여러 개를 호출해서 구체적인 추천 결과를 제공하세요:
- recommend_music: 감정에 맞는 음악 추천
- recommend_book: 감정에 맞는 책 추천
- recommend_place: 주변 장소 추천
"""

    try:
        response = llm_with_tools.invoke([HumanMessage(content=prompt)])

        tool_results = []
        if response.tool_calls:
            tool_map = {t.name: t for t in ALL_TOOLS}
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"[Node] Tool 호출: {tool_name}({tool_args})")

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    tool_results.append(result)

            state["recommendation"] = "\n\n".join(tool_results)
        else:
            state["recommendation"] = str(response.content)

        print("[Node] 추천 완료")

    except Exception as e:
        print(f"[Node] Tool Calling 실패: {e}")
        state["recommendation"] = "추천을 생성하지 못했습니다."

    return state

# ── 조건부 엣지 ────────────────────────────────────────
def should_recommend(state: AgentState) -> str:
    if state["recommendation"] == "개입 불필요":
        return "skip"
    return "recommend"