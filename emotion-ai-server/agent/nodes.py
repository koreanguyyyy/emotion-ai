# agent/nodes.py

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from llm.model import llm
from agent.state import AgentState
from tools import ALL_TOOLS

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/agent.log", encoding="utf-8"), # 로그 파일로 저장
        logging.StreamHandler() # 콘솔에도 출력
    ]
)

logger = logging.getLogger(__name__)


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
        logger.info(f"[Node] 텍스트 없음 → VAD 값 사용: {state['stress_level']}/10")
        return state

    try:
        result = stress_chain.invoke({
            "vad_stress": state["vad_stress"],
            "stt_text": state["stt_text"],
            "face_emotion": state["face_emotion"]
        })
        stress = int("".join(filter(str.isdigit, result.strip()))[:1])
        state["stress_level"] = max(0, min(10, stress))
        logger.info(f"[Node] LLM 보정: {state['vad_stress']} → {state['stress_level']}/10")
    except Exception as e:
        logger.error(f"[Node] LLM 보정 실패: {e} → VAD 값 사용")
        state["stress_level"] = round(state["vad_stress"])

    return state

# ── 노드 2: 개입 판단 ──────────────────────────────────
def decide_intervention(state: AgentState) -> AgentState:
    if state["stress_level"] >= 7:
        state["recommendation"] = "개입 필요"
    else:
        state["recommendation"] = "개입 불필요"
    logger.info(f"[Node] 판단: {state['recommendation']} (스트레스: {state['stress_level']}/10)")
    return state

# ── 노드 3: Tool Calling 추천 ──────────────────────────
# agent/nodes.py - recommend 노드만 수정
def recommend(state: AgentState) -> AgentState:
    logger.info("[Node] Tool Calling 추천 시작...")

    prompt = f""" 
    사용자의 현재 상태:
    - 감정: {state['face_emotion']}
    - 스트레스 수준: {state['stress_level']}/10
    - 최근 발화: "{state['stt_text']}"

    [Tool 선택 규칙]
    1. 발화에서 사용자가 **구체적으로 요청한 대상**을 최우선으로 판단하세요:
    - "노래", "음악", "들을", "듣고 싶다" → recommend_music 호출
    - "책", "읽을", "읽고 싶다" → recommend_book 호출
    - "장소", "어디", "카페", "공원", "산책" → recommend_place 호출

    ⚠️ "풀다", "해소", "전환" 같은 동사는 Tool 선택 기준이 아닙니다.
    "기분을 풀 수 있는 노래" → "노래"가 핵심이므로 recommend_music입니다.

    2. 발화에 위 키워드가 전혀 없을 때만 감정 기반으로 선택하세요:
    - 슬픔/공포/혐오 → recommend_music
    - 분노 → recommend_place
    - 무감정 → recommend_book
    - 행복/놀람 → 자유롭게 선택

    반드시 Tool을 호출하세요. 분석 과정을 텍스트로 설명하지 말고, Tool function을 직접 호출하세요.
    """
    try:
        response = llm_with_tools.invoke([HumanMessage(content=prompt)])


        tool_results = []
        if response.tool_calls:
            tool_map = {t.name: t for t in ALL_TOOLS}
            logger.info(f"[Node] Tool 맵: {tool_map}")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]



                logger.info(f"[Node] Tool 호출: {tool_name}({tool_args})")
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    tool_results.append(result)

            state["recommendation"] = "\n\n".join(tool_results)
            logger.info(f"[Node] Tool 결과:\n{state['recommendation']}")
        else:
            logger.info("[Node] Tool 호출 없음 → LLM이 텍스트로만 응답")
            state["recommendation"] = "추천을 생성하지 못했습니다. 다시 시도해주세요."

        logger.info("[Node] 추천 완료")

    except Exception as e:
        logger.error(f"[Node] Tool Calling 실패: {e}")
        state["recommendation"] = "추천을 생성하지 못했습니다."

    return state
# ── 조건부 엣지 ────────────────────────────────────────
def should_recommend(state: AgentState) -> str:
    if state["recommendation"] == "개입 불필요":
        return "skip"
    return "recommend"