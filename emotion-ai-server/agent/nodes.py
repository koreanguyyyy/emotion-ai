# agent/nodes.py
import os
import re
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from llm.model import llm
from agent.state import AgentState
from tools import ALL_TOOLS

# logs 폴더 없으면 자동 생성
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/agent.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 개입 기준 (LLM이 보정한 스트레스 0~10). 1학기 기본 7. 시연에서는 .env INTERVENE_THRESHOLD로 낮출 수 있다
INTERVENE_THRESHOLD = int(os.getenv("INTERVENE_THRESHOLD", "7").split("#")[0].strip() or 7)

# 추천(음악·책·장소 도구) 사용 여부. false면 추천 없이 상담형으로 먼저 말을 건다 (2026-11-04 기본값 false)
RECOMMEND_ENABLED = os.getenv("RECOMMEND_ENABLED", "false").split("#")[0].strip().lower() == "true"

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
표정 기반 감정: {face_emotion}
최근 30초 발화: "{stt_text}"

최종 스트레스 수준 (0~10):""")
])

stress_chain = stress_prompt | llm | StrOutputParser()

# ── 헬퍼 함수 ──────────────────────────────────────────
def _format_recent_history(interventions: list) -> str:
    if not interventions:
        return "없음"
    lines = []
    for i, inv in enumerate(interventions[:3], 1):
        created = inv.get("createdAt", "")[:10]
        recs = inv.get("recommendations", [])
        rec_summary = ", ".join(
            f"{r.get('category', '')}:{r.get('title', '')}"
            for r in recs[:2]
        )
        lines.append(f"{i}. [{created}] 추천: {rec_summary or '없음'}")
    return "\n".join(lines)

def _format_recent_recommendations(recommendations: list) -> str:
    if not recommendations:
        return "없음"
    lines = []
    for rec in recommendations[:5]:
        category = rec.get("category", "")
        title = rec.get("title", "")
        lines.append(f"- [{category}] {title}")
    return "\n".join(lines)

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
        # 첫 번째 정수 전체를 읽음 (예전 코드는 첫 글자만 읽어 "10"이 1이 됐음)
        match = re.search(r"\d+", result)
        if match is None:
            raise ValueError(f"숫자 없음: {result!r}")
        stress = int(match.group())
        state["stress_level"] = max(0, min(10, stress))
        logger.info(f"[Node] LLM 보정: {state['vad_stress']} → {state['stress_level']}/10")
    except Exception as e:
        logger.error(f"[Node] LLM 보정 실패: {e} → VAD 값 사용")
        state["stress_level"] = round(state["vad_stress"])

    return state

# ── 노드 2: 개입 판단 ──────────────────────────────────
def decide_intervention(state: AgentState) -> AgentState:
    if state.get("force"):
        state["recommendation"] = "개입 필요"
        logger.info(f"[Node] 시연 요청(force) → 기준과 상관없이 개입 (스트레스: {state['stress_level']}/10)")
        return state
    if state["stress_level"] >= INTERVENE_THRESHOLD:
        state["recommendation"] = "개입 필요"
    else:
        state["recommendation"] = "개입 불필요"
    logger.info(f"[Node] 판단: {state['recommendation']} (스트레스: {state['stress_level']}/10)")
    return state

# ── 노드 3-a: 상담형 첫 마디 (추천 없음) ───────────────
COUNSEL_OPENING_PROMPT = """당신은 사용자의 마음을 살피는 AI 상담 친구입니다. 반드시 한국어로, 따뜻한 존댓말로 말하세요.
지금 사용자의 스트레스가 높아 보여서 당신이 먼저 말을 거는 상황입니다. 당신의 말은 음성으로 읽힙니다.

[감지된 상태 — 숫자를 그대로 말하지 마세요]
- 표정: {face_emotion}
- 스트레스 수준: {stress_level}/10
- 최근 30초 발화: "{stt_text}"
{extra}
다음을 지켜 2~3문장으로 말하세요.
- 먼저 지금 상태를 부드럽게 알아차려 주세요(예: "조금 지쳐 보이세요", "많이 긴장되신 것 같아요"). 단정하거나 진단하지 마세요.
- 최근 발화가 있으면 그 내용을 자연스럽게 짚어 주세요.
- 마지막은 편하게 답할 수 있는 열린 질문 하나로 끝내세요(예: "무슨 일이 있었는지 이야기해 주실래요?").
- 음악·책·장소 추천, 목록, 이모지, URL은 쓰지 마세요.
- 소리 내어 말할 문장만 쓰세요. 따옴표로 감싸거나 괄호로 설명을 붙이지 말고, 영어 단어를 섞지 마세요.
- 정보에 없는 일(예: 전에 했던 이야기)을 지어내지 마세요.{memory_rule}"""


def counsel_opening(state: AgentState) -> AgentState:
    logger.info("[Node] 상담형 첫 마디 생성 (추천 끔)")
    from agent.memory import get_memory, speech_history_text
    extra = ""
    sh = speech_history_text(state.get("speech_history"))
    if sh:
        extra += f"\n[최근 몇 분 동안의 발화]\n{sh}\n"
    mem = get_memory()
    mem_text = mem.as_prompt(state.get("memory_key")) if mem else ""
    memory_rule = ""
    if mem_text:
        extra += f"\n[지난 대화에서 기억하는 내용]\n{mem_text}\n"
        memory_rule = ("\n- 위의 지난 대화 기억과 관련 있으면 자연스럽게 이어 주세요(예: \"어제 말씀하신 발표는 잘 끝나셨어요?\"). "
                       "힘든 이야기를 캐묻지는 마세요.")
        logger.info(f"[Node] 지난 대화 기억 {len(mem.recent(state.get('memory_key')))}개 사용")
    try:
        msg = llm.invoke([HumanMessage(content=COUNSEL_OPENING_PROMPT.format(
            face_emotion=state["face_emotion"], stress_level=state["stress_level"],
            stt_text=state["stt_text"] or "없음", extra=extra, memory_rule=memory_rule))])
        from agent.text_clean import clean_reply
        text = clean_reply(str(msg.content))
        state["recommendation"] = text or "조금 지쳐 보이세요. 괜찮으시면 지금 어떤 기분인지 이야기해 주실래요?"
    except Exception as e:
        logger.error(f"[Node] 첫 마디 생성 실패: {e}")
        state["recommendation"] = "조금 지쳐 보이세요. 괜찮으시면 지금 어떤 기분인지 이야기해 주실래요?"
    logger.info(f"[Node] 첫 마디: {state['recommendation']}")
    return state


# ── 노드 3: Tool Calling 추천 ──────────────────────────
def recommend(state: AgentState) -> AgentState:
    if not RECOMMEND_ENABLED:
        return counsel_opening(state)
    logger.info("[Node] Tool Calling 추천 시작...")

    # 대화 메모리 조회
    from backend_client import get_recent_interventions, get_recent_recommendations

    user_id = state.get("user_id", 1)
    recent_interventions = get_recent_interventions(user_id, size=3)
    recent_recommendations = get_recent_recommendations(user_id, size=5)

    history_text = _format_recent_history(recent_interventions)
    already_recommended = _format_recent_recommendations(recent_recommendations)

    logger.info(f"[Node] 최근 개입 이력:\n{history_text}")
    logger.info(f"[Node] 이미 추천한 항목:\n{already_recommended}")

    # Tool 선택 프롬프트
    prompt = f"""
사용자의 현재 상태:
- 감정: {state['face_emotion']}
- 스트레스 수준: {state['stress_level']}/10
- 최근 발화: "{state['stt_text']}"

[최근 개입 이력]
{history_text}

[이미 추천한 항목 (중복 금지)]
{already_recommended}

[Tool 선택 규칙]
1. 발화에 키워드가 있으면 해당 Tool 우선 호출:
   - 장소/어디/카페/공원/산책/스트레스 풀 → recommend_place
   - 노래/음악/듣고 싶다 → recommend_music
   - 책/읽고 싶다 → recommend_book

2. 키워드 없으면 감정({state['face_emotion']})에 맞는 Tool 선택:
   - 슬픔/공포/혐오 → recommend_music
   - 분노 → recommend_place
   - 무감정 → recommend_book
   - 행복/놀람 → 자유롭게 선택

3. 이미 추천한 항목은 절대 중복 추천하지 마세요.

반드시 Tool을 호출하세요.
"""

    try:
        response = llm_with_tools.invoke([HumanMessage(content=prompt)])

        tool_results = []
        if response.tool_calls:
            tool_map = {t.name: t for t in ALL_TOOLS}
            logger.info(f"[Node] Tool 맵: {list(tool_map.keys())}")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                logger.info(f"[Node] Tool 호출: {tool_name}({tool_args})")

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    tool_results.append(result)

            raw_results = "\n\n".join(tool_results)
            logger.info(f"[Node] Tool 원본 결과:\n{raw_results}")

            # ── Tool 결과 → 대화체 변환 ──────────────
            convert_prompt = f"""
당신은 감정 케어 AI 상담사입니다. 반드시 한국어로 답변하세요.

사용자 감정: {state['face_emotion']}
스트레스 수준: {state['stress_level']}/10

아래는 추천 결과입니다:
{raw_results}

위 추천 결과를 바탕으로 친근한 대화체로 자연스럽게 전달해주세요.
- 목록 나열 형식 절대 사용 금지 (-, * 등 불릿 포인트 사용 금지)
- 위로의 말 먼저, 그 다음 추천 1~2개만 자연스럽게 언급
- 곡명/책명/장소명은 따옴표로 감싸서 표현
- URL은 추천 항목 문장 바로 뒤에 자연스럽게 포함
- 전체 3~4문장으로 간결하게 작성
"""
            convert_response = llm.invoke([HumanMessage(content=convert_prompt)])
            state["recommendation"] = str(convert_response.content)
            logger.info(f"[Node] 대화체 변환 완료:\n{state['recommendation']}")

        else:
            logger.warning("[Node] Tool 호출 없음 → 텍스트 응답만 반환")
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
