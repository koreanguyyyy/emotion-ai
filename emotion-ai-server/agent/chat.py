# agent/chat.py
"""
이어지는 대화 — 에이전트가 먼저 말을 건 뒤(/analyze) 사용자가 대답하면 대화를 이어 간다.

이전에는 위로 한마디 + 추천을 읽어 주면 끝났다. 여기서는 개입마다 대화 세션을 만들고
지난 대화를 기억한 채로 답한다. 사용자가 "다른 노래 없어?"처럼 추천을 원하면 추천 도구를 다시 부르고,
새 추천은 DB(recommendations)에 그 개입의 추천으로 저장한다.

  POST /chat  {session_id, user_id, message, intervention_id?, opening?, context?}
       → {session_id, reply, ended, recommendations: [{category, title, artist, url}]}
  GET  /chat/{session_id} → 대화 기록

- reply는 음성으로 읽기 때문에 URL을 넣지 않는다. 링크는 recommendations로 따로 돌려준다.
- 사용자가 대화를 끝내려 하면 LLM이 인사 뒤에 <END>를 붙이고, ended=true로 알려 준다.
- 대화 내용은 logs/chat_sessions.jsonl에 한 줄씩 남는다(평가 자료). 세션은 메모리에만 두고 2시간 뒤 지운다.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

SESSION_TTL_SEC = 2 * 3600
MAX_SESSIONS = 50
MAX_HISTORY = 40          # LLM에 넘기는 최근 대화 수 (사용자·에이전트 합쳐서)
MEMORY_GUIDE = ("[지난 대화에서 기억하는 내용 — 자연스럽게 이어 가세요(예: 지난번 걱정하던 일의 안부). "
                "사용자가 먼저 꺼내지 않은 힘든 이야기를 캐묻지는 마세요]\n")
SPEECH_GUIDE = "[대화 전 최근 몇 분 동안 사용자가 혼잣말처럼 한 말 — 참고만 하세요]\n"
MAX_TOOL_ROUNDS = 2
CHAT_LOG = os.path.join("logs", "chat_sessions.jsonl")
END_MARK = "<END>"

CATEGORY_BY_TOOL = {"recommend_music": "MUSIC", "recommend_book": "BOOK", "recommend_place": "PLACE"}
URL_RE = re.compile(r"https?://\S+")

SYSTEM_PROMPT = """당신은 사용자의 감정을 돌보는 AI 친구입니다. 반드시 한국어로, 친근한 존댓말로 답하세요.
당신의 말은 음성으로 읽힙니다.
- 한 번에 1~3문장으로 짧게 말하세요. 목록·불릿·이모지·URL은 쓰지 마세요.
- 먼저 사용자의 말에 공감하고, 필요하면 부드러운 질문 하나로 대화를 이어 가세요. 훈계나 진단은 하지 마세요.
- 사용자가 음악·책·장소를 원하거나 기분 전환이 필요해 보이면 추천 도구를 부르세요.
  도구의 face_emotion에는 지금 원하는 분위기에 맞는 감정을 넣으세요(예: 신나는 노래 → 행복, 차분한 노래 → 무감정).
  추천할 때는 결과 중 1~2개만 제목을 작은따옴표로 감싸 말하세요. 링크는 화면에 따로 보여 주니 말하지 마세요.
- 이미 추천한 항목은 다시 추천하지 마세요.
- 사용자가 고맙다며 마무리하거나, 그만 이야기하고 싶어 하거나, 작별 인사를 하면 짧게 인사한 뒤 맨 끝에 <END>를 붙이세요.
- 사용자가 스스로를 해치고 싶다는 뜻을 비치면 진지하게 걱정을 표현하고, 자살예방상담전화 109(24시간)나
  가까운 사람에게 지금 연락해 보라고 권하세요. 이때는 대화를 끝내지 마세요.

[지금 감지된 상태 — 참고만 하고 숫자를 그대로 말하지는 마세요]
{context}"""

# 추천을 끈 상담형 대화 (RECOMMEND_ENABLED=false, 2026-11-04)
COUNSEL_PROMPT = """당신은 사용자의 마음을 살피는 AI 상담 친구입니다. 반드시 한국어로, 따뜻한 존댓말로 답하세요.
당신의 말은 음성으로 읽힙니다.
- 한 번에 1~3문장으로 짧게 말하세요. 목록·불릿·이모지·URL은 쓰지 마세요.
- 소리 내어 말할 문장만 쓰세요. 따옴표로 감싸거나 괄호로 설명을 붙이지 말고, 영어 단어를 섞지 마세요.
- 사용자의 말을 잘 듣고, 느끼는 감정을 짚어 주며 공감하세요(예: "그 상황이면 정말 속상하셨겠어요").
- 질문은 한 번에 하나만, 편하게 답할 수 있는 열린 질문으로 하세요. 캐묻거나 훈계하지 마세요.
- 해결책을 서두르지 마세요. 사용자가 원하거나 대화가 충분히 이어졌을 때만, 지금 바로 할 수 있는 작은 마음 돌보기
  (천천히 숨 쉬기, 잠깐 일어나 스트레칭, 물 한 잔, 생각 정리하기 등)를 하나 부드럽게 제안하세요.
- 음악·책·장소·영상 추천은 하지 마세요. 진단하거나 약을 권하지 마세요.
- 사용자가 고맙다며 마무리하거나, 그만 이야기하고 싶어 하거나, 작별 인사를 하면 짧게 인사한 뒤 맨 끝에 <END>를 붙이세요.
- 사용자가 스스로를 해치고 싶다는 뜻을 비치면 진지하게 걱정을 표현하고, 자살예방상담전화 109(24시간)나
  가까운 사람에게 지금 연락해 보라고 권하세요. 이때는 대화를 끝내지 마세요.

[지금 감지된 상태 — 참고만 하고 숫자를 그대로 말하지는 마세요]
{context}"""


def _context_text(ctx: dict | None) -> str:
    if not ctx:
        return "정보 없음"
    parts = []
    if ctx.get("face_emotion"):
        parts.append(f"표정: {ctx['face_emotion']}")
    if ctx.get("stress_index") is not None:
        parts.append(f"스트레스 지수: {float(ctx['stress_index']):.1f}/10")
    if ctx.get("heart_rate") is not None:
        hr = f"심박: {float(ctx['heart_rate']):.0f}bpm"
        if ctx.get("resting_heart_rate") is not None:
            hr += f" (평상시 {float(ctx['resting_heart_rate']):.0f})"
        parts.append(hr)
    return ", ".join(parts) or "정보 없음"


def parse_tool_items(tool_name: str, raw: str) -> list[dict]:
    """추천 도구 출력("- 제목 / 아티스트\\n  URL")을 항목 목록으로 바꾼다."""
    items, current = [], None
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("- "):
            body = s[2:]
            title, _, artist = body.partition(" / ")
            current = {"category": CATEGORY_BY_TOOL.get(tool_name, "MUSIC"),
                       "title": title.strip(), "artist": artist.strip() or None, "url": None}
            items.append(current)
        elif current is not None and s.startswith("http") and current["url"] is None:
            current["url"] = s.split()[0]
    return [i for i in items if i["title"]]


def pick_mentioned(items: list[dict], reply: str, limit: int = 2) -> list[dict]:
    """답에 제목이 나온 항목만 고른다. 하나도 없으면 첫 항목(도구를 불렀다면 최소 하나는 보여 준다)."""
    norm = lambda s: re.sub(r"\s+", "", s or "").lower()
    r = norm(reply)
    hit = [i for i in items if norm(i["title"]) and norm(i["title"]) in r]
    return (hit or items[:1])[:limit]


class ChatManager:
    """llm: .invoke(messages) → AIMessage, llm_with_tools: 도구가 묶인 LLM, tools: LangChain tool 목록,
    save_rec: 추천 저장 함수(intervention_id, category, title, artist_or_author, external_url, reason) 또는 None.
    시험에서 가짜 LLM을 넣을 수 있게 밖에서 받는다."""

    def __init__(self, llm, llm_with_tools, tools, save_rec=None, log_path: str | None = CHAT_LOG,
                 use_tools: bool = True, memory=None):
        self.llm, self.llm_tools = llm, llm_with_tools
        self.use_tools = use_tools   # False면 추천 도구 없이 상담형으로 대화
        self.memory = memory         # agent.memory.MemoryStore (없으면 지난 대화를 기억하지 않음)
        self.tools = {t.name: t for t in tools}
        self.save_rec = save_rec
        self.log_path = log_path
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()

    # ── 세션 ──
    def _cleanup(self) -> None:
        now = time.time()
        for sid in [s for s, v in self.sessions.items() if now - v["updated"] > SESSION_TTL_SEC]:
            del self.sessions[sid]
        while len(self.sessions) > MAX_SESSIONS:
            del self.sessions[min(self.sessions, key=lambda s: self.sessions[s]["updated"])]

    def get_or_create(self, session_id: str, user_id: int, intervention_id: int | None = None,
                      opening: str | None = None, memory_key: str | None = None,
                      speech_history: list | None = None) -> dict:
        with self.lock:
            self._cleanup()
            s = self.sessions.get(session_id)
            if s is None:
                s = {"session_id": session_id, "user_id": user_id, "intervention_id": intervention_id,
                     "messages": [], "recommended": [], "ended": False, "summarized": False,
                     "memory_key": memory_key, "speech_history": speech_history or [],
                     "created": time.time(), "updated": time.time()}
                if opening:
                    s["messages"].append({"role": "agent", "text": opening, "time": time.time()})
                    s["recommended"] += re.findall(r"'([^']+)'", opening)
                self.sessions[session_id] = s
                self._log(s, "agent", opening, kind="opening") if opening else None
            return s

    def history(self, session_id: str) -> dict | None:
        s = self.sessions.get(session_id)
        return None if s is None else {k: s[k] for k in ("session_id", "intervention_id", "messages", "ended")}

    def _log(self, s: dict, role: str, text: str, **extra) -> None:
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "session_id": s["session_id"],
                   "user_id": s["user_id"], "intervention_id": s["intervention_id"],
                   "role": role, "text": text, **extra}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:   # 기록 실패가 대화를 막으면 안 된다
            logger.warning(f"[Chat] 대화 기록 실패: {e}")

    # ── 대화 한 턴 ──
    def _begin(self, session_id, user_id, message, intervention_id, opening, context,
               memory_key=None, speech_history=None):
        """사용자 말을 기록하고 LLM에 넘길 메시지 목록을 만든다. 빈 말이면 (s, None)."""
        s = self.get_or_create(session_id, user_id, intervention_id, opening, memory_key, speech_history)
        if memory_key:
            s["memory_key"] = memory_key
        message = (message or "").strip()
        if not message:
            return s, None
        s["messages"].append({"role": "user", "text": message, "time": time.time()})
        s["updated"] = time.time()
        s["ended"] = False   # 끝난 뒤에 다시 말하면 이어서 대화한다
        s["summarized"] = False
        self._log(s, "user", message, context=context)

        sys = (SYSTEM_PROMPT if self.use_tools else COUNSEL_PROMPT).format(context=_context_text(context))
        mem = self.memory.as_prompt(s.get("memory_key")) if self.memory else ""
        if mem:
            sys += "\n\n" + MEMORY_GUIDE + mem
        if s.get("speech_history"):
            from agent.memory import speech_history_text
            sh = speech_history_text(s["speech_history"])
            if sh:
                sys += "\n\n" + SPEECH_GUIDE + sh
        if s["recommended"] and self.use_tools:
            sys += "\n\n[이 대화에서 이미 추천한 항목]\n" + "\n".join(f"- {t}" for t in s["recommended"][-10:])
        msgs = [SystemMessage(content=sys)]
        for m in s["messages"][-MAX_HISTORY:]:
            msgs.append(HumanMessage(content=m["text"]) if m["role"] == "user" else AIMessage(content=m["text"]))
        return s, msgs

    def _finish(self, s, raw_reply: str, items: list, tools_used: list) -> dict:
        """<END>·URL을 정리하고 추천을 저장·기록한 뒤 결과 dict를 만든다."""
        from agent.text_clean import clean_reply
        ended = END_MARK in raw_reply
        reply = clean_reply(URL_RE.sub("", raw_reply.replace(END_MARK, "")))
        recs = pick_mentioned(items, reply) if items else []
        for r in recs:
            s["recommended"].append(r["title"])
            if self.save_rec and s["intervention_id"]:
                try:
                    self.save_rec(intervention_id=s["intervention_id"], category=r["category"], title=r["title"],
                                  artist_or_author=r["artist"], external_url=r["url"], reason="대화 중 추천")
                except Exception as e:
                    logger.warning(f"[Chat] 추천 저장 실패: {e}")
        s["messages"].append({"role": "agent", "text": reply, "time": time.time(), "links": recs})
        s["ended"] = ended
        s["updated"] = time.time()
        self._log(s, "agent", reply, ended=ended, tools=tools_used, recommendations=recs)
        if ended:
            self.end_session(s["session_id"])
        return {"session_id": s["session_id"], "reply": reply, "ended": ended, "recommendations": recs}

    def end_session(self, session_id: str, background: bool = True) -> bool:
        """대화가 끝났을 때(마무리 인사·웹 '끝내기'·무응답) 요약해서 기억에 남긴다. 요약했으면 True."""
        s = self.sessions.get(session_id)
        if (s is None or s.get("summarized") or not self.memory or not s.get("memory_key")
                or not any(m["role"] == "user" for m in s["messages"])):
            return False
        s["summarized"] = True
        msgs, key = list(s["messages"]), s["memory_key"]

        def run():
            from agent.memory import summarize_dialog
            try:
                summary = summarize_dialog(self.llm, msgs)
                self.memory.add(key, summary, session_id)
                self._log(s, "memory", summary)
                logger.info(f"[Memory] 대화 요약 저장 ({key}):\n{summary}")
            except Exception as e:
                s["summarized"] = False
                logger.warning(f"[Memory] 요약 실패: {e}")
        if background:
            threading.Thread(target=run, daemon=True).start()
        else:
            run()
        return True

    def _generate(self, msgs):
        """도구 호출을 포함해 답을 한 번에 만든다. (답, 추천 항목, 쓴 도구)"""
        items: list[dict] = []
        tools_used: list[str] = []
        try:
            reply = None
            for _ in range(MAX_TOOL_ROUNDS):
                resp = (self.llm_tools if self.use_tools else self.llm).invoke(msgs)
                calls = getattr(resp, "tool_calls", None) or []
                if not calls:
                    reply = str(resp.content)
                    break
                msgs.append(resp)
                for call in calls:
                    name, args = call["name"], call.get("args") or {}
                    tool = self.tools.get(name)
                    out = str(tool.invoke(args)) if tool else f"알 수 없는 도구: {name}"
                    logger.info(f"[Chat] 도구 호출 {name}({args})")
                    tools_used.append(name)
                    items += parse_tool_items(name, out)
                    msgs.append(ToolMessage(content=out, tool_call_id=call.get("id") or name))
            if reply is None:   # 도구를 계속 부르면 도구 없이 한 번 더 답을 받는다
                reply = str(self.llm.invoke(msgs).content)
        except Exception as e:
            logger.error(f"[Chat] 답변 생성 실패: {e}")
            reply = FAIL_REPLY
        return reply, items, tools_used

    def chat(self, session_id: str, user_id: int, message: str, intervention_id: int | None = None,
             opening: str | None = None, context: dict | None = None,
             memory_key: str | None = None, speech_history: list | None = None) -> dict:
        s, msgs = self._begin(session_id, user_id, message, intervention_id, opening, context,
                              memory_key, speech_history)
        if msgs is None:
            return {"session_id": session_id, "reply": "", "ended": s["ended"], "recommendations": []}
        reply, items, tools_used = self._generate(msgs)
        return self._finish(s, reply, items, tools_used)

    def chat_stream(self, session_id: str, user_id: int, message: str, intervention_id: int | None = None,
                    opening: str | None = None, context: dict | None = None,
                    memory_key: str | None = None, speech_history: list | None = None):
        """답을 조각으로 흘려보낸다: {"type": "delta", "text": ...} … 마지막에 {"type": "done", ...결과}.
        상담형(도구 없음)은 LLM 스트리밍을 그대로 쓰고, 추천 도구를 쓰는 경우는 한 번에 만든 답을 한 조각으로 보낸다.
        <END> 표시는 조각으로 보내지 않는다(글자 단위로 나뉘어 와도 화면에 보이지 않게 끝부분을 잠시 붙잡아 둔다)."""
        s, msgs = self._begin(session_id, user_id, message, intervention_id, opening, context,
                              memory_key, speech_history)
        if msgs is None:
            yield {"type": "done", "session_id": session_id, "reply": "", "ended": s["ended"], "recommendations": []}
            return
        if self.use_tools:
            reply, items, tools_used = self._generate(msgs)
            result = self._finish(s, reply, items, tools_used)
            if result["reply"]:
                yield {"type": "delta", "text": result["reply"]}
            yield {"type": "done", **result}
            return

        full, sent = "", 0
        try:
            for chunk in self.llm.stream(msgs):
                piece = getattr(chunk, "content", chunk)
                if not isinstance(piece, str) or not piece:
                    continue
                full += piece
                safe = _safe_prefix_len(full)
                if safe > sent:
                    yield {"type": "delta", "text": full[sent:safe]}
                    sent = safe
        except Exception as e:
            logger.error(f"[Chat] 스트리밍 실패: {e}")
            if not full:
                full = FAIL_REPLY
        result = self._finish(s, full, [], [])
        # 붙잡아 둔 끝부분(또는 <END>를 뺀 나머지)을 마저 보낸다
        visible = full.replace(END_MARK, "")
        if len(visible.rstrip()) > sent:
            yield {"type": "delta", "text": visible[sent:].rstrip()}
        yield {"type": "done", **result}


FAIL_REPLY = "미안해요, 잠깐 생각이 끊겼어요. 한 번만 다시 말해 주실래요?"


def _safe_prefix_len(text: str) -> int:
    """화면에 내보내도 되는 길이: <END>와, <END>의 앞부분일 수 있는 끝 글자는 빼고 센다."""
    i = text.find(END_MARK)
    n = i if i >= 0 else len(text)
    if i < 0:
        for k in range(len(END_MARK) - 1, 0, -1):
            if text.endswith(END_MARK[:k]):
                n = len(text) - k
                break
    return len(text[:n].rstrip())   # 끝 공백도 다음 글자가 올 때까지 붙잡는다 (최종 답은 앞뒤 공백을 지움)
