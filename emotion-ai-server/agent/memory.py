# agent/memory.py
"""
지난 대화 기억 (2026-11-04).

대화가 끝날 때마다 LLM이 그 대화를 3~5줄로 요약해 기억 공간(memory_key)별로 저장한다.
다음에 먼저 말을 걸 때와 대화할 때 최근 요약 몇 개를 프롬프트에 넣어, 지난 이야기를 이어 갈 수 있게 한다.

- memory_key: 감지 프로그램이 정한다. 평소에는 "user-<id>", 전시에서 '새 관람객'을 누르면 새 키로 바뀌어
  앞사람의 이야기를 다음 사람에게 꺼내지 않는다.
- 저장: logs/user_memory.json (이 노트북 안). 영상·음성은 없고 요약 글만 있다.
.env
  MEMORY_ENABLED=true   MEMORY_DAYS=7   MEMORY_MAX=8
- 대화 요약 말고도, 감지 프로그램이 30분마다 보내는 '대화 밖에서 한 말'을 요약해 같은 곳에 쌓는다(평소 켜 두고 쓰는 경우).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def _env(name, default):
    v = os.getenv(name)
    return default if v is None or v.split("#")[0].strip() == "" else v.split("#")[0].strip()


MEMORY_ENABLED = _env("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_DAYS = float(_env("MEMORY_DAYS", "7"))
MEMORY_MAX = int(_env("MEMORY_MAX", "8"))
MEMORY_PATH = os.path.join("logs", "user_memory.json")
KEEP_PER_KEY = 30   # 키마다 파일에 남기는 최대 요약 수

SUMMARY_PROMPT = """다음은 AI 상담 친구와 사용자의 대화입니다. 다음에 다시 이야기할 때 참고할 수 있게 한국어로 3~5줄 요약하세요.
- 사용자가 이야기한 상황·고민 (구체적인 일, 사람, 일정이 있으면 포함)
- 사용자가 표현한 감정과 대화 끝 무렵의 상태
- 도움이 된 것 / 하기로 한 것이 있으면
- 다음에 안부를 물어볼 만한 것
대화에 없는 내용은 추측해서 쓰지 마세요. 각 줄은 '- '로 시작하세요.

[대화]
{dialog}"""


class MemoryStore:
    def __init__(self, path: str | None = MEMORY_PATH, clock=time.time):
        self.path, self.clock = path, clock
        self.lock = threading.Lock()
        self.data: dict[str, list[dict]] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning(f"[Memory] 기억 파일을 읽지 못했습니다 ({e}) → 새로 시작")

    def add(self, key: str, summary: str, session_id: str | None = None) -> None:
        if not key or not summary.strip():
            return
        with self.lock:
            items = self.data.setdefault(key, [])
            if session_id:   # 같은 대화를 다시 요약하면(끝난 뒤 이어서 말한 경우) 앞 요약을 바꾼다
                items[:] = [x for x in items if x.get("session_id") != session_id]
            items.append({"time": self.clock(), "session_id": session_id, "summary": summary.strip()})
            del items[:-KEEP_PER_KEY]
            self._save()

    def recent(self, key: str | None, days: float = MEMORY_DAYS, limit: int = MEMORY_MAX) -> list[dict]:
        if not key:
            return []
        cutoff = self.clock() - days * 86400
        with self.lock:
            items = [x for x in self.data.get(key, []) if x["time"] >= cutoff]
        return items[-limit:]

    def as_prompt(self, key: str | None) -> str:
        """프롬프트에 넣을 글. 없으면 빈 문자열."""
        items = self.recent(key)
        if not items:
            return ""
        lines = []
        for x in items:
            when = _when(x["time"], self.clock())
            lines.append(f"({when})\n{x['summary']}")
        return "\n\n".join(lines)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except Exception as e:
            logger.warning(f"[Memory] 기억 저장 실패: {e}")


def _when(t: float, now: float) -> str:
    d = datetime.fromtimestamp(t)
    days = (datetime.fromtimestamp(now).date() - d.date()).days
    hm = d.strftime("%H:%M")
    return f"오늘 {hm}" if days == 0 else f"어제 {hm}" if days == 1 else f"{days}일 전 {hm}"


def summarize_dialog(llm, messages: list[dict]) -> str:
    dialog = "\n".join(("사용자: " if m["role"] == "user" else "AI: ") + m["text"] for m in messages if m.get("text"))
    resp = llm.invoke([_human(SUMMARY_PROMPT.format(dialog=dialog[-6000:]))])
    return str(getattr(resp, "content", resp)).strip()


SPEECH_SUMMARY_PROMPT = """다음은 사용자가 컴퓨터 앞에서 대화 밖에 혼잣말처럼 한 말을 음성 인식한 기록입니다(인식 오류가 섞여 있을 수 있음).
다음에 AI 상담 친구가 안부를 물을 때 참고할 수 있게, 의미 있는 내용만 한국어 1~3줄로 요약하세요.
- 무슨 일을 하고 있었는지, 어떤 걱정·감정이 드러났는지
- 의미 없는 잡음·인사·혼잣말뿐이면 '없음'이라고만 쓰세요.
추측하지 마세요. 각 줄은 '- '로 시작하세요.

[기록]
{lines}"""


def summarize_speech(llm, items: list[dict]) -> str:
    """대화 밖 발화 [{time, text}] → 요약. 의미 없으면 ''."""
    lines = "\n".join(f"[{x.get('time', '')}] {x.get('text', '')}" for x in items if x.get("text"))
    if not lines.strip():
        return ""
    resp = llm.invoke([_human(SPEECH_SUMMARY_PROMPT.format(lines=lines[-6000:]))])
    text = str(getattr(resp, "content", resp)).strip()
    return "" if not text or text.replace("-", "").strip().startswith("없음") else "[혼잣말 요약]\n" + text


def _human(text):
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=text)


def speech_history_text(history: list[dict] | None) -> str:
    """감지 프로그램이 보낸 최근 발화 [{time:'HH:MM:SS', text}] → 프롬프트용 글."""
    if not history:
        return ""
    return "\n".join(f"[{h.get('time', '')}] {h.get('text', '')}" for h in history[-30:] if h.get("text"))


_store: MemoryStore | None = None


def get_memory() -> MemoryStore | None:
    global _store
    if not MEMORY_ENABLED:
        return None
    if _store is None:
        _store = MemoryStore()
    return _store
