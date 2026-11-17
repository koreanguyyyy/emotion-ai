# ai/conversation.py
"""
대화 모드 — 에이전트가 먼저 말을 건 뒤, 사용자가 말(음성)이나 글(웹 채팅)로 대답하면 대화를 이어 간다.

  개입 → start() : 첫 메시지를 읽어 주고 대화 모드로 들어간다
  음성 인식 결과 / 웹 채팅 → submit() : 에이전트 서버 /chat 에 보내고 답을 음성으로 읽는다
  끝 : 에이전트가 작별 인사로 끝냈을 때(ended), 웹의 '대화 끝내기', 또는 CHAT_IDLE_SEC 동안 말이 없을 때

- 대화 중에는 새 개입을 하지 않는다(main.py의 스트레스 루프가 is_active()를 확인).
- 에이전트가 말하는 동안 마이크에 들어온 소리는 버린다(detection/tts.py의 was_speaking_since).
- 웹 화면은 snapshot()을 실시간 상태 통로(/state)로 1초마다 읽는다.

.env
  CHAT_ENABLED=true     대화 모드 사용 (false면 1학기처럼 한 번 말하고 끝)
  CHAT_IDLE_SEC=90      이 시간 동안 아무 말이 없으면 대화를 끝낸다
  CHAT_TIMEOUT_SEC=40   에이전트 서버 답변 대기 시간
  CHAT_SPEAK_WEB=true   웹 채팅으로 쓴 말에 대한 답도 음성으로 읽기
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable

URL_RE = re.compile(r"https?://\S+")

# Whisper가 조용할 때 자주 지어내는 문장 (대화 중에 이걸 사용자 말로 보내면 안 된다)
WHISPER_HALLUCINATIONS = ("시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", "구독과 좋아요",
                          "다음 영상에서 만나요", "MBC 뉴스", "자막 제공", "감사합니다.")
# 짧은 인식 결과에 이 말이 들어 있으면 지어낸 문장으로 본다 ("오늘도 시청해주셔서 정말 감사합니다!" 등)
HALLUCINATION_PARTS = ("시청해", "시청 해", "구독", "좋아요 눌러", "알림 설정", "MBC", "KBS", "SBS", "자막", "영상에서 만나")
MIN_VOICE_CHARS = 2
FREE_OPENING = "안녕하세요. 요즘 기분은 어떠세요? 편하게 이야기해 주세요."


def for_speech(text: str) -> str:
    """음성으로 읽을 문장: URL과 마크다운 기호를 뺀다(예전에는 URL까지 소리 내어 읽었다)."""
    t = URL_RE.sub("", text or "")
    t = re.sub(r"[*#`>\[\]()]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def is_noise(text: str) -> bool:
    t = (text or "").strip()
    if len(t.replace(" ", "")) < MIN_VOICE_CHARS:
        return True
    if len(t) <= 30 and any(p in t for p in HALLUCINATION_PARTS):
        return True
    return any(t == h or t.startswith(h) for h in WHISPER_HALLUCINATIONS) and len(t) <= 20


class Conversation:
    """send(payload) → dict | None : 에이전트 서버 /chat 호출 (시험에서 가짜로 바꿀 수 있게 주입)
    speak(text) → None : 음성 출력 (끝날 때까지 기다림)
    get_context() → dict : 지금 표정·스트레스·심박"""

    def __init__(self, send: Callable[[dict], dict | None], speak: Callable[[str], object],
                 get_context: Callable[[], dict], user_id: int = 1,
                 idle_sec: float | None = None, clock: Callable[[], float] = time.time,
                 send_stream=None, speak_part=None, speak_wait=None,
                 get_memory_key=None, get_speech_history=None, on_end=None):
        """send_stream(payload) → 이벤트 반복자({"type":"delta"|"done", ...}) — 있으면 답을 글자 단위로 받아
        웹에 바로 보여주고, 문장이 완성되는 대로 speak_part(문장)으로 읽는다(끝나면 speak_wait())."""
        self.send, self.speak, self.get_context = send, speak, get_context
        self.send_stream, self.speak_part, self.speak_wait = send_stream, speak_part, speak_wait
        # 지난 대화 기억 (2026-11-04): 기억 공간 키, 대화 전 최근 발화, 대화가 끝났을 때 서버에 알리기(요약 저장)
        self.get_memory_key, self.get_speech_history, self.on_end = get_memory_key, get_speech_history, on_end
        self.user_id = user_id
        self.idle_sec = float(os.getenv("CHAT_IDLE_SEC", "90")) if idle_sec is None else idle_sec
        self.clock = clock
        self.speak_web = os.getenv("CHAT_SPEAK_WEB", "true").lower() == "true"
        self.lock = threading.Lock()
        self.turn_lock = threading.Lock()   # 한 번에 한 턴만 (음성과 웹이 동시에 들어와도 순서대로)
        self.active = False
        self.busy = False                   # 에이전트가 답을 만드는 중
        self.session_id: str | None = None
        self.intervention_id: int | None = None
        self.opening: str | None = None
        self.messages: list[dict] = []
        self.last_activity = 0.0
        self.end_reason: str | None = None
        self._seq = 0

    # ── 상태 ──
    def is_active(self) -> bool:
        self.check_idle()
        return self.active

    def _add(self, role: str, text: str, source: str, links=None) -> dict:
        self._seq += 1
        msg = {"id": self._seq, "role": role, "text": text, "source": source,
               "time": self.clock(), "links": links or []}
        self.messages.append(msg)
        self.messages = self.messages[-60:]
        return msg

    def snapshot(self) -> dict:
        self.check_idle()
        with self.lock:
            return {"active": self.active, "busy": self.busy, "sessionId": self.session_id,
                    "interventionId": self.intervention_id, "endReason": self.end_reason,
                    "idleLeftSec": (max(0, round(self.idle_sec - (self.clock() - self.last_activity)))
                                    if self.active else None),
                    "messages": list(self.messages)}

    # ── 시작 / 끝 ──
    def start(self, opening: str, intervention_id: int | None = None, speak: bool = True) -> None:
        """개입 직후 호출. 첫 메시지를 기록하고(음성으로 읽고) 대화 모드로 들어간다."""
        with self.lock:
            self.active, self.busy, self.end_reason = True, False, None
            self.intervention_id = intervention_id
            self.session_id = f"iv-{intervention_id}" if intervention_id else f"free-{int(self.clock())}"
            self.opening = opening
            self.messages = []
            self._add("agent", opening, "intervention")
            self.last_activity = self.clock()
        if speak:
            self._speak(opening)

    def reset(self) -> None:
        """새 관람객: 대화를 끝내고 화면의 대화 기록도 비운다."""
        if self.active:
            self._notify_end()   # 앞사람 대화는 앞사람 기억 공간에 요약해 둔다
        with self.lock:
            self.active, self.busy = False, False
            self.session_id = self.intervention_id = self.opening = None
            self.messages, self.end_reason = [], None

    def start_free(self) -> None:
        """개입 없이 사용자가 먼저 대화를 시작할 때 (웹의 '대화 시작' 버튼)."""
        self.start(FREE_OPENING, None)

    def end(self, reason: str = "user") -> None:
        with self.lock:
            ended = self.active
            if self.active:
                self.active, self.end_reason = False, reason
                print(f"[Chat] 대화 종료 ({reason})")
        if ended:
            self._notify_end()

    def _notify_end(self) -> None:
        """대화가 끝났다고 서버에 알린다 → 서버가 요약해서 기억에 남긴다. 대화가 오가지 않았으면 생략."""
        sid = self.session_id
        if not self.on_end or not sid or not any(m["role"] == "user" for m in self.messages):
            return
        def run():
            try:
                self.on_end(sid)
            except Exception as e:
                print(f"[Chat] 대화 종료 알림 실패: {e}")
        threading.Thread(target=run, daemon=True).start()

    def check_idle(self) -> None:
        idle = False
        with self.lock:
            if self.active and not self.busy and self.clock() - self.last_activity > self.idle_sec:
                self.active, self.end_reason = False, "idle"
                idle = True
                print(f"[Chat] {self.idle_sec:.0f}초 동안 말이 없어 대화를 끝냅니다")
        if idle:
            self._notify_end()

    def _speak(self, text: str) -> None:
        t = for_speech(text)
        if t:
            try:
                self.speak(t)
            except Exception as e:
                print(f"[Chat] 음성 출력 실패: {e}")
        with self.lock:
            self.last_activity = self.clock()   # 말이 끝난 때부터 기다리는 시간을 센다

    # ── 한 턴 ──
    def submit(self, text: str, source: str = "voice") -> dict | None:
        """사용자 말 한 마디를 처리하고 에이전트의 답(dict)을 돌려준다. 대화 중이 아니면 None.
        음성 인식 결과는 대화 중일 때만 받는다. 웹 채팅은 대화가 없으면 새로 시작한다."""
        text = (text or "").strip()
        if not text:
            return None
        if source == "voice":
            if not self.is_active() or self.busy or is_noise(text):
                return None
        elif not self.is_active():
            self.start(FREE_OPENING, None, speak=False)

        with self.turn_lock:
            with self.lock:
                self.busy = True
                self.last_activity = self.clock()
                self._add("user", text, source)
                payload = {"session_id": self.session_id, "user_id": self.user_id, "message": text,
                           "intervention_id": self.intervention_id, "opening": self.opening,
                           "context": self.get_context()}
                if self.get_memory_key:
                    payload["memory_key"] = self.get_memory_key()
                if self.get_speech_history:
                    payload["speech_history"] = self.get_speech_history()
            print(f"[Chat] 사용자({source}): {text}")
            want_voice = source == "voice" or self.speak_web
            result = self._stream_turn(payload, want_voice) if self.send_stream else None
            if result is None:   # 스트리밍이 없거나, 한 글자도 못 받고 실패 → 한 번에 받기
                try:
                    result = self.send(payload)
                except Exception as e:
                    print(f"[Chat] 에이전트 서버 오류: {e}")
                if not result or not result.get("reply"):
                    reply = "미안해요, 지금 답을 가져오지 못했어요. 잠시 뒤에 다시 말해 주세요."
                    result = {"reply": reply, "ended": False, "recommendations": []}
                with self.lock:
                    self._add("agent", result["reply"], "agent", result.get("recommendations"))
                if want_voice:
                    self._speak(result["reply"])
            print(f"[Chat] 에이전트: {result['reply']}")
            with self.lock:
                self.busy = False
                self.last_activity = self.clock()
                if result.get("ended"):
                    self.active, self.end_reason = False, "agent"
            if result.get("ended"):
                print("[Chat] 에이전트가 대화를 마무리했습니다")
            return result

    def _stream_turn(self, payload: dict, want_voice: bool) -> dict | None:
        """답을 조각으로 받아 화면 메시지를 늘려 가고, 완성된 문장부터 읽는다. 아무것도 못 받으면 None."""
        from detection.tts import SentenceBuffer   # 가벼운 모듈 (음성 장치 불필요)
        with self.lock:
            msg = self._add("agent", "", "agent")
            msg["streaming"] = True
        buf, result, got = SentenceBuffer(), None, False
        speak_live = want_voice and self.speak_part is not None
        try:
            for ev in self.send_stream(payload):
                if ev.get("type") == "delta" and ev.get("text"):
                    got = True
                    with self.lock:
                        msg["text"] += ev["text"]
                        self.last_activity = self.clock()
                    if speak_live:
                        for sent in buf.feed(ev["text"]):
                            self._speak_part(sent)
                elif ev.get("type") == "done":
                    result = ev
        except Exception as e:
            print(f"[Chat] 스트리밍 오류: {e}")
        if result is None and not got:
            with self.lock:   # 빈 말풍선을 치우고 한 번에 받기로 넘어간다
                if msg in self.messages:
                    self.messages.remove(msg)
            return None
        if result is None:   # 중간에 끊김 → 받은 데까지
            result = {"reply": msg["text"].strip(), "ended": False, "recommendations": []}
        with self.lock:
            msg["text"] = result.get("reply") or msg["text"]
            msg["links"] = result.get("recommendations") or []
            msg["streaming"] = False
        if want_voice:
            if speak_live:
                self._speak_part(buf.flush())
                if self.speak_wait:
                    self.speak_wait()
                with self.lock:
                    self.last_activity = self.clock()
            else:
                self._speak(result["reply"])
        return result

    def _speak_part(self, text: str) -> None:
        t = for_speech(text)
        if t:
            try:
                self.speak_part(t)
            except Exception as e:
                print(f"[Chat] 음성 출력 실패: {e}")

    def submit_async(self, text: str, source: str = "voice") -> None:
        threading.Thread(target=self.submit, args=(text, source), daemon=True).start()


# ── 프로그램 전체에서 쓰는 하나의 대화 (main.py에서 configure) ──
_conversation: Conversation | None = None


def configure(conv: Conversation) -> None:
    global _conversation
    _conversation = conv


def get_conversation() -> Conversation | None:
    return _conversation


def on_voice_text(text: str) -> bool:
    """음성 인식 결과가 나올 때마다 호출. 대화 중이면 대화로 넘기고 True."""
    c = _conversation
    if c is None or not c.is_active() or c.busy:
        return False
    if is_noise(text):
        return True   # 대화 중의 잡음은 대화로도, 스트레스 버퍼로도 보내지 않는다
    c.submit_async(text, "voice")
    return True
