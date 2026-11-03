# agent/text_clean.py
"""LLM 답 다듬기 (2026-11-04). 작은 로컬 모델(llama3.1 등)이 붙이는 군더더기를 지운다:
'[문장](음성으로 읽힙니다)' 같은 마크다운, 메타 설명 괄호, 전체를 감싼 따옴표, 'AI:' 같은 이름표, 별표."""
import re

_META_PAREN = re.compile(r"\((?:[^()]*?(?:음성|읽힙|읽어|TTS|tts|말투|예시|참고)[^()]*?)\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LABEL = re.compile(r"^\s*(?:AI|에이전트|상담사|assistant|Assistant|답변|응답)\s*[:：]\s*")


def clean_reply(text: str) -> str:
    t = (text or "").strip()
    t = _MD_LINK.sub(r"\1", t)
    t = _META_PAREN.sub("", t)
    t = re.sub(r"[*#`]+", "", t)
    t = _LABEL.sub("", t)
    t = re.sub(r"\s*\n\s*\n\s*", " ", t)            # 문단 나눔은 한 칸 띄우기로
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    if len(t) >= 2 and t[0] in "\"'“‘「" and t[-1] in "\"'”’」":
        t = t[1:-1].strip()
    return t
