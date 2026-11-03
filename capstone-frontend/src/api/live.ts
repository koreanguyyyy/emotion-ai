import type { LiveState } from "@/types";

// 감지 프로그램(emotion-detection-fusion)이 이 컴퓨터에서 여는 실시간 상태 통로.
// 백엔드 기록은 1분마다·개입 기준을 넘을 때만 저장되므로, 모니터링 화면은 이 통로를 1초마다 읽는다.
const LIVE_URL = import.meta.env.VITE_LIVE_URL || "http://127.0.0.1:8765/state";

const LIVE_BASE = LIVE_URL.replace(/\/state\/?$/, "");

async function post(path: string, body: object = {}): Promise<{ ok: boolean; reason?: string }> {
  const res = await fetch(`${LIVE_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`live ${res.status}`);
  return res.json();
}

// 웹 채팅: 보낸 말에 대한 답은 /state의 chat.messages에 나타난다 (음성으로도 읽힘)
export const sendChatMessage = (message: string) => post("/chat", { message });
export const startChat = () => post("/chat/start");
export const endChat = () => post("/chat/end");
// 시연용 '지금 말 걸기': 지수와 상관없이 다음 판단 주기(기본 5초 안)에 개입 흐름 실행
export const triggerIntervention = () => post("/intervene");
// 전시용 '새 관람객': 기준선 다시 보정 + 대화·쿨다운 비우기
export const resetVisitor = () => post("/reset");

export async function getLiveState(timeoutMs = 800): Promise<LiveState> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(LIVE_URL, { signal: ctrl.signal, cache: "no-store" });
    if (!res.ok) throw new Error(`live ${res.status}`);
    return (await res.json()) as LiveState;
  } finally {
    clearTimeout(timer);
  }
}
