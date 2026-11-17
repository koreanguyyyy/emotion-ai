import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import ChatBubble from "@/components/chat/ChatBubble";
import ConversationBubble from "@/components/chat/ConversationBubble";
import { Button } from "@/components/ui/button";
import { Camera, CameraOff, Loader2, MessageCircle, Send, UserPlus, X, Zap } from "lucide-react";
import { getRecentEmotionLogs } from "@/api/emotion";
import {
  getInterventions,
  submitInterventionFeedback,
} from "@/api/intervention";
import { endChat, getLiveState, LIVE_VIDEO_URL, resetVisitor, sendChatMessage, startChat, triggerIntervention } from "@/api/live";
import type { EmotionLog, Intervention, InterventionRating, LiveState } from "@/types";
import { EMOTION_LABELS, HR_MODE_LABELS, logEmotion, logStress, normalizeEmotion } from "@/types";
import { format } from "date-fns";
import { useTheme } from "@/hooks/useTheme";

const USER_ID = Number(import.meta.env.VITE_USER_ID) || 1;
const POLL_INTERVAL = 5000;
const LIVE_INTERVAL = 1000;
const LIVE_INTERVAL_BUSY = 200;   // 에이전트가 답하는 중에는 자주 읽어 글자가 이어지듯 보이게
const MAX_BUBBLES = 20;

const EMOTION_EMOJI: Record<string, string> = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  fear: "😰",
  disgust: "🤢",
  surprise: "😲",
  neutral: "😐",
};

function stressColor(level: number): string {
  if (level <= 3) return "#22c55e";
  if (level <= 6) return "#eab308";
  if (level <= 8) return "#f97316";
  return "#ef4444";
}

interface ChatMessage {
  interventionId: number;
  message: string;
  timestamp: string;
  feedbackGiven: boolean;
}

export default function Monitor() {
  const { dark } = useTheme();
  const [latestLog, setLatestLog] = useState<EmotionLog | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const lastInterventionIdRef = useRef<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [live, setLive] = useState<LiveState | null>(null);
  const [liveOk, setLiveOk] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 3000);
  }, []);

  // Initial load: fetch recent 3 interventions as history
  useEffect(() => {
    async function init() {
      try {
        const [logs, interventions] = await Promise.all([
          getRecentEmotionLogs(USER_ID, 1).catch(() => []),
          getInterventions(USER_ID, 3).catch(() => []),
        ]);

        if (logs.length > 0) setLatestLog(logs[0]);

        const history: ChatMessage[] = interventions
          .reverse()
          .map((iv: Intervention) => ({
            interventionId: iv.interventionId,
            message: iv.agentMessage,
            timestamp: iv.createdAt,
            feedbackGiven: iv.interventionRating !== null,
          }));

        setMessages(history);

        if (interventions.length > 0) {
          const newest = interventions[interventions.length - 1];
          lastInterventionIdRef.current = newest.interventionId;
        }

        setLastUpdate(new Date());
      } catch {
        showToast("데이터를 불러오는 데 실패했습니다.");
      } finally {
        setLoading(false);
      }
    }
    init();
  }, [showToast]);

  // Polling
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [logs, interventions] = await Promise.all([
          getRecentEmotionLogs(USER_ID, 1).catch(() => []),
          getInterventions(USER_ID, 1).catch(() => []),
        ]);

        if (logs.length > 0) setLatestLog(logs[0]);

        if (interventions.length > 0) {
          const latest = interventions[0];
          if (
            lastInterventionIdRef.current !== null &&
            latest.interventionId !== lastInterventionIdRef.current
          ) {
            const newMsg: ChatMessage = {
              interventionId: latest.interventionId,
              message: latest.agentMessage,
              timestamp: latest.createdAt,
              feedbackGiven: false,
            };
            setMessages((prev) => [...prev.slice(-MAX_BUBBLES + 1), newMsg]);
          }
          lastInterventionIdRef.current = latest.interventionId;
        }

        setLastUpdate(new Date());
      } catch {
        // Silently fail on polling errors
      }
    }, POLL_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  // 실시간 상태: 감지 프로그램의 상태 통로를 1초마다(답이 오는 중에는 0.2초마다) 읽는다.
  // 연결이 안 되면 저장된 최근 기록을 보여준다.
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      let busy = false;
      try {
        const s = await getLiveState();
        if (!alive) return;
        setLive(s);
        setLiveOk(true);
        setLastUpdate(new Date());
        busy = !!s.chat?.busy;
      } catch {
        if (alive) setLiveOk(false);
      }
      if (alive) timer = setTimeout(tick, busy ? LIVE_INTERVAL_BUSY : LIVE_INTERVAL);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  // 이어지는 대화: 감지 프로그램이 /state의 chat으로 알려 준다 (음성으로 한 말도 여기 나타남)
  const chat = liveOk ? live?.chat ?? null : null;
  const chatActive = !!chat?.active;
  const chatBusy = !!chat?.busy;
  // 첫 메시지는 개입 말풍선(피드백 버튼 있음)으로 이미 보이면 다시 그리지 않는다
  const convo = (chat?.messages ?? []).filter(
    (m) =>
      !(m.source === "intervention" && chat?.interventionId != null &&
        messages.some((b) => b.interventionId === chat.interventionId))
  );
  const convoCount = convo.length;
  const streamingLen = convo.length ? convo[convo.length - 1].text.length : 0;
  const lastStreaming = convo.length > 0 && !!convo[convo.length - 1].streaming;

  // Auto-scroll on new messages
  useEffect(() => {
    // 채팅 칸 안에서만 내린다 (scrollIntoView는 페이지 전체까지 밀어 올려 상단 상태 카드가 잘렸다)
    const box = chatEndRef.current?.parentElement;
    if (box) box.scrollTo({ top: box.scrollHeight, behavior: "smooth" });
  }, [messages, convoCount, chatBusy, streamingLen]);

  async function handleSend(e?: React.FormEvent) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || sending || chatBusy) return;
    setSending(true);
    try {
      const r = await sendChatMessage(text);
      if (r.ok) setInput("");
      else showToast("에이전트가 답하는 중이에요. 잠시 뒤에 보내 주세요.");
    } catch {
      showToast("감지 프로그램(main.py)이 실행 중이어야 대화할 수 있어요.");
    } finally {
      setSending(false);
    }
  }

  async function handleStartOrEnd() {
    try {
      if (chatActive) await endChat();
      else await startChat();
    } catch {
      showToast("감지 프로그램(main.py)이 실행 중이어야 대화할 수 있어요.");
    }
  }

  async function handleReset() {
    try {
      await resetVisitor();
      setMessages([]);   // 앞사람의 개입 말풍선도 화면에서 지운다 (DB 기록은 그대로)
      setInput("");
      showToast("새 관람객 — 잠깐 편한 얼굴로 화면을 봐 주세요");
    } catch {
      showToast("감지 프로그램(main.py)이 실행 중이어야 해요.");
    }
  }

  async function handleIntervene() {
    try {
      const r = await triggerIntervention();
      showToast(r.ok ? "몇 초 안에 에이전트가 말을 걸어요" : "이미 대화 중이에요");
    } catch {
      showToast("감지 프로그램(main.py)이 실행 중이어야 해요.");
    }
  }

  const END_TEXT: Record<string, string> = {
    agent: "에이전트가 대화를 마무리했어요",
    user: "대화를 끝냈어요",
    idle: "한동안 말이 없어 대화를 마쳤어요",
  };

  async function handleFeedback(
    interventionId: number,
    rating: InterventionRating
  ) {
    try {
      await submitInterventionFeedback(interventionId, rating);
      setMessages((prev) =>
        prev.map((m) =>
          m.interventionId === interventionId
            ? { ...m, feedbackGiven: true }
            : m
        )
      );
      showToast("피드백이 전송되었습니다 😊");
    } catch {
      showToast("피드백 전송에 실패했습니다.");
    }
  }

  // 실시간 통로가 연결되면 그 값을, 아니면 백엔드의 최근 기록을 쓴다
  const useLive = liveOk && live !== null;
  const emotion = useLive
    ? live.faceDetected ? normalizeEmotion(live.faceEmotion) : ""
    : latestLog ? logEmotion(latestLog) : "";
  const stressLevel = useLive ? live.stressIndex ?? 0 : latestLog ? logStress(latestLog) : 0;
  const heartRate = useLive ? live.heartRate : latestLog?.heartRate ?? null;
  const restingHr = useLive ? live.restingHeartRate : latestLog?.restingHeartRate ?? null;
  const hrMode = useLive ? live.hrMode : latestLog?.hrMode ?? null;
  const speech = useLive ? live.speechText ?? latestLog?.speechText : latestLog?.speechText;
  const parts: { label: string; value: number | null | undefined; weight: string }[] = [
    { label: "부정 표정", value: useLive ? live.stressNeg : latestLog?.stressNeg, weight: "35%" },
    { label: "심박 상승", value: useLive ? live.stressHr : latestLog?.stressHr, weight: "35%" },
    { label: "표정 각성", value: useLive ? live.stressArousal : latestLog?.stressArousal, weight: "10%" },
    { label: "감정 기복", value: useLive ? live.stressInstab : latestLog?.stressInstab, weight: "20%" },
  ];

  if (loading) {
    return (
      <div className="flex flex-col gap-4 animate-pulse">
        <div className="h-24 rounded-lg bg-muted" />
        <div className="flex-1 space-y-4">
          <div className="h-16 w-3/4 rounded-lg bg-muted" />
          <div className="h-16 w-2/3 rounded-lg bg-muted" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col lg:flex-row h-full min-h-0 gap-4">
      {/* ── 왼쪽: 상태 (얼굴 인식 + 표정·스트레스·심박) ── */}
      <div className="lg:w-[44%] shrink-0 flex flex-col gap-4 min-h-0 lg:overflow-y-auto">
      <FaceCamera connected={liveOk} />

      {/* Status */}
      <Card className="shrink-0">
        <CardContent className="p-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">감지된 표정</p>
              <div className="flex items-center gap-2">
                <span className="text-2xl">
                  {EMOTION_EMOJI[emotion] ?? "❓"}
                </span>
                <span className="font-medium text-sm">
                  {emotion ? EMOTION_LABELS[emotion] : "얼굴 인식 대기"}
                </span>
                {useLive && emotion === "neutral" && live.emotionLean && (
                  <span className="text-xs text-muted-foreground">
                    · {EMOTION_LABELS[live.emotionLean]} 쪽 {EMOTION_EMOJI[live.emotionLean]}
                  </span>
                )}
              </div>
            </div>

            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">스트레스 수준</p>
              <div className="flex items-center gap-2">
                <Progress
                  value={stressLevel * 10}
                  className="flex-1 h-3"
                  style={
                    {
                      "--progress-foreground": stressColor(stressLevel),
                    } as React.CSSProperties
                  }
                />
                <span className="text-sm font-bold w-14 text-right">
                  {stressLevel.toFixed(1)}/10
                </span>
              </div>
            </div>

            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">
                심박 ({useLive && live.hrSource === "watch" ? "워치" : "웹캠"})
              </p>
              <p className="text-sm font-medium">
                {heartRate != null ? `${Math.round(heartRate)} bpm` : "측정 대기"}
                {restingHr != null && (
                  <span className="text-muted-foreground font-normal"> · 평상시 {Math.round(restingHr)}</span>
                )}
              </p>
              <p className="text-xs text-muted-foreground">{hrMode ? HR_MODE_LABELS[hrMode] ?? hrMode : "-"}</p>
              {useLive && live.watchHeartRate != null && (
                <p className="text-xs text-muted-foreground">
                  워치 {Math.round(live.watchHeartRate)} · 웹캠{" "}
                  {live.webcamHeartRate != null ? Math.round(live.webcamHeartRate) : "--"}
                </p>
              )}
            </div>

            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">감지 프로그램</p>
              <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  {liveOk && (
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  )}
                  <span className={`relative inline-flex rounded-full h-2 w-2 ${liveOk ? "bg-green-500" : "bg-gray-400"}`} />
                </span>
                <span className="text-sm font-medium">
                  {liveOk ? "실시간 연결" : "연결 안 됨 · 저장된 기록 표시"}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                {lastUpdate ? `마지막 업데이트 ${format(lastUpdate, "HH:mm:ss")}` : "-"}
              </p>
            </div>
          </div>

          {useLive && live.emotionScores && (
            <EmotionBars scores={live.emotionScores} active={emotion !== "neutral" ? emotion : live.emotionLean ?? null} />
          )}

          <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t pt-4">
            {parts.map((p) => (
              <div key={p.label} className="space-y-1">
                <p className="text-xs text-muted-foreground">
                  {p.label} <span className="opacity-70">({p.weight})</span>
                </p>
                <div className="flex items-center gap-2">
                  <Progress value={(p.value ?? 0) * 100} className="flex-1 h-2" />
                  <span className="text-xs w-10 text-right">{p.value != null ? p.value.toFixed(2) : "모름"}</span>
                </div>
              </div>
            ))}
            <div className="space-y-1 col-span-2">
              <p className="text-xs text-muted-foreground">최근 발화</p>
              <p className="text-sm truncate font-medium">{speech || "-"}</p>
            </div>
          </div>
          {useLive && !live.faceBaselineReady && (
            <p className="mt-3 text-xs text-muted-foreground">
              평소 표정을 기록하는 중입니다(잠깐이면 돼요). 편한 얼굴로 화면을 봐 주세요.
            </p>
          )}
        </CardContent>
      </Card>

      </div>

      {/* ── 오른쪽: 대화 ── */}
      <Card className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between gap-2 border-b px-4 py-2">
          <div className="flex items-center gap-2 text-sm">
            <MessageCircle className="h-4 w-4" />
            {chatActive ? (
              <span className="font-medium">
                대화 중
                <span className="ml-2 text-xs font-normal text-muted-foreground hidden 2xl:inline">
                  말로 대답하거나 아래에 입력하세요
                  {chat?.idleLeftSec != null && !chatBusy && ` · ${chat.idleLeftSec}초 뒤 자동 종료`}
                </span>
              </span>
            ) : (
              <span className="text-muted-foreground">
                {chat?.endReason ? END_TEXT[chat.endReason] : "스트레스가 높아지면 에이전트가 먼저 말을 걸어요"}
              </span>
            )}
          </div>
          <div className="flex gap-2">
          <Button variant="outline" size="sm" className="h-8" disabled={!liveOk} onClick={handleReset}
            title="전시용: 기준선을 다시 잡고 대화를 비워요">
            <UserPlus className="h-3.5 w-3.5" /> 새 관람객
          </Button>
          {!chatActive && (
            <Button variant="outline" size="sm" className="h-8" disabled={!liveOk} onClick={handleIntervene}
              title="시연용: 스트레스 지수와 상관없이 지금 개입 흐름을 실행해요">
              <Zap className="h-3.5 w-3.5" /> 지금 말 걸기
            </Button>
          )}
          <Button variant="outline" size="sm" className="h-8" disabled={!liveOk} onClick={handleStartOrEnd}>
            {chatActive ? (
              <><X className="h-3.5 w-3.5" /> 대화 끝내기</>
            ) : (
              <><MessageCircle className="h-3.5 w-3.5" /> 대화 시작</>
            )}
          </Button>
          </div>
        </div>
        <CardContent className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.length === 0 && convo.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-2">
              <span className="text-4xl">🤖</span>
              <p className="text-sm">에이전트가 말을 걸면 여기에 대화가 표시됩니다</p>
              <Badge variant="outline" className="text-xs">
                5초마다 자동 업데이트
              </Badge>
            </div>
          ) : (
            messages.map((msg) => (
              <ChatBubble
                key={msg.interventionId}
                message={msg.message}
                timestamp={msg.timestamp}
                interventionId={msg.interventionId}
                onFeedback={handleFeedback}
                feedbackGiven={msg.feedbackGiven}
                darkMode={dark}
              />
            ))
          )}
          {convo.map((m) => (
            <ConversationBubble key={`${chat?.sessionId}-${m.id}`} message={m} darkMode={dark} />
          ))}
          {chatBusy && !lastStreaming && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground pl-11">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> 에이전트가 답을 생각하는 중…
            </div>
          )}
          <div ref={chatEndRef} />
        </CardContent>
        <form onSubmit={handleSend} className="flex gap-2 border-t p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={500}
            disabled={!liveOk}
            placeholder={liveOk ? "에이전트에게 하고 싶은 말을 입력하세요" : "감지 프로그램이 연결되면 대화할 수 있어요"}
            className="flex-1 h-10 rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
          />
          <Button type="submit" disabled={!liveOk || !input.trim() || sending || chatBusy}>
            {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            보내기
          </Button>
        </form>
      </Card>

      {/* Toast */}
      {toastMsg && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 animate-in fade-in slide-in-from-bottom-4 duration-300">
          <div className="rounded-lg bg-primary text-primary-foreground px-4 py-3 text-sm font-medium shadow-lg">
            {toastMsg}
          </div>
        </div>
      )}
    </div>
  );
}

// 얼굴 인식 화면: 감지 프로그램이 그린 화면(얼굴 박스·심박·지수)을 MJPEG로 받아 그대로 보여 준다.
function FaceCamera({ connected }: { connected: boolean }) {
  const [show, setShow] = useState(true);
  const [failed, setFailed] = useState(false);
  const [nonce, setNonce] = useState(0);

  // 감지 프로그램이 다시 연결되면 영상도 다시 연결
  useEffect(() => {
    if (connected) {
      setFailed(false);
      setNonce((n) => n + 1);
    }
  }, [connected]);

  const ready = connected && show && !failed;
  return (
    <Card className="shrink-0 flex flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b px-4 py-2">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Camera className="h-4 w-4" /> 얼굴 인식
          <span className="text-xs font-normal text-muted-foreground">저장하지 않아요</span>
        </div>
        <Button variant="outline" size="sm" className="h-8" onClick={() => { setShow((v) => !v); setFailed(false); setNonce((n) => n + 1); }}>
          {show ? <><CameraOff className="h-3.5 w-3.5" /> 숨기기</> : <><Camera className="h-3.5 w-3.5" /> 보이기</>}
        </Button>
      </div>
      <div className="aspect-video bg-black flex items-center justify-center">
        {ready ? (
          <img
            key={nonce}
            src={`${LIVE_VIDEO_URL}?n=${nonce}`}
            alt="얼굴 인식 화면"
            className="w-full h-full object-contain"
            onError={() => setFailed(true)}
          />
        ) : (
          <div className="flex flex-col items-center gap-2 p-6 text-center text-sm text-white/70">
            <CameraOff className="h-8 w-8" />
            {!show
              ? "얼굴 인식 화면을 숨겼어요"
              : !connected
                ? "감지 프로그램이 연결되면 얼굴 인식 화면이 나타나요"
                : (
                  <>
                    얼굴 인식 화면을 불러오지 못했어요
                    <Button variant="secondary" size="sm" className="h-7" onClick={() => { setFailed(false); setNonce((n) => n + 1); }}>
                      다시 연결
                    </Button>
                  </>
                )}
          </div>
        )}
      </div>
    </Card>
  );
}

// 표정 점수 막대: 라벨이 '무감정'이어도 어느 감정 쪽으로 얼마나 움직였는지 바로 보이게
const BAR_KEYS = ["happy", "sad", "angry", "fear", "disgust", "surprise"] as const;
const BAR_FULL = 0.3;   // 평소보다 이만큼 움직이면 막대가 꽉 참 (라벨 기준 약 0.12~0.15의 2배)
const BAR_COLOR: Record<string, string> = {
  happy: "#22c55e", sad: "#3b82f6", angry: "#ef4444", fear: "#a855f7", disgust: "#84cc16", surprise: "#f59e0b",
};

function EmotionBars({ scores, active }: { scores: Record<string, number>; active: string | null }) {
  return (
    <div className="mt-4 border-t pt-4">
      <p className="mb-2 text-xs text-muted-foreground">표정 점수 <span className="opacity-70">(평소 얼굴 대비 움직임)</span></p>
      <div className="grid grid-cols-3 gap-x-4 gap-y-2">
        {BAR_KEYS.map((k) => {
          const v = scores[k] ?? 0;
          const pct = Math.min(100, (v / BAR_FULL) * 100);
          const on = k === active;
          return (
            <div key={k} className="space-y-1">
              <div className={`flex items-center gap-1 text-xs ${on ? "font-semibold" : "text-muted-foreground"}`}>
                <span>{EMOTION_EMOJI[k]}</span>{EMOTION_LABELS[k]}
              </div>
              <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${pct}%`, backgroundColor: BAR_COLOR[k], opacity: on ? 1 : 0.55 }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
