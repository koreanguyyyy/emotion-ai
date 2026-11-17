export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message: string | null;
}

export interface UserInfo {
  userId: number;
  loginId: string;
  email: string;
  nickname: string;
  address: string | null;
  addressDetail: string | null;
  zipcode: string | null;
  createdAt: string;
}

export interface UserPreference {
  preferenceId: number;
  userId: number;
  preferredMusicGenres: string[];
  preferredMovieGenres: string[];
  preferredBookGenres: string[];
  preferredPlaceTypes: string[];
  ttsEnabled: boolean;
  cooldownMinutes: number;
}

export interface EmotionLog {
  emotionLogId: number;
  userId: number;
  primaryEmotion: string;
  confidenceScore: number;
  faceEmotion: string | null;
  faceConfidence: number | null;
  speechText: string | null;
  speechSentiment: string | null;
  vadStressScore: number | null;
  finalStressLevel: number;
  sighDetected: boolean;
  motionPattern: string | null;
  interventionTriggered: boolean;
  detectedAt: string;
  // 2026-11-03 추가 (fusion 감지 프로그램). 1학기 기록에는 없으므로 모두 선택 항목
  heartRate?: number | null;
  restingHeartRate?: number | null;
  stressNeg?: number | null;
  stressHr?: number | null;
  stressArousal?: number | null;
  stressInstab?: number | null;
  hrMode?: string | null;
  faceMethod?: string | null;
  modelEmotion?: string | null;
  logType?: string | null;
}

// 감지 프로그램의 실시간 상태 통로(http://127.0.0.1:8765/state) 응답
// 이어지는 대화 (감지 프로그램의 /state 안 chat)
export interface LiveChatLink {
  category: string;
  title: string;
  artist: string | null;
  url: string | null;
}

export interface LiveChatMessage {
  id: number;
  role: "agent" | "user";
  text: string;
  source: "intervention" | "agent" | "voice" | "web";
  time: number;
  links: LiveChatLink[];
  streaming?: boolean;   // 답이 아직 오는 중
}

export interface LiveChat {
  active: boolean;
  busy: boolean;
  sessionId: string | null;
  interventionId: number | null;
  endReason: "agent" | "user" | "idle" | null;
  idleLeftSec: number | null;
  messages: LiveChatMessage[];
}

export interface LiveState {
  chat?: LiveChat | null;
  time: number;
  faceDetected: boolean;
  faceEmotion: string;
  emotionKey: string;
  emotionScores?: Record<string, number> | null;   // 감정별 점수(평소 대비) — 표정 막대
  emotionLean?: string | null;                     // 무감정이어도 기울어진 감정 (영문 키)
  faceConfidence: number | null;
  faceBaselineReady: boolean;
  faceMethod: string | null;
  modelEmotion: string | null;
  valence: number | null;
  arousal: number | null;
  heartRate: number | null;
  heartRateQuality: number | null;
  restingHeartRate: number | null;
  hrMode: string | null;
  hrSource?: "webcam" | "watch" | null;     // 지금 심박의 출처
  watchHeartRate?: number | null;           // 갤럭시 워치 심박
  webcamHeartRate?: number | null;          // 웹캠 심박 (비교용)
  stressIndex: number | null;
  stressNeg: number | null;
  stressHr: number | null;
  stressArousal: number | null;
  stressInstab: number | null;
  agentStressLevel: number | null;
  speechText: string | null;
  lastInterventionId: number | null;
}

export const HR_MODE_LABELS: Record<string, string> = {
  fusion: "표정 + 심박",
  face_only: "표정만 (심박 미측정)",
  face_only_gap: "표정만 (심박 일시 끊김)",
  calibrating: "평상시 심박 보정 중",
};

export type InterventionRating = "HELPFUL" | "NEUTRAL" | "UNNECESSARY" | "ANNOYING";
export type EmotionAccuracy = string;
export type RecommendationCategory = "MUSIC" | "MOVIE" | "BOOK" | "PLACE";
export type UserFeedback = "LIKED" | "DISLIKED" | "SKIPPED";
export type RelevanceRating = string;

export interface Intervention {
  interventionId: number;
  userId: number;
  emotionLogId: number;
  interventionType: string;
  agentMessage: string;
  ttsPlayed: boolean;
  cooldownExpiresAt: string | null;
  interventionRating: InterventionRating | null;
  emotionAccuracy: EmotionAccuracy | null;
  interventionFeedbackText: string | null;
  feedbackAt: string | null;
  createdAt: string;
  recommendations: Recommendation[];
}

export interface Recommendation {
  recommendationId: number;
  interventionId: number;
  category: RecommendationCategory;
  title: string;
  artistOrAuthor: string | null;
  externalId: string | null;
  externalUrl: string | null;
  thumbnailUrl: string | null;
  reason: string | null;
  userFeedback: UserFeedback | null;
  played: boolean;
  relevanceRating: RelevanceRating | null;
  feedbackText: string | null;
  createdAt: string;
}

export type EmotionType = "happy" | "sad" | "angry" | "fear" | "disgust" | "surprise" | "neutral";

export const EMOTION_COLORS: Record<string, string> = {
  happy: "#FFD700",
  sad: "#4A90E2",
  angry: "#E74C3C",
  fear: "#9B59B6",
  disgust: "#8B4513",
  surprise: "#FF8C00",
  neutral: "#95A5A6",
};

export const EMOTION_LABELS: Record<string, string> = {
  happy: "행복",
  sad: "슬픔",
  angry: "분노",
  fear: "공포",
  disgust: "혐오",
  surprise: "놀람",
  neutral: "중립",
};

// ── 감정 이름 통일 (2026-11-03) ─────────────────────────────────────
// 감지 프로그램(1학기 DeepFace 버전·2학기 fusion 버전)은 감정을 한국어("슬픔", "무감정")로 저장하고,
// 이 웹페이지의 색·이모지·이름 표는 영어 키(sad, neutral)를 쓴다. 키가 맞지 않으면 모니터링 화면이
// "대기 중"으로 나오고 그래프 색이 회색이 된다. 모든 화면은 이 함수로 감정 키를 정리해서 쓴다.
const KO_TO_EN: Record<string, EmotionType> = {
  행복: "happy",
  슬픔: "sad",
  분노: "angry",
  공포: "fear",
  혐오: "disgust",
  놀람: "surprise",
  무감정: "neutral",
  중립: "neutral",
};

export function normalizeEmotion(raw?: string | null): EmotionType {
  if (!raw) return "neutral";
  const s = raw.trim();
  if (s in KO_TO_EN) return KO_TO_EN[s];
  const lower = s.toLowerCase();
  return (lower in EMOTION_LABELS ? lower : "neutral") as EmotionType;
}

export function logEmotion(log: EmotionLog): EmotionType {
  return normalizeEmotion(log.primaryEmotion ?? log.faceEmotion);
}

// 스트레스 값: 최종 판단(finalStressLevel)이 비어 있으면 감지 프로그램의 지수(vadStressScore)를 쓴다.
// 예전 기록은 최종 값을 비운 채 저장돼서, 이 대비 없이는 0으로 보였다.
export function logStress(log: EmotionLog): number {
  if (log.finalStressLevel != null) return log.finalStressLevel;
  if (log.vadStressScore != null) return Math.round(log.vadStressScore * 10) / 10;
  return 0;
}
