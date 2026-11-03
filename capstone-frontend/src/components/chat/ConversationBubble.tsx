import { Bot, User, Mic, Music, BookOpen, MapPin, ExternalLink } from "lucide-react";
import SpotifyPlayer, { extractSpotifyTrackIds } from "./SpotifyPlayer";
import { format } from "date-fns";
import type { LiveChatMessage } from "@/types";

const LINK_ICON: Record<string, typeof Music> = { MUSIC: Music, BOOK: BookOpen, PLACE: MapPin };

interface Props {
  message: LiveChatMessage;
  darkMode: boolean;
}

// 이어지는 대화의 말풍선. 에이전트는 왼쪽, 사용자는 오른쪽.
export default function ConversationBubble({ message, darkMode }: Props) {
  const isUser = message.role === "user";
  const time = format(new Date(message.time * 1000), "HH:mm");

  if (isUser) {
    return (
      <div className="flex gap-3 items-start justify-end">
        <div className="max-w-[75%] space-y-1">
          <div className="rounded-2xl rounded-tr-sm bg-primary text-primary-foreground px-4 py-2.5">
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
          </div>
          <p className="text-xs text-muted-foreground text-right flex items-center justify-end gap-1">
            {message.source === "voice" && <Mic className="h-3 w-3" />}
            {message.source === "voice" ? "음성" : "채팅"} · {time}
          </p>
        </div>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary">
          <User className="h-4 w-4" />
        </div>
      </div>
    );
  }

  const spotifyIds = message.links.flatMap((l) => (l.url ? extractSpotifyTrackIds(l.url) : []));
  return (
    <div className="flex gap-3 items-start">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex-1 space-y-1 max-w-[85%]">
        <div className="rounded-2xl rounded-tl-sm bg-muted px-4 py-2.5">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.text}
            {message.streaming && (
              <span className="ml-0.5 inline-block h-4 w-1.5 -mb-0.5 animate-pulse rounded-sm bg-foreground/60 align-middle" />
            )}
          </p>
          {message.links.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {message.links.map((l) => {
                const Icon = LINK_ICON[l.category] ?? ExternalLink;
                return (
                  <a
                    key={`${l.title}-${l.url}`}
                    href={l.url ?? undefined}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-full border bg-background px-3 py-1 text-xs hover:bg-accent"
                  >
                    <Icon className="h-3.5 w-3.5" />
                    <span className="font-medium">{l.title}</span>
                    {l.artist && <span className="text-muted-foreground">· {l.artist}</span>}
                  </a>
                );
              })}
            </div>
          )}
          {spotifyIds.map((id) => (
            <SpotifyPlayer key={id} trackId={id} darkMode={darkMode} />
          ))}
        </div>
        <p className="text-xs text-muted-foreground">{time}</p>
      </div>
    </div>
  );
}
