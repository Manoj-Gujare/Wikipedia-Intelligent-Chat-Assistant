"use client";

import type { ReactNode, RefObject } from "react";
import type { Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";

export function MessageList({
  messages,
  busy,
  focusedId,
  panelOpen,
  emptyState,
  bottomRef,
  onPickDisambiguation,
  onShowSources,
}: {
  messages: Message[];
  busy: boolean;
  focusedId: string | null;
  panelOpen: boolean;
  emptyState: ReactNode;
  bottomRef: RefObject<HTMLDivElement | null>;
  onPickDisambiguation: (title: string) => void;
  onShowSources: (id: string) => void;
}) {
  return (
    <main className="messages">
      {messages.length === 0
        ? emptyState
        : messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              busy={busy}
              isActive={message.id === focusedId && panelOpen}
              onPickDisambiguation={onPickDisambiguation}
              onShowSources={onShowSources}
            />
          ))}
      <div ref={bottomRef} />
    </main>
  );
}
