"use client";

import type { Message } from "@/lib/types";
import { AnswerText } from "./AnswerText";
import { DisambiguationList } from "./DisambiguationList";

export function MessageBubble({
  message,
  onPickDisambiguation,
  onShowSources,
  isActive,
  busy,
}: {
  message: Message;
  onPickDisambiguation: (title: string) => void;
  onShowSources: (messageId: string) => void;
  isActive: boolean;
  busy: boolean;
}) {
  if (message.role === "user") {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user">{message.content}</div>
      </div>
    );
  }

  const empty = message.content.length === 0;
  const sourceCount = message.sources?.length ?? 0;
  const articleCount = message.articles?.length ?? 0;
  const elapsed = message.timings?.total_ms ?? 0;
  // One compact line instead of an inline citation block: enough to know
  // evidence exists, without breaking the flow of the conversation.
  const showFooter = !message.streaming && (sourceCount > 0 || articleCount > 0);
  // A conversational reply cites nothing, so it gets the timing on its own —
  // as text, not a button, because there is no source panel to open. Without
  // this the answers that skip retrieval were the only ones with no visible
  // cost, which read as though they had not been timed.
  const showTimeOnly = !message.streaming && !showFooter && elapsed > 0;

  return (
    <div className="row row-assistant">
      <div className="avatar" aria-hidden="true">
        W
      </div>

      <div className={`bubble bubble-assistant${message.error ? " bubble-error" : ""}`}>
        {message.resolvedQuery && (
          <p className="resolved-query" title="History-aware query used for retrieval">
            searching for “{message.resolvedQuery}”
          </p>
        )}

        {/* Shown on every turn. This used to be suppressed for chitchat, back
            when a greeting was answered from a lookup table in ~16ms and the
            indicator would flash and vanish. A conversational reply now costs a
            model call like any other answer, so suppressing it left the bubble
            blank for about a second. */}
        {empty && message.streaming ? (
          <div className="typing" aria-label="Thinking">
            <span />
            <span />
            <span />
          </div>
        ) : (
          <AnswerText text={message.content} sources={message.sources ?? []} />
        )}

        {/* Disambiguation stays inline: it is a question back to the user, not
            supporting evidence. */}
        {message.disambiguation && message.disambiguation.length > 0 && (
          <DisambiguationList
            options={message.disambiguation}
            onPick={onPickDisambiguation}
            disabled={busy}
          />
        )}

        {showFooter && (
          <button
            className={`answer-footer${isActive ? " active" : ""}`}
            onClick={() => onShowSources(message.id)}
          >
            {sourceCount > 0
              ? `${sourceCount} cited ${sourceCount === 1 ? "source" : "sources"}`
              : message.usedLiveSearch
                ? `${articleCount} suggested ${articleCount === 1 ? "article" : "articles"}`
                : `${articleCount} related ${articleCount === 1 ? "article" : "articles"}`}
            {elapsed > 0 && (
              <span className="answer-footer-time">{(elapsed / 1000).toFixed(2)}s</span>
            )}
          </button>
        )}

        {showTimeOnly && (
          <p className="answer-footer answer-footer-plain">
            <span className="answer-footer-time">{(elapsed / 1000).toFixed(2)}s</span>
          </p>
        )}
      </div>
    </div>
  );
}
