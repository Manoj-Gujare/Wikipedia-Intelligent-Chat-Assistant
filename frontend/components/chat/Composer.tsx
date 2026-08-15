"use client";

import type { FormEvent, KeyboardEvent, RefObject } from "react";

export function Composer({
  input,
  busy,
  hasApiKey,
  textareaRef,
  onChange,
  onSubmit,
  onStop,
}: {
  input: string;
  busy: boolean;
  hasApiKey: boolean;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
}) {
  const submit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={textareaRef}
        value={input}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder={
          hasApiKey
            ? "Ask a question…"
            : "Add your OpenAI API key above to start asking"
        }
        rows={1}
        disabled={busy || !hasApiKey}
      />
      {busy ? (
        <button type="button" className="send stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" className="send" disabled={!input.trim()}>
          Send
        </button>
      )}
    </form>
  );
}
