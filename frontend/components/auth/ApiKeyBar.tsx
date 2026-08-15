"use client";

import { FormEvent, useState } from "react";
import { verifyKey } from "@/lib/api";
import { maskKey } from "@/lib/credentials";
import type { Credentials } from "@/lib/types";

/**
 * The API key control that sits at the top of the app.
 *
 * The key is a runtime credential, not an identity: you are signed in without
 * it, but questions cost money so they need one. Keeping it here (rather than
 * on the login form) means it can be added, replaced or removed at any time
 * without touching the account.
 */
export function ApiKeyBar({
  credentials,
  onSave,
  onClear,
}: {
  credentials: Credentials;
  onSave: (apiKey: string) => void;
  onClear: () => void;
}) {
  const hasKey = Boolean(credentials.apiKey);
  const [editing, setEditing] = useState(!hasKey);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const apiKey = draft.trim();
    if (!apiKey || busy) return;

    setBusy(true);
    setError(null);
    try {
      // Shape checks cannot tell a revoked key from a live one — only OpenAI
      // can, so confirm before storing and letting the user ask a question.
      await verifyKey({ ...credentials, apiKey });
      onSave(apiKey);
      setDraft("");
      setEditing(false);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!editing && hasKey) {
    return (
      <div className="keybar keybar-set">
        <span className="keybar-label">OpenAI key</span>
        <code className="keybar-value">{maskKey(credentials.apiKey)}</code>
        <button className="link-button" onClick={() => setEditing(true)}>
          Replace
        </button>
        <button className="link-button" onClick={onClear}>
          Remove
        </button>
      </div>
    );
  }

  return (
    <form className="keybar" onSubmit={submit}>
      <label className="keybar-label" htmlFor="openai-key">
        {hasKey ? "New OpenAI key" : "Add your OpenAI API key to start asking"}
      </label>
      <input
        id="openai-key"
        type="password"
        className="keybar-input"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder="sk-…"
        autoComplete="off"
        spellCheck={false}
      />
      <button className="send keybar-save" type="submit" disabled={busy || !draft.trim()}>
        {busy ? "Checking…" : "Save"}
      </button>
      {hasKey && (
        <button
          type="button"
          className="link-button"
          onClick={() => {
            setEditing(false);
            setDraft("");
            setError(null);
          }}
        >
          Cancel
        </button>
      )}
      {error && <span className="keybar-error">{error}</span>}
    </form>
  );
}
