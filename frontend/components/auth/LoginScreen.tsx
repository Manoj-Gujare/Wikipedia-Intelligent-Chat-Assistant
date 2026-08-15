"use client";

import { FormEvent, useState } from "react";
import { login, register } from "@/lib/api";
import { saveCredentials } from "@/lib/credentials";
import type { Credentials, Session } from "@/lib/types";

export function LoginScreen({
  onSignedIn,
}: {
  onSignedIn: (credentials: Credentials, session: Session) => void;
}) {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const signingUp = mode === "signup";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (busy) return;

    setBusy(true);
    setError(null);

    try {
      const action = signingUp ? register : login;
      const { credentials, session } = await action(email.trim(), password);
      // The API key is supplied later, from the header — it is a runtime
      // credential, not part of signing in.
      saveCredentials(credentials);
      onSignedIn(credentials, session);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="login-card" onSubmit={submit}>
        <div className="brand-mark login-mark" aria-hidden="true">
          W
        </div>
        <h1>{signingUp ? "Create an account" : "Wikipedia Assistant"}</h1>
        <p className="login-sub">
          {signingUp
            ? "Your chats and knowledge base stay private to your account."
            : "Answers grounded in Wikipedia, cited down to the section."}
        </p>

        <label className="field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            autoComplete="email"
            required
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={signingUp ? "At least 8 characters" : "Your password"}
            autoComplete={signingUp ? "new-password" : "current-password"}
            minLength={signingUp ? 8 : undefined}
            required
          />
        </label>

        {error && <p className="login-error">{error}</p>}

        <button className="send login-submit" type="submit" disabled={busy}>
          {busy ? "Please wait…" : signingUp ? "Create account" : "Sign in"}
        </button>

        <button
          type="button"
          className="link-button"
          onClick={() => {
            setMode(signingUp ? "signin" : "signup");
            setError(null);
          }}
        >
          {signingUp
            ? "Already have an account? Sign in"
            : "New here? Create an account"}
        </button>

        <p className="login-note">
          You&apos;ll add your OpenAI API key after signing in. It stays in this
          browser, is sent only to bill your own usage, and is never stored on
          the server or placed in your session token.
        </p>
      </form>
    </div>
  );
}
