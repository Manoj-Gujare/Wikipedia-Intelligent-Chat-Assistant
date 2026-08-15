"use client";

import { useEffect, useState } from "react";
import { ChatScreen } from "@/components/chat/ChatScreen";
import { LoginScreen } from "@/components/auth/LoginScreen";
import { startSession } from "@/lib/api";
import { clearCredentials, loadCredentials } from "@/lib/credentials";
import type { Credentials, Session } from "@/lib/types";

export default function Home() {
  const [credentials, setCredentials] = useState<Credentials | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [restoring, setRestoring] = useState(true);

  // Credentials live only in this browser, so a reload has to re-establish the
  // session from local storage rather than from a server-side cookie. A stored
  // token may have expired (or been signed with a secret that changed when the
  // server restarted), so fall back to a fresh sign-in with the saved key
  // before giving up and sending the user back to the login screen.
  useEffect(() => {
    const saved = loadCredentials();
    if (!saved) {
      setRestoring(false);
      return;
    }

    void (async () => {
      try {
        const restored = await startSession(saved);
        setCredentials(saved);
        setSession(restored);
      } catch {
        // Token expired or was signed with a secret that changed on restart.
        // There is no stored password to renew with, so sign in again.
        clearCredentials();
      } finally {
        setRestoring(false);
      }
    })();
  }, []);

  if (restoring) {
    return (
      <div className="login">
        <div className="typing" aria-label="Loading">
          <span />
          <span />
          <span />
        </div>
      </div>
    );
  }

  if (!credentials || !session) {
    return (
      <LoginScreen
        onSignedIn={(next, nextSession) => {
          setCredentials(next);
          setSession(nextSession);
        }}
      />
    );
  }

  return (
    <ChatScreen
      credentials={credentials}
      session={session}
      onCredentialsChange={setCredentials}
      onSignOut={() => {
        setCredentials(null);
        setSession(null);
      }}
    />
  );
}
