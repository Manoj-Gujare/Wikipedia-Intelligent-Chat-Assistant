import type { Credentials, LoginResponse, Session } from "../types";
import { API_URL, request, throwApiError } from "./client";

/**
 * Sign in or sign up. Deliberately does not go through `request`: there are no
 * credentials to send yet, which is the whole point of these two endpoints.
 */
async function authenticate(
  path: string,
  email: string,
  password: string,
  failureMessage: string,
): Promise<{ credentials: Credentials; session: LoginResponse }> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) await throwApiError(response, failureMessage);

  const session = (await response.json()) as LoginResponse;
  // No API key yet — it is supplied separately, after signing in.
  return { credentials: { email, apiKey: "", token: session.access_token }, session };
}

/** Sign in with email and password. The API key is not involved. */
export function login(email: string, password: string) {
  return authenticate("/api/auth/login", email, password, "Sign-in failed");
}

export function register(email: string, password: string) {
  return authenticate("/api/auth/register", email, password, "Sign-up failed");
}

/** Confirm a key works before storing it — only OpenAI can tell us that. */
export function verifyKey(credentials: Credentials): Promise<{ valid: boolean }> {
  return request<{ valid: boolean }>("/api/auth/verify-key", credentials, {
    method: "POST",
  });
}

export function startSession(credentials: Credentials): Promise<Session> {
  return request<Session>("/api/session", credentials, { method: "POST" });
}
