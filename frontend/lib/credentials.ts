import type { Credentials } from "./types";

/**
 * Credentials live in this browser and nowhere else.
 *
 * The API key is sent as a request header on each call and is never persisted
 * server-side, so "signing out" genuinely removes it — there is no server
 * record to revoke. Storage is `localStorage` so a reload keeps you signed in;
 * anyone with access to this browser profile can read it, which is the same
 * trust boundary as a saved password in the browser's own manager.
 */
const STORAGE_KEY = "wiki-chat:credentials";

export function loadCredentials(): Credentials | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Credentials;
    if (!parsed?.email || !parsed?.apiKey) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function saveCredentials(credentials: Credentials): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(credentials));
}

export function clearCredentials(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/**
 * Auth headers for every API call.
 *
 * Two distinct things: the bearer token proves *who you are* (the server
 * verifies its signature, so nothing here can assert an identity), while the
 * API key is what your OpenAI calls are billed to. The key is deliberately not
 * inside the token — a JWT payload is readable by anyone holding it.
 */
export function authHeaders(credentials: Credentials): Record<string, string> {
  const headers: Record<string, string> = {};
  if (credentials.token) headers.Authorization = `Bearer ${credentials.token}`;
  // Absent until the user adds one; endpoints that spend money answer 428.
  if (credentials.apiKey) headers["X-OpenAI-Key"] = credentials.apiKey;
  return headers;
}

/** Never render a full key back to the user; enough to recognise, not to reuse. */
export function maskKey(apiKey: string): string {
  if (apiKey.length < 12) return "sk-…";
  return `${apiKey.slice(0, 7)}…${apiKey.slice(-4)}`;
}
