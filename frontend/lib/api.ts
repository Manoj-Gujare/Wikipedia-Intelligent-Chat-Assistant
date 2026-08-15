import { authHeaders } from "./credentials";
import type {
  ArticleLink,
  ChatResponse,
  ConversationSummary,
  Credentials,
  IngestJob,
  KnowledgeBase,
  LoginResponse,
  Session,
  Source,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(
  path: string,
  credentials: Credentials,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(credentials),
      ...(init.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // Non-JSON error body; the status-based message stands.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

// ------------------------------------------------------------------- session

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

  if (!response.ok) {
    let detail = `${failureMessage} (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // Non-JSON error body; the status-based message stands.
    }
    throw new ApiError(detail, response.status);
  }

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

// ---------------------------------------------------------------------- chat

export interface StreamHandlers {
  onMeta?: (conversationId: string) => void;
  /** Which path the graph took: chitchat answers skip retrieval entirely. */
  onIntent?: (intent: string) => void;
  onRewrite?: (resolvedQuery: string) => void;
  onRetrieval?: (info: { chunks: number }) => void;
  onArticles?: (articles: ArticleLink[]) => void;
  onToken?: (text: string) => void;
  onDone?: (response: ChatResponse) => void;
  onError?: (message: string) => void;
}

/**
 * Streams a chat answer over SSE.
 *
 * `EventSource` cannot POST or send auth headers, so we read the response body
 * directly and parse the SSE framing ourselves. Events arrive as
 * `event: <name>\ndata: <json>` separated by blank lines; partial frames are
 * buffered until complete.
 */
export async function streamChat(
  credentials: Credentials,
  body: { message: string; session_id: string | null; lang: string },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(credentials) },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") return;
    handlers.onError?.(`Could not reach the backend at ${API_URL}. Is it running?`);
    return;
  }

  if (response.status === 401) {
    handlers.onError?.("Your credentials were rejected. Sign in again.");
    return;
  }
  if (!response.ok || !response.body) {
    handlers.onError?.(`Backend returned ${response.status}.`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let separator = buffer.indexOf("\n\n");
      while (separator !== -1) {
        dispatch(buffer.slice(0, separator), handlers);
        buffer = buffer.slice(separator + 2);
        separator = buffer.indexOf("\n\n");
      }
    }
  } catch (error) {
    if ((error as Error).name !== "AbortError") {
      handlers.onError?.("The connection dropped while streaming the answer.");
    }
  }
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }

  if (dataLines.length === 0) return;

  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return;
  }

  switch (event) {
    case "meta":
      handlers.onMeta?.(payload.conversation_id as string);
      break;
    case "intent":
      handlers.onIntent?.(payload.intent as string);
      break;
    case "rewrite":
      handlers.onRewrite?.(payload.resolved_query as string);
      break;
    case "retrieval":
      handlers.onRetrieval?.(payload as unknown as { chunks: number });
      break;
    case "articles":
      handlers.onArticles?.(payload.articles as ArticleLink[]);
      break;
    case "token":
      handlers.onToken?.(payload.text as string);
      break;
    case "done":
      handlers.onDone?.(payload as unknown as ChatResponse);
      break;
    case "error":
      handlers.onError?.((payload.detail as string) ?? "Unknown error.");
      break;
  }
}

// ------------------------------------------------------------- conversations

export function listConversations(
  credentials: Credentials,
): Promise<ConversationSummary[]> {
  return request<ConversationSummary[]>("/api/conversations", credentials);
}

export interface RestoredTurn {
  role: "user" | "assistant";
  content: string;
  meta: { sources?: Source[]; articles?: ArticleLink[] } | null;
}

export interface RestoredConversation {
  conversation_id: string;
  lang: string;
  turns: RestoredTurn[];
}

export async function fetchConversation(
  credentials: Credentials,
  conversationId: string,
): Promise<RestoredConversation | null> {
  try {
    return await request<RestoredConversation>(
      `/api/conversations/${conversationId}`,
      credentials,
    );
  } catch {
    return null;
  }
}

export async function deleteConversation(
  credentials: Credentials,
  conversationId: string,
): Promise<void> {
  try {
    await request(`/api/conversations/${conversationId}`, credentials, {
      method: "DELETE",
    });
  } catch {
    // Best effort: the server evicts conversations on TTL anyway.
  }
}

// ------------------------------------------------------------ knowledge base

export function fetchKnowledgeBase(credentials: Credentials): Promise<KnowledgeBase> {
  return request<KnowledgeBase>("/api/kb", credentials);
}

export function startIngest(
  credentials: Credentials,
  topics: string[],
  articlesPerTopic: number,
  lang: string,
): Promise<IngestJob> {
  return request<IngestJob>("/api/kb/ingest", credentials, {
    method: "POST",
    body: JSON.stringify({
      topics,
      articles_per_topic: articlesPerTopic,
      lang,
    }),
  });
}

/** Add specific articles by exact title — the "that wasn't indexed" path. */
export function addArticles(
  credentials: Credentials,
  titles: string[],
  lang: string,
): Promise<IngestJob> {
  return request<IngestJob>("/api/kb/articles", credentials, {
    method: "POST",
    body: JSON.stringify({ titles, lang }),
  });
}

export function fetchJob(credentials: Credentials, jobId: string): Promise<IngestJob> {
  return request<IngestJob>(`/api/kb/jobs/${jobId}`, credentials);
}

export function cancelJob(
  credentials: Credentials,
  jobId: string,
): Promise<{ cancelled: boolean }> {
  return request(`/api/kb/jobs/${jobId}/cancel`, credentials, { method: "POST" });
}

export function resetKnowledgeBase(
  credentials: Credentials,
): Promise<{ collections_removed: number }> {
  return request("/api/kb", credentials, { method: "DELETE" });
}
