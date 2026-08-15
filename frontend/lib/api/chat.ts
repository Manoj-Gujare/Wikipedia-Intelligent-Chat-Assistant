import { authHeaders } from "../credentials";
import type { ArticleLink, ChatResponse, Credentials } from "../types";
import { API_URL } from "./client";

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
