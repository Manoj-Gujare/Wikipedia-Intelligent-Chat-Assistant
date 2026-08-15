import type {
  ArticleLink,
  ConversationSummary,
  Credentials,
  Source,
} from "../types";
import { request } from "./client";

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
