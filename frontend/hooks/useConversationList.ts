"use client";

import { useCallback, useEffect, useState } from "react";
import { deleteConversation, listConversations } from "@/lib/api";
import type { ConversationSummary, Credentials } from "@/lib/types";

/**
 * The sidebar's thread list.
 *
 * Every failure here is swallowed on purpose: the list is navigation, not the
 * conversation, and a failed refresh must never take the chat down with it.
 */
export function useConversationList(credentials: Credentials) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);

  const refresh = useCallback(async () => {
    try {
      setConversations(await listConversations(credentials));
    } catch {
      // Sidebar is non-essential; a failure here must not break the chat.
    }
  }, [credentials]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const remove = useCallback(
    async (id: string) => {
      await deleteConversation(credentials, id);
      void refresh();
    },
    [credentials, refresh],
  );

  return { conversations, refresh, remove };
}
