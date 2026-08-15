"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchConversation, streamChat } from "@/lib/api";
import type { Credentials, Message } from "@/lib/types";

function newId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

/**
 * The active thread: its messages, the request in flight, and the language it
 * is being held in.
 *
 * The conversation id is a ref rather than state because it must be readable
 * synchronously inside `send` — a second turn fired before React re-rendered
 * would otherwise start a new thread on the server.
 */
export function useConversation({
  credentials,
  onFirstTurn,
}: {
  credentials: Credentials;
  onFirstTurn: () => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [lang, setLang] = useState("en");
  const [busy, setBusy] = useState(false);

  const conversationId = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || busy || !credentials.apiKey) return;

      const assistantId = newId();
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "user", content: question },
        { id: assistantId, role: "assistant", content: "", streaming: true },
      ]);
      setInput("");
      setBusy(true);

      const patch = (updater: (message: Message) => Message) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? updater(m) : m)),
        );

      const controller = new AbortController();
      abortRef.current = controller;
      const isFirstTurn = conversationId.current === null;

      await streamChat(
        credentials,
        { message: question, session_id: conversationId.current, lang },
        {
          onMeta: (id) => {
            conversationId.current = id;
          },
          onIntent: (intent) => patch((m) => ({ ...m, intent })),
          onRewrite: (resolvedQuery) => patch((m) => ({ ...m, resolvedQuery })),
          onArticles: (articles) => patch((m) => ({ ...m, articles })),
          onToken: (token) => patch((m) => ({ ...m, content: m.content + token })),
          onDone: (response) =>
            patch((m) => ({
              ...m,
              content: response.answer,
              sources: response.sources,
              articles: response.articles,
              disambiguation: response.disambiguation,
              disambiguationTerm: response.disambiguation_term,
              resolvedQuery: response.resolved_query,
              timings: response.timings,
              usedLiveSearch: response.used_live_search,
              streaming: false,
            })),
          onError: (detail) =>
            patch((m) => ({
              ...m,
              content: m.content || detail,
              streaming: false,
              error: true,
            })),
        },
        controller.signal,
      );

      // Guarantees the bubble leaves its streaming state even if `done` never
      // arrived (aborted request, dropped connection).
      patch((m) => (m.streaming ? { ...m, streaming: false } : m));
      abortRef.current = null;
      setBusy(false);
      textareaRef.current?.focus();
      // A new thread needs to appear in the sidebar; later turns only move it.
      if (isFirstTurn) onFirstTurn();
    },
    [busy, credentials, lang, onFirstTurn],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const startNewChat = useCallback(() => {
    abortRef.current?.abort();
    conversationId.current = null;
    setMessages([]);
    setBusy(false);
  }, []);

  const openConversation = useCallback(
    async (id: string) => {
      abortRef.current?.abort();
      const restored = await fetchConversation(credentials, id);
      if (!restored) return;
      conversationId.current = id;
      setLang(restored.lang);
      setMessages(
        restored.turns.map((turn) => ({
          id: newId(),
          role: turn.role,
          content: turn.content,
          sources: turn.meta?.sources,
          articles: turn.meta?.articles,
        })),
      );
    },
    [credentials],
  );

  const changeLanguage = useCallback(
    (code: string) => {
      setLang(code);
      // Retrieval is language-scoped, so mixing editions inside one thread would
      // produce citations the user cannot follow. Start a clean thread instead.
      if (messages.length > 0) startNewChat();
    },
    [messages.length, startNewChat],
  );

  return {
    messages,
    input,
    setInput,
    busy,
    lang,
    changeLanguage,
    send,
    stop,
    startNewChat,
    openConversation,
    activeConversationId: conversationId.current,
    conversationIdRef: conversationId,
    bottomRef,
    textareaRef,
  };
}
