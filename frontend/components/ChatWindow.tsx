"use client";

import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  addArticles,
  deleteConversation,
  fetchConversation,
  fetchJob,
  fetchKnowledgeBase,
  listConversations,
  streamChat,
} from "@/lib/api";
import { clearCredentials, saveCredentials } from "@/lib/credentials";
import {
  LANGUAGES,
  type ConversationSummary,
  type Credentials,
  type Message,
  type Session,
} from "@/lib/types";
import { ApiKeyBar } from "./ApiKeyBar";
import { KnowledgeBasePanel } from "./KnowledgeBasePanel";
import { MessageBubble } from "./MessageBubble";
import { Sidebar } from "./Sidebar";
import { SourcePanel } from "./SourcePanel";

const SUGGESTIONS = [
  "What is the event horizon of a black hole?",
  "How does photosynthesis work?",
  "Who was Alan Turing and why does he matter?",
  "Tell me about Mercury",
];

function newId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function ChatWindow({
  credentials,
  session,
  onSignOut,
  onCredentialsChange,
}: {
  credentials: Credentials;
  session: Session;
  onSignOut: () => void;
  onCredentialsChange: (credentials: Credentials) => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [input, setInput] = useState("");
  const [lang, setLang] = useState("en");
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [kbOpen, setKbOpen] = useState(false);
  const [personalArticles, setPersonalArticles] = useState(session.personal_articles);
  const [addingTitle, setAddingTitle] = useState<string | null>(null);
  const [addedTitles, setAddedTitles] = useState<string[]>([]);
  const [panelOpen, setPanelOpen] = useState(false);
  const [focusedId, setFocusedId] = useState<string | null>(null);

  const conversationId = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await listConversations(credentials));
    } catch {
      // Sidebar is non-essential; a failure here must not break the chat.
    }
  }, [credentials]);

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const openConversation = useCallback(
    async (id: string) => {
      abortRef.current?.abort();
      setSidebarOpen(false);
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
      // Unpin, so the panel tracks the answer being written rather than an
      // older one the user happened to inspect.
      setFocusedId(null);

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
      if (isFirstTurn) void refreshConversations();
    },
    [busy, credentials, lang, refreshConversations],
  );

  /**
   * Add an article the index didn't have, then re-ask the question that
   * failed. Closes the loop the user is actually in: "you clearly know this
   * page exists, so use it."
   */
  const addArticle = useCallback(
    async (title: string) => {
      if (addingTitle) return;
      setAddingTitle(title);

      // The question that prompted this answer is the user turn before it.
      const index = messages.findIndex((m) => m.articles?.some((a) => a.title === title));
      const question = index > 0 ? messages[index - 1].content : null;

      try {
        const job = await addArticles(credentials, [title], lang);
        for (let i = 0; i < 40; i += 1) {
          await new Promise((resolve) => setTimeout(resolve, 1200));
          const status = await fetchJob(credentials, job.job_id);
          if (status.status === "done") {
            setAddedTitles((prev) => [...prev, title]);
            const kb = await fetchKnowledgeBase(credentials);
            setPersonalArticles(kb.personal_articles);
            if (question) await send(question);
            break;
          }
          if (status.status === "failed" || status.status === "cancelled") break;
        }
      } catch {
        // Surfaced by the unchanged answer; the chip simply re-enables.
      } finally {
        setAddingTitle(null);
      }
    },
    [addingTitle, credentials, lang, messages, send],
  );

  // The panel follows the newest answer unless the user pinned an older one,
  // so asking a question refreshes the evidence beside it without a click.
  const latestWithEvidence = [...messages]
    .reverse()
    .find((m) => m.role === "assistant" && !m.streaming && (m.sources?.length || m.articles?.length));
  const focused =
    messages.find((m) => m.id === focusedId) ?? latestWithEvidence ?? null;

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void send(input);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send(input);
    }
  };

  const startNewChat = () => {
    abortRef.current?.abort();
    conversationId.current = null;
    setMessages([]);
    setBusy(false);
    setSidebarOpen(false);
  };

  const removeConversation = async (id: string) => {
    await deleteConversation(credentials, id);
    if (conversationId.current === id) startNewChat();
    void refreshConversations();
  };

  const signOut = () => {
    abortRef.current?.abort();
    clearCredentials();
    onSignOut();
  };

  const onLanguageChange = (code: string) => {
    setLang(code);
    // Retrieval is language-scoped, so mixing editions inside one thread would
    // produce citations the user cannot follow. Start a clean thread instead.
    if (messages.length > 0) startNewChat();
  };

  return (
    <div className="shell">
      <Sidebar
        credentials={credentials}
        conversations={conversations}
        activeId={conversationId.current}
        open={sidebarOpen}
        onSelect={(id) => void openConversation(id)}
        onNewChat={startNewChat}
        onDelete={(id) => void removeConversation(id)}
        onManageKb={() => {
          setKbOpen(true);
          setSidebarOpen(false);
        }}
        onSignOut={signOut}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="app">
        <header className="header">
          <div className="brand">
            <button
              className="menu-button"
              onClick={() => setSidebarOpen((v) => !v)}
              aria-label="Toggle sidebar"
            >
              ☰
            </button>
            <div className="brand-mark" aria-hidden="true">
              W
            </div>
            <div>
              <h1>Wikipedia Assistant</h1>
              <p className="brand-sub">
                Answers grounded in Wikipedia, with the sections they came from
              </p>
            </div>
          </div>

          <div className="header-actions">
            <label className="lang-picker">
              <span className="sr-only">Wikipedia language</span>
              <select
                value={lang}
                onChange={(event) => onLanguageChange(event.target.value)}
                disabled={busy}
              >
                {LANGUAGES.map((language) => (
                  <option key={language.code} value={language.code}>
                    {language.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              className="ghost-button"
              onClick={() => setPanelOpen((v) => !v)}
              aria-pressed={panelOpen}
            >
              Sources
            </button>
          </div>
        </header>

        <ApiKeyBar
          credentials={credentials}
          onSave={(apiKey) => {
            const next = { ...credentials, apiKey };
            saveCredentials(next);
            onCredentialsChange(next);
          }}
          onClear={() => {
            const next = { ...credentials, apiKey: "" };
            saveCredentials(next);
            onCredentialsChange(next);
          }}
        />

        <main className="messages">
          {messages.length === 0 ? (
            <div className="empty">
              <h2>Ask anything covered by Wikipedia</h2>
              <p>
                Every answer is built from indexed article text and cites the
                exact sections it used.
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    className="suggestion"
                    onClick={() => void send(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
              <p className="index-note">
                {session.shared_chunks.toLocaleString()} shared passages
                {personalArticles > 0
                  ? ` · ${personalArticles.toLocaleString()} of your own articles`
                  : " · add your own topics from the sidebar"}
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                busy={busy}
                isActive={message.id === focusedId && panelOpen}
                onPickDisambiguation={(title) => void send(title)}
                onShowSources={(id) => {
                  setFocusedId(id);
                  setPanelOpen(true);
                }}
              />
            ))
          )}
          <div ref={bottomRef} />
        </main>

        <form className="composer" onSubmit={onSubmit}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              credentials.apiKey
                ? "Ask a question…"
                : "Add your OpenAI API key above to start asking"
            }
            rows={1}
            disabled={busy || !credentials.apiKey}
          />
          {busy ? (
            <button
              type="button"
              className="send stop"
              onClick={() => abortRef.current?.abort()}
            >
              Stop
            </button>
          ) : (
            <button type="submit" className="send" disabled={!input.trim()}>
              Send
            </button>
          )}
        </form>
      </div>

      <SourcePanel
        message={focused}
        open={panelOpen}
        onClose={() => setPanelOpen(false)}
        onAddArticle={(title) => void addArticle(title)}
        addingTitle={addingTitle}
        addedTitles={addedTitles}
      />

      {kbOpen && (
        <KnowledgeBasePanel
          credentials={credentials}
          lang={lang}
          onClose={() => setKbOpen(false)}
          onChanged={() => {
            void (async () => {
              try {
                const { fetchKnowledgeBase } = await import("@/lib/api");
                const kb = await fetchKnowledgeBase(credentials);
                setPersonalArticles(kb.personal_articles);
              } catch {
                // Non-critical: the count refreshes on next sign-in.
              }
            })();
          }}
        />
      )}
    </div>
  );
}
