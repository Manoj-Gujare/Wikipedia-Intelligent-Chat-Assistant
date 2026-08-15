"use client";

import { useCallback, useState } from "react";
import { ApiKeyBar } from "@/components/auth/ApiKeyBar";
import { KnowledgeBasePanel } from "@/components/knowledge-base/KnowledgeBasePanel";
import { SourcePanel } from "@/components/sources/SourcePanel";
import { useArticleIngest } from "@/hooks/useArticleIngest";
import { useConversation } from "@/hooks/useConversation";
import { useConversationList } from "@/hooks/useConversationList";
import { clearCredentials, saveCredentials } from "@/lib/credentials";
import type { Credentials, Session } from "@/lib/types";
import { ChatHeader } from "./ChatHeader";
import { Composer } from "./Composer";
import { EmptyState } from "./EmptyState";
import { MessageList } from "./MessageList";
import { Sidebar } from "./Sidebar";

/**
 * The signed-in screen: sidebar, conversation, and the evidence panel beside
 * it. State lives in the three hooks; this component is the wiring between
 * them and the layout they render into.
 */
export function ChatScreen({
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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [kbOpen, setKbOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [focusedId, setFocusedId] = useState<string | null>(null);

  const threads = useConversationList(credentials);

  const chat = useConversation({
    credentials,
    onFirstTurn: useCallback(() => void threads.refresh(), [threads.refresh]),
  });

  const ask = useCallback(
    (text: string) => {
      // Unpin, so the panel tracks the answer being written rather than an
      // older one the user happened to inspect.
      setFocusedId(null);
      void chat.send(text);
    },
    [chat.send],
  );

  const ingest = useArticleIngest({
    credentials,
    lang: chat.lang,
    initialCount: session.personal_articles,
    onIngested: useCallback(
      async (question: string | null) => {
        if (question) await chat.send(question);
      },
      [chat.send],
    ),
  });

  // The panel follows the newest answer unless the user pinned an older one,
  // so asking a question refreshes the evidence beside it without a click.
  const latestWithEvidence = [...chat.messages]
    .reverse()
    .find(
      (m) =>
        m.role === "assistant" &&
        !m.streaming &&
        (m.sources?.length || m.articles?.length),
    );
  const focused =
    chat.messages.find((m) => m.id === focusedId) ?? latestWithEvidence ?? null;

  const startNewChat = () => {
    chat.startNewChat();
    setSidebarOpen(false);
  };

  return (
    <div className="shell">
      <Sidebar
        credentials={credentials}
        conversations={threads.conversations}
        activeId={chat.activeConversationId}
        open={sidebarOpen}
        onSelect={(id) => {
          setSidebarOpen(false);
          void chat.openConversation(id);
        }}
        onNewChat={startNewChat}
        onDelete={(id) => {
          void (async () => {
            await threads.remove(id);
            if (chat.conversationIdRef.current === id) startNewChat();
          })();
        }}
        onManageKb={() => {
          setKbOpen(true);
          setSidebarOpen(false);
        }}
        onSignOut={() => {
          chat.stop();
          clearCredentials();
          onSignOut();
        }}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="app">
        <ChatHeader
          lang={chat.lang}
          busy={chat.busy}
          panelOpen={panelOpen}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
          onLanguageChange={(code) => {
            chat.changeLanguage(code);
            setSidebarOpen(false);
          }}
          onTogglePanel={() => setPanelOpen((v) => !v)}
        />

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

        <MessageList
          messages={chat.messages}
          busy={chat.busy}
          focusedId={focusedId}
          panelOpen={panelOpen}
          bottomRef={chat.bottomRef}
          emptyState={
            <EmptyState
              sharedChunks={session.shared_chunks}
              personalArticles={ingest.personalArticles}
              onPick={ask}
            />
          }
          onPickDisambiguation={ask}
          onShowSources={(id) => {
            setFocusedId(id);
            setPanelOpen(true);
          }}
        />

        <Composer
          input={chat.input}
          busy={chat.busy}
          hasApiKey={Boolean(credentials.apiKey)}
          textareaRef={chat.textareaRef}
          onChange={chat.setInput}
          onSubmit={() => ask(chat.input)}
          onStop={chat.stop}
        />
      </div>

      <SourcePanel
        message={focused}
        open={panelOpen}
        onClose={() => setPanelOpen(false)}
        onAddArticle={(title) => {
          // The question that prompted this answer is the user turn before it.
          const index = chat.messages.findIndex((m) =>
            m.articles?.some((a) => a.title === title),
          );
          const question = index > 0 ? chat.messages[index - 1].content : null;
          void ingest.addArticle(title, question);
        }}
        addingTitle={ingest.addingTitle}
        addedTitles={ingest.addedTitles}
      />

      {kbOpen && (
        <KnowledgeBasePanel
          credentials={credentials}
          lang={chat.lang}
          onClose={() => setKbOpen(false)}
          onChanged={() => void ingest.refreshCount()}
        />
      )}
    </div>
  );
}
