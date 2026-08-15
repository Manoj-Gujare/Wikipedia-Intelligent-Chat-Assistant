"use client";

import type { ConversationSummary, Credentials } from "@/lib/types";
import { maskKey } from "@/lib/credentials";

export function Sidebar({
  credentials,
  conversations,
  activeId,
  open,
  onSelect,
  onNewChat,
  onDelete,
  onManageKb,
  onSignOut,
  onClose,
}: {
  credentials: Credentials;
  conversations: ConversationSummary[];
  activeId: string | null;
  open: boolean;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
  onManageKb: () => void;
  onSignOut: () => void;
  onClose: () => void;
}) {
  return (
    <>
      {open && <div className="scrim" onClick={onClose} aria-hidden="true" />}
      <aside className={`sidebar${open ? " sidebar-open" : ""}`}>
        <div className="sidebar-top">
          <button className="primary-button" onClick={onNewChat}>
            + New chat
          </button>
        </div>

        <nav className="conversation-list" aria-label="Previous chats">
          {conversations.length === 0 ? (
            <p className="sidebar-empty">No conversations yet.</p>
          ) : (
            conversations.map((conversation) => (
              <div
                key={conversation.conversation_id}
                className={`conversation-item${
                  conversation.conversation_id === activeId ? " active" : ""
                }`}
              >
                <button
                  className="conversation-open"
                  onClick={() => onSelect(conversation.conversation_id)}
                  title={conversation.title}
                >
                  {conversation.title}
                </button>
                <button
                  className="conversation-delete"
                  onClick={() => onDelete(conversation.conversation_id)}
                  aria-label={`Delete ${conversation.title}`}
                  title="Delete"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </nav>

        <div className="sidebar-bottom">
          <button className="sidebar-action" onClick={onManageKb}>
            Manage knowledge base
          </button>
          <div className="account-row">
            <div className="account-detail">
              <span className="account-email" title={credentials.email}>
                {credentials.email}
              </span>
              <span className="account-key">{maskKey(credentials.apiKey)}</span>
            </div>
            <button className="sidebar-action subtle" onClick={onSignOut}>
              Sign out
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
