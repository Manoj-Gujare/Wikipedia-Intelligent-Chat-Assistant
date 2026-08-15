"use client";

import { LANGUAGES } from "@/lib/languages";

export function ChatHeader({
  lang,
  busy,
  panelOpen,
  onToggleSidebar,
  onLanguageChange,
  onTogglePanel,
}: {
  lang: string;
  busy: boolean;
  panelOpen: boolean;
  onToggleSidebar: () => void;
  onLanguageChange: (code: string) => void;
  onTogglePanel: () => void;
}) {
  return (
    <header className="header">
      <div className="brand">
        <button
          className="menu-button"
          onClick={onToggleSidebar}
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
          onClick={onTogglePanel}
          aria-pressed={panelOpen}
        >
          Sources
        </button>
      </div>
    </header>
  );
}
