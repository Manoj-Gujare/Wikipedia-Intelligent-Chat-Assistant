"use client";

import type { Message } from "@/lib/types";

/**
 * Citations and routing links for one answer, moved out of the conversation.
 *
 * Inline citation blocks pushed every answer apart and made a thread hard to
 * read. The evidence still matters, so it lives beside the chat rather than
 * inside it: the transcript reads as prose, and the sources for whichever
 * answer you are looking at are one glance away.
 */
export function SourcePanel({
  message,
  open,
  onClose,
  onAddArticle,
  addingTitle,
  addedTitles,
}: {
  message: Message | null;
  open: boolean;
  onClose: () => void;
  onAddArticle?: (title: string) => void;
  addingTitle?: string | null;
  addedTitles?: string[];
}) {
  const sources = message?.sources ?? [];
  const articles = message?.articles ?? [];
  const timings = message?.timings;
  const canAdd = Boolean(message?.usedLiveSearch && onAddArticle);
  const isEmpty = sources.length === 0 && articles.length === 0;

  return (
    <>
      {open && <div className="scrim" onClick={onClose} aria-hidden="true" />}
      <aside
        className={`source-panel${open ? " source-panel-open" : ""}`}
        aria-label="Sources for the selected answer"
      >
        <header className="source-panel-header">
          <h2>Sources</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close sources">
            ×
          </button>
        </header>

        <div className="source-panel-body">
          {isEmpty ? (
            <p className="source-panel-empty">
              Citations and Wikipedia links for an answer appear here.
            </p>
          ) : (
            <>
              {sources.length > 0 && (
                <section>
                  <h3 className="kb-heading">Cited passages</h3>
                  <ol className="source-items">
                    {sources.map((source) => (
                      <li key={`${source.index}-${source.url}`} className="source-item">
                        <span className="source-index">{source.index}</span>
                        <div className="source-body">
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-title"
                          >
                            {source.title}
                            {source.section && source.section !== "Introduction" && (
                              <span className="source-section"> § {source.section}</span>
                            )}
                          </a>
                          <p className="source-snippet">{source.snippet}</p>
                          <span className="source-score">
                            relevance {(source.score * 100).toFixed(0)}%
                          </span>
                        </div>
                      </li>
                    ))}
                  </ol>
                </section>
              )}

              {articles.length > 0 && (
                <section>
                  <h3 className="kb-heading">
                    {canAdd ? "Not in your knowledge base" : "Read on Wikipedia"}
                  </h3>
                  <div className="panel-articles">
                    {articles.map((article) => {
                      const adding = addingTitle === article.title;
                      const added = addedTitles?.includes(article.title);
                      return (
                        <div key={article.url} className="panel-article">
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="panel-article-title"
                          >
                            {article.title}
                          </a>
                          {canAdd && (
                            <button
                              className="article-chip-add"
                              onClick={() => onAddArticle?.(article.title)}
                              disabled={adding || added}
                              title={`Add "${article.title}" to your knowledge base`}
                            >
                              {added ? "added" : adding ? "adding…" : "+ add"}
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </section>
              )}
            </>
          )}

          {timings && timings.total_ms > 0 && (
            <p className="timings">
              {(timings.total_ms / 1000).toFixed(2)}s
              {timings.retrieval_ms > 0 && ` · retrieval ${timings.retrieval_ms}ms`}
              {timings.generation_ms > 0 && ` · generation ${timings.generation_ms}ms`}
            </p>
          )}
        </div>
      </aside>
    </>
  );
}
