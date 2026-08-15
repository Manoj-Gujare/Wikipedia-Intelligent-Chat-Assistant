"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelJob,
  fetchJob,
  fetchKnowledgeBase,
  resetKnowledgeBase,
  startIngest,
} from "@/lib/api";
import type { Credentials, IngestJob, KnowledgeBase } from "@/lib/types";

const POLL_MS = 1500;

export function KnowledgeBasePanel({
  credentials,
  lang,
  onClose,
  onChanged,
}: {
  credentials: Credentials;
  lang: string;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [custom, setCustom] = useState("");
  const [perTopic, setPerTopic] = useState(25);
  const [job, setJob] = useState<IngestJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchKnowledgeBase(credentials);
      setKb(data);
      if (data.active_job) setJob(data.active_job);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [credentials]);

  useEffect(() => {
    void load();
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, [load]);

  // Poll while a build is running; stop as soon as it reaches a terminal state.
  useEffect(() => {
    if (!job || (job.status !== "running" && job.status !== "queued")) return;

    pollRef.current = setTimeout(async () => {
      try {
        const updated = await fetchJob(credentials, job.job_id);
        setJob(updated);
        if (updated.status === "done") {
          await load();
          onChanged();
        }
      } catch {
        // Transient failure; the next tick retries.
      }
    }, POLL_MS);

    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, [job, credentials, load, onChanged]);

  const toggle = (category: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(category)) next.delete(category);
      else next.add(category);
      return next;
    });
  };

  const topics = () => {
    const list = [...selected];
    for (const entry of custom.split(",")) {
      const trimmed = entry.trim();
      if (trimmed) list.push(trimmed);
    }
    return list.slice(0, 10);
  };

  const build = async () => {
    const chosen = topics();
    if (chosen.length === 0) {
      setError("Pick at least one topic, or type your own.");
      return;
    }
    setError(null);
    try {
      const started = await startIngest(credentials, chosen, perTopic, lang);
      setJob(started);
      setSelected(new Set());
      setCustom("");
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const reset = async () => {
    setError(null);
    try {
      await resetKnowledgeBase(credentials);
      await load();
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const running = job?.status === "running" || job?.status === "queued";
  const progress =
    job && job.articles_total > 0
      ? Math.min(100, Math.round((job.articles_done / job.articles_total) * 100))
      : 0;
  const chosenCount = topics().length;

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Manage knowledge base"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal-header">
          <h2>Knowledge base</h2>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="modal-body">
          {kb && (
            <div className="kb-stats">
              <div className="kb-stat">
                <strong>{kb.shared_chunks.toLocaleString()}</strong>
                <span>shared passages</span>
              </div>
              <div className="kb-stat">
                <strong>{kb.personal_articles.toLocaleString()}</strong>
                <span>your articles</span>
              </div>
              <div className="kb-stat">
                <strong>{kb.personal_chunks.toLocaleString()}</strong>
                <span>your passages</span>
              </div>
            </div>
          )}

          <p className="kb-note">
            Everyone can search the shared corpus. Anything you add here is
            private to your workspace and embedded with your own API key, so you
            are billed only for what you add.
          </p>

          {running ? (
            <section className="kb-progress">
              <h3>Building{job?.current ? `: ${job.current}` : "…"}</h3>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <p className="kb-progress-detail">
                {job?.articles_done} / {job?.articles_total} articles ·{" "}
                {job?.chunks_written.toLocaleString()} passages indexed
              </p>
              <button
                className="sidebar-action subtle"
                onClick={async () => {
                  if (job) await cancelJob(credentials, job.job_id);
                  await load();
                  setJob(null);
                }}
              >
                Cancel build
              </button>
            </section>
          ) : (
            <>
              <section>
                <h3 className="kb-heading">Add topics</h3>
                <div className="topic-grid">
                  {(kb?.suggested_topics ?? []).map((topic) => (
                    <button
                      key={topic.category}
                      className={`topic-chip${
                        selected.has(topic.category) ? " selected" : ""
                      }`}
                      onClick={() => toggle(topic.category)}
                    >
                      {topic.label}
                    </button>
                  ))}
                </div>

                <label className="field">
                  <span>Or type your own (comma separated)</span>
                  <input
                    type="text"
                    value={custom}
                    onChange={(e) => setCustom(e.target.value)}
                    placeholder="cricket, quantum computing, Category:Volcanoes"
                  />
                </label>

                <label className="field">
                  <span>Articles per topic: {perTopic}</span>
                  <input
                    type="range"
                    min={5}
                    max={100}
                    step={5}
                    value={perTopic}
                    onChange={(e) => setPerTopic(Number(e.target.value))}
                  />
                </label>

                {chosenCount > 0 && (
                  <p className="kb-estimate">
                    {chosenCount} topic{chosenCount === 1 ? "" : "s"} ×{" "}
                    {perTopic} articles ≈ {chosenCount * perTopic} articles to
                    fetch and embed.
                  </p>
                )}
              </section>

              {job && job.status === "failed" && (
                <p className="login-error">{job.error}</p>
              )}
              {error && <p className="login-error">{error}</p>}

              <div className="modal-actions">
                <button className="send" onClick={build} disabled={chosenCount === 0}>
                  Build knowledge base
                </button>
                {(kb?.personal_articles ?? 0) > 0 && (
                  <button className="sidebar-action subtle" onClick={reset}>
                    Remove my articles
                  </button>
                )}
              </div>
            </>
          )}

          {kb && kb.titles.length > 0 && (
            <section>
              <h3 className="kb-heading">In your knowledge base</h3>
              <div className="article-chips">
                {kb.titles.slice(0, 40).map((title) => (
                  <span key={title} className="article-chip">
                    <span className="article-chip-title">{title}</span>
                  </span>
                ))}
              </div>
              {kb.titles.length > 40 && (
                <p className="kb-note">…and {kb.titles.length - 40} more.</p>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
