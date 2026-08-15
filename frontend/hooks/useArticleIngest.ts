"use client";

import { useCallback, useState } from "react";
import { addArticles, fetchJob, fetchKnowledgeBase } from "@/lib/api";
import type { Credentials } from "@/lib/types";

/** Poll interval and ceiling for a single-article ingest job. */
const POLL_MS = 1200;
const MAX_POLLS = 40;

/**
 * Adding an article the index didn't have, then closing the loop the user is
 * actually in: "you clearly know this page exists, so use it."
 *
 * The re-ask is not done here. This hook knows about ingestion; it does not
 * know what a message is. `onIngested` is awaited rather than fired and
 * forgotten, so the chip stays disabled until the re-asked answer has arrived
 * — which is what it did when both halves lived in one component.
 */
export function useArticleIngest({
  credentials,
  lang,
  initialCount,
  onIngested,
}: {
  credentials: Credentials;
  lang: string;
  initialCount: number;
  onIngested: (question: string | null) => void | Promise<void>;
}) {
  const [personalArticles, setPersonalArticles] = useState(initialCount);
  const [addingTitle, setAddingTitle] = useState<string | null>(null);
  const [addedTitles, setAddedTitles] = useState<string[]>([]);

  const refreshCount = useCallback(async () => {
    try {
      const kb = await fetchKnowledgeBase(credentials);
      setPersonalArticles(kb.personal_articles);
    } catch {
      // Non-critical: the count refreshes on next sign-in.
    }
  }, [credentials]);

  const addArticle = useCallback(
    async (title: string, question: string | null) => {
      if (addingTitle) return;
      setAddingTitle(title);

      try {
        const job = await addArticles(credentials, [title], lang);
        for (let i = 0; i < MAX_POLLS; i += 1) {
          await new Promise((resolve) => setTimeout(resolve, POLL_MS));
          const status = await fetchJob(credentials, job.job_id);
          if (status.status === "done") {
            setAddedTitles((prev) => [...prev, title]);
            const kb = await fetchKnowledgeBase(credentials);
            setPersonalArticles(kb.personal_articles);
            await onIngested(question);
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
    [addingTitle, credentials, lang, onIngested],
  );

  return { personalArticles, addingTitle, addedTitles, addArticle, refreshCount };
}
