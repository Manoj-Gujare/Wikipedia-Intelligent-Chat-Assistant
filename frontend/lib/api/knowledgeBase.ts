import type { Credentials, IngestJob, KnowledgeBase } from "../types";
import { request } from "./client";

export function fetchKnowledgeBase(credentials: Credentials): Promise<KnowledgeBase> {
  return request<KnowledgeBase>("/api/kb", credentials);
}

export function startIngest(
  credentials: Credentials,
  topics: string[],
  articlesPerTopic: number,
  lang: string,
): Promise<IngestJob> {
  return request<IngestJob>("/api/kb/ingest", credentials, {
    method: "POST",
    body: JSON.stringify({
      topics,
      articles_per_topic: articlesPerTopic,
      lang,
    }),
  });
}

/** Add specific articles by exact title — the "that wasn't indexed" path. */
export function addArticles(
  credentials: Credentials,
  titles: string[],
  lang: string,
): Promise<IngestJob> {
  return request<IngestJob>("/api/kb/articles", credentials, {
    method: "POST",
    body: JSON.stringify({ titles, lang }),
  });
}

export function fetchJob(credentials: Credentials, jobId: string): Promise<IngestJob> {
  return request<IngestJob>(`/api/kb/jobs/${jobId}`, credentials);
}

export function cancelJob(
  credentials: Credentials,
  jobId: string,
): Promise<{ cancelled: boolean }> {
  return request(`/api/kb/jobs/${jobId}/cancel`, credentials, { method: "POST" });
}

export function resetKnowledgeBase(
  credentials: Credentials,
): Promise<{ collections_removed: number }> {
  return request("/api/kb", credentials, { method: "DELETE" });
}
