export interface Source {
  index: number;
  title: string;
  section: string;
  url: string;
  article_url: string;
  lang: string;
  snippet: string;
  score: number;
}

export interface ArticleLink {
  title: string;
  url: string;
  lang: string;
  summary: string;
}

export interface DisambiguationOption {
  title: string;
  url: string;
}

export interface Timings {
  rewrite_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface ChatResponse {
  conversation_id: string;
  answer: string;
  sources: Source[];
  articles: ArticleLink[];
  disambiguation: DisambiguationOption[];
  disambiguation_term: string | null;
  lang: string;
  resolved_query: string | null;
  cached: boolean;
  used_live_search: boolean;
  timings: Timings;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Graph path taken; "chitchat" suppresses the thinking indicator. */
  intent?: string;
  sources?: Source[];
  articles?: ArticleLink[];
  disambiguation?: DisambiguationOption[];
  disambiguationTerm?: string | null;
  resolvedQuery?: string | null;
  timings?: Timings;
  streaming?: boolean;
  error?: boolean;
  /** True when the index couldn't answer and links came from live search —
   *  these articles can be added to the user's knowledge base. */
  usedLiveSearch?: boolean;
}

export interface Credentials {
  email: string;
  apiKey: string;
  /** Session token issued by /auth/login; identity comes from this. */
  token?: string;
}

export interface Session {
  account_id: string;
  email: string;
  role: string;
  shared_chunks: number;
  personal_chunks: number;
  personal_articles: number;
}

export interface LoginResponse extends Session {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  lang: string;
  updated_at: number;
}

export interface IngestJob {
  job_id: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  topics: string[];
  articles_done: number;
  articles_total: number;
  chunks_written: number;
  current: string;
  error: string;
}

export interface SuggestedTopic {
  label: string;
  category: string;
}

export interface KnowledgeBase {
  shared_chunks: number;
  personal_chunks: number;
  personal_articles: number;
  articles_by_language: Record<string, number>;
  titles: string[];
  suggested_topics: SuggestedTopic[];
  active_job: IngestJob | null;
  recent_jobs: IngestJob[];
}

export interface Language {
  code: string;
  label: string;
}
