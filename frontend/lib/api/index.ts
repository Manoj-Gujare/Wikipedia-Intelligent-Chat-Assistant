/**
 * The backend API, grouped by resource.
 *
 * Re-exported from one place so callers keep importing `@/lib/api` without
 * caring which module a call lives in.
 */

export { ApiError, API_URL } from "./client";
export { login, register, startSession, verifyKey } from "./auth";
export { streamChat, type StreamHandlers } from "./chat";
export {
  deleteConversation,
  fetchConversation,
  listConversations,
  type RestoredConversation,
  type RestoredTurn,
} from "./conversations";
export {
  addArticles,
  cancelJob,
  fetchJob,
  fetchKnowledgeBase,
  resetKnowledgeBase,
  startIngest,
} from "./knowledgeBase";
