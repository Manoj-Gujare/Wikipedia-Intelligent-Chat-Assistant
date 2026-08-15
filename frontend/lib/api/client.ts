import { authHeaders } from "../credentials";
import type { Credentials } from "../types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/**
 * Raise an ApiError carrying the server's own message where it sent one.
 *
 * FastAPI puts the useful text in `detail`; a non-JSON body (a proxy error
 * page, a dropped connection) leaves the status-based fallback standing.
 */
export async function throwApiError(
  response: Response,
  fallback: string,
): Promise<never> {
  let detail = `${fallback} (${response.status})`;
  try {
    const body = await response.json();
    if (body?.detail) detail = body.detail;
  } catch {
    // Non-JSON error body; the status-based message stands.
  }
  throw new ApiError(detail, response.status);
}

export async function request<T>(
  path: string,
  credentials: Credentials,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(credentials),
      ...(init.headers ?? {}),
    },
  });

  if (!response.ok) await throwApiError(response, "Request failed");

  return (await response.json()) as T;
}
