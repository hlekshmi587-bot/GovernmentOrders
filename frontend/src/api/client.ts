/**
 * Base HTTP Client for Kerala Government Orders API.
 * Uses browser fetch through the Vite proxy (/api -> http://127.0.0.1:8000).
 */

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error ${status}: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

interface FastAPIValidationError {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

/**
 * Perform a typed HTTP request against the backend API.
 *
 * @param path Relative path (e.g. '/api/v1/search')
 * @param options Standard RequestInit options
 */
export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers || {});

  // Do not set Content-Type if body is FormData; browser sets multipart/form-data boundary automatically
  if (options.body && !(options.body instanceof FormData)) {
    if (!headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }
  }

  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }

  let response: Response;
  try {
    response = await fetch(path, {
      ...options,
      headers,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Network request failed';
    throw new ApiError(0, `${message}. Please ensure the backend server is running.`);
  }

  // Handle successful responses
  if (response.ok) {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return (await response.json()) as T;
    }
    return {} as T;
  }

  // Parse structured errors from FastAPI
  let errorDetail = `Request failed with status ${response.status} (${response.statusText})`;
  try {
    const data = await response.json();
    if (data && typeof data === 'object') {
      if (typeof data.detail === 'string') {
        errorDetail = data.detail;
      } else if (Array.isArray(data.detail)) {
        // FastAPI / Pydantic validation error list
        errorDetail = (data.detail as FastAPIValidationError[])
          .map((e) => {
            const loc = e.loc ? e.loc.filter((p) => p !== 'body').join('.') : '';
            return loc ? `${loc}: ${e.msg || 'Invalid value'}` : e.msg || 'Validation error';
          })
          .join(', ');
      } else if (data.message && typeof data.message === 'string') {
        errorDetail = data.message;
      }
    }
  } catch {
    // Non-JSON response body or empty error body; keep fallback errorDetail
  }

  throw new ApiError(response.status, errorDetail);
}
