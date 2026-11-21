/**
 * Protected Administrator API services.
 * All requests route via the relative Vite proxy (/api -> http://127.0.0.1:8000).
 *
 * Security Requirements:
 * - Admin API keys are passed solely as runtime function parameters and transmitted
 *   only in the 'X-Admin-API-Key' HTTP header.
 * - Keys are never persisted in localStorage, sessionStorage, cookies, or environment variables.
 */

import { apiFetch } from './client';
import type { DocumentUploadResponse, RetryIndexResponse } from '../types/api';

/**
 * Upload and ingest a new Government Order PDF document.
 *
 * @param file The PDF File object to upload (must be .pdf, <= 25 MB, %PDF- signature).
 * @param adminApiKey In-memory administrator API key for authentication.
 */
export async function uploadDocument(
  file: File,
  adminApiKey: string,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  return apiFetch<DocumentUploadResponse>('/api/v1/admin/documents/upload', {
    method: 'POST',
    headers: {
      'X-Admin-API-Key': adminApiKey.trim(),
    },
    body: formData,
  });
}

/**
 * Trigger on-demand vector re-indexing for an already-saved Government Order.
 *
 * @param goId Database primary key ID of the Government Order.
 * @param adminApiKey In-memory administrator API key for authentication.
 */
export async function retryIndexDocument(
  goId: number,
  adminApiKey: string,
): Promise<RetryIndexResponse> {
  return apiFetch<RetryIndexResponse>(`/api/v1/admin/documents/${goId}/retry-index`, {
    method: 'POST',
    headers: {
      'X-Admin-API-Key': adminApiKey.trim(),
    },
  });
}
