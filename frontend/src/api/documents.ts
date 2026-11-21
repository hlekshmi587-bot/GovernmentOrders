/**
 * Public Document and Search API services.
 * All requests route via the relative Vite proxy (/api -> http://127.0.0.1:8000).
 */

import { apiFetch } from './client';
import type {
  DocumentDetailResponse,
  DocumentRelationshipsResponse,
  QueryRequest,
  QueryResponse,
  SearchRequest,
  SearchResponse,
} from '../types/api';

/**
 * Submit a natural-language question or search inquiry to the LangGraph multi-agent workflow.
 * Dispatches to the Supervisor Agent and returns evidence synthesis alongside structured result cards.
 *
 * @param params Query string and optional session thread ID.
 * @param signal Optional AbortSignal for user cancellation or timeout.
 */
export async function submitQuery(
  params: QueryRequest,
  signal?: AbortSignal,
): Promise<QueryResponse> {
  return apiFetch<QueryResponse>('/api/v1/query', {
    method: 'POST',
    body: JSON.stringify(params),
    signal,
  });
}

/**
 * Execute public search across Government Orders.
 * Supports HYBRID, KEYWORD, and SEMANTIC modes, optional department filtering, and pagination limits.
 */
export async function searchDocuments(params: SearchRequest): Promise<SearchResponse> {
  return apiFetch<SearchResponse>('/api/v1/search', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

/**
 * Retrieve detailed metadata, "Read" references, and verified PDF availability for a single Government Order.
 *
 * @param goId Database primary key ID of the Government Order.
 */
export async function getDocumentDetail(goId: number): Promise<DocumentDetailResponse> {
  return apiFetch<DocumentDetailResponse>(`/api/v1/documents/${goId}`, {
    method: 'GET',
  });
}

/**
 * Retrieve extracted incoming and outgoing relationships for a single Government Order.
 * Resolves verification statuses and maintains legal_effect_verified=false.
 *
 * @param goId Database primary key ID of the Government Order.
 */
export async function getDocumentRelationships(
  goId: number,
): Promise<DocumentRelationshipsResponse> {
  return apiFetch<DocumentRelationshipsResponse>(`/api/v1/documents/${goId}/relationships`, {
    method: 'GET',
  });
}

/**
 * Construct the relative URL for inline viewing or downloading a Government Order PDF.
 *
 * @param goId Database primary key ID of the Government Order.
 * @param download If true, triggers attachment download; if false, serves inline for viewing.
 */
export function getPdfUrl(goId: number, download = false): string {
  return `/api/v1/documents/${goId}/pdf?download=${download}`;
}
