/**
 * TypeScript Type Definitions for Kerala Government Orders System.
 * Strictly aligned with backend Pydantic models in app/api/schemas.py.
 */

export type RetrievalMode = 'HYBRID' | 'KEYWORD' | 'SEMANTIC';

export type RelationshipDirection = 'OUTGOING' | 'INCOMING';

export type VerificationStatus =
  | 'VERIFIED'
  | 'REJECTED'
  | 'PENDING'
  | 'PARTIALLY_VERIFIED'
  | 'UNRECORDED'
  | (string & {});

export type UploadStatus = 'SUCCESS' | 'ALREADY_EXISTS' | 'INDEXING_FAILED' | 'FAILED';

export type RetryIndexStatus = 'SUCCESS' | 'UNCHANGED' | 'FAILED';

/**
 * Validated request schema for public Government Orders search.
 */
export interface SearchRequest {
  query: string;
  retrieval_mode?: RetrievalMode;
  department?: string | null;
  limit?: number;
}

/**
 * Structured search result item representing an indexed Government Order.
 */
export interface SearchResultItem {
  go_id: number | null;
  go_number: string;
  department: string | null;
  issue_date: string | null;
  source_filename: string;
  text_snippet: string | null;
  relevance_score: number | null;
  matched_sources: string[];
  has_pdf: boolean;
  pdf_url: string | null;
}

/**
 * Structured response schema for public Government Orders search.
 */
export interface SearchResponse {
  status: string;
  query: string;
  returned_count: number;
  limit_applied: number;
  has_more: boolean;
  is_exhaustive: boolean;
  results: SearchResultItem[];
  disclaimer: string;
  error?: string | null;
}

/**
 * Structured representation of a cited reference paper ('Read' section).
 */
export interface ReferenceItem {
  id?: number;
  ordinal: number;
  referenced_go_number: string;
  reference_date: string | null;
  page_number: number | null;
  evidence_text: string | null;
}

/**
 * Detailed metadata and reference inspection for a single Government Order.
 */
export interface DocumentDetailResponse {
  id: number;
  go_number: string;
  department: string | null;
  issue_date: string | null;
  source_filename: string;
  extracted_text_preview?: string | null;
  verification_status: string | null;
  has_pdf: boolean;
  pdf_url: string | null;
  references_count: number;
  relationships_count: number;
  references: ReferenceItem[];
  /** Mandatory invariant: legal validity is not verified */
  legal_effect_verified: false;
  disclaimer: string;
}

/**
 * Structured representation of an incoming or outgoing relationship.
 */
export interface RelationshipItem {
  id: number;
  direction: RelationshipDirection;
  source_go_id: number;
  source_go_number: string;
  target_go_number: string | null;
  relationship_type: string;
  matched_phrase: string | null;
  evidence_text: string | null;
  page_number: number | null;
  confidence: string | null;
  verification_status: VerificationStatus;
}

/**
 * Detailed inspection of incoming and outgoing relationships for a Government Order.
 */
export interface DocumentRelationshipsResponse {
  go_id: number;
  go_number: string;
  outgoing_count: number;
  incoming_count: number;
  outgoing_relationships: RelationshipItem[];
  incoming_relationships: RelationshipItem[];
  /** Mandatory invariant: legal validity is not verified */
  legal_effect_verified: false;
  disclaimer: string;
}

/**
 * Validated request schema for natural-language Government Order queries.
 * Strictly aligned with QueryRequest in app/api/schemas.py.
 */
export interface QueryRequest {
  query: string;
  thread_id?: string | null;
}

/**
 * Unified response representing LangGraph multi-agent workflow output.
 * Strictly aligned with QueryResponse in app/api/schemas.py.
 */
export interface QueryResponse {
  thread_id: string;
  workflow_status: string;
  final_answer?: string | null;
  verification_status?: string | null;
  confidence?: string | null;
  evidence_sufficient?: boolean | null;
  /** Mandatory invariant: legal validity is not verified */
  legal_effect_verified: false;
  intent?: string | null;
  identified_go_numbers: string[];
  discovered_go_numbers: string[];
  human_review_required: boolean;
  human_review_prompt?: Record<string, unknown> | null;
  execution_trace_count: number;
  errors: string[];
  search_results: SearchResultItem[];
  relationships: RelationshipItem[];
}

/**
 * Unified response for administrator Government Order PDF upload.
 */
export interface DocumentUploadResponse {
  status: UploadStatus;
  message: string;
  go_id?: number | null;
  go_number?: string | null;
  department?: string | null;
  issue_date?: string | null;
  source_filename?: string | null;
  references_count: number;
  relationships_count: number;
  chunks_indexed: number;
  qdrant_version?: number | null;
  retry_index_url?: string | null;
}

/**
 * Response for administrative on-demand vector re-indexing.
 */
export interface RetryIndexResponse {
  status: RetryIndexStatus;
  message: string;
  go_id: number;
  go_number: string;
  chunks_indexed: number;
  qdrant_version?: number | null;
  indexing_status?: string | null;
}
