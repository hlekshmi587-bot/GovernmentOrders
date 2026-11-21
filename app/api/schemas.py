"""
Pydantic Request and Response Schemas for the FastAPI Backend.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check status response."""
    status: Literal["healthy"] = "healthy"
    service: str = "kerala-government-orders-multi-agent-system"
    version: str = "1.0.0"
    ollama_online: bool
    postgres_online: bool
    qdrant_online: bool


class QueryRequest(BaseModel):
    """Natural-language Government Order query request."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language question or Government Order inquiry",
        examples=["Show details for G.O.(Rt)No.24/2026/ITD"],
    )
    thread_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional session thread ID for conversation continuity. Auto-generated if omitted.",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query cannot be empty or whitespace only.")
        return clean


class SearchResultItemSchema(BaseModel):
    """Structured search result item representing an indexed Government Order."""
    go_id: Optional[int] = Field(
        default=None,
        description="Primary key ID in PostgreSQL database",
    )
    go_number: str = Field(
        ...,
        description="Canonical Government Order number",
    )
    department: Optional[str] = Field(
        default=None,
        description="Department name",
    )
    issue_date: Optional[str] = Field(
        default=None,
        description="Issue date in ISO format (YYYY-MM-DD)",
    )
    source_filename: str = Field(
        ...,
        description="Physical source PDF filename",
    )
    text_snippet: Optional[str] = Field(
        default=None,
        description="Excerpt containing matched keywords or relevant text context",
    )
    relevance_score: Optional[float] = Field(
        default=None,
        description="Relevance score (fused RRF or cosine similarity)",
    )
    matched_sources: List[str] = Field(
        default_factory=list,
        description="Retrieval sources: KEYWORD, SEMANTIC, or both",
    )
    has_pdf: bool = Field(
        default=False,
        description="Whether the original PDF is available on disk",
    )
    pdf_url: Optional[str] = Field(
        default=None,
        description="API endpoint to view or download original PDF",
    )


class RelationshipItemSchema(BaseModel):
    """Structured representation of an incoming or outgoing relationship."""
    id: int = Field(
        ...,
        description="Primary key ID of the relationship record in database",
    )
    direction: Literal["OUTGOING", "INCOMING"] = Field(
        ...,
        description="Relationship direction: OUTGOING or INCOMING",
    )
    source_go_id: int = Field(
        ...,
        description="Database ID of the acting source Government Order",
    )
    source_go_number: str = Field(
        ...,
        description="Government Order number of the acting source document",
    )
    target_go_number: Optional[str] = Field(
        default=None,
        description="Government Order number of the referenced target document, or None if unspecified/statutory",
    )
    relationship_type: str = Field(
        ...,
        description="Relationship type (e.g. PARTIALLY_AMENDS, SUPERSEDES, REFERENCES)",
    )
    matched_phrase: Optional[str] = Field(
        default=None,
        description="Exact phrase matching relationship pattern",
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description="Documentary context excerpt from source document",
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number where the relationship signal was found",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Extraction confidence score (HIGH, MEDIUM, LOW)",
    )
    verification_status: str = Field(
        default="UNRECORDED",
        description="Actual status from linked verification record or UNRECORDED",
    )


class QueryResponse(BaseModel):
    """Unified response representing multi-agent workflow output."""
    thread_id: str
    workflow_status: str  # "COMPLETED", "WAITING_FOR_HUMAN_REVIEW", "FAILED"
    final_answer: Optional[str] = None
    verification_status: Optional[str] = None
    confidence: Optional[str] = None
    evidence_sufficient: Optional[bool] = None
    legal_effect_verified: Literal[False] = False
    intent: Optional[str] = None
    identified_go_numbers: List[str] = Field(default_factory=list)
    discovered_go_numbers: List[str] = Field(default_factory=list)
    human_review_required: bool = False
    human_review_prompt: Optional[Dict[str, Any]] = None
    execution_trace_count: int = 0
    errors: List[str] = Field(default_factory=list)
    search_results: List[SearchResultItemSchema] = Field(
        default_factory=list,
        description="Structured search results if search agent executed",
    )
    relationships: List[RelationshipItemSchema] = Field(
        default_factory=list,
        description="Structured relationship records if relationship investigation executed",
    )

    @field_validator("legal_effect_verified")
    @classmethod
    def reject_true_legal_effect(cls, v: Any) -> Literal[False]:
        if v is not False:
            raise ValueError("legal_effect_verified must strictly remain False.")
        return False


class ResumeRequest(BaseModel):
    """Request to resume a workflow paused at human_review_node."""
    thread_id: str = Field(..., min_length=1, max_length=128, description="Thread ID of the paused workflow")
    feedback: str = Field(..., min_length=1, max_length=1000, description="User clarification or feedback")

    @field_validator("thread_id", "feedback")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field cannot be empty or whitespace only.")
        return clean


class ThreadStatusResponse(BaseModel):
    """Detailed inspection of an in-memory thread checkpoint."""
    thread_id: str
    workflow_status: str  # "COMPLETED", "WAITING_FOR_HUMAN_REVIEW", "NOT_FOUND", "INITIALIZED"
    next_nodes: List[str] = Field(default_factory=list)
    human_review_required: bool = False
    human_review_prompt: Optional[Dict[str, Any]] = None
    is_persisted_across_restarts: Literal[False] = False


class DocumentUploadResponse(BaseModel):
    """Unified response for administrator Government Order PDF upload."""
    status: Literal["SUCCESS", "ALREADY_EXISTS", "INDEXING_FAILED", "FAILED"]
    message: str
    go_id: Optional[int] = None
    go_number: Optional[str] = None
    department: Optional[str] = None
    issue_date: Optional[str] = None
    source_filename: Optional[str] = None
    references_count: int = 0
    relationships_count: int = 0
    chunks_indexed: int = 0
    qdrant_version: Optional[int] = None
    retry_index_url: Optional[str] = None


class RetryIndexResponse(BaseModel):
    """Response for administrative on-demand vector re-indexing."""
    status: Literal["SUCCESS", "UNCHANGED", "FAILED"]
    message: str
    go_id: int
    go_number: str
    chunks_indexed: int = 0
    qdrant_version: Optional[int] = None
    indexing_status: Optional[str] = None


class SearchRequest(BaseModel):
    """Validated request schema for public Government Orders search."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query, keywords, topic, or Government Order number",
        examples=["Electricity subsidy", "G.O.(Rt)No.24/2026/ITD"],
    )
    retrieval_mode: Literal["HYBRID", "KEYWORD", "SEMANTIC"] = Field(
        default="HYBRID",
        description="Retrieval strategy: HYBRID, KEYWORD, or SEMANTIC",
    )
    department: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional exact department name filter",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of Government Orders to return (1 to 50)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query cannot be empty or whitespace only.")
        return clean

    @field_validator("department")
    @classmethod
    def validate_department_optional(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        return clean if clean else None


class SearchResponse(BaseModel):
    """Structured response schema for public Government Orders search."""
    status: str = Field(
        ...,
        description="Search status: SUCCESS, NO_MATCH, PARTIAL_SUCCESS, or ERROR",
    )
    query: str = Field(
        ...,
        description="The executed search query",
    )
    returned_count: int = Field(
        default=0,
        ge=0,
        description="Number of results returned in this response",
    )
    limit_applied: int = Field(
        default=10,
        ge=1,
        description="Limit applied to the candidate retrieval window",
    )
    has_more: bool = Field(
        default=False,
        description="Whether more candidate orders were found beyond the returned limit",
    )
    is_exhaustive: bool = Field(
        ...,
        description="Whether the search scanned all available candidates without reaching candidate ceiling",
    )
    results: List[SearchResultItemSchema] = Field(
        default_factory=list,
        description="Ranked list of matching Government Orders",
    )
    disclaimer: str = Field(
        default="Document relationships and search extractions are assistive; verify official gazetted status.",
        description="Standard legal advisory notice",
    )


class ReferenceItemSchema(BaseModel):
    """Structured representation of a cited reference paper ('Read' section)."""
    ordinal: int = Field(
        ...,
        description="Ordinal reference number as cited in 'Read' section",
    )
    referenced_go_number: str = Field(
        ...,
        description="Referenced Government Order number",
    )
    reference_date: Optional[str] = Field(
        default=None,
        description="Date of referenced order in ISO format (YYYY-MM-DD)",
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number where reference appears",
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description="Verbatim text of the reference citation",
    )


class DocumentDetailResponse(BaseModel):
    """Detailed metadata and reference inspection for a single Government Order."""
    id: int = Field(
        ...,
        description="Database primary key ID of the Government Order",
    )
    go_number: str = Field(
        ...,
        description="Canonical Government Order number",
    )
    department: Optional[str] = Field(
        default=None,
        description="Originating government department",
    )
    issue_date: Optional[str] = Field(
        default=None,
        description="Date of issuance in ISO format (YYYY-MM-DD)",
    )
    source_filename: str = Field(
        ...,
        description="Physical source PDF filename in storage",
    )
    extracted_text_preview: Optional[str] = Field(
        default=None,
        description="Preview of the extracted text content",
    )
    verification_status: Optional[str] = Field(
        default=None,
        description="Document-level verification status in database",
    )
    has_pdf: bool = Field(
        default=False,
        description="Whether the original PDF file exists on the server disk",
    )
    pdf_url: Optional[str] = Field(
        default=None,
        description="API endpoint URL to stream or download the original PDF",
    )
    references_count: int = Field(
        default=0,
        ge=0,
        description="Total number of cited references",
    )
    references: List[ReferenceItemSchema] = Field(
        default_factory=list,
        description="Detailed list of cited references from 'Read' section",
    )
    relationships_count: int = Field(
        default=0,
        ge=0,
        description="Total number of extracted relationships",
    )
    legal_effect_verified: Literal[False] = Field(
        default=False,
        description="Mandatory invariant: legal validity is not verified",
    )
    disclaimer: str = Field(
        default="Legal validity and operative status must be verified against official notifications.",
        description="Standard legal advisory notice",
    )

    @field_validator("legal_effect_verified")
    @classmethod
    def reject_true_legal_effect(cls, v: Any) -> Literal[False]:
        if v is not False:
            raise ValueError("legal_effect_verified must strictly remain False.")
        return False


class DocumentRelationshipsResponse(BaseModel):
    """Detailed inspection of incoming and outgoing relationships for a Government Order."""
    go_id: int = Field(
        ...,
        description="Database primary key ID of the inspected Government Order",
    )
    go_number: str = Field(
        ...,
        description="Canonical Government Order number",
    )
    outgoing_count: int = Field(
        default=0,
        ge=0,
        description="Number of outgoing relationships (earlier orders referenced or modified)",
    )
    outgoing_relationships: List[RelationshipItemSchema] = Field(
        default_factory=list,
        description="List of outgoing relationship records",
    )
    incoming_count: int = Field(
        default=0,
        ge=0,
        description="Number of incoming relationships (subsequent orders modifying this order)",
    )
    incoming_relationships: List[RelationshipItemSchema] = Field(
        default_factory=list,
        description="List of incoming relationship records",
    )
    legal_effect_verified: Literal[False] = Field(
        default=False,
        description="Mandatory invariant: legal validity is not verified",
    )
    disclaimer: str = Field(
        default=(
            "Document relationships reflect automated textual analysis. "
            "Official legal validity and operative effect must be verified against official gazetted notifications."
        ),
        description="Standard legal advisory notice",
    )

    @field_validator("legal_effect_verified")
    @classmethod
    def reject_true_legal_effect(cls, v: Any) -> Literal[False]:
        if v is not False:
            raise ValueError("legal_effect_verified must strictly remain False.")
        return False


