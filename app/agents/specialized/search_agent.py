"""
Search Agent Implementation for Kerala Government Orders System.

Coordinates keyword search (PostgreSQL SearchTool) and semantic vector search
(QdrantTool) to provide unified KEYWORD, SEMANTIC, and HYBRID retrieval with
Reciprocal Rank Fusion (RRF), PostgreSQL ground-truth validation, and complete provenance.
"""

from datetime import date
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select

from app.database.connection import SessionLocal
from app.database.models import GovernmentOrderDB
from app.agents.tools.search_tool import SearchTool
from app.agents.tools.qdrant_tool import QdrantTool

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    """Supported retrieval modes for SearchAgent."""
    KEYWORD = "KEYWORD"
    SEMANTIC = "SEMANTIC"
    HYBRID = "HYBRID"


class SearchAgentRequest(BaseModel):
    """
    Validated request contract for SearchAgent.
    Treats search query as literal text without blacklisting natural query words.
    Enforces strict typing and bounds (forbids extra fields, booleans as ints, etc.).
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Search query or statement")
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Retrieval strategy: KEYWORD, SEMANTIC, or HYBRID"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of Government Orders to return (strictly 1 to 50)"
    )
    department_filter: Optional[str] = Field(
        default=None,
        description="Exact department name filter"
    )
    issue_date_start: Optional[date] = Field(
        default=None,
        description="Earliest issue date (inclusive)"
    )
    issue_date_end: Optional[date] = Field(
        default=None,
        description="Latest issue date (inclusive)"
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Smoothing constant for Reciprocal Rank Fusion"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Search query cannot be empty or whitespace.")
        return v.strip()

    @field_validator("limit", mode="before")
    @classmethod
    def validate_limit_strict_int(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"Limit must be an integer, got {type(v).__name__}")
        if v < 1 or v > 50:
            raise ValueError(f"Limit must be strictly between 1 and 50 (got {v})")
        return v

    @field_validator("rrf_k", mode="before")
    @classmethod
    def validate_rrf_k_strict_int(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"rrf_k must be a positive integer >= 1 (got {repr(v)})")
        return v

    @field_validator("department_filter")
    @classmethod
    def validate_department_filter(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("department_filter must be a non-empty string or None.")
        return v.strip()

    @model_validator(mode="after")
    def validate_date_range(self) -> "SearchAgentRequest":
        if self.issue_date_start and self.issue_date_end:
            if self.issue_date_start > self.issue_date_end:
                raise ValueError(
                    f"issue_date_start ({self.issue_date_start}) cannot be later than issue_date_end ({self.issue_date_end})."
                )
        return self


class SupportingChunk(BaseModel):
    """Individual chunk supporting a semantic or hybrid match."""
    chunk_index: int
    chunk_text: str
    page_number: Optional[int] = None
    score: Optional[float] = None


class SearchResultItem(BaseModel):
    """
    Unified Government Order search result item.
    Omits verification_status so search retrieval is never conflated with legal verification.
    """
    go_id: Optional[int] = None
    go_number: str
    department: Optional[str] = None
    issue_date: Optional[str] = None
    source_filename: str

    # Granular retrieval provenance
    matched_sources: List[str]
    keyword_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    rrf_score: Optional[float] = None

    # Textual evidence
    text_snippet: Optional[str] = None
    supporting_chunks: List[SupportingChunk] = Field(default_factory=list)


class SearchAgentResponse(BaseModel):
    """Structured output contract returned by SearchAgent."""
    status: str  # "SUCCESS", "NO_MATCH", "PARTIAL_SUCCESS", "PARTIAL_FAILURE", "ERROR"
    query: str
    requested_mode: RetrievalMode
    attempted_backends: List[str] = Field(default_factory=list)
    successful_backends: List[str] = Field(default_factory=list)
    failed_backends: List[str] = Field(default_factory=list)
    candidates_retrieved: int = 0
    candidates_retained: int = 0
    candidate_ceiling_reached: bool = False
    is_exhaustive: bool = True
    stale_qdrant_points_discarded: int = 0
    returned_count: int = 0
    results: List[SearchResultItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)


class SearchAgent:
    """
    Specialized agent coordinating PostgreSQL keyword search and Qdrant semantic search.
    Enforces PostgreSQL as authoritative source of truth, performs RRF fusion,
    and returns complete retrieval provenance.
    """

    def __init__(
        self,
        search_tool: Optional[SearchTool] = None,
        qdrant_tool: Optional[QdrantTool] = None,
    ) -> None:
        self.search_tool = search_tool or SearchTool()
        self.qdrant_tool = qdrant_tool or QdrantTool()

    def execute_search(self, request_data: Any) -> SearchAgentResponse:
        """
        Main entry point for executing search requests.
        Validates request_data into SearchAgentRequest and executes corresponding mode.
        """
        # Step 1: Validate request contract
        try:
            if isinstance(request_data, SearchAgentRequest):
                req = request_data
            elif isinstance(request_data, dict):
                req = SearchAgentRequest(**request_data)
            else:
                raise ValueError(f"Invalid request data type: {type(request_data).__name__}")
        except Exception as exc:
            logger.warning("SearchAgentRequest validation failed: %s", exc)
            query_str = ""
            if isinstance(request_data, dict):
                query_str = str(request_data.get("query", ""))
            elif hasattr(request_data, "query"):
                query_str = str(getattr(request_data, "query", ""))

            return SearchAgentResponse(
                status="ERROR",
                query=query_str,
                requested_mode=RetrievalMode.HYBRID,
                attempted_backends=[],
                successful_backends=[],
                failed_backends=[],
                returned_count=0,
                results=[],
                errors=[f"Request validation error: {exc}"],
                execution_trace=[{"action": "validate_request", "status": "FAILED", "error": str(exc)}],
            )

        # Step 2: Route by retrieval mode
        if req.retrieval_mode == RetrievalMode.KEYWORD:
            return self._execute_keyword_mode(req)
        elif req.retrieval_mode == RetrievalMode.SEMANTIC:
            return self._execute_semantic_mode(req)
        elif req.retrieval_mode == RetrievalMode.HYBRID:
            return self._execute_hybrid_mode(req)
        else:
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=[],
                successful_backends=[],
                failed_backends=[],
                returned_count=0,
                results=[],
                errors=[f"Unsupported retrieval mode: {req.retrieval_mode}"],
            )

    def _execute_keyword_mode(self, req: SearchAgentRequest) -> SearchAgentResponse:
        """Execute PostgreSQL keyword search with evidence excerpting and bounded filtering."""
        trace: List[Dict[str, Any]] = []
        errors: List[str] = []
        attempted = ["KEYWORD"]
        successful: List[str] = []
        failed: List[str] = []

        has_filters = bool(req.department_filter or req.issue_date_start or req.issue_date_end)
        # Apply bounded over-fetch when filtering is requested to reduce candidate starvation
        fetch_limit = 50 if has_filters else req.limit

        try:
            kw_res = self.search_tool.search(keyword=req.query, limit=fetch_limit)
        except Exception as exc:
            logger.error("Keyword backend exception: %s", exc, exc_info=True)
            failed.append("KEYWORD")
            errors.append("Keyword search backend encountered an operational error.")
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=successful,
                failed_backends=failed,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=[{
                    "action": "execute_keyword_search",
                    "status": "ERROR",
                    "error": "Keyword search backend encountered an operational error.",
                }],
            )

        trace.append({
            "action": "execute_keyword_search",
            "backend": "PostgreSQL SearchTool",
            "fetch_limit": fetch_limit,
            "status": kw_res.get("status"),
            "raw_returned_count": kw_res.get("returned_count", 0),
        })

        if kw_res.get("status") == "ERROR":
            logger.error("Keyword backend returned error status: %s", kw_res.get("error"))
            failed.append("KEYWORD")
            errors.append("Keyword search backend encountered an operational error.")
            trace.append({
                "action": "execute_keyword_search",
                "status": "ERROR",
                "error": "Keyword search backend encountered an operational error.",
            })
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=successful,
                failed_backends=failed,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=trace,
            )

        successful.append("KEYWORD")
        raw_items = kw_res.get("results", [])
        candidates_retrieved = len(raw_items)

        if not raw_items:
            return SearchAgentResponse(
                status="NO_MATCH",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=successful,
                failed_backends=failed,
                candidates_retrieved=0,
                candidates_retained=0,
                candidate_ceiling_reached=False,
                is_exhaustive=True,
                returned_count=0,
                results=[],
                errors=[],
                execution_trace=trace,
            )

        # Lookup DB records to get go_id, extract true text snippets, and apply date/dept filters
        go_numbers = [item["go_number"] for item in raw_items if item.get("go_number")]
        try:
            db_records = self._fetch_db_orders_by_numbers(go_numbers)
        except Exception as exc:
            logger.error("PostgreSQL lookup failed during keyword processing: %s", exc, exc_info=True)
            failed.append("KEYWORD")
            errors.append("Database query encountered an operational error.")
            trace.append({
                "action": "fetch_db_orders",
                "status": "ERROR",
                "error": "Database query encountered an operational error.",
            })
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=[],
                failed_backends=failed,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=trace,
            )

        filtered_items: List[SearchResultItem] = []
        # Rank policy: original backend position is preserved for valid results
        for orig_idx, raw_item in enumerate(raw_items, start=1):
            num = raw_item.get("go_number")
            db_order = db_records.get(num)
            if not db_order:
                continue

            # Exact department filter
            if req.department_filter:
                clean_dept = req.department_filter.strip()
                if (db_order.department or "").strip() != clean_dept:
                    continue

            # Date range filters (using actual date objects)
            if req.issue_date_start and (not db_order.issue_date or db_order.issue_date < req.issue_date_start):
                continue
            if req.issue_date_end and (not db_order.issue_date or db_order.issue_date > req.issue_date_end):
                continue

            # Extract genuine text excerpt around query if keyword appears in extracted text
            snippet = self._extract_keyword_snippet(db_order.extracted_text, req.query)

            filtered_items.append(
                SearchResultItem(
                    go_id=db_order.id,
                    go_number=db_order.go_number,
                    department=db_order.department,
                    issue_date=db_order.issue_date.isoformat() if db_order.issue_date else None,
                    source_filename=db_order.source_filename,
                    matched_sources=["KEYWORD"],
                    keyword_rank=orig_idx,  # Preserves original backend rank
                    semantic_rank=None,
                    semantic_score=None,
                    rrf_score=None,
                    text_snippet=snippet,
                    supporting_chunks=[],
                )
            )

        candidates_retained = len(filtered_items)
        fetch_limit_reached = (candidates_retrieved >= fetch_limit)
        candidate_ceiling_reached = fetch_limit_reached and (candidates_retained < req.limit)
        is_exhaustive = not fetch_limit_reached

        trace.append({
            "action": "keyword_candidate_evaluation",
            "candidates_retrieved": candidates_retrieved,
            "candidates_retained": candidates_retained,
            "fetch_limit_reached": fetch_limit_reached,
            "candidate_ceiling_reached": candidate_ceiling_reached,
            "is_exhaustive": is_exhaustive,
        })

        final_results = filtered_items[:req.limit]

        # Status resolution per specification
        if not final_results:
            final_status = "PARTIAL_FAILURE" if candidate_ceiling_reached else "NO_MATCH"
        else:
            final_status = "PARTIAL_SUCCESS" if candidate_ceiling_reached else "SUCCESS"

        return SearchAgentResponse(
            status=final_status,
            query=req.query,
            requested_mode=req.retrieval_mode,
            attempted_backends=attempted,
            successful_backends=successful,
            failed_backends=failed,
            candidates_retrieved=candidates_retrieved,
            candidates_retained=candidates_retained,
            candidate_ceiling_reached=candidate_ceiling_reached,
            is_exhaustive=is_exhaustive,
            returned_count=len(final_results),
            results=final_results,
            errors=errors,
            execution_trace=trace,
        )

    def _execute_semantic_mode(self, req: SearchAgentRequest) -> SearchAgentResponse:
        """Execute Qdrant semantic vector search with chunk grouping and PostgreSQL validation."""
        trace: List[Dict[str, Any]] = []
        errors: List[str] = []
        attempted = ["SEMANTIC"]
        successful: List[str] = []
        failed: List[str] = []

        has_date_filter = bool(req.issue_date_start or req.issue_date_end)
        chunk_fetch_limit = 50 if has_date_filter else min(50, max(req.limit * 3, 15))

        try:
            qdrant_res = self.qdrant_tool.search(
                query=req.query,
                limit=chunk_fetch_limit,
                department_filter=req.department_filter,
            )
        except Exception as exc:
            logger.error("QdrantTool exception: %s", exc, exc_info=True)
            failed.append("SEMANTIC")
            errors.append("Semantic search backend encountered an operational error.")
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=successful,
                failed_backends=failed,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=[{
                    "action": "execute_semantic_search",
                    "status": "ERROR",
                    "error": "Semantic search backend encountered an operational error.",
                }],
            )

        trace.append({
            "action": "execute_semantic_search",
            "backend": "Qdrant Vector DB",
            "chunk_fetch_limit": chunk_fetch_limit,
            "status": qdrant_res.get("status"),
            "raw_returned_count": qdrant_res.get("returned_count", 0),
        })

        if qdrant_res.get("status") == "ERROR":
            logger.error("QdrantTool returned error status: %s", qdrant_res.get("error"))
            failed.append("SEMANTIC")
            errors.append("Semantic search backend encountered an operational error.")
            trace.append({
                "action": "execute_semantic_search",
                "status": "ERROR",
                "error": "Semantic search backend encountered an operational error.",
            })
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=successful,
                failed_backends=failed,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=trace,
            )

        raw_hits = qdrant_res.get("results", [])
        candidates_retrieved = len(raw_hits)

        if not raw_hits:
            successful.append("SEMANTIC")
            return SearchAgentResponse(
                status="NO_MATCH",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=successful,
                failed_backends=failed,
                candidates_retrieved=0,
                candidates_retained=0,
                candidate_ceiling_reached=False,
                is_exhaustive=True,
                returned_count=0,
                results=[],
                errors=[],
                execution_trace=trace,
            )

        # Validate hits against PostgreSQL and group chunks by go_id
        try:
            grouped_orders, stale_count = self._validate_and_group_qdrant_hits(
                hits=raw_hits,
                issue_date_start=req.issue_date_start,
                issue_date_end=req.issue_date_end,
                trace=trace,
            )
            successful.append("SEMANTIC")
        except Exception as exc:
            logger.error("PostgreSQL validation failed during semantic search: %s", exc, exc_info=True)
            failed.append("SEMANTIC")
            errors.append("Database validation encountered an operational error.")
            trace.append({
                "action": "postgres_validation",
                "status": "ERROR",
                "error": "Database validation encountered an operational error.",
            })
            # Semantic search CANNOT report SUCCESS if PostgreSQL validation fails!
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=[],
                failed_backends=failed,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=trace,
            )

        candidates_retained = len(grouped_orders)
        fetch_limit_reached = (candidates_retrieved >= chunk_fetch_limit)
        candidate_ceiling_reached = fetch_limit_reached and (candidates_retained < req.limit)
        is_exhaustive = not fetch_limit_reached

        trace.append({
            "action": "semantic_candidate_evaluation",
            "chunk_fetch_limit": chunk_fetch_limit,
            "candidates_retrieved": candidates_retrieved,
            "candidates_retained": candidates_retained,
            "fetch_limit_reached": fetch_limit_reached,
            "candidate_ceiling_reached": candidate_ceiling_reached,
            "is_exhaustive": is_exhaustive,
        })

        if not grouped_orders:
            final_status = "PARTIAL_FAILURE" if candidate_ceiling_reached else "NO_MATCH"
            return SearchAgentResponse(
                status=final_status,
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted,
                successful_backends=successful,
                failed_backends=failed,
                candidates_retrieved=candidates_retrieved,
                candidates_retained=0,
                candidate_ceiling_reached=candidate_ceiling_reached,
                is_exhaustive=is_exhaustive,
                stale_qdrant_points_discarded=stale_count,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=trace,
            )

        # Sort grouped GOs by their best chunk score descending, break ties by go_id ascending
        sorted_gos = sorted(
            grouped_orders.values(),
            key=lambda item: (-item["best_score"], item["go_id"]),
        )

        results: List[SearchResultItem] = []
        for rank, go_info in enumerate(sorted_gos[:req.limit], start=1):
            results.append(
                SearchResultItem(
                    go_id=go_info["go_id"],
                    go_number=go_info["go_number"],
                    department=go_info["department"],
                    issue_date=go_info["issue_date"],
                    source_filename=go_info["source_filename"],
                    matched_sources=["SEMANTIC"],
                    keyword_rank=None,
                    semantic_rank=go_info["original_best_rank"],  # Preserves best chunk's original rank
                    semantic_score=go_info["best_score"],
                    rrf_score=None,
                    text_snippet=go_info["best_chunk_text"],
                    supporting_chunks=go_info["supporting_chunks"],
                )
            )

        final_status = "PARTIAL_SUCCESS" if candidate_ceiling_reached else "SUCCESS"

        return SearchAgentResponse(
            status=final_status,
            query=req.query,
            requested_mode=req.retrieval_mode,
            attempted_backends=attempted,
            successful_backends=successful,
            failed_backends=failed,
            candidates_retrieved=candidates_retrieved,
            candidates_retained=candidates_retained,
            candidate_ceiling_reached=candidate_ceiling_reached,
            is_exhaustive=is_exhaustive,
            stale_qdrant_points_discarded=stale_count,
            returned_count=len(results),
            results=results,
            errors=errors,
            execution_trace=trace,
        )

    def _execute_hybrid_mode(self, req: SearchAgentRequest) -> SearchAgentResponse:
        """
        Execute Hybrid retrieval combining keyword search and semantic search via RRF.
        Applies transparent partial failure reporting, rank preservation, and deterministic tie-breaking.
        """
        trace: List[Dict[str, Any]] = []
        errors: List[str] = []
        attempted_backends = ["KEYWORD", "SEMANTIC"]
        successful_backends: List[str] = []
        failed_backends: List[str] = []

        has_date_filter = bool(req.issue_date_start or req.issue_date_end)
        has_filters = bool(req.department_filter or has_date_filter)

        # 1. Execute Keyword Search
        kw_failed = False
        raw_kw_items: List[Dict[str, Any]] = []
        kw_fetch_limit = 50 if has_filters else max(req.limit * 2, 20)
        try:
            kw_res = self.search_tool.search(keyword=req.query, limit=kw_fetch_limit)
            if kw_res.get("status") == "ERROR":
                kw_failed = True
                logger.error("Keyword backend returned error in hybrid: %s", kw_res.get("error"))
            else:
                successful_backends.append("KEYWORD")
                raw_kw_items = kw_res.get("results", [])
        except Exception as exc:
            logger.error("Keyword backend exception in hybrid: %s", exc, exc_info=True)
            kw_failed = True

        if kw_failed:
            failed_backends.append("KEYWORD")
            errors.append("Keyword search backend encountered an operational error.")
            trace.append({
                "action": "execute_keyword_search",
                "status": "ERROR",
                "error": "Keyword search backend encountered an operational error.",
            })

        # 2. Execute Semantic Search
        sem_failed = False
        raw_sem_hits: List[Dict[str, Any]] = []
        sem_fetch_limit = 50 if has_date_filter else min(50, max(req.limit * 3, 20))
        try:
            sem_res = self.qdrant_tool.search(
                query=req.query,
                limit=sem_fetch_limit,
                department_filter=req.department_filter,
            )
            if sem_res.get("status") == "ERROR":
                sem_failed = True
                logger.error("Semantic backend returned error in hybrid: %s", sem_res.get("error"))
            else:
                successful_backends.append("SEMANTIC")
                raw_sem_hits = sem_res.get("results", [])
        except Exception as exc:
            logger.error("Semantic backend exception in hybrid: %s", exc, exc_info=True)
            sem_failed = True

        if sem_failed:
            failed_backends.append("SEMANTIC")
            errors.append("Semantic search backend encountered an operational error.")
            trace.append({
                "action": "execute_semantic_search",
                "status": "ERROR",
                "error": "Semantic search backend encountered an operational error.",
            })

        trace.append({
            "action": "execute_hybrid_backends",
            "keyword_status": "ERROR" if kw_failed else ("SUCCESS" if raw_kw_items else "NO_MATCH"),
            "semantic_status": "ERROR" if sem_failed else ("SUCCESS" if raw_sem_hits else "NO_MATCH"),
            "keyword_candidates": len(raw_kw_items),
            "semantic_candidates": len(raw_sem_hits),
        })

        # 3. Handle Complete Backend Failure
        if kw_failed and sem_failed:
            return SearchAgentResponse(
                status="ERROR",
                query=req.query,
                requested_mode=req.retrieval_mode,
                attempted_backends=attempted_backends,
                successful_backends=successful_backends,
                failed_backends=failed_backends,
                candidates_retrieved=0,
                candidates_retained=0,
                candidate_ceiling_reached=False,
                is_exhaustive=False,
                returned_count=0,
                results=[],
                errors=errors,
                execution_trace=trace,
            )

        # 4. Filter and process Keyword candidates
        kw_ranked_gos: Dict[str, Dict[str, Any]] = {}
        if raw_kw_items and not kw_failed:
            go_numbers = [item["go_number"] for item in raw_kw_items if item.get("go_number")]
            try:
                db_records = self._fetch_db_orders_by_numbers(go_numbers)
                for orig_rank, item in enumerate(raw_kw_items, start=1):
                    num = item.get("go_number")
                    db_order = db_records.get(num)
                    if not db_order:
                        continue
                    if req.department_filter and (db_order.department or "").strip() != req.department_filter.strip():
                        continue
                    if req.issue_date_start and (not db_order.issue_date or db_order.issue_date < req.issue_date_start):
                        continue
                    if req.issue_date_end and (not db_order.issue_date or db_order.issue_date > req.issue_date_end):
                        continue

                    snippet = self._extract_keyword_snippet(db_order.extracted_text, req.query)
                    kw_ranked_gos[db_order.go_number] = {
                        "go_id": db_order.id,
                        "go_number": db_order.go_number,
                        "department": db_order.department,
                        "issue_date": db_order.issue_date.isoformat() if db_order.issue_date else None,
                        "source_filename": db_order.source_filename,
                        "keyword_rank": orig_rank,  # Preserves original position
                        "text_snippet": snippet,
                    }
            except Exception as exc:
                logger.error("PostgreSQL error in hybrid keyword processing: %s", exc, exc_info=True)
                if "KEYWORD" in successful_backends:
                    successful_backends.remove("KEYWORD")
                if "KEYWORD" not in failed_backends:
                    failed_backends.append("KEYWORD")
                errors.append("Database query encountered an operational error.")
                trace.append({
                    "action": "fetch_db_orders",
                    "status": "ERROR",
                    "error": "Database query encountered an operational error.",
                })

        # 5. Process and validate Semantic candidates
        sem_ranked_gos: Dict[str, Dict[str, Any]] = {}
        stale_discarded_count = 0
        if raw_sem_hits and not sem_failed:
            try:
                grouped_sem, stale_discarded_count = self._validate_and_group_qdrant_hits(
                    hits=raw_sem_hits,
                    issue_date_start=req.issue_date_start,
                    issue_date_end=req.issue_date_end,
                    trace=trace,
                )
                for go_id, go_info in grouped_sem.items():
                    sem_ranked_gos[go_info["go_number"]] = {
                        "go_id": go_info["go_id"],
                        "go_number": go_info["go_number"],
                        "department": go_info["department"],
                        "issue_date": go_info["issue_date"],
                        "source_filename": go_info["source_filename"],
                        "semantic_rank": go_info["original_best_rank"],  # Preserves original rank
                        "semantic_score": go_info["best_score"],
                        "text_snippet": go_info["best_chunk_text"],
                        "supporting_chunks": go_info["supporting_chunks"],
                    }
            except Exception as exc:
                logger.error("PostgreSQL validation error in hybrid semantic processing: %s", exc, exc_info=True)
                if "SEMANTIC" in successful_backends:
                    successful_backends.remove("SEMANTIC")
                if "SEMANTIC" not in failed_backends:
                    failed_backends.append("SEMANTIC")
                errors.append("Database validation encountered an operational error.")
                trace.append({
                    "action": "postgres_validation",
                    "status": "ERROR",
                    "error": "Database validation encountered an operational error.",
                })

        # 6. Apply Reciprocal Rank Fusion (RRF)
        all_go_numbers = set(kw_ranked_gos.keys()) | set(sem_ranked_gos.keys())
        k = req.rrf_k
        fused_items: List[SearchResultItem] = []

        for go_num in all_go_numbers:
            kw_entry = kw_ranked_gos.get(go_num)
            sem_entry = sem_ranked_gos.get(go_num)

            matched_sources: List[str] = []
            rrf_score = 0.0

            if kw_entry:
                matched_sources.append("KEYWORD")
                rrf_score += 1.0 / (k + kw_entry["keyword_rank"])

            if sem_entry:
                matched_sources.append("SEMANTIC")
                rrf_score += 1.0 / (k + sem_entry["semantic_rank"])

            base_entry = sem_entry or kw_entry
            assert base_entry is not None

            snippet = None
            if sem_entry and sem_entry.get("text_snippet"):
                snippet = sem_entry["text_snippet"]
            elif kw_entry and kw_entry.get("text_snippet"):
                snippet = kw_entry["text_snippet"]

            fused_items.append(
                SearchResultItem(
                    go_id=base_entry["go_id"],
                    go_number=go_num,
                    department=base_entry["department"],
                    issue_date=base_entry["issue_date"],
                    source_filename=base_entry["source_filename"],
                    matched_sources=matched_sources,
                    keyword_rank=kw_entry["keyword_rank"] if kw_entry else None,
                    semantic_rank=sem_entry["semantic_rank"] if sem_entry else None,
                    semantic_score=sem_entry["semantic_score"] if sem_entry else None,
                    rrf_score=rrf_score,
                    text_snippet=snippet,
                    supporting_chunks=sem_entry["supporting_chunks"] if sem_entry else [],
                )
            )

        # 7. Deterministic Sorting:
        # Priority 1: RRF score descending
        # Priority 2: Multi-source matches preferred (len(matched_sources) descending)
        # Priority 3: Semantic similarity score descending
        # Priority 4: go_id ascending (stable deterministic ground truth)
        fused_items.sort(
            key=lambda item: (
                -(item.rrf_score or 0.0),
                -len(item.matched_sources),
                -(item.semantic_score or 0.0),
                (item.go_id or 0),
            )
        )

        final_results = fused_items[:req.limit]

        # Candidate completeness calculation
        candidates_retrieved = len(raw_kw_items) + len(raw_sem_hits)
        candidates_retained = len(fused_items)
        kw_fetch_reached = (len(raw_kw_items) >= kw_fetch_limit) if raw_kw_items else False
        sem_fetch_reached = (len(raw_sem_hits) >= sem_fetch_limit) if raw_sem_hits else False
        fetch_limit_reached = kw_fetch_reached or sem_fetch_reached
        candidate_ceiling_reached = fetch_limit_reached and (candidates_retained < req.limit)
        is_exhaustive = not fetch_limit_reached

        trace.append({
            "action": "hybrid_candidate_evaluation",
            "candidates_retrieved": candidates_retrieved,
            "candidates_retained": candidates_retained,
            "fetch_limit_reached": fetch_limit_reached,
            "candidate_ceiling_reached": candidate_ceiling_reached,
            "is_exhaustive": is_exhaustive,
        })

        # 8. Consistent Status Resolution per Table:
        # Both succeed w/ results -> SUCCESS
        # Both succeed w/ no results -> NO_MATCH
        # One fails, one returns results -> PARTIAL_SUCCESS
        # One fails, one returns 0 results -> PARTIAL_FAILURE
        # Both fail -> ERROR
        # Candidate ceiling reached -> PARTIAL_SUCCESS (w/ results) or PARTIAL_FAILURE (w/o results)
        has_failure = bool(failed_backends)
        if has_failure:
            if final_results:
                final_status = "PARTIAL_SUCCESS"
            else:
                final_status = "PARTIAL_FAILURE"
        else:
            if candidate_ceiling_reached:
                final_status = "PARTIAL_SUCCESS" if final_results else "PARTIAL_FAILURE"
            else:
                final_status = "SUCCESS" if final_results else "NO_MATCH"

        return SearchAgentResponse(
            status=final_status,
            query=req.query,
            requested_mode=req.retrieval_mode,
            attempted_backends=attempted_backends,
            successful_backends=successful_backends,
            failed_backends=failed_backends,
            candidates_retrieved=candidates_retrieved,
            candidates_retained=candidates_retained,
            candidate_ceiling_reached=candidate_ceiling_reached,
            is_exhaustive=is_exhaustive,
            stale_qdrant_points_discarded=stale_discarded_count,
            returned_count=len(final_results),
            results=final_results,
            errors=errors,
            execution_trace=trace,
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _fetch_db_orders_by_numbers(self, go_numbers: List[str]) -> Dict[str, GovernmentOrderDB]:
        """Fetch PostgreSQL GovernmentOrderDB objects by go_number."""
        if not go_numbers:
            return {}
        with SessionLocal() as session:
            statement = select(GovernmentOrderDB).where(GovernmentOrderDB.go_number.in_(go_numbers))
            return {order.go_number: order for order in session.scalars(statement).all()}

    def _validate_and_group_qdrant_hits(
        self,
        hits: List[Dict[str, Any]],
        issue_date_start: Optional[date],
        issue_date_end: Optional[date],
        trace: List[Dict[str, Any]],
    ) -> tuple[Dict[int, Dict[str, Any]], int]:
        """
        Validate Qdrant hits against PostgreSQL to discard stale vector points,
        apply date bounds, and group multiple chunks under each unique Government Order.
        Preserves original 1-based backend rank of the best chunk and chunk_index=0.
        """
        if not hits:
            return {}, 0

        valid_dict_hits = [h for h in hits if isinstance(h, dict)]
        if not valid_dict_hits:
            return {}, 0

        raw_go_ids = {h.get("go_id") for h in valid_dict_hits if h.get("go_id") is not None}
        if not raw_go_ids:
            return {}, 0

        with SessionLocal() as session:
            stmt = select(GovernmentOrderDB).where(GovernmentOrderDB.id.in_(raw_go_ids))
            db_records = {order.id: order for order in session.scalars(stmt).all()}

        stale_discarded = 0
        grouped: Dict[int, Dict[str, Any]] = {}

        for orig_rank, hit in enumerate(valid_dict_hits, start=1):
            gid = hit.get("go_id")
            db_order = db_records.get(gid)
            if not db_order:
                stale_discarded += 1
                continue

            # Date range filtering against authoritative PostgreSQL date
            if issue_date_start and (not db_order.issue_date or db_order.issue_date < issue_date_start):
                continue
            if issue_date_end and (not db_order.issue_date or db_order.issue_date > issue_date_end):
                continue

            try:
                chunk_score = float(hit.get("score") or 0.0)
            except (ValueError, TypeError):
                chunk_score = 0.0

            raw_c_idx = hit.get("chunk_index")
            # Preserve chunk_index=0 cleanly!
            safe_c_idx = 0 if raw_c_idx == 0 else (raw_c_idx if raw_c_idx is not None else 1)
            try:
                safe_c_idx = int(safe_c_idx)
            except (ValueError, TypeError):
                safe_c_idx = 1

            chunk_obj = SupportingChunk(
                chunk_index=safe_c_idx,
                chunk_text=hit.get("chunk_text") or "",
                page_number=hit.get("page_number"),
                score=chunk_score,
            )

            if gid not in grouped:
                grouped[gid] = {
                    "go_id": db_order.id,
                    "go_number": db_order.go_number,
                    "department": db_order.department,
                    "issue_date": db_order.issue_date.isoformat() if db_order.issue_date else None,
                    "source_filename": db_order.source_filename,
                    "best_score": chunk_score,
                    "best_chunk_text": chunk_obj.chunk_text,
                    "original_best_rank": orig_rank,  # First time this GO appears is its best rank in raw hits
                    "supporting_chunks": [chunk_obj],
                }
            else:
                grouped[gid]["supporting_chunks"].append(chunk_obj)
                if chunk_score > grouped[gid]["best_score"]:
                    grouped[gid]["best_score"] = chunk_score
                    grouped[gid]["best_chunk_text"] = chunk_obj.chunk_text

        if stale_discarded > 0:
            trace.append({
                "action": "postgres_validation",
                "stale_qdrant_points_discarded": stale_discarded,
            })

        return grouped, stale_discarded

    def _extract_keyword_snippet(self, full_text: Optional[str], keyword: str) -> Optional[str]:
        """
        Extract a genuine ~180-character excerpt window around the first occurrence
        of the keyword in the extracted text. Returns None if the keyword does not appear.
        """
        if not full_text or not keyword:
            return None

        clean_text = full_text.strip()
        lower_text = clean_text.lower()
        lower_kw = keyword.strip().lower()

        idx = lower_text.find(lower_kw)
        if idx == -1:
            return None

        window_radius = 90
        start = max(0, idx - window_radius)
        end = min(len(clean_text), idx + len(keyword) + window_radius)

        snippet = clean_text[start:end].replace("\n", " ").strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(clean_text) else ""

        return f"{prefix}{snippet}{suffix}"
