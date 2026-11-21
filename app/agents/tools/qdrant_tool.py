"""
Qdrant Tool Adapter.

Provides a structured adapter wrapping QdrantService and EmbeddingService
for semantic vector retrieval across indexed Kerala Government Orders.
Searches only active, reconciled document chunks with dense BGE embeddings.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from app.config import EMBEDDING_DIMENSION, QDRANT_COLLECTION_NAME, QDRANT_STORAGE_PATH
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class QdrantTool:
    """
    Adapter tool for executing semantic vector search in Qdrant.
    Reuses existing EmbeddingService and QdrantService singletons.
    """

    def __init__(
        self,
        qdrant_service: Optional[QdrantService] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        """
        Initialize the Qdrant tool adapter.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService(
            storage_path=QDRANT_STORAGE_PATH,
            collection_name=QDRANT_COLLECTION_NAME,
            dimension=EMBEDDING_DIMENSION,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        department_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute semantic vector search for text chunks relevant to the query.

        Args:
            query: Natural language query or search statement.
            limit: Maximum number of points/chunks to return (1 to 50).
            department_filter: Optional exact department name for native Qdrant filtering.
            min_score: Optional minimum cosine similarity threshold in [-1.0, 1.0].

        Returns:
            Structured dictionary containing status, query, returned_count, limit_applied,
            department_filter, results, and error details.
        """
        # A1. Input Validation: Query
        if not isinstance(query, str) or not query.strip():
            return {
                "status": "EMPTY_QUERY",
                "query": query if isinstance(query, str) else "",
                "returned_count": 0,
                "limit_applied": limit if isinstance(limit, int) and not isinstance(limit, bool) else 0,
                "department_filter": None,
                "results": [],
                "error": "Search query cannot be empty or whitespace.",
            }
        cleaned_query = query.strip()

        # A1. Input Validation: Limit (reject bools, floats, non-ints, 0, > 50)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 50:
            return {
                "status": "ERROR",
                "query": cleaned_query,
                "returned_count": 0,
                "limit_applied": limit,
                "department_filter": None,
                "results": [],
                "error": f"Search limit must be an integer between 1 and 50 (got {repr(limit)}).",
            }

        # A1. Input Validation: Department filter
        normalized_dept: Optional[str] = None
        if department_filter is not None:
            if not isinstance(department_filter, str) or not department_filter.strip():
                return {
                    "status": "ERROR",
                    "query": cleaned_query,
                    "returned_count": 0,
                    "limit_applied": limit,
                    "department_filter": department_filter,
                    "results": [],
                    "error": "Department filter must be a non-empty string or None.",
                }
            normalized_dept = department_filter.strip()

        # A1. Input Validation: min_score
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            return {
                "status": "ERROR",
                "query": cleaned_query,
                "returned_count": 0,
                "limit_applied": limit,
                "department_filter": normalized_dept,
                "results": [],
                "error": f"min_score must be a finite numeric value (got {repr(min_score)}).",
            }
        if math.isnan(min_score) or math.isinf(min_score):
            return {
                "status": "ERROR",
                "query": cleaned_query,
                "returned_count": 0,
                "limit_applied": limit,
                "department_filter": normalized_dept,
                "results": [],
                "error": "min_score must be a finite numeric value and cannot be NaN or infinity.",
            }
        if min_score < -1.0 or min_score > 1.0:
            return {
                "status": "ERROR",
                "query": cleaned_query,
                "returned_count": 0,
                "limit_applied": limit,
                "department_filter": normalized_dept,
                "results": [],
                "error": f"min_score must be within the cosine similarity range [-1.0, 1.0] (got {min_score}).",
            }

        try:
            # A2. Generate query embedding via existing EmbeddingService (handles BGE prefix once)
            query_vector = self.embedding_service.embed_query(cleaned_query)

            # A3. Search active chunks in Qdrant with native department filtering
            raw_hits: List[Dict[str, Any]] = self.qdrant_service.search_similar(
                query_vector=query_vector,
                limit=limit,
                department_filter=normalized_dept,
                min_score=float(min_score),
            )

            if not raw_hits:
                return {
                    "status": "NO_MATCH",
                    "query": cleaned_query,
                    "returned_count": 0,
                    "limit_applied": limit,
                    "department_filter": normalized_dept,
                    "results": [],
                    "error": None,
                }

            # A4. Validate structure of returned Qdrant hits before reporting SUCCESS
            validated_hits: List[Dict[str, Any]] = []
            for hit in raw_hits:
                if not isinstance(hit, dict):
                    logger.warning("QdrantTool: Skipped malformed non-dict hit: %s", hit)
                    continue

                # Ensure required keys exist; explicitly preserve chunk_index=0!
                chunk_idx = hit.get("chunk_index")
                if chunk_idx is not None and not isinstance(chunk_idx, int):
                    try:
                        chunk_idx = int(chunk_idx)
                    except (ValueError, TypeError):
                        chunk_idx = None

                validated_hits.append(
                    {
                        "point_id": str(hit.get("point_id", "")),
                        "score": float(hit.get("score") or 0.0),
                        "go_id": hit.get("go_id"),
                        "go_number": hit.get("go_number"),
                        "department": hit.get("department"),
                        "source_filename": hit.get("source_filename"),
                        "issue_date": hit.get("issue_date"),
                        "chunk_index": chunk_idx,  # Preserves 0 cleanly
                        "chunk_text": str(hit.get("chunk_text") or ""),
                        "page_number": hit.get("page_number"),
                        "version": hit.get("version"),
                    }
                )

            if not validated_hits:
                return {
                    "status": "NO_MATCH",
                    "query": cleaned_query,
                    "returned_count": 0,
                    "limit_applied": limit,
                    "department_filter": normalized_dept,
                    "results": [],
                    "error": None,
                }

            return {
                "status": "SUCCESS",
                "query": cleaned_query,
                "returned_count": len(validated_hits),
                "limit_applied": limit,
                "department_filter": normalized_dept,
                "results": validated_hits,
                "error": None,
            }

        except Exception as exc:
            # A4. Log detailed internal exception; return sanitized error without paths/credentials
            logger.error("QdrantTool search error for query '%s': %s", cleaned_query, exc, exc_info=True)
            return {
                "status": "ERROR",
                "query": cleaned_query,
                "returned_count": 0,
                "limit_applied": limit,
                "department_filter": normalized_dept,
                "results": [],
                "error": "Semantic vector search service encountered an internal error.",
            }
