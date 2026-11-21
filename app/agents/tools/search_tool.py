"""
Search Tool Adapter.

Wraps the existing SearchService to provide safe, structured keyword and
department searching across Kerala Government Orders stored in PostgreSQL.
Preserves existing SearchService without modification.
"""

import logging
from typing import Any, Dict, List, Optional
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


class SearchTool:
    """
    Adapter tool for executing keyword searches against the Government Orders database.
    Reuses existing SearchService.
    """

    def __init__(self, search_service: Optional[SearchService] = None) -> None:
        """
        Initialize the search tool adapter.
        """
        self.service = search_service or SearchService()

    def search(self, keyword: str, limit: int = 10) -> Dict[str, Any]:
        """
        Execute a search for orders matching the keyword.

        Note on counts:
            The fields 'returned_count' and 'total_found' represent the number of
            records returned within the current query limit ('limit_applied'),
            not the total number of matching records in the entire database.

        Args:
            keyword: The search phrase (e.g. GO number, department, topic).
            limit: Maximum number of records to return (default: 10, max: 50).

        Returns:
            Structured dictionary containing status, query, returned_count, total_found,
            limit_applied, results, and error.
        """
        cleaned_keyword = keyword.strip() if keyword else ""

        if not cleaned_keyword:
            return {
                "status": "EMPTY_QUERY",
                "query": keyword,
                "returned_count": 0,
                "total_found": 0,
                "limit_applied": limit,
                "results": [],
                "error": "Search keyword cannot be empty.",
            }

        safe_limit = max(1, min(limit, 50))

        try:
            raw_results: List[Dict[str, Any]] = self.service.search_orders(
                keyword=cleaned_keyword,
                limit=safe_limit,
            )

            if not raw_results:
                return {
                    "status": "NO_MATCH",
                    "query": cleaned_keyword,
                    "returned_count": 0,
                    "total_found": 0,
                    "limit_applied": safe_limit,
                    "results": [],
                    "error": None,
                }

            formatted_results = []
            for item in raw_results:
                formatted_results.append(
                    {
                        "go_number": item.get("go_number"),
                        "issue_date": item.get("issue_date"),
                        "department": item.get("department"),
                        "source_filename": item.get("source_filename"),
                        "verification_status": item.get("verification_status"),
                    }
                )

            count = len(formatted_results)
            return {
                "status": "SUCCESS",
                "query": cleaned_keyword,
                "returned_count": count,
                "total_found": count,  # Backward-compatible alias
                "limit_applied": safe_limit,
                "results": formatted_results,
                "error": None,
            }

        except Exception as exc:
            logger.error("Search failed for keyword '%s': %s", cleaned_keyword, exc, exc_info=True)
            return {
                "status": "ERROR",
                "query": cleaned_keyword,
                "returned_count": 0,
                "total_found": 0,
                "limit_applied": safe_limit,
                "results": [],
                "error": "Search operation could not be completed due to a database error.",
            }
