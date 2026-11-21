"""
Relationship Tool Adapter.

Wraps the existing rule-based RelationshipExtractionService to extract
relationship signals, evidence sentences, and numbered references from
Kerala Government Order text.
Distinguishes unverified extraction candidates from legally verified relationships.
"""

import logging
from typing import Any, Dict, List, Optional
from app.services.relationship_extraction_service import RelationshipExtractionService
from app.agents.tools.database_tool import DatabaseTool

logger = logging.getLogger(__name__)


class RelationshipTool:
    """
    Adapter tool for deterministic rule-based relationship and reference extraction.
    Reuses existing RelationshipExtractionService.
    """

    def __init__(
        self,
        relationship_service: Optional[RelationshipExtractionService] = None,
        database_tool: Optional[DatabaseTool] = None,
    ) -> None:
        """
        Initialize the relationship tool adapter.
        """
        self.service = relationship_service or RelationshipExtractionService()
        self.db_tool = database_tool or DatabaseTool()

    def extract_from_text(
        self,
        text: str,
        source_go_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract relationships and references from raw Government Order text.
        Tags extracted relationships as unverified candidates requiring evidence verification.

        Args:
            text: Extracted text of the Government Order.
            source_go_number: Optional identifier of the source GO for context.

        Returns:
            Structured dictionary containing references, relationships, evidence, and status.
        """
        cleaned_text = text.strip() if text else ""
        if not cleaned_text:
            return {
                "status": "EMPTY_TEXT",
                "source_go_number": source_go_number,
                "total_references": 0,
                "references": [],
                "total_relationships": 0,
                "relationships": [],
                "error": "Input text cannot be empty.",
            }

        try:
            # 1. Extract numbered references from "Read" section
            raw_references = self.service.extract_references(cleaned_text)

            # 2. Extract relationship patterns and evidence sentences
            raw_relationships = self.service.extract_relationships(cleaned_text)

            # 3. Resolve ordinals (e.g. "1st paper") to actual referenced GO numbers
            resolved_relationships = self.service.resolve_referenced_go_numbers(
                relationships=raw_relationships,
                references=raw_references,
            )

            # Format references with full evidence preservation
            formatted_refs = []
            for ref in raw_references:
                formatted_refs.append(
                    {
                        "ordinal": ref.ordinal,
                        "go_number": ref.go_number,
                        "issue_date": ref.issue_date,
                        "page_number": ref.page_number,
                        "evidence_text": ref.evidence_text,
                    }
                )

            # Format relationships with clear unverified candidate status
            formatted_rels = []
            for rel in resolved_relationships:
                formatted_rels.append(
                    {
                        "relationship_type": rel.relationship_type.value,
                        "matched_phrase": rel.matched_phrase,
                        "evidence_text": rel.evidence_text,
                        "page_number": rel.page_number,
                        "reference_ordinals": rel.reference_ordinals,
                        "referenced_go_numbers": rel.referenced_go_numbers,
                        "confidence": rel.confidence.value,
                        "extraction_method": rel.extraction_method,
                        "review_required": rel.review_required,
                        # Clarification tags: clearly distinguish rule extraction from legal verification
                        "verification_status": "UNVERIFIED_CANDIDATE",
                        "legal_effect_verified": False,
                    }
                )

            return {
                "status": "SUCCESS",
                "source_go_number": source_go_number,
                "total_references": len(formatted_refs),
                "references": formatted_refs,
                "total_relationships": len(formatted_rels),
                "relationships": formatted_rels,
                "error": None,
            }

        except Exception as exc:
            logger.error("Relationship extraction failed for '%s': %s", source_go_number, exc, exc_info=True)
            return {
                "status": "ERROR",
                "source_go_number": source_go_number,
                "total_references": 0,
                "references": [],
                "total_relationships": 0,
                "relationships": [],
                "error": "Relationship extraction operation could not be completed.",
            }

    def extract_for_order(self, go_number: str) -> Dict[str, Any]:
        """
        Convenience method: Fetches order text from PostgreSQL, then extracts
        all references and relationships. Safely handles missing or ambiguous source orders.

        Args:
            go_number: The Government Order number to analyze.

        Returns:
            Structured dictionary containing extracted references, relationships, and evidence.
        """
        order_res = self.db_tool.get_order_by_number(go_number, include_full_text=True)

        if not order_res.get("found"):
            # Distinct handling for ambiguity vs not found
            status = (
                "AMBIGUOUS_SOURCE_ORDER"
                if order_res.get("status") == "AMBIGUOUS_MATCH"
                else "ORDER_NOT_FOUND"
            )
            return {
                "status": status,
                "source_go_number": go_number,
                "candidates": order_res.get("candidates", []),
                "total_references": 0,
                "references": [],
                "total_relationships": 0,
                "relationships": [],
                "error": order_res.get("error") or f"Order '{go_number}' could not be uniquely resolved.",
            }

        order_data = order_res.get("data", {})
        text = order_data.get("extracted_text", "")
        official_go_number = order_data.get("go_number", go_number)

        return self.extract_from_text(
            text=text,
            source_go_number=official_go_number,
        )
