"""
Database Tool Adapter.

Provides direct, safe read-only queries against PostgreSQL to retrieve verified
Government Orders, references, relationship records, and extracted text.
Supports exact matching with priority, partial-match ambiguity detection,
and bidirectional (incoming and outgoing) relationship inspection.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select, or_

from app.database.connection import SessionLocal
from app.database.models import (
    GovernmentOrderDB,
    GOReferenceDB,
    GORelationshipDB,
    VerificationRecordDB,
)

logger = logging.getLogger(__name__)


class DatabaseTool:
    """
    Adapter tool for deterministic database lookups in PostgreSQL.
    """

    def __init__(self) -> None:
        pass

    def list_all_go_numbers(self) -> Dict[str, Any]:
        """
        Retrieve all Government Order numbers registered in the database.

        Returns:
            Dictionary with status, count, and list of all GO numbers.
        """
        try:
            with SessionLocal() as session:
                statement = select(GovernmentOrderDB.go_number).order_by(GovernmentOrderDB.id)
                numbers = list(session.scalars(statement).all())

                return {
                    "status": "SUCCESS",
                    "total_count": len(numbers),
                    "go_numbers": numbers,
                    "error": None,
                }
        except Exception as exc:
            logger.error("Failed to list GO numbers: %s", exc, exc_info=True)
            return {
                "status": "ERROR",
                "total_count": 0,
                "go_numbers": [],
                "error": "Database operation could not be completed due to an internal error.",
            }

    @staticmethod
    def _parse_go_structure(identifier: Optional[str]) -> Optional[Tuple[str, str, str]]:
        """
        Extract (number, year, department) from a structurally valid GO identifier.
        Rejects incomplete fragments like '24', 'ITD', 'G.O.'.
        """
        if not identifier:
            return None
        clean = re.sub(r"\s+", "", str(identifier).strip().upper())
        pattern = re.compile(
            r"^(?:G\.O\.?(?:\([A-Za-z]+\))?(?:NO\.?)?)?(\d+)/(\d{4})(?:/([A-Za-z0-9&.-]+))?$",
            re.IGNORECASE,
        )
        m = pattern.match(clean)
        if not m:
            return None
        num = m.group(1)
        year = m.group(2)
        dept = (m.group(3) or "").strip().upper()
        return num, year, dept

    @classmethod
    def _are_structurally_equivalent(cls, id1: Optional[str], id2: Optional[str]) -> bool:
        """
        Check whether two GO identifiers are structurally equivalent complete orders.
        Requires identical number, year, and department code.
        """
        p1 = cls._parse_go_structure(id1)
        p2 = cls._parse_go_structure(id2)
        if not p1 or not p2:
            return False
        num1, year1, dept1 = p1
        num2, year2, dept2 = p2
        if num1 != num2 or year1 != year2:
            return False
        if dept1 != dept2:
            return False
        return True

    def _resolve_order(self, session, query: str) -> Dict[str, Any]:
        """
        Internal helper to resolve a GO query using exact match priority
        and explicit ambiguity detection.

        Returns:
            dict with:
                - 'status': SUCCESS, AMBIGUOUS_MATCH, NOT_FOUND, or EMPTY_QUERY
                - 'order': GovernmentOrderDB instance if uniquely resolved, else None
                - 'match_type': EXACT, PARTIAL, or None
                - 'candidates': list of candidate go_numbers if ambiguous
                - 'error': error message if any
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            return {
                "status": "EMPTY_QUERY",
                "order": None,
                "match_type": None,
                "candidates": [],
                "error": "GO number cannot be empty.",
            }

        # 1. Priority 1: Exact raw-string match
        exact_stmt = select(GovernmentOrderDB).where(
            GovernmentOrderDB.go_number == cleaned_query
        )
        exact_order = session.scalar(exact_stmt)
        if exact_order:
            return {
                "status": "SUCCESS",
                "order": exact_order,
                "match_type": "EXACT",
                "candidates": [exact_order.go_number],
                "error": None,
            }

        # 1B. Priority 1B: Exact structural equivalence match (e.g. '24/2026/ITD' vs 'G.O.(Rt)No.24/2026/ITD')
        parsed_q = self._parse_go_structure(cleaned_query)
        if parsed_q:
            q_num, q_year, _ = parsed_q
            struct_stmt = select(GovernmentOrderDB).where(
                GovernmentOrderDB.go_number.ilike(f"%{q_num}/{q_year}%")
            ).order_by(GovernmentOrderDB.id)
            struct_candidates = session.scalars(struct_stmt).all()
            exact_struct_matches = [
                c for c in struct_candidates if self._are_structurally_equivalent(cleaned_query, c.go_number)
            ]
            if len(exact_struct_matches) == 1:
                return {
                    "status": "SUCCESS",
                    "order": exact_struct_matches[0],
                    "match_type": "EXACT",
                    "candidates": [exact_struct_matches[0].go_number],
                    "error": None,
                }
            elif len(exact_struct_matches) > 1:
                candidate_numbers = [c.go_number for c in exact_struct_matches]
                return {
                    "status": "AMBIGUOUS_MATCH",
                    "order": None,
                    "match_type": None,
                    "candidates": candidate_numbers,
                    "error": (
                        f"Query '{cleaned_query}' is ambiguous. "
                        f"Found {len(candidate_numbers)} matching orders: {candidate_numbers}"
                    ),
                }

        # 2. Priority 2: Case-insensitive partial match
        partial_stmt = select(GovernmentOrderDB).where(
            GovernmentOrderDB.go_number.ilike(f"%{cleaned_query}%")
        ).order_by(GovernmentOrderDB.id)
        candidates = session.scalars(partial_stmt).all()

        if len(candidates) == 1:
            m_type = "EXACT" if self._are_structurally_equivalent(cleaned_query, candidates[0].go_number) else "PARTIAL"
            return {
                "status": "SUCCESS",
                "order": candidates[0],
                "match_type": m_type,
                "candidates": [candidates[0].go_number],
                "error": None,
            }
        elif len(candidates) > 1:
            candidate_numbers = [c.go_number for c in candidates]
            return {
                "status": "AMBIGUOUS_MATCH",
                "order": None,
                "match_type": None,
                "candidates": candidate_numbers,
                "error": (
                    f"Query '{cleaned_query}' is ambiguous. "
                    f"Found {len(candidate_numbers)} matching orders: {candidate_numbers}"
                ),
            }
        else:
            return {
                "status": "NOT_FOUND",
                "order": None,
                "match_type": None,
                "candidates": [],
                "error": f"No Government Order found matching '{cleaned_query}'.",
            }

    def get_order_by_number(
        self,
        go_number: str,
        include_full_text: bool = True,
        text_preview_chars: int = 500,
    ) -> Dict[str, Any]:
        """
        Retrieve details and document text for a specific Government Order.
        Prioritizes exact match; detects and reports ambiguous partial matches.

        Args:
            go_number: The GO number or identifier to look up.
            include_full_text: If True, returns the complete extracted text.
            text_preview_chars: Length of snippet preview text to include.

        Returns:
            Dictionary containing order metadata, text, match_type, and status.
        """
        query = go_number.strip() if go_number else ""
        if not query:
            return {
                "status": "EMPTY_QUERY",
                "found": False,
                "query": query,
                "match_type": None,
                "data": None,
                "error": "GO number cannot be empty.",
            }

        try:
            with SessionLocal() as session:
                resolution = self._resolve_order(session, query)

                if resolution["status"] != "SUCCESS":
                    return {
                        "status": resolution["status"],
                        "found": False,
                        "query": query,
                        "match_type": resolution["match_type"],
                        "candidates": resolution["candidates"],
                        "data": None,
                        "error": resolution["error"],
                    }

                order = resolution["order"]
                full_text = order.extracted_text or ""
                preview = full_text[:text_preview_chars] if full_text else ""

                data_payload: Dict[str, Any] = {
                    "id": order.id,
                    "go_number": order.go_number,
                    "issue_date": order.issue_date.isoformat() if order.issue_date else None,
                    "department": order.department,
                    "source_filename": order.source_filename,
                    "extracted_text_filename": order.extracted_text_filename,
                    "verification_status": order.verification_status,
                    "extracted_text_length": len(full_text),
                    "text_preview": preview,
                }

                if include_full_text:
                    data_payload["extracted_text"] = full_text

                return {
                    "status": "SUCCESS",
                    "found": True,
                    "query": query,
                    "match_type": resolution["match_type"],
                    "candidates": resolution["candidates"],
                    "data": data_payload,
                    "error": None,
                }

        except Exception as exc:
            logger.error("Failed to get order details for '%s': %s", query, exc, exc_info=True)
            return {
                "status": "ERROR",
                "found": False,
                "query": query,
                "match_type": None,
                "data": None,
                "error": "Database operation could not be completed due to an internal error.",
            }

    def get_references_for_order(self, go_number: str) -> Dict[str, Any]:
        """
        Retrieve all numbered references ('Read' section papers) for a given GO.

        Args:
            go_number: Source Government Order number.

        Returns:
            Dictionary with status, count, and list of reference records.
        """
        query = go_number.strip() if go_number else ""
        if not query:
            return {
                "status": "EMPTY_QUERY",
                "source_go_number": query,
                "references": [],
                "error": "GO number cannot be empty.",
            }

        try:
            with SessionLocal() as session:
                resolution = self._resolve_order(session, query)

                if resolution["status"] != "SUCCESS":
                    return {
                        "status": resolution["status"],
                        "source_go_number": query,
                        "candidates": resolution["candidates"],
                        "references": [],
                        "error": resolution["error"],
                    }

                order = resolution["order"]
                ref_statement = (
                    select(GOReferenceDB)
                    .where(GOReferenceDB.source_go_id == order.id)
                    .order_by(GOReferenceDB.ordinal)
                )

                references = session.scalars(ref_statement).all()

                formatted_refs = []
                for ref in references:
                    formatted_refs.append(
                        {
                            "id": ref.id,
                            "ordinal": ref.ordinal,
                            "referenced_go_number": ref.referenced_go_number,
                            "reference_date": ref.reference_date.isoformat() if ref.reference_date else None,
                            "page_number": ref.page_number,
                            "evidence_text": ref.evidence_text,
                        }
                    )

                return {
                    "status": "SUCCESS",
                    "source_go_id": order.id,
                    "source_go_number": order.go_number,
                    "match_type": resolution["match_type"],
                    "total_references": len(formatted_refs),
                    "references": formatted_refs,
                    "error": None,
                }

        except Exception as exc:
            logger.error("Failed to retrieve references for '%s': %s", query, exc, exc_info=True)
            return {
                "status": "ERROR",
                "source_go_number": query,
                "references": [],
                "error": "Database operation could not be completed due to an internal error.",
            }

    def get_outgoing_relationships(self, go_number: str) -> Dict[str, Any]:
        """
        Retrieve outgoing relationships: Which earlier orders does this GO
        modify, amend, cancel, or supersede?

        Args:
            go_number: Source Government Order number.

        Returns:
            Dictionary containing list of outgoing relationship records with evidence.
        """
        query = go_number.strip() if go_number else ""
        if not query:
            return {
                "status": "EMPTY_QUERY",
                "source_go_number": query,
                "direction": "OUTGOING",
                "total_relationships": 0,
                "relationships": [],
                "error": "GO number cannot be empty.",
            }

        try:
            with SessionLocal() as session:
                resolution = self._resolve_order(session, query)

                if resolution["status"] != "SUCCESS":
                    return {
                        "status": resolution["status"],
                        "source_go_number": query,
                        "direction": "OUTGOING",
                        "candidates": resolution["candidates"],
                        "total_relationships": 0,
                        "relationships": [],
                        "error": resolution["error"],
                    }

                order = resolution["order"]
                statement = (
                    select(GORelationshipDB)
                    .where(GORelationshipDB.source_go_id == order.id)
                    .order_by(GORelationshipDB.id)
                )
                relationships = session.scalars(statement).all()

                formatted_rels = []
                for rel in relationships:
                    formatted_rels.append(
                        {
                            "id": rel.id,
                            "direction": "OUTGOING",
                            "source_go_id": order.id,
                            "source_go_number": order.go_number,
                            "relationship_type": rel.relationship_type,
                            "target_go_number": rel.referenced_go_number,
                            "referenced_go_number": rel.referenced_go_number,  # Backward compatibility
                            "matched_phrase": rel.matched_phrase,
                            "evidence_text": rel.evidence_text,
                            "page_number": rel.page_number,
                            "confidence": rel.confidence,
                            "extraction_method": rel.extraction_method,
                            "review_required": rel.review_required,
                        }
                    )

                return {
                    "status": "SUCCESS",
                    "source_go_id": order.id,
                    "source_go_number": order.go_number,
                    "direction": "OUTGOING",
                    "total_relationships": len(formatted_rels),
                    "relationships": formatted_rels,
                    "error": None,
                }

        except Exception as exc:
            logger.error("Failed to retrieve outgoing relationships for '%s': %s", query, exc, exc_info=True)
            return {
                "status": "ERROR",
                "source_go_number": query,
                "direction": "OUTGOING",
                "total_relationships": 0,
                "relationships": [],
                "error": "Database operation could not be completed due to an internal error.",
            }

    def get_incoming_relationships(self, go_number: str) -> Dict[str, Any]:
        """
        Retrieve incoming relationships: Which later orders modify, amend,
        cancel, or supersede this Government Order?

        Searches for records where this GO is referenced as the target in GORelationshipDB,
        and joins GovernmentOrderDB to identify the later acting order.

        Args:
            go_number: Target Government Order number being modified/cancelled.

        Returns:
            Dictionary containing list of incoming relationship records with acting GO details.
        """
        query = go_number.strip() if go_number else ""
        if not query:
            return {
                "status": "EMPTY_QUERY",
                "target_go_number": query,
                "direction": "INCOMING",
                "total_relationships": 0,
                "relationships": [],
                "error": "GO number cannot be empty.",
            }

        try:
            import re
            with SessionLocal() as session:
                # Resolve the target order if it exists in DB, or search by query pattern
                resolution = self._resolve_order(session, query)
                target_official_number = (
                    resolution["order"].go_number if resolution["status"] == "SUCCESS" else query
                )

                # Extract core identifier (e.g. '114/2025/ITD') to handle punctuation variations
                core_match = re.search(r"\d+/\d{4}/[\w&.-]+", query)
                core_ident = core_match.group(0) if core_match else query

                # Search GORelationshipDB where referenced_go_number matches target
                # Check for structural equivalence to prevent substring collisions on number/year prefixes
                parsed_target = self._parse_go_structure(target_official_number) or self._parse_go_structure(query)

                candidate_filter = [
                    GORelationshipDB.referenced_go_number == target_official_number,
                    GORelationshipDB.referenced_go_number.ilike(f"%{query}%"),
                ]
                if core_ident:
                    candidate_filter.append(GORelationshipDB.referenced_go_number.ilike(f"%{core_ident}%"))

                statement = (
                    select(GORelationshipDB, GovernmentOrderDB)
                    .join(GovernmentOrderDB, GORelationshipDB.source_go_id == GovernmentOrderDB.id)
                    .where(or_(*candidate_filter))
                    .order_by(GORelationshipDB.id)
                )

                results = session.execute(statement).all()

                formatted_rels = []
                for rel, source_order in results:
                    # Validate that the referenced_go_number genuinely matches the target order
                    ref_no = rel.referenced_go_number
                    is_match = (
                        ref_no == target_official_number
                        or self._are_structurally_equivalent(ref_no, target_official_number)
                        or (parsed_target and self._are_structurally_equivalent(ref_no, query))
                    )
                    if not is_match:
                        # Fallback for non-canonical target strings: require exact substring match of query
                        if query.lower() not in (ref_no or "").lower():
                            continue

                    formatted_rels.append(
                        {
                            "id": rel.id,
                            "direction": "INCOMING",
                            "source_go_id": source_order.id,
                            "source_go_number": source_order.go_number,
                            "source_issue_date": source_order.issue_date.isoformat() if source_order.issue_date else None,
                            "source_department": source_order.department,
                            "relationship_type": rel.relationship_type,
                            "target_go_number": rel.referenced_go_number,
                            "matched_phrase": rel.matched_phrase,
                            "evidence_text": rel.evidence_text,
                            "page_number": rel.page_number,
                            "confidence": rel.confidence,
                            "extraction_method": rel.extraction_method,
                            "review_required": rel.review_required,
                        }
                    )

                return {
                    "status": "SUCCESS",
                    "target_go_number": target_official_number,
                    "direction": "INCOMING",
                    "total_relationships": len(formatted_rels),
                    "relationships": formatted_rels,
                    "error": None,
                }

        except Exception as exc:
            logger.error("Failed to retrieve incoming relationships for '%s': %s", query, exc, exc_info=True)
            return {
                "status": "ERROR",
                "target_go_number": query,
                "direction": "INCOMING",
                "total_relationships": 0,
                "relationships": [],
                "error": "Database operation could not be completed due to an internal error.",
            }

    def get_relationships_for_order(
        self,
        go_number: str,
        direction: str = "ALL",
    ) -> Dict[str, Any]:
        """
        Consolidated relationship retrieval supporting OUTGOING, INCOMING, or ALL directions.
        Backward-compatible with original get_relationships_for_order method.

        Args:
            go_number: Government Order number to analyze.
            direction: 'OUTGOING', 'INCOMING', or 'ALL' (default: 'ALL').

        Returns:
            Dictionary containing combined relationship records with direction indicators.
        """
        dir_upper = direction.upper().strip()

        if dir_upper == "OUTGOING":
            return self.get_outgoing_relationships(go_number)
        elif dir_upper == "INCOMING":
            return self.get_incoming_relationships(go_number)
        else:
            # Combine both outgoing and incoming
            outgoing_res = self.get_outgoing_relationships(go_number)
            incoming_res = self.get_incoming_relationships(go_number)

            if outgoing_res.get("status") == "ERROR" and incoming_res.get("status") == "ERROR":
                return outgoing_res

            combined = []
            if outgoing_res.get("status") == "SUCCESS":
                combined.extend(outgoing_res.get("relationships", []))
            if incoming_res.get("status") == "SUCCESS":
                combined.extend(incoming_res.get("relationships", []))

            source_number = outgoing_res.get("source_go_number") or incoming_res.get("target_go_number") or go_number

            return {
                "status": "SUCCESS",
                "source_go_number": source_number,
                "direction": "ALL",
                "total_relationships": len(combined),
                "outgoing_count": len(outgoing_res.get("relationships", [])),
                "incoming_count": len(incoming_res.get("relationships", [])),
                "relationships": combined,
                "error": None,
            }
