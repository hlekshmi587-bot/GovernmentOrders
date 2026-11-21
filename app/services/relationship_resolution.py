"""
Relationship resolution and verification service.
Provides shared, authoritative functions for querying and resolving incoming
relationships for Government Orders. Used by both API endpoints and workflow nodes.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.agents.tools.database_tool import DatabaseTool
from app.database.models import (
    GORelationshipDB,
    GovernmentOrderDB,
    VerificationRecordDB,
)
from app.api.schemas import RelationshipItemSchema

logger = logging.getLogger(__name__)


def resolve_relationship_verification_status(
    records: List[VerificationRecordDB],
) -> str:
    """
    Resolve verification status for a relationship from its verification records.
    Resolution rule:
    - If no records exist, return "UNRECORDED".
    - Prefer the record with the latest effective timestamp:
      rec.verified_at if rec.verified_at is not None else rec.created_at.
    - Deterministic tie-breaker when timestamps are equal: rec.id descending.
    - An older VERIFIED record does not override a newer REJECTED record.
    """
    if not records:
        return "UNRECORDED"

    def sort_key(rec: VerificationRecordDB):
        effective_ts = (
            rec.verified_at.timestamp()
            if rec.verified_at
            else (rec.created_at.timestamp() if rec.created_at else 0.0)
        )
        rec_id = rec.id or 0
        return (effective_ts, rec_id)

    latest_record = max(records, key=sort_key)
    return latest_record.verification_status or "PENDING"


def resolve_incoming_relationships_for_order(
    order_id: int,
    go_number: str,
    db: Session,
) -> Tuple[List[RelationshipItemSchema], bool]:
    """
    Retrieve incoming relationships for a Government Order and resolve their
    latest verification status.

    Returns:
        Tuple[List[RelationshipItemSchema], bool]:
            (resolved_incoming_relationships, lookup_success)
            If an exception occurs during the database query, returns ([], False).
    """
    if not order_id or not go_number:
        return [], True

    try:
        incoming_filters = [GORelationshipDB.referenced_go_number == go_number]
        parsed_target = DatabaseTool._parse_go_structure(go_number)
        if parsed_target:
            num, year, dept = parsed_target
            incoming_filters.append(
                GORelationshipDB.referenced_go_number.ilike(f"%{num}/{year}%")
            )
            if dept:
                incoming_filters.append(
                    GORelationshipDB.referenced_go_number.ilike(
                        f"%{num}/{year}/{dept}%"
                    )
                )

        core_match = re.search(r"\d+/\d{4}/[\w&.-]+", go_number)
        if core_match:
            core_ident = core_match.group(0)
            incoming_filters.append(
                GORelationshipDB.referenced_go_number.ilike(f"%{core_ident}%")
            )

        in_stmt = (
            select(GORelationshipDB, GovernmentOrderDB)
            .join(
                GovernmentOrderDB,
                GORelationshipDB.source_go_id == GovernmentOrderDB.id,
            )
            .where(
                GORelationshipDB.source_go_id != order_id,
                or_(*incoming_filters),
            )
            .order_by(GORelationshipDB.id)
        )
        incoming_candidate_results = db.execute(in_stmt).all()

        valid_incoming = []
        for rel, source_order in incoming_candidate_results:
            ref_no = rel.referenced_go_number
            if ref_no == go_number or DatabaseTool._are_structurally_equivalent(
                ref_no, go_number
            ):
                valid_incoming.append((rel, source_order))

        rel_ids = [r.id for r, _ in valid_incoming]
        ver_records_by_rel: dict[int, list[VerificationRecordDB]] = {}
        if rel_ids:
            ver_stmt = select(VerificationRecordDB).where(
                VerificationRecordDB.relationship_id.in_(rel_ids)
            )
            for vrec in db.scalars(ver_stmt).all():
                if vrec.relationship_id is not None:
                    ver_records_by_rel.setdefault(
                        vrec.relationship_id, []
                    ).append(vrec)

        incoming_schemas = [
            RelationshipItemSchema(
                id=rel.id,
                direction="INCOMING",
                source_go_id=source_order.id,
                source_go_number=source_order.go_number,
                target_go_number=rel.referenced_go_number,
                relationship_type=rel.relationship_type,
                matched_phrase=rel.matched_phrase,
                evidence_text=rel.evidence_text,
                page_number=rel.page_number,
                confidence=rel.confidence or "MEDIUM",
                extraction_method=rel.extraction_method or "RULE_BASED",
                review_required=(
                    rel.review_required
                    if rel.review_required is not None
                    else True
                ),
                verification_status=resolve_relationship_verification_status(
                    ver_records_by_rel.get(rel.id, [])
                ),
                legal_effect_verified=False,
            )
            for rel, source_order in valid_incoming
        ]
        return incoming_schemas, True

    except Exception as exc:
        logger.error(
            "Error querying incoming relationships for order %s (%s): %s",
            order_id,
            go_number,
            exc,
            exc_info=True,
        )
        return [], False
