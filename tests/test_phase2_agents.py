"""
Phase 2 Step 2.3 Test Suite: Database Agent and Relationship Investigation Agent.

Covers all 15 required core scenarios + 23 specialized regression categories (38 tests total):
1. Exact GO lookup (match_type="EXACT", status="SUCCESS").
2. Partial lookup (match_type="PARTIAL", status="SUCCESS").
3. Missing GO (status="NOT_FOUND", clean error).
4. Ambiguous GO matches (status="AMBIGUOUS_MATCH", candidate list returned, zero guessing).
5. Metadata retrieval default (include_full_text=False excludes text, preserves metadata/preview).
6. Incoming relationships retrieval (incoming direction and acting order details).
7. Outgoing relationships retrieval (outgoing direction and referenced order numbers).
8. Missing source records in relationship investigation (status="NOT_FOUND").
9. Unverified relationship handling (verification_status="UNVERIFIED_CANDIDATE", legal_effect_verified=False).
10. Evidence and page-number preservation (verbatim evidence_text, matched_phrase, page_number).
11. Conservative conflict detection (complementary types preserved without conflict; contradictory types flagged).
12. Preservation and typing of all 8 existing relationship types.
13. Preservation of unresolved ordinal references (never inventing a referenced GO number).
14. Pair-wise relationship investigation (INVESTIGATE_PAIR targeting specific source-target pairs).
15. Database cross-examination (CROSS_EXAMINE_DATABASE identifying unrecorded and missing records).
16. Discovered GO numbers accumulation (iterative graph discovery from references and relationships).
17. Invalid or unauthorized tool operations / parameters (strict Pydantic validation).
18. Malformed input & safe fallback handling (non-dict, empty strings, invalid types).
19. Execution trace and decision-source correctness (ExecutionTraceItem, decision_source="DETERMINISTIC_RULES").
20. Compatibility with existing tool interfaces and live PostgreSQL environment guard.
21. Boundary-aware pair-wise matching (114/2025 matches 114/2025/ITD, rejects 2114/2025/ITD).
22. Cross-examination reverse discrepancy (MISSING_FROM_TEXT when DB record is absent from text).
23. Malformed tool records resilience (non-dict records, missing fields, unexpected status with is_complete=False).
24. Database agent dedicated direction methods (direct dispatch to get_outgoing/incoming_relationships).
25. Database agent ERROR status handling (ERROR with found=False returns ERROR, never NOT_FOUND).
26. Relationship agent unknown type handling (unknown types preserved as diagnostic, never mapped to MODIFIES).
27. Cross-examine complementary types (MODIFIES vs EXTENDS produces zero false TYPE_MISMATCH).
28. Cross-examine multi-relationship on same target (EXTENDS + CANCELS in DB: missing CANCELS not masked by EXTENDS).
29. Cross-examine genuine differing classification (CANCELS vs EXTENDS produces TYPE_MISMATCH for review).
30. Pair-wise target in evidence text only (strictly excluded from confirmed matches and preserved as unresolved mention).
31. Exact match plus separate genuine mismatch on same target (EXTENDS matched; CANCELS vs READ_AS reported).
32. Two text relationships against one DB record (unmatched second candidate reported as UNRECORDED_IN_DATABASE).
33. One text relationship against duplicate DB records (unconsumed duplicate reported as MISSING_FROM_TEXT).
34. Multiple complementary relationships for same target (independent 1-to-1 matching without reuse).
35. Malformed DB records discarded with incomplete flag and diagnostics (is_complete=False, human_review_required=True).
36. Excluded incompatible types generate TYPE_MISMATCH (AMENDS vs EXTENDS, READ_AS vs MODIFIES, CANCELS vs SUPERSEDES).
37. Identical unsupported relationship types can never match (_is_complementary_type and cross-examination).
38. Unsupported DB type alongside valid match preserves valid match, marks result incomplete, and records diagnostic.
"""

from datetime import date
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from sqlalchemy import select

from app.database.connection import SessionLocal
from app.models.identifier_extraction import RelationshipType, ExtractionConfidence
from app.agents.tools.database_tool import DatabaseTool
from app.agents.tools.relationship_tool import RelationshipTool
from app.agents.specialized.database_agent import (
    DatabaseAgent,
    DatabaseAgentRequest,
    DatabaseAgentResponse,
    DatabaseOperation,
    RelationshipDirection,
    AgentResponseStatus,
)
from app.agents.specialized.relationship_agent import (
    RelationshipAgent,
    RelationshipAgentRequest,
    RelationshipAgentResponse,
    RelationshipInvestigationMode,
    RelationshipAgentStatus,
    RelationshipCandidate,
)


def is_postgres_available() -> bool:
    """Check if the local PostgreSQL service on port 5433 is live and responding."""
    try:
        with SessionLocal() as session:
            session.execute(select(1))
            return True
    except Exception:
        return False


# =========================================================================
# TEST 1: Exact GO Lookup
# =========================================================================
def test_exact_go_lookup() -> None:
    print("\n--- 1. Exact GO Lookup ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_order_by_number.return_value = {
        "status": "SUCCESS",
        "found": True,
        "query": "G.O.(Rt)No.24/2026/ITD",
        "match_type": "EXACT",
        "candidates": ["G.O.(Rt)No.24/2026/ITD"],
        "data": {
            "id": 1,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "issue_date": "2026-01-23",
            "department": "Electronics & Information Technology Department",
            "source_filename": "24_2026_ITD.pdf",
            "extracted_text_filename": "24_2026_ITD.txt",
            "verification_status": "UNVERIFIED",
            "extracted_text_length": 1500,
            "text_preview": "Sanction accorded for incubation project grant.",
            "extracted_text": "Full extracted text of the Government Order.",
        },
        "error": None,
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    req = DatabaseAgentRequest(
        operation=DatabaseOperation.LOOKUP_ORDER,
        go_number="G.O.(Rt)No.24/2026/ITD",
        include_full_text=True,
    )
    res = agent.execute(req)

    assert res.status == AgentResponseStatus.SUCCESS
    assert res.match_type == "EXACT"
    assert res.order_data is not None
    assert res.order_data.id == 1
    assert res.order_data.go_number == "G.O.(Rt)No.24/2026/ITD"
    assert res.order_data.extracted_text == "Full extracted text of the Government Order."
    assert res.legal_effect_verified is False
    assert "G.O.(Rt)No.24/2026/ITD" in res.discovered_go_numbers
    print("PASS: Exact GO lookup succeeded with exact match type, metadata, and full text.")


# =========================================================================
# TEST 2: Partial GO Lookup
# =========================================================================
def test_partial_go_lookup() -> None:
    print("\n--- 2. Partial GO Lookup ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_order_by_number.return_value = {
        "status": "SUCCESS",
        "found": True,
        "query": "24/2026",
        "match_type": "PARTIAL",
        "candidates": ["G.O.(Rt)No.24/2026/ITD"],
        "data": {
            "id": 1,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "issue_date": "2026-01-23",
            "department": "Electronics & Information Technology Department",
            "source_filename": "24_2026_ITD.pdf",
            "verification_status": "UNVERIFIED",
            "extracted_text_length": 1500,
            "text_preview": "Sanction accorded for incubation project grant.",
        },
        "error": None,
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    res = agent.execute({"operation": "LOOKUP_ORDER", "go_number": "24/2026"})

    assert res.status == AgentResponseStatus.SUCCESS
    assert res.match_type == "PARTIAL"
    assert res.order_data is not None
    assert res.order_data.go_number == "G.O.(Rt)No.24/2026/ITD"
    print("PASS: Partial GO lookup successfully resolved unique match with match_type='PARTIAL'.")


# =========================================================================
# TEST 3: Missing GO
# =========================================================================
def test_missing_go() -> None:
    print("\n--- 3. Missing GO Lookup ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_order_by_number.return_value = {
        "status": "NOT_FOUND",
        "found": False,
        "query": "G.O.(Rt)No.999/2026/NONEXISTENT",
        "match_type": None,
        "candidates": [],
        "data": None,
        "error": "No Government Order found matching 'G.O.(Rt)No.999/2026/NONEXISTENT'.",
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    res = agent.execute(DatabaseAgentRequest(go_number="G.O.(Rt)No.999/2026/NONEXISTENT"))

    assert res.status == AgentResponseStatus.NOT_FOUND
    assert res.order_data is None
    assert len(res.errors) == 1
    assert "No Government Order found" in res.errors[0]
    print("PASS: Missing GO correctly returned status=NOT_FOUND with empty order_data.")


# =========================================================================
# TEST 4: Ambiguous GO Matches
# =========================================================================
def test_ambiguous_go_matches() -> None:
    print("\n--- 4. Ambiguous GO Matches ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_order_by_number.return_value = {
        "status": "AMBIGUOUS_MATCH",
        "found": False,
        "query": "2026",
        "match_type": None,
        "candidates": ["G.O.(Rt)No.24/2026/ITD", "G.O.(P)No.5/2026/FIN", "G.O.(Ms)No.12/2026/GAD"],
        "data": None,
        "error": "Query '2026' is ambiguous. Found 3 matching orders.",
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    res = agent.execute(DatabaseAgentRequest(go_number="2026"))

    assert res.status == AgentResponseStatus.AMBIGUOUS_MATCH
    assert res.order_data is None
    assert len(res.ambiguous_candidates) == 3
    assert "G.O.(Rt)No.24/2026/ITD" in res.ambiguous_candidates
    assert res.total_records == 3
    print("PASS: Ambiguous query returned status=AMBIGUOUS_MATCH and complete candidates list without guessing.")


# =========================================================================
# TEST 5: Metadata Retrieval Default (include_full_text=False)
# =========================================================================
def test_metadata_retrieval_default() -> None:
    print("\n--- 5. Metadata Retrieval Default ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_order_by_number.return_value = {
        "status": "SUCCESS",
        "found": True,
        "query": "G.O.(Rt)No.24/2026/ITD",
        "match_type": "EXACT",
        "candidates": ["G.O.(Rt)No.24/2026/ITD"],
        "data": {
            "id": 1,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "issue_date": "2026-01-23",
            "department": "Electronics & Information Technology Department",
            "source_filename": "24_2026_ITD.pdf",
            "verification_status": "UNVERIFIED",
            "extracted_text_length": 5000,
            "text_preview": "Text preview without full document.",
            "extracted_text": None,
        },
        "error": None,
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    # Default include_full_text is False
    req = DatabaseAgentRequest(operation=DatabaseOperation.GET_METADATA, go_number="G.O.(Rt)No.24/2026/ITD")
    assert req.include_full_text is False
    res = agent.execute(req)

    assert res.status == AgentResponseStatus.SUCCESS
    assert res.order_data is not None
    assert res.order_data.extracted_text is None
    assert res.order_data.text_preview == "Text preview without full document."
    assert mock_db_tool.get_order_by_number.call_args[1]["include_full_text"] is False
    print("PASS: Default include_full_text=False strictly excluded full text while preserving preview and metadata.")


# =========================================================================
# TEST 6: Incoming Relationships Retrieval
# =========================================================================
def test_incoming_relationships() -> None:
    print("\n--- 6. Incoming Relationships Retrieval ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_incoming_relationships.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Ms)No.114/2025/ITD",
        "direction": "INCOMING",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 101,
                "direction": "INCOMING",
                "source_go_id": 5,
                "source_go_number": "G.O.(Rt)No.24/2026/ITD",
                "source_issue_date": "2026-01-23",
                "source_department": "ITD",
                "relationship_type": "AMENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
                "matched_phrase": "amended as",
                "evidence_text": "The government order is hereby amended as follows.",
                "page_number": 2,
                "confidence": "HIGH",
                "review_required": False,
            }
        ],
        "error": None,
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    req = DatabaseAgentRequest(
        operation=DatabaseOperation.GET_RELATIONSHIPS,
        go_number="G.O.(Ms)No.114/2025/ITD",
        direction=RelationshipDirection.INCOMING,
    )
    res = agent.execute(req)

    assert res.status == AgentResponseStatus.SUCCESS
    assert len(res.relationships) == 1
    rel = res.relationships[0]
    assert rel.direction == "INCOMING"
    assert rel.source_go_number == "G.O.(Rt)No.24/2026/ITD"
    assert rel.target_go_number == "G.O.(Ms)No.114/2025/ITD"
    assert rel.relationship_type == "AMENDS"
    assert rel.legal_effect_verified is False
    assert "G.O.(Rt)No.24/2026/ITD" in res.discovered_go_numbers
    print("PASS: Incoming relationships retrieved with acting source GO details and discovered GO extraction.")


# =========================================================================
# TEST 7: Outgoing Relationships Retrieval
# =========================================================================
def test_outgoing_relationships() -> None:
    print("\n--- 7. Outgoing Relationships Retrieval ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "direction": "OUTGOING",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 102,
                "direction": "OUTGOING",
                "source_go_id": 5,
                "source_go_number": "G.O.(Rt)No.24/2026/ITD",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
                "relationship_type": "EXTENDS",
                "matched_phrase": "period is extended",
                "evidence_text": "The project period is extended up to 31-03-2027.",
                "page_number": 1,
                "confidence": "HIGH",
                "review_required": False,
            }
        ],
        "error": None,
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    req = DatabaseAgentRequest(
        operation=DatabaseOperation.GET_RELATIONSHIPS,
        go_number="G.O.(Rt)No.24/2026/ITD",
        direction=RelationshipDirection.OUTGOING,
    )
    res = agent.execute(req)

    assert res.status == AgentResponseStatus.SUCCESS
    assert len(res.relationships) == 1
    rel = res.relationships[0]
    assert rel.direction == "OUTGOING"
    assert rel.target_go_number == "G.O.(Ms)No.114/2025/ITD"
    assert rel.relationship_type == "EXTENDS"
    assert "G.O.(Ms)No.114/2025/ITD" in res.discovered_go_numbers
    print("PASS: Outgoing relationships retrieved with referenced target GO and discovered GO extraction.")


# =========================================================================
# TEST 8: Missing Source Records in Relationship Investigation
# =========================================================================
def test_missing_source_record_in_relationship_agent() -> None:
    print("\n--- 8. Missing Source Records in Relationship Investigation ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_for_order.return_value = {
        "status": "ORDER_NOT_FOUND",
        "source_go_number": "G.O.(Rt)No.999/2026/NONEXISTENT",
        "candidates": [],
        "total_references": 0,
        "references": [],
        "total_relationships": 0,
        "relationships": [],
        "error": "Order 'G.O.(Rt)No.999/2026/NONEXISTENT' could not be uniquely resolved.",
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    req = RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.INVESTIGATE_ORDER,
        go_number="G.O.(Rt)No.999/2026/NONEXISTENT",
    )
    res = agent.execute(req)

    assert res.status == RelationshipAgentStatus.NOT_FOUND
    assert res.total_relationships == 0
    assert len(res.errors) == 1
    assert any(term in res.errors[0].lower() for term in ["not found", "uniquely resolved"])
    print("PASS: Missing source order in relationship investigation returned status=NOT_FOUND safely.")


# =========================================================================
# TEST 9: Unverified Relationship Handling Invariant
# =========================================================================
def test_unverified_relationship_handling() -> None:
    print("\n--- 9. Unverified Relationship Handling Invariant ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "total_references": 1,
        "references": [{"ordinal": 1, "go_number": "G.O.(Ms)No.114/2025/ITD"}],
        "total_relationships": 1,
        "relationships": [
            {
                "relationship_type": "AMENDS",
                "matched_phrase": "amended",
                "evidence_text": "Clause 4 is hereby amended.",
                "page_number": 1,
                "reference_ordinals": [1],
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
                "extraction_method": "rule_based",
                "review_required": False,
            }
        ],
        "error": None,
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    res = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
        text="Clause 4 is hereby amended.",
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))

    assert res.status == RelationshipAgentStatus.SUCCESS
    assert len(res.relationships) == 1
    cand = res.relationships[0]
    # Invariant: Must always be tagged UNVERIFIED_CANDIDATE and legal_effect_verified=False
    assert cand.verification_status == "UNVERIFIED_CANDIDATE"
    assert cand.legal_effect_verified is False
    assert res.legal_effect_verified is False
    print("PASS: Relationship candidate strictly tagged UNVERIFIED_CANDIDATE with legal_effect_verified=False.")


# =========================================================================
# TEST 10: Evidence and Page-Number Preservation
# =========================================================================
def test_evidence_and_page_number_preservation() -> None:
    print("\n--- 10. Evidence and Page-Number Preservation ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "total_references": 1,
        "references": [
            {
                "ordinal": 1,
                "go_number": "G.O.(Ms)No.114/2025/ITD",
                "issue_date": "2025-08-14",
                "page_number": 1,
                "evidence_text": "1. G.O.(Ms)No.114/2025/ITD dated 14/08/2025.",
            }
        ],
        "total_relationships": 1,
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "matched_phrase": "period is extended",
                "evidence_text": "The tenure of the project is extended for one more year.",
                "page_number": 3,
                "reference_ordinals": [1],
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
                "extraction_method": "rule_based",
                "review_required": False,
            }
        ],
        "error": None,
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    res = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
        text="Sample text",
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))

    # Reference evidence
    ref = res.references[0]
    assert ref.page_number == 1
    assert ref.evidence_text == "1. G.O.(Ms)No.114/2025/ITD dated 14/08/2025."

    # Relationship evidence
    rel = res.relationships[0]
    assert rel.page_number == 3
    assert rel.matched_phrase == "period is extended"
    assert rel.evidence_text == "The tenure of the project is extended for one more year."
    print("PASS: Exact evidence text, matched phrases, and page numbers preserved verbatim.")


# =========================================================================
# TEST 11: Conservative Conflict Detection
# =========================================================================
def test_conservative_conflict_detection() -> None:
    print("\n--- 11. Conservative Conflict Detection ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)

    # 11A. Complementary signals (EXTENDS + MODIFIES) must NOT be flagged as conflict!
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "total_references": 1,
        "references": [{"ordinal": 1, "go_number": "G.O.(Ms)No.114/2025/ITD"}],
        "total_relationships": 2,
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "matched_phrase": "period is extended",
                "evidence_text": "The period is extended.",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
            },
            {
                "relationship_type": "MODIFIES",
                "matched_phrase": "modified to that extent",
                "evidence_text": "The order is modified to that extent.",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
            },
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    res_comp = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
        text="text",
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))

    assert res_comp.human_review_required is False
    assert res_comp.relationships[0].potential_conflict is False
    assert res_comp.relationships[1].potential_conflict is False

    # 11B. Contradictory lifecycle signals (CANCELS + MODIFIES for same target)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "total_references": 1,
        "references": [{"ordinal": 1, "go_number": "G.O.(Ms)No.114/2025/ITD"}],
        "total_relationships": 2,
        "relationships": [
            {
                "relationship_type": "CANCELS",
                "matched_phrase": "stands cancelled",
                "evidence_text": "The earlier order stands cancelled.",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
            },
            {
                "relationship_type": "MODIFIES",
                "matched_phrase": "modified",
                "evidence_text": "The order is hereby modified.",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
            },
        ],
    }

    res_contra = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
        text="text",
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))

    # Must flag for human review without deleting either candidate
    assert res_contra.human_review_required is True
    assert "potentially conflicting" in res_contra.human_review_reason.lower()
    assert len(res_contra.relationships) == 2  # Both candidates preserved!
    assert res_contra.relationships[0].potential_conflict is True
    assert res_contra.relationships[1].potential_conflict is True
    print("PASS: Conservative conflict detection correctly distinguished layered modifications from contradictory lifecycle actions.")


# =========================================================================
# TEST 12: Coverage of All 8 Existing Relationship Types
# =========================================================================
def test_coverage_of_all_eight_relationship_types() -> None:
    print("\n--- 12. Coverage of All 8 Existing Relationship Types ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    all_types = [
        "AMENDS", "PARTIALLY_AMENDS", "MODIFIES", "EXTENDS",
        "SUPERSEDES", "CANCELS", "REVOKES", "READ_AS"
    ]
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "total_references": 0,
        "references": [],
        "total_relationships": len(all_types),
        "relationships": [
            {
                "relationship_type": t,
                "matched_phrase": f"matched {t}",
                "evidence_text": f"Evidence for {t}",
                "referenced_go_numbers": [f"GO-{t}"],
                "confidence": "HIGH",
            }
            for t in all_types
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    res = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
        text="Sample text",
    ))

    assert res.total_relationships == 8
    returned_types = [r.relationship_type for r in res.relationships]
    for expected_type in RelationshipType:
        assert expected_type in returned_types, f"Type {expected_type} not found in returned types!"
    print("PASS: All 8 existing RelationshipType enums successfully parsed, validated, and typed.")


# =========================================================================
# TEST 13: Preservation of Unresolved Ordinal References
# =========================================================================
def test_unresolved_ordinal_references_preserved() -> None:
    print("\n--- 13. Unresolved Ordinal References Preserved ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "total_references": 0,
        "references": [],
        "total_relationships": 1,
        "relationships": [
            {
                "relationship_type": "AMENDS",
                "matched_phrase": "read as 2nd paper",
                "evidence_text": "The paper read as 2nd paper above is amended.",
                "reference_ordinals": [2],
                "referenced_go_numbers": [],  # Ordinal could not be resolved to a GO number
                "confidence": "MEDIUM",
            }
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    res = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
        text="text",
    ))

    assert res.total_relationships == 1
    rel = res.relationships[0]
    # Unresolved ordinal must be preserved without inventing a target GO number
    assert rel.reference_ordinals == [2]
    assert rel.referenced_go_numbers == []
    print("PASS: Unresolved ordinal references preserved verbatim without inventing target numbers.")


# =========================================================================
# TEST 14: Pair-Wise Relationship Investigation (INVESTIGATE_PAIR)
# =========================================================================
def test_pairwise_relationship_investigation() -> None:
    print("\n--- 14. Pair-Wise Relationship Investigation ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "total_references": 2,
        "references": [
            {"ordinal": 1, "go_number": "G.O.(Ms)No.114/2025/ITD"},
            {"ordinal": 2, "go_number": "G.O.(P)No.5/2026/FIN"},
        ],
        "total_relationships": 2,
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "matched_phrase": "period is extended",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
            },
            {
                "relationship_type": "MODIFIES",
                "matched_phrase": "modified",
                "referenced_go_numbers": ["G.O.(P)No.5/2026/FIN"],
                "confidence": "HIGH",
            },
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    req = RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.INVESTIGATE_PAIR,
        text="sample text",
        target_go_number="114/2025",  # Target specifically 114/2025
    )
    res = agent.execute(req)

    assert res.status == RelationshipAgentStatus.SUCCESS
    assert res.mode == RelationshipInvestigationMode.INVESTIGATE_PAIR
    assert res.target_go_number == "114/2025"
    assert res.total_relationships == 1
    assert res.relationships[0].referenced_go_numbers == ["G.O.(Ms)No.114/2025/ITD"]
    assert res.relationships[0].relationship_type == RelationshipType.EXTENDS
    print("PASS: Pair-wise investigation filtered relationships targeting the specific requested GO.")


# =========================================================================
# TEST 15: Database Cross-Examination (CROSS_EXAMINE_DATABASE)
# =========================================================================
def test_database_cross_examination() -> None:
    print("\n--- 15. Database Cross-Examination ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "total_references": 1,
        "references": [{"ordinal": 1, "go_number": "G.O.(Ms)No.114/2025/ITD"}],
        "total_relationships": 2,
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "matched_phrase": "extended",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
            },
            {
                "relationship_type": "MODIFIES",
                "matched_phrase": "modified",
                "referenced_go_numbers": ["G.O.(P)No.99/2024/ITD"],  # Newly extracted, unrecorded in DB
                "confidence": "HIGH",
            },
        ],
    }

    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "direction": "OUTGOING",
        "total_relationships": 1,
        "relationships": [
            {
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
                "relationship_type": "EXTENDS",
            }
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    req = RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        go_number="G.O.(Rt)No.24/2026/ITD",
    )
    res = agent.execute(req)

    assert res.status == RelationshipAgentStatus.SUCCESS
    assert res.mode == RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE
    assert len(res.discrepancies) == 1
    disc = res.discrepancies[0]
    assert disc.category == "UNRECORDED_IN_DATABASE"
    assert "99/2024" in disc.referenced_go_number
    assert disc.text_relationship_type == "MODIFIES"
    print("PASS: Cross-examination detected unrecorded candidate relationship between text and database.")


# =========================================================================
# TEST 16: Discovered GO Numbers Accumulation
# =========================================================================
def test_discovered_go_numbers_accumulation() -> None:
    print("\n--- 16. Discovered GO Numbers Accumulation ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_references_for_order.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "total_references": 3,
        "references": [
            {"ordinal": 1, "referenced_go_number": "GO-REF-1"},
            {"ordinal": 2, "referenced_go_number": "GO-REF-2"},
            {"ordinal": 3, "referenced_go_number": "GO-REF-1"}, # Duplicate
        ],
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    res = agent.execute(DatabaseAgentRequest(
        operation=DatabaseOperation.GET_REFERENCES,
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))

    assert res.status == AgentResponseStatus.SUCCESS
    assert len(res.discovered_go_numbers) == 2
    assert "GO-REF-1" in res.discovered_go_numbers
    assert "GO-REF-2" in res.discovered_go_numbers
    print("PASS: References extracted deduplicated set of discovered GO numbers.")


# =========================================================================
# TEST 17: Invalid or Unauthorized Tool Operations & Parameters
# =========================================================================
def test_invalid_parameters_and_operations() -> None:
    print("\n--- 17. Invalid Parameters & Operations Validation ---")
    # A. Forbid extra parameters
    try:
        DatabaseAgentRequest(operation=DatabaseOperation.LOOKUP_ORDER, go_number="GO-1", unauthorized_sql="DROP TABLE")
        assert False, "Expected ValidationError on unauthorized extra fields"
    except ValidationError:
        pass

    # B. Reject boolean as int
    try:
        DatabaseAgentRequest(operation=DatabaseOperation.LOOKUP_ORDER, go_number="GO-1", text_preview_chars=True)
        assert False, "Expected ValidationError on boolean text_preview_chars"
    except ValidationError:
        pass

    # C. Missing required go_number for order operation
    try:
        DatabaseAgentRequest(operation=DatabaseOperation.LOOKUP_ORDER, go_number=None)
        assert False, "Expected ValidationError when go_number missing for LOOKUP_ORDER"
    except ValidationError:
        pass

    print("PASS: Strict Pydantic contracts rejected unauthorized fields, boolean coercions, and missing identifiers.")


# =========================================================================
# TEST 18: Malformed Input & Safe Fallback Handling
# =========================================================================
def test_malformed_input_safe_fallback() -> None:
    print("\n--- 18. Malformed Input & Safe Fallback Handling ---")
    agent_db = DatabaseAgent()
    # Non-dict, non-request input
    res_db = agent_db.execute("not a valid dict or request object")
    assert res_db.status == AgentResponseStatus.ERROR
    assert len(res_db.errors) == 1
    assert "validation" in res_db.errors[0].lower()

    agent_rel = RelationshipAgent()
    # Empty string text
    res_rel = agent_rel.execute({"mode": "EXTRACT_FROM_TEXT", "text": "   "})
    assert res_rel.status == RelationshipAgentStatus.ERROR
    assert "validation" in res_rel.errors[0].lower()
    print("PASS: Malformed inputs returned safe error responses without unhandled crashes.")


# =========================================================================
# TEST 19: Execution Trace and Decision-Source Correctness
# =========================================================================
def test_execution_trace_and_decision_source() -> None:
    print("\n--- 19. Execution Trace and Decision Source ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.list_all_go_numbers.return_value = {
        "status": "SUCCESS",
        "total_count": 2,
        "go_numbers": ["GO-1", "GO-2"],
    }
    agent = DatabaseAgent(database_tool=mock_db_tool)
    res = agent.execute(DatabaseAgentRequest(operation=DatabaseOperation.LIST_ALL_ORDERS))

    assert res.status == AgentResponseStatus.SUCCESS
    assert len(res.execution_trace) == 1
    trace_item = res.execution_trace[0]
    assert trace_item["agent"] == "database_agent"
    assert trace_item["action"] == "list_all_go_numbers"
    assert trace_item["tool_called"] == "database_tool"
    assert trace_item["decision_source"] == "DETERMINISTIC_RULES"
    assert "timestamp" in trace_item
    print("PASS: Execution trace verified adhering to ExecutionTraceItem with decision_source='DETERMINISTIC_RULES'.")


# =========================================================================
# TEST 20: Compatibility with Live PostgreSQL / Environment Guard
# =========================================================================
def test_live_database_compatibility_guard() -> None:
    print("\n--- 20. Live Database Compatibility & Environment Guard ---")
    if is_postgres_available():
        agent = DatabaseAgent()
        res = agent.execute(DatabaseAgentRequest(operation=DatabaseOperation.LIST_ALL_ORDERS))
        assert res.status == AgentResponseStatus.SUCCESS
        assert res.total_records >= 11
        print(f"PASS (Live DB): Live PostgreSQL queried successfully ({res.total_records} orders).")
    else:
        print("NOT RUN (Docker/PostgreSQL unavailable): Live PostgreSQL integration test safely skipped; mock tests verified 100%.")


# =========================================================================
# TEST 21: Boundary-Aware Pair-Wise Matching
# =========================================================================
def test_boundary_aware_pairwise_matching() -> None:
    print("\n--- 21. Boundary-Aware Pair-Wise Matching ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "AMENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "target_go_raw": "G.O.(Ms)No.114/2025/ITD",
                "confidence": "HIGH",
                "evidence_text": "This order amends G.O.(Ms)No.114/2025/ITD dated 15/05/2025.",
                "matched_phrase": "amends",
                "page_number": 1,
            },
            {
                "relationship_type": "MODIFIES",
                "referenced_go_numbers": ["G.O.(Ms)No.2114/2025/ITD"],
                "target_go_raw": "G.O.(Ms)No.2114/2025/ITD",
                "confidence": "HIGH",
                "evidence_text": "In partial modification of G.O.(Ms)No.2114/2025/ITD dated 20/06/2025.",
                "matched_phrase": "in partial modification of",
                "page_number": 1,
            },
        ],
        "references": [
            {
                "ordinal": 1,
                "go_number": "G.O.(Ms)No.114/2025/ITD",
                "raw_text": "1. G.O.(Ms)No.114/2025/ITD dated 15/05/2025.",
                "page_number": 1,
            },
            {
                "ordinal": 2,
                "go_number": "G.O.(Ms)No.2114/2025/ITD",
                "raw_text": "2. G.O.(Ms)No.2114/2025/ITD dated 20/06/2025.",
                "page_number": 1,
            },
        ],
        "unresolved_ordinals": [],
        "total_relationships": 2,
        "total_references": 2,
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)

    # 1. Query with "114/2025" - MUST match G.O.(Ms)No.114/2025/ITD, MUST NOT match 2114/2025/ITD
    res_114 = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.INVESTIGATE_PAIR,
            text="Dummy text containing 114/2025 and 2114/2025",
            target_go_number="114/2025",
        )
    )
    assert res_114.status == RelationshipAgentStatus.SUCCESS
    assert res_114.total_relationships == 1
    assert res_114.relationships[0].referenced_go_numbers == ["G.O.(Ms)No.114/2025/ITD"]
    assert res_114.total_references == 1
    assert res_114.references[0].go_number == "G.O.(Ms)No.114/2025/ITD"

    # 2. Query with "2114/2025" - MUST match G.O.(Ms)No.2114/2025/ITD, MUST NOT match 114/2025/ITD
    res_2114 = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.INVESTIGATE_PAIR,
            text="Dummy text containing 114/2025 and 2114/2025",
            target_go_number="2114/2025",
        )
    )
    assert res_2114.status == RelationshipAgentStatus.SUCCESS
    assert res_2114.total_relationships == 1
    assert res_2114.relationships[0].referenced_go_numbers == ["G.O.(Ms)No.2114/2025/ITD"]
    assert res_2114.total_references == 1
    assert res_2114.references[0].go_number == "G.O.(Ms)No.2114/2025/ITD"

    # 3. Query with non-matching boundary substring "14/2025" - MUST NOT match 114/2025 or 2114/2025
    res_14 = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.INVESTIGATE_PAIR,
            text="Dummy text containing 114/2025 and 2114/2025",
            target_go_number="14/2025",
        )
    )
    assert res_14.status == RelationshipAgentStatus.NO_RELATIONSHIPS_FOUND
    assert res_14.total_relationships == 0
    assert res_14.total_references == 0

    print("PASS: Boundary-aware matching correctly isolated '114/2025' from '2114/2025' without false positive substring overlaps.")


# =========================================================================
# TEST 22: Cross-Examination Reverse Discrepancy (MISSING_FROM_TEXT)
# =========================================================================
def test_cross_examine_missing_from_text_discrepancy() -> None:
    print("\n--- 22. Cross-Examination Reverse Discrepancy (MISSING_FROM_TEXT) ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Relationship tool extracts MODIFIES to 10/2025 (not in DB)
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "MODIFIES",
                "referenced_go_numbers": ["G.O.(Ms)No.10/2025/ITD"],
                "target_go_raw": "G.O.(Ms)No.10/2025/ITD",
                "confidence": "HIGH",
                "evidence_text": "In modification of G.O.(Ms)No.10/2025/ITD.",
                "matched_phrase": "in modification of",
                "page_number": 1,
            }
        ],
        "references": [],
        "unresolved_ordinals": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    # DB records outgoing relationship AMENDS to 50/2024 (not in extracted text)
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 99,
                "relationship_type": "AMENDS",
                "target_go_number": "G.O.(P)No.50/2024/Fin",
                "target_go_id": 42,
                "referenced_go_number": "G.O.(P)No.50/2024/Fin",
            }
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    assert len(res.discrepancies) == 2

    # Check direction 1: Extracted text candidate missing in DB
    unrecorded = [d for d in res.discrepancies if d.category == "UNRECORDED_IN_DATABASE"]
    assert len(unrecorded) == 1
    assert unrecorded[0].referenced_go_number == "G.O.(Ms)No.10/2025/ITD"
    assert unrecorded[0].text_relationship_type == "MODIFIES"
    assert unrecorded[0].db_relationship_type is None

    # Check direction 2: DB record missing from extracted text
    missing_text = [d for d in res.discrepancies if d.category == "MISSING_FROM_TEXT"]
    assert len(missing_text) == 1
    assert "50/2024" in missing_text[0].referenced_go_number
    assert missing_text[0].db_relationship_type == "AMENDS"
    assert missing_text[0].text_relationship_type is None

    # Legal invariant preserved: neither source declared authoritative truth
    assert res.legal_effect_verified is False
    print("PASS: Cross-examination verified both UNRECORDED_IN_DATABASE and MISSING_FROM_TEXT discrepancies with full evidence preservation.")


# =========================================================================
# TEST 23: Malformed Tool Records Resilience
# =========================================================================
def test_malformed_tool_records_resilience() -> None:
    print("\n--- 23. Malformed Tool Records Resilience ---")
    # 1. DatabaseAgent: Malformed reference records returned by tool
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_references_for_order.return_value = {
        "status": "SUCCESS",
        "match_type": "EXACT",
        "total_references": 3,
        "references": [
            # Valid record
            {
                "id": 1,
                "ordinal": 1,
                "referenced_go_number": "G.O.(P)No.10/2024/Fin",
                "reference_date": "2024-01-10",
                "evidence_text": "1. G.O.(P)No.10/2024/Fin",
            },
            # Malformed record: not a dict
            "MALFORMED_NON_DICT_STRING",
            # Malformed record: missing referenced_go_number
            {"id": 2, "ordinal": 2},
        ],
    }

    agent_db = DatabaseAgent(database_tool=mock_db_tool)
    res_db = agent_db.execute(
        DatabaseAgentRequest(
            operation=DatabaseOperation.GET_REFERENCES,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )
    assert res_db.status == AgentResponseStatus.SUCCESS
    assert res_db.total_records == 1  # Only the valid record is preserved
    assert res_db.references[0].referenced_go_number == "G.O.(P)No.10/2024/Fin"
    assert res_db.is_complete is False  # Incomplete response due to discarded malformed records
    assert res_db.malformed_records_discarded == 2
    assert len(res_db.errors) >= 1
    assert "malformed reference record" in res_db.errors[0].lower()

    # 2. DatabaseAgent: Corrupted lookup data missing required fields (e.g. no "id" or "go_number")
    mock_db_tool.get_order_by_number.return_value = {
        "status": "SUCCESS",
        "found": True,
        "data": {"some_corrupted_key": 123},  # missing "id" and "go_number"
    }
    res_lookup_corrupt = agent_db.execute(
        DatabaseAgentRequest(
            operation=DatabaseOperation.LOOKUP_ORDER,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )
    assert res_lookup_corrupt.status == AgentResponseStatus.ERROR
    assert res_lookup_corrupt.is_complete is False
    assert "malformed order record" in res_lookup_corrupt.errors[0].lower()

    # 3. RelationshipAgent: Malformed relationship records in tool response
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "relationships": [
            # Valid record
            {
                "relationship_type": "AMENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "confidence": "HIGH",
                "evidence_text": "Amends G.O. 114/2025",
            },
            # Malformed record: not a dict
            None,
            # Malformed record: missing relationship_type
            {"referenced_go_numbers": ["GO-999"]},
        ],
        "references": [],
        "unresolved_ordinals": [],
        "total_relationships": 3,
        "total_references": 0,
    }
    agent_rel = RelationshipAgent(relationship_tool=mock_rel_tool)
    res_rel = agent_rel.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
            text="Sample order text",
        )
    )
    assert res_rel.status == RelationshipAgentStatus.SUCCESS
    assert res_rel.total_relationships == 1
    assert res_rel.relationships[0].relationship_type == RelationshipType.AMENDS
    assert res_rel.is_complete is False
    assert res_rel.malformed_records_discarded == 2
    assert len(res_rel.errors) >= 1
    assert "malformed record" in res_rel.errors[0].lower()

    # 4. RelationshipAgent: Unexpected tool status handled gracefully
    mock_rel_tool.extract_from_text.return_value = {
        "status": "UNEXPECTED_CORRUPTED_STATUS",
        "error": "Backend tokenizer fault",
    }
    res_unexpected = agent_rel.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
            text="Sample order text",
        )
    )
    assert res_unexpected.status == RelationshipAgentStatus.ERROR
    assert res_unexpected.is_complete is False
    assert any("unexpected" in err.lower() or "unknown" in err.lower() for err in res_unexpected.errors)

    print("PASS: Malformed tool outputs filtered safely, valid records preserved, and incomplete responses flagged.")


# =========================================================================
# TEST 24: Database Agent Dedicated Direction Methods Dispatch
# =========================================================================
def test_database_agent_dedicated_direction_methods() -> None:
    print("\n--- 24. Database Agent Dedicated Direction Methods Dispatch ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)

    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 101,
                "relationship_type": "MODIFIES",
                "target_go_number": "G.O.(Ms)No.12/2024/ITD",
                "target_go_id": 5,
                "referenced_go_number": "G.O.(Ms)No.12/2024/ITD",
            }
        ],
    }

    mock_db_tool.get_incoming_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 102,
                "relationship_type": "AMENDS",
                "acting_go_number": "G.O.(Rt)No.99/2026/ITD",
                "acting_go_id": 8,
                "source_go_number": "G.O.(Rt)No.99/2026/ITD",
            }
        ],
    }

    mock_db_tool.get_relationships_for_order.return_value = {
        "status": "SUCCESS",
        "total_relationships": 2,
        "relationships": [
            {"id": 101, "relationship_type": "MODIFIES", "target_go_number": "G.O.(Ms)No.12/2024/ITD"},
            {"id": 102, "relationship_type": "AMENDS", "acting_go_number": "G.O.(Rt)No.99/2026/ITD"},
        ],
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)

    # 1. OUTGOING direction -> calls get_outgoing_relationships
    res_out = agent.execute(
        DatabaseAgentRequest(
            operation=DatabaseOperation.GET_RELATIONSHIPS,
            go_number="G.O.(Rt)No.24/2026/ITD",
            direction=RelationshipDirection.OUTGOING,
        )
    )
    assert res_out.status == AgentResponseStatus.SUCCESS
    assert res_out.total_records == 1
    assert res_out.relationships[0].relationship_type == "MODIFIES"
    assert res_out.relationships[0].direction == "OUTGOING"
    mock_db_tool.get_outgoing_relationships.assert_called_once_with(go_number="G.O.(Rt)No.24/2026/ITD")
    assert any(t["action"] == "get_outgoing_relationships" for t in res_out.execution_trace)

    # 2. INCOMING direction -> calls get_incoming_relationships
    res_in = agent.execute(
        DatabaseAgentRequest(
            operation=DatabaseOperation.GET_RELATIONSHIPS,
            go_number="G.O.(Rt)No.24/2026/ITD",
            direction=RelationshipDirection.INCOMING,
        )
    )
    assert res_in.status == AgentResponseStatus.SUCCESS
    assert res_in.total_records == 1
    assert res_in.relationships[0].relationship_type == "AMENDS"
    assert res_in.relationships[0].direction == "INCOMING"
    mock_db_tool.get_incoming_relationships.assert_called_once_with(go_number="G.O.(Rt)No.24/2026/ITD")
    assert any(t["action"] == "get_incoming_relationships" for t in res_in.execution_trace)

    # 3. ALL direction -> calls get_relationships_for_order(..., direction="ALL")
    res_all = agent.execute(
        DatabaseAgentRequest(
            operation=DatabaseOperation.GET_RELATIONSHIPS,
            go_number="G.O.(Rt)No.24/2026/ITD",
            direction=RelationshipDirection.ALL,
        )
    )
    assert res_all.status == AgentResponseStatus.SUCCESS
    assert res_all.total_records == 2
    mock_db_tool.get_relationships_for_order.assert_called_once_with(
        go_number="G.O.(Rt)No.24/2026/ITD",
        direction="ALL",
    )
    assert any(t["action"] == "get_relationships_for_order" for t in res_all.execution_trace)

    print("PASS: Dedicated direction methods dispatched cleanly with verified trace records.")


# =========================================================================
# TEST 25: Database Agent ERROR Status Not Masked by found=False
# =========================================================================
def test_database_agent_error_status_not_masked_by_found_false() -> None:
    print("\n--- 25. Database Agent ERROR Status Not Masked by found=False ---")
    mock_db_tool = MagicMock(spec=DatabaseTool)
    mock_db_tool.get_order_by_number.return_value = {
        "status": "ERROR",
        "found": False,
        "error": "PostgreSQL internal timeout or connection failure.",
        "data": None,
    }

    agent = DatabaseAgent(database_tool=mock_db_tool)
    res = agent.execute(
        DatabaseAgentRequest(
            operation=DatabaseOperation.LOOKUP_ORDER,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    # Must return ERROR, NEVER NOT_FOUND!
    assert res.status == AgentResponseStatus.ERROR
    assert res.status != AgentResponseStatus.NOT_FOUND
    assert res.is_complete is False
    assert len(res.errors) >= 1
    assert "timeout or connection failure" in res.errors[0]
    print("PASS: DatabaseAgent returned status=ERROR and did not erroneously treat ERROR with found=False as NOT_FOUND.")


# =========================================================================
# TEST 26: Relationship Agent Unknown Type Never Converted to MODIFIES
# =========================================================================
def test_relationship_agent_unknown_type_not_converted_to_modifies() -> None:
    print("\n--- 26. Unknown Relationship Type Not Converted to MODIFIES ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "UNKNOWN_FABRICATED_TYPE",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "fabricated phrasing",
                "evidence_text": "This order does something unknown to G.O. 114/2025.",
                "confidence": "HIGH",
            }
        ],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT,
            text="Text with unknown relationship type.",
        )
    )

    # Must NOT convert unknown type to RelationshipType.MODIFIES or any of the 8 valid types
    assert res.total_relationships == 0
    assert len(res.relationships) == 0
    # Must preserve as an unrecognized diagnostic candidate
    assert len(res.unrecognized_relationships) == 1
    unrecognized = res.unrecognized_relationships[0]
    assert unrecognized["raw_relationship_type"] == "UNKNOWN_FABRICATED_TYPE"
    assert "114/2025" in str(unrecognized["referenced_go_numbers"])
    # Must mark for human review and mark response incomplete
    assert res.human_review_required is True
    assert res.is_complete is False
    assert res.malformed_records_discarded == 1
    assert any("UNKNOWN_FABRICATED_TYPE" in err for err in res.errors)
    print("PASS: Unknown relationship type preserved as unrecognized diagnostic, marked for review, and never mapped to MODIFIES.")


# =========================================================================
# TEST 27: Cross-Examine Complementary Types (MODIFIES vs EXTENDS) No Mismatch
# =========================================================================
def test_cross_examine_complementary_types_no_mismatch() -> None:
    print("\n--- 27. Cross-Examine Complementary Types (MODIFIES vs EXTENDS) No Mismatch ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Text extracts MODIFIES
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "MODIFIES",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "modified",
                "evidence_text": "This order modifies G.O.(Ms)No.114/2025/ITD.",
                "confidence": "HIGH",
            }
        ],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    # DB records EXTENDS (both are continuing/complementary administrative types)
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 1,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            }
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    # Complementary types MUST NOT produce automatic TYPE_MISMATCH
    assert len(res.discrepancies) == 0
    print("PASS: Complementary types (MODIFIES and EXTENDS) recognized as compatible without generating false TYPE_MISMATCH.")


# =========================================================================
# TEST 28: Cross-Examine Multi-Relationship on Same Target Not Masked
# =========================================================================
def test_cross_examine_multi_relationship_not_masked() -> None:
    print("\n--- 28. Cross-Examine Multi-Relationship on Same Target Not Masked ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Text extracts ONLY EXTENDS for target 114/2025
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "extended",
                "evidence_text": "Period is extended for G.O.(Ms)No.114/2025/ITD.",
                "confidence": "HIGH",
            }
        ],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    # DB records BOTH EXTENDS and CANCELS for target 114/2025
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 2,
        "relationships": [
            {
                "id": 1,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
            {
                "id": 2,
                "relationship_type": "CANCELS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    # Missing CANCELS signal must NOT be silently considered covered merely because EXTENDS matched!
    assert len(res.discrepancies) == 1
    disc = res.discrepancies[0]
    assert disc.category == "MISSING_FROM_TEXT"
    assert disc.db_relationship_type == "CANCELS"
    assert "114/2025" in disc.referenced_go_number
    print("PASS: Missing CANCELS relationship in database was not masked by an existing EXTENDS match on the same target.")


# =========================================================================
# TEST 29: Cross-Examine Genuine Differing Classification (TYPE_MISMATCH)
# =========================================================================
def test_cross_examine_genuine_differing_classification() -> None:
    print("\n--- 29. Cross-Examine Genuine Differing Classification (TYPE_MISMATCH) ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Text extracts CANCELS (terminating action)
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "CANCELS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "is hereby cancelled",
                "evidence_text": "G.O.(Ms)No.114/2025/ITD is hereby cancelled.",
                "confidence": "HIGH",
            }
        ],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    # DB records EXTENDS (continuing action) - genuinely non-complementary!
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 1,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            }
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    assert len(res.discrepancies) == 1
    disc = res.discrepancies[0]
    assert disc.category == "TYPE_MISMATCH"
    assert disc.text_relationship_type == "CANCELS"
    assert disc.db_relationship_type == "EXTENDS"
    assert "114/2025" in disc.referenced_go_number
    # Neither is claimed as legally authoritative
    assert res.legal_effect_verified is False
    print("PASS: Genuine differing non-complementary classification reported as TYPE_MISMATCH for human review.")


# =========================================================================
# TEST 30: Pair-Wise Target in Evidence Text Only Not Matched
# =========================================================================
def test_pairwise_target_in_evidence_text_only_not_matched() -> None:
    print("\n--- 30. Pair-Wise Target in Evidence Text Only Not Matched ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_rel_tool.extract_from_text.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "MODIFIES",
                # Structured referenced_go_numbers points to GO-888, NOT 114/2025!
                "referenced_go_numbers": ["G.O.(Rt)No.888/2025/ITD"],
                "matched_phrase": "in modification of",
                # Target 114/2025 only appears incidentally in narrative evidence text
                "evidence_text": "In modification of G.O. 888/2025, while referencing background context from G.O.(Ms)No.114/2025/ITD.",
                "confidence": "HIGH",
                "page_number": 1,
            }
        ],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.INVESTIGATE_PAIR,
            text="Narrative order text referencing background context",
            target_go_number="114/2025",
        )
    )

    # Must NOT be included as a confirmed pair match!
    assert res.status == RelationshipAgentStatus.NO_RELATIONSHIPS_FOUND
    assert res.total_relationships == 0
    assert len(res.relationships) == 0

    # Must be preserved separately as an unresolved mention
    assert len(res.unresolved_mentions) == 1
    mention = res.unresolved_mentions[0]
    assert mention["target_query"] == "114/2025"
    assert "114/2025" in mention["evidence_text"]
    assert res.human_review_required is True
    assert "unconfirmed narrative mention" in res.human_review_reason.lower()
    print("PASS: Target appearing only in narrative evidence_text was strictly excluded from confirmed pair matches and preserved separately.")


# =========================================================================
# TEST 31: Exact Match Plus Separate Genuine Mismatch for Same Target
# =========================================================================
def test_cross_examine_exact_match_plus_separate_mismatch() -> None:
    print("\n--- 31. Exact Match Plus Separate Genuine Mismatch for Same Target ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Text extracts EXTENDS and CANCELS for target 114/2025
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "extended",
                "evidence_text": "Extension granted.",
                "confidence": "HIGH",
            },
            {
                "relationship_type": "CANCELS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "cancelled",
                "evidence_text": "Cancelled earlier provisions.",
                "confidence": "HIGH",
            },
        ],
        "references": [],
        "total_relationships": 2,
        "total_references": 0,
    }

    # DB records EXTENDS and READ_AS for target 114/2025
    # (EXTENDS matches exactly; CANCELS vs READ_AS is a genuine mismatch)
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 2,
        "relationships": [
            {
                "id": 1,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
            {
                "id": 2,
                "relationship_type": "READ_AS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    # Exactly one mismatch: the exact match EXTENDS cannot hide the CANCELS vs READ_AS mismatch!
    assert len(res.discrepancies) == 1
    disc = res.discrepancies[0]
    assert disc.category == "TYPE_MISMATCH"
    assert disc.text_relationship_type == "CANCELS"
    assert disc.db_relationship_type == "READ_AS"
    assert "114/2025" in disc.referenced_go_number
    assert res.legal_effect_verified is False
    print("PASS: Exact match EXTENDS and genuine mismatch CANCELS vs READ_AS on the same target evaluated independently without masking.")


# =========================================================================
# TEST 32: Two Text Relationships Against One Database Relationship
# =========================================================================
def test_cross_examine_two_text_against_one_db_unrecorded() -> None:
    print("\n--- 32. Two Text Relationships Against One Database Relationship ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Text extracts BOTH EXTENDS and MODIFIES for target 114/2025
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "extended",
                "evidence_text": "Tenure extended.",
                "confidence": "HIGH",
            },
            {
                "relationship_type": "MODIFIES",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "modified",
                "evidence_text": "Provisions modified.",
                "confidence": "HIGH",
            },
        ],
        "references": [],
        "total_relationships": 2,
        "total_references": 0,
    }

    # DB records ONLY EXTENDS for target 114/2025
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [
            {
                "id": 1,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            }
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    # The unmatched second text relationship MODIFIES must be reported as UNRECORDED_IN_DATABASE!
    assert len(res.discrepancies) == 1
    disc = res.discrepancies[0]
    assert disc.category == "UNRECORDED_IN_DATABASE"
    assert disc.text_relationship_type == "MODIFIES"
    assert "114/2025" in disc.referenced_go_number
    assert res.legal_effect_verified is False
    print("PASS: Two text relationships against one DB record correctly reported the unmatched candidate as UNRECORDED_IN_DATABASE.")


# =========================================================================
# TEST 33: One Text Relationship Against Duplicate Database Records
# =========================================================================
def test_cross_examine_one_text_against_duplicate_db_records() -> None:
    print("\n--- 33. One Text Relationship Against Duplicate Database Records ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Text extracts ONE EXTENDS candidate
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "extended",
                "evidence_text": "Extension order.",
                "confidence": "HIGH",
            }
        ],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    # DB records DUPLICATE records for EXTENDS to target 114/2025 (ids 101 and 102)
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 2,
        "relationships": [
            {
                "id": 101,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
            {
                "id": 102,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    # One-to-one matching: first DB record matches text; second duplicate DB record remains unmatched and is reported!
    assert len(res.discrepancies) == 1
    disc = res.discrepancies[0]
    assert disc.category == "MISSING_FROM_TEXT"
    assert disc.db_relationship_type == "EXTENDS"
    assert "114/2025" in disc.referenced_go_number
    assert res.legal_effect_verified is False
    print("PASS: Duplicate DB records handled transparently; unconsumed duplicate reported as MISSING_FROM_TEXT.")


# =========================================================================
# TEST 34: Multiple Complementary Relationships with Independent 1-to-1 Matching
# =========================================================================
def test_cross_examine_multiple_complementary_independent_matching() -> None:
    print("\n--- 34. Multiple Complementary Relationships with Independent 1-to-1 Matching ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    # Text has PARTIALLY_AMENDS and MODIFIES for target 114/2025
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "PARTIALLY_AMENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "partially amended",
                "evidence_text": "Partially amended.",
                "confidence": "HIGH",
            },
            {
                "relationship_type": "MODIFIES",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "modified",
                "evidence_text": "Tenure modified.",
                "confidence": "HIGH",
            },
        ],
        "references": [],
        "total_relationships": 2,
        "total_references": 0,
    }

    # DB has AMENDS and EXTENDS for target 114/2025
    # Pair 1: PARTIALLY_AMENDS complementary to AMENDS
    # Pair 2: MODIFIES complementary to EXTENDS
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 2,
        "relationships": [
            {
                "id": 1,
                "relationship_type": "AMENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
            {
                "id": 2,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    # Both pairs match independently via 1-to-1 complementary pairing without generating false TYPE_MISMATCH
    assert len(res.discrepancies) == 0
    assert res.legal_effect_verified is False
    print("PASS: Multiple complementary relationships matched independently one-to-one without cross-reusing items.")


# =========================================================================
# TEST 35: Malformed Database Relationship Records Discarded with Incomplete Flag
# =========================================================================
def test_cross_examine_malformed_db_records_discarded_incomplete() -> None:
    print("\n--- 35. Malformed Database Records Discarded with Incomplete Flag ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [
            {
                "relationship_type": "EXTENDS",
                "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
                "matched_phrase": "extended",
                "evidence_text": "Extension order.",
                "confidence": "HIGH",
            }
        ],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    # DB returns malformed records: non-dict, missing target_go_number, missing relationship_type, plus 1 valid record
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 4,
        "relationships": [
            # Valid record
            {
                "id": 1,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
            # Malformed: non-dict string
            "MALFORMED_STRING_NOT_A_DICT",
            # Malformed: missing target
            {"id": 2, "relationship_type": "AMENDS"},
            # Malformed: missing relationship_type
            {"id": 3, "target_go_number": "G.O.(Ms)No.99/2025/ITD"},
        ],
    }

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)
    res = agent.execute(
        RelationshipAgentRequest(
            mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
            go_number="G.O.(Rt)No.24/2026/ITD",
        )
    )

    assert res.status == RelationshipAgentStatus.SUCCESS
    # Valid record matched cleanly (no false discrepancy for valid record)
    assert len(res.discrepancies) == 0
    # Malformed records MUST mark response incomplete and require human review!
    assert res.is_complete is False
    assert res.malformed_records_discarded == 3
    assert res.human_review_required is True
    assert any("malformed database relationship record" in err.lower() for err in res.errors)
    print("PASS: Malformed DB records safely filtered, valid record matched, and result marked incomplete with diagnostics.")


# =========================================================================
# TEST 36: Excluded Incompatible Types Independently Generate TYPE_MISMATCH
# =========================================================================
def test_cross_examine_excluded_incompatible_types_generate_mismatch() -> None:
    print("\n--- 36. Excluded Incompatible Types Generate TYPE_MISMATCH ---")
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)

    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)

    # Incompatible Case 1: AMENDS vs EXTENDS
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [{
            "relationship_type": "AMENDS",
            "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
            "matched_phrase": "amends",
            "evidence_text": "Amends GO 114/2025.",
            "confidence": "HIGH",
        }],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [{
            "id": 1,
            "relationship_type": "EXTENDS",
            "target_go_number": "G.O.(Ms)No.114/2025/ITD",
        }],
    }
    res1 = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))
    assert len(res1.discrepancies) == 1
    assert res1.discrepancies[0].category == "TYPE_MISMATCH"
    assert res1.discrepancies[0].text_relationship_type == "AMENDS"
    assert res1.discrepancies[0].db_relationship_type == "EXTENDS"

    # Incompatible Case 2: READ_AS vs MODIFIES
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [{
            "relationship_type": "READ_AS",
            "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
            "matched_phrase": "read as",
            "evidence_text": "Read as GO 114/2025.",
            "confidence": "HIGH",
        }],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [{
            "id": 1,
            "relationship_type": "MODIFIES",
            "target_go_number": "G.O.(Ms)No.114/2025/ITD",
        }],
    }
    res2 = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))
    assert len(res2.discrepancies) == 1
    assert res2.discrepancies[0].category == "TYPE_MISMATCH"
    assert res2.discrepancies[0].text_relationship_type == "READ_AS"
    assert res2.discrepancies[0].db_relationship_type == "MODIFIES"

    # Incompatible Case 3: CANCELS vs SUPERSEDES
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "relationships": [{
            "relationship_type": "CANCELS",
            "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
            "matched_phrase": "cancels",
            "evidence_text": "Cancels GO 114/2025.",
            "confidence": "HIGH",
        }],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [{
            "id": 1,
            "relationship_type": "SUPERSEDES",
            "target_go_number": "G.O.(Ms)No.114/2025/ITD",
        }],
    }
    res3 = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))
    assert len(res3.discrepancies) == 1
    assert res3.discrepancies[0].category == "TYPE_MISMATCH"
    assert res3.discrepancies[0].text_relationship_type == "CANCELS"
    assert res3.discrepancies[0].db_relationship_type == "SUPERSEDES"

    print("PASS: All three excluded incompatible pairs (AMENDS vs EXTENDS, READ_AS vs MODIFIES, CANCELS vs SUPERSEDES) correctly generated TYPE_MISMATCH.")


def test_cross_examine_identical_unsupported_types_do_not_match() -> None:
    """Test 37: Identical unsupported relationship types can never match in _is_complementary_type or cross-examination."""
    # 1. Direct check on _is_complementary_type
    assert RelationshipAgent._is_complementary_type("CUSTOM", "CUSTOM") is False
    assert RelationshipAgent._is_complementary_type("UNKNOWN_REL", "UNKNOWN_REL") is False
    assert RelationshipAgent._is_complementary_type("INVALID_TYPE", "INVALID_TYPE") is False
    assert RelationshipAgent._is_complementary_type("MODIFIES", "CUSTOM") is False
    assert RelationshipAgent._is_complementary_type("CUSTOM", "MODIFIES") is False
    # Valid enum types continue to match or follow narrow complementary pairs
    assert RelationshipAgent._is_complementary_type("MODIFIES", "MODIFIES") is True
    assert RelationshipAgent._is_complementary_type("MODIFIES", "EXTENDS") is True
    assert RelationshipAgent._is_complementary_type("AMENDS", "EXTENDS") is False

    # 2. Check during cross-examination: an unsupported type in DB is discarded and never matched
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)
    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)

    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "relationships": [{
            "id": "rel-1",
            "relationship_type": "EXTENDS",
            "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
            "evidence_text": "Extends GO 114/2025.",
            "confidence": "HIGH",
        }],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }
    # Database returns an unsupported relationship type
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 1,
        "relationships": [{
            "id": 1,
            "relationship_type": "UNSUPPORTED_CUSTOM_TYPE",
            "target_go_number": "G.O.(Ms)No.114/2025/ITD",
        }],
    }

    res = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))
    # DB record was unsupported, discarded from matching, and recorded as diagnostic
    assert res.is_complete is False
    assert res.human_review_required is True
    # Structurally valid DB record with unsupported type is NOT counted as malformed
    assert res.malformed_records_discarded == 0
    assert len(res.unrecognized_relationships) == 1
    assert res.unrecognized_relationships[0]["source"] == "database"
    assert res.unrecognized_relationships[0]["raw_relationship_type"] == "UNSUPPORTED_CUSTOM_TYPE"
    # Text relationship remained unmatched against DB -> UNRECORDED_IN_DATABASE
    assert len(res.discrepancies) == 1
    assert res.discrepancies[0].category == "UNRECORDED_IN_DATABASE"

    print("PASS: Identical unsupported relationship types rejected from matching and correctly flagged for review.")


def test_cross_examine_unsupported_db_type_alongside_valid_match() -> None:
    """Test 38: Unsupported DB type alongside valid match preserves valid match, marks incomplete, and distinguishes malformed records."""
    mock_rel_tool = MagicMock(spec=RelationshipTool)
    mock_db_tool = MagicMock(spec=DatabaseTool)
    agent = RelationshipAgent(relationship_tool=mock_rel_tool, database_tool=mock_db_tool)

    # Text extraction has one valid relationship for target 114/2025
    mock_rel_tool.extract_for_order.return_value = {
        "status": "SUCCESS",
        "source_go_number": "G.O.(Rt)No.24/2026/ITD",
        "relationships": [{
            "id": "rel-1",
            "relationship_type": "EXTENDS",
            "referenced_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
            "evidence_text": "Extends GO 114/2025.",
            "confidence": "HIGH",
        }],
        "references": [],
        "total_relationships": 1,
        "total_references": 0,
    }

    # DB returns three records:
    # 1. Valid record (EXTENDS to 114/2025)
    # 2. Structurally valid record with unsupported type (CUSTOM_ACTION to 120/2025)
    # 3. Structurally malformed record (missing target and type)
    mock_db_tool.get_outgoing_relationships.return_value = {
        "status": "SUCCESS",
        "total_relationships": 3,
        "relationships": [
            {
                "id": 1,
                "relationship_type": "EXTENDS",
                "target_go_number": "G.O.(Ms)No.114/2025/ITD",
            },
            {
                "id": 2,
                "relationship_type": "CUSTOM_ACTION",
                "target_go_number": "G.O.(Ms)No.120/2025/ITD",
            },
            {
                "id": 3,
                "note": "structurally malformed record without target_go_number or relationship_type",
            },
        ],
    }

    res = agent.execute(RelationshipAgentRequest(
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        go_number="G.O.(Rt)No.24/2026/ITD",
    ))

    # 1. Valid relationship matched cleanly with zero discrepancies
    assert len(res.discrepancies) == 0
    # 2. Unsupported DB relationship captured in diagnostics, NOT counted as malformed
    assert len(res.unrecognized_relationships) == 1
    assert res.unrecognized_relationships[0]["source"] == "database"
    assert res.unrecognized_relationships[0]["target_go_number"] == "G.O.(Ms)No.120/2025/ITD"
    assert res.unrecognized_relationships[0]["raw_relationship_type"] == "CUSTOM_ACTION"
    # 3. ONLY the structurally malformed record (record 3) is counted in malformed_records_discarded
    assert res.malformed_records_discarded == 1
    # 4. Incomplete comparison and human review required
    assert res.is_complete is False
    assert res.human_review_required is True
    # 5. Explanatory errors exist for both unsupported type and malformed record
    assert any("CUSTOM_ACTION" in err for err in res.errors)
    assert any("Skipped 1 malformed database relationship record" in err for err in res.errors)

    print("PASS: Unsupported DB type alongside valid match preserves valid match, marks incomplete, and distinguishes malformed records.")


# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    try:
        test_exact_go_lookup()
        test_partial_go_lookup()
        test_missing_go()
        test_ambiguous_go_matches()
        test_metadata_retrieval_default()
        test_incoming_relationships()
        test_outgoing_relationships()
        test_missing_source_record_in_relationship_agent()
        test_unverified_relationship_handling()
        test_evidence_and_page_number_preservation()
        test_conservative_conflict_detection()
        test_coverage_of_all_eight_relationship_types()
        test_unresolved_ordinal_references_preserved()
        test_pairwise_relationship_investigation()
        test_database_cross_examination()
        test_discovered_go_numbers_accumulation()
        test_invalid_parameters_and_operations()
        test_malformed_input_safe_fallback()
        test_execution_trace_and_decision_source()
        test_live_database_compatibility_guard()
        test_boundary_aware_pairwise_matching()
        test_cross_examine_missing_from_text_discrepancy()
        test_malformed_tool_records_resilience()
        test_database_agent_dedicated_direction_methods()
        test_database_agent_error_status_not_masked_by_found_false()
        test_relationship_agent_unknown_type_not_converted_to_modifies()
        test_cross_examine_complementary_types_no_mismatch()
        test_cross_examine_multi_relationship_not_masked()
        test_cross_examine_genuine_differing_classification()
        test_pairwise_target_in_evidence_text_only_not_matched()
        test_cross_examine_exact_match_plus_separate_mismatch()
        test_cross_examine_two_text_against_one_db_unrecorded()
        test_cross_examine_one_text_against_duplicate_db_records()
        test_cross_examine_multiple_complementary_independent_matching()
        test_cross_examine_malformed_db_records_discarded_incomplete()
        test_cross_examine_excluded_incompatible_types_generate_mismatch()
        test_cross_examine_identical_unsupported_types_do_not_match()
        test_cross_examine_unsupported_db_type_alongside_valid_match()

        print("\n=================================================================")
        print("ALL 38 PHASE 2 STEP 2.3 AGENT TESTS COMPLETED (37 PASS, 1 NOT RUN)!")
        print("=================================================================")

    except Exception as exc:
        print(f"\nTEST SUITE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
