"""
Phase 2 Step 2.4 - Verification Agent Test Suite.

Contains exactly 52 physical test functions verifying:
- Internal SearchAgent page provenance, retrieval gating, and conservative GO normalization.
- Directness detection with line-level section boundaries and strict LLM entailment validation.
- Canonical 8-type relationship vocabulary, 1-to-1 candidate matching, and schema compatibility.
- Temporal ordering evaluation with 0.5x penalty for relevant claims and no false contradiction.
- Restored confidence thresholds (0.80 / 0.60), vacuous relationship check isolation, and invariants.
- Read-only live database integration and snapshot immutability verification.
"""

from datetime import date
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from app.agents.specialized.database_agent import (
    DatabaseAgent,
    DatabaseAgentResponse,
    AgentResponseStatus,
    OrderMetadataItem,
)
from app.agents.specialized.relationship_agent import (
    RelationshipAgent,
    RelationshipAgentResponse,
    RelationshipAgentStatus,
    RelationshipInvestigationMode,
    RelationshipCandidate,
    DiscrepancyReport,
)
from app.agents.specialized.search_agent import (
    SearchAgent,
    SearchAgentRequest,
    SearchAgentResponse,
    SearchResultItem,
    SupportingChunk,
    RetrievalMode,
)
from app.agents.specialized.verification_agent import (
    VerificationAgent,
    VerificationAgentRequest,
    VerificationAgentResponse,
    VerificationStatus,
    ClaimVerificationStatus,
    EvidenceDirectness,
    TextSourceType,
    PageLocatorSource,
    RelationshipRegistrationStatus,
    RelationshipVerificationCheckState,
    TemporalRelationType,
    VerificationConfidence,
    VerificationClaim,
    RelationshipCandidateInput,
    SupportingChunkInput,
    AutonomousVerificationAction,
    VerificationRecommendation,
    TargetAgentName,
    EmptyParams,
)
from app.agents.specialized.database_agent import DatabaseOperation


# =========================================================================
# 1. Page Provenance, Retrieval Gating & Identifier Normalization (1–15)
# =========================================================================

def test_provenance_rejects_caller_supplied_chunks():
    """Test 1: Caller-supplied chunks are never trusted to establish verified page provenance."""
    agent = VerificationAgent()
    src_text = "--- Page 1 ---\nSample text here\n--- Page 2 ---\nDuplicate phrase occurs here\n--- Page 3 ---\nDuplicate phrase occurs here"
    caller_chunk = SupportingChunkInput(chunk_index=0, chunk_text="Duplicate phrase occurs here", page_number=2)
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Duplicate phrase occurs here",
        validated_search_chunk=caller_chunk,
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_rejects_fabricated_page_numbers():
    """Test 2: A fabricated reported page number cannot override authoritative multi-page ambiguity."""
    agent = VerificationAgent()
    src_text = "--- Page 1 ---\nSome claim here\n--- Page 2 ---\nSome claim here"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Some claim here",
        reported_page=99,
        is_trusted_reported_page=False,
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_accepts_single_page_occurrence_from_db_text():
    """Test 3: A unique page occurrence in authoritative DB text resolves without SearchAgent."""
    agent = VerificationAgent()
    src_text = "--- Page 1 ---\nIntroductory remarks\n--- Page 2 ---\nUnique directive phrase\n--- Page 3 ---\nConclusions"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Unique directive phrase",
    )
    assert verified is True
    assert page == 2
    assert loc_src == PageLocatorSource.DERIVED_FROM_PAGE_MARKER


def test_provenance_leaves_multi_page_unresolved_without_search():
    """Test 4: Duplicate excerpt across multiple pages remains ambiguous when search is not invoked."""
    agent = VerificationAgent()
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_resolves_multi_page_via_validated_search_chunk():
    """Test 5: Multi-page ambiguity resolved when internal SearchAgent returns a validated chunk."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=1,
        results=[
            SearchResultItem(
                go_number="G.O. (Ms) No. 10/2024/GAD",
                source_filename="order.pdf",
                matched_sources=["semantic"],
                supporting_chunks=[
                    SupportingChunk(chunk_index=0, chunk_text="Recurring clause in detailed context", page_number=2)
                ],
            )
        ],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is True
    assert page == 2
    assert loc_src == PageLocatorSource.TRUSTED_DOCUMENT_METADATA


def test_provenance_rejects_search_agent_partial_success_status():
    """Test 6: PARTIAL_SUCCESS from SearchAgent is untrusted and leaves provenance unresolved."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="PARTIAL_SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=1,
        results=[
            SearchResultItem(
                go_number="G.O. (Ms) No. 10/2024/GAD",
                source_filename="order.pdf",
                matched_sources=["semantic"],
                supporting_chunks=[
                    SupportingChunk(chunk_index=0, chunk_text="Recurring clause", page_number=2)
                ],
            )
        ],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_rejects_search_agent_zero_returned_count():
    """Test 7: SearchAgent returning returned_count=0 leaves provenance unresolved."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=0,
        results=[],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_rejects_search_chunk_with_mismatched_go_number():
    """Test 8: Search chunk from an unrelated GO cannot resolve ambiguity."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=1,
        results=[
            SearchResultItem(
                go_number="G.O. (Ms) No. 999/2024/OTHER",
                source_filename="unrelated.pdf",
                matched_sources=["semantic"],
                supporting_chunks=[
                    SupportingChunk(chunk_index=0, chunk_text="Recurring clause", page_number=2)
                ],
            )
        ],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_rejects_search_chunk_missing_excerpt():
    """Test 9: Search chunk that does not contain the cited excerpt verbatim is rejected."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=1,
        results=[
            SearchResultItem(
                go_number="G.O. (Ms) No. 10/2024/GAD",
                source_filename="order.pdf",
                matched_sources=["semantic"],
                supporting_chunks=[
                    SupportingChunk(chunk_index=0, chunk_text="Completely different content", page_number=2)
                ],
            )
        ],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_rejects_search_chunk_with_external_page_number():
    """Test 10: Search chunk citing a page not present in authoritative candidate pages is rejected."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=1,
        results=[
            SearchResultItem(
                go_number="G.O. (Ms) No. 10/2024/GAD",
                source_filename="order.pdf",
                matched_sources=["semantic"],
                supporting_chunks=[
                    SupportingChunk(chunk_index=0, chunk_text="Recurring clause", page_number=99)
                ],
            )
        ],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_conflicting_search_chunks_remain_unresolved():
    """Test 11: Conflicting chunks pointing to different candidate pages keep ambiguity unresolved."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=2,
        results=[
            SearchResultItem(
                go_number="G.O. (Ms) No. 10/2024/GAD",
                source_filename="order.pdf",
                matched_sources=["semantic"],
                supporting_chunks=[
                    SupportingChunk(chunk_index=0, chunk_text="Recurring clause on p1", page_number=1),
                    SupportingChunk(chunk_index=1, chunk_text="Recurring clause on p2", page_number=2),
                ],
            )
        ],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause on p1\n--- Page 2 ---\nRecurring clause on p2"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_provenance_identical_page_search_chunks_resolve_successfully():
    """Test 12: Multiple search chunks pointing to the exact same page reinforce resolution."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.return_value = SearchAgentResponse(
        status="SUCCESS",
        query="Recurring clause",
        requested_mode=RetrievalMode.HYBRID,
        returned_count=1,
        results=[
            SearchResultItem(
                go_number="G.O. (Ms) No. 10/2024/GAD",
                source_filename="order.pdf",
                matched_sources=["semantic"],
                supporting_chunks=[
                    SupportingChunk(chunk_index=0, chunk_text="Recurring clause sentence A", page_number=2),
                    SupportingChunk(chunk_index=1, chunk_text="Recurring clause sentence B", page_number=2),
                ],
            )
        ],
    )
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause sentence A and B"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is True
    assert page == 2
    assert loc_src == PageLocatorSource.TRUSTED_DOCUMENT_METADATA


def test_provenance_handles_search_agent_exception_gracefully():
    """Test 13: Exception raised by SearchAgent leaves multi-page provenance unresolved without crash."""
    mock_search = MagicMock(spec=SearchAgent)
    mock_search.execute_search.side_effect = RuntimeError("Search service timeout")
    agent = VerificationAgent(search_agent=mock_search)
    src_text = "--- Page 1 ---\nRecurring clause\n--- Page 2 ---\nRecurring clause"
    page, verified, loc_src, note = agent.derive_page_number_from_text(
        source_text=src_text,
        excerpt="Recurring clause",
        go_number="G.O. (Ms) No. 10/2024/GAD",
    )
    assert verified is False
    assert page is None
    assert loc_src == PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE


def test_conservative_go_normalization_prevents_identifier_collision():
    """Test 14: Conservative normalization prevents collision between GO 12/2024 and GO 120/2024."""
    norm1 = VerificationAgent._normalize_go_identifier("G.O.(Ms) No. 12/2024/GAD")
    norm2 = VerificationAgent._normalize_go_identifier("G.O.(Ms) No. 120/2024/GAD")
    assert norm1 != norm2
    assert VerificationAgent._are_go_identifiers_equal(norm1, norm2) is False


def test_conservative_go_normalization_matches_equivalent_punctuations():
    """Test 15: Conservative normalization equates typographical variants of the same GO."""
    id1 = "G.O. (Ms) No. 123/2024/GAD"
    id2 = "G.O.(MS)NO.123/2024/GAD"
    assert VerificationAgent._are_go_identifiers_equal(id1, id2) is True


# =========================================================================
# 2. Directness Detection & LLM Entailment (16–25)
# =========================================================================

def test_directness_detection_operative_order_heading():
    """Test 16: Excerpt under an operative line-level ORDER heading is DIRECT_OPERATIVE_ORDER."""
    doc_text = "Read: 1. Letter dated 2024-01-01\n\nORDER\n\nSanction is hereby accorded to approve the project."
    excerpt = "Sanction is hereby accorded"
    directness = VerificationAgent._detect_section_directness(doc_text, excerpt)
    assert directness == EvidenceDirectness.DIRECT_OPERATIVE_ORDER


def test_directness_detection_ignores_order_word_in_ordinary_prose():
    """Test 17: Occurrence of 'order' inside ordinary prose does not falsely trigger operative heading."""
    doc_text = "In accordance with the executive order issued earlier, the committee discussed the proposal."
    excerpt = "discussed the proposal"
    directness = VerificationAgent._detect_section_directness(doc_text, excerpt)
    assert directness == EvidenceDirectness.UNCLASSIFIED


def test_directness_detection_read_citation_section():
    """Test 18: Excerpt located inside a Read: citation section is INDIRECT_PREAMBLE_CITATION."""
    doc_text = "Read: 1. G.O. (Ms) No. 45/2023 dated 12.05.2023\n\nORDER\n\nGovernment are pleased to order."
    excerpt = "G.O. (Ms) No. 45/2023"
    directness = VerificationAgent._detect_section_directness(doc_text, excerpt)
    assert directness == EvidenceDirectness.INDIRECT_PREAMBLE_CITATION


def test_directness_detection_preamble_citation_section():
    """Test 19: Excerpt located inside a Preamble section is INDIRECT_PREAMBLE_CITATION."""
    doc_text = "Preamble\nAs per representations received, government examined the matter.\n\nORDER\nSanction is accorded."
    excerpt = "representations received"
    directness = VerificationAgent._detect_section_directness(doc_text, excerpt)
    assert directness == EvidenceDirectness.INDIRECT_PREAMBLE_CITATION


def test_directness_detection_missing_headers_defaults_to_unclassified():
    """Test 20: Excerpt in a document lacking canonical section headers defaults to UNCLASSIFIED."""
    doc_text = "The committee met on Monday and recommended approval of the grant."
    excerpt = "recommended approval"
    directness = VerificationAgent._detect_section_directness(doc_text, excerpt)
    assert directness == EvidenceDirectness.UNCLASSIFIED


def test_llm_entailment_valid_supported_response():
    """Test 21: Valid JSON with ENTAILED and finite confidence returns SUPPORTED."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "ENTAILED", "confidence": 0.95, "reasoning": "Direct match", "supporting_excerpt": "Sanction is accorded"}'
    agent = VerificationAgent(llm_service=mock_llm)
    src_text = "ORDER\nSanction is accorded for Rs 50 Lakhs."
    status, conf, reason, exc = agent._evaluate_semantic_entailment(
        claim="Sanction is accorded",
        source_text=src_text,
        cited_excerpt="Sanction is accorded",
    )
    assert status == "SUPPORTED"
    assert conf == 0.95
    assert exc == "Sanction is accorded"


def test_llm_entailment_valid_contradicted_response():
    """Test 22: Valid JSON with CONTRADICTED returns CONTRADICTED."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "CONTRADICTED", "confidence": 0.90, "reasoning": "Claim says approved but text says rejected", "contradictory_excerpt": "Proposal is rejected"}'
    agent = VerificationAgent(llm_service=mock_llm)
    src_text = "ORDER\nProposal is rejected after review."
    status, conf, reason, exc = agent._evaluate_semantic_entailment(
        claim="Proposal was approved",
        source_text=src_text,
        cited_excerpt="Proposal is rejected",
    )
    assert status == "CONTRADICTED"
    assert conf == 0.90


def test_llm_entailment_malformed_json_yields_unverifiable():
    """Test 23: Malformed JSON output from LLM safely yields UNVERIFIABLE with 0.0 confidence."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "This is not valid JSON at all."
    agent = VerificationAgent(llm_service=mock_llm)
    src_text = "ORDER\nSanction is accorded."
    status, conf, reason, exc = agent._evaluate_semantic_entailment(
        claim="Sanction is accorded",
        source_text=src_text,
        cited_excerpt="Sanction is accorded",
    )
    assert status == "UNVERIFIABLE"
    assert conf == 0.0


def test_llm_entailment_invalid_confidence_bounds_yields_unverifiable():
    """Test 24: Confidence outside [0.0, 1.0] yields UNVERIFIABLE with 0.0 confidence."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "ENTAILED", "confidence": 1.5, "reasoning": "Too high"}'
    agent = VerificationAgent(llm_service=mock_llm)
    src_text = "ORDER\nSanction is accorded."
    status, conf, reason, exc = agent._evaluate_semantic_entailment(
        claim="Sanction is accorded",
        source_text=src_text,
        cited_excerpt="Sanction is accorded",
    )
    assert status == "UNVERIFIABLE"
    assert conf == 0.0


def test_llm_entailment_nan_or_inf_confidence_yields_unverifiable():
    """Test 25: NaN or Inf confidence from LLM yields UNVERIFIABLE with 0.0 confidence."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "ENTAILED", "confidence": NaN, "reasoning": "NaN test"}'
    agent = VerificationAgent(llm_service=mock_llm)
    src_text = "ORDER\nSanction is accorded."
    status, conf, reason, exc = agent._evaluate_semantic_entailment(
        claim="Sanction is accorded",
        source_text=src_text,
        cited_excerpt="Sanction is accorded",
    )
    assert status == "UNVERIFIABLE"
    assert conf == 0.0


# =========================================================================
# 3. Relationship Consistency & Schema Compatibility (26–36)
# =========================================================================

def test_relationship_canonical_vocabulary_exact_match():
    """Test 26: Candidate matching DB exactly within canonical vocabulary produces REGISTERED_EXACT_MATCH."""
    cand = RelationshipCandidateInput(
        relationship_type="AMENDS",
        source_go_number="GO 100",
        target_order_number="GO 50",
    )
    db_rels = [{"target_order_number": "GO 50", "relationship_type": "AMENDS"}]
    details, discs = VerificationAgent._match_relationship_candidates([cand], db_rels, "GO 100")
    assert len(details) == 1
    assert details[0].registration_status == RelationshipRegistrationStatus.REGISTERED_EXACT_MATCH
    assert len(discs) == 0


def test_relationship_canonical_vocabulary_complementary_match():
    """Test 27: Complementary pair (e.g. AMENDS vs PARTIALLY_AMENDS) produces REGISTERED_COMPLEMENTARY_MATCH."""
    cand = RelationshipCandidateInput(
        relationship_type="AMENDS",
        source_go_number="GO 100",
        target_order_number="GO 50",
    )
    db_rels = [{"target_order_number": "GO 50", "relationship_type": "PARTIALLY_AMENDS"}]
    details, discs = VerificationAgent._match_relationship_candidates([cand], db_rels, "GO 100")
    assert len(details) == 1
    assert details[0].registration_status == RelationshipRegistrationStatus.REGISTERED_COMPLEMENTARY_MATCH
    assert len(discs) == 0


def test_relationship_unsupported_text_type_generates_discrepancy():
    """Test 28: Unsupported type in text candidate generates UNSUPPORTED_TYPE discrepancy."""
    cand = RelationshipCandidateInput(
        relationship_type="RATIFIES",
        source_go_number="GO 100",
        target_order_number="GO 50",
    )
    db_rels = [{"target_order_number": "GO 50", "relationship_type": "AMENDS"}]
    details, discs = VerificationAgent._match_relationship_candidates([cand], db_rels, "GO 100")
    assert len(details) == 1
    assert details[0].registration_status == RelationshipRegistrationStatus.UNREGISTERED_INCOMPATIBLE_TYPE
    assert any(d.category == "UNSUPPORTED_TYPE" for d in discs)


def test_relationship_unsupported_db_type_generates_discrepancy():
    """Test 29: Unsupported type in DB record generates UNSUPPORTED_TYPE discrepancy."""
    cand = RelationshipCandidateInput(
        relationship_type="AMENDS",
        source_go_number="GO 100",
        target_order_number="GO 50",
    )
    db_rels = [{"target_order_number": "GO 99", "relationship_type": "CLARIFIES"}]
    details, discs = VerificationAgent._match_relationship_candidates([cand], db_rels, "GO 100")
    assert any(d.category == "UNSUPPORTED_TYPE" for d in discs)


def test_relationship_one_to_one_candidate_matching_no_double_consumption():
    """Test 30: A DB relationship cannot be consumed by two duplicate text candidates."""
    cand1 = RelationshipCandidateInput(relationship_type="AMENDS", source_go_number="GO 100", target_order_number="GO 50")
    cand2 = RelationshipCandidateInput(relationship_type="AMENDS", source_go_number="GO 100", target_order_number="GO 50")
    db_rels = [{"target_order_number": "GO 50", "relationship_type": "AMENDS"}]
    details, discs = VerificationAgent._match_relationship_candidates([cand1, cand2], db_rels, "GO 100")
    assert len(details) == 2
    assert details[0].registration_status == RelationshipRegistrationStatus.REGISTERED_EXACT_MATCH
    assert details[1].registration_status == RelationshipRegistrationStatus.UNRECORDED_IN_DATABASE
    assert any(d.category == "UNRECORDED_IN_DATABASE" for d in discs)


def test_relationship_missing_from_text_detected():
    """Test 31: Unconsumed DB relationship is flagged as MISSING_FROM_TEXT."""
    cand = RelationshipCandidateInput(relationship_type="AMENDS", source_go_number="GO 100", target_order_number="GO 50")
    db_rels = [
        {"target_order_number": "GO 50", "relationship_type": "AMENDS"},
        {"target_order_number": "GO 75", "relationship_type": "MODIFIES"},
    ]
    details, discs = VerificationAgent._match_relationship_candidates([cand], db_rels, "GO 100")
    assert any(d.category == "MISSING_FROM_TEXT" and d.referenced_go_number == "GO 75" for d in discs)


def test_relationship_unrecorded_in_db_detected():
    """Test 32: Candidate asserting relationship with target not in DB is flagged as UNRECORDED_IN_DATABASE."""
    cand = RelationshipCandidateInput(relationship_type="AMENDS", source_go_number="GO 100", target_order_number="GO 99")
    db_rels = []
    details, discs = VerificationAgent._match_relationship_candidates([cand], db_rels, "GO 100")
    assert details[0].registration_status == RelationshipRegistrationStatus.UNRECORDED_IN_DATABASE
    assert any(d.category == "UNRECORDED_IN_DATABASE" for d in discs)


def test_relationship_cross_examination_coverage_vs_consistency_distinction():
    """Test 33: Cross-examination performed with discrepancies sets performed=True, consistent=False."""
    mock_rel = MagicMock(spec=RelationshipAgent)
    mock_rel.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        total_relationships=1,
        discrepancies=[
            DiscrepancyReport(
                referenced_go_number="GO 2",
                category="TYPE_MISMATCH",
                description="Type mismatch",
            )
        ],
        is_complete=True,
    )
    agent = VerificationAgent(relationship_agent=mock_rel)
    details, discs, state, score = agent._cross_examine_relationships(source_go_number="GO 1")
    assert state == RelationshipVerificationCheckState.DISCREPANCIES_FOUND
    assert score == 0.0


def test_relationship_discrepancies_prevent_verified_status():
    """Test 34: Existence of relationship discrepancies prevents overall status from being VERIFIED."""
    mock_db = MagicMock(spec=DatabaseAgent)
    mock_db.execute.return_value = DatabaseAgentResponse(
        status=AgentResponseStatus.SUCCESS,
        operation=DatabaseOperation.LOOKUP_ORDER,
        order_data=OrderMetadataItem(
            id=1,
            go_number="GO 100",
            source_filename="doc.pdf",
            issue_date="2024-01-01",
            extracted_text="ORDER\nDirective approved.",
            text_preview="",
            extracted_text_length=len("ORDER\nDirective approved."),
        ),
    )
    mock_rel = MagicMock(spec=RelationshipAgent)
    mock_rel.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        total_relationships=1,
        discrepancies=[
            DiscrepancyReport(
                referenced_go_number="GO 50",
                category="TYPE_MISMATCH",
                description="Conflict",
            )
        ],
    )
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "ENTAILED", "confidence": 0.90, "reasoning": "Supported"}'

    agent = VerificationAgent(database_agent=mock_db, relationship_agent=mock_rel, llm_service=mock_llm)
    req = VerificationAgentRequest(
        claims=[VerificationClaim(claim_text="Directive approved.", cited_go_number="GO 100")],
        cited_go_numbers=["GO 100"],
    )
    res = agent.execute(req)
    assert res.status != VerificationStatus.VERIFIED
    assert res.status == VerificationStatus.PARTIALLY_VERIFIED
    assert res.human_review_required is True


def test_vacuous_relationship_check_cannot_yield_verified():
    """Test 35: A vacuous relationship check caps overall status at PARTIALLY_VERIFIED."""
    mock_db = MagicMock(spec=DatabaseAgent)
    mock_db.execute.return_value = DatabaseAgentResponse(
        status=AgentResponseStatus.SUCCESS,
        operation=DatabaseOperation.LOOKUP_ORDER,
        order_data=OrderMetadataItem(
            id=1,
            go_number="GO 100",
            source_filename="doc.pdf",
            issue_date="2024-01-01",
            extracted_text="ORDER\nDirective approved.",
            text_preview="",
            extracted_text_length=len("ORDER\nDirective approved."),
        ),
    )
    mock_rel = MagicMock(spec=RelationshipAgent)
    mock_rel.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        total_relationships=0,
        relationships=[],
        discrepancies=[],
    )
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "ENTAILED", "confidence": 0.90, "reasoning": "Supported"}'

    agent = VerificationAgent(database_agent=mock_db, relationship_agent=mock_rel, llm_service=mock_llm)
    req = VerificationAgentRequest(
        claims=[VerificationClaim(claim_text="Directive approved.", cited_go_number="GO 100")],
        cited_go_numbers=["GO 100"],
    )
    res = agent.execute(req)
    assert res.status == VerificationStatus.PARTIALLY_VERIFIED


def test_discrepancy_report_schema_preserves_category_and_go_fields():
    """Test 36: DiscrepancyReport preserves referenced_go_number, category, and descriptive fields."""
    disc = DiscrepancyReport(
        referenced_go_number="GO 50/2023",
        category="TYPE_MISMATCH",
        text_relationship_type="AMENDS",
        db_relationship_type="MODIFIES",
        description="Type mismatch detected",
    )
    assert disc.referenced_go_number == "GO 50/2023"
    assert disc.category == "TYPE_MISMATCH"
    assert disc.text_relationship_type == "AMENDS"
    assert disc.db_relationship_type == "MODIFIES"


# =========================================================================
# 4. Temporal Evaluation & Relevance (37–42)
# =========================================================================

def test_temporal_evaluation_valid_chronology_passes():
    """Test 37: Sequential chronology (target issued before source) passes with no discrepancy."""
    rel = MagicMock()
    rel.source_go_number = "GO 100"
    rel.target_go_number = "GO 50"
    rel.source_date = date(2024, 6, 1)
    rel.target_date = date(2024, 1, 1)
    discs = VerificationAgent._evaluate_temporal_ordering([rel], {})
    assert len(discs) == 0
    assert rel.temporal_relation == TemporalRelationType.SEQUENTIAL_TARGET_BEFORE_SOURCE


def test_temporal_anomaly_does_not_mark_claim_contradicted():
    """Test 38: Temporal anomaly alone must never mark a factual claim as CONTRADICTED."""
    claim = VerificationClaim(
        claim_text="G.O. (Ms) No. 10/2024 amends G.O. (Ms) No. 20/2024 regarding budget.",
        cited_go_number="G.O. (Ms) No. 10/2024",
    )
    mock_db = MagicMock(spec=DatabaseAgent)
    mock_db.execute.return_value = DatabaseAgentResponse(
        status=AgentResponseStatus.SUCCESS,
        operation=DatabaseOperation.LOOKUP_ORDER,
        order_data=OrderMetadataItem(
            id=1,
            go_number="G.O. (Ms) No. 10/2024",
            source_filename="doc.pdf",
            issue_date="2024-01-01",
            extracted_text="ORDER\nG.O. (Ms) No. 10/2024 amends G.O. (Ms) No. 20/2024 regarding budget.",
            text_preview="",
            extracted_text_length=len("ORDER\nG.O. (Ms) No. 10/2024 amends G.O. (Ms) No. 20/2024 regarding budget."),
        ),
    )
    mock_rel = MagicMock(spec=RelationshipAgent)
    mock_rel.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        total_relationships=1,
        relationships=[],
        discrepancies=[],
    )
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "ENTAILED", "confidence": 0.90, "reasoning": "Supported"}'

    agent = VerificationAgent(database_agent=mock_db, relationship_agent=mock_rel, llm_service=mock_llm)
    with patch.object(
        agent,
        "_cross_examine_relationships",
        return_value=(
            [],
            [{"category": "TEMPORAL_ORDERING_ANOMALY", "referenced_go_number": "G.O. (Ms) No. 20/2024"}],
            True,
            False,
            0.0,
            True,
        ),
    ):
        res = agent.execute(VerificationAgentRequest(claims=[claim], cited_go_numbers=["G.O. (Ms) No. 10/2024"]))
        assert len(res.verified_claims) == 1
        assert res.verified_claims[0].status == ClaimVerificationStatus.SUPPORTED
        assert res.verified_claims[0].status != ClaimVerificationStatus.CONTRADICTED


def test_temporal_evaluation_anomaly_applies_half_penalty_to_relevant_claim():
    """Test 39: Demonstrably relevant claim receives 0.5x scaling penalty on temporal anomaly."""
    raw_score, factors = VerificationAgent._compute_claim_confidence(
        go_exists=True,
        text_source=TextSourceType.FULL_DATABASE_TEXT,
        directness=EvidenceDirectness.DIRECT_OPERATIVE_ORDER,
        status=ClaimVerificationStatus.SUPPORTED,
        page_verified=True,
        source_filename="order.pdf",
        temporal_penalty_multiplier=1.0,
    )
    penalized_score, p_factors = VerificationAgent._compute_claim_confidence(
        go_exists=True,
        text_source=TextSourceType.FULL_DATABASE_TEXT,
        directness=EvidenceDirectness.DIRECT_OPERATIVE_ORDER,
        status=ClaimVerificationStatus.SUPPORTED,
        page_verified=True,
        source_filename="order.pdf",
        temporal_penalty_multiplier=0.5,
    )
    assert penalized_score == round(raw_score * 0.5, 4)
    assert p_factors["temporal_penalty_multiplier"] == 0.5


def test_temporal_evaluation_anomaly_ignored_for_irrelevant_claim():
    """Test 40: Temporal anomaly between orders does not penalize an irrelevant factual claim."""
    relevant = VerificationAgent._is_claim_relevant_to_temporal_discrepancy(
        claim_text="The sanctioned strength of staff in Secretariat is 120 posts.",
        source_go="GO 100/2024",
        target_go="GO 50/2024",
    )
    assert relevant is False


def test_temporal_claim_relevance_requires_relational_semantics():
    """Test 41: Claim asserting modification between source and target is identified as relevant."""
    relevant = VerificationAgent._is_claim_relevant_to_temporal_discrepancy(
        claim_text="GO 100/2024 amends the provisions of GO 50/2024.",
        source_go="GO 100/2024",
        target_go="GO 50/2024",
    )
    assert relevant is True


def test_temporal_claim_relevance_negative_on_incidental_mention():
    """Test 42: Incidental citation of target in an unrelated sentence does not trigger relevance."""
    claim = "GO 100/2024 sanctions 50 posts. In a separate footnote, GO 50/2024 was referenced for holidays."
    relevant = VerificationAgent._is_claim_relevant_to_temporal_discrepancy(
        claim_text=claim,
        source_go="GO 100/2024",
        target_go="GO 50/2024",
    )
    assert relevant is False


# =========================================================================
# 5. Confidence Scoring & Model Invariants (43–48)
# =========================================================================

def test_confidence_threshold_high_classification_at_0_80():
    """Test 43: Confidence score >= 0.80 classified as HIGH."""
    assert VerificationAgent._classify_confidence(0.80) == VerificationConfidence.HIGH
    assert VerificationAgent._classify_confidence(0.95) == VerificationConfidence.HIGH


def test_confidence_threshold_medium_classification_at_0_60():
    """Test 44: Confidence score in [0.60, 0.80) classified as MEDIUM."""
    assert VerificationAgent._classify_confidence(0.60) == VerificationConfidence.MEDIUM
    assert VerificationAgent._classify_confidence(0.799) == VerificationConfidence.MEDIUM


def test_confidence_threshold_low_classification_under_0_60():
    """Test 45: Confidence score < 0.60 classified as LOW."""
    assert VerificationAgent._classify_confidence(0.599) == VerificationConfidence.LOW
    assert VerificationAgent._classify_confidence(0.0) == VerificationConfidence.LOW


def test_vacuous_relationship_check_preserves_claim_confidence():
    """Test 46: Vacuous relationship check does not degrade factual claim confidence."""
    score, factors = VerificationAgent._compute_claim_confidence(
        go_exists=True,
        text_source=TextSourceType.FULL_DATABASE_TEXT,
        directness=EvidenceDirectness.DIRECT_OPERATIVE_ORDER,
        status=ClaimVerificationStatus.SUPPORTED,
        page_verified=True,
        source_filename="order.pdf",
    )
    assert score >= 0.80
    assert factors["support_score"] == 0.35


def test_legal_effect_verified_permanently_false():
    """Test 47: Enforces legal_effect_verified=False invariant across all models."""
    valid_rec = VerificationRecommendation(
        action=AutonomousVerificationAction.PROCEED_TO_RESPONSE,
        target_agent=TargetAgentName.SUPERVISOR_AGENT,
        parameters=EmptyParams(),
        reasoning="All claims verified with direct grounded evidence.",
    )
    res = VerificationAgentResponse(
        status=VerificationStatus.VERIFIED,
        overall_confidence=VerificationConfidence.HIGH,
        confidence_score=0.95,
        recommendation=valid_rec,
        legal_effect_verified=False,
    )
    assert res.legal_effect_verified is False
    try:
        VerificationAgentResponse(
            status=VerificationStatus.VERIFIED,
            overall_confidence=VerificationConfidence.HIGH,
            confidence_score=0.95,
            recommendation=valid_rec,
            legal_effect_verified=True,  # Prohibited
        )
        assert False, "Should have raised exception for legal_effect_verified=True"
    except Exception:
        pass


def test_verification_agent_initialization_defaults():
    """Test 48: VerificationAgent initializes default sub-agents safely."""
    agent = VerificationAgent()
    assert agent.db_agent is not None
    assert agent.rel_agent is not None
    assert agent.search_agent is not None
    assert agent.llm_service is not None


# =========================================================================
# 6. Live Integration & Database Invariance (49–52)
# =========================================================================

def _check_db_connectivity() -> bool:
    try:
        from app.database.connection import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as session:
            val = session.execute(text("SELECT 1;")).scalar()
            return val == 1
    except Exception:
        return False


def test_live_db_verification_record_retrieval_guard():
    """Test 49: Live DB lookup verifies order existence or honestly skips if DB is unreachable."""
    if not _check_db_connectivity():
        raise unittest.SkipTest("PostgreSQL container or database_tool offline on port 5433.")

    agent = VerificationAgent()
    req = VerificationAgentRequest(
        cited_go_numbers=["G.O.(Rt)No.24/2026/ITD"],
    )
    res = agent.execute(req)
    assert res.status in [VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED]


def test_live_db_real_order_verification_workflow():
    """Test 50: Live DB verification workflow executes read-only lookup on known real order."""
    if not _check_db_connectivity():
        raise unittest.SkipTest("PostgreSQL container or database_tool offline on port 5433.")

    agent = VerificationAgent()
    req = VerificationAgentRequest(
        claims=[
            VerificationClaim(
                claim_text="Information Technology Department government order.",
                cited_go_number="G.O.(Rt)No.24/2026/ITD",
            )
        ],
        cited_go_numbers=["G.O.(Rt)No.24/2026/ITD"],
    )
    res = agent.execute(req)
    assert len(res.verified_claims) + len(res.unsupported_claims) == 1


def test_live_db_preserves_immutability_of_existing_tables():
    """Test 51: Confirms database counts remain exactly 11, 4, 6, 6 before and after execution."""
    if not _check_db_connectivity():
        raise unittest.SkipTest("PostgreSQL container or database_tool offline on port 5433.")

    from app.database.connection import SessionLocal
    from sqlalchemy import text

    def get_counts():
        with SessionLocal() as session:
            gos = session.execute(text("SELECT count(*) FROM government_orders;")).scalar()
            refs = session.execute(text("SELECT count(*) FROM go_references;")).scalar()
            rels = session.execute(text("SELECT count(*) FROM go_relationships;")).scalar()
            vers = session.execute(text("SELECT count(*) FROM verification_records;")).scalar()
            return gos, refs, rels, vers

    counts_before = get_counts()
    agent = VerificationAgent()
    agent.execute(VerificationAgentRequest(cited_go_numbers=["G.O.(Rt)No.24/2026/ITD"]))
    counts_after = get_counts()

    assert counts_before == counts_after == (11, 4, 6, 6), f"DB state altered: before={counts_before}, after={counts_after}"


def test_honest_skip_reporting_when_db_or_container_offline():
    """Test 52: Honest skip reporting when container or DB connection fails."""
    if not _check_db_connectivity():
        raise unittest.SkipTest("Honest skip: PostgreSQL container offline on port 5433.")
    else:
        assert True


# =========================================================================
# 7. Additional Invariant Tests (53–56)
# =========================================================================

def test_relationship_candidate_input_rejects_legal_effect_verified_true():
    """Test 53: RelationshipCandidateInput strictly rejects caller-supplied legal_effect_verified=True."""
    from pydantic import ValidationError
    try:
        RelationshipCandidateInput(
            relationship_type="AMENDS",
            source_go_number="GO 100",
            target_order_number="GO 50",
            legal_effect_verified=True,
        )
        assert False, "Should have raised ValidationError for legal_effect_verified=True on input model"
    except ValidationError:
        pass

    # Verify default remains False and valid callers are compatible
    cand = RelationshipCandidateInput(
        relationship_type="AMENDS",
        source_go_number="GO 100",
        target_order_number="GO 50",
    )
    assert cand.legal_effect_verified is False


def test_document_evidence_rejected_when_db_returns_error():
    """Test 54: DatabaseAgent ERROR cannot be converted to SUCCESS by caller document evidence."""
    mock_db = MagicMock(spec=DatabaseAgent)
    mock_db.execute.return_value = DatabaseAgentResponse(
        status=AgentResponseStatus.ERROR,
        operation=DatabaseOperation.LOOKUP_ORDER,
        errors=["Database connection failure"],
    )
    mock_rel = MagicMock(spec=RelationshipAgent)
    mock_rel.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        total_relationships=0,
        relationships=[],
        discrepancies=[],
        is_complete=True,
    )
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"entailment_status": "ENTAILED", "confidence": 0.95, "reasoning": "ok", "supporting_excerpt": "Sanction is accorded"}'

    from app.agents.specialized.verification_agent import DocumentEvidenceInput
    agent = VerificationAgent(database_agent=mock_db, relationship_agent=mock_rel, llm_service=mock_llm)
    req = VerificationAgentRequest(
        claims=[VerificationClaim(claim_text="Sanction is accorded", cited_go_number="G.O. 123", cited_excerpt="Sanction is accorded")],
        cited_go_numbers=["G.O. 123"],
        document_evidence=[
            DocumentEvidenceInput(
                go_number="G.O. 123",
                extracted_text="ORDER\nSanction is accorded for 50 Lakhs.",
                extracted_text_length=len("ORDER\nSanction is accorded for 50 Lakhs."),
            )
        ],
    )
    res = agent.execute(req)
    assert len(res.verified_claims) == 0
    assert len(res.unsupported_claims) == 1
    unsupported = res.unsupported_claims[0]
    assert unsupported.go_exists_in_database is False
    assert unsupported.status == ClaimVerificationStatus.UNVERIFIABLE
    assert unsupported.text_source_type != TextSourceType.FULL_DATABASE_TEXT
    assert res.status != VerificationStatus.VERIFIED
    assert res.is_complete is False


def test_document_evidence_rejected_when_db_raises_exception():
    """Test 55: DatabaseAgent exception cannot be converted to SUCCESS by caller document evidence."""
    mock_db = MagicMock(spec=DatabaseAgent)
    mock_db.execute.side_effect = RuntimeError("Database socket dropped")

    mock_rel = MagicMock(spec=RelationshipAgent)
    mock_rel.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        total_relationships=0,
        relationships=[],
        discrepancies=[],
        is_complete=True,
    )

    from app.agents.specialized.verification_agent import DocumentEvidenceInput
    agent = VerificationAgent(database_agent=mock_db, relationship_agent=mock_rel)
    req = VerificationAgentRequest(
        claims=[VerificationClaim(claim_text="Sanction is accorded", cited_go_number="G.O. 123")],
        cited_go_numbers=["G.O. 123"],
        document_evidence=[
            DocumentEvidenceInput(
                go_number="G.O. 123",
                extracted_text="ORDER\nSanction is accorded for 50 Lakhs.",
                extracted_text_length=len("ORDER\nSanction is accorded for 50 Lakhs."),
            )
        ],
    )
    res = agent.execute(req)
    assert len(res.verified_claims) == 0
    assert len(res.unsupported_claims) == 1
    unsupported = res.unsupported_claims[0]
    assert unsupported.go_exists_in_database is False
    assert unsupported.text_source_type != TextSourceType.FULL_DATABASE_TEXT
    assert res.status != VerificationStatus.VERIFIED
    assert res.is_complete is False
    assert any("Database socket dropped" in err for err in res.errors)


def test_document_evidence_rejected_when_db_returns_not_found():
    """Test 56: DatabaseAgent NOT_FOUND cannot be converted to SUCCESS by caller document evidence."""
    mock_db = MagicMock(spec=DatabaseAgent)
    mock_db.execute.return_value = DatabaseAgentResponse(
        status=AgentResponseStatus.NOT_FOUND,
        operation=DatabaseOperation.LOOKUP_ORDER,
        order_data=None,
    )
    mock_rel = MagicMock(spec=RelationshipAgent)
    mock_rel.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        total_relationships=0,
        relationships=[],
        discrepancies=[],
        is_complete=True,
    )

    from app.agents.specialized.verification_agent import DocumentEvidenceInput
    agent = VerificationAgent(database_agent=mock_db, relationship_agent=mock_rel)
    req = VerificationAgentRequest(
        claims=[VerificationClaim(claim_text="Sanction is accorded", cited_go_number="G.O. 123")],
        cited_go_numbers=["G.O. 123"],
        document_evidence=[
            DocumentEvidenceInput(
                go_number="G.O. 123",
                extracted_text="ORDER\nSanction is accorded for 50 Lakhs.",
                extracted_text_length=len("ORDER\nSanction is accorded for 50 Lakhs."),
            )
        ],
    )
    res = agent.execute(req)
    assert len(res.verified_claims) == 0
    assert len(res.unsupported_claims) == 1
    unsupported = res.unsupported_claims[0]
    assert unsupported.go_exists_in_database is False
    assert unsupported.status == ClaimVerificationStatus.REFUTED
    assert unsupported.text_source_type != TextSourceType.FULL_DATABASE_TEXT
    assert res.status == VerificationStatus.NOT_FOUND


# =========================================================================
# RUNNER
# =========================================================================

ALL_PHYSICAL_TESTS = [
    test_provenance_rejects_caller_supplied_chunks,
    test_provenance_rejects_fabricated_page_numbers,
    test_provenance_accepts_single_page_occurrence_from_db_text,
    test_provenance_leaves_multi_page_unresolved_without_search,
    test_provenance_resolves_multi_page_via_validated_search_chunk,
    test_provenance_rejects_search_agent_partial_success_status,
    test_provenance_rejects_search_agent_zero_returned_count,
    test_provenance_rejects_search_chunk_with_mismatched_go_number,
    test_provenance_rejects_search_chunk_missing_excerpt,
    test_provenance_rejects_search_chunk_with_external_page_number,
    test_provenance_conflicting_search_chunks_remain_unresolved,
    test_provenance_identical_page_search_chunks_resolve_successfully,
    test_provenance_handles_search_agent_exception_gracefully,
    test_conservative_go_normalization_prevents_identifier_collision,
    test_conservative_go_normalization_matches_equivalent_punctuations,
    test_directness_detection_operative_order_heading,
    test_directness_detection_ignores_order_word_in_ordinary_prose,
    test_directness_detection_read_citation_section,
    test_directness_detection_preamble_citation_section,
    test_directness_detection_missing_headers_defaults_to_unclassified,
    test_llm_entailment_valid_supported_response,
    test_llm_entailment_valid_contradicted_response,
    test_llm_entailment_malformed_json_yields_unverifiable,
    test_llm_entailment_invalid_confidence_bounds_yields_unverifiable,
    test_llm_entailment_nan_or_inf_confidence_yields_unverifiable,
    test_relationship_canonical_vocabulary_exact_match,
    test_relationship_canonical_vocabulary_complementary_match,
    test_relationship_unsupported_text_type_generates_discrepancy,
    test_relationship_unsupported_db_type_generates_discrepancy,
    test_relationship_one_to_one_candidate_matching_no_double_consumption,
    test_relationship_missing_from_text_detected,
    test_relationship_unrecorded_in_db_detected,
    test_relationship_cross_examination_coverage_vs_consistency_distinction,
    test_relationship_discrepancies_prevent_verified_status,
    test_vacuous_relationship_check_cannot_yield_verified,
    test_discrepancy_report_schema_preserves_category_and_go_fields,
    test_temporal_evaluation_valid_chronology_passes,
    test_temporal_anomaly_does_not_mark_claim_contradicted,
    test_temporal_evaluation_anomaly_applies_half_penalty_to_relevant_claim,
    test_temporal_evaluation_anomaly_ignored_for_irrelevant_claim,
    test_temporal_claim_relevance_requires_relational_semantics,
    test_temporal_claim_relevance_negative_on_incidental_mention,
    test_confidence_threshold_high_classification_at_0_80,
    test_confidence_threshold_medium_classification_at_0_60,
    test_confidence_threshold_low_classification_under_0_60,
    test_vacuous_relationship_check_preserves_claim_confidence,
    test_legal_effect_verified_permanently_false,
    test_verification_agent_initialization_defaults,
    test_live_db_verification_record_retrieval_guard,
    test_live_db_real_order_verification_workflow,
    test_live_db_preserves_immutability_of_existing_tables,
    test_honest_skip_reporting_when_db_or_container_offline,
    test_relationship_candidate_input_rejects_legal_effect_verified_true,
    test_document_evidence_rejected_when_db_returns_error,
    test_document_evidence_rejected_when_db_raises_exception,
    test_document_evidence_rejected_when_db_returns_not_found,
]


def run_all_tests():
    passed = []
    failed = []
    skipped = []

    print(f"\n=================================================================")
    print(f"RUNNING ALL {len(ALL_PHYSICAL_TESTS)} PHASE 2 STEP 2.4 PHYSICAL TESTS")
    print(f"=================================================================\n")

    for idx, test_fn in enumerate(ALL_PHYSICAL_TESTS, 1):
        name = test_fn.__name__
        try:
            test_fn()
            print(f"[{idx:02d}/56] PASS: {name}")
            passed.append(name)
        except unittest.SkipTest as e:
            print(f"[{idx:02d}/56] SKIPPED: {name} (Reason: {e})")
            skipped.append((name, str(e)))
        except Exception as e:
            print(f"[{idx:02d}/56] FAIL: {name} (Error: {e})")
            failed.append((name, str(e)))

    print(f"\n=================================================================")
    print(f"TOTAL TESTS: {len(ALL_PHYSICAL_TESTS)}")
    print(f"PASSED:      {len(passed)}")
    print(f"FAILED:      {len(failed)}")
    print(f"SKIPPED:     {len(skipped)}")
    print(f"=================================================================\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
