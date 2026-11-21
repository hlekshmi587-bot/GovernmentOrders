"""
Focused Offline Regression Test Suite for Phase 5 Scenario 1 Corrections.

Covers:
1. Complete and incomplete GO identifiers (formats with and without department suffix).
2. Relevant and unrelated search fallback candidates (meaningful alignment).
3. Rejection of generic token matches and positive semantic scores without meaningful query alignment.
4. Queries that genuinely require relationship investigation vs queries that do not.
5. Evidence provenance preservation using genuine model fields only (no unsupported synthetic keys).
6. Page-level verification isolation: verified page vs unverified page never merged or falsely marked verified.
7. Direction checking: conflicting directions and missing-versus-present directions never merged.
8. Deduplication boundaries (reversed directions, distinct relationship types).
9. Database structural matching & ResponseAgent metadata rendering.
"""

import unittest
from unittest.mock import MagicMock
from app.agents.tools.database_tool import DatabaseTool
from app.agents.workflow.graph_builder import WorkflowNodes
from app.agents.specialized.response_agent import ResponseAgent, ResponseAgentInput
from app.agents.specialized.verification_agent import RelationshipVerificationDetail


class TestPhase5Scenario1Corrections(unittest.TestCase):
    """Offline regression tests for Scenario 1 corrections."""

    # -------------------------------------------------------------------------
    # 1. Complete and Incomplete GO Identifiers
    # -------------------------------------------------------------------------
    def test_complete_go_identifier_with_department(self):
        """Valid format with department suffix is accepted."""
        self.assertTrue(WorkflowNodes._is_valid_go_identifier("24/2026/ITD"))
        self.assertTrue(WorkflowNodes._is_valid_go_identifier("G.O.(Rt)No.24/2026/ITD"))
        self.assertTrue(WorkflowNodes._is_valid_go_identifier("G.O.(P)No.54/2026/FIN"))
        self.assertTrue(WorkflowNodes._is_valid_go_identifier("G.O.(Ms)No.123/2024/GAD"))

        p = DatabaseTool._parse_go_structure("G.O.(Rt)No.24/2026/ITD")
        self.assertIsNotNone(p)
        self.assertEqual(p, ("24", "2026", "ITD"))

    def test_complete_go_identifier_without_department(self):
        """Valid format without department suffix is accepted and not rejected."""
        self.assertTrue(WorkflowNodes._is_valid_go_identifier("114/2025"))
        self.assertTrue(WorkflowNodes._is_valid_go_identifier("G.O.114/2025"))
        self.assertTrue(WorkflowNodes._is_valid_go_identifier("G.O.(Ms)No.114/2025"))

        p = DatabaseTool._parse_go_structure("G.O.114/2025")
        self.assertIsNotNone(p)
        self.assertEqual(p, ("114", "2025", ""))

    def test_incomplete_fragments_strictly_rejected(self):
        """Bare numbers, bare departments, and bare prefixes are rejected."""
        self.assertFalse(WorkflowNodes._is_valid_go_identifier("24"))
        self.assertFalse(WorkflowNodes._is_valid_go_identifier("ITD"))
        self.assertFalse(WorkflowNodes._is_valid_go_identifier("2026"))
        self.assertFalse(WorkflowNodes._is_valid_go_identifier("G.O."))
        self.assertFalse(WorkflowNodes._is_valid_go_identifier("No.24"))
        self.assertFalse(WorkflowNodes._is_valid_go_identifier(""))
        self.assertFalse(WorkflowNodes._is_valid_go_identifier(None))

        self.assertIsNone(DatabaseTool._parse_go_structure("24"))
        self.assertIsNone(DatabaseTool._parse_go_structure("ITD"))
        self.assertIsNone(DatabaseTool._parse_go_structure("G.O."))

    # -------------------------------------------------------------------------
    # 2. Search Fallback Candidate Relevance vs Unrelated Candidates
    # -------------------------------------------------------------------------
    def test_relevant_search_fallback_candidate_accepted(self):
        """Candidate with matching department, substantive query terms, and relational context is accepted."""
        query = "Which order modified the IT software export policy?"
        candidate = {
            "go_number": "G.O.(Ms)No.10/2025/ITD",
            "department": "Information Technology Department",
            "text_snippet": "In partial modification of software export policy 2024, the government hereby orders...",
            "source_filename": "GO_10_2025_ITD.pdf",
        }
        state = {"identified_departments": ["ITD"]}
        self.assertTrue(WorkflowNodes._is_search_hit_meaningfully_aligned(candidate, query, state))

    def test_unrelated_search_fallback_candidate_rejected(self):
        """Candidate from conflicting department is rejected even if it is top hit."""
        query = "Which order modified the IT software export policy?"
        candidate = {
            "go_number": "G.O.(P)No.54/2026/FIN",
            "department": "Finance Department",
            "text_snippet": "In partial modification of pension rules...",
            "source_filename": "GO_54_2026_FIN.pdf",
        }
        state = {"identified_departments": ["ITD"]}
        self.assertFalse(WorkflowNodes._is_search_hit_meaningfully_aligned(candidate, query, state))

    # -------------------------------------------------------------------------
    # 3. Generic Token Matches & Positive Semantic Scores Without Meaningful Alignment
    # -------------------------------------------------------------------------
    def test_generic_tokens_alone_not_sufficient(self):
        """Candidate matching only generic administrative stopwords is rejected."""
        query = "Show details of the Kerala government order dated 2026"
        candidate = {
            "go_number": "G.O.(Rt)No.99/2026/GAD",
            "department": "General Administration",
            "text_snippet": "Government of Kerala order dated 15-01-2026 regarding general office administration.",
            "source_filename": "GO_99_2026_GAD.pdf",
            "semantic_score": 0.89,
        }
        self.assertFalse(WorkflowNodes._is_search_hit_meaningfully_aligned(candidate, query))

    def test_positive_semantic_score_without_substantive_alignment_rejected(self):
        """High semantic score without query subject alignment does not establish relevance."""
        query = "Find orders revoking the coastal aquaculture regulation"
        candidate = {
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "department": "Information Technology",
            "text_snippet": "Sanction is accorded for IT infrastructure upgrade.",
            "source_filename": "GO_24_2026_ITD.pdf",
            "semantic_score": 0.95,
        }
        self.assertFalse(WorkflowNodes._is_search_hit_meaningfully_aligned(candidate, query))

    # -------------------------------------------------------------------------
    # 4. Intent Restriction & Prioritized Focal Scoping
    # -------------------------------------------------------------------------
    def test_fallback_activated_only_for_relationship_intents(self):
        """Fallback evaluation operates under relationship intents, not general lookups."""
        builder = WorkflowNodes(supervisor_agent=MagicMock())
        search_hit = {
            "go_number": "G.O.(Ms)No.10/2025/ITD",
            "department": "Information Technology",
            "text_snippet": "In partial modification of software export policy 2024...",
            "source_filename": "GO_10_2025_ITD.pdf",
        }

        # Case A: Intent = GO_LOOKUP (No focal GO, but lookup intent) -> fallback bypassed
        state_lookup = {
            "identified_go_numbers": [],
            "discovered_go_numbers": [],
            "search_results": [search_hit],
            "intent": "GO_LOOKUP",
            "user_query": "Which order modified the software export policy?",
            "step_id": "step_1",
        }
        res = builder.verification_node(state_lookup)
        called_input = builder.supervisor.create_trace_item.call_args[1]["tool_input"]
        self.assertEqual(called_input["cited_gos"], [])

        # Case B: Intent = INCOMING_RELATIONSHIPS -> fallback accepts aligned hit
        builder.verification_agent.execute = MagicMock(return_value=MagicMock(
            status=MagicMock(value="VERIFIED"),
            confidence_score=0.9,
            evidence_sufficient=True,
            model_dump=MagicMock(return_value={}),
        ))
        state_rel = {
            "identified_go_numbers": [],
            "discovered_go_numbers": [],
            "search_results": [search_hit],
            "intent": "INCOMING_RELATIONSHIPS",
            "user_query": "Which order modified the software export policy?",
            "step_id": "step_2",
        }
        res_rel = builder.verification_node(state_rel)
        called_input_rel = builder.supervisor.create_trace_item.call_args[1]["tool_input"]
        self.assertEqual(called_input_rel["cited_gos"], ["G.O.(Ms)No.10/2025/ITD"])

    def test_explicit_identified_go_prevents_collateral_search_hits_as_focal(self):
        """When an explicit GO is requested, collateral search hits are never treated as focal."""
        builder = WorkflowNodes(supervisor_agent=MagicMock())
        builder.verification_agent.execute = MagicMock(return_value=MagicMock(
            status=MagicMock(value="VERIFIED"),
            confidence_score=0.9,
            evidence_sufficient=True,
            model_dump=MagicMock(return_value={}),
        ))
        state = {
            "identified_go_numbers": ["G.O.(Rt)No.24/2026/ITD"],
            "discovered_go_numbers": ["G.O.(P)No.54/2026/FIN"],  # Collateral hit
            "search_results": [{"go_number": "G.O.(P)No.54/2026/FIN", "department": "FIN"}],
            "intent": "GO_LOOKUP",
            "user_query": "Show details of G.O.(Rt)No.24/2026/ITD",
            "step_id": "step_1",
        }
        builder.verification_node(state)
        called_input = builder.supervisor.create_trace_item.call_args[1]["tool_input"]
        # Only explicit focal order is passed, not the collateral FIN order
        self.assertEqual(called_input["cited_gos"], ["G.O.(Rt)No.24/2026/ITD"])

    # -------------------------------------------------------------------------
    # 5. Provenance Preservation with Real Model Fields (No Extra Synthetic Keys)
    # -------------------------------------------------------------------------
    def test_deduplication_preserves_provenance_and_validates_against_model(self):
        """Deduplicated items use genuine fields and validate against RelationshipVerificationDetail(extra='forbid')."""
        rel1 = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "candidate_relationship_type": "MODIFIES",
            "relationship_type": "MODIFIES",
            "evidence_text": "This order modifies Clause 4.",
            "page_number": 1,
            "notes": "Source document: doc_a.pdf | Primary clause edit",
        }
        rel2 = {
            "source_go_number": "G.O.(Rt)No.24/2026/ITD",
            "target_go_number": "G.O.(Ms)No.12/2024/ITD",
            "candidate_relationship_type": "MODIFIES",
            "relationship_type": "MODIFIES",
            "evidence_text": "In continuation of earlier notification.",
            "page_number": 1,
            "notes": "Source document: doc_b.pdf | Notification continuity",
        }

        deduped = WorkflowNodes._deduplicate_supported_relationships([rel1, rel2])
        self.assertEqual(len(deduped), 1)
        item = deduped[0]

        # Canonical longer GO numbers preserved
        self.assertEqual(item["source_go_number"], "G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(item["target_go_number"], "G.O.(Ms)No.12/2024/ITD")

        # Distinct excerpts merged into standard evidence_text field
        self.assertIn("This order modifies Clause 4.", item["evidence_text"])
        self.assertIn("In continuation of earlier notification.", item["evidence_text"])

        # Notes merged into standard notes field
        self.assertIn("Source document: doc_a.pdf", item["notes"])
        self.assertIn("Source document: doc_b.pdf", item["notes"])
        self.assertIn("Primary clause edit", item["notes"])
        self.assertIn("Notification continuity", item["notes"])

        # Real model validation check: Must not raise ValidationError (no extra forbidden fields)
        model_obj = RelationshipVerificationDetail.model_validate(item)
        self.assertEqual(model_obj.source_go_number, "G.O.(Rt)No.24/2026/ITD")

    # -------------------------------------------------------------------------
    # 6. Page-Level Verification Isolation
    # -------------------------------------------------------------------------
    def test_verified_and_unverified_pages_never_merged(self):
        """Records with different page_number_verified status must NOT be merged."""
        rel_verified = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
            "page_number": 1,
            "page_number_verified": True,
        }
        rel_unverified = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
            "page_number": 3,
            "page_number_verified": False,
        }
        deduped = WorkflowNodes._deduplicate_supported_relationships([rel_verified, rel_unverified])
        # Must retain both separately to prevent unverified page 3 from being marked verified
        self.assertEqual(len(deduped), 2)

        # Formatted output verification: Verified page and unverified page formatted accurately
        agent = ResponseAgent()
        out1 = agent._format_verified_item(deduped[0])
        out2 = agent._format_verified_item(deduped[1])
        formatted = "\n".join(out1 + out2)
        self.assertIn("Page: 1 (Verified)", formatted)
        self.assertIn("Page: 3 (Unverified)", formatted)

    # -------------------------------------------------------------------------
    # 7. Direction Checking During Deduplication
    # -------------------------------------------------------------------------
    def test_missing_vs_present_direction_not_merged(self):
        """A record with direction=None and a record with direction='OUTGOING' must not be merged."""
        rel_known = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
            "direction": "OUTGOING",
        }
        rel_missing = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
            "direction": None,
        }
        deduped = WorkflowNodes._deduplicate_supported_relationships([rel_known, rel_missing])
        self.assertEqual(len(deduped), 2)

    def test_conflicting_directions_not_merged(self):
        """Conflicting directions ('OUTGOING' vs 'INCOMING') must not be merged."""
        rel_out = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
            "direction": "OUTGOING",
        }
        rel_in = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
            "direction": "INCOMING",
        }
        deduped = WorkflowNodes._deduplicate_supported_relationships([rel_out, rel_in])
        self.assertEqual(len(deduped), 2)

    def test_reversed_direction_not_merged(self):
        """Reversed source and target must not be merged."""
        rel_fwd = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
        }
        rel_rev = {
            "source_go_number": "12/2024/ITD",
            "target_go_number": "24/2026/ITD",
            "relationship_type": "MODIFIES",
        }
        deduped = WorkflowNodes._deduplicate_supported_relationships([rel_fwd, rel_rev])
        self.assertEqual(len(deduped), 2)

    def test_different_relationship_types_not_merged(self):
        """MODIFIES and REVOKES must not be merged."""
        rel_mod = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "MODIFIES",
        }
        rel_rev = {
            "source_go_number": "24/2026/ITD",
            "target_go_number": "12/2024/ITD",
            "relationship_type": "REVOKES",
        }
        deduped = WorkflowNodes._deduplicate_supported_relationships([rel_mod, rel_rev])
        self.assertEqual(len(deduped), 2)

    # -------------------------------------------------------------------------
    # 8. Database Structural Matching & Safe Match Classification
    # -------------------------------------------------------------------------
    def test_database_structural_equivalence(self):
        """Structurally equivalent GO identifiers match; departmental mismatches do not."""
        self.assertTrue(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "G.O.(Rt)No.24/2026/ITD"))
        self.assertTrue(DatabaseTool._are_structurally_equivalent("114/2025", "G.O.114/2025"))
        self.assertFalse(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "24/2026/FIN"))
        self.assertFalse(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "25/2026/ITD"))
        self.assertFalse(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "24/2025/ITD"))
        # Number/year only vs complete order with department: Not structurally equivalent (partial match)
        self.assertFalse(DatabaseTool._are_structurally_equivalent("114/2025", "G.O.(Ms)No.114/2025/ITD"))

    def test_response_agent_renders_order_details_even_without_verification(self):
        """ResponseAgent renders Indexed Database Records whenever document_analysis is populated."""
        agent = ResponseAgent()
        resp_input = ResponseAgentInput(
            user_query="Show details of G.O.(Rt)No.24/2026/ITD",
            verification_status=None,
            document_analysis=[{
                "go_number": "G.O.(Rt)No.24/2026/ITD",
                "department": "Information Technology",
                "issue_date": "2026-01-15",
                "text_preview": "Sanction is accorded for State Data Center upgrade.",
            }],
        )
        output = agent.generate_response(resp_input)
        self.assertIn("### 1. Indexed Database Records", output)
        self.assertIn("G.O.(Rt)No.24/2026/ITD", output)
        self.assertIn("State Data Center upgrade", output)


if __name__ == "__main__":
    unittest.main()
