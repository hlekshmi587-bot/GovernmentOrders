"""
Focused Offline Regression Test Suite for Phase 5 Scenario 2: Incoming Relationships.

Verifies:
1. All specified incoming queries trigger INCOMING_RELATIONSHIPS in heuristic fallback.
2. Existing intent classifications remain unchanged.
3. Incoming lookup for 24/2026/ITD returns zero relationships (unit logic with mock session).
4. Outgoing lookup returns the three existing relationships (unit logic with mock session).
5. Incoming lookup never returns those outgoing relationships.
6. A mocked incoming relationship is displayed in the correct source-to-target direction.
7. When zero incoming relationships exist, the response clearly says none were found while retaining the focal GO metadata.
8. The verification agent remains scoped to the focal GO.
9. legal_effect_verified=False remains strictly unchanged.
10. Scenario 1 corrections continue to work without regression.
"""

import unittest
from unittest.mock import MagicMock, patch
from app.agents.specialized.supervisor_agent import SupervisorAgent, QueryIntent
from app.agents.tools.database_tool import DatabaseTool
from app.agents.specialized.response_agent import ResponseAgent, ResponseAgentInput
from app.agents.workflow.graph_builder import WorkflowNodes


class TestPhase5Scenario2IncomingRelationships(unittest.TestCase):
    """Offline test suite for Scenario 2: Incoming Relationships."""

    def setUp(self):
        self.supervisor = SupervisorAgent(llm_service=MagicMock())
        self.db_tool = DatabaseTool()
        self.response_agent = ResponseAgent()

    # -------------------------------------------------------------------------
    # 1. Supervisor Heuristic Fallback Intent Classification
    # -------------------------------------------------------------------------
    def test_incoming_queries_trigger_incoming_relationships_intent(self):
        """Verify all approved incoming queries trigger INCOMING_RELATIONSHIPS in heuristic fallback."""
        test_queries = [
            "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "Show incoming relationships for 24/2026/ITD",
            "Find incoming relationship for G.O.(Rt)No.24/2026/ITD",
            "Are there incoming orders for G.O.(Rt)No.24/2026/ITD",
            "List orders that modify G.O.(Rt)No.24/2026/ITD",
            "Show orders that reference G.O.(Rt)No.24/2026/ITD",
            "Which later order modified G.O.(Rt)No.24/2026/ITD?",
            "Who modified G.O.(Rt)No.24/2026/ITD?",
            "Any subsequent order for G.O.(Rt)No.24/2026/ITD?",
            "Find later government order for G.O.(Rt)No.24/2026/ITD",
        ]
        for query in test_queries:
            plan = self.supervisor._heuristic_fallback_plan(query)
            self.assertEqual(
                plan.intent,
                QueryIntent.INCOMING_RELATIONSHIPS,
                f"Query '{query}' failed to classify as INCOMING_RELATIONSHIPS (got {plan.intent})",
            )
            # Step 1 should be database_agent targeting direction=INCOMING
            self.assertTrue(len(plan.execution_steps) >= 1)
            first_step = plan.execution_steps[0]
            self.assertEqual(first_step.target_agent, "database_agent")
            self.assertEqual(first_step.target_tool, "database_tool")
            self.assertEqual(first_step.tool_parameters.get("direction"), "INCOMING")

    # -------------------------------------------------------------------------
    # 2. Existing Intent Classifications Remain Unchanged
    # -------------------------------------------------------------------------
    def test_existing_intents_remain_unchanged(self):
        """Verify GO_LOOKUP, EXPLAIN_ORDER, OUTGOING_RELATIONSHIPS, and KEYWORD_SEARCH intents remain intact."""
        # Exact GO lookup
        p_lookup = self.supervisor._heuristic_fallback_plan("Show details of G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(p_lookup.intent, QueryIntent.GO_LOOKUP)

        # Explain order
        p_explain = self.supervisor._heuristic_fallback_plan("Explain the clause of G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(p_explain.intent, QueryIntent.EXPLAIN_ORDER)

        # Outgoing relationships
        p_outgoing = self.supervisor._heuristic_fallback_plan("Which orders are modified by G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(p_outgoing.intent, QueryIntent.OUTGOING_RELATIONSHIPS)

        # General search
        p_search = self.supervisor._heuristic_fallback_plan("Show guidelines on remote work policy")
        self.assertEqual(p_search.intent, QueryIntent.KEYWORD_SEARCH)

    # -------------------------------------------------------------------------
    # 3. Database Incoming vs Outgoing Separation (Offline Mocked Session)
    # -------------------------------------------------------------------------
    @patch("app.agents.tools.database_tool.SessionLocal")
    def test_incoming_lookup_returns_zero_relationships(self, mock_session_cls):
        """Incoming lookup for 24/2026/ITD returns zero relationships."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        # Mock resolve order returning order 24/2026/ITD
        mock_order = MagicMock()
        mock_order.go_number = "G.O.(Rt)No.24/2026/ITD"
        mock_order.id = 1
        with patch.object(self.db_tool, "_resolve_order", return_value={"status": "SUCCESS", "order": mock_order}):
            # Mock empty incoming query result
            mock_session.execute.return_value.all.return_value = []
            res = self.db_tool.get_incoming_relationships("24/2026/ITD")

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["total_relationships"], 0)
            self.assertEqual(res["relationships"], [])
            self.assertEqual(res["direction"], "INCOMING")

    @patch("app.agents.tools.database_tool.SessionLocal")
    def test_outgoing_lookup_returns_three_relationships(self, mock_session_cls):
        """Outgoing lookup returns the three existing relationships (114/2025, 151/2025, 264/2025)."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        mock_order = MagicMock()
        mock_order.go_number = "G.O.(Rt)No.24/2026/ITD"
        mock_order.id = 1

        rel1 = MagicMock(id=1, relationship_type="MODIFIES", referenced_go_number="G.O.(Rt)No.114/2025/ITD", matched_phrase=None, evidence_text="modifies 114", page_number=1, confidence="HIGH", extraction_method="RULE", review_required=False)
        rel2 = MagicMock(id=2, relationship_type="MODIFIES", referenced_go_number="G.O.(Rt)No.151/2025/ITD", matched_phrase=None, evidence_text="modifies 151", page_number=1, confidence="HIGH", extraction_method="RULE", review_required=False)
        rel3 = MagicMock(id=3, relationship_type="MODIFIES", referenced_go_number="G.O.(Rt)No.264/2025/ITD", matched_phrase=None, evidence_text="modifies 264", page_number=1, confidence="HIGH", extraction_method="RULE", review_required=False)

        with patch.object(self.db_tool, "_resolve_order", return_value={"status": "SUCCESS", "order": mock_order}):
            mock_session.scalars.return_value.all.return_value = [rel1, rel2, rel3]
            res = self.db_tool.get_outgoing_relationships("24/2026/ITD")

            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["total_relationships"], 3)
            self.assertEqual(res["direction"], "OUTGOING")

            target_numbers = [r["target_go_number"] for r in res["relationships"]]
            self.assertTrue(any("114/2025" in num for num in target_numbers))
            self.assertTrue(any("151/2025" in num for num in target_numbers))
            self.assertTrue(any("264/2025" in num for num in target_numbers))

    @patch("app.agents.tools.database_tool.SessionLocal")
    def test_incoming_lookup_never_returns_outgoing_relationships(self, mock_session_cls):
        """Incoming lookup must never return outgoing relationships or invert them."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        mock_order = MagicMock()
        mock_order.go_number = "G.O.(Rt)No.24/2026/ITD"
        mock_order.id = 1

        with patch.object(self.db_tool, "_resolve_order", return_value={"status": "SUCCESS", "order": mock_order}):
            mock_session.execute.return_value.all.return_value = []
            res_incoming = self.db_tool.get_incoming_relationships("24/2026/ITD")
            incoming_targets = [r.get("target_go_number") for r in res_incoming["relationships"]]
            incoming_sources = [r.get("source_go_number") for r in res_incoming["relationships"]]

            # None of the outgoing target orders should appear
            self.assertNotIn("114/2025/ITD", incoming_targets)
            self.assertNotIn("151/2025/ITD", incoming_targets)
            self.assertNotIn("264/2025/ITD", incoming_targets)
            self.assertEqual(len(incoming_sources), 0)

    # -------------------------------------------------------------------------
    # 4. Partial Matching Collision Guard in Incoming Lookup
    # -------------------------------------------------------------------------
    def test_partial_matching_does_not_collide_on_similar_numbers(self):
        """Ensures searching for 24/2026 does not match 124/2026 or 24/2025."""
        # Structurally, 24/2026/ITD should only match 24/2026/ITD
        self.assertFalse(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "124/2026/ITD"))
        self.assertFalse(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "24/2025/ITD"))
        self.assertFalse(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "24/2026/FIN"))
        self.assertTrue(DatabaseTool._are_structurally_equivalent("24/2026/ITD", "G.O.(Rt)No.24/2026/ITD"))

    # -------------------------------------------------------------------------
    # 5. Mocked Incoming Relationship Display Direction (Source-to-Target)
    # -------------------------------------------------------------------------
    def test_mocked_incoming_relationship_display_direction(self):
        """A mocked incoming relationship must be displayed in the correct source-to-target direction."""
        # Mock an incoming order: 999/2026/ITD modifies 24/2026/ITD
        mock_incoming_finding = {
            "source_go_number": "G.O.(Rt)No.999/2026/ITD",
            "target_go_number": "G.O.(Rt)No.24/2026/ITD",
            "relationship_type": "MODIFIES",
            "evidence_directness": "DIRECT_OPERATIVE_ORDER",
            "page_number": 1,
            "page_number_verified": True,
            "source_filename": "go_rt_999_2026_itd.pdf",
            "claim_id": "CLM-999-01",
        }
        resp_input = ResponseAgentInput(
            user_query="Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            verification_status="VERIFIED",
            verification_confidence_score=0.95,
            legal_effect_verified=False,
            verified_findings=[mock_incoming_finding],
            document_analysis=[{
                "go_number": "G.O.(Rt)No.24/2026/ITD",
                "department": "Information Technology",
                "issue_date": "2026-01-24",
                "text_preview": "Order text preview...",
            }],
        )
        response_text = self.response_agent.generate_response(resp_input)

        # Must display source order pointing to target order
        self.assertIn("`G.O.(Rt)No.999/2026/ITD` $\\rightarrow$ `MODIFIES` $\\rightarrow$ `G.O.(Rt)No.24/2026/ITD`", response_text)
        self.assertIn("Page: 1 (Verified)", response_text)
        self.assertIn("Document: `go_rt_999_2026_itd.pdf`", response_text)
        self.assertIn("Ref: CLM-999-01", response_text)

    # -------------------------------------------------------------------------
    # 6. Zero Incoming Relationships Response Formatting
    # -------------------------------------------------------------------------
    def test_zero_incoming_relationships_response_formatting(self):
        """When zero incoming relationships exist, response says none found while retaining focal GO metadata."""
        resp_input = ResponseAgentInput(
            user_query="Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            intent="INCOMING_RELATIONSHIPS",
            verification_status=None,
            legal_effect_verified=False,
            verified_findings=[],
            partially_verified_supported=[],
            unverified_candidates=[],
            document_analysis=[{
                "go_number": "G.O.(Rt)No.24/2026/ITD",
                "department": "Information Technology",
                "issue_date": "2026-01-24",
                "text_preview": "Sanctioning revised project scope...",
            }],
        )
        response_text = self.response_agent.generate_response(resp_input)

        # Metadata must be present
        self.assertIn("### 1. Indexed Database Records", response_text)
        self.assertIn("`G.O.(Rt)No.24/2026/ITD`", response_text)
        self.assertIn("Information Technology", response_text)
        self.assertIn("2026-01-24", response_text)

        # Clear message stating zero incoming relationships found
        self.assertIn("### 1. Incoming Relationships", response_text)
        self.assertIn("No incoming relationships (subsequent Government Orders modifying, superseding, or referencing this order) were found in the indexed database.", response_text)

    # -------------------------------------------------------------------------
    # 7. Verification Agent Scoping & Invariants
    # -------------------------------------------------------------------------
    def test_verification_agent_scoping_to_focal_go(self):
        """VerificationAgent receives only the focal GO."""
        nodes = WorkflowNodes(supervisor_agent=MagicMock())
        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "identified_go_numbers": ["24/2026/ITD"],
            "discovered_go_numbers": ["114/2025/ITD", "151/2025/ITD"],
            "search_results": [{"go_number": "999/2024/FIN", "department": "Finance"}],
            "workflow_steps": [{"step_number": 1, "target_agent": "verification_agent"}],
            "current_step_index": 0,
        }
        # Mock verification agent response
        mock_verif_res = MagicMock()
        mock_verif_res.status.value = "VERIFIED"
        mock_verif_res.confidence_score = 0.95
        mock_verif_res.evidence_sufficient = True
        mock_verif_res.model_dump.return_value = {"status": "VERIFIED"}
        mock_verif_res.relationships = []
        nodes.verification_agent.execute = MagicMock(return_value=mock_verif_res)

        res_update = nodes.verification_node(state)
        # Check call arguments to nodes.verification_agent.execute
        call_args = nodes.verification_agent.execute.call_args[0][0]
        # Should contain focal order only, not unrelated search results
        self.assertIn("24/2026/ITD", call_args.cited_go_numbers)
        self.assertNotIn("999/2024/FIN", call_args.cited_go_numbers)

    def test_legal_effect_verified_invariant_enforced(self):
        """legal_effect_verified=False must be strictly enforced."""
        with self.assertRaises(ValueError):
            ResponseAgentInput(
                user_query="Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
                legal_effect_verified=True,  # type: ignore
            )

    # -------------------------------------------------------------------------
    # 8. Issue 1 Corrections: LLM Plan Normalization & database_node Inference
    # -------------------------------------------------------------------------
    def test_llm_plan_normalization_for_incoming_relationships(self):
        """SupervisorPlan normalizes missing operation and direction to GET_RELATIONSHIPS and INCOMING."""
        from app.agents.specialized.supervisor_agent import SupervisorPlan, PlannedStep

        plan = SupervisorPlan(
            intent=QueryIntent.INCOMING_RELATIONSHIPS,
            identified_go_numbers=["24/2026/ITD"],
            reasoning="Testing normalization for incoming relationships",
            selected_agents=["database_agent"],
            selected_tools=["database_tool"],
            execution_steps=[
                PlannedStep(
                    step_number=1,
                    action_name="lookup_order",
                    target_agent="database_agent",
                    target_tool="database_tool",
                    tool_parameters={"go_number": "24/2026/ITD"},  # Missing operation and direction
                    expected_outcome="Incoming records",
                )
            ],
            expected_evidence="Incoming relationships",
            next_action="database",
            confidence=0.9,
        )

        step_params = plan.execution_steps[0].tool_parameters
        self.assertEqual(step_params.get("operation"), "GET_RELATIONSHIPS")
        self.assertEqual(step_params.get("direction"), "INCOMING")
        self.assertEqual(step_params.get("go_number"), "24/2026/ITD")

    def test_database_node_safe_operation_inference_incoming(self):
        """database_node infers GET_RELATIONSHIPS and INCOMING when omitted in plan parameters."""
        from app.agents.specialized.database_agent import DatabaseOperation, RelationshipDirection, DatabaseAgentResponse, AgentResponseStatus

        nodes = WorkflowNodes(supervisor_agent=MagicMock())
        mock_db_res = DatabaseAgentResponse(
            status=AgentResponseStatus.SUCCESS,
            operation=DatabaseOperation.GET_RELATIONSHIPS,
            query_identifier="24/2026/ITD",
            relationships=[],
            total_records=0,
        )
        nodes.db_agent.execute = MagicMock(return_value=mock_db_res)
        nodes.db_agent.db_tool.get_order_by_number = MagicMock(return_value={
            "status": "SUCCESS",
            "order": {
                "id": 1,
                "go_number": "G.O.(Rt)No.24/2026/ITD",
                "issue_date": "2026-01-20",
                "department": "Information Technology",
                "source_filename": "go_rt_24_2026_itd.pdf",
                "extracted_text_length": 1500,
                "text_preview": "Sanctioning revised project...",
            }
        })

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": QueryIntent.INCOMING_RELATIONSHIPS,
            "identified_go_numbers": ["24/2026/ITD"],
            "workflow_steps": [{
                "step_number": 1,
                "target_agent": "database_agent",
                "target_tool": "database_tool",
                "tool_parameters": {"go_number": "24/2026/ITD"},  # No operation or direction supplied
            }],
            "current_step_index": 0,
            "document_analysis": [],
        }

        update = nodes.database_node(state)

        # Verify db_agent.execute was called with GET_RELATIONSHIPS and INCOMING
        call_req = nodes.db_agent.execute.call_args[0][0]
        self.assertEqual(call_req.operation, DatabaseOperation.GET_RELATIONSHIPS)
        self.assertEqual(call_req.direction, RelationshipDirection.INCOMING)
        self.assertEqual(call_req.go_number, "24/2026/ITD")

        # Verify focal order metadata was attached to document_analysis
        self.assertIn("document_analysis", update)
        self.assertEqual(len(update["document_analysis"]), 1)
        self.assertEqual(update["document_analysis"][0]["go_number"], "G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(update["document_analysis"][0]["department"], "Information Technology")

    # -------------------------------------------------------------------------
    # 9. Issue 2 Corrections: Outgoing Candidates & Discrepancies Suppression
    # -------------------------------------------------------------------------
    def test_verification_node_excludes_outgoing_candidates_for_incoming_intent(self):
        """verification_node excludes outgoing candidates (where focal order is source) for incoming queries."""
        nodes = WorkflowNodes(supervisor_agent=MagicMock())
        mock_verif_res = MagicMock()
        mock_verif_res.status.value = "VERIFIED"
        mock_verif_res.confidence_score = 0.95
        mock_verif_res.evidence_sufficient = True
        mock_verif_res.model_dump.return_value = {"status": "VERIFIED"}
        mock_verif_res.relationships = []
        nodes.verification_agent.execute = MagicMock(return_value=mock_verif_res)

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": QueryIntent.INCOMING_RELATIONSHIPS,
            "identified_go_numbers": ["24/2026/ITD"],
            "relationship_results": [
                {
                    "source_go_number": "24/2026/ITD",
                    "target_go_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                    "direction": "OUTGOING",
                }
            ],
            "workflow_steps": [{"step_number": 1, "target_agent": "verification_agent"}],
            "current_step_index": 0,
        }

        nodes.verification_node(state)
        call_req = nodes.verification_agent.execute.call_args[0][0]
        # Outgoing candidate must be filtered out for incoming query
        self.assertEqual(len(call_req.relationship_candidates), 0)

    def test_response_node_excludes_outgoing_candidates_and_discrepancies(self):
        """response_node excludes outgoing candidates and discrepancies for incoming queries."""
        nodes = WorkflowNodes(supervisor_agent=MagicMock())
        nodes.response_agent.generate_response = MagicMock(return_value="Final Answer")

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": QueryIntent.INCOMING_RELATIONSHIPS,
            "identified_go_numbers": ["24/2026/ITD"],
            "relationship_results": [
                {
                    "source_go_number": "24/2026/ITD",
                    "target_go_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                }
            ],
            "verification_status": "VERIFIED",
            "verification_results": {
                "confidence_score": 0.95,
                "verified_relationships": [],
                "discrepancies": [
                    {
                        "source_order_number": "24/2026/ITD",
                        "target_order_number": "114/2025/ITD",
                        "description": "UNREGISTERED_INCOMPATIBLE_TYPE",
                    }
                ],
            },
            "document_analysis": [
                {
                    "go_number": "G.O.(Rt)No.24/2026/ITD",
                    "department": "Information Technology",
                    "issue_date": "2026-01-20",
                }
            ],
            "workflow_steps": [{"step_number": 1, "target_agent": "response_agent"}],
            "current_step_index": 0,
        }

        nodes.response_node(state)
        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # Outgoing candidates and discrepancies must be filtered out
        self.assertEqual(len(resp_input.unverified_candidates), 0)
        self.assertEqual(len(resp_input.discrepancies), 0)

    def test_response_agent_suppresses_outgoing_discrepancies_and_shows_zero_incoming(self):
        """ResponseAgent suppresses outgoing findings and formats clear zero incoming relationships notice."""
        resp_input = ResponseAgentInput(
            user_query="Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            intent="INCOMING_RELATIONSHIPS",
            verification_status="VERIFIED",
            verification_confidence_score=0.95,
            legal_effect_verified=False,
            verified_findings=[],
            partially_verified_supported=[],
            # Outgoing candidate and discrepancy passed into input
            unverified_candidates=[
                {
                    "source_go_number": "24/2026/ITD",
                    "target_order_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                }
            ],
            discrepancies=[
                {
                    "source_order_number": "24/2026/ITD",
                    "target_order_number": "114/2025/ITD",
                    "description": "UNREGISTERED_INCOMPATIBLE_TYPE",
                }
            ],
            document_analysis=[{
                "go_number": "G.O.(Rt)No.24/2026/ITD",
                "department": "Information Technology",
                "issue_date": "2026-01-20",
                "text_preview": "Sanctioning revised project...",
            }],
        )

        response_text = self.response_agent.generate_response(resp_input)

        # Metadata must be retained
        self.assertIn("`G.O.(Rt)No.24/2026/ITD`", response_text)
        self.assertIn("Information Technology", response_text)

        # Zero incoming relationships clearly stated
        self.assertIn("### 1. Incoming Relationships", response_text)
        self.assertIn("No incoming relationships (subsequent Government Orders modifying, superseding, or referencing this order) were found in the indexed database.", response_text)

        # Outgoing findings and discrepancies must NOT appear
        self.assertNotIn("### 2. Unverified Relationship Candidates", response_text)
        self.assertNotIn("### 3. Database Discrepancies", response_text)
        self.assertNotIn("114/2025/ITD", response_text)
        self.assertNotIn("UNREGISTERED_INCOMPATIBLE_TYPE", response_text)

    # -------------------------------------------------------------------------
    # 10. Edge Case: Focal GO with Discovered GOs — Outgoing to Discovered Excluded
    # -------------------------------------------------------------------------
    def test_outgoing_relationship_from_focal_to_discovered_go_strictly_excluded(self):
        """
        Verify that when state contains a focal GO alongside discovered/cited GOs,
        an outgoing relationship from the focal GO to a discovered GO is strictly excluded
        across verification_node, response_node, and ResponseAgent.
        """
        nodes = WorkflowNodes(supervisor_agent=MagicMock())

        # State contains focal GO AND discovered GOs (simulating database discovery of 114 and 151)
        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": QueryIntent.INCOMING_RELATIONSHIPS,
            "identified_go_numbers": ["24/2026/ITD", "114/2025/ITD"],  # Even if multi-identified
            "discovered_go_numbers": ["114/2025/ITD", "151/2025/ITD"],
            "relationship_results": [
                {
                    "source_go_number": "24/2026/ITD",
                    "target_go_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                }
            ],
            "workflow_steps": [{"step_number": 1, "target_agent": "verification_agent"}],
            "current_step_index": 0,
            "document_analysis": [
                {
                    "go_number": "G.O.(Rt)No.24/2026/ITD",
                    "department": "Information Technology",
                    "issue_date": "2026-01-20",
                }
            ],
        }

        # 1. VerificationNode scoping check:
        # Must scope focal_cited_gos strictly to the user-requested focal GO
        mock_verif_res = MagicMock()
        mock_verif_res.status.value = "VERIFIED"
        mock_verif_res.confidence_score = 0.95
        mock_verif_res.evidence_sufficient = True
        mock_verif_res.model_dump.return_value = {
            "confidence_score": 0.95,
            "verified_relationships": [],
            "discrepancies": [
                {
                    "source_order_number": "24/2026/ITD",
                    "target_order_number": "114/2025/ITD",
                    "description": "UNREGISTERED_INCOMPATIBLE_TYPE",
                }
            ],
        }
        mock_verif_res.relationships = []
        nodes.verification_agent.execute = MagicMock(return_value=mock_verif_res)

        nodes.verification_node(state)
        call_req = nodes.verification_agent.execute.call_args[0][0]

        # VerificationAgent cited_gos must contain ONLY the user-requested focal GO
        self.assertEqual(len(call_req.cited_go_numbers), 1)
        self.assertEqual(call_req.cited_go_numbers[0], "G.O.(Rt)No.24/2026/ITD")
        self.assertNotIn("114/2025/ITD", call_req.cited_go_numbers)

        # Outgoing candidate to discovered GO must be excluded
        self.assertEqual(len(call_req.relationship_candidates), 0)

        # 2. ResponseNode filtering check:
        state["verification_status"] = "VERIFIED"
        state["verification_results"] = mock_verif_res.model_dump()
        state["workflow_steps"] = [{"step_number": 2, "target_agent": "response_agent"}]
        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")

        nodes.response_node(state)
        resp_input = nodes.response_agent.generate_response.call_args[0][0]

        # Candidates and discrepancies to discovered GO must be excluded from response input
        self.assertEqual(len(resp_input.unverified_candidates), 0)
        self.assertEqual(len(resp_input.discrepancies), 0)

        # 3. Direct ResponseAgent output check:
        direct_resp_input = ResponseAgentInput(
            user_query="Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            intent="INCOMING_RELATIONSHIPS",
            verification_status="VERIFIED",
            verification_confidence_score=0.95,
            legal_effect_verified=False,
            verified_findings=[],
            partially_verified_supported=[],
            unverified_candidates=[
                {
                    "source_go_number": "24/2026/ITD",
                    "target_order_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                }
            ],
            discrepancies=[
                {
                    "source_order_number": "24/2026/ITD",
                    "target_order_number": "114/2025/ITD",
                    "description": "UNREGISTERED_INCOMPATIBLE_TYPE",
                }
            ],
            document_analysis=[{
                "go_number": "G.O.(Rt)No.24/2026/ITD",
                "department": "Information Technology",
                "issue_date": "2026-01-20",
                "text_preview": "Sanctioning revised project...",
            }],
        )
        final_text = self.response_agent.generate_response(direct_resp_input)

        # Must report zero incoming relationships and retain focal metadata
        self.assertIn("### 1. Incoming Relationships", final_text)
        self.assertIn("No incoming relationships (subsequent Government Orders modifying, superseding, or referencing this order) were found in the indexed database.", final_text)
        self.assertIn("`G.O.(Rt)No.24/2026/ITD`", final_text)

        # Must NOT leak discovered GO 114/2025/ITD as candidate or discrepancy
        self.assertNotIn("114/2025/ITD", final_text)
        self.assertNotIn("### 2. Unverified Relationship Candidates", final_text)
        self.assertNotIn("### 3. Database Discrepancies", final_text)

    # -------------------------------------------------------------------------
    # 13. Explicit focal_go in ResponseAgentInput & Missing Target Filtering
    # -------------------------------------------------------------------------
    def test_response_agent_input_focal_go_explicit_filtering(self):
        """Verify ResponseAgent uses input_data.focal_go explicitly for incoming filtering."""
        resp_input = ResponseAgentInput(
            user_query="Generic query without explicit GO in text",
            intent="INCOMING_RELATIONSHIPS",
            focal_go="24/2026/ITD",
            verification_status="VERIFIED",
            verification_confidence_score=0.95,
            legal_effect_verified=False,
            unverified_candidates=[
                # Incoming pointing to focal GO -> KEEP
                {"source_go_number": "100/2026/ITD", "target_order_number": "24/2026/ITD", "relationship_type": "MODIFIES"},
                # Outgoing from focal GO -> EXCLUDE
                {"source_go_number": "24/2026/ITD", "target_order_number": "114/2025/ITD", "relationship_type": "REFERENCES"},
                # Irrelevant relationship -> EXCLUDE
                {"source_go_number": "50/2026/ITD", "target_order_number": "99/2025/ITD", "relationship_type": "MODIFIES"},
            ],
            discrepancies=[
                # Incoming discrepancy targeting focal GO -> KEEP
                {"source_order_number": "100/2026/ITD", "target_order_number": "24/2026/ITD", "description": "VALID_INCOMING"},
                # Discrepancy targeting other GO -> EXCLUDE
                {"source_order_number": "24/2026/ITD", "target_order_number": "114/2025/ITD", "description": "OUTGOING"},
            ],
            document_analysis=[{
                "go_number": "24/2026/ITD",
                "department": "ITD",
                "issue_date": "2026-01-20",
                "text_preview": "Sample order...",
            }],
        )

        final_text = self.response_agent.generate_response(resp_input)
        # Should keep candidate 100/2026/ITD targeting focal GO
        self.assertIn("100/2026/ITD", final_text)
        self.assertIn("VALID_INCOMING", final_text)
        # Should exclude outgoing/irrelevant GOs
        self.assertNotIn("114/2025/ITD", final_text)
        self.assertNotIn("50/2026/ITD", final_text)

    def test_response_node_excludes_discrepancy_with_missing_target_go(self):
        """Verify response_node excludes discrepancies with missing or empty target GO for incoming queries."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "VERIFIED",
            "supported_relationships": [],
            "unsupported_claims": [],
            "discrepancies": [
                {"source_order_number": "100/2026/ITD", "target_order_number": None, "description": "MISSING_TARGET"},
                {"source_order_number": "100/2026/ITD", "target_order_number": "", "description": "EMPTY_TARGET"},
                {"source_order_number": "100/2026/ITD", "target_order_number": "24/2026/ITD", "description": "VALID_INCOMING_TARGET"},
                {"source_order_number": "24/2026/ITD", "target_order_number": "114/2025/ITD", "description": "OUTGOING_DISCREPANCY"},
            ],
        }

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": "INCOMING_RELATIONSHIPS",
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "database_agent", "action_type": "GET_RELATIONSHIPS", "tool_parameters": {"go_number": "24/2026/ITD", "direction": "INCOMING"}}]},
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "unverified_candidates": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # Only VALID_INCOMING_TARGET should survive filtering
        self.assertEqual(len(resp_input.discrepancies), 1)
        self.assertEqual(resp_input.discrepancies[0]["description"], "VALID_INCOMING_TARGET")
        self.assertEqual(resp_input.discrepancies[0]["target_order_number"], "24/2026/ITD")
        # Ensure focal_go was passed into resp_input
        self.assertEqual(resp_input.focal_go, "G.O.(Rt)No.24/2026/ITD")

    def test_outgoing_query_discrepancies_unfiltered_in_response_node(self):
        """Verify response_node retains all discrepancies for outgoing / general relationship queries."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "VERIFIED",
            "supported_relationships": [],
            "unsupported_claims": [],
            "discrepancies": [
                {"source_order_number": "24/2026/ITD", "target_order_number": None, "description": "MISSING_TARGET"},
                {"source_order_number": "24/2026/ITD", "target_order_number": "114/2025/ITD", "description": "OUTGOING_DISCREPANCY"},
            ],
        }

        state = {
            "user_query": "What are the relationships for G.O.(Rt)No.24/2026/ITD?",
            "intent": "RELATIONSHIPS",
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "database_agent", "action_type": "GET_RELATIONSHIPS", "tool_parameters": {"go_number": "24/2026/ITD", "direction": "OUTGOING"}}]},
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "unverified_candidates": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # In outgoing queries, discrepancies must NOT be filtered
        self.assertEqual(len(resp_input.discrepancies), 2)

    def test_all_three_stages_use_same_focal_go(self):
        """Confirm that verification_node, response_node, and ResponseAgent use the exact same focal GO."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.status.value = "VERIFIED"
        mock_verif_res.overall_status = "VERIFIED"
        mock_verif_res.confidence_score = 0.95
        mock_verif_res.evidence_sufficient = True
        mock_verif_res.relationships = []
        mock_verif_res.model_dump.return_value = {
            "overall_status": "VERIFIED",
            "confidence_score": 0.95,
            "supported_relationships": [],
            "unsupported_claims": [],
            "discrepancies": [],
        }
        nodes.verification_agent.execute = MagicMock(return_value=mock_verif_res)

        focal_query_go = "G.O.(Rt)No.24/2026/ITD"
        discovered_go = "114/2025/ITD"

        state = {
            "user_query": f"Show incoming relationships for {focal_query_go}",
            "intent": "INCOMING_RELATIONSHIPS",
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "database_agent", "action_type": "GET_RELATIONSHIPS", "tool_parameters": {"go_number": "24/2026/ITD", "direction": "INCOMING"}}]},
            "identified_go_numbers": ["24/2026/ITD", discovered_go],
            "relationship_candidates": [
                {"source_go_number": "24/2026/ITD", "target_order_number": discovered_go, "relationship_type": "MODIFIES", "direction": "OUTGOING"}
            ],
            "document_analysis": [{"go_number": focal_query_go, "department": "ITD", "text_preview": "Sample"}],
            "search_results": [],
            "workflow_steps": [{"step_number": 1, "target_agent": "verification_agent"}],
        }

        # Stage 1: verification_node
        res_verif = nodes.verification_node(state)
        call_req = nodes.verification_agent.execute.call_args[0][0]
        stage1_focal_go = call_req.cited_go_numbers[0]
        self.assertEqual(stage1_focal_go, focal_query_go)

        # Stage 2: response_node
        state["verification_status"] = "VERIFIED"
        state["verification_results"] = mock_verif_res.model_dump()
        state["workflow_steps"].append({"step_number": 2, "target_agent": "response_agent"})
        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")

        res_resp = nodes.response_node(state)
        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        stage2_focal_go = resp_input.focal_go
        self.assertEqual(stage2_focal_go, focal_query_go)

        # Stage 3: ResponseAgent.generate_response
        # Test directly with resp_input to see that resp_input.focal_go is the one used
        actual_response_agent = ResponseAgent()
        with patch.object(actual_response_agent, "_are_go_matching", wraps=actual_response_agent._are_go_matching) as mock_matching:
            final_text = actual_response_agent.generate_response(resp_input)
            # Check that _are_go_matching was called with stage2_focal_go
            match_calls = [c[0] for c in mock_matching.call_args_list]
            self.assertTrue(any(stage2_focal_go in str(c) for c in match_calls) or focal_query_go in final_text)

        # Assert stage 1 and stage 2 use identical focal GO string
        self.assertEqual(stage1_focal_go, stage2_focal_go)
        self.assertEqual(stage2_focal_go, focal_query_go)

    # -------------------------------------------------------------------------
    # 14. Directional Filtering for Verified and Supported Findings
    # -------------------------------------------------------------------------
    def test_response_node_excludes_outgoing_verified_findings_for_incoming_query(self):
        """Verify response_node excludes outgoing findings from verified_findings for incoming queries."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "VERIFIED",
            "verified_relationships": [
                {
                    "source_go_number": "G.O.(Rt)No.24/2026/ITD",
                    "target_go_number": "G.O (Rt) No. 114/2025/ITD",
                    "relationship_type": "MODIFIES",
                    "registration_status": "REGISTERED_EXACT_MATCH",
                }
            ],
            "discrepancies": [],
        }

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": "INCOMING_RELATIONSHIPS",
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "relationship_results": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # Outgoing verified relationship must be excluded
        self.assertEqual(len(resp_input.verified_findings), 0)

    def test_response_node_retains_incoming_verified_findings_for_incoming_query(self):
        """Verify response_node retains genuine incoming findings in verified_findings."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "VERIFIED",
            "verified_relationships": [
                {
                    "source_go_number": "G.O.(Rt)No.100/2026/ITD",
                    "target_go_number": "G.O.(Rt)No.24/2026/ITD",
                    "relationship_type": "MODIFIES",
                    "registration_status": "REGISTERED_EXACT_MATCH",
                }
            ],
            "discrepancies": [],
        }

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": "INCOMING_RELATIONSHIPS",
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "relationship_results": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # Incoming verified relationship must be retained
        self.assertEqual(len(resp_input.verified_findings), 1)
        self.assertEqual(resp_input.verified_findings[0]["source_go_number"], "G.O.(Rt)No.100/2026/ITD")
        self.assertEqual(resp_input.verified_findings[0]["target_go_number"], "G.O.(Rt)No.24/2026/ITD")

    def test_response_node_excludes_outgoing_partially_verified_findings_for_incoming_query(self):
        """Verify response_node excludes outgoing findings from partially_verified_supported for incoming queries."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "PARTIALLY_VERIFIED",
            "verified_relationships": [
                {
                    "source_go_number": "24/2026/ITD",
                    "target_go_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                    "registration_status": "REGISTERED_EXACT_MATCH",
                }
            ],
            "discrepancies": [],
        }

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": "INCOMING_RELATIONSHIPS",
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "PARTIALLY_VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "relationship_results": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # Outgoing partially verified supported relationship must be excluded
        self.assertEqual(len(resp_input.partially_verified_supported), 0)

    def test_response_node_retains_incoming_partially_verified_findings_for_incoming_query(self):
        """Verify response_node retains incoming findings in partially_verified_supported."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "PARTIALLY_VERIFIED",
            "verified_relationships": [
                {
                    "source_go_number": "100/2026/ITD",
                    "target_go_number": "24/2026/ITD",
                    "relationship_type": "MODIFIES",
                    "registration_status": "REGISTERED_EXACT_MATCH",
                }
            ],
            "discrepancies": [],
        }

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": "INCOMING_RELATIONSHIPS",
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "PARTIALLY_VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "relationship_results": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # Incoming supported relationship must be retained
        self.assertEqual(len(resp_input.partially_verified_supported), 1)

    def test_response_node_excludes_missing_target_verified_findings(self):
        """Verify response_node excludes relationships with missing or empty target GO."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "VERIFIED",
            "verified_relationships": [
                {
                    "source_go_number": "100/2026/ITD",
                    "target_go_number": None,
                    "relationship_type": "MODIFIES",
                    "registration_status": "REGISTERED_EXACT_MATCH",
                },
                {
                    "source_go_number": "100/2026/ITD",
                    "target_go_number": "",
                    "relationship_type": "MODIFIES",
                    "registration_status": "REGISTERED_EXACT_MATCH",
                },
            ],
            "discrepancies": [],
        }

        state = {
            "user_query": "Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            "intent": "INCOMING_RELATIONSHIPS",
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "relationship_results": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        self.assertEqual(len(resp_input.verified_findings), 0)

    def test_outgoing_and_general_query_verified_findings_unfiltered(self):
        """Verify outgoing/general queries retain all verified findings without directional filtering."""
        nodes = WorkflowNodes()
        mock_verif_res = MagicMock()
        mock_verif_res.model_dump.return_value = {
            "overall_status": "VERIFIED",
            "verified_relationships": [
                {
                    "source_go_number": "24/2026/ITD",
                    "target_go_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                    "registration_status": "REGISTERED_EXACT_MATCH",
                }
            ],
            "discrepancies": [],
        }

        state = {
            "user_query": "What are the relationships for G.O.(Rt)No.24/2026/ITD?",
            "intent": "RELATIONSHIPS",
            "identified_go_numbers": ["24/2026/ITD"],
            "verification_status": "VERIFIED",
            "verification_results": mock_verif_res.model_dump(),
            "workflow_steps": [{"step_number": 2, "target_agent": "response_agent"}],
            "document_analysis": [],
            "search_results": [],
            "relationship_results": [],
        }

        nodes.response_agent.generate_response = MagicMock(return_value="Final Response")
        nodes.response_node(state)

        resp_input = nodes.response_agent.generate_response.call_args[0][0]
        # In general queries, outgoing verified relationship must NOT be excluded
        self.assertEqual(len(resp_input.verified_findings), 1)
        self.assertEqual(resp_input.verified_findings[0]["target_go_number"], "114/2025/ITD")

    def test_response_agent_renders_zero_incoming_notice_for_verified_and_partially_verified(self):
        """Verify ResponseAgent renders zero incoming notice when all findings are outgoing, for both VERIFIED and PARTIALLY_VERIFIED."""
        # 1. VERIFIED status with outgoing finding supplied
        resp_input_verified = ResponseAgentInput(
            user_query="Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            intent="INCOMING_RELATIONSHIPS",
            focal_go="24/2026/ITD",
            verification_status="VERIFIED",
            verification_confidence_score=1.0,
            legal_effect_verified=False,
            verified_findings=[
                {
                    "source_go_number": "24/2026/ITD",
                    "target_go_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                }
            ],
            document_analysis=[{
                "go_number": "24/2026/ITD",
                "department": "ITD",
                "issue_date": "2026-01-20",
                "text_preview": "Sample order...",
            }],
        )
        text_verified = self.response_agent.generate_response(resp_input_verified)
        self.assertIn("### 1. Incoming Relationships", text_verified)
        self.assertIn("No incoming relationships (subsequent Government Orders modifying, superseding, or referencing this order) were found in the indexed database.", text_verified)
        self.assertNotIn("114/2025/ITD", text_verified)
        self.assertNotIn("### 1. Verified Textual Findings", text_verified)

        # 2. PARTIALLY_VERIFIED status with outgoing supported finding supplied
        resp_input_pv = ResponseAgentInput(
            user_query="Show incoming relationships for G.O.(Rt)No.24/2026/ITD",
            intent="INCOMING_RELATIONSHIPS",
            focal_go="24/2026/ITD",
            verification_status="PARTIALLY_VERIFIED",
            verification_confidence_score=0.6,
            legal_effect_verified=False,
            partially_verified_supported=[
                {
                    "source_go_number": "24/2026/ITD",
                    "target_go_number": "114/2025/ITD",
                    "relationship_type": "MODIFIES",
                }
            ],
            document_analysis=[{
                "go_number": "24/2026/ITD",
                "department": "ITD",
                "issue_date": "2026-01-20",
                "text_preview": "Sample order...",
            }],
        )
        text_pv = self.response_agent.generate_response(resp_input_pv)
        self.assertIn("### 1. Incoming Relationships", text_pv)
        self.assertIn("No incoming relationships (subsequent Government Orders modifying, superseding, or referencing this order) were found in the indexed database.", text_pv)
        self.assertNotIn("114/2025/ITD", text_pv)
        self.assertNotIn("### 1. Partially Verified Textual Findings", text_pv)


if __name__ == "__main__":
    unittest.main()
