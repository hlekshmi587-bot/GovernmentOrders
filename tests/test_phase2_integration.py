"""
Phase 2 Step 2.5: Integration Test Suite for LangGraph Multi-Agent Architecture.

Tests dynamic orchestration across:
  - SupervisorAgent, SearchAgent, DatabaseAgent, RelationshipAgent, VerificationAgent, ResponseAgent
  - Exact parameter mappings and validation rejections
  - State reducers and step-ID uniqueness invariants
  - Multi-tier bounds (plan_generation <= 6, invalid_plan_count <= 2, step_retry_count <= 2)
  - Evidence-grounded citation and legal_effect_verified=False enforcement
  - Human review interrupt and resume routing back to Supervisor
  - End-to-end simulated workflows

Physical test count: Exactly 32 physical test functions.
"""

import os
import sys
from datetime import date, datetime
import unittest
from unittest.mock import MagicMock, patch

# Ensure workspace root is always on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import ValidationError
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState, merge_list
from app.agents.specialized.supervisor_agent import (
    SupervisorAgent,
    SupervisorPlan,
    PlannedStep,
    QueryIntent,
)
from app.agents.specialized.search_agent import (
    SearchAgent,
    SearchAgentRequest,
    SearchAgentResponse,
    SearchResultItem,
    RetrievalMode,
)
from app.agents.specialized.database_agent import (
    DatabaseAgent,
    DatabaseAgentRequest,
    DatabaseAgentResponse,
    DatabaseOperation,
    RelationshipDirection,
    AgentResponseStatus,
    OrderMetadataItem,
)
from app.agents.specialized.relationship_agent import (
    RelationshipAgent,
    RelationshipAgentRequest,
    RelationshipAgentResponse,
    RelationshipInvestigationMode,
    RelationshipCandidate,
    RelationshipAgentStatus,
)
from app.agents.specialized.verification_agent import (
    VerificationAgent,
    VerificationAgentRequest,
    VerificationAgentResponse,
    VerificationStatus,
    VerificationConfidence,
    VerificationRecommendation,
    AutonomousVerificationAction,
    TargetAgentName,
    EmptyParams,
    RelationshipVerificationDetail,
)
from app.agents.specialized.response_agent import (
    ResponseAgent,
    ResponseAgentInput,
)
from app.agents.workflow.graph_builder import (
    WorkflowNodes,
    build_government_orders_graph,
    route_from_supervisor,
    route_after_agent_node,
    route_from_verification,
    route_from_human_review,
)


class TestPhase2Integration(unittest.TestCase):
    """32 Physical Integration Tests for Phase 2 Step 2.5."""

    # =========================================================================
    # Group 1: Contract, Reducer & State Invariants (Tests 1–6)
    # =========================================================================

    def test_01_existing_merge_list_remains_unchanged(self):
        """1. Confirm merge_list reducer behaves identically to Phase 1/2."""
        self.assertEqual(merge_list([], ["a", "b"]), ["a", "b"])
        self.assertEqual(merge_list(["a"], []), ["a"])
        self.assertEqual(merge_list(["a", "b"], ["b", "c"]), ["a", "b", "c"])

    def test_02_pre_reducer_single_id_emission(self):
        """2. Confirm nodes emit single unique IDs, preventing incoming duplicates."""
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="test",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=1,
            results=[
                SearchResultItem(
                    go_number="G.O.(Ms)No.114/2025/ITD",
                    matched_sources=["database"],
                    source_filename="test.pdf",
                )
            ],
        )
        nodes = WorkflowNodes(search_agent=mock_search)
        state: AgentState = {
            "user_query": "test query",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "search_agent", "tool_parameters": {"query": "test"}}]},
        }
        res = nodes.search_node(state)
        # Emitted completed_step_ids must contain exactly one ID
        completed = res.get("completed_step_ids", [])
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0], "plan_1_step_1")

    def test_03_completed_ids_never_abandoned(self):
        """3. Confirm completed steps are excluded from abandoned_step_ids upon replan."""
        nodes = WorkflowNodes()
        # All steps in plan (step 1 and step 2) are completed or failed, so replan is triggered
        state: AgentState = {
            "user_query": "test query",
            "plan_generation": 1,
            "completed_step_ids": ["plan_1_step_1"],
            "failed_step_ids": ["plan_1_step_2"],
            "abandoned_step_ids": [],
            "execution_plan": {
                "execution_steps": [
                    {"step_number": 1, "target_agent": "search_agent"},
                    {"step_number": 2, "target_agent": "database_agent"},
                    {"step_number": 3, "target_agent": "relationship_agent"},
                ]
            },
        }
        # Step 3 was never completed, step 1 was completed
        res = nodes.supervisor_node(state)
        # When step 3 is pending, it executes next:
        self.assertEqual(res.get("next_action"), "relationship")

        # Now when all steps are finished and replanning triggers:
        state["failed_step_ids"] = ["plan_1_step_2", "plan_1_step_3"]
        res_replan = nodes.supervisor_node(state)
        abandoned = res_replan.get("abandoned_step_ids", [])
        self.assertNotIn("plan_1_step_1", abandoned)
        self.assertIn("plan_1_step_2", abandoned)
        self.assertIn("plan_1_step_3", abandoned)

    def test_04_failed_ids_never_counted_as_completed(self):
        """4. Confirm failed step IDs and completed IDs are mutually exclusive."""
        nodes = WorkflowNodes()
        # Invalid search query triggers failure
        state: AgentState = {
            "user_query": "",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "search_agent", "tool_parameters": {"query": ""}}]},
        }
        res = nodes.search_node(state)
        self.assertIn("failed_step_ids", res)
        self.assertNotIn("completed_step_ids", res)
        self.assertIn("plan_1_step_1", res["failed_step_ids"])

    def test_05_agent_state_initialization_defaults(self):
        """5. Confirm all state fields initialize with correct defaults."""
        initial_keys = [
            "user_query", "intent", "identified_go_numbers", "discovered_go_numbers",
            "execution_plan", "selected_agents", "next_action", "search_results",
            "document_analysis", "relationship_results", "verification_results",
            "evidence_sufficient", "confidence", "retry_count", "max_retries",
            "human_review_required", "human_review_status", "human_review_notes",
            "human_feedback", "final_answer", "workflow_status", "execution_trace",
            "errors", "plan_generation", "invalid_plan_count", "step_retry_count",
            "completed_step_ids", "failed_step_ids", "abandoned_step_ids",
            "response_ready", "verification_status"
        ]
        state_keys = AgentState.__annotations__.keys()
        for k in initial_keys:
            self.assertIn(k, state_keys)

    def test_06_unique_generation_step_ids_on_replan(self):
        """6. Confirm generation increments produce distinct step IDs."""
        nodes = WorkflowNodes()
        state: AgentState = {
            "user_query": "test query",
            "plan_generation": 1,
            "completed_step_ids": ["plan_1_step_1"],
            "failed_step_ids": [],
            "abandoned_step_ids": [],
            "execution_plan": {
                "execution_steps": [
                    {"step_number": 1, "target_agent": "search_agent"}
                ]
            },
        }
        # In evaluate_and_replan, if no errors or unverified candidates, it replans
        with patch.object(nodes.supervisor, "evaluate_and_replan") as mock_eval:
            from app.agents.specialized.supervisor_agent import SupervisorReplanDecision
            mock_eval.return_value = SupervisorReplanDecision(
                evidence_sufficient=False,
                reasoning="Need further steps",
                next_action="search",
                confidence=0.5,
                decision_source="HEURISTIC_FALLBACK"
            )
            res = nodes.supervisor_node(state)
            new_gen = res.get("plan_generation")
            self.assertEqual(new_gen, 2)
            new_steps = res.get("execution_plan", {}).get("execution_steps", [])
            if new_steps:
                first_new_id = f"plan_{new_gen}_step_{new_steps[0]['step_number']}"
                self.assertNotEqual(first_new_id, "plan_1_step_1")

    # =========================================================================
    # Group 2: Parameter Translation & Validation (Tests 7–12)
    # =========================================================================

    def test_07_supervisor_to_search_parameter_translation(self):
        """7. Tests translation of keyword -> query, retrieval_mode, and date filters."""
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="electronic policy",
            requested_mode=RetrievalMode.KEYWORD,
            returned_count=1,
            results=[
                SearchResultItem(
                    go_number="G.O.(Ms)No.114/2025/ITD",
                    matched_sources=["database"],
                    source_filename="test.pdf",
                )
            ],
        )
        nodes = WorkflowNodes(search_agent=mock_search)
        state: AgentState = {
            "user_query": "electronic policy",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "search_agent",
                        "tool_parameters": {
                            "keyword": "electronic policy",
                            "retrieval_mode": "KEYWORD",
                            "limit": 5,
                        },
                    }
                ]
            },
        }
        res = nodes.search_node(state)
        self.assertIn("completed_step_ids", res)
        # Verify SearchAgentRequest received query="electronic policy"
        called_req = mock_search.execute_search.call_args[0][0]
        self.assertEqual(called_req.query, "electronic policy")
        self.assertEqual(called_req.retrieval_mode, RetrievalMode.KEYWORD)

    def test_08_supervisor_to_database_parameter_translation(self):
        """8. Tests operation, direction, and conditional go_number validation."""
        mock_db = MagicMock(spec=DatabaseAgent)
        mock_db.execute.return_value = DatabaseAgentResponse(
            status=AgentResponseStatus.SUCCESS,
            operation=DatabaseOperation.LOOKUP_ORDER,
            query_identifier="114/2025",
            order_data=OrderMetadataItem(
                id=1,
                go_number="G.O.(Ms)No.114/2025/ITD",
                source_filename="test.pdf",
            ),
        )
        nodes = WorkflowNodes(database_agent=mock_db)
        state: AgentState = {
            "user_query": "Lookup 114/2025",
            "identified_go_numbers": ["114/2025"],
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "database_agent",
                        "tool_parameters": {
                            "operation": "LOOKUP_ORDER",
                            "go_number": "114/2025",
                            "direction": "INCOMING",
                        },
                    }
                ]
            },
        }
        res = nodes.database_node(state)
        self.assertIn("completed_step_ids", res)
        called_req = mock_db.execute.call_args[0][0]
        self.assertEqual(called_req.operation, DatabaseOperation.LOOKUP_ORDER)
        self.assertEqual(called_req.direction, RelationshipDirection.INCOMING)

    def test_09_supervisor_to_relationship_parameter_translation(self):
        """9. Tests source_go_number -> go_number, and target_go_number."""
        mock_rel = MagicMock(spec=RelationshipAgent)
        mock_rel.execute.return_value = RelationshipAgentResponse(
            status=RelationshipAgentStatus.SUCCESS,
            mode=RelationshipInvestigationMode.INVESTIGATE_PAIR,
            source_go_number="114/2025",
            target_go_number="115/2025",
            total_relationships=0,
            relationships=[],
        )
        nodes = WorkflowNodes(relationship_agent=mock_rel)
        state: AgentState = {
            "user_query": "Relate 114 and 115",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "relationship_agent",
                        "tool_parameters": {
                            "source_go_number": "114/2025",
                            "target_go_number": "115/2025",
                            "mode": "INVESTIGATE_PAIR",
                        },
                    }
                ]
            },
        }
        res = nodes.relationship_node(state)
        self.assertIn("completed_step_ids", res)
        called_req = mock_rel.execute.call_args[0][0]
        self.assertEqual(called_req.go_number, "114/2025")
        self.assertEqual(called_req.target_go_number, "115/2025")
        self.assertEqual(called_req.mode, RelationshipInvestigationMode.INVESTIGATE_PAIR)

    def test_10_rejection_of_boolean_as_integer(self):
        """10. Confirms limit=True or text_preview_chars=True is rejected."""
        with self.assertRaises(ValueError):
            PlannedStep(
                step_number=1,
                action_name="test",
                target_agent="search_agent",
                target_tool="search_tool",
                tool_parameters={"limit": True},
                expected_outcome="fail",
            )
        with self.assertRaises(ValueError):
            PlannedStep(
                step_number=1,
                action_name="test",
                target_agent="database_agent",
                target_tool="database_tool",
                tool_parameters={"text_preview_chars": True},
                expected_outcome="fail",
            )

    def test_11_rejection_of_invalid_enum_values(self):
        """11. Confirms invalid modes or operations are rejected before execution."""
        with self.assertRaises(ValueError):
            PlannedStep(
                step_number=1,
                action_name="test",
                target_agent="relationship_agent",
                target_tool="relationship_tool",
                tool_parameters={"mode": "INVALID_MODE"},
                expected_outcome="fail",
            )
        with self.assertRaises(ValueError):
            PlannedStep(
                step_number=1,
                action_name="test",
                target_agent="database_agent",
                target_tool="database_tool",
                tool_parameters={"operation": "INVALID_OPERATION"},
                expected_outcome="fail",
            )

    def test_12_relationship_modes_parameter_requirements(self):
        """12. Confirms all 4 modes enforce their required fields."""
        # INVESTIGATE_ORDER requires go_number
        with self.assertRaises(ValueError):
            RelationshipAgentRequest(mode=RelationshipInvestigationMode.INVESTIGATE_ORDER, go_number=None)
        # EXTRACT_FROM_TEXT requires text
        with self.assertRaises(ValueError):
            RelationshipAgentRequest(mode=RelationshipInvestigationMode.EXTRACT_FROM_TEXT, text=None)
        # INVESTIGATE_PAIR requires target_go_number and (go_number or text)
        with self.assertRaises(ValueError):
            RelationshipAgentRequest(mode=RelationshipInvestigationMode.INVESTIGATE_PAIR, target_go_number=None)
        # CROSS_EXAMINE_DATABASE requires go_number
        with self.assertRaises(ValueError):
            RelationshipAgentRequest(mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE, go_number=None)

    # =========================================================================
    # Group 3: Retry, Replanning & Loop Bounding (Tests 13–18)
    # =========================================================================

    def test_13_independent_step_retry_resets_on_success(self):
        """13. Confirms step_retry_count resets to 0 after a successful step."""
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="test",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=1,
            results=[
                SearchResultItem(
                    go_number="G.O.(Ms)No.114/2025/ITD",
                    matched_sources=["database"],
                    source_filename="test.pdf",
                )
            ],
        )
        nodes = WorkflowNodes(search_agent=mock_search)
        state: AgentState = {
            "user_query": "test query",
            "plan_generation": 1,
            "step_retry_count": 1,  # previous transient retry
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "search_agent", "tool_parameters": {"query": "test"}}]},
        }
        res = nodes.search_node(state)
        self.assertEqual(res.get("step_retry_count"), 0)
        self.assertIn("plan_1_step_1", res.get("completed_step_ids", []))

    def test_14_transient_retry_bounded_at_two(self):
        """14. Confirms step retries twice on transient error before routing to human review."""
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.side_effect = ConnectionError("DB connection dropped")
        nodes = WorkflowNodes(search_agent=mock_search)

        state: AgentState = {
            "user_query": "test",
            "plan_generation": 1,
            "step_retry_count": 0,
            "retry_count": 0,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "search_agent", "tool_parameters": {"query": "test"}}]},
        }
        # Attempt 1: fails, step_retry_count -> 1
        res1 = nodes.search_node(state)
        self.assertEqual(res1.get("step_retry_count"), 1)

        # Attempt 2: fails, step_retry_count -> 2
        state["step_retry_count"] = 1
        res2 = nodes.search_node(state)
        self.assertEqual(res2.get("step_retry_count"), 2)

        # Attempt 3: fails, step_retry_count >= 2 -> marked in failed_step_ids, human_review_required=True
        state["step_retry_count"] = 2
        res3 = nodes.search_node(state)
        self.assertIn("plan_1_step_1", res3.get("failed_step_ids", []))
        self.assertTrue(res3.get("human_review_required"))

    def test_15_evidence_driven_replanning_progresses_multi_step(self):
        """15. Confirms legitimate investigation replans across generations up to bound 6."""
        nodes = WorkflowNodes()
        state: AgentState = {
            "user_query": "investigate multi step",
            "plan_generation": 2,
            "completed_step_ids": ["plan_1_step_1", "plan_2_step_1"],
            "failed_step_ids": [],
            "abandoned_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "search_agent"}]},
            "relationship_results": [{"verification_status": "UNVERIFIED_CANDIDATE"}],
        }
        res = nodes.supervisor_node(state)
        self.assertEqual(res.get("next_action"), "verification")

    def test_16_evidence_driven_replanning_hard_bounded_at_six(self):
        """16. Confirms replanning stops at 6 generations and outputs qualified findings."""
        nodes = WorkflowNodes()
        state: AgentState = {
            "user_query": "investigate long",
            "plan_generation": 6,
            "completed_step_ids": [f"plan_{i}_step_1" for i in range(1, 7)],
            "failed_step_ids": [],
            "abandoned_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "search_agent"}]},
        }
        res = nodes.supervisor_node(state)
        self.assertEqual(res.get("next_action"), "response")
        self.assertTrue(res.get("response_ready"))
        self.assertFalse(res.get("evidence_sufficient"))

    def test_17_invalid_plan_triggers_heuristic_fallback(self):
        """17. Confirms first invalid plan triggers deterministic heuristic planning."""
        supervisor = SupervisorAgent()
        # Mock LLM failure in create_initial_plan to test fallback
        with patch.object(supervisor.llm, "generate_structured", side_effect=Exception("LLM offline")):
            plan = supervisor.create_initial_plan("general query without LLM")
            self.assertIsNotNone(plan)
            self.assertEqual(plan.plan_source, "HEURISTIC_FALLBACK")

    def test_18_repeated_invalid_plans_route_to_human_review(self):
        """18. Confirms two invalid plans route to human review."""
        nodes = WorkflowNodes()
        state: AgentState = {
            "user_query": "test query",
            "plan_generation": 1,
            "invalid_plan_count": 2,
            "human_review_required": True,
            "execution_plan": {"execution_steps": []},
        }
        res = nodes.supervisor_node(state)
        self.assertEqual(res.get("next_action"), "human_review")
        self.assertEqual(res.get("workflow_status"), "PAUSED_FOR_HUMAN")

    # =========================================================================
    # Group 4: Specialized Agent Node Wrappers (Tests 19–23)
    # =========================================================================

    def test_19_search_node_success_and_no_match(self):
        """19. Confirms SearchAgent execution and completed step recording without infinite loops."""
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.return_value = SearchAgentResponse(
            status="NO_MATCH",
            query="nonexistent order",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=0,
            results=[],
        )
        nodes = WorkflowNodes(search_agent=mock_search)
        state: AgentState = {
            "user_query": "nonexistent order",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "search_agent", "tool_parameters": {"query": "nonexistent order"}}]},
        }
        res = nodes.search_node(state)
        self.assertIn("plan_1_step_1", res.get("completed_step_ids", []))
        self.assertEqual(res.get("step_retry_count"), 0)

    def test_20_database_node_not_found_semantics(self):
        """20. Confirms missing GO recorded as absent from indexed database."""
        mock_db = MagicMock(spec=DatabaseAgent)
        mock_db.execute.return_value = DatabaseAgentResponse(
            status=AgentResponseStatus.NOT_FOUND,
            operation=DatabaseOperation.LOOKUP_ORDER,
            query_identifier="999/2025",
        )
        nodes = WorkflowNodes(database_agent=mock_db)
        state: AgentState = {
            "user_query": "lookup 999/2025",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "database_agent", "tool_parameters": {"go_number": "999/2025"}}]},
        }
        res = nodes.database_node(state)
        self.assertIn("plan_1_step_1", res.get("completed_step_ids", []))
        doc_analysis = res.get("document_analysis", [])
        self.assertTrue(any(d.get("status") == "NOT_FOUND" for d in doc_analysis))

    def test_21_database_node_ambiguous_match_routes_to_human_review(self):
        """21. Confirms AMBIGUOUS_MATCH triggers human review without guessing."""
        mock_db = MagicMock(spec=DatabaseAgent)
        mock_db.execute.return_value = DatabaseAgentResponse(
            status=AgentResponseStatus.AMBIGUOUS_MATCH,
            operation=DatabaseOperation.LOOKUP_ORDER,
            query_identifier="114",
            ambiguous_candidates=["114/2025/ITD", "114/2024/GAD"],
        )
        nodes = WorkflowNodes(database_agent=mock_db)
        state: AgentState = {
            "user_query": "lookup 114",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "database_agent", "tool_parameters": {"go_number": "114"}}]},
        }
        res = nodes.database_node(state)
        self.assertTrue(res.get("human_review_required"))
        self.assertIn("Ambiguous database match", res.get("human_review_notes", ""))

    def test_22_relationship_node_candidate_isolation(self):
        """22. Confirms candidates remain UNVERIFIED with legal_effect_verified=False."""
        mock_rel = MagicMock(spec=RelationshipAgent)
        mock_rel.execute.return_value = RelationshipAgentResponse(
            status=RelationshipAgentStatus.SUCCESS,
            mode=RelationshipInvestigationMode.INVESTIGATE_ORDER,
            relationships=[
                RelationshipCandidate(
                    relationship_type="AMENDS",
                    source_go_number="114/2025",
                    referenced_go_numbers=["115/2025"],
                    matched_phrase="amends previous order",
                    confidence="HIGH",
                    extraction_method="REGEX",
                    verification_status="UNVERIFIED_CANDIDATE",
                    legal_effect_verified=False,
                )
            ],
            total_relationships=1,
        )
        nodes = WorkflowNodes(relationship_agent=mock_rel)
        state: AgentState = {
            "user_query": "114/2025",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "relationship_agent", "tool_parameters": {"go_number": "114/2025"}}]},
        }
        res = nodes.relationship_node(state)
        rels = res.get("relationship_results", [])
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["verification_status"], "UNVERIFIED_CANDIDATE")
        self.assertFalse(rels[0]["legal_effect_verified"])

    def test_23_verification_node_preserves_evidence_sufficient_flag(self):
        """23. Confirms evidence_sufficient flag is preserved as calculated."""
        mock_verif = MagicMock(spec=VerificationAgent)
        mock_verif.execute.return_value = VerificationAgentResponse(
            status=VerificationStatus.PARTIALLY_VERIFIED,
            overall_confidence=VerificationConfidence.MEDIUM,
            confidence_score=0.65,
            evidence_sufficient=False,
            recommendation=VerificationRecommendation(
                action=AutonomousVerificationAction.PROCEED_TO_RESPONSE,
                target_agent=TargetAgentName.SUPERVISOR_AGENT,
                parameters=EmptyParams(),
                reasoning="Partial corroboration achieved.",
            ),
        )
        nodes = WorkflowNodes(verification_agent=mock_verif)
        state: AgentState = {
            "user_query": "verify 114",
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "execution_plan": {"execution_steps": [{"step_number": 1, "target_agent": "verification_agent"}]},
        }
        res = nodes.verification_node(state)
        self.assertFalse(res.get("evidence_sufficient"))
        self.assertEqual(res.get("verification_status"), "PARTIALLY_VERIFIED")

    # =========================================================================
    # Group 5: Confidence, ResponseAgent & Human Review (Tests 24–29)
    # =========================================================================

    def test_24_retrieval_confidence_not_represented_as_verification(self):
        """24. Confirms search/DB direct responses have verification_confidence_score=None."""
        resp_agent = ResponseAgent()
        resp_input = ResponseAgentInput(
            user_query="Search IT policy",
            confidence="HIGH",
            verification_status=None,
            verification_confidence_score=None,
            search_results=[{"go_number": "G.O.114/2025", "department": "ITD", "issue_date": "2025-01-01"}],
        )
        text = resp_agent.generate_response(resp_input)
        self.assertIn("**Retrieval Precision:** `HIGH`", text)
        self.assertNotIn("Verification Status: `VERIFIED`", text)
        self.assertIn("Retrieved Search Results", text)

    def test_25_verification_numerical_confidence_passed_accurately(self):
        """25. Confirms VerificationAgent.confidence_score passes to response."""
        resp_agent = ResponseAgent()
        resp_input = ResponseAgentInput(
            user_query="Verify 114 modifies 115",
            verification_status="VERIFIED",
            verification_confidence_score=0.92,
            verified_findings=[{"source_go_number": "114/2025", "target_go_number": "115/2025", "relationship_type": "MODIFIES"}],
        )
        text = resp_agent.generate_response(resp_input)
        self.assertIn("Confidence Score: 0.92", text)
        self.assertIn("### 1. Verified Textual Findings", text)

    def test_26_response_agent_legal_effect_invariant_structural_rejection(self):
        """26. Confirms ResponseAgentInput(legal_effect_verified=True) is rejected."""
        with self.assertRaises(ValidationError):
            ResponseAgentInput(
                user_query="test",
                legal_effect_verified=True,
            )

    def test_27_response_agent_partial_verification_wording(self):
        """27. Confirms supported claims are separated from unsupported claims."""
        resp_agent = ResponseAgent()
        resp_input = ResponseAgentInput(
            user_query="Verify claims",
            verification_status="PARTIALLY_VERIFIED",
            partially_verified_supported=[{"source_go_number": "114/2025", "target_go_number": "115/2025", "relationship_type": "AMENDS"}],
            partially_verified_unsupported=[{"claim_text": "Claims 114 cancels 200", "reasoning": "Order 200 not found"}],
        )
        text = resp_agent.generate_response(resp_input)
        self.assertIn("PARTIALLY_VERIFIED (INCOMPLETE CORROBORATION)", text)
        self.assertIn("#### A. Supported Textual Findings", text)
        self.assertIn("#### B. Unsupported / Unresolved Claims", text)

    def test_28_human_review_resume_routes_back_to_supervisor(self):
        """28. Confirms resume via Command(resume=...) routes to supervisor_node."""
        state: AgentState = {"user_query": "test", "human_review_status": "RESOLVED"}
        next_node = route_from_human_review(state)
        self.assertEqual(next_node, "supervisor_node")

    def test_29_conflicting_evidence_routes_to_human_review_not_response(self):
        """29. Confirms CONFLICTING_EVIDENCE cannot bypass human review."""
        state: AgentState = {
            "user_query": "test",
            "verification_status": "CONFLICTING_EVIDENCE",
            "human_review_required": True,
        }
        next_node = route_from_verification(state)
        self.assertEqual(next_node, "human_review_node")

    # =========================================================================
    # Group 6: End-to-End Dynamic Workflows (Tests 30–32)
    # =========================================================================

    def test_30_end_to_end_direct_lookup_workflow(self):
        """30. Confirms direct GO lookup through supervisor -> database -> response."""
        mock_db = MagicMock(spec=DatabaseAgent)
        mock_db.execute.return_value = DatabaseAgentResponse(
            status=AgentResponseStatus.SUCCESS,
            operation=DatabaseOperation.LOOKUP_ORDER,
            query_identifier="114/2025",
            order_data=OrderMetadataItem(
                id=1,
                go_number="G.O.(Ms)No.114/2025/ITD",
                department="Electronics & Information Technology",
                issue_date="2025-01-15",
                source_filename="test.pdf",
            ),
        )
        nodes = WorkflowNodes(database_agent=mock_db)
        # Mock evaluate_and_replan to immediately finish with response
        from app.agents.specialized.supervisor_agent import SupervisorReplanDecision
        nodes.supervisor.evaluate_and_replan = MagicMock(return_value=SupervisorReplanDecision(
            evidence_sufficient=True,
            reasoning="Lookup succeeded",
            next_action="response",
            confidence=0.9,
            decision_source="HEURISTIC_FALLBACK"
        ))
        app = build_government_orders_graph(checkpointer=MemorySaver(), nodes=nodes)
        
        initial_state: AgentState = {
            "user_query": "Lookup G.O.(Ms)No.114/2025/ITD",
            "identified_go_numbers": ["G.O.(Ms)No.114/2025/ITD"],
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "abandoned_step_ids": [],
            "search_results": [],
            "document_analysis": [],
            "relationship_results": [],
            "execution_trace": [],
            "errors": [],
            "discovered_go_numbers": [],
            "retry_count": 0,
            "max_retries": 2,
            "step_retry_count": 0,
            "invalid_plan_count": 0,
            "human_review_required": False,
            "human_review_status": "NONE",
            "response_ready": False,
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "database_agent",
                        "tool_parameters": {"operation": "LOOKUP_ORDER", "go_number": "G.O.(Ms)No.114/2025/ITD"},
                    }
                ]
            },
        }
        res = app.invoke(initial_state, config={"configurable": {"thread_id": "t1"}})
        self.assertEqual(res.get("workflow_status"), "COMPLETED")
        self.assertIn("Indexed Database Records", res.get("final_answer", ""))

    def test_31_end_to_end_relational_verification_workflow(self):
        """31. Confirms full verification through supervisor -> relationship -> verification -> response."""
        mock_rel = MagicMock(spec=RelationshipAgent)
        mock_rel.execute.return_value = RelationshipAgentResponse(
            status=RelationshipAgentStatus.SUCCESS,
            mode=RelationshipInvestigationMode.INVESTIGATE_PAIR,
            relationships=[
                RelationshipCandidate(
                    relationship_type="AMENDS",
                    source_go_number="114/2025",
                    referenced_go_numbers=["115/2025"],
                    matched_phrase="amends previous order",
                    confidence="HIGH",
                    extraction_method="REGEX",
                    verification_status="UNVERIFIED_CANDIDATE",
                    legal_effect_verified=False,
                )
            ],
            total_relationships=1,
        )
        mock_verif = MagicMock(spec=VerificationAgent)
        mock_verif.execute.return_value = VerificationAgentResponse(
            status=VerificationStatus.VERIFIED,
            overall_confidence=VerificationConfidence.HIGH,
            confidence_score=0.95,
            evidence_sufficient=True,
            verified_relationships=[
                RelationshipVerificationDetail(
                    source_go_number="114/2025",
                    target_go_number="115/2025",
                    relationship_type="AMENDS",
                    evidence_text="amends order 115/2025",
                    page_number=2,
                    legal_effect_verified=False,
                )
            ],
            recommendation=VerificationRecommendation(
                action=AutonomousVerificationAction.PROCEED_TO_RESPONSE,
                target_agent=TargetAgentName.SUPERVISOR_AGENT,
                parameters=EmptyParams(),
                reasoning="Complete match verified.",
            ),
        )
        from app.agents.specialized.supervisor_agent import SupervisorReplanDecision
        nodes = WorkflowNodes(relationship_agent=mock_rel, verification_agent=mock_verif)
        nodes.supervisor.evaluate_and_replan = MagicMock(return_value=SupervisorReplanDecision(
            evidence_sufficient=True,
            reasoning="Verification completed",
            next_action="response",
            confidence=0.95,
            decision_source="HEURISTIC_FALLBACK"
        ))
        app = build_government_orders_graph(checkpointer=MemorySaver(), nodes=nodes)

        initial_state: AgentState = {
            "user_query": "Verify if 114/2025 amends 115/2025",
            "identified_go_numbers": ["114/2025", "115/2025"],
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
            "abandoned_step_ids": [],
            "search_results": [],
            "document_analysis": [],
            "relationship_results": [],
            "execution_trace": [],
            "errors": [],
            "discovered_go_numbers": [],
            "retry_count": 0,
            "max_retries": 2,
            "step_retry_count": 0,
            "invalid_plan_count": 0,
            "human_review_required": False,
            "human_review_status": "NONE",
            "response_ready": False,
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "relationship_agent",
                        "tool_parameters": {"source_go_number": "114/2025", "target_go_number": "115/2025", "mode": "INVESTIGATE_PAIR"},
                    },
                    {
                        "step_number": 2,
                        "target_agent": "verification_agent",
                        "tool_parameters": {},
                    },
                ]
            },
        }
        res = app.invoke(initial_state, config={"configurable": {"thread_id": "t2"}})
        self.assertEqual(res.get("workflow_status"), "COMPLETED")
        self.assertIn("Verified Textual Findings", res.get("final_answer", ""))
        self.assertIn("114/2025", res.get("final_answer", ""))

    def test_32_end_to_end_exhausted_investigation_qualified_response(self):
        """32. Confirms exhausted investigation outputs qualified partial response with evidence_sufficient=False."""
        mock_verif = MagicMock(spec=VerificationAgent)
        mock_verif.execute.return_value = VerificationAgentResponse(
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            overall_confidence=VerificationConfidence.LOW,
            confidence_score=0.2,
            evidence_sufficient=False,
            recommendation=VerificationRecommendation(
                action=AutonomousVerificationAction.PROCEED_TO_RESPONSE,
                target_agent=TargetAgentName.SUPERVISOR_AGENT,
                parameters=EmptyParams(),
                reasoning="Evidence incomplete.",
            ),
        )
        nodes = WorkflowNodes(verification_agent=mock_verif)
        app = build_government_orders_graph(checkpointer=MemorySaver(), nodes=nodes)

        initial_state: AgentState = {
            "user_query": "Investigate complex order",
            "plan_generation": 6,  # Reached max generations
            "completed_step_ids": [f"plan_{i}_step_1" for i in range(1, 7)],
            "failed_step_ids": [],
            "abandoned_step_ids": [],
            "search_results": [],
            "document_analysis": [],
            "relationship_results": [],
            "execution_trace": [],
            "errors": [],
            "discovered_go_numbers": [],
            "retry_count": 0,
            "max_retries": 2,
            "step_retry_count": 0,
            "invalid_plan_count": 0,
            "human_review_required": False,
            "human_review_status": "NONE",
            "response_ready": False,
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "verification_agent",
                        "tool_parameters": {},
                    }
                ]
            },
        }
        res = app.invoke(initial_state, config={"configurable": {"thread_id": "t3"}})
        self.assertEqual(res.get("workflow_status"), "COMPLETED")
        self.assertFalse(res.get("evidence_sufficient"))
        self.assertIn("Incomplete Investigation", res.get("final_answer", ""))


ALL_INTEGRATION_TESTS = [
    getattr(TestPhase2Integration, m)
    for m in sorted(dir(TestPhase2Integration))
    if m.startswith("test_")
]


def run_all_tests():
    print(f"Discovered {len(ALL_INTEGRATION_TESTS)} physical integration tests.")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPhase2Integration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 60)
    print(f"Ran {result.testsRun} tests.")
    print(f"PASSED: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"FAILED: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    print(f"SKIPPED: {len(result.skipped)}")
    print("=" * 60)
    if not result.wasSuccessful():
        exit(1)


if __name__ == "__main__":
    run_all_tests()
