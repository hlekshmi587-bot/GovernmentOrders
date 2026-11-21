"""
Tests for Phase 4.1 Step 4.3 Stage 1: Agentic Search Integration.

Validates:
1. Mapping of document_analysis to structured search results (SearchResultItemSchema) for exact GO_LOOKUP.
2. Missing or unsafe filename handling (has_pdf=False, pdf_url=None).
3. Search-node limit resolution priority:
   - Priority 1: Explicit plan step limit
   - Priority 2: Citizen dashboard search_parameters={"limit": 10}
   - Priority 3: Existing caller default limit 5
4. Department-name query classification (KEYWORD_SEARCH instead of AMBIGUOUS_CLARIFICATION).
5. Strict preservation of legal_effect_verified=False invariant.
"""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from app.agents.specialized.database_agent import OrderMetadataItem
from app.agents.specialized.search_agent import (
    RetrievalMode,
    SearchAgent,
    SearchAgentRequest,
    SearchAgentResponse,
    SearchResultItem,
)
from app.agents.specialized.supervisor_agent import (
    QueryIntent,
    SupervisorAgent,
    SupervisorPlan,
)
from app.agents.state import AgentState
from app.agents.workflow.graph_builder import WorkflowNodes
from app.api.schemas import QueryResponse, SearchResultItemSchema
from app.api.service import WorkflowService


class TestAgenticSearchResultMapping(unittest.TestCase):
    """Unit tests for Agentic Search result mapping and limit priorities."""

    def setUp(self) -> None:
        WorkflowService.reset_instance()
        self.qdrant_patcher = patch("app.services.qdrant_service.QdrantService.__init__", return_value=None)
        self.qdrant_patcher.start()
        self.service = WorkflowService.get_instance()

    def tearDown(self) -> None:
        self.qdrant_patcher.stop()

    def test_01_document_analysis_mapping_to_search_results(self) -> None:
        """
        Verify that WorkflowService._format_response() maps document_analysis items into
        structured search results with relevance_score=None and matched_sources=['DATABASE'].
        """
        doc_item = OrderMetadataItem(
            id=42,
            go_number="G.O.(P) No. 45/2024/Fin",
            issue_date="2024-03-15",
            department="Finance Department",
            source_filename="order45.pdf",
            extracted_text_length=500,
            text_preview="Government sanction is accorded for the revision of dearness allowance...",
        ).model_dump()

        state: AgentState = {
            "user_query": "G.O.(P) No. 45/2024/Fin",
            "intent": "GO_LOOKUP",
            "search_results": [],
            "document_analysis": [doc_item],
            "relationship_results": [],
            "workflow_status": "COMPLETED",
            "final_answer": "Found order G.O.(P) No. 45/2024/Fin in database records.",
            "verification_status": "VERIFIED",
            "confidence": "HIGH",
            "evidence_sufficient": True,
            "identified_go_numbers": ["G.O.(P) No. 45/2024/Fin"],
            "discovered_go_numbers": [],
            "execution_trace": [],
            "errors": [],
        }

        with patch("app.api.service.Path.is_file", return_value=True):
            resp: QueryResponse = self.service._format_response("thread-test-1", state)

        self.assertEqual(len(resp.search_results), 1)
        card: SearchResultItemSchema = resp.search_results[0]
        self.assertEqual(card.go_id, 42)
        self.assertEqual(card.go_number, "G.O.(P) No. 45/2024/Fin")
        self.assertEqual(card.department, "Finance Department")
        self.assertEqual(card.issue_date, "2024-03-15")
        self.assertEqual(card.source_filename, "order45.pdf")
        self.assertIn("dearness allowance", card.text_snippet)
        self.assertIsNone(card.relevance_score, "Relevance score must be None for exact database lookup")
        self.assertEqual(card.matched_sources, ["DATABASE"])
        self.assertTrue(card.has_pdf)
        self.assertEqual(card.pdf_url, "/api/v1/documents/42/pdf")
        self.assertFalse(resp.legal_effect_verified)

    def test_02_missing_or_unsafe_filename_handling(self) -> None:
        """
        Verify that missing, empty, or path-traversal filenames safely yield has_pdf=False and pdf_url=None.
        """
        unsafe_items = [
            {"id": 1, "go_number": "GO 1", "source_filename": "../../secret.pdf", "text_preview": "t1"},
            {"id": 2, "go_number": "GO 2", "source_filename": "nonexistent.pdf", "text_preview": "t2"},
            {"id": 3, "go_number": "GO 3", "source_filename": "", "text_preview": "t3"},
        ]

        state: AgentState = {
            "user_query": "lookup",
            "search_results": [],
            "document_analysis": unsafe_items,
            "workflow_status": "COMPLETED",
        }

        with patch("app.api.service.Path.is_file", return_value=False):
            resp: QueryResponse = self.service._format_response("thread-test-2", state)

        self.assertEqual(len(resp.search_results), 3)
        for card in resp.search_results:
            self.assertFalse(card.has_pdf)
            self.assertIsNone(card.pdf_url)

    def test_03_limit_priority_1_explicit_step_limit(self) -> None:
        """
        Priority 1: Explicit Supervisor plan step limit must override caller search_parameters limit.
        """
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="pension",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=0,
            results=[],
        )
        nodes = WorkflowNodes(search_agent=mock_search)

        state: AgentState = {
            "user_query": "pension",
            "search_parameters": {"limit": 10},  # Citizen dashboard limit
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "search_agent",
                        "tool_parameters": {"query": "pension", "limit": 3},  # Explicit step limit
                    }
                ]
            },
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
        }

        nodes.search_node(state)
        mock_search.execute_search.assert_called_once()
        req: SearchAgentRequest = mock_search.execute_search.call_args[0][0]
        self.assertEqual(req.limit, 3, "Explicit step limit of 3 must take priority over caller limit of 10")

    def test_04_limit_priority_2_citizen_dashboard_limit_10(self) -> None:
        """
        Priority 2: Citizen dashboard search_parameters={"limit": 10} is used when no step limit exists.
        """
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="pension",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=0,
            results=[],
        )
        nodes = WorkflowNodes(search_agent=mock_search)

        state: AgentState = {
            "user_query": "pension",
            "search_parameters": {"limit": 10},  # Citizen dashboard limit
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "search_agent",
                        "tool_parameters": {"query": "pension"},  # No explicit limit
                    }
                ]
            },
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
        }

        nodes.search_node(state)
        mock_search.execute_search.assert_called_once()
        req: SearchAgentRequest = mock_search.execute_search.call_args[0][0]
        self.assertEqual(req.limit, 10, "Caller search_parameters limit of 10 must be applied")

    def test_05_limit_priority_3_preserved_caller_default_5(self) -> None:
        """
        Priority 3: Callers without search_parameters retain the existing default limit of 5.
        """
        mock_search = MagicMock(spec=SearchAgent)
        mock_search.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="pension",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=0,
            results=[],
        )
        nodes = WorkflowNodes(search_agent=mock_search)

        state: AgentState = {
            "user_query": "pension",
            "search_parameters": None,  # No search parameters from caller
            "execution_plan": {
                "execution_steps": [
                    {
                        "step_number": 1,
                        "target_agent": "search_agent",
                        "tool_parameters": {"query": "pension"},  # No explicit limit
                    }
                ]
            },
            "plan_generation": 1,
            "completed_step_ids": [],
            "failed_step_ids": [],
        }

        nodes.search_node(state)
        mock_search.execute_search.assert_called_once()
        req: SearchAgentRequest = mock_search.execute_search.call_args[0][0]
        self.assertEqual(req.limit, 5, "Default limit of 5 must be preserved when no parameters are passed")

    def test_06_department_name_search_classification(self) -> None:
        """
        Verify that department-name queries like 'Finance Department' and 'Revenue Department'
        are classified as KEYWORD_SEARCH instead of AMBIGUOUS_CLARIFICATION.
        """
        mock_llm = MagicMock()
        mock_llm.generate_structured.side_effect = RuntimeError("Simulate LLM fallback")
        supervisor = SupervisorAgent(llm_service=mock_llm)

        # 1. Department queries should classify as KEYWORD_SEARCH
        dept_queries = [
            "Finance Department",
            "Revenue Department",
            "Finance",
            "General Administration Department",
            "P&ARD",
        ]
        for q in dept_queries:
            plan: SupervisorPlan = supervisor.create_initial_plan(q)
            self.assertEqual(
                plan.intent,
                QueryIntent.KEYWORD_SEARCH,
                f"Query '{q}' should be classified as KEYWORD_SEARCH, got {plan.intent}",
            )
            self.assertEqual(plan.next_action, "search")
            self.assertIn("search_agent", plan.selected_agents)

        # 2. Vague non-search queries must remain AMBIGUOUS_CLARIFICATION
        vague_queries = ["check order", "what happened", "hello"]
        for v in vague_queries:
            plan_vague: SupervisorPlan = supervisor.create_initial_plan(v)
            self.assertEqual(
                plan_vague.intent,
                QueryIntent.AMBIGUOUS_CLARIFICATION,
                f"Query '{v}' should remain AMBIGUOUS_CLARIFICATION, got {plan_vague.intent}",
            )
            self.assertEqual(plan_vague.next_action, "human_review")

    def test_07_legal_effect_invariant_strictly_false(self) -> None:
        """
        Confirm legal_effect_verified=False invariant is preserved across all response formatting.
        """
        state_search: AgentState = {
            "user_query": "pension",
            "search_results": [
                SearchResultItem(
                    go_id=1,
                    go_number="G.O. 1",
                    source_filename="o1.pdf",
                    matched_sources=["KEYWORD"],
                ).model_dump()
            ],
            "document_analysis": [],
            "workflow_status": "COMPLETED",
        }
        resp = self.service._format_response("t-search", state_search)
        self.assertFalse(resp.legal_effect_verified)

        state_doc: AgentState = {
            "user_query": "G.O. 1",
            "search_results": [],
            "document_analysis": [{"id": 1, "go_number": "G.O. 1", "source_filename": "o1.pdf"}],
            "workflow_status": "COMPLETED",
        }
        resp_doc = self.service._format_response("t-doc", state_doc)
        self.assertFalse(resp_doc.legal_effect_verified)

    def test_08_heuristic_short_topic_search_classification(self) -> None:
        """
        Verify heuristic fallback planning for short topic searches and routing preservation:
        1. 'dearness allowance' -> KEYWORD_SEARCH
        2. 'pension revision' -> KEYWORD_SEARCH
        3. 'pay revision' -> KEYWORD_SEARCH
        4. 'finance' -> KEYWORD_SEARCH
        5. 'finance department' -> KEYWORD_SEARCH
        6. 'hi' -> AMBIGUOUS_CLARIFICATION
        7. 'help' -> AMBIGUOUS_CLARIFICATION
        8. Exact GO number retains GO_LOOKUP
        9. Incoming relationship query retains INCOMING_RELATIONSHIPS
        10. Outgoing relationship query retains OUTGOING_RELATIONSHIPS
        Plus additional short topic queries (pension, salary, leave benefits, electricity, etc.).
        """
        supervisor = SupervisorAgent()

        # 1. Topic queries that MUST trigger KEYWORD_SEARCH
        keyword_search_cases = [
            "dearness allowance",
            "pension revision",
            "pay revision",
            "finance",
            "finance department",
            "pension",
            "salary",
            "leave benefits",
            "electricity",
            "revenue department",
            "government orders about salary",
        ]
        for q in keyword_search_cases:
            plan = supervisor._heuristic_fallback_plan(q)
            self.assertEqual(
                plan.intent,
                QueryIntent.KEYWORD_SEARCH,
                f"Query '{q}' should classify as KEYWORD_SEARCH, got {plan.intent}",
            )
            self.assertEqual(plan.next_action, "search")
            self.assertIn("search_agent", plan.selected_agents)
            self.assertFalse(plan.human_review_required)
            self.assertEqual(plan.execution_steps[0].target_agent, "search_agent")

        # 2. Genuinely uninformative queries that MUST trigger AMBIGUOUS_CLARIFICATION
        clarification_cases = [
            "hi",
            "help",
            "hello",
            "thanks",
            "   ",
            "",
            "check order",
            "what happened",
        ]
        for q in clarification_cases:
            plan = supervisor._heuristic_fallback_plan(q)
            self.assertEqual(
                plan.intent,
                QueryIntent.AMBIGUOUS_CLARIFICATION,
                f"Query '{q}' should classify as AMBIGUOUS_CLARIFICATION, got {plan.intent}",
            )
            self.assertEqual(plan.next_action, "human_review")
            self.assertIn("human_review", plan.selected_agents)
            self.assertTrue(plan.human_review_required)

        # 3. Exact GO number retains GO_LOOKUP
        plan_lookup = supervisor._heuristic_fallback_plan("G.O.(P) No. 45/2024/Fin")
        self.assertEqual(plan_lookup.intent, QueryIntent.GO_LOOKUP)
        self.assertEqual(plan_lookup.next_action, "database")
        self.assertIn("database_agent", plan_lookup.selected_agents)
        self.assertEqual(plan_lookup.identified_go_numbers, ["G.O.(P) No. 45/2024/Fin"])

        # 4. Incoming relationship query retains INCOMING_RELATIONSHIPS
        plan_incoming = supervisor._heuristic_fallback_plan(
            "Which later order modified G.O.(P) No. 45/2024/Fin"
        )
        self.assertEqual(plan_incoming.intent, QueryIntent.INCOMING_RELATIONSHIPS)
        self.assertEqual(plan_incoming.next_action, "database")
        self.assertEqual(plan_incoming.execution_steps[0].tool_parameters.get("direction"), "INCOMING")

        # 5. Outgoing relationship query retains OUTGOING_RELATIONSHIPS
        plan_outgoing = supervisor._heuristic_fallback_plan(
            "Which orders are modified by G.O.(P) No. 45/2024/Fin"
        )
        self.assertEqual(plan_outgoing.intent, QueryIntent.OUTGOING_RELATIONSHIPS)
        self.assertEqual(plan_outgoing.next_action, "database")
        self.assertEqual(plan_outgoing.execution_steps[0].tool_parameters.get("direction"), "OUTGOING")


if __name__ == "__main__":
    unittest.main()
