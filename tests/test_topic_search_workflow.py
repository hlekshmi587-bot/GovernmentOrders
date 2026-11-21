r"""
Comprehensive Automated Tests for LangGraph Multi-Agent Document-Discovery Workflow:
Validates:
1. Actual LangGraph node execution for KEYWORD_SEARCH:
   - supervisor_node -> search_node -> supervisor_node (evaluation) -> response_node
   - verification_node is strictly NOT executed.
   - verification_status is None and confidence is None.
2. Actual LangGraph node execution for INCOMING_RELATIONSHIPS and EXPLAIN_ORDER:
   - VerificationAgent is strictly executed.
   - verification_status is populated.
3. Actual LangGraph node execution for GO_LOOKUP:
   - Existing database lookup workflow preserved.
4. ResponseAgent document discovery synthesis:
   - Clear topic heading, no INSUFFICIENT_EVIDENCE, no unwanted backslashes.
   - Strict preservation of legal_effect_verified=False invariant.
5. Markdown normalization rules:
   - Normalizes \##, \*\*, \` into standard markdown markers.
   - Strictly preserves legitimate Windows paths (e.g. D:\GovernmentOrders, C:\Users) and LaTeX arrows ($\\rightarrow$).
"""

from datetime import date
import re
import unittest
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver

from app.agents.specialized.database_agent import (
    AgentResponseStatus,
    DatabaseAgent,
    DatabaseAgentResponse,
    OrderMetadataItem,
)
from app.agents.specialized.relationship_agent import (
    RelationshipAgent,
    RelationshipAgentResponse,
    RelationshipAgentStatus,
    RelationshipCandidate,
    RelationshipInvestigationMode,
)
from app.agents.specialized.response_agent import ResponseAgent, ResponseAgentInput
from app.agents.specialized.search_agent import (
    RetrievalMode,
    SearchAgent,
    SearchAgentRequest,
    SearchAgentResponse,
    SearchResultItem,
)
from app.agents.specialized.supervisor_agent import (
    PlannedStep,
    QueryIntent,
    SupervisorAgent,
    SupervisorPlan,
    SupervisorReplanDecision,
)
from app.agents.specialized.verification_agent import (
    AutonomousVerificationAction,
    EmptyParams,
    TargetAgentName,
    VerificationAgent,
    VerificationAgentResponse,
    VerificationConfidence,
    VerificationRecommendation,
    VerificationStatus,
)
from app.agents.state import AgentState
from app.agents.workflow.graph_builder import (
    WorkflowNodes,
    build_government_orders_graph,
)


def normalize_markdown_py(text: str) -> str:
  """Python mirror of frontend normalizeMarkdown function for parity validation."""
  if not text:
    return ""
  s = re.sub(r"^\\(#{1,6}\s)", r"\1", text, flags=re.MULTILINE)
  s = re.sub(r"\\(\*{1,2})(.*?)\\(\*{1,2})", r"\1\2\3", s)
  s = re.sub(r"(^|[^\\])\\(\*{1,2})", r"\1\2", s)
  s = re.sub(r"^\\(>\s*)", r"\1", s, flags=re.MULTILINE)
  s = re.sub(r"\\(`)(.*?)\\(`)", r"\1\2\3", s)
  return s


class TestTopicSearchWorkflow(unittest.TestCase):
  """Automated tests validating actual compiled LangGraph node execution."""

  def setUp(self) -> None:
    # Patch Qdrant initialization to prevent portalocker file locks when background server is running
    self.qdrant_patcher = patch(
        "app.services.qdrant_service.QdrantService.__init__", return_value=None
    )
    self.qdrant_patcher.start()

  def tearDown(self) -> None:
    self.qdrant_patcher.stop()

  def test_01_actual_node_execution_for_keyword_search(self) -> None:
    """Test that KEYWORD_SEARCH actually executes supervisor_node, search_node,

    supervisor_node evaluation, and response_node, strictly omitting
    verification_node.
    """
    mock_search_agent = MagicMock(spec=SearchAgent)
    mock_search_resp = SearchAgentResponse(
        status="SUCCESS",
        query="Dearness Allowance",
        requested_mode=RetrievalMode.HYBRID,
        candidates_retrieved=2,
        candidates_retained=2,
        returned_count=2,
        results=[
            SearchResultItem(
                go_id=7,
                go_number="G.O.(P)No.27/2026/FIN",
                department="Finance Department",
                issue_date="2026-02-15",
                source_filename="GO_27_2026_FIN.pdf",
                matched_sources=["KEYWORD", "VECTOR"],
                text_snippet="Dearness Allowance revision for state employees.",
            ),
            SearchResultItem(
                go_id=8,
                go_number="G.O.(P)No.28/2026/FIN",
                department="Finance Department",
                issue_date="2026-02-20",
                source_filename="GO_28_2026_FIN.pdf",
                matched_sources=["KEYWORD"],
                text_snippet="Dearness Relief for pensioners.",
            ),
        ],
    )
    mock_search_agent.execute_search.return_value = mock_search_resp

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    initial_plan = SupervisorPlan(
        intent=QueryIntent.KEYWORD_SEARCH,
        reasoning="Topic search regarding Dearness Allowance.",
        selected_agents=["search_agent"],
        selected_tools=["search_tool"],
        execution_steps=[
            PlannedStep(
                step_number=1,
                action_name="search_dearness_allowance",
                target_agent="search_agent",
                target_tool="search_tool",
                tool_parameters={"keyword": "Dearness Allowance", "limit": 10},
                expected_outcome="Matching Government Orders.",
            )
        ],
        expected_evidence="Relevant orders on Dearness Allowance.",
        next_action="search",
        confidence=0.9,
        plan_source="LLM",
    )
    mock_supervisor.create_initial_plan.return_value = initial_plan

    # Supervisor evaluation after search step completes
    replan_decision = SupervisorReplanDecision(
        evidence_sufficient=True,
        reasoning=(
            "Document discovery completed: search returned relevant indexed"
            " Government Orders."
        ),
        next_action="response",
        confidence=0.9,
        decision_source="HEURISTIC_FALLBACK",
    )
    mock_supervisor.evaluate_and_replan.return_value = replan_decision

    def _create_trace(agent, action, **kwargs):
      return {
          "agent": agent,
          "action": action,
          "decision": kwargs.get("decision"),
          "tool_called": kwargs.get("tool_called"),
      }

    mock_supervisor.create_trace_item.side_effect = _create_trace

    # Mock response agent
    mock_response_agent = MagicMock(spec=ResponseAgent)
    mock_response_agent.generate_response.return_value = (
        "## Government Orders Analysis Report\n\n### Indexed Government Orders"
        " on This Topic\n\nRetrieved 2 Government Order(s) from the indexed"
        " database."
    )

    # Mock verification agent (should never be called!)
    mock_verification_agent = MagicMock(spec=VerificationAgent)

    workflow_nodes = WorkflowNodes(
        supervisor_agent=mock_supervisor,
        search_agent=mock_search_agent,
        response_agent=mock_response_agent,
        verification_agent=mock_verification_agent,
    )

    graph = build_government_orders_graph(
        checkpointer=MemorySaver(), nodes=workflow_nodes
    )

    initial_state: AgentState = {
        "user_query": "Dearness Allowance",
        "intent": None,
        "identified_go_numbers": [],
        "discovered_go_numbers": [],
        "search_parameters": {"limit": 10},
        "execution_plan": None,
        "selected_agents": [],
        "next_action": None,
        "search_results": [],
        "document_analysis": [],
        "relationship_results": [],
        "verification_results": None,
        "evidence_sufficient": None,
        "confidence": None,
        "retry_count": 0,
        "max_retries": 2,
        "human_review_required": False,
        "human_review_status": "NONE",
        "human_review_notes": None,
        "human_feedback": None,
        "final_answer": None,
        "workflow_status": "INITIALIZED",
        "execution_trace": [],
        "errors": [],
        "plan_generation": 1,
        "invalid_plan_count": 0,
        "step_retry_count": 0,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "abandoned_step_ids": [],
        "response_ready": False,
        "verification_status": None,
    }

    config = {"configurable": {"thread_id": "test-keyword-node-execution"}}
    final_state = graph.invoke(initial_state, config=config)

    # 1. Assert search_node executed
    mock_search_agent.execute_search.assert_called_once()

    # 2. Assert response_node executed
    mock_response_agent.generate_response.assert_called_once()

    # 3. Assert verification_node was strictly NOT executed
    mock_verification_agent.execute.assert_not_called()

    # 4. Assert execution trace captures node progression
    agents_in_trace = [
        item["agent"] for item in final_state.get("execution_trace", [])
    ]
    self.assertIn("supervisor", agents_in_trace)
    self.assertIn("search_agent", agents_in_trace)
    self.assertIn("response_agent", agents_in_trace)
    self.assertNotIn("verification_agent", agents_in_trace)

    # 5. Assert verification_status and confidence are strictly None for discovery search
    self.assertIsNone(final_state.get("verification_status"))
    self.assertIsNone(final_state.get("confidence"))
    self.assertTrue(final_state.get("response_ready"))
    self.assertEqual(final_state.get("workflow_status"), "COMPLETED")

  def test_02_actual_node_execution_for_relationship_investigation(
      self,
  ) -> None:
    """Test that relationship queries strictly execute verification_node."""
    mock_db_agent = MagicMock(spec=DatabaseAgent)
    mock_db_agent.execute.return_value = DatabaseAgentResponse(
        status=AgentResponseStatus.SUCCESS,
        operation="GET_RELATIONSHIPS",
        results=[],
    )

    mock_rel_agent = MagicMock(spec=RelationshipAgent)
    mock_rel_agent.execute.return_value = RelationshipAgentResponse(
        status=RelationshipAgentStatus.SUCCESS,
        mode=RelationshipInvestigationMode.INVESTIGATE_ORDER,
        candidates=[
            RelationshipCandidate(
                source_go_number="G.O.(P) No.30/2026/FIN",
                target_go_number="G.O.(P) No.24/2026/ITD",
                relationship_type="PARTIALLY_AMENDS",
                evidence_text="Order 24 is hereby amended in part.",
                matched_phrase="amended in part",
                confidence="HIGH",
                extraction_method="PATTERN",
            )
        ],
    )

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    initial_plan = SupervisorPlan(
        intent=QueryIntent.INCOMING_RELATIONSHIPS,
        reasoning="Investigation of orders modifying 24/2026/ITD.",
        selected_agents=["database_agent", "relationship_agent"],
        selected_tools=["database_tool", "relationship_tool"],
        execution_steps=[
            PlannedStep(
                step_number=1,
                action_name="lookup_relationships",
                target_agent="database_agent",
                target_tool="database_tool",
                tool_parameters={
                    "go_number": "G.O.(P) No.24/2026/ITD",
                    "operation": "GET_RELATIONSHIPS",
                },
                expected_outcome="Relationship candidates.",
            )
        ],
        expected_evidence="Relationship candidates to verify.",
        next_action="database",
        confidence=0.9,
        plan_source="LLM",
    )
    mock_supervisor.create_initial_plan.return_value = initial_plan

    # Replan decision after database step: MUST route to verification!
    replan_decision = SupervisorReplanDecision(
        evidence_sufficient=False,
        reasoning=(
            "Retrieved relationship evidence must be verified by"
            " VerificationAgent before response synthesis."
        ),
        next_action="verification",
        confidence=0.9,
        decision_source="HEURISTIC_FALLBACK",
    )
    mock_supervisor.evaluate_and_replan.return_value = replan_decision

    def _create_trace(agent, action, **kwargs):
      return {"agent": agent, "action": action}

    mock_supervisor.create_trace_item.side_effect = _create_trace

    # Mock verification agent (MUST BE CALLED!)
    mock_verification_agent = MagicMock(spec=VerificationAgent)
    mock_verification_agent.execute.return_value = VerificationAgentResponse(
        status=VerificationStatus.PARTIALLY_VERIFIED,
        overall_confidence=VerificationConfidence.HIGH,
        confidence_score=0.85,
        is_complete=True,
        evidence_sufficient=True,
        legal_effect_verified=False,
        recommendation=VerificationRecommendation(
            action=AutonomousVerificationAction.PROCEED_TO_RESPONSE,
            target_agent=TargetAgentName.SUPERVISOR_AGENT,
            parameters=EmptyParams(),
            reasoning="Evidence sufficient to proceed to response.",
        ),
    )

    mock_response_agent = MagicMock(spec=ResponseAgent)
    mock_response_agent.generate_response.return_value = (
        "## Report\nVerified relationship report."
    )

    workflow_nodes = WorkflowNodes(
        supervisor_agent=mock_supervisor,
        database_agent=mock_db_agent,
        relationship_agent=mock_rel_agent,
        verification_agent=mock_verification_agent,
        response_agent=mock_response_agent,
    )

    graph = build_government_orders_graph(
        checkpointer=MemorySaver(), nodes=workflow_nodes
    )

    initial_state: AgentState = {
        "user_query": "Which orders modified G.O.(P) No.24/2026/ITD?",
        "intent": "INCOMING_RELATIONSHIPS",
        "identified_go_numbers": ["G.O.(P) No.24/2026/ITD"],
        "discovered_go_numbers": [],
        "search_parameters": {},
        "execution_plan": None,
        "selected_agents": [],
        "next_action": None,
        "search_results": [],
        "document_analysis": [],
        "relationship_results": [
            {
                "source_go_number": "G.O.(P) No.30/2026/FIN",
                "target_go_number": "G.O.(P) No.24/2026/ITD",
                "relationship_type": "PARTIALLY_AMENDS",
                "evidence_text": "Order 24 is hereby amended in part.",
            }
        ],
        "verification_results": None,
        "evidence_sufficient": None,
        "confidence": None,
        "retry_count": 0,
        "max_retries": 2,
        "human_review_required": False,
        "human_review_status": "NONE",
        "human_review_notes": None,
        "human_feedback": None,
        "final_answer": None,
        "workflow_status": "INITIALIZED",
        "execution_trace": [],
        "errors": [],
        "plan_generation": 1,
        "invalid_plan_count": 0,
        "step_retry_count": 0,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "abandoned_step_ids": [],
        "response_ready": False,
        "verification_status": None,
    }

    config = {"configurable": {"thread_id": "test-relationship-node-execution"}}
    final_state = graph.invoke(initial_state, config=config)

    # Assert verification_node strictly executed!
    mock_verification_agent.execute.assert_called_once()
    self.assertEqual(final_state.get("verification_status"), "PARTIALLY_VERIFIED")
    self.assertEqual(final_state.get("confidence"), "HIGH")

    agents_in_trace = [
        item["agent"] for item in final_state.get("execution_trace", [])
    ]
    self.assertIn("verification_agent", agents_in_trace)

  def test_03_response_agent_discovery_summary_generation(self) -> None:
    """Test ResponseAgent produces clean document discovery synthesis when

    verification_status is None.
    """
    resp_agent = ResponseAgent()
    input_data = ResponseAgentInput(
        user_query="Dearness Allowance",
        intent="KEYWORD_SEARCH",
        verification_status=None,
        confidence=None,
        evidence_sufficient=True,
        legal_effect_verified=False,
        search_results=[
            {
                "go_number": "G.O.(P)No.27/2026/FIN",
                "department": "Finance Department",
                "issue_date": "2026-02-15",
                "text_snippet": (
                    "Sanctioning of Dearness Allowance to State Government"
                    " Employees."
                ),
            },
            {
                "go_number": "G.O.(P)No.28/2026/FIN",
                "department": "Finance Department",
                "issue_date": "2026-02-20",
                "text_snippet": (
                    "Sanctioning of Dearness Relief to State Pensioners."
                ),
            },
        ],
    )

    out = resp_agent.generate_response(input_data)

    # 1. Assert citizen-friendly heading
    self.assertTrue(
        "### Retrieved Search Results" in out or "### Indexed Government Orders on This Topic" in out
    )
    self.assertIn("Retrieved 2 Government Order(s)", out)

    # 2. Assert orders and snippets present
    self.assertIn("G.O.(P)No.27/2026/FIN", out)
    self.assertIn("G.O.(P)No.28/2026/FIN", out)
    self.assertIn("Finance Department", out)

    # 3. Assert INSUFFICIENT_EVIDENCE is NOT present
    self.assertNotIn("INSUFFICIENT_EVIDENCE", out)
    self.assertNotIn("INCONCLUSIVE", out)

    # 4. Assert no unwanted backslashes before headings or bold markers
    self.assertNotIn(r"\##", out)
    self.assertNotIn(r"\**", out)
    self.assertNotIn(r"\`", out)

    # 5. Assert legal disclaimer is present
    self.assertIn("Legal effect remains unverified", out)

  def test_04_markdown_normalization_preserves_paths_and_latex(self) -> None:
    """Test Markdown normalization unescapes markers while preserving Windows

    paths and LaTeX arrows.
    """
    # 1. Escaped markdown markers should be unescaped
    escaped_input = (
        r"\## Analysis Report"
        + "\n"
        + r"\### 1. Topic Orders"
        + "\n"
        + r"\**Order:\** \`G.O. 27\`"
        + "\n"
        + r"\> [!IMPORTANT]"
    )
    normalized = normalize_markdown_py(escaped_input)
    self.assertIn("## Analysis Report", normalized)
    self.assertIn("### 1. Topic Orders", normalized)
    self.assertIn("**Order:** `G.O. 27`", normalized)
    self.assertIn("> [!IMPORTANT]", normalized)
    self.assertNotIn(r"\##", normalized)
    self.assertNotIn(r"\**", normalized)

    # 2. Windows file paths MUST be preserved without corrupting backslashes!
    win_path = r"D:\GovernmentOrders\dataset\orders\GO_27.pdf"
    norm_path = normalize_markdown_py(win_path)
    self.assertEqual(norm_path, win_path)

    win_user = r"C:\Users\ACER\AppData\Local"
    norm_user = normalize_markdown_py(win_user)
    self.assertEqual(norm_user, win_user)

    # 3. LaTeX arrows MUST be preserved intact!
    latex_text = r"- Order A $\rightarrow$ Action $\rightarrow$ Order B"
    norm_latex = normalize_markdown_py(latex_text)
    self.assertEqual(norm_latex, latex_text)


  def test_05_actual_node_execution_for_exact_go_lookup(self) -> None:
    """Test that GO_LOOKUP executes supervisor_node, database_node, and existing verification workflow."""
    mock_db_agent = MagicMock(spec=DatabaseAgent)
    mock_db_agent.execute.return_value = DatabaseAgentResponse(
        status=AgentResponseStatus.SUCCESS,
        operation="LOOKUP_ORDER",
        results=[
            OrderMetadataItem(
                id=7,
                go_number="G.O.(P) No.27/2026/FIN",
                department="Finance Department",
                issue_date="2026-02-15",
                source_filename="GO_27_2026_FIN.pdf",
            )
        ],
    )

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    initial_plan = SupervisorPlan(
        intent=QueryIntent.GO_LOOKUP,
        reasoning="Exact lookup for G.O.(P) No.27/2026/FIN.",
        selected_agents=["database_agent"],
        selected_tools=["database_tool"],
        execution_steps=[
            PlannedStep(
                step_number=1,
                action_name="lookup_exact_order",
                target_agent="database_agent",
                target_tool="database_tool",
                tool_parameters={"go_number": "G.O.(P) No.27/2026/FIN"},
                expected_outcome="Order details and metadata.",
            )
        ],
        expected_evidence="Authoritative metadata for order.",
        next_action="database",
        confidence=0.9,
        plan_source="LLM",
    )
    mock_supervisor.create_initial_plan.return_value = initial_plan

    replan_decision = SupervisorReplanDecision(
        evidence_sufficient=True,
        reasoning="Database lookup retrieved order record; proceeding to verification.",
        next_action="verification",
        confidence=0.9,
        decision_source="HEURISTIC_FALLBACK",
    )
    mock_supervisor.evaluate_and_replan.return_value = replan_decision

    def _create_trace(agent, action, **kwargs):
      return {"agent": agent, "action": action}

    mock_supervisor.create_trace_item.side_effect = _create_trace

    mock_verification_agent = MagicMock(spec=VerificationAgent)
    mock_verification_agent.execute.return_value = VerificationAgentResponse(
        status=VerificationStatus.VERIFIED,
        overall_confidence=VerificationConfidence.HIGH,
        confidence_score=0.95,
        is_complete=True,
        evidence_sufficient=True,
        legal_effect_verified=False,
        recommendation=VerificationRecommendation(
            action=AutonomousVerificationAction.PROCEED_TO_RESPONSE,
            target_agent=TargetAgentName.SUPERVISOR_AGENT,
            parameters=EmptyParams(),
            reasoning="Metadata verified in authoritative records.",
        ),
    )

    mock_response_agent = MagicMock(spec=ResponseAgent)
    mock_response_agent.generate_response.return_value = (
        "## Order Analysis\nOrder 27 verified."
    )

    workflow_nodes = WorkflowNodes(
        supervisor_agent=mock_supervisor,
        database_agent=mock_db_agent,
        verification_agent=mock_verification_agent,
        response_agent=mock_response_agent,
    )

    graph = build_government_orders_graph(
        checkpointer=MemorySaver(), nodes=workflow_nodes
    )

    initial_state: AgentState = {
        "user_query": "G.O.(P) No.27/2026/FIN",
        "intent": "GO_LOOKUP",
        "identified_go_numbers": ["G.O.(P) No.27/2026/FIN"],
        "discovered_go_numbers": [],
        "search_parameters": {},
        "execution_plan": None,
        "selected_agents": [],
        "next_action": None,
        "search_results": [],
        "document_analysis": [],
        "relationship_results": [],
        "verification_results": None,
        "evidence_sufficient": None,
        "confidence": None,
        "retry_count": 0,
        "max_retries": 2,
        "human_review_required": False,
        "human_review_status": "NONE",
        "human_review_notes": None,
        "human_feedback": None,
        "final_answer": None,
        "workflow_status": "INITIALIZED",
        "execution_trace": [],
        "errors": [],
        "plan_generation": 1,
        "invalid_plan_count": 0,
        "step_retry_count": 0,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "abandoned_step_ids": [],
        "response_ready": False,
        "verification_status": None,
    }

    config = {"configurable": {"thread_id": "test-go-lookup-node-execution"}}
    final_state = graph.invoke(initial_state, config=config)

    # 1. Assert database_node executed
    mock_db_agent.execute.assert_called_once()

    # 2. Assert verification_node executed (existing behavior preserved!)
    mock_verification_agent.execute.assert_called_once()

    # 3. Assert response_node executed
    mock_response_agent.generate_response.assert_called_once()

    # 4. Assert trace contains supervisor, database_agent, verification_agent, response_agent
    agents_in_trace = [
        item["agent"] for item in final_state.get("execution_trace", [])
    ]
    self.assertIn("supervisor", agents_in_trace)
    self.assertIn("database_agent", agents_in_trace)
    self.assertIn("verification_agent", agents_in_trace)
    self.assertIn("response_agent", agents_in_trace)
    self.assertEqual(final_state.get("verification_status"), "VERIFIED")
    self.assertEqual(final_state.get("confidence"), "HIGH")


if __name__ == "__main__":
  unittest.main()
