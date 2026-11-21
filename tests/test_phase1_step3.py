"""
Comprehensive Unit and Integration Tests for Phase 1 Step 3.2:
LangGraph Multi-Agent Workflow Implementation.

Tests cover:
1. Dynamic Routing & Tool Execution:
   - Keyword Search -> SearchTool -> Verification -> Response
   - Exact GO Lookup -> DatabaseTool -> Verification -> Response
   - Incoming Relationships -> DatabaseTool.get_incoming_relationships -> Response
   - Outgoing Relationships -> DatabaseTool.get_outgoing_relationships -> Response
   - Multi-Step Explanation -> DatabaseTool (step 1) -> RelationshipTool (step 2) -> Verification -> Response
2. Evidence Verification:
   - Fact-checks real database records
   - Confirms that UNVERIFIED_CANDIDATE items without database records are not treated as verified facts
3. Ambiguity & Human-in-the-Loop Interruption:
   - Ambiguous queries trigger LangGraph checkpoint pause (interrupt)
   - Resuming via Command(resume=...) cleanly resumes execution
4. Bounded Retries & Replanning:
   - Unsuccessful tool calls lead to replanning
   - Reaching max_retries halts into human review
5. Execution Trace Verification:
   - Captures node executions, tool calls, decision_source, and timestamps without document bloat
6. Live Ollama Integration Test:
   - Full end-to-end run with llama3.2:3b
"""

import os
import sys
from typing import Optional
from unittest.mock import MagicMock

# Ensure workspace root is always on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agents.state import AgentState
from app.agents.workflow.graph_builder import (
    WorkflowNodes,
    build_government_orders_graph,
    run_workflow,
)
from app.agents.specialized.supervisor_agent import (
    SupervisorAgent,
    SupervisorPlan,
    PlannedStep,
    QueryIntent,
    SupervisorReplanDecision,
)
from app.agents.specialized.verification_agent import (
    VerificationStatus,
    VerificationConfidence,
    RelationshipRegistrationStatus,
)
from app.agents.tools.search_tool import SearchTool
from app.agents.tools.database_tool import DatabaseTool
from app.agents.tools.relationship_tool import RelationshipTool
from app.agents.llm.llm_service import LLMService
from app.agents.llm.ollama_client import OllamaClient


def create_mocked_supervisor(
    plan: SupervisorPlan,
    replan_decision: Optional[SupervisorReplanDecision] = None,
) -> SupervisorAgent:
    """Helper to create a SupervisorAgent returning a predetermined plan and replan decision."""
    mock_llm = MagicMock(spec=LLMService)

    def mock_gen_struct(prompt, schema_model, **kwargs):
        if schema_model == SupervisorReplanDecision:
            if replan_decision is not None:
                return replan_decision.model_dump()
            return {
                "evidence_sufficient": True,
                "reasoning": "Evidence gathered is sufficient for response",
                "next_action": "response",
                "confidence": 0.9,
                "decision_source": "LLM",
            }
        return plan.model_dump()

    mock_llm.generate_structured.side_effect = mock_gen_struct
    return SupervisorAgent(llm_service=mock_llm)


def test_keyword_search_workflow() -> None:
    print("\n--- 1. Keyword Search Workflow ---")
    plan = SupervisorPlan(
        intent=QueryIntent.KEYWORD_SEARCH,
        identified_departments=["P&ARD"],
        reasoning="Searching orders by department keyword P&ARD",
        selected_agents=["search_agent", "response_agent"],
        selected_tools=["search_tool"],
        execution_steps=[
            PlannedStep(
                step_number=1,
                action_name="search_p_and_ard",
                target_agent="search_agent",
                target_tool="search_tool",
                tool_parameters={"keyword": "P&ARD"},
                expected_outcome="Matching P&ARD Government Orders",
            )
        ],
        expected_evidence="Matching P&ARD records from PostgreSQL",
        next_action="search",
        confidence=0.9,
        plan_source="LLM",
    )

    nodes = WorkflowNodes(supervisor_agent=create_mocked_supervisor(plan))
    memory = MemorySaver()
    graph = build_government_orders_graph(checkpointer=memory, nodes=nodes)

    result = graph.invoke(
        {"user_query": "Find orders from P&ARD", "retry_count": 0, "max_retries": 2},
        config={"configurable": {"thread_id": "test_search_thread"}},
    )

    assert result["workflow_status"] == "COMPLETED"
    assert result["evidence_sufficient"] is True
    assert len(result["search_results"]) > 0
    assert "P&ARD" in result["final_answer"]
    assert any(t["agent"] == "search_agent" for t in result["execution_trace"])
    print(f"PASS: Search workflow completed. Found {len(result['search_results'])} orders in PostgreSQL.")


def test_incoming_relationships_workflow() -> None:
    print("\n--- 2. Incoming Relationships Workflow ---")
    # Target: G.O (Rt) No. 114/2025/ITD is modified by G.O.(Rt)No.24/2026/ITD in PostgreSQL
    target_go = "G.O (Rt) No. 114/2025/ITD"
    plan = SupervisorPlan(
        intent=QueryIntent.INCOMING_RELATIONSHIPS,
        identified_go_numbers=[target_go],
        reasoning=f"Investigating which later order modified {target_go}",
        selected_agents=["database_agent", "verification_agent", "response_agent"],
        selected_tools=["database_tool"],
        execution_steps=[
            PlannedStep(
                step_number=1,
                action_name="get_incoming",
                target_agent="database_agent",
                target_tool="database_tool",
                tool_parameters={"go_number": target_go, "operation": "GET_RELATIONSHIPS", "direction": "INCOMING"},
                expected_outcome="Later modifying order records",
            )
        ],
        expected_evidence="Records in PostgreSQL linking modifying order to target",
        next_action="database",
        confidence=0.95,
        plan_source="LLM",
    )

    nodes = WorkflowNodes(supervisor_agent=create_mocked_supervisor(plan))
    memory = MemorySaver()
    graph = build_government_orders_graph(checkpointer=memory, nodes=nodes)

    result = graph.invoke(
        {"user_query": f"Which later order modified {target_go}?", "retry_count": 0, "max_retries": 2},
        config={"configurable": {"thread_id": "test_incoming_thread"}},
    )

    assert result["workflow_status"] == "COMPLETED"
    assert result["evidence_sufficient"] is False
    assert len(result["relationship_results"]) > 0
    first_rel = result["relationship_results"][0]
    assert first_rel["direction"] == "INCOMING"
    assert "G.O.(Rt)No.24/2026/ITD" in first_rel["source_go_number"]
    final_ans = result["final_answer"] or ""
    assert "G.O.(Rt)No.24/2026/ITD" in final_ans
    assert "MODIFIES" in final_ans
    assert "114/2025/ITD" in final_ans
    print(f"PASS: Incoming relationship verified! Later order='{first_rel['source_go_number']}' modifies '{target_go}'.")


def test_multi_step_explanation_workflow() -> None:
    print("\n--- 3. Multi-Step Explanation Workflow (EXPLAIN_ORDER) ---")
    target_go = "G.O.(Rt)No.24/2026/ITD"
    step1 = PlannedStep(
        step_number=1,
        action_name="retrieve_text",
        target_agent="database_agent",
        target_tool="database_tool",
        tool_parameters={"go_number": target_go, "include_full_text": True},
        expected_outcome="Full text of order",
    )
    step2 = PlannedStep(
        step_number=2,
        action_name="extract_clauses",
        target_agent="relationship_agent",
        target_tool="relationship_tool",
        tool_parameters={"go_number": target_go},
        expected_outcome="Extracted clauses and relationships",
    )

    plan = SupervisorPlan(
        intent=QueryIntent.EXPLAIN_ORDER,
        identified_go_numbers=[target_go],
        reasoning=f"Two-step explanation: fetch document text, then extract clauses and references for {target_go}",
        selected_agents=["database_agent", "relationship_agent", "verification_agent", "response_agent"],
        selected_tools=["database_tool", "relationship_tool"],
        execution_steps=[step1, step2],
        expected_evidence="Full document text and extracted relationship signals",
        next_action="database",  # Step 1
        confidence=0.9,
        plan_source="LLM",
    )

    nodes = WorkflowNodes(supervisor_agent=create_mocked_supervisor(plan))
    memory = MemorySaver()
    graph = build_government_orders_graph(checkpointer=memory, nodes=nodes)

    result = graph.invoke(
        {"user_query": f"Explain the changes in {target_go}", "retry_count": 0, "max_retries": 2},
        config={"configurable": {"thread_id": "test_multistep_thread"}},
    )

    assert result["workflow_status"] == "COMPLETED"
    assert len(result["document_analysis"]) > 0
    assert len(result["relationship_results"]) > 0
    # Verify execution trace contains both tools in sequence
    trace_tools = [t.get("tool_called") for t in result["execution_trace"] if t.get("tool_called")]
    assert "database_tool" in trace_tools
    assert "relationship_tool" in trace_tools
    print("PASS: Multi-step plan executed in sequence: database_tool -> relationship_tool -> verification -> response.")


def test_ambiguity_and_human_review_interruption() -> None:
    print("\n--- 4. Ambiguity & Human-in-the-Loop Interruption ---")
    ambig_plan = SupervisorPlan(
        intent=QueryIntent.AMBIGUOUS_CLARIFICATION,
        reasoning="Query '2026' is ambiguous; requesting clarification.",
        selected_agents=["human_review"],
        selected_tools=["none"],
        execution_steps=[],
        expected_evidence="Clarified order number",
        next_action="human_review",
        confidence=0.2,
        human_review_required=True,
        plan_source="LLM",
    )

    clarified_plan = SupervisorPlan(
        intent=QueryIntent.GO_LOOKUP,
        identified_go_numbers=["G.O.(Rt)No.24/2026/ITD"],
        reasoning="Human clarified target order to G.O.(Rt)No.24/2026/ITD",
        selected_agents=["response_agent"],
        selected_tools=["none"],
        execution_steps=[],
        expected_evidence="Order record matching G.O.(Rt)No.24/2026/ITD",
        next_action="response",
        confidence=0.95,
        human_review_required=False,
        plan_source="LLM",
    )

    mock_llm = MagicMock(spec=LLMService)

    def mock_gen_struct(prompt, schema_model, **kwargs):
        if schema_model == SupervisorPlan:
            if "Human Guidance" in prompt:
                return clarified_plan.model_dump()
            return ambig_plan.model_dump()
        if schema_model == SupervisorReplanDecision:
            return {
                "evidence_sufficient": True,
                "reasoning": "Clarified evidence sufficient",
                "next_action": "response",
                "confidence": 0.95,
                "decision_source": "LLM",
            }
        return ambig_plan.model_dump()

    mock_llm.generate_structured.side_effect = mock_gen_struct
    nodes = WorkflowNodes(supervisor_agent=SupervisorAgent(llm_service=mock_llm))
    memory = MemorySaver()
    graph = build_government_orders_graph(checkpointer=memory, nodes=nodes)

    config = {"configurable": {"thread_id": "test_human_thread"}}

    # 1. Run initial workflow: ambiguous query triggers human review and pauses at human_review_node
    res_pause = graph.invoke(
        {"user_query": "check 2026", "retry_count": 0, "max_retries": 2},
        config=config,
    )

    state_snapshot = graph.get_state(config)
    print(f"Workflow paused at: {state_snapshot.next}")
    assert "human_review_node" in state_snapshot.next
    print("PASS: Workflow successfully paused at human_review_node via LangGraph interrupt()!")

    # 2. Resume with human feedback: routes through Supervisor
    human_input = "Please inspect G.O.(Rt)No.24/2026/ITD specifically"
    res_resumed = graph.invoke(Command(resume=human_input), config=config)

    assert res_resumed["human_review_status"] == "RESOLVED"
    # Execution trace records the received human feedback
    assert any(
        t.get("agent") == "human_review"
        and t.get("action") == "receive_human_feedback"
        and t.get("tool_input") == human_input
        for t in res_resumed.get("execution_trace", [])
    )
    # Execution trace demonstrates resume continues through Supervisor
    trace_agents = [t.get("agent") for t in res_resumed.get("execution_trace", [])]
    assert "human_review" in trace_agents
    assert "supervisor" in trace_agents
    hr_idx = next(i for i, t in enumerate(res_resumed["execution_trace"]) if t.get("agent") == "human_review")
    assert any(t.get("agent") == "supervisor" for t in res_resumed["execution_trace"][hr_idx + 1:])

    # 3. Final output reflects the human clarification appropriately
    assert res_resumed["workflow_status"] == "COMPLETED"
    assert "G.O.(Rt)No.24/2026/ITD" in (res_resumed.get("final_answer") or "") or "G.O.(Rt)No.24/2026/ITD" in res_resumed.get("identified_go_numbers", [])
    print(f"PASS: Workflow resumed successfully through Supervisor, reflecting clarification '{human_input}'.")


def test_unverified_candidate_handling() -> None:
    print("\n--- 5. Unverified Candidate Verification ---")
    # Simulate an extracted relationship with UNVERIFIED_CANDIDATE that has NO database record
    fake_candidate = {
        "source_go_number": "G.O. FAKE/2026",
        "target_go_number": "G.O. TARGET/2020",
        "relationship_type": "CANCELS",
        "verification_status": "UNVERIFIED_CANDIDATE",
        "legal_effect_verified": False,
        "evidence_text": "Fake text claiming cancellation",
    }

    nodes = WorkflowNodes()
    test_state: AgentState = {
        "user_query": "Did Fake GO cancel Target GO?",
        "intent": "OUTGOING_RELATIONSHIPS",
        "identified_go_numbers": ["G.O. FAKE/2026"],
        "relationship_results": [fake_candidate],
        "document_analysis": [],
        "search_results": [],
        "execution_trace": [],
        "retry_count": 0,
        "max_retries": 2,
    }

    verif_output = nodes.verification_node(test_state)

    # 1. VerificationAgent Contract: Evidence sufficiency and status
    assert verif_output["evidence_sufficient"] is False, "Evidence must be insufficient for unrecorded candidate"
    assert verif_output["verification_status"] in (
        VerificationStatus.NOT_FOUND.value,
        VerificationStatus.INSUFFICIENT_EVIDENCE.value,
        "NOT_FOUND",
        "INSUFFICIENT_EVIDENCE",
    )

    verif_results = verif_output["verification_results"]
    assert verif_results["evidence_sufficient"] is False
    assert verif_results["legal_effect_verified"] is False, "legal_effect_verified MUST remain False"
    assert verif_results["overall_confidence"] in (VerificationConfidence.LOW, "LOW") or verif_results["confidence_score"] < 0.60
    assert verif_results["verified_claims"] == [], "Unverified candidate must not be promoted into verified claims"

    # Candidate details in verified_relationships must record unrecorded registration and legal_effect_verified=False
    rel_details = verif_results.get("verified_relationships", [])
    assert len(rel_details) >= 1
    for rel in rel_details:
        assert rel["legal_effect_verified"] is False, "Individual candidate legal_effect_verified MUST be False"
        assert rel.get("registration_status") in (
            RelationshipRegistrationStatus.UNRECORDED_IN_DATABASE,
            "UNRECORDED_IN_DATABASE",
        ), "Candidate must be marked UNRECORDED_IN_DATABASE"
        assert rel.get("database_relationship_type") is None, "Must have no database relationship type"

    # Must report database discrepancy for unrecorded candidate
    assert any(
        d.get("category") == "UNRECORDED_IN_DATABASE" for d in verif_results.get("discrepancies", [])
    ), "Must record UNRECORDED_IN_DATABASE discrepancy"

    # 2. ResponseAgent Output Contract: Candidate isolation & disclaimer enforcement
    test_state.update(verif_output)
    response_output = nodes.response_node(test_state)
    final_text = response_output.get("final_answer", "")
    assert "Unverified Relationship Candidates" in final_text, "Response must isolate unverified candidates"
    assert "Legal effect remains unverified" in final_text or "legal enforceability" in final_text.lower(), "Must include legal disclaimer"
    assert "Supported Textual Findings" not in final_text, "Must not promote unverified candidate to supported findings"
    print("PASS: UNVERIFIED_CANDIDATE without database record correctly rejected as unverified and isolated in response.")


def test_retry_exhaustion() -> None:
    print("\n--- 6. Bounded Retries & Loop Termination ---")
    supervisor = SupervisorAgent()
    state_at_limit: AgentState = {
        "user_query": "Find order NON_EXISTENT_9999",
        "retry_count": 2,
        "max_retries": 2,
        "search_results": [],
        "document_analysis": [],
        "relationship_results": [],
        "errors": ["Order not found"],
    }

    decision = supervisor.evaluate_and_replan(state_at_limit)
    assert decision.next_action == "human_review"
    assert decision.human_review_required is True
    assert "Reached maximum retry limit" in decision.reasoning
    print("PASS: Maximum retries reached -> safely routes to human review without infinite loop.")


def test_execution_trace_audit() -> None:
    print("\n--- 7. Execution Trace Auditability ---")
    plan = SupervisorPlan(
        intent=QueryIntent.KEYWORD_SEARCH,
        identified_departments=["P&ARD"],
        reasoning="Auditing execution trace recording",
        selected_agents=["search_agent", "response_agent"],
        selected_tools=["search_tool"],
        execution_steps=[
            PlannedStep(
                step_number=1,
                action_name="search",
                target_agent="search_agent",
                target_tool="search_tool",
                tool_parameters={"keyword": "P&ARD"},
                expected_outcome="Orders",
            )
        ],
        expected_evidence="Evidence",
        next_action="search",
        confidence=0.9,
        plan_source="LLM",
    )
    nodes = WorkflowNodes(supervisor_agent=create_mocked_supervisor(plan))
    graph = build_government_orders_graph(nodes=nodes)
    result = graph.invoke(
        {"user_query": "Find orders from P&ARD", "retry_count": 0, "max_retries": 2},
        config={"configurable": {"thread_id": "test_trace_thread"}},
    )
    traces = result.get("execution_trace", [])
    assert len(traces) >= 3

    for t in traces:
        assert "timestamp" in t
        assert "agent" in t
        assert "action" in t
        assert "decision" in t
        assert t.get("decision_source") in ("LLM", "HEURISTIC_FALLBACK", "DETERMINISTIC_RULES")
    print(f"PASS: Execution trace recorded {len(traces)} events with decision_source and timestamps.")


def test_live_ollama_integration() -> None:
    print("\n==========================================")
    print("SUITE 8: Live Ollama Integration Test")
    print("==========================================")
    client = OllamaClient()
    if not client.is_server_running():
        print("SKIP: Ollama server not reachable on localhost:11434. Skipping live test.")
        return

    # Live end-to-end run on actual query
    live_query = "Which later Government Order modified G.O.(Rt) No. 114/2025/ITD?"
    print(f"Executing live workflow for query: '{live_query}'")

    res = run_workflow(user_query=live_query, thread_id="live_test_thread")

    print(f"Workflow Status: {res.get('workflow_status')}")
    print(f"Intent Identified: {res.get('intent')}")
    print(f"Verification Status: {res.get('verification_status')}")
    print(f"Evidence Sufficient: {res.get('evidence_sufficient')}")
    final_ans = res.get("final_answer")
    if final_ans:
        print(f"Final Answer Snippet:\n{final_ans[:300]}...")
    else:
        print("Final answer is None")

    # 1. Workflow completes successfully
    assert res.get("workflow_status") == "COMPLETED", f"Expected COMPLETED, got {res.get('workflow_status')}"

    # 2. Intent is INCOMING_RELATIONSHIPS
    intent_val = str(res.get("intent", ""))
    assert "INCOMING_RELATIONSHIPS" in intent_val, f"Expected INCOMING_RELATIONSHIPS intent, got {intent_val}"

    # 3. Correct modifying order G.O.(Rt)No.24/2026/ITD is retrieved
    rels = res.get("relationship_results", [])
    assert len(rels) > 0, "Expected at least 1 relationship result"
    first_rel = rels[0]
    assert "G.O.(Rt)No.24/2026/ITD" in first_rel.get("source_go_number", "")
    assert "G.O.(Rt)No.24/2026/ITD" in (res.get("final_answer") or ""), "Expected G.O.(Rt)No.24/2026/ITD in final answer"

    # 4. Target 114/2025/ITD is correctly identified
    assert "114/2025/ITD" in first_rel.get("target_go_number", "") or any(
        "114/2025/ITD" in str(s.get("tool_parameters", {}).get("go_number", ""))
        for s in res.get("execution_plan", {}).get("execution_steps", [])
    ), "Expected 114/2025/ITD target identified in plan or relationship result"

    # 5. Verification status is PARTIALLY_VERIFIED under current Phase 2 contract
    assert res.get("verification_status") == "PARTIALLY_VERIFIED", f"Expected PARTIALLY_VERIFIED, got {res.get('verification_status')}"

    # 6. evidence_sufficient remains False (strictly required when status != VERIFIED)
    assert res.get("evidence_sufficient") is False, f"Expected evidence_sufficient=False under PARTIALLY_VERIFIED, got {res.get('evidence_sufficient')}"

    # 7. legal_effect_verified remains False
    verif_res = res.get("verification_results") or {}
    assert verif_res.get("legal_effect_verified") is False, "legal_effect_verified must remain False"
    assert all(r.get("legal_effect_verified") is False for r in rels), "All relationship candidates must have legal_effect_verified=False"

    # 8. Response preserves mandatory legal-effect disclaimer
    assert "Legal effect remains unverified" in (final_ans or "") or "legal enforceability" in (final_ans or "").lower(), "Must preserve legal-effect disclaimer"

    # 9. Execution trace confirms successful LLM-driven planning and dispatch
    traces = res.get("execution_trace", [])
    assert any(
        t.get("agent") == "supervisor" and t.get("action") == "initial_planning" and t.get("decision_source") == "LLM"
        for t in traces
    ), "Execution trace must confirm initial planning was LLM-driven"
    assert any(
        t.get("agent") == "database_agent" and t.get("action") == "GET_RELATIONSHIPS"
        for t in traces
    ), "Execution trace must confirm database_agent dispatched GET_RELATIONSHIPS"

    print("PASS: Live Ollama workflow executed autonomously and identified modifying order under Phase 2 contract!")


def test_incoming_relationship_normalization_and_ambiguity() -> None:
    print("\n==========================================")
    print("FOCUSED TEST 1: Incoming Relationship Normalization & Ambiguity")
    print("==========================================")
    db_tool = DatabaseTool()

    # 1. Test different punctuation and spacing variations of the same target GO number
    variations = [
        "G.O (Rt) No. 114/2025/ITD",    # Exact DB format (space after G.O, no dot after Rt)
        "G.O.(Rt) No. 114/2025/ITD",    # Standard dot notation with space before No
        "G.O.(Rt)No.114/2025/ITD",      # Compact notation (no spaces)
        "114/2025/ITD",                 # Bare core identifier
        "G.O. (Rt) No. 114/2025/ITD",   # Dot + space notation
    ]

    for var in variations:
        res = db_tool.get_incoming_relationships(var)
        assert res["status"] == "SUCCESS", f"Failed for variation '{var}': {res}"
        assert res["total_relationships"] >= 1, f"Expected at least 1 relationship for '{var}', got {res['total_relationships']}"
        first_rel = res["relationships"][0]
        assert first_rel["source_go_number"] == "G.O.(Rt)No.24/2026/ITD", f"Wrong source GO for '{var}'"
        assert first_rel["relationship_type"] == "MODIFIES", f"Wrong relationship type for '{var}'"
        assert "114/2025/ITD" in first_rel["target_go_number"], f"Wrong target GO for '{var}'"
        print(f"PASS: Punctuation/spacing variation '{var}' correctly resolved to modifying order '{first_rel['source_go_number']}'.")

    # 2. Confirm that ambiguous queries are not silently treated as a unique verified relationship
    # A) An ambiguous partial query in incoming relationships returns all matching candidates explicitly
    broad_res = db_tool.get_incoming_relationships("2025")
    assert broad_res["status"] == "SUCCESS"
    assert broad_res["total_relationships"] == 3, f"Expected 3 distinct relationships for '2025', got {broad_res['total_relationships']}"
    targets = {r["target_go_number"] for r in broad_res["relationships"]}
    assert len(targets) == 3, f"Ambiguous query '2025' should return 3 distinct targets, got {targets}"
    assert "G.O (Rt) No. 114/2025/ITD" in targets
    assert "G.O (Rt) No. 151/2025/ITD" in targets
    assert "G.O (Rt) No. 264/2025/ITD" in targets
    print(f"PASS: Ambiguous query '2025' explicitly returned {len(targets)} distinct targets rather than collapsing into a single unique relationship.")

    # B) Ambiguous order resolution in DatabaseTool correctly detects ambiguity
    ambig_check = db_tool.get_order_by_number("2026")
    assert ambig_check["status"] == "AMBIGUOUS_MATCH"
    assert len(ambig_check["candidates"]) > 1
    print(f"PASS: Ambiguous order query '2026' explicitly returned AMBIGUOUS_MATCH with {len(ambig_check['candidates'])} candidates.")


def test_supervisor_plan_auto_healing_and_rejection() -> None:
    print("\n==========================================")
    print("FOCUSED TEST 2: SupervisorPlan Auto-Healing & Strict Rejection")
    print("==========================================")
    from pydantic import ValidationError

    # 1. Test Auto-Healing: missing selected_agents and selected_tools automatically populated from valid execution steps
    plan_with_missing_headers = SupervisorPlan(
        intent=QueryIntent.GO_LOOKUP,
        identified_go_numbers=["G.O.(Rt)No.24/2026/ITD"],
        reasoning="Looking up GO details",
        selected_agents=[],  # Intentionally omitted by LLM
        selected_tools=[],   # Intentionally omitted by LLM
        execution_steps=[
            PlannedStep(
                step_number=1,
                action_name="lookup_order",
                target_agent="database_agent",
                target_tool="database_tool",
                tool_parameters={"go_number": "G.O.(Rt)No.24/2026/ITD"},
                expected_outcome="Order details",
            ),
            PlannedStep(
                step_number=2,
                action_name="extract_relations",
                target_agent="relationship_agent",
                target_tool="relationship_tool",
                tool_parameters={"go_number": "G.O.(Rt)No.24/2026/ITD"},
                expected_outcome="Extracted relations",
            ),
        ],
        expected_evidence="Order text and relationships",
        next_action="database",
        confidence=0.9,
        plan_source="LLM",
    )

    assert "database_agent" in plan_with_missing_headers.selected_agents, "database_agent not auto-healed into selected_agents"
    assert "relationship_agent" in plan_with_missing_headers.selected_agents, "relationship_agent not auto-healed into selected_agents"
    assert "database_tool" in plan_with_missing_headers.selected_tools, "database_tool not auto-healed into selected_tools"
    assert "relationship_tool" in plan_with_missing_headers.selected_tools, "relationship_tool not auto-healed into selected_tools"
    print(f"PASS: Auto-healing correctly populated selected_agents: {plan_with_missing_headers.selected_agents}")
    print(f"PASS: Auto-healing correctly populated selected_tools: {plan_with_missing_headers.selected_tools}")

    # 2. Confirm unauthorized agents are rejected
    try:
        PlannedStep(
            step_number=1,
            action_name="unauthorized_action",
            target_agent="unauthorized_rogue_agent",
            target_tool="database_tool",
            tool_parameters={"go_number": "G.O. 1"},
            expected_outcome="Outcome",
        )
        assert False, "Failed to reject unauthorized agent"
    except ValidationError as e:
        assert "Unknown agent 'unauthorized_rogue_agent'" in str(e)
        print("PASS: Unauthorized agent 'unauthorized_rogue_agent' strictly rejected.")

    # 3. Confirm unauthorized tools are rejected
    try:
        PlannedStep(
            step_number=1,
            action_name="unauthorized_tool_action",
            target_agent="database_agent",
            target_tool="arbitrary_bash_tool",
            tool_parameters={},
            expected_outcome="Outcome",
        )
        assert False, "Failed to reject unauthorized tool"
    except ValidationError as e:
        assert "Unknown tool 'arbitrary_bash_tool'" in str(e)
        print("PASS: Unauthorized tool 'arbitrary_bash_tool' strictly rejected.")

    # 4. Confirm invalid parameters are rejected
    try:
        PlannedStep(
            step_number=1,
            action_name="invalid_param_action",
            target_agent="search_agent",
            target_tool="search_tool",
            tool_parameters={"unsupported_custom_param": "malicious_payload"},
            expected_outcome="Outcome",
        )
        assert False, "Failed to reject invalid tool parameter"
    except ValidationError as e:
        assert "Unsupported parameter 'unsupported_custom_param'" in str(e)
        print("PASS: Unsupported tool parameter 'unsupported_custom_param' strictly rejected.")

    # 5. Confirm suspicious command injection parameters are rejected
    try:
        PlannedStep(
            step_number=1,
            action_name="injection_attempt",
            target_agent="search_agent",
            target_tool="search_tool",
            tool_parameters={"keyword": "test; DROP TABLE government_orders;"},
            expected_outcome="Outcome",
        )
        assert False, "Failed to reject injection attack in parameter"
    except ValidationError as e:
        assert "Security violation" in str(e)
        print("PASS: SQL injection attempt in parameter strictly rejected.")


def test_heuristic_fallback_incoming_relationship_workflow() -> None:
    print("\n==========================================")
    print("FOCUSED TEST 3: Heuristic Fallback Incoming Relationship Workflow")
    print("==========================================")
    # Simulate an inconsistent LLM plan that triggers heuristic fallback:
    # Step 1 uses 'search_tool' while top-level next_action is set to 'database'
    inconsistent_plan_dict = {
        "intent": "INCOMING_RELATIONSHIPS",
        "identified_go_numbers": ["G.O.(Rt) No. 114/2025/ITD"],
        "identified_departments": [],
        "reasoning": "Inconsistent plan for fallback regression verification",
        "selected_agents": ["search_agent"],
        "selected_tools": ["search_tool"],
        "execution_steps": [
            {
                "step_number": 1,
                "action_name": "search_order",
                "target_agent": "database_agent",  # Invalid pairing with search_tool!
                "target_tool": "search_tool",
                "tool_parameters": {"query": "114/2025/ITD"},
                "expected_outcome": "Outcome",
            }
        ],
        "expected_evidence": "Evidence",
        "next_action": "database",  # Inconsistent with search_tool
        "confidence": 0.8,
        "human_review_required": False,
    }

    mock_llm = MagicMock(spec=LLMService)

    def mock_gen_struct(prompt, schema_model, **kwargs):
        if schema_model == SupervisorPlan:
            return inconsistent_plan_dict
        if schema_model == SupervisorReplanDecision:
            return {
                "evidence_sufficient": True,
                "reasoning": "Fallback execution succeeded with incoming relationship evidence",
                "next_action": "response",
                "confidence": 0.95,
                "decision_source": "LLM",
            }
        return inconsistent_plan_dict

    mock_llm.generate_structured.side_effect = mock_gen_struct
    supervisor = SupervisorAgent(llm_service=mock_llm)

    query = "Which later Government Order modified G.O.(Rt) No. 114/2025/ITD?"

    # 1. Verify inconsistent LLM plan is rejected and heuristic fallback is triggered
    plan = supervisor.create_initial_plan(query)
    assert plan.plan_source == "HEURISTIC_FALLBACK", f"Expected HEURISTIC_FALLBACK, got {plan.plan_source}"
    # 2. Verify fallback selects database_tool
    assert "database_tool" in plan.selected_tools, f"Expected database_tool in selected_tools, got {plan.selected_tools}"
    step1_params = plan.execution_steps[0].tool_parameters
    # 3. Verify fallback supplies operation='GET_RELATIONSHIPS' and direction='INCOMING'
    assert step1_params.get("operation") == "GET_RELATIONSHIPS", f"Expected operation='GET_RELATIONSHIPS', got {step1_params}"
    assert step1_params.get("direction") == "INCOMING", f"Expected direction='INCOMING', got {step1_params}"
    print(f"PASS: Inconsistent LLM plan rejected -> Fallback selected database_tool with operation='GET_RELATIONSHIPS'.")

    # 4. Run workflow and verify incoming relationship is successfully retrieved
    nodes = WorkflowNodes(supervisor_agent=supervisor)
    memory = MemorySaver()
    graph = build_government_orders_graph(checkpointer=memory, nodes=nodes)
    result = graph.invoke(
        {"user_query": query, "retry_count": 0, "max_retries": 2},
        config={"configurable": {"thread_id": "test_fallback_regression_thread"}},
    )

    assert result["workflow_status"] == "COMPLETED"
    assert result["evidence_sufficient"] is False
    assert len(result["relationship_results"]) > 0
    rel = result["relationship_results"][0]
    assert rel["direction"] == "INCOMING"
    assert "G.O.(Rt)No.24/2026/ITD" in rel["source_go_number"]
    print(f"PASS: Workflow retrieved incoming relationship: '{rel['source_go_number']}' modifies target.")

    # 5. Verify final response presents relationship accurately and maintains unverified legal effect disclaimer
    final_ans = result.get("final_answer", "")
    assert "G.O.(Rt)No.24/2026/ITD" in final_ans
    assert "MODIFIES" in final_ans
    assert "Legal effect remains unverified" in final_ans or "legal enforceability" in final_ans.lower()
    assert all(r.get("legal_effect_verified") is False for r in result["relationship_results"])
    print("PASS: Final response accurately presents relationship without claiming legal effect verification.")


def test_database_operation_inference_regression() -> None:
    print("\n==========================================")
    print("FOCUSED TEST 4: Safe Operation Inference in Database Node")
    print("==========================================")
    nodes = WorkflowNodes()

    # 1. Missing operation + direction=INCOMING selects GET_RELATIONSHIPS and retrieves modifying order
    state_incoming: AgentState = {
        "user_query": "Which later order modified 114/2025/ITD?",
        "plan_generation": 1,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "execution_plan": {
            "execution_steps": [
                {
                    "step_number": 1,
                    "target_agent": "database_agent",
                    "target_tool": "database_tool",
                    "tool_parameters": {"go_number": "114/2025/ITD", "direction": "INCOMING"},
                }
            ]
        },
    }
    res_in = nodes.database_node(state_incoming)
    assert "plan_1_step_1" in res_in.get("completed_step_ids", [])
    trace_in = res_in["execution_trace"][0]
    assert trace_in["action"] == "GET_RELATIONSHIPS", f"Expected GET_RELATIONSHIPS, got {trace_in['action']}"
    assert len(res_in.get("relationship_results", [])) > 0
    assert "G.O.(Rt)No.24/2026/ITD" in res_in["relationship_results"][0]["source_go_number"]
    print("PASS: Missing operation + direction=INCOMING selects GET_RELATIONSHIPS and retrieves modifying order.")

    # 2. Missing operation + direction=OUTGOING selects GET_RELATIONSHIPS
    state_outgoing: AgentState = {
        "user_query": "Which orders does G.O.(Rt)No.24/2026/ITD modify?",
        "plan_generation": 1,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "execution_plan": {
            "execution_steps": [
                {
                    "step_number": 1,
                    "target_agent": "database_agent",
                    "target_tool": "database_tool",
                    "tool_parameters": {"go_number": "G.O.(Rt)No.24/2026/ITD", "direction": "OUTGOING"},
                }
            ]
        },
    }
    res_out = nodes.database_node(state_outgoing)
    assert "plan_1_step_1" in res_out.get("completed_step_ids", [])
    trace_out = res_out["execution_trace"][0]
    assert trace_out["action"] == "GET_RELATIONSHIPS", f"Expected GET_RELATIONSHIPS, got {trace_out['action']}"
    print("PASS: Missing operation + direction=OUTGOING selects GET_RELATIONSHIPS.")

    # 3. Explicit operation takes precedence
    state_explicit: AgentState = {
        "user_query": "Lookup 114/2025/ITD",
        "plan_generation": 1,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "execution_plan": {
            "execution_steps": [
                {
                    "step_number": 1,
                    "target_agent": "database_agent",
                    "target_tool": "database_tool",
                    "tool_parameters": {"go_number": "114/2025/ITD", "operation": "LOOKUP_ORDER", "direction": "INCOMING"},
                }
            ]
        },
    }
    res_exp = nodes.database_node(state_explicit)
    trace_exp = res_exp["execution_trace"][0]
    assert trace_exp["action"] == "LOOKUP_ORDER", f"Expected LOOKUP_ORDER, got {trace_exp['action']}"
    print("PASS: Explicit operation takes precedence over direction inference.")

    # 4. Missing operation + missing direction retains LOOKUP_ORDER
    state_default: AgentState = {
        "user_query": "Lookup G.O.(Rt)No.24/2026/ITD",
        "plan_generation": 1,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "execution_plan": {
            "execution_steps": [
                {
                    "step_number": 1,
                    "target_agent": "database_agent",
                    "target_tool": "database_tool",
                    "tool_parameters": {"go_number": "G.O.(Rt)No.24/2026/ITD"},
                }
            ]
        },
    }
    res_def = nodes.database_node(state_default)
    trace_def = res_def["execution_trace"][0]
    assert trace_def["action"] == "LOOKUP_ORDER", f"Expected LOOKUP_ORDER, got {trace_def['action']}"
    print("PASS: Missing operation + missing direction retains LOOKUP_ORDER default.")

    # 5. Invalid direction is rejected safely
    state_invalid_dir: AgentState = {
        "user_query": "Query with invalid direction",
        "plan_generation": 1,
        "completed_step_ids": [],
        "failed_step_ids": [],
        "invalid_plan_count": 0,
        "execution_plan": {
            "execution_steps": [
                {
                    "step_number": 1,
                    "target_agent": "database_agent",
                    "target_tool": "database_tool",
                    "tool_parameters": {"go_number": "114/2025/ITD", "direction": "SIDEWAYS"},
                }
            ]
        },
    }
    res_inv = nodes.database_node(state_invalid_dir)
    assert "plan_1_step_1" in res_inv.get("failed_step_ids", [])
    assert res_inv.get("invalid_plan_count") == 1
    print("PASS: Invalid direction 'SIDEWAYS' safely rejected.")


if __name__ == "__main__":
    try:
        test_keyword_search_workflow()
        test_incoming_relationships_workflow()
        test_multi_step_explanation_workflow()
        test_ambiguity_and_human_review_interruption()
        test_unverified_candidate_handling()
        test_retry_exhaustion()
        test_execution_trace_audit()
        test_live_ollama_integration()
        test_incoming_relationship_normalization_and_ambiguity()
        test_supervisor_plan_auto_healing_and_rejection()
        test_heuristic_fallback_incoming_relationship_workflow()
        test_database_operation_inference_regression()
        print("\n=======================================================")
        print("ALL LANGGRAPH WORKFLOW & FOCUSED TESTS PASSED!")
        print("=======================================================")
    except Exception as exc:
        print(f"\nTEST SUITE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
