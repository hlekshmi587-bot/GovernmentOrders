"""
Comprehensive Unit and Integration Tests for Supervisor Agent & Planning Schema:
Phase 1, Step 3.1

Tests cover:
1. Intent Differentiation:
   - KEYWORD_SEARCH
   - GO_LOOKUP
   - INCOMING_RELATIONSHIPS
   - OUTGOING_RELATIONSHIPS
   - EXPLAIN_ORDER
   - AMBIGUOUS_CLARIFICATION
2. Multi-Step Planning:
   - Sequential step numbering (1, 2, ...)
   - Step-to-action coherence (next_action matches step 1)
   - Multiple agents/tools coordination
3. Tool Parameter Validation:
   - Rejection of unauthorized tools
   - Rejection of unsupported parameters
   - Rejection of suspicious command injection keywords (cmd, exec, sql)
4. Evidence Evaluation:
   - Empty search results not treated as sufficient
   - Direct lookup NOT_FOUND handled with meaningful alternative
   - UNVERIFIED_CANDIDATE routed for verification, not treated as legally verified
5. Replanning & Bounded Retries:
   - Failed tool calls trigger alternative actions, not repeated failures
   - Bounded retries exhaust safely into human review
6. Fallback Transparency:
   - Simulated LLM failure explicitly marks plan_source='HEURISTIC_FALLBACK'
7. Execution Trace:
   - Captures decision_source, tool, input, output summary, timestamp
8. Live Ollama Integration Test:
   - Strict assertions on Query B ('INCOMING_RELATIONSHIPS', 'database_tool')
"""

import sys
from unittest.mock import MagicMock
from pydantic import ValidationError

from app.agents.specialized.supervisor_agent import (
    SupervisorAgent,
    SupervisorPlan,
    PlannedStep,
    QueryIntent,
    SupervisorReplanDecision,
)
from app.agents.state import AgentState
from app.agents.llm.llm_service import LLMService
from app.agents.llm.ollama_client import OllamaClient


def test_intent_differentiation() -> None:
    print("\n==========================================")
    print("SUITE 1: Intent Differentiation Tests")
    print("==========================================")
    mock_llm = MagicMock(spec=LLMService)
    supervisor = SupervisorAgent(llm_service=mock_llm)

    intents_to_test = [
        (
            "Find Government Orders from P&ARD",
            QueryIntent.KEYWORD_SEARCH,
            "search",
            ["search_tool"],
            ["search_agent", "response_agent"],
        ),
        (
            "Show details of G.O.(P) No.4/2026/P&ARD",
            QueryIntent.GO_LOOKUP,
            "database",
            ["database_tool"],
            ["database_agent", "response_agent"],
        ),
        (
            "Which later Government Order modified G.O.(Rt) No. 114/2025/ITD?",
            QueryIntent.INCOMING_RELATIONSHIPS,
            "database",
            ["database_tool"],
            ["database_agent", "verification_agent", "response_agent"],
        ),
        (
            "Which earlier orders does G.O.(Rt)No.24/2026/ITD modify?",
            QueryIntent.OUTGOING_RELATIONSHIPS,
            "database",
            ["database_tool"],
            ["database_agent", "verification_agent", "response_agent"],
        ),
        (
            "Explain the changes introduced by G.O.(Rt) No. 24/2026/ITD",
            QueryIntent.EXPLAIN_ORDER,
            "database",
            ["database_tool", "relationship_tool"],
            ["database_agent", "relationship_agent", "verification_agent", "response_agent"],
        ),
        (
            "Check order",
            QueryIntent.AMBIGUOUS_CLARIFICATION,
            "human_review",
            ["none"],
            ["human_review"],
        ),
    ]

    for query, expected_intent, expected_action, tools, agents in intents_to_test:
        steps = []
        if tools != ["none"]:
            for idx, t in enumerate(tools, start=1):
                steps.append(
                    PlannedStep(
                        step_number=idx,
                        action_name=f"step_{idx}",
                        target_agent=agents[0] if idx == 1 else agents[1],
                        target_tool=t,
                        tool_parameters={"query": query} if t != "none" else {},
                        expected_outcome="expected_outcome",
                    )
                )

        mock_llm.generate_structured.return_value = {
            "intent": expected_intent.value,
            "identified_go_numbers": ["G.O. 114/2025/ITD"] if "114" in query else [],
            "identified_departments": ["P&ARD"] if "P&ARD" in query else [],
            "reasoning": f"Reasoning for {expected_intent.value}",
            "selected_agents": agents,
            "selected_tools": tools,
            "execution_steps": [s.model_dump() for s in steps],
            "expected_evidence": "Expected evidence",
            "next_action": expected_action,
            "confidence": 0.9,
            "human_review_required": (expected_action == "human_review"),
            "plan_source": "LLM",
        }

        plan = supervisor.create_initial_plan(query)
        assert plan.intent == expected_intent, f"Expected intent {expected_intent}, got {plan.intent}"
        assert plan.next_action == expected_action, f"Expected action {expected_action}, got {plan.next_action}"
        print(f"PASS: Intent '{expected_intent.value}' correctly planned (next_action={plan.next_action}).")


def test_multi_step_planning() -> None:
    print("\n==========================================")
    print("SUITE 2: Multi-Step Planning Tests")
    print("==========================================")

    # 1. Valid multi-step plan
    step1 = PlannedStep(
        step_number=1,
        action_name="retrieve_full_text",
        target_agent="database_agent",
        target_tool="database_tool",
        tool_parameters={"go_number": "G.O.(Rt)No.24/2026/ITD", "include_full_text": True},
        expected_outcome="Full document text",
    )
    step2 = PlannedStep(
        step_number=2,
        action_name="extract_relationships",
        target_agent="relationship_agent",
        target_tool="relationship_tool",
        tool_parameters={"go_number": "G.O.(Rt)No.24/2026/ITD"},
        expected_outcome="Extracted relationship signals and evidence",
    )

    valid_multi_plan = SupervisorPlan(
        intent=QueryIntent.EXPLAIN_ORDER,
        identified_go_numbers=["G.O.(Rt)No.24/2026/ITD"],
        identified_departments=["ITD"],
        reasoning="Multi-step explanation plan: fetch text then extract relationships.",
        selected_agents=["database_agent", "relationship_agent", "verification_agent", "response_agent"],
        selected_tools=["database_tool", "relationship_tool"],
        execution_steps=[step1, step2],
        expected_evidence="Full text and parsed relationship evidence",
        next_action="database",  # Matches step 1 tool (database_tool)
        confidence=0.95,
        human_review_required=False,
        plan_source="LLM",
    )
    assert len(valid_multi_plan.execution_steps) == 2
    assert valid_multi_plan.execution_steps[0].step_number == 1
    assert valid_multi_plan.execution_steps[1].step_number == 2
    assert valid_multi_plan.next_action == "database"
    print("PASS: Valid 2-step plan verified with sequential steps and matching next_action.")

    # 2. Rejection of non-sequential step numbering (e.g. 1, 3)
    step_bad = PlannedStep(
        step_number=3,  # Invalid: skips step 2
        action_name="extract_relationships",
        target_agent="relationship_agent",
        target_tool="relationship_tool",
        tool_parameters={"go_number": "G.O. 1"},
        expected_outcome="output",
    )
    try:
        SupervisorPlan(
            intent=QueryIntent.EXPLAIN_ORDER,
            reasoning="test",
            selected_agents=["database_agent", "relationship_agent"],
            selected_tools=["database_tool", "relationship_tool"],
            execution_steps=[step1, step_bad],
            expected_evidence="evidence",
            next_action="database",
            confidence=0.9,
            plan_source="LLM",
        )
        assert False, "Should have rejected non-sequential steps (1, 3)"
    except ValueError as e:
        assert "strictly sequential" in str(e)
        print("PASS: Non-sequential step numbering (1, 3) rejected by validator.")

    # 3. Alignment of conflicting next_action with valid step 1 pairing
    plan_aligned = SupervisorPlan(
        intent=QueryIntent.EXPLAIN_ORDER,
        reasoning="test",
        selected_agents=["database_agent", "relationship_agent"],
        selected_tools=["database_tool", "relationship_tool"],
        execution_steps=[step1, step2],
        expected_evidence="evidence",
        next_action="search",  # Conflicted with step 1 database_tool!
        confidence=0.9,
        plan_source="LLM",
    )
    assert plan_aligned.next_action == "database", f"Expected next_action to align to 'database', got {plan_aligned.next_action}"
    print("PASS: Conflicting next_action ('search') safely aligned to step 1 valid pairing ('database').")

    # 4. Rejection of invalid agent/tool pairing in step
    step_invalid_pair = PlannedStep(
        step_number=1,
        action_name="invalid_pair",
        target_agent="database_agent",
        target_tool="search_tool",  # Invalid pairing!
        tool_parameters={"query": "test"},
        expected_outcome="outcome",
    )
    try:
        SupervisorPlan(
            intent=QueryIntent.GO_LOOKUP,
            reasoning="test",
            selected_agents=["database_agent"],
            selected_tools=["search_tool"],
            execution_steps=[step_invalid_pair],
            expected_evidence="evidence",
            next_action="database",
            confidence=0.9,
            plan_source="LLM",
        )
        assert False, "Should have rejected invalid agent/tool pairing"
    except ValueError as e:
        assert "Invalid agent/tool pairing" in str(e)
        print("PASS: Invalid agent/tool pairing ('database_agent' with 'search_tool') strictly rejected.")


def test_tool_parameter_validation() -> None:
    print("\n==========================================")
    print("SUITE 3: Tool Parameter Validation Tests")
    print("==========================================")

    # 1. Valid parameters for search_tool
    valid_search_step = PlannedStep(
        step_number=1,
        action_name="search",
        target_agent="search_agent",
        target_tool="search_tool",
        tool_parameters={"keyword": "Finance", "limit": 5},
        expected_outcome="Orders",
    )
    assert valid_search_step.tool_parameters["keyword"] == "Finance"
    print("PASS: Supported parameters for search_tool accepted.")

    # 2. Rejection of unsupported parameters for search_tool (e.g. 'arbitrary_flag')
    try:
        PlannedStep(
            step_number=1,
            action_name="search",
            target_agent="search_agent",
            target_tool="search_tool",
            tool_parameters={"keyword": "Finance", "arbitrary_unsupported_flag": 123},
            expected_outcome="Orders",
        )
        assert False, "Should have rejected unsupported parameter for search_tool"
    except ValueError as e:
        assert "Unsupported parameter" in str(e)
        print("PASS: Unsupported parameter for search_tool correctly rejected.")

    # 3. Rejection of suspicious command injection keywords (e.g. 'cmd', 'exec', 'sql')
    for bad_param in ["cmd", "exec", "eval", "sql", "code", "script"]:
        try:
            PlannedStep(
                step_number=1,
                action_name="attack",
                target_agent="database_agent",
                target_tool="database_tool",
                tool_parameters={bad_param: "rm -rf /"},
                expected_outcome="Orders",
            )
            assert False, f"Should have rejected command parameter '{bad_param}'"
        except ValueError as e:
            assert "Security violation" in str(e)
    print("PASS: Command injection parameter names rejected by security validator.")


def test_evidence_evaluation_and_replanning() -> None:
    print("\n==========================================")
    print("SUITE 4: Evidence Evaluation & Replanning Tests")
    print("==========================================")
    supervisor = SupervisorAgent()

    # 4.1 Empty search results NOT treated as sufficient evidence
    state_empty_search: AgentState = {
        "user_query": "Find order XYZ_NOT_FOUND",
        "retry_count": 0,
        "max_retries": 2,
        "search_results": [{"status": "NO_MATCH", "returned_count": 0, "results": []}],
        "document_analysis": [],
        "relationship_results": [],
        "errors": [],
    }
    decision_empty = supervisor.evaluate_and_replan(state_empty_search)
    assert decision_empty.evidence_sufficient is False
    assert decision_empty.next_action == "human_review"
    assert decision_empty.human_review_required is True
    print("PASS: Empty search results (0 records) correctly evaluated as insufficient.")

    # 4.2 Database NOT_FOUND replans to search_tool (actionable alternative, not repeating failed call)
    state_db_failed: AgentState = {
        "user_query": "G.O.(P) No.999/2026/P&ARD",
        "retry_count": 0,
        "max_retries": 2,
        "document_analysis": [{"status": "NOT_FOUND", "found": False}],
        "relationship_results": [],
        "search_results": [],
        "errors": [],
    }
    decision_db_retry = supervisor.evaluate_and_replan(state_db_failed)
    assert decision_db_retry.evidence_sufficient is False
    assert decision_db_retry.next_action == "search"
    assert decision_db_retry.suggested_tool == "search_tool"
    assert "Replanning to search_tool" in decision_db_retry.reasoning
    print("PASS: Failed DB lookup triggers meaningful alternative (search_tool), not repeating failed DB query.")

    # 4.3 UNVERIFIED_CANDIDATE routed for verification, NOT treated as confirmed evidence
    state_unverified_cand: AgentState = {
        "user_query": "Did G.O. 24/2026 modify earlier orders?",
        "retry_count": 0,
        "max_retries": 2,
        "relationship_results": [
            {
                "relationship_type": "MODIFIES",
                "matched_phrase": "hereby modified",
                "verification_status": "UNVERIFIED_CANDIDATE",
                "legal_effect_verified": False,
                "evidence_text": "Order is hereby modified",
            }
        ],
        "document_analysis": [],
        "search_results": [],
        "errors": [],
    }
    decision_unverified = supervisor.evaluate_and_replan(state_unverified_cand)
    assert decision_unverified.evidence_sufficient is False
    assert decision_unverified.next_action == "verification"
    assert "UNVERIFIED_CANDIDATE" in decision_unverified.reasoning
    print("PASS: UNVERIFIED_CANDIDATE relationship routed for verification, not treated as confirmed evidence.")

    # 4.4 Bounded retry exhaustion routes to human review
    state_exhausted: AgentState = {
        "user_query": "Find order XYZ",
        "retry_count": 2,
        "max_retries": 2,
        "search_results": [],
        "document_analysis": [],
        "relationship_results": [],
        "errors": ["Failed multiple attempts"],
    }
    decision_exhausted = supervisor.evaluate_and_replan(state_exhausted)
    assert decision_exhausted.next_action == "human_review"
    assert decision_exhausted.human_review_required is True
    assert "Reached maximum retry limit" in decision_exhausted.reasoning
    print("PASS: Retry exhaustion terminates bounded loop and routes to human review.")


def test_fallback_transparency_and_traces() -> None:
    print("\n==========================================")
    print("SUITE 5: Fallback Transparency & Trace Tests")
    print("==========================================")

    # 1. Fallback transparency: simulate LLM failure
    mock_failing_llm = MagicMock(spec=LLMService)
    mock_failing_llm.generate_structured.side_effect = RuntimeError("Ollama connection timed out")

    supervisor_fallback = SupervisorAgent(llm_service=mock_failing_llm)
    plan_fallback = supervisor_fallback.create_initial_plan("Which later Government Order modified G.O. 114?")

    assert plan_fallback.plan_source == "HEURISTIC_FALLBACK"
    assert "Heuristic fallback" in plan_fallback.reasoning
    assert plan_fallback.intent == QueryIntent.INCOMING_RELATIONSHIPS
    print(f"PASS: Fallback explicitly marked plan_source='{plan_fallback.plan_source}' (not reported as LLM).")

    # 2. Execution trace captures decision_source, tool, and timestamp
    trace_llm = SupervisorAgent.create_trace_item(
        agent="supervisor",
        action="initial_plan",
        reasoning="Classified as INCOMING_RELATIONSHIPS",
        tool_called="database_tool",
        tool_input={"go_number": "G.O. 114"},
        tool_output_summary="Planned incoming relationship lookup",
        decision="database",
        decision_source="LLM",
    )
    assert trace_llm["decision_source"] == "LLM"
    assert trace_llm["agent"] == "supervisor"
    assert "timestamp" in trace_llm

    trace_fb = SupervisorAgent.create_trace_item(
        agent="supervisor",
        action="fallback_plan",
        reasoning="Heuristic fallback plan generated",
        tool_called="search_tool",
        decision="search",
        decision_source="HEURISTIC_FALLBACK",
    )
    assert trace_fb["decision_source"] == "HEURISTIC_FALLBACK"
    print("PASS: Execution trace records decision_source ('LLM' vs 'HEURISTIC_FALLBACK') cleanly.")


def test_live_ollama_integration_strict() -> None:
    print("\n==========================================")
    print("SUITE 6: Live Ollama Integration Test (Strict)")
    print("==========================================")
    client = OllamaClient()
    if not client.is_server_running():
        print("SKIP: Ollama server not reachable on localhost:11434. Skipping live inference test.")
        return

    llm_service = LLMService(client=client, model="llama3.2:3b")
    supervisor = SupervisorAgent(llm_service=llm_service)

    live_query = "Which later Government Order modified G.O.(Rt) No. 114/2025/ITD?"
    print(f"Live Query submitted to llama3.2:3b: '{live_query}'")

    plan = supervisor.create_initial_plan(live_query)

    print(f"Model plan_source: {plan.plan_source}")
    print(f"Model intent: {plan.intent.value}")
    print(f"Model next_action: {plan.next_action}")
    print(f"Model selected_tools: {plan.selected_tools}")
    print(f"Model reasoning: {plan.reasoning}")
    print(f"Model confidence: {plan.confidence}")

    # Strict assertions per specification: do not weaken!
    assert plan.plan_source == "LLM", f"Expected live plan_source to be 'LLM', got {plan.plan_source}"
    assert plan.intent == QueryIntent.INCOMING_RELATIONSHIPS, f"Expected INCOMING_RELATIONSHIPS, got {plan.intent}"
    assert plan.next_action == "database", f"Expected next_action 'database', got {plan.next_action}"
    assert "database_tool" in plan.selected_tools, f"Expected database_tool in selected_tools, got {plan.selected_tools}"
    assert 0.0 <= plan.confidence <= 1.0

    print("PASS: Live Ollama model strictly satisfied all architectural assertions!")


if __name__ == "__main__":
    try:
        test_intent_differentiation()
        test_multi_step_planning()
        test_tool_parameter_validation()
        test_evidence_evaluation_and_replanning()
        test_fallback_transparency_and_traces()
        test_live_ollama_integration_strict()
        print("\n=======================================================")
        print("ALL SUPERVISOR AGENT STRENGTHENED TESTS PASSED!")
        print("=======================================================")
    except Exception as exc:
        print(f"\nTEST SUITE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
