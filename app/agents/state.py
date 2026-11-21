"""
Agent State Definition for Kerala Government Orders Multi-Agent System.

This module defines the shared state structure used across all agents in the
LangGraph workflow. Every agent reads from and writes to this state.
"""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict


def merge_list(existing: List[Any], new_items: List[Any]) -> List[Any]:
    """Helper reducer to append new items to an existing list in LangGraph state."""
    if not existing:
        return list(new_items)
    if not new_items:
        return list(existing)
    return list(existing) + [item for item in new_items if item not in existing]


def append_trace(existing: List[Dict[str, Any]], new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper reducer to append execution trace events."""
    if not existing:
        return list(new_records)
    if not new_records:
        return list(existing)
    return list(existing) + list(new_records)


class ExecutionTraceItem(TypedDict, total=False):
    """
    A single event in the execution trace for complete auditability.
    """
    timestamp: str
    agent: str
    action: str
    reasoning: Optional[str]
    tool_called: Optional[str]
    tool_input: Optional[Any]
    tool_output_summary: Optional[str]
    decision: Optional[str]
    decision_source: Optional[str]  # "LLM" or "HEURISTIC_FALLBACK"


class AgentState(TypedDict, total=False):
    """
    The central shared state for the multi-agent system.
    Passed between nodes in the LangGraph workflow.
    """

    # 1. User Input & Natural Language Understanding
    user_query: str
    intent: Optional[str]
    identified_go_numbers: List[str]
    search_parameters: Optional[Dict[str, Any]]  # Optional caller-provided search constraints (e.g. limit, department, retrieval_mode)
    
    # 2. Iterative Discovery (New GO references found during investigation)
    discovered_go_numbers: Annotated[List[str], merge_list]

    # 3. Dynamic Planning & Coordination (Driven by Supervisor Agent)
    execution_plan: Optional[Dict[str, Any]]
    selected_agents: List[str]
    next_action: Optional[str]

    # 4. Agent Outputs & Gathered Evidence
    search_results: Annotated[List[Dict[str, Any]], merge_list]
    document_analysis: Annotated[List[Dict[str, Any]], merge_list]
    relationship_results: Annotated[List[Dict[str, Any]], merge_list]

    # 5. Verification & Evidence Sufficiency Evaluation
    verification_results: Optional[Dict[str, Any]]
    evidence_sufficient: Optional[bool]
    confidence: Optional[str]  # "HIGH", "MEDIUM", "LOW"

    # 6. Retry & Loop Control (Guards against infinite loops)
    retry_count: int
    max_retries: int

    # 7. Human-in-the-Loop Review & Resume Support
    human_review_required: bool
    human_review_status: str  # "NONE", "PENDING", "APPROVED", "REJECTED", "CORRECTED"
    human_review_notes: Optional[str]
    human_feedback: Optional[str]  # Human corrections provided to resume the agent

    # 8. Final Synthesis
    final_answer: Optional[str]

    # 9. Observability, Traceability, and Error Handling
    workflow_status: str  # "INITIALIZED", "PLANNING", "RUNNING", "PAUSED_FOR_HUMAN", "COMPLETED", "FAILED"
    execution_trace: Annotated[List[Dict[str, Any]], append_trace]
    errors: Annotated[List[str], merge_list]

    # 10. Phase 2 Step 2.5 Dynamic Plan-Generation & Routing Tracking
    plan_generation: int                                  # Starts at 1; increments on evidence-driven replan (max 6)
    invalid_plan_count: int                               # Starts at 0; increments on invalid plan/params (max 2)
    step_retry_count: int                                 # Starts at 0; tracks retries for current step (max 2 retries = 3 attempts total)
    completed_step_ids: Annotated[List[str], merge_list]  # e.g., ["plan_1_step_1", "plan_2_step_1"]
    failed_step_ids: Annotated[List[str], merge_list]     # e.g., ["plan_1_step_2"]
    abandoned_step_ids: Annotated[List[str], merge_list]  # Unexecuted steps retired upon replan
    response_ready: bool                                  # True when final answer is ready for synthesis
    verification_status: Optional[str]                    # Exact status from VerificationAgent or None
