"""
LangGraph Workflow Builder for Kerala Government Orders Multi-Agent Architecture.

Orchestrates the Supervisor Agent, specialized agent execution nodes:
  - SearchAgent (keyword & semantic vector retrieval with RRF fusion)
  - DatabaseAgent (exact/partial PostgreSQL order lookup and references)
  - RelationshipAgent (four investigation modes and candidate isolation)
  - VerificationAgent (fact-checking, candidate corroboration, legal-effect invariant)
  - ResponseAgent (evidence-grounded final synthesis with strict sectioning)
  - Human-in-the-Loop review (LangGraph interrupt/resume checkpointing)
"""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.agents.state import AgentState
from app.agents.specialized.supervisor_agent import (
    SupervisorAgent,
    SupervisorPlan,
    PlannedStep,
    QueryIntent,
)
from app.agents.specialized.search_agent import (
    SearchAgent,
    SearchAgentRequest,
    RetrievalMode,
)
from app.agents.specialized.database_agent import (
    DatabaseAgent,
    DatabaseAgentRequest,
    DatabaseOperation,
    RelationshipDirection,
    AgentResponseStatus,
)
from app.agents.specialized.relationship_agent import (
    RelationshipAgent,
    RelationshipAgentRequest,
    RelationshipInvestigationMode,
    ExtractionConfidence,
    RelationshipAgentStatus,
)
from app.agents.specialized.verification_agent import (
    VerificationAgent,
    VerificationAgentRequest,
    VerificationClaim,
    RelationshipCandidateInput,
    SearchEvidenceInput,
    SupportingChunkInput,
    VerificationStatus,
)
from app.agents.specialized.response_agent import (
    ResponseAgent,
    ResponseAgentInput,
)

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """
    Container for the executable graph nodes.
    Each node receives AgentState and returns updated fields.
    """

    def __init__(
        self,
        supervisor_agent: Optional[SupervisorAgent] = None,
        search_agent: Optional[SearchAgent] = None,
        database_agent: Optional[DatabaseAgent] = None,
        relationship_agent: Optional[RelationshipAgent] = None,
        verification_agent: Optional[VerificationAgent] = None,
        response_agent: Optional[ResponseAgent] = None,
    ) -> None:
        self.supervisor = supervisor_agent or SupervisorAgent()
        self.search_agent = search_agent or SearchAgent()
        self.db_agent = database_agent or DatabaseAgent()
        self.rel_agent = relationship_agent or RelationshipAgent()
        self.verification_agent = verification_agent or VerificationAgent()
        self.response_agent = response_agent or ResponseAgent()

    def supervisor_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Supervisor decision node: creates initial plan, replans, or manages step progression.
        """
        query = state.get("user_query", "")
        plan = state.get("execution_plan")
        human_feedback = state.get("human_feedback")
        plan_gen = state.get("plan_generation", 1)
        completed_step_ids = state.get("completed_step_ids", [])
        failed_step_ids = state.get("failed_step_ids", [])
        abandoned_step_ids = state.get("abandoned_step_ids", [])
        invalid_plan_count = state.get("invalid_plan_count", 0)
        step_retry_count = state.get("step_retry_count", 0)
        global_retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        logger.info("SupervisorNode: Entering supervisor decision (plan_generation=%d, steps_completed=%d)", plan_gen, len(completed_step_ids))

        # 1. Initial Planning or Replanning with Human Feedback
        if not plan or human_feedback:
            effective_query = f"{query} [Human Guidance: {human_feedback}]" if human_feedback else query
            logger.info("SupervisorNode: Generating initial plan via LLM for query: '%s'", effective_query[:120])
            try:
                new_plan = self.supervisor.create_initial_plan(effective_query)
            except Exception as exc:
                logger.warning("Supervisor initial planning failed: %s", exc)
                # Fallback to heuristic
                new_plan = self.supervisor.create_initial_plan(effective_query)
            
            trace_item = self.supervisor.create_trace_item(
                agent="supervisor",
                action="initial_planning",
                reasoning=new_plan.reasoning,
                decision=new_plan.next_action,
                decision_source=new_plan.plan_source,
            )

            logger.info(
                "SupervisorNode: Initial plan generated -> intent=%s, next_action=%s, plan_source=%s, steps=%d",
                new_plan.intent.value, new_plan.next_action, new_plan.plan_source, len(new_plan.execution_steps)
            )

            return {
                "execution_plan": new_plan.model_dump(),
                "intent": new_plan.intent.value,
                "identified_go_numbers": new_plan.identified_go_numbers,
                "selected_agents": new_plan.selected_agents,
                "next_action": new_plan.next_action,
                "confidence": None,
                "human_review_required": new_plan.human_review_required,
                "human_feedback": None,  # Consumed
                "workflow_status": "PLANNING",
                "execution_trace": [trace_item],
            }

        # 2. Check for Ambiguity / Human Review Required in state
        if state.get("human_review_required"):
            logger.info("SupervisorNode: Routing to human_review_node (human_review_required=True)")
            return {
                "next_action": "human_review",
                "workflow_status": "PAUSED_FOR_HUMAN",
            }

        # 3. Step Progression: Find Next Pending Step in Current Generation
        steps = plan.get("execution_steps", [])
        pending_step = None
        for step in steps:
            s_num = step.get("step_number", 1)
            step_id = f"plan_{plan_gen}_step_{s_num}"
            if step_id not in completed_step_ids and step_id not in failed_step_ids:
                pending_step = step
                break

        # If a pending step exists in the current plan, route to it
        if pending_step:
            target_agent = pending_step.get("target_agent", "")
            action_map = {
                "search_agent": "search",
                "database_agent": "database",
                "relationship_agent": "relationship",
                "verification_agent": "verification",
                "response_agent": "response",
                "human_review": "human_review",
            }
            next_act = action_map.get(target_agent, "search")
            logger.info(
                "SupervisorNode: Routing to pending step %d ('%s') -> target_agent=%s, next_action=%s",
                pending_step.get("step_number", 1), pending_step.get("action_name", ""), target_agent, next_act
            )
            return {
                "next_action": next_act,
                "workflow_status": "RUNNING",
            }

        # 4. Evidence-Based Replanning or Response Finalization
        # All steps in current plan have executed. Check if replanning is needed or we can finish.
        if plan_gen >= 6:
            # Reached generation limit (max 6): conclude with qualified response
            logger.info("SupervisorNode: Plan generation limit reached (%d >= 6). Finalizing qualified response.", plan_gen)
            return {
                "next_action": "response",
                "response_ready": True,
                "evidence_sufficient": False,
                "workflow_status": "COMPLETED",
            }

        logger.info("SupervisorNode: Evaluating collected evidence for replanning (plan_generation=%d)", plan_gen)
        replan_decision = self.supervisor.evaluate_and_replan(state)

        trace_item = self.supervisor.create_trace_item(
            agent="supervisor",
            action="replanning_evaluation",
            reasoning=replan_decision.reasoning,
            decision=replan_decision.next_action,
            decision_source=replan_decision.decision_source,
        )

        logger.info(
            "SupervisorNode: Replan evaluation complete -> next_action=%s, evidence_sufficient=%s, confidence=%.2f, source=%s",
            replan_decision.next_action, replan_decision.evidence_sufficient, replan_decision.confidence, replan_decision.decision_source
        )

        # If replan decision says proceed to verification or response
        if replan_decision.next_action in ("response", "verification", "human_review"):
            # For response when verification was not run (e.g. topic discovery), confidence remains None.
            # When verification runs, confidence will be assigned by VerificationAgent.
            confidence_val = state.get("confidence")
            if replan_decision.next_action == "verification":
                confidence_val = None
            elif replan_decision.next_action == "response" and not state.get("verification_status"):
                confidence_val = None

            update_dict: Dict[str, Any] = {
                "next_action": replan_decision.next_action,
                "evidence_sufficient": replan_decision.evidence_sufficient,
                "confidence": confidence_val,
                "human_review_required": replan_decision.human_review_required,
                "human_review_notes": replan_decision.human_review_reason,
                "workflow_status": "PLANNING",
                "execution_trace": [trace_item],
            }
            if replan_decision.next_action == "response":
                update_dict["response_ready"] = True
            return update_dict

        # If replanning requires a new generation of steps
        new_gen = plan_gen + 1
        newly_abandoned = []
        for s in steps:
            s_num = s.get("step_number", 1)
            s_id = f"plan_{plan_gen}_step_{s_num}"
            if s_id not in completed_step_ids and s_id not in abandoned_step_ids:
                newly_abandoned.append(s_id)

        logger.info("SupervisorNode: Triggering replanned plan generation %d via LLM", new_gen)
        # Create new execution plan for new generation
        try:
            new_plan = self.supervisor.create_initial_plan(f"{query} [Refined Plan Generation {new_gen}]")
        except Exception:
            new_plan = self.supervisor.create_initial_plan(query)

        return {
            "plan_generation": new_gen,
            "execution_plan": new_plan.model_dump(),
            "abandoned_step_ids": newly_abandoned,
            "next_action": new_plan.next_action,
            "step_retry_count": 0,
            "workflow_status": "PLANNING",
            "execution_trace": [trace_item],
        }

    def _get_current_step_info(self, state: AgentState) -> tuple[Optional[Dict[str, Any]], str]:
        """Helper to get current PlannedStep dict and unique step_id."""
        plan = state.get("execution_plan") or {}
        plan_gen = state.get("plan_generation", 1)
        completed = state.get("completed_step_ids", [])
        failed = state.get("failed_step_ids", [])
        steps = plan.get("execution_steps", [])

        for s in steps:
            s_num = s.get("step_number", 1)
            step_id = f"plan_{plan_gen}_step_{s_num}"
            if step_id not in completed and step_id not in failed:
                return s, step_id

        # Fallback step_id
        return None, f"plan_{plan_gen}_step_1"

    def search_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes SearchAgent: maps parameters, validates, and runs keyword/semantic search.
        """
        step, step_id = self._get_current_step_info(state)
        params = (step.get("tool_parameters", {}) if step else {})
        user_query = state.get("user_query", "")

        # 1. Parameter Translation & Validation
        raw_query = params.get("query") or params.get("keyword") or user_query
        if not raw_query or not str(raw_query).strip():
            logger.warning("Empty search query rejected.")
            return {
                "failed_step_ids": [step_id],
                "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                "step_retry_count": 0,
                "workflow_status": "RUNNING",
            }

        retrieval_mode = RetrievalMode.HYBRID
        raw_mode = params.get("retrieval_mode") or (state.get("search_parameters") or {}).get("retrieval_mode")
        if raw_mode:
            try:
                retrieval_mode = RetrievalMode(str(raw_mode).strip().upper())
            except ValueError:
                return {
                    "failed_step_ids": [step_id],
                    "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                    "step_retry_count": 0,
                }

        limit_val = params.get("limit")
        if limit_val is None:
            search_params = state.get("search_parameters") or {}
            limit_val = search_params.get("limit", 5)

        if isinstance(limit_val, bool) or not isinstance(limit_val, int) or limit_val < 1 or limit_val > 50:
            return {
                "failed_step_ids": [step_id],
                "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                "step_retry_count": 0,
            }

        dept_filter = params.get("department_filter") or (state.get("search_parameters") or {}).get("department")

        logger.info("SearchNode: Entering search execution (step_id='%s', retrieval_mode=%s, limit=%s)", step_id, retrieval_mode.value, limit_val)

        try:
            req = SearchAgentRequest(
                query=str(raw_query).strip(),
                retrieval_mode=retrieval_mode,
                limit=limit_val,
                department_filter=dept_filter,
                issue_date_start=params.get("issue_date_start"),
                issue_date_end=params.get("issue_date_end"),
                rrf_k=60,
            )
            resp = self.search_agent.execute_search(req)

        except Exception as exc:
            logger.warning("SearchAgent execution failed with technical error: %s", exc)
            curr_step_retry = state.get("step_retry_count", 0)
            if curr_step_retry < 2:
                return {
                    "step_retry_count": curr_step_retry + 1,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "workflow_status": "RUNNING",
                }
            return {
                "failed_step_ids": [step_id],
                "step_retry_count": 0,
                "retry_count": state.get("retry_count", 0) + 1,
                "human_review_required": True,
                "human_review_notes": f"SearchAgent failed after 3 attempts: {exc}",
            }

        # Convert SearchResultItems to serializable dicts
        serializable_results = [r.model_dump() for r in resp.results]

        # Extract and deduplicate genuine GO identifiers from search results
        retrieved_gos: List[str] = []
        seen_retrieved: Set[str] = set()
        for r in resp.results:
            go_val = r.go_number.strip() if hasattr(r, "go_number") else str(r.get("go_number", "")).strip()
            if go_val and go_val.lower() not in seen_retrieved:
                seen_retrieved.add(go_val.lower())
                retrieved_gos.append(go_val)

        trace_item = self.supervisor.create_trace_item(
            agent="search_agent",
            action="execute_search",
            tool_called="search_tool",
            tool_input=req.model_dump(mode="json"),
            tool_output_summary=f"Search status: {resp.status}, returned {resp.returned_count} hits",
            decision="proceed_to_next_step",
            decision_source="DETERMINISTIC_RULES",
        )

        logger.info("SearchNode: Search execution finished -> status=%s, returned_count=%d", resp.status, len(serializable_results))

        # Handle Error status
        if resp.status == "ERROR":
            curr_step_retry = state.get("step_retry_count", 0)
            if curr_step_retry < 2:
                return {
                    "step_retry_count": curr_step_retry + 1,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "execution_trace": [trace_item],
                }
            return {
                "failed_step_ids": [step_id],
                "step_retry_count": 0,
                "human_review_required": True,
                "human_review_notes": f"SearchAgent returned ERROR: {resp.errors}",
                "execution_trace": [trace_item],
            }

        return {
            "search_results": serializable_results,
            "completed_step_ids": [step_id],
            "step_retry_count": 0,
            "workflow_status": "RUNNING",
            "execution_trace": [trace_item],
        }

    def database_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes DatabaseAgent: looks up orders, metadata, references, or relationships.
        """
        step, step_id = self._get_current_step_info(state)
        params = (step.get("tool_parameters", {}) if step else {})
        identified_gos = state.get("identified_go_numbers", [])

        # 1. Parameter Extraction & Safe Operation / Direction Inference
        target_go = params.get("go_number") or (identified_gos[0] if identified_gos else None)

        intent = state.get("intent")
        plan = state.get("execution_plan") or {}
        intent_val = intent or plan.get("intent")
        intent_str = intent_val.value if hasattr(intent_val, "value") else str(intent_val or "").strip().upper()
        is_incoming_intent = intent_str in ("INCOMING_RELATIONSHIPS", "QUERYINTENT.INCOMING_RELATIONSHIPS")
        is_outgoing_intent = intent_str in ("OUTGOING_RELATIONSHIPS", "QUERYINTENT.OUTGOING_RELATIONSHIPS")

        direction_val = RelationshipDirection.ALL
        if "direction" in params and params["direction"] is not None:
            raw_dir = str(params["direction"]).strip().upper()
            if raw_dir not in ("INCOMING", "OUTGOING"):
                return {
                    "failed_step_ids": [step_id],
                    "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                    "step_retry_count": 0,
                }
            try:
                direction_val = RelationshipDirection(raw_dir)
            except ValueError:
                return {
                    "failed_step_ids": [step_id],
                    "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                    "step_retry_count": 0,
                }
        elif is_incoming_intent:
            direction_val = RelationshipDirection.INCOMING
        elif is_outgoing_intent:
            direction_val = RelationshipDirection.OUTGOING

        # Safe operation inference:
        # 1. If operation is explicitly supplied, preserve and validate it exactly as currently implemented.
        # 2. If operation is absent but direction is supplied, infer GET_RELATIONSHIPS.
        # 3. If intent is INCOMING_RELATIONSHIPS or OUTGOING_RELATIONSHIPS, infer GET_RELATIONSHIPS.
        # 4. If absent, preserve the existing LOOKUP_ORDER default.
        if "operation" in params and params["operation"] is not None:
            raw_op = params["operation"]
        elif ("direction" in params and params["direction"] is not None) or is_incoming_intent or is_outgoing_intent:
            raw_op = "GET_RELATIONSHIPS"
        else:
            raw_op = "LOOKUP_ORDER"

        try:
            operation = DatabaseOperation(str(raw_op).strip().upper())
        except ValueError:
            return {
                "failed_step_ids": [step_id],
                "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                "step_retry_count": 0,
            }

        # Conditional go_number enforcement
        if operation != DatabaseOperation.LIST_ALL_ORDERS and not target_go:
            logger.warning("DatabaseAgent operation '%s' requires non-empty go_number.", operation)
            return {
                "failed_step_ids": [step_id],
                "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                "step_retry_count": 0,
            }

        include_ft = bool(params.get("include_full_text", True))

        try:
            req = DatabaseAgentRequest(
                operation=operation,
                go_number=target_go,
                include_full_text=include_ft,
                text_preview_chars=int(params.get("text_preview_chars", 500)),
                direction=direction_val,
            )
            resp = self.db_agent.execute(req)

        except Exception as exc:
            logger.warning("DatabaseAgent execution failed with technical error: %s", exc)
            curr_step_retry = state.get("step_retry_count", 0)
            if curr_step_retry < 2:
                return {
                    "step_retry_count": curr_step_retry + 1,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "workflow_status": "RUNNING",
                }
            return {
                "failed_step_ids": [step_id],
                "step_retry_count": 0,
                "human_review_required": True,
                "human_review_notes": f"DatabaseAgent failed after 3 attempts: {exc}",
            }

        trace_item = self.supervisor.create_trace_item(
            agent="database_agent",
            action=operation.value,
            tool_called="database_tool",
            tool_input=req.model_dump(mode="json"),
            tool_output_summary=f"DB status: {resp.status.value}, match: {resp.match_type}",
            decision="proceed_to_next_step",
            decision_source="DETERMINISTIC_RULES",
        )

        # Ambiguous match requires human review
        if resp.status == AgentResponseStatus.AMBIGUOUS_MATCH:
            return {
                "completed_step_ids": [step_id],
                "step_retry_count": 0,
                "human_review_required": True,
                "human_review_notes": f"Ambiguous database match for '{target_go}': {resp.ambiguous_candidates}",
                "execution_trace": [trace_item],
            }

        # Technical error
        if resp.status in (AgentResponseStatus.ERROR, AgentResponseStatus.EMPTY_INPUT):
            curr_step_retry = state.get("step_retry_count", 0)
            if curr_step_retry < 2:
                return {
                    "step_retry_count": curr_step_retry + 1,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "execution_trace": [trace_item],
                }
            return {
                "failed_step_ids": [step_id],
                "step_retry_count": 0,
                "human_review_required": True,
                "human_review_notes": f"DatabaseAgent returned {resp.status.value}: {resp.errors}",
                "execution_trace": [trace_item],
            }

        # Success or NOT_FOUND: valid completion
        update_dict: Dict[str, Any] = {
            "completed_step_ids": [step_id],
            "step_retry_count": 0,
            "workflow_status": "RUNNING",
            "execution_trace": [trace_item],
        }

        if resp.order_data:
            update_dict["document_analysis"] = [resp.order_data.model_dump()]
        elif operation == DatabaseOperation.GET_RELATIONSHIPS and target_go:
            existing_docs = state.get("document_analysis", [])
            has_order_data = any(
                isinstance(d, dict) and d.get("go_number") and d.get("department")
                for d in existing_docs
            )
            if not has_order_data:
                try:
                    order_lookup = self.db_agent.db_tool.get_order_by_number(go_number=target_go)
                    if isinstance(order_lookup, dict) and order_lookup.get("status") == "SUCCESS" and order_lookup.get("order"):
                        o = order_lookup["order"]
                        update_dict["document_analysis"] = [
                            {
                                "id": o.get("id"),
                                "go_number": o.get("go_number"),
                                "issue_date": str(o.get("issue_date")) if o.get("issue_date") else None,
                                "department": o.get("department"),
                                "source_filename": o.get("source_filename"),
                                "verification_status": o.get("verification_status"),
                                "extracted_text_length": o.get("extracted_text_length", 0),
                                "text_preview": o.get("text_preview", ""),
                            }
                        ]
                except Exception as ex:
                    logger.debug("Could not attach focal order metadata during GET_RELATIONSHIPS: %s", ex)
        elif resp.status == AgentResponseStatus.NOT_FOUND:
            update_dict["document_analysis"] = [{"go_number": target_go, "status": "NOT_FOUND"}]

        if resp.relationships:
            update_dict["relationship_results"] = [r.model_dump() for r in resp.relationships]

        if resp.discovered_go_numbers:
            update_dict["discovered_go_numbers"] = resp.discovered_go_numbers

        return update_dict

    def relationship_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes RelationshipAgent: analyzes references and relational text.
        """
        step, step_id = self._get_current_step_info(state)
        params = (step.get("tool_parameters", {}) if step else {})
        identified_gos = state.get("identified_go_numbers", [])

        # Check for conflicting parameters
        if "source_go_number" in params and "go_number" in params:
            s_go = str(params["source_go_number"]).strip().lower()
            g_go = str(params["go_number"]).strip().lower()
            if s_go != g_go:
                logger.warning("Conflicting parameters in relationship_tool: %s vs %s", s_go, g_go)
                return {
                    "failed_step_ids": [step_id],
                    "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                    "step_retry_count": 0,
                }

        target_go = params.get("go_number") or params.get("source_go_number") or (identified_gos[0] if identified_gos else None)
        target_pair = params.get("target_go_number")

        # Determine mode
        raw_mode = params.get("mode")
        if raw_mode:
            try:
                mode = RelationshipInvestigationMode(str(raw_mode).strip().upper())
            except ValueError:
                return {
                    "failed_step_ids": [step_id],
                    "invalid_plan_count": state.get("invalid_plan_count", 0) + 1,
                    "step_retry_count": 0,
                }
        else:
            if target_go and target_pair:
                mode = RelationshipInvestigationMode.INVESTIGATE_PAIR
            elif params.get("text"):
                mode = RelationshipInvestigationMode.EXTRACT_FROM_TEXT
            else:
                mode = RelationshipInvestigationMode.INVESTIGATE_ORDER

        # Extract text from document_analysis if available and needed
        doc_text = params.get("text")
        if not doc_text and state.get("document_analysis"):
            for doc in state["document_analysis"]:
                if isinstance(doc, dict) and doc.get("extracted_text"):
                    doc_text = doc["extracted_text"]
                    break

        try:
            req = RelationshipAgentRequest(
                mode=mode,
                go_number=target_go,
                target_go_number=target_pair,
                text=doc_text,
            )
            resp = self.rel_agent.execute(req)

        except Exception as exc:
            logger.warning("RelationshipAgent execution failed: %s", exc)
            curr_step_retry = state.get("step_retry_count", 0)
            if curr_step_retry < 2:
                return {
                    "step_retry_count": curr_step_retry + 1,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "workflow_status": "RUNNING",
                }
            return {
                "failed_step_ids": [step_id],
                "step_retry_count": 0,
                "human_review_required": True,
                "human_review_notes": f"RelationshipAgent failed: {exc}",
            }

        trace_item = self.supervisor.create_trace_item(
            agent="relationship_agent",
            action=mode.value,
            tool_called="relationship_tool",
            tool_input=req.model_dump(mode="json"),
            tool_output_summary=f"Rel status: {resp.status.value}, found {resp.total_relationships} candidates",
            decision="proceed_to_verification",
            decision_source="DETERMINISTIC_RULES",
        )

        if resp.status == RelationshipAgentStatus.AMBIGUOUS_MATCH:
            return {
                "completed_step_ids": [step_id],
                "step_retry_count": 0,
                "human_review_required": True,
                "human_review_notes": f"Ambiguous reference match in relationship analysis for {target_go}",
                "execution_trace": [trace_item],
            }

        # Serialized relationship candidates (preserving UNVERIFIED_CANDIDATE & False legal invariant)
        rel_candidates = [r.model_dump() for r in resp.relationships]

        return {
            "relationship_results": rel_candidates,
            "completed_step_ids": [step_id],
            "step_retry_count": 0,
            "workflow_status": "RUNNING",
            "execution_trace": [trace_item],
        }

    @classmethod
    def _get_user_focal_go(cls, state: AgentState) -> Optional[str]:
        """
        Extract the single, actual user-requested focal Government Order from workflow state.
        Ensures incoming relationship filtering scopes strictly to the user-requested focal order,
        preventing discovered or multi-identified GOs from corrupting directional filtering.
        """
        user_query = state.get("user_query", "")
        if user_query:
            m = re.search(
                r"G\.?\s*O\.?\s*(?:\([A-Za-z]+\))?\s*(?:No\.?)?\s*\d+(?:/\d{4}/[A-Za-z&.-]+)?|"
                r"സ\s*\.?\s*ഉ\s*\.?\s*(?:\([^)]*\))?\s*(?:നം|നമ്പർ)?\s*\.?\s*\d+(?:/\d{4}/[\w&.-]+)?|"
                r"\b\d+/\d{4}/[A-Za-z&.-]+",
                user_query,
                re.IGNORECASE,
            )
            if m:
                return m.group(0).rstrip(".?, ")

        plan = state.get("execution_plan") or {}
        for s in plan.get("execution_steps", []):
            params = s.get("tool_parameters", {})
            if params.get("go_number"):
                return str(params["go_number"]).strip()

        docs = state.get("document_analysis", [])
        for d in docs:
            if isinstance(d, dict) and d.get("go_number"):
                return str(d["go_number"]).strip()

        identified = state.get("identified_go_numbers", [])
        if identified:
            return str(identified[0]).strip()

        return None

    def verification_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes VerificationAgent: fact-checks claims and candidate relationships against database authority.
        """
        step, step_id = self._get_current_step_info(state)
        user_query = state.get("user_query", "")
        identified_gos = state.get("identified_go_numbers", [])
        discovered_gos = state.get("discovered_go_numbers", [])
        search_results = state.get("search_results", [])

        # Determine focal orders for relationship cross-examination
        # Priority 1: Explicit identified GOs from user query / supervisor
        # Priority 2: Discovered GOs from database lookups (when no explicit GO exists)
        if identified_gos:
            raw_focal = [str(g).strip() for g in identified_gos if g and str(g).strip()]
        elif discovered_gos:
            raw_focal = [str(g).strip() for g in discovered_gos if g and str(g).strip()]
        else:
            raw_focal = []

        # Deduplicate focal orders using structural equivalence (e.g. 24/2026/ITD vs G.O.(Rt)No.24/2026/ITD)
        deduped_focal: List[str] = []
        for g in raw_focal:
            matched_idx = -1
            for idx, existing in enumerate(deduped_focal):
                if self._are_go_numbers_matching(g, existing):
                    matched_idx = idx
                    break
            if matched_idx >= 0:
                if len(g) > len(deduped_focal[matched_idx]):
                    deduped_focal[matched_idx] = g
            else:
                deduped_focal.append(g)

        # Fallback to search results ONLY when:
        # 1. No explicit or discovered focal GO is available
        # 2. User intent genuinely requires relationship investigation
        # 3. Top search hit is sufficiently identifiable, not ambiguous, and meaningfully aligned
        if not deduped_focal and search_results:
            plan = state.get("execution_plan") or {}
            intent_val = str(state.get("intent") or plan.get("intent") or "").upper()
            rel_intents = {"INCOMING_RELATIONSHIPS", "OUTGOING_RELATIONSHIPS", "EXPLAIN_ORDER"}
            if intent_val in rel_intents:
                top_hit = search_results[0]
                if isinstance(top_hit, dict):
                    if self._is_search_hit_meaningfully_aligned(top_hit, user_query, state):
                        raw_go = str(top_hit.get("go_number") or "").strip()
                        deduped_focal = [raw_go]

        focal_cited_gos = deduped_focal
        raw_candidates = state.get("relationship_results", [])

        plan = state.get("execution_plan") or {}
        intent_val = str(state.get("intent") or plan.get("intent") or "").upper()
        is_incoming = "INCOMING" in intent_val
        user_focal_go = self._get_user_focal_go(state) if is_incoming else None
        if is_incoming and user_focal_go:
            focal_cited_gos = [user_focal_go]

        # Build RelationshipCandidateInput objects strictly enforcing legal_effect_verified=False
        typed_candidates = []
        for rc in raw_candidates:
            if isinstance(rc, dict):
                src = rc.get("source_go_number") or rc.get("source_order_number")
                tgt = rc.get("target_order_number") or rc.get("target_go_number") or (rc.get("referenced_go_numbers", [None])[0] if rc.get("referenced_go_numbers") else None)
                rel_type = rc.get("relationship_type", "REFERENCES")
                if src and tgt:
                    # In an incoming-only query, filter strictly against user_focal_go
                    if is_incoming and user_focal_go:
                        c_dir = str(rc.get("direction", "")).upper()
                        if c_dir == "OUTGOING":
                            continue
                        if self._are_go_numbers_matching(src, user_focal_go):
                            continue
                        if not self._are_go_numbers_matching(tgt, user_focal_go):
                            continue
                    typed_candidates.append(
                        RelationshipCandidateInput(
                            relationship_type=str(rel_type),
                            source_go_number=str(src),
                            target_order_number=str(tgt),
                            matched_phrase=rc.get("matched_phrase"),
                            evidence_text=rc.get("evidence_text"),
                            page_number=rc.get("page_number"),
                            confidence=rc.get("confidence", "MEDIUM"),
                            verification_status="UNVERIFIED_CANDIDATE",
                            legal_effect_verified=False,
                        )
                    )

        # Build SearchEvidenceInput objects preserving search results provenance
        typed_search_evidence: List[SearchEvidenceInput] = []
        for sr in search_results:
            if isinstance(sr, dict) and sr.get("go_number"):
                chunks = []
                for sc in sr.get("supporting_chunks", []):
                    if isinstance(sc, dict):
                        chunks.append(
                            SupportingChunkInput(
                                chunk_index=sc.get("chunk_index", 0),
                                chunk_text=sc.get("chunk_text") or sc.get("content") or "",
                                page_number=sc.get("page_number"),
                                score=sc.get("score"),
                            )
                        )
                typed_search_evidence.append(
                    SearchEvidenceInput(
                        go_number=str(sr["go_number"]).strip(),
                        go_id=sr.get("go_id"),
                        department=sr.get("department"),
                        issue_date=str(sr.get("issue_date")) if sr.get("issue_date") is not None else None,
                        source_filename=sr.get("source_filename"),
                        text_snippet=sr.get("text_snippet"),
                        supporting_chunks=chunks,
                    )
                )

        logger.info(
            "VerificationNode: Entering verification (step_id='%s', cited_gos=%s, candidates_count=%d, search_evidence_count=%d)",
            step_id, focal_cited_gos, len(typed_candidates), len(typed_search_evidence)
        )

        try:
            req = VerificationAgentRequest(
                user_query=user_query,
                claims=[],  # Direct relationship verification; no fabricated claims
                cited_go_numbers=focal_cited_gos,
                relationship_candidates=typed_candidates,
                document_evidence=[],  # Caller never fabricates authoritative full text
                search_evidence=typed_search_evidence,
                max_retries=state.get("max_retries", 2),
                current_retry=state.get("retry_count", 0),
            )
            resp = self.verification_agent.execute(req)

        except Exception as exc:
            logger.warning("VerificationAgent execution failed: %s", exc)
            curr_step_retry = state.get("step_retry_count", 0)
            if curr_step_retry < 2:
                return {
                    "step_retry_count": curr_step_retry + 1,
                    "retry_count": state.get("retry_count", 0) + 1,
                    "workflow_status": "RUNNING",
                }
            return {
                "failed_step_ids": [step_id],
                "step_retry_count": 0,
                "human_review_required": True,
                "human_review_notes": f"VerificationAgent failed: {exc}",
            }

        trace_item = self.supervisor.create_trace_item(
            agent="verification_agent",
            action="execute_verification",
            tool_called="none",
            tool_input={"cited_gos": focal_cited_gos, "candidates_count": len(typed_candidates)},
            tool_output_summary=f"Verification status: {resp.status.value}, confidence: {resp.confidence_score:.2f}",
            decision="proceed_to_response",
            decision_source="DETERMINISTIC_RULES",
        )

        logger.info(
            "VerificationNode: Verification completed -> status=%s, evidence_sufficient=%s, confidence=%.2f",
            resp.status.value, resp.evidence_sufficient, resp.confidence_score
        )

        conf_val = resp.overall_confidence.value if hasattr(resp.overall_confidence, "value") else str(resp.overall_confidence)

        update_dict: Dict[str, Any] = {
            "completed_step_ids": [step_id],
            "step_retry_count": 0,
            "verification_status": resp.status.value,
            "verification_results": resp.model_dump(),
            "confidence": conf_val,
            "evidence_sufficient": resp.evidence_sufficient,
            "workflow_status": "RUNNING",
            "execution_trace": [trace_item],
        }

        # Status routing flags
        if resp.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED, VerificationStatus.NOT_FOUND):
            update_dict["response_ready"] = True

        elif resp.status == VerificationStatus.CONFLICTING_EVIDENCE:
            update_dict["human_review_required"] = True
            update_dict["human_review_notes"] = "VerificationAgent detected conflicting evidence between orders."

        elif resp.status == VerificationStatus.INSUFFICIENT_EVIDENCE:
            # If primary evidence exists (e.g. search results without specific relationship targets)
            # or generation bound reached, proceed to response with qualified limitations rather than looping
            if state.get("search_results") or state.get("document_analysis") or state.get("plan_generation", 1) >= 6:
                update_dict["response_ready"] = True
                update_dict["evidence_sufficient"] = False

        return update_dict

    @staticmethod
    def _is_relationship_supported(item: Any) -> bool:
        """Check if an item in verified_relationships was corroborated by the database."""
        if isinstance(item, dict):
            status = item.get("registration_status") or item.get("status")
        else:
            status = getattr(item, "registration_status", None) or getattr(item, "status", None)
        if status is None:
            return True
        status_str = status.value if hasattr(status, "value") else str(status)
        return status_str in (
            "REGISTERED_EXACT_MATCH",
            "REGISTERED_COMPLEMENTARY_MATCH",
            "RECORDED_EXACT",
            "RECORDED_COMPLEMENTARY",
        )

    @staticmethod
    def _extract_relationship_tuple(item: Any) -> Tuple[str, str, str]:
        """Extract source, target, and relationship_type from dict or object."""
        if isinstance(item, dict):
            src = (
                item.get("source_go_number")
                or item.get("source_order_number")
                or item.get("source_go")
                or item.get("cited_go_number")
                or item.get("cited_go")
                or ""
            )
            tgt = (
                item.get("target_go_number")
                or item.get("target_order_number")
                or item.get("target_go")
                or item.get("referenced_go_number")
                or item.get("referenced_order_number")
            )
            if not tgt and item.get("referenced_go_numbers"):
                refs = item["referenced_go_numbers"]
                if isinstance(refs, (list, tuple)) and refs:
                    tgt = refs[0]
            tgt = tgt or ""
            rtype = (
                item.get("relationship_type")
                or item.get("candidate_relationship_type")
                or item.get("database_relationship_type")
                or ""
            )
        else:
            src = (
                getattr(item, "source_go_number", None)
                or getattr(item, "source_order_number", None)
                or getattr(item, "source_go", None)
                or getattr(item, "cited_go_number", None)
                or getattr(item, "cited_go", "")
                or ""
            )
            tgt = (
                getattr(item, "target_go_number", None)
                or getattr(item, "target_order_number", None)
                or getattr(item, "target_go", None)
                or getattr(item, "referenced_go_number", None)
                or getattr(item, "referenced_order_number", None)
            )
            if not tgt and getattr(item, "referenced_go_numbers", None):
                refs = item.referenced_go_numbers
                if isinstance(refs, (list, tuple)) and refs:
                    tgt = refs[0]
            tgt = tgt or ""
            rtype = (
                getattr(item, "relationship_type", None)
                or getattr(item, "candidate_relationship_type", None)
                or getattr(item, "database_relationship_type", "")
                or ""
            )
        rtype_str = rtype.value if hasattr(rtype, "value") else str(rtype)
        return str(src).strip(), str(tgt).strip(), rtype_str.strip().upper()

    @staticmethod
    def _are_go_numbers_matching(go1: Optional[str], go2: Optional[str]) -> bool:
        """
        Check if two Government Order identifiers refer to the same order.
        Requires matching order number and year, with compatible department codes.
        Prevents substring collisions on fragments (e.g. 'G.O.', 'ITD', '2026', '24').
        """
        if not go1 or not go2:
            return False

        norm1 = VerificationAgent._normalize_go_identifier(go1)
        norm2 = VerificationAgent._normalize_go_identifier(go2)

        clean1 = re.sub(r"\s+", "", norm1)
        clean2 = re.sub(r"\s+", "", norm2)

        # 1. Strict validation: both inputs must contain a complete, structurally valid GO identifier (number/year)
        pattern = re.compile(
            r"^(?:G\.O\.?(?:\([A-Za-z]+\))?(?:NO\.?)?)?(\d+)/(\d{4})(?:/([A-Za-z0-9&.-]+))?$",
            re.IGNORECASE,
        )
        m1 = pattern.match(clean1)
        m2 = pattern.match(clean2)

        # Both must pass structural validation; fragments (e.g. '24', 'ITD', 'G.O.') are strictly rejected
        if not m1 or not m2:
            return False

        # 2. Structural comparison: order number and year must match exactly
        num1, year1, dept1 = m1.group(1), m1.group(2), (m1.group(3) or "").strip().upper()
        num2, year2, dept2 = m2.group(1), m2.group(2), (m2.group(3) or "").strip().upper()

        if num1 != num2 or year1 != year2:
            return False

        # 3. Department comparison: both must match exactly; one present and one missing must never match
        if dept1 != dept2:
            return False

        return True

    @classmethod
    def _is_valid_go_identifier(cls, identifier: Optional[str]) -> bool:
        """
        Check if an identifier is a complete, valid GO format (at least number and 4-digit year).
        Does not require a department suffix, but strictly rejects incomplete fragments
        such as '24', 'ITD', 'G.O.', '2026'.
        """
        if not identifier:
            return False
        clean = re.sub(r"\s+", "", str(identifier).strip().upper())
        pattern = re.compile(
            r"^(?:G\.O\.?(?:\([A-Za-z]+\))?(?:NO\.?)?)?(\d+)/(\d{4})(?:/([A-Za-z0-9&.-]+))?$",
            re.IGNORECASE,
        )
        return pattern.match(clean) is not None

    @classmethod
    def _is_department_compatible(cls, dept1: Optional[str], dept2: Optional[str]) -> bool:
        """Check compatibility between department strings or codes."""
        if not dept1 or not dept2:
            return True
        d1 = dept1.strip().lower()
        d2 = dept2.strip().lower()
        if d1 == d2 or d1 in d2 or d2 in d1:
            return True
        known_aliases = {
            "itd": "information technology",
            "fin": "finance",
            "gad": "general administration",
            "h_edn": "higher education",
            "p_ard": "personnel",
            "pwd": "public works",
        }
        for code, full_name in known_aliases.items():
            if (code in d1 or full_name in d1) and (code in d2 or full_name in d2):
                return True
        return False

    @classmethod
    def _is_search_hit_meaningfully_aligned(
        cls, top_hit: Dict[str, Any], user_query: str, state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine whether a top search hit is meaningfully aligned with the user query
        to serve as a fallback focal order for relationship cross-examination.
        Rejects candidates based on generic tokens, positive semantic scores alone,
        or department mismatches.
        """
        cand_go = str(top_hit.get("go_number") or "").strip()
        if not cls._is_valid_go_identifier(cand_go):
            return False

        q_clean = user_query.strip().lower()
        cand_dept = str(top_hit.get("department") or "").strip()
        cand_snippet = str(top_hit.get("text_snippet") or "").strip().lower()
        cand_file = str(top_hit.get("source_filename") or "").strip().lower()

        # 1. Department consistency check
        state_depts = [str(d).strip() for d in (state.get("identified_departments", []) if state else []) if d]
        if state_depts and cand_dept:
            if not any(cls._is_department_compatible(sd, cand_dept) for sd in state_depts):
                return False

        known_aliases = {
            "itd": "information technology",
            "fin": "finance",
            "gad": "general administration",
            "h_edn": "higher education",
            "p_ard": "personnel",
            "pwd": "public works",
        }
        for code, full_name in known_aliases.items():
            if re.search(rf"\b{code}\b", q_clean) or full_name in q_clean:
                if cand_dept and not cls._is_department_compatible(code, cand_dept):
                    return False

        # 2. Substantive query terms alignment (excluding administrative/query stopwords)
        stopwords = {
            "order", "orders", "government", "kerala", "dated", "date", "no", "number",
            "show", "details", "find", "give", "what", "which", "list", "view",
            "regarding", "about", "state", "please", "the", "and", "for", "with",
            "from", "in", "of", "on", "to", "is", "are", "were", "was", "any", "all",
            "rt", "ms", "p", "g.o.", "g.o", "go"
        }
        query_words = re.findall(r"[a-z0-9&.-]+", q_clean)
        substantive_tokens = [w for w in query_words if len(w) >= 3 and w not in stopwords and not w.isdigit()]

        cand_corpus = f"{cand_go.lower()} {cand_dept} {cand_snippet} {cand_file}"
        if substantive_tokens:
            has_substantive_match = any(token in cand_corpus for token in substantive_tokens)
            if not has_substantive_match:
                return False

        # 3. Relational context alignment
        relational_terms = {
            "amend", "modify", "supersede", "cancel", "revoke", "read as",
            "reference", "cited", "in partial modification", "relationship",
            "subsequent", "prior", "earlier", "later"
        }
        has_query_rel = any(rt in q_clean for rt in relational_terms)
        has_cand_rel = any(rt in cand_snippet for rt in relational_terms)
        if not (has_query_rel or has_cand_rel):
            return False

        return True

    @classmethod
    def _deduplicate_supported_relationships(
        cls, relationships: List[Any]
    ) -> List[Any]:
        """
        Deduplicate equivalent supported relationships while preserving complete evidence provenance.
        Merges distinct excerpts, page numbers, source documents, and claim references without merging
        different relationship types, directions, source orders, or target orders.
        """
        deduped: List[Dict[str, Any]] = []

        for rel_item in relationships:
            rel = dict(rel_item) if isinstance(rel_item, dict) else (
                rel_item.model_dump() if hasattr(rel_item, "model_dump") else dict(rel_item)
            )
            src, tgt, r_type = cls._extract_relationship_tuple(rel)
            if not src or not tgt or not r_type:
                deduped.append(rel)
                continue

            r_dir = rel.get("direction")
            r_page_ver = rel.get("page_number_verified")

            matched_idx = -1
            for idx, existing in enumerate(deduped):
                e_src, e_tgt, e_type = cls._extract_relationship_tuple(existing)
                e_dir = existing.get("direction")
                e_page_ver = existing.get("page_number_verified")

                # 1. Strictly compare relationship type, source, and target
                if r_type != e_type:
                    continue
                if not cls._are_go_numbers_matching(src, e_src):
                    continue
                if not cls._are_go_numbers_matching(tgt, e_tgt):
                    continue

                # 2. Strictly compare direction: conflicting or missing-vs-present directions must not be merged
                if r_dir != e_dir:
                    continue

                # 3. Strictly compare page verification status: verified and unverified page records must not be merged
                if r_page_ver != e_page_ver:
                    continue

                matched_idx = idx
                break

            if matched_idx == -1:
                deduped.append(dict(rel))
            else:
                existing = deduped[matched_idx]

                # Prefer longer / more canonical source and target display strings
                if len(src) > len(existing.get("source_go_number", "")):
                    existing["source_go_number"] = src
                if len(tgt) > len(existing.get("target_go_number", "")):
                    existing["target_go_number"] = tgt

                # Preserve distinct operative excerpts in standard evidence_text field
                existing_excerpt = existing.get("evidence_text") or ""
                new_excerpt = rel.get("evidence_text") or ""
                if new_excerpt and new_excerpt not in existing_excerpt:
                    if existing_excerpt:
                        existing["evidence_text"] = f"{existing_excerpt} | {new_excerpt}"
                    else:
                        existing["evidence_text"] = new_excerpt

                # Preserve source document filenames in standard source_filename field
                existing_fn = existing.get("source_filename") or ""
                new_fn = rel.get("source_filename") or ""
                if new_fn and new_fn not in existing_fn:
                    if existing_fn:
                        existing["source_filename"] = f"{existing_fn}, {new_fn}"
                    else:
                        existing["source_filename"] = new_fn

                # Preserve secondary page citation in standard notes field if pages differ
                e_page = existing.get("page_number")
                r_page = rel.get("page_number")
                if e_page is not None and r_page is not None and e_page != r_page:
                    note_str = f"Also cited on page {r_page}"
                    existing_notes = existing.get("notes") or ""
                    if note_str not in existing_notes:
                        existing["notes"] = f"{existing_notes} | {note_str}" if existing_notes else note_str

                # Preserve claim reference in standard claim_id field if missing
                if not existing.get("claim_id") and rel.get("claim_id"):
                    existing["claim_id"] = rel["claim_id"]

                # Preserve temporal dates & order IDs
                if not existing.get("source_date") and rel.get("source_date"):
                    existing["source_date"] = rel["source_date"]
                if not existing.get("target_date") and rel.get("target_date"):
                    existing["target_date"] = rel["target_date"]
                if not existing.get("source_order_id") and rel.get("source_order_id"):
                    existing["source_order_id"] = rel["source_order_id"]
                if not existing.get("target_order_id") and rel.get("target_order_id"):
                    existing["target_order_id"] = rel["target_order_id"]

                # Preserve notes
                existing_notes = existing.get("notes") or ""
                new_notes = rel.get("notes") or ""
                if new_notes and new_notes not in existing_notes:
                    if existing_notes:
                        existing["notes"] = f"{existing_notes} | {new_notes}"
                    else:
                        existing["notes"] = new_notes

                # 8. Preserve directness classification (prefer DIRECT over INDIRECT or UNCLASSIFIED)
                dir_order = {
                    "DIRECT_OPERATIVE_ORDER": 3,
                    "DIRECT": 3,
                    "INDIRECT_PREAMBLE_CITATION": 2,
                    "INDIRECT": 2,
                    "UNCLASSIFIED": 1,
                    "NONE": 0,
                }
                curr_rank = dir_order.get(str(existing.get("evidence_directness", "")), 0)
                new_rank = dir_order.get(str(rel.get("evidence_directness", "")), 0)
                if new_rank > curr_rank:
                    existing["evidence_directness"] = rel.get("evidence_directness")

                # 9. Preserve registration status (prefer REGISTERED_EXACT_MATCH over UNRECORDED)
                if rel.get("registration_status") == "REGISTERED_EXACT_MATCH":
                    existing["registration_status"] = "REGISTERED_EXACT_MATCH"

        return deduped

    @classmethod
    def _matches_supported_relationship(cls, cand: Any, supported_items: List[Any]) -> bool:
        """
        Check if a candidate matches any supported relationship.
        Preserves relationship direction strictly (source vs source, target vs target).
        """
        c_src, c_tgt, c_type = cls._extract_relationship_tuple(cand)
        if not c_src or not c_tgt:
            return False
        for supp in supported_items:
            s_src, s_tgt, s_type = cls._extract_relationship_tuple(supp)
            if not s_src or not s_tgt:
                continue
            # 1. Type match
            if c_type != s_type:
                continue
            # 2. Source match (preserving direction: c_src vs s_src)
            if not cls._are_go_numbers_matching(c_src, s_src):
                continue
            # 3. Target match (preserving direction: c_tgt vs s_tgt)
            if not cls._are_go_numbers_matching(c_tgt, s_tgt):
                continue
            return True
        return False

    def response_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesizes the evidence-backed final response using ResponseAgent.
        """
        verif_data = state.get("verification_results") or {}
        verif_status = state.get("verification_status")

        # Numerical confidence passed only if VerificationAgent executed
        conf_score = verif_data.get("confidence_score") if verif_status else None

        logger.info(
            "ResponseNode: Entering response synthesis (intent=%s, verification_status=%s, evidence_sufficient=%s)",
            state.get("intent"), verif_status, bool(state.get("evidence_sufficient", False))
        )

        # 1. Collect supported relationships when overall status is VERIFIED or PARTIALLY_VERIFIED
        supported_raw = (
            verif_data.get("verified_relationships", [])
            if verif_status in ("VERIFIED", "PARTIALLY_VERIFIED")
            else []
        )
        supported_filtered = [r for r in supported_raw if self._is_relationship_supported(r)]
        supported_rels = self._deduplicate_supported_relationships(supported_filtered)

        plan = state.get("execution_plan") or {}
        intent_val = str(state.get("intent") or plan.get("intent") or "").upper()
        is_incoming = "INCOMING" in intent_val
        user_focal_go = self._get_user_focal_go(state) if is_incoming else None

        # Directional incoming filtering for verified/supported relationships
        filtered_supported_rels = []
        for r in supported_rels:
            if is_incoming and user_focal_go:
                r_src, r_tgt, _ = self._extract_relationship_tuple(r)
                if r_src and self._are_go_numbers_matching(r_src, user_focal_go):
                    continue
                if not r_tgt or not self._are_go_numbers_matching(r_tgt, user_focal_go):
                    continue
            filtered_supported_rels.append(r)

        # 2. Exclude only candidates that match an already-supported relationship; preserve genuinely unverified candidates
        raw_candidates = state.get("relationship_results", [])
        unverified_candidates = []

        for cand in raw_candidates:
            if not self._matches_supported_relationship(cand, supported_rels):
                if is_incoming and user_focal_go:
                    c_src, c_tgt, _ = self._extract_relationship_tuple(cand)
                    c_dir = str(cand.get("direction", "")).upper() if isinstance(cand, dict) else ""
                    if c_dir == "OUTGOING":
                        continue
                    if self._are_go_numbers_matching(c_src, user_focal_go):
                        continue
                    if not self._are_go_numbers_matching(c_tgt, user_focal_go):
                        continue
                unverified_candidates.append(cand)

        # Discrepancies directional filtering for incoming queries
        raw_discrepancies = verif_data.get("discrepancies", [])
        filtered_discrepancies = []
        for d in raw_discrepancies:
            if is_incoming and user_focal_go:
                d_src = d.get("source_order_number") or d.get("source_go_number") or d.get("cited_go_number")
                d_tgt = d.get("target_order_number") or d.get("target_go_number") or d.get("referenced_go_number")
                if d_src and self._are_go_numbers_matching(d_src, user_focal_go):
                    continue
                if not d_tgt or not self._are_go_numbers_matching(d_tgt, user_focal_go):
                    continue
            filtered_discrepancies.append(d)

        # For discovery searches without prior VerificationAgent execution, resolve incoming
        # recorded status for the top relevant candidate orders with strict session recovery
        order_recorded_statuses: Dict[int, Dict[str, Any]] = {}
        search_results = state.get("search_results", [])
        if not verif_status and search_results:
            top_candidates = self.response_agent.select_relevant_orders(
                query=state.get("user_query", ""),
                search_results=search_results,
                max_orders=2,
            )
            try:
                from app.database.connection import SessionLocal
                from app.services.relationship_resolution import resolve_incoming_relationships_for_order

                with SessionLocal() as db_session:
                    for cand in top_candidates:
                        cand_id = cand.get("id") or cand.get("go_id")
                        cand_go = cand.get("go_number")
                        if cand_id is None or not cand_go:
                            continue
                        try:
                            rels, success = resolve_incoming_relationships_for_order(
                                order_id=cand_id,
                                go_number=cand_go,
                                db=db_session,
                            )
                            if not success:
                                db_session.rollback()
                            status_str, rel_go = self.response_agent.compute_order_recorded_status(
                                relationships=rels,
                                lookup_success=success,
                            )
                            order_recorded_statuses[cand_id] = {
                                "status": status_str,
                                "related_go": rel_go,
                            }
                        except Exception as cand_exc:
                            logger.error(
                                "Failed relationship lookup for candidate order %s (%s): %s",
                                cand_id,
                                cand_go,
                                cand_exc,
                            )
                            try:
                                db_session.rollback()
                            except Exception:
                                pass
                            order_recorded_statuses[cand_id] = {
                                "status": "Status requires further verification",
                                "related_go": None,
                            }
            except Exception as db_exc:
                logger.error("Database session creation failed in response_node: %s", db_exc)
                for cand in top_candidates:
                    cand_id = cand.get("id") or cand.get("go_id")
                    if cand_id is not None and cand_id not in order_recorded_statuses:
                        order_recorded_statuses[cand_id] = {
                            "status": "Status requires further verification",
                            "related_go": None,
                        }

        # Build ResponseAgentInput
        resp_input = ResponseAgentInput(
            user_query=state.get("user_query", ""),
            intent=state.get("intent"),
            focal_go=user_focal_go,
            verification_status=verif_status,
            verification_confidence_score=conf_score,
            confidence=state.get("confidence"),
            evidence_sufficient=bool(state.get("evidence_sufficient", False)),
            legal_effect_verified=False,
            verified_findings=filtered_supported_rels if verif_status == "VERIFIED" else [],
            partially_verified_supported=filtered_supported_rels if verif_status == "PARTIALLY_VERIFIED" else [],
            partially_verified_unsupported=verif_data.get("unsupported_claims", []) if verif_status == "PARTIALLY_VERIFIED" else [],
            unverified_candidates=unverified_candidates,
            discrepancies=filtered_discrepancies,
            missing_indexed_orders=[d["go_number"] for d in state.get("document_analysis", []) if isinstance(d, dict) and d.get("status") == "NOT_FOUND"],
            search_results=state.get("search_results", []),
            order_recorded_statuses=order_recorded_statuses,
            document_analysis=state.get("document_analysis", []),
            is_incomplete_investigation=(state.get("plan_generation", 1) >= 6 or state.get("retry_count", 0) >= state.get("max_retries", 2)),
            incomplete_reason="Investigation stopped due to generation or retry limit." if (state.get("plan_generation", 1) >= 6 or state.get("retry_count", 0) >= state.get("max_retries", 2)) else None,
        )

        final_text = self.response_agent.generate_response(resp_input)

        trace_item = self.supervisor.create_trace_item(
            agent="response_agent",
            action="synthesize_response",
            tool_called="none",
            tool_input={"verification_status": verif_status},
            tool_output_summary="Final evidence-grounded response synthesized.",
            decision="workflow_completed",
            decision_source="DETERMINISTIC_RULES",
        )

        logger.info("ResponseNode: Response synthesis completed -> length=%d characters", len(final_text))

        return {
            "final_answer": final_text,
            "workflow_status": "COMPLETED",
            "execution_trace": [trace_item],
        }

    def human_review_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Human-in-the-loop checkpoint: Pauses workflow execution via LangGraph interrupt().
        Resumes when caller supplies human feedback.
        """
        reason = state.get("human_review_notes") or "Clarification or review required before continuing"
        query = state.get("user_query")

        user_feedback = interrupt({
            "prompt": "Human review required",
            "reason": reason,
            "user_query": query,
            "current_intent": state.get("intent"),
            "retry_count": state.get("retry_count", 0),
        })

        trace_item = self.supervisor.create_trace_item(
            agent="human_review",
            action="receive_human_feedback",
            tool_called="none",
            tool_input=str(user_feedback),
            tool_output_summary="Human feedback received; resuming workflow.",
            decision="resume_to_supervisor",
            decision_source="DETERMINISTIC_RULES",
        )

        return {
            "human_feedback": str(user_feedback),
            "human_review_status": "RESOLVED",
            "human_review_required": False,
            "workflow_status": "RUNNING",
            "execution_trace": [trace_item],
        }


# ============================================================================
# CONDITIONAL ROUTERS
# ============================================================================

def route_from_supervisor(state: AgentState) -> str:
    """Route from supervisor to the selected agent node."""
    action = state.get("next_action", "search")
    if action == "human_review" or state.get("human_review_required"):
        return "human_review_node"
    if action == "database":
        return "database_node"
    if action == "relationship":
        return "relationship_node"
    if action == "verification":
        return "verification_node"
    if action == "response" or state.get("response_ready"):
        return "response_node"
    return "search_node"


def route_after_agent_node(state: AgentState) -> str:
    """
    Route after an agent executes: returns to supervisor for the next step,
    unless human review is required or response is ready.
    """
    if state.get("human_review_required"):
        return "human_review_node"
    if state.get("response_ready"):
        return "response_node"
    return "supervisor_node"


def route_from_verification(state: AgentState) -> str:
    """
    Route from verification:
    If sufficient or response_ready, proceed to response.
    If human review required (conflicting evidence), route to human review.
    Otherwise return to supervisor for replanning.
    """
    if state.get("human_review_required"):
        return "human_review_node"
    if state.get("response_ready"):
        return "response_node"
    return "supervisor_node"


def route_from_human_review(state: AgentState) -> str:
    """Route from human review strictly back to supervisor to replan with guidance."""
    return "supervisor_node"


# ============================================================================
# GRAPH BUILDER FUNCTION
# ============================================================================

def build_government_orders_graph(
    checkpointer: Optional[MemorySaver] = None,
    nodes: Optional[WorkflowNodes] = None,
) -> Any:
    """
    Construct and compile the LangGraph StateGraph.
    """
    workflow_nodes = nodes or WorkflowNodes()
    memory = checkpointer if checkpointer is not None else MemorySaver()

    builder = StateGraph(AgentState)

    # 1. Add All Nodes
    builder.add_node("supervisor_node", workflow_nodes.supervisor_node)
    builder.add_node("search_node", workflow_nodes.search_node)
    builder.add_node("database_node", workflow_nodes.database_node)
    builder.add_node("relationship_node", workflow_nodes.relationship_node)
    builder.add_node("verification_node", workflow_nodes.verification_node)
    builder.add_node("response_node", workflow_nodes.response_node)
    builder.add_node("human_review_node", workflow_nodes.human_review_node)

    # 2. Add Edges
    builder.add_edge(START, "supervisor_node")

    # Supervisor conditional routing
    builder.add_conditional_edges(
        "supervisor_node",
        route_from_supervisor,
        {
            "search_node": "search_node",
            "database_node": "database_node",
            "relationship_node": "relationship_node",
            "verification_node": "verification_node",
            "response_node": "response_node",
            "human_review_node": "human_review_node",
        },
    )

    # Specialized agent nodes return to supervisor or proceed
    builder.add_conditional_edges(
        "search_node",
        route_after_agent_node,
        {
            "supervisor_node": "supervisor_node",
            "response_node": "response_node",
            "human_review_node": "human_review_node",
        },
    )
    builder.add_conditional_edges(
        "database_node",
        route_after_agent_node,
        {
            "supervisor_node": "supervisor_node",
            "response_node": "response_node",
            "human_review_node": "human_review_node",
        },
    )
    builder.add_conditional_edges(
        "relationship_node",
        route_after_agent_node,
        {
            "supervisor_node": "supervisor_node",
            "response_node": "response_node",
            "human_review_node": "human_review_node",
        },
    )

    # Verification conditional routing
    builder.add_conditional_edges(
        "verification_node",
        route_from_verification,
        {
            "response_node": "response_node",
            "supervisor_node": "supervisor_node",
            "human_review_node": "human_review_node",
        },
    )

    # Human review strictly routes back to supervisor
    builder.add_conditional_edges(
        "human_review_node",
        route_from_human_review,
        {
            "supervisor_node": "supervisor_node",
        },
    )

    # Response completes the graph
    builder.add_edge("response_node", END)

    return builder.compile(checkpointer=memory)


def run_workflow(
    user_query: str,
    thread_id: str = "default_thread",
    checkpointer: Optional[MemorySaver] = None,
    graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Helper function to execute the compiled graph on a query.
    """
    app = graph or build_government_orders_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "user_query": user_query,
        "intent": None,
        "identified_go_numbers": [],
        "discovered_go_numbers": [],
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

    return app.invoke(initial_state, config=config)
