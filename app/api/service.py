"""
Workflow Service Layer coordinating the shared LangGraph application and MemorySaver.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.config import DATASET_DIR
from app.agents.state import AgentState
from app.agents.workflow.graph_builder import WorkflowNodes, build_government_orders_graph
from app.agents.specialized.search_agent import (
    RetrievalMode,
    SearchAgent,
    SearchAgentRequest,
    SearchAgentResponse,
)
from app.agents.llm.ollama_client import OllamaClient
from app.database.connection import SessionLocal
from app.services.qdrant_service import QdrantService
from app.api.schemas import (
    HealthResponse,
    QueryResponse,
    RelationshipItemSchema,
    SearchRequest,
    SearchResponse,
    SearchResultItemSchema,
    ThreadStatusResponse,
)
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _is_safe_filename(filename: Optional[str]) -> bool:
    """
    Strictly validate that a filename is a plain filename without directory components.
    Rejects:
    - None or empty/whitespace
    - Any forward slash '/' or backslash '\\'
    - Directory traversal sequences '..'
    - Absolute paths (drive letters like 'C:' or leading slashes)
    - Any file not ending with '.pdf' (case-insensitive)
    Never silently sanitizes using Path.name.
    """
    if not filename or not isinstance(filename, str):
        return False
    fn = filename.strip()
    if not fn:
        return False
    if "/" in fn or "\\" in fn:
        return False
    if ".." in fn:
        return False
    if ":" in fn:  # Drive letters like C:
        return False
    if Path(fn).is_absolute():
        return False
    if not fn.lower().endswith(".pdf"):
        return False
    return True


class WorkflowService:
    """
    Singleton service managing the shared LangGraph state machine, in-memory checkpoints,
    and dispatch of query, interruption, and resume operations.
    """

    _instance: Optional["WorkflowService"] = None

    def __init__(
        self,
        checkpointer: Optional[MemorySaver] = None,
        search_agent: Optional[SearchAgent] = None,
    ) -> None:
        self.checkpointer = checkpointer or MemorySaver()
        self.search_agent = search_agent or SearchAgent()
        self.workflow_nodes = WorkflowNodes(search_agent=self.search_agent)
        self.graph = build_government_orders_graph(
            checkpointer=self.checkpointer,
            nodes=self.workflow_nodes,
        )
        logger.info("WorkflowService initialized with shared LangGraph application and MemorySaver.")

    @classmethod
    def get_instance(cls) -> "WorkflowService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for isolated test environments."""
        cls._instance = None

    def execute_query(
        self,
        query: str,
        thread_id: Optional[str] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
    ) -> QueryResponse:
        """
        Execute a user query through the shared compiled LangGraph workflow.
        """
        tid = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": tid}}

        initial_state: AgentState = {
            "user_query": query,
            "intent": None,
            "identified_go_numbers": [],
            "discovered_go_numbers": [],
            "search_parameters": search_parameters,
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

        # Invoke the graph
        res = self.graph.invoke(initial_state, config=config)

        # Inspect current state checkpoint
        state_snapshot = self.graph.get_state(config)
        is_interrupted = bool(state_snapshot.next and "human_review_node" in state_snapshot.next)

        if is_interrupted:
            prompt_data = None
            if hasattr(state_snapshot, "tasks") and state_snapshot.tasks:
                for task in state_snapshot.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        prompt_data = task.interrupts[0].value
                        break

            return QueryResponse(
                thread_id=tid,
                workflow_status="WAITING_FOR_HUMAN_REVIEW",
                final_answer=None,
                verification_status=None,
                confidence=res.get("confidence"),
                evidence_sufficient=False,
                legal_effect_verified=False,
                intent=str(res.get("intent")) if res.get("intent") else None,
                identified_go_numbers=res.get("identified_go_numbers", []),
                discovered_go_numbers=res.get("discovered_go_numbers", []),
                human_review_required=True,
                human_review_prompt=prompt_data,
                execution_trace_count=len(res.get("execution_trace", [])),
                errors=res.get("errors", []),
            )

        return self._format_response(tid, res)

    def resume_thread(self, thread_id: str, feedback: str) -> QueryResponse:
        """
        Resume a thread paused at human_review_node using Command(resume=feedback).
        """
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.graph.get_state(config)

        if not state_snapshot.values:
            raise KeyError(f"Thread '{thread_id}' was not found in checkpoint memory.")

        if not (state_snapshot.next and "human_review_node" in state_snapshot.next):
            raise ValueError(
                f"Thread '{thread_id}' is not in a paused human review state (next={state_snapshot.next})."
            )

        # Resume the paused thread
        res = self.graph.invoke(Command(resume=feedback), config=config)

        # Check if paused again or completed
        post_snapshot = self.graph.get_state(config)
        is_still_interrupted = bool(post_snapshot.next and "human_review_node" in post_snapshot.next)

        if is_still_interrupted:
            prompt_data = None
            if hasattr(post_snapshot, "tasks") and post_snapshot.tasks:
                for task in post_snapshot.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        prompt_data = task.interrupts[0].value
                        break
            return QueryResponse(
                thread_id=thread_id,
                workflow_status="WAITING_FOR_HUMAN_REVIEW",
                final_answer=None,
                verification_status=None,
                confidence=res.get("confidence"),
                evidence_sufficient=False,
                legal_effect_verified=False,
                intent=str(res.get("intent")) if res.get("intent") else None,
                identified_go_numbers=res.get("identified_go_numbers", []),
                discovered_go_numbers=res.get("discovered_go_numbers", []),
                human_review_required=True,
                human_review_prompt=prompt_data,
                execution_trace_count=len(res.get("execution_trace", [])),
                errors=res.get("errors", []),
            )

        return self._format_response(thread_id, res)

    def get_thread_status(self, thread_id: str) -> ThreadStatusResponse:
        """
        Inspect the status of a specific thread in checkpoint memory.
        """
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.graph.get_state(config)

        if not state_snapshot.values:
            return ThreadStatusResponse(
                thread_id=thread_id,
                workflow_status="NOT_FOUND",
                next_nodes=[],
                human_review_required=False,
                human_review_prompt=None,
                is_persisted_across_restarts=False,
            )

        next_nodes = list(state_snapshot.next) if state_snapshot.next else []
        is_paused = "human_review_node" in next_nodes

        prompt_data = None
        if is_paused and hasattr(state_snapshot, "tasks") and state_snapshot.tasks:
            for task in state_snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    prompt_data = task.interrupts[0].value
                    break

        status = "WAITING_FOR_HUMAN_REVIEW" if is_paused else ("COMPLETED" if not next_nodes else "RUNNING")

        return ThreadStatusResponse(
            thread_id=thread_id,
            workflow_status=status,
            next_nodes=next_nodes,
            human_review_required=is_paused,
            human_review_prompt=prompt_data,
            is_persisted_across_restarts=False,
        )

    def check_health(self) -> HealthResponse:
        """
        Check health and liveness of dependent services without exposing secrets.
        """
        ollama_ok = False
        try:
            ollama_client = OllamaClient()
            ollama_ok = bool(ollama_client.is_server_running())
        except Exception:
            ollama_ok = False

        postgres_ok = False
        try:
            with SessionLocal() as session:
                count = session.execute(text("SELECT count(*) FROM government_orders")).scalar()
                postgres_ok = count is not None
        except Exception:
            postgres_ok = False

        qdrant_ok = False
        try:
            qdrant = QdrantService()
            qdrant_ok = qdrant.count_active_points() >= 0
        except Exception:
            qdrant_ok = False

        return HealthResponse(
            status="healthy",
            service="kerala-government-orders-multi-agent-system",
            version="1.0.0",
            ollama_online=ollama_ok,
            postgres_online=postgres_ok,
            qdrant_online=qdrant_ok,
        )

    def search_documents(self, request: SearchRequest) -> SearchResponse:
        """
        Execute structured search for Government Orders reusing the shared SearchAgent.
        Supports HYBRID, KEYWORD, and SEMANTIC modes, department filtering, and exact GO lookup.
        """
        try:
            mode = RetrievalMode(request.retrieval_mode.upper())
            agent_req = SearchAgentRequest(
                query=request.query,
                retrieval_mode=mode,
                limit=request.limit,
                department_filter=request.department,
                rrf_k=60,
            )
            agent_resp: SearchAgentResponse = self.search_agent.execute_search(agent_req)
        except Exception as exc:
            logger.exception("Error executing search in WorkflowService: %s", exc)
            return SearchResponse(
                status="ERROR",
                query=request.query,
                returned_count=0,
                limit_applied=request.limit,
                has_more=False,
                is_exhaustive=False,
                results=[],
                disclaimer="Document relationships and search extractions are assistive; verify official gazetted status.",
            )

        results: List[SearchResultItemSchema] = []
        for r in agent_resp.results:
            results.append(self._map_search_result_item(r))

        # Rule 6: has_more=True only when additional distinct orders actually exist beyond returned limit
        has_more = bool(agent_resp.candidates_retained > len(results))

        # Rule 6: is_exhaustive=False when retrieval is incomplete or ceiling reached
        is_exhaustive = bool(agent_resp.is_exhaustive and not agent_resp.candidate_ceiling_reached)

        return SearchResponse(
            status=agent_resp.status,
            query=agent_resp.query,
            returned_count=len(results),
            limit_applied=request.limit,
            has_more=has_more,
            is_exhaustive=is_exhaustive,
            results=results,
            disclaimer="Document relationships and search extractions are assistive; verify official gazetted status.",
        )

    def _map_search_result_item(
        self,
        item: Any,
        default_source: Optional[str] = None,
    ) -> SearchResultItemSchema:
        """Map SearchResultItem, OrderMetadataItem, or dictionary to SearchResultItemSchema with strict disk-verified PDF status."""
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = item
        else:
            data = getattr(item, "__dict__", {})

        go_id = data.get("go_id") if data.get("go_id") is not None else data.get("id")
        source_fn = data.get("source_filename")

        has_pdf = False
        pdf_url = None

        # Rule 5: Strict PDF filename validation; never silently sanitize via Path.name
        if _is_safe_filename(source_fn):
            clean_fn = source_fn.strip()
            try:
                dataset_base = DATASET_DIR.resolve()
                target_path = (dataset_base / clean_fn).resolve()
                if target_path.is_relative_to(dataset_base) and target_path.is_file():
                    has_pdf = True
                    if go_id is not None:
                        pdf_url = f"/api/v1/documents/{go_id}/pdf"
            except Exception:
                has_pdf = False
                pdf_url = None

        rel_score = data.get("relevance_score")
        if rel_score is None:
            if data.get("rrf_score") is not None:
                rel_score = float(data["rrf_score"])
            elif data.get("semantic_score") is not None:
                rel_score = float(data["semantic_score"])
            elif data.get("keyword_rank") is not None:
                rel_score = float(1.0 / (60 + int(data["keyword_rank"])))

        matched_sources = data.get("matched_sources") or ([default_source] if default_source else [])

        raw_issue_date = data.get("issue_date")
        formatted_issue_date = str(raw_issue_date) if raw_issue_date is not None else None

        text_snippet = data.get("text_snippet") or data.get("text_preview")

        return SearchResultItemSchema(
            go_id=go_id,
            go_number=str(data.get("go_number") or ""),
            department=data.get("department"),
            issue_date=formatted_issue_date,
            source_filename=str(source_fn or ""),
            text_snippet=text_snippet,
            relevance_score=rel_score,
            matched_sources=matched_sources,
            has_pdf=has_pdf,
            pdf_url=pdf_url,
        )

    def _format_response(self, thread_id: str, state: Dict[str, Any]) -> QueryResponse:
        """Format final completed AgentState into standard QueryResponse."""
        intent = state.get("intent")
        intent_str = intent.value if hasattr(intent, "value") else (str(intent) if intent else None)

        formatted_search_results: List[SearchResultItemSchema] = []
        for raw_item in state.get("search_results", []):
            try:
                formatted_search_results.append(self._map_search_result_item(raw_item))
            except Exception as e:
                logger.warning("Failed to map search result item in QueryResponse: %s", e)

        # Map document_analysis if search_results is empty (e.g. exact GO_LOOKUP)
        if not formatted_search_results:
            for raw_doc in state.get("document_analysis", []):
                if isinstance(raw_doc, dict) and raw_doc.get("status") != "NOT_FOUND" and raw_doc.get("go_number"):
                    try:
                        formatted_search_results.append(
                            self._map_search_result_item(raw_doc, default_source="DATABASE")
                        )
                    except Exception as e:
                        logger.warning("Failed to map document analysis item in QueryResponse: %s", e)

        formatted_relationships: List[RelationshipItemSchema] = []
        for raw_rel in state.get("relationship_results", []):
            try:
                if isinstance(raw_rel, dict) and "id" in raw_rel and "direction" in raw_rel:
                    formatted_relationships.append(RelationshipItemSchema(**raw_rel))
            except Exception as e:
                logger.warning("Failed to map relationship item in QueryResponse: %s", e)

        return QueryResponse(
            thread_id=thread_id,
            workflow_status=state.get("workflow_status", "COMPLETED"),
            final_answer=state.get("final_answer"),
            verification_status=state.get("verification_status"),
            confidence=state.get("confidence"),
            evidence_sufficient=bool(state.get("evidence_sufficient", False)),
            legal_effect_verified=False,  # strictly False invariant
            intent=intent_str,
            identified_go_numbers=state.get("identified_go_numbers", []),
            discovered_go_numbers=state.get("discovered_go_numbers", []),
            human_review_required=bool(state.get("human_review_required", False)),
            human_review_prompt=None,
            execution_trace_count=len(state.get("execution_trace", [])),
            errors=state.get("errors", []),
            search_results=formatted_search_results,
            relationships=formatted_relationships,
        )

