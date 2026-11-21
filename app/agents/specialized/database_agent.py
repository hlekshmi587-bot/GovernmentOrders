"""
Database Agent Implementation for Kerala Government Orders System.

Specialized agent coordinating PostgreSQL database lookups via DatabaseTool.
Supports exact matching, partial matching, ambiguity detection, metadata retrieval,
references inspection, and bidirectional relationship retrieval with safe parameter
validation, discovered identifier extraction, complete provenance, defensive validation
of tool outputs, and error sanitization.
"""

from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.tools.database_tool import DatabaseTool

logger = logging.getLogger(__name__)


class DatabaseOperation(str, Enum):
    """Supported operations for DatabaseAgent."""
    LOOKUP_ORDER = "LOOKUP_ORDER"
    GET_METADATA = "GET_METADATA"
    GET_REFERENCES = "GET_REFERENCES"
    GET_RELATIONSHIPS = "GET_RELATIONSHIPS"
    LIST_ALL_ORDERS = "LIST_ALL_ORDERS"


class RelationshipDirection(str, Enum):
    """Direction for relationship queries."""
    INCOMING = "INCOMING"
    OUTGOING = "OUTGOING"
    ALL = "ALL"


class AgentResponseStatus(str, Enum):
    """Standardized response status for DatabaseAgent."""
    SUCCESS = "SUCCESS"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    EMPTY_INPUT = "EMPTY_INPUT"
    ERROR = "ERROR"


class DecisionSource(str, Enum):
    """Decision provenance source for audit traces."""
    DETERMINISTIC_RULES = "DETERMINISTIC_RULES"
    LLM = "LLM"
    HEURISTIC_FALLBACK = "HEURISTIC_FALLBACK"


class DatabaseAgentRequest(BaseModel):
    """
    Validated request contract for DatabaseAgent.
    Enforces strict typing, parameter allowlists, and bounds.
    """
    model_config = ConfigDict(extra="forbid")

    operation: DatabaseOperation = Field(
        default=DatabaseOperation.LOOKUP_ORDER,
        description="Database operation to perform"
    )
    go_number: Optional[str] = Field(
        default=None,
        description="Government Order number or query identifier to look up"
    )
    include_full_text: bool = Field(
        default=False,
        description="Whether to include full extracted text (default False to conserve bandwidth/memory)"
    )
    text_preview_chars: int = Field(
        default=500,
        ge=0,
        le=50000,
        description="Number of characters of extracted text to include in preview"
    )
    direction: RelationshipDirection = Field(
        default=RelationshipDirection.ALL,
        description="Relationship direction: INCOMING, OUTGOING, or ALL"
    )

    @field_validator("go_number")
    @classmethod
    def validate_go_number(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("GO number cannot be empty or whitespace.")
        return v.strip()

    @field_validator("include_full_text", mode="before")
    @classmethod
    def validate_include_full_text(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        raise ValueError(f"include_full_text must be a boolean, got {type(v).__name__}")

    @field_validator("text_preview_chars", mode="before")
    @classmethod
    def validate_text_preview_chars(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"text_preview_chars must be an integer, got {type(v).__name__}")
        if v < 0 or v > 50000:
            raise ValueError(f"text_preview_chars must be between 0 and 50000 (got {v})")
        return v

    @model_validator(mode="after")
    def validate_operation_requirements(self) -> "DatabaseAgentRequest":
        requires_go_number = {
            DatabaseOperation.LOOKUP_ORDER,
            DatabaseOperation.GET_METADATA,
            DatabaseOperation.GET_REFERENCES,
            DatabaseOperation.GET_RELATIONSHIPS,
        }
        if self.operation in requires_go_number and not self.go_number:
            raise ValueError(f"Operation '{self.operation.value}' requires a non-empty 'go_number'.")
        return self


class OrderMetadataItem(BaseModel):
    """Structured representation of a Government Order database record."""
    id: int
    go_number: str
    issue_date: Optional[str] = None
    department: Optional[str] = None
    source_filename: str
    verification_status: Optional[str] = None
    extracted_text_length: int = 0
    text_preview: str = ""
    extracted_text: Optional[str] = None


class ReferenceItem(BaseModel):
    """Structured representation of a numbered reference paper ('Read' section)."""
    id: Optional[int] = None
    ordinal: int
    referenced_go_number: str
    reference_date: Optional[str] = None
    page_number: Optional[int] = None
    evidence_text: Optional[str] = None


class DatabaseRelationshipItem(BaseModel):
    """Structured representation of a Government Order relationship from the database."""
    id: Optional[int] = None
    direction: str  # "INCOMING" or "OUTGOING"
    source_go_id: Optional[int] = None
    source_go_number: str
    source_issue_date: Optional[str] = None
    source_department: Optional[str] = None
    target_go_number: str
    relationship_type: str
    matched_phrase: Optional[str] = None
    evidence_text: Optional[str] = None
    page_number: Optional[int] = None
    confidence: Optional[str] = None
    review_required: bool = False
    legal_effect_verified: bool = False  # Explicit invariant: DB match != legal verification


class DatabaseAgentResponse(BaseModel):
    """Structured output contract returned by DatabaseAgent."""
    status: AgentResponseStatus
    operation: DatabaseOperation
    query_identifier: Optional[str] = None
    match_type: Optional[str] = None  # "EXACT", "PARTIAL", or None
    ambiguous_candidates: List[str] = Field(default_factory=list)
    order_data: Optional[OrderMetadataItem] = None
    references: List[ReferenceItem] = Field(default_factory=list)
    relationships: List[DatabaseRelationshipItem] = Field(default_factory=list)
    discovered_go_numbers: List[str] = Field(default_factory=list)
    total_records: int = 0
    is_complete: bool = True  # Set to False if tool returned malformed records or unexpected states
    malformed_records_discarded: int = 0
    errors: List[str] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    legal_effect_verified: bool = False  # Retrieval invariant


class DatabaseAgent:
    """
    Specialized agent coordinating safe, deterministic queries against PostgreSQL
    via DatabaseTool.
    """

    def __init__(self, database_tool: Optional[DatabaseTool] = None) -> None:
        self.db_tool = database_tool or DatabaseTool()

    def execute(self, request_data: Any) -> DatabaseAgentResponse:
        """
        Main entry point for executing database operations.
        Validates input against DatabaseAgentRequest and routes to appropriate handler.
        """
        # Step 1: Validate request contract
        try:
            if isinstance(request_data, DatabaseAgentRequest):
                req = request_data
            elif isinstance(request_data, dict):
                req = DatabaseAgentRequest(**request_data)
            else:
                raise ValueError(f"Invalid request data type: {type(request_data).__name__}")
        except Exception as exc:
            logger.warning("DatabaseAgentRequest validation failed: %s", exc)
            query_str = None
            if isinstance(request_data, dict):
                query_str = request_data.get("go_number")

            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=DatabaseOperation.LOOKUP_ORDER,
                query_identifier=query_str,
                is_complete=False,
                errors=[f"Request validation error: {exc}"],
                execution_trace=[{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "database_agent",
                    "action": "validate_request",
                    "reasoning": "Validate input contract parameters and operational bounds",
                    "tool_called": "none",
                    "tool_input": None,
                    "tool_output_summary": f"Validation failed: {exc}",
                    "decision": "terminate_with_error",
                    "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
                }],
            )

        # Step 2: Route by requested operation
        if req.operation == DatabaseOperation.LOOKUP_ORDER:
            return self._lookup_order(req)
        elif req.operation == DatabaseOperation.GET_METADATA:
            return self._get_metadata(req)
        elif req.operation == DatabaseOperation.GET_REFERENCES:
            return self._get_references(req)
        elif req.operation == DatabaseOperation.GET_RELATIONSHIPS:
            return self._get_relationships(req)
        elif req.operation == DatabaseOperation.LIST_ALL_ORDERS:
            return self._list_all_orders(req)
        else:
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=[f"Unsupported database operation: {req.operation}"],
            )

    def _lookup_order(self, req: DatabaseAgentRequest) -> DatabaseAgentResponse:
        """Execute exact/partial Government Order lookup with metadata and optional text."""
        assert req.go_number is not None
        trace: List[Dict[str, Any]] = []

        try:
            tool_res = self.db_tool.get_order_by_number(
                go_number=req.go_number,
                include_full_text=req.include_full_text,
                text_preview_chars=req.text_preview_chars,
            )
        except Exception as exc:
            logger.error("DatabaseTool exception in lookup_order: %s", exc, exc_info=True)
            return self._error_response(
                req.operation, req.go_number,
                "Database operation could not be completed due to an internal error.",
                action="get_order_by_number", trace=trace
            )

        status_raw = tool_res.get("status", "ERROR") if isinstance(tool_res, dict) else "ERROR"
        summary = f"Status={status_raw}, found={tool_res.get('found', False) if isinstance(tool_res, dict) else False}"
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "database_agent",
            "action": "get_order_by_number",
            "reasoning": f"Lookup Government Order for query '{req.go_number}'",
            "tool_called": "database_tool",
            "tool_input": {
                "go_number": req.go_number,
                "include_full_text": req.include_full_text,
                "text_preview_chars": req.text_preview_chars,
            },
            "tool_output_summary": summary,
            "decision": "evaluate_order_resolution",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        # Step 1: Process ERROR explicitly before checking found
        if status_raw == "ERROR":
            err_msg = tool_res.get("error") if isinstance(tool_res, dict) and tool_res.get("error") else "Database operation could not be completed due to an internal error."
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=[err_msg],
                execution_trace=trace,
            )

        # Step 2: Process AMBIGUOUS_MATCH explicitly
        if status_raw == "AMBIGUOUS_MATCH":
            candidates = tool_res.get("candidates", []) if isinstance(tool_res, dict) else []
            return DatabaseAgentResponse(
                status=AgentResponseStatus.AMBIGUOUS_MATCH,
                operation=req.operation,
                query_identifier=req.go_number,
                match_type=None,
                ambiguous_candidates=candidates,
                total_records=len(candidates),
                errors=[tool_res.get("error") or f"Query '{req.go_number}' matched multiple orders."],
                execution_trace=trace,
            )

        # Step 3: Process NOT_FOUND explicitly
        if status_raw == "NOT_FOUND" or (isinstance(tool_res, dict) and tool_res.get("found") is False):
            return DatabaseAgentResponse(
                status=AgentResponseStatus.NOT_FOUND,
                operation=req.operation,
                query_identifier=req.go_number,
                match_type=None,
                total_records=0,
                errors=[tool_res.get("error") if isinstance(tool_res, dict) and tool_res.get("error") else f"No Government Order found matching '{req.go_number}'."],
                execution_trace=trace,
            )

        # Step 4: Reject unknown/unexpected tool statuses safely as ERROR
        if status_raw != "SUCCESS":
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=[f"Database returned unknown operational status '{status_raw}'."],
                execution_trace=trace,
            )

        # Step 5: Validate successful order data contains required fields 'id' and 'go_number'
        raw_data = tool_res.get("data")
        if not isinstance(raw_data, dict) or "id" not in raw_data or "go_number" not in raw_data:
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=["Database returned an invalid or malformed order record."],
                execution_trace=trace,
            )

        order_item = OrderMetadataItem(
            id=raw_data["id"],
            go_number=raw_data["go_number"],
            issue_date=raw_data.get("issue_date"),
            department=raw_data.get("department"),
            source_filename=raw_data.get("source_filename", ""),
            verification_status=raw_data.get("verification_status"),
            extracted_text_length=raw_data.get("extracted_text_length", 0),
            text_preview=raw_data.get("text_preview", ""),
            extracted_text=raw_data.get("extracted_text") if req.include_full_text else None,
        )

        discovered = [order_item.go_number]

        return DatabaseAgentResponse(
            status=AgentResponseStatus.SUCCESS,
            operation=req.operation,
            query_identifier=req.go_number,
            match_type=tool_res.get("match_type"),
            candidates=tool_res.get("candidates", []),
            order_data=order_item,
            discovered_go_numbers=discovered,
            total_records=1,
            is_complete=True,
            errors=[],
            execution_trace=trace,
        )

    def _get_metadata(self, req: DatabaseAgentRequest) -> DatabaseAgentResponse:
        """Retrieve metadata only without full document text."""
        req_copy = req.model_copy(update={"include_full_text": False, "text_preview_chars": 0})
        res = self._lookup_order(req_copy)
        res.operation = DatabaseOperation.GET_METADATA
        return res

    def _get_references(self, req: DatabaseAgentRequest) -> DatabaseAgentResponse:
        """Retrieve numbered references ('Read' section papers) and extract discovered identifiers."""
        assert req.go_number is not None
        trace: List[Dict[str, Any]] = []

        try:
            tool_res = self.db_tool.get_references_for_order(go_number=req.go_number)
        except Exception as exc:
            logger.error("DatabaseTool exception in get_references_for_order: %s", exc, exc_info=True)
            return self._error_response(
                req.operation, req.go_number,
                "Database operation could not be completed due to an internal error.",
                action="get_references_for_order", trace=trace
            )

        status_raw = tool_res.get("status", "ERROR") if isinstance(tool_res, dict) else "ERROR"
        summary = f"Status={status_raw}, total_references={tool_res.get('total_references', 0) if isinstance(tool_res, dict) else 0}"
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "database_agent",
            "action": "get_references_for_order",
            "reasoning": f"Retrieve references for GO '{req.go_number}'",
            "tool_called": "database_tool",
            "tool_input": {"go_number": req.go_number},
            "tool_output_summary": summary,
            "decision": "evaluate_references_result",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        if status_raw == "ERROR":
            err_msg = tool_res.get("error") if isinstance(tool_res, dict) and tool_res.get("error") else "Database operation could not be completed due to an internal error."
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=[err_msg],
                execution_trace=trace,
            )

        if status_raw == "AMBIGUOUS_MATCH":
            candidates = tool_res.get("candidates", []) if isinstance(tool_res, dict) else []
            return DatabaseAgentResponse(
                status=AgentResponseStatus.AMBIGUOUS_MATCH,
                operation=req.operation,
                query_identifier=req.go_number,
                ambiguous_candidates=candidates,
                total_records=len(candidates),
                errors=[tool_res.get("error") or f"Query '{req.go_number}' matched multiple orders."],
                execution_trace=trace,
            )

        if status_raw == "NOT_FOUND":
            return DatabaseAgentResponse(
                status=AgentResponseStatus.NOT_FOUND,
                operation=req.operation,
                query_identifier=req.go_number,
                total_records=0,
                errors=[tool_res.get("error") if isinstance(tool_res, dict) else f"No Government Order found matching '{req.go_number}'."],
                execution_trace=trace,
            )

        if status_raw != "SUCCESS":
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=[f"Database returned unknown operational status '{status_raw}'."],
                execution_trace=trace,
            )

        raw_refs = tool_res.get("references", [])
        if not isinstance(raw_refs, list):
            raw_refs = []

        ref_items: List[ReferenceItem] = []
        discovered_gos: List[str] = []
        malformed_count = 0
        errors: List[str] = []

        # Validate individual records before processing; preserve valid records
        for r in raw_refs:
            if not isinstance(r, dict) or "ordinal" not in r or "referenced_go_number" not in r:
                malformed_count += 1
                continue

            ref_num = str(r.get("referenced_go_number", "")).strip()
            ref_items.append(
                ReferenceItem(
                    id=r.get("id"),
                    ordinal=r.get("ordinal", 0),
                    referenced_go_number=ref_num,
                    reference_date=r.get("reference_date"),
                    page_number=r.get("page_number"),
                    evidence_text=r.get("evidence_text"),
                )
            )
            if ref_num and ref_num not in discovered_gos:
                discovered_gos.append(ref_num)

        is_complete = True
        if malformed_count > 0:
            is_complete = False
            errors.append(f"Detected and skipped {malformed_count} malformed reference record(s) in tool output.")

        return DatabaseAgentResponse(
            status=AgentResponseStatus.SUCCESS,
            operation=req.operation,
            query_identifier=req.go_number,
            match_type=tool_res.get("match_type"),
            references=ref_items,
            discovered_go_numbers=discovered_gos,
            total_records=len(ref_items),
            is_complete=is_complete,
            malformed_records_discarded=malformed_count,
            errors=errors,
            execution_trace=trace,
        )

    def _get_relationships(self, req: DatabaseAgentRequest) -> DatabaseAgentResponse:
        """
        Retrieve incoming, outgoing, or bidirectional relationships from PostgreSQL.
        Dispatches directly to:
        - OUTGOING -> get_outgoing_relationships
        - INCOMING -> get_incoming_relationships
        - ALL -> get_relationships_for_order
        """
        assert req.go_number is not None
        trace: List[Dict[str, Any]] = []
        direction_str = req.direction.value

        try:
            if req.direction == RelationshipDirection.OUTGOING:
                tool_res = self.db_tool.get_outgoing_relationships(go_number=req.go_number)
                tool_action = "get_outgoing_relationships"
            elif req.direction == RelationshipDirection.INCOMING:
                tool_res = self.db_tool.get_incoming_relationships(go_number=req.go_number)
                tool_action = "get_incoming_relationships"
            else:
                tool_res = self.db_tool.get_relationships_for_order(
                    go_number=req.go_number,
                    direction="ALL",
                )
                tool_action = "get_relationships_for_order"
        except Exception as exc:
            logger.error("DatabaseTool exception in %s: %s", req.operation, exc, exc_info=True)
            return self._error_response(
                req.operation, req.go_number,
                "Database operation could not be completed due to an internal error.",
                action=f"get_{direction_str.lower()}_relationships", trace=trace
            )

        status_raw = tool_res.get("status", "ERROR") if isinstance(tool_res, dict) else "ERROR"
        summary = f"Status={status_raw}, direction={direction_str}, total_relationships={tool_res.get('total_relationships', 0) if isinstance(tool_res, dict) else 0}"
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "database_agent",
            "action": tool_action,
            "reasoning": f"Retrieve {direction_str} relationships for GO '{req.go_number}'",
            "tool_called": "database_tool",
            "tool_input": {"go_number": req.go_number, "direction": direction_str},
            "tool_output_summary": summary,
            "decision": "evaluate_relationships_result",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        if status_raw == "ERROR":
            err_msg = tool_res.get("error") if isinstance(tool_res, dict) and tool_res.get("error") else "Database operation could not be completed due to an internal error."
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=[err_msg],
                execution_trace=trace,
            )

        if status_raw == "AMBIGUOUS_MATCH":
            candidates = tool_res.get("candidates", []) if isinstance(tool_res, dict) else []
            return DatabaseAgentResponse(
                status=AgentResponseStatus.AMBIGUOUS_MATCH,
                operation=req.operation,
                query_identifier=req.go_number,
                ambiguous_candidates=candidates,
                total_records=len(candidates),
                errors=[tool_res.get("error") or f"Query '{req.go_number}' matched multiple orders."],
                execution_trace=trace,
            )

        if status_raw == "NOT_FOUND":
            return DatabaseAgentResponse(
                status=AgentResponseStatus.NOT_FOUND,
                operation=req.operation,
                query_identifier=req.go_number,
                total_records=0,
                errors=[tool_res.get("error") if isinstance(tool_res, dict) else f"No Government Order found matching '{req.go_number}'."],
                execution_trace=trace,
            )

        if status_raw != "SUCCESS":
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                query_identifier=req.go_number,
                is_complete=False,
                errors=[f"Database returned unknown operational status '{status_raw}'."],
                execution_trace=trace,
            )

        raw_rels = tool_res.get("relationships", [])
        if not isinstance(raw_rels, list):
            raw_rels = []

        rel_items: List[DatabaseRelationshipItem] = []
        discovered_gos: List[str] = []
        malformed_count = 0
        errors: List[str] = []

        # Validate individual records before processing; preserve valid records
        for r in raw_rels:
            if not isinstance(r, dict) or "relationship_type" not in r:
                malformed_count += 1
                continue

            src_num = str(r.get("source_go_number", "")).strip()
            tgt_num = str(r.get("target_go_number") or r.get("referenced_go_number", "")).strip()

            rel_items.append(
                DatabaseRelationshipItem(
                    id=r.get("id"),
                    direction=r.get("direction", direction_str),
                    source_go_id=r.get("source_go_id"),
                    source_go_number=src_num,
                    source_issue_date=r.get("source_issue_date"),
                    source_department=r.get("source_department"),
                    target_go_number=tgt_num,
                    relationship_type=str(r.get("relationship_type", "")),
                    matched_phrase=r.get("matched_phrase"),
                    evidence_text=r.get("evidence_text"),
                    page_number=r.get("page_number"),
                    confidence=str(r.get("confidence", "HIGH")),
                    review_required=bool(r.get("review_required", False)),
                    legal_effect_verified=False,  # Invariant: DB match is not legal verification
                )
            )
            for num in [src_num, tgt_num]:
                if num and num != req.go_number and num not in discovered_gos:
                    discovered_gos.append(num)

        is_complete = True
        if malformed_count > 0:
            is_complete = False
            errors.append(f"Detected and skipped {malformed_count} malformed relationship record(s) in tool output.")

        return DatabaseAgentResponse(
            status=AgentResponseStatus.SUCCESS,
            operation=req.operation,
            query_identifier=req.go_number,
            match_type=tool_res.get("match_type"),
            relationships=rel_items,
            discovered_go_numbers=discovered_gos,
            total_records=len(rel_items),
            is_complete=is_complete,
            malformed_records_discarded=malformed_count,
            errors=errors,
            execution_trace=trace,
        )

    def _list_all_orders(self, req: DatabaseAgentRequest) -> DatabaseAgentResponse:
        """Retrieve full catalog list of registered Government Order numbers."""
        trace: List[Dict[str, Any]] = []

        try:
            tool_res = self.db_tool.list_all_go_numbers()
        except Exception as exc:
            logger.error("DatabaseTool exception in list_all_go_numbers: %s", exc, exc_info=True)
            return self._error_response(
                req.operation, None,
                "Database operation could not be completed due to an internal error.",
                action="list_all_go_numbers", trace=trace
            )

        status_raw = tool_res.get("status", "ERROR") if isinstance(tool_res, dict) else "ERROR"
        numbers = tool_res.get("go_numbers", []) if isinstance(tool_res, dict) else []
        if not isinstance(numbers, list):
            numbers = []

        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "database_agent",
            "action": "list_all_go_numbers",
            "reasoning": "Retrieve list of all registered Government Order numbers",
            "tool_called": "database_tool",
            "tool_input": {},
            "tool_output_summary": f"Status={status_raw}, total_count={len(numbers)}",
            "decision": "evaluate_catalog_result",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        if status_raw == "ERROR" or status_raw != "SUCCESS":
            return DatabaseAgentResponse(
                status=AgentResponseStatus.ERROR,
                operation=req.operation,
                is_complete=False,
                errors=["Database operation could not be completed due to an internal error."],
                execution_trace=trace,
            )

        return DatabaseAgentResponse(
            status=AgentResponseStatus.SUCCESS,
            operation=req.operation,
            discovered_go_numbers=numbers,
            total_records=len(numbers),
            is_complete=True,
            errors=[],
            execution_trace=trace,
        )

    def _error_response(
        self,
        operation: DatabaseOperation,
        query_identifier: Optional[str],
        error_message: str,
        action: str,
        trace: List[Dict[str, Any]],
    ) -> DatabaseAgentResponse:
        """Helper to construct safe, sanitized error responses."""
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "database_agent",
            "action": action,
            "reasoning": "Execute database query",
            "tool_called": "database_tool",
            "tool_input": {"query": query_identifier},
            "tool_output_summary": f"Failed with sanitized message: {error_message}",
            "decision": "terminate_with_error",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })
        return DatabaseAgentResponse(
            status=AgentResponseStatus.ERROR,
            operation=operation,
            query_identifier=query_identifier,
            is_complete=False,
            errors=[error_message],
            execution_trace=trace,
        )
