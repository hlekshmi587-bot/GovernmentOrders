"""
Relationship Investigation Agent Implementation for Kerala Government Orders System.

Specialized agent coordinating relationship analysis between source and referenced
Government Orders. Uses RelationshipTool for rule-based signal and reference extraction,
and DatabaseTool for source document lookup and cross-examination.
Enforces conservative conflict detection, preserves evidence and page numbers,
supports boundary-aware pair matching, handles malformed tool records with diagnostics,
and strictly marks all outputs as UNVERIFIED_CANDIDATE with legal_effect_verified=False.
"""

from datetime import datetime, timezone
from enum import Enum
import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.identifier_extraction import RelationshipType, ExtractionConfidence
from app.agents.tools.relationship_tool import RelationshipTool
from app.agents.tools.database_tool import DatabaseTool

logger = logging.getLogger(__name__)


def _is_target_go_match(target_query: str, candidate_text: str) -> bool:
    """
    Boundary-aware, punctuation-normalized matcher ensuring target numbers
    like '114/2025' match 'G.O.(Ms)No.114/2025/ITD' but strictly do NOT match
    'G.O.(Ms)No.2114/2025/ITD'.
    """
    if not target_query or not candidate_text:
        return False
    clean_target = target_query.strip().lower()
    clean_cand = candidate_text.strip().lower()
    if clean_target == clean_cand:
        return True

    # Normalize multiple whitespace
    clean_target = re.sub(r"\s+", " ", clean_target)
    clean_cand = re.sub(r"\s+", " ", clean_cand)

    escaped_target = re.escape(clean_target)
    escaped_cand = re.escape(clean_cand)

    # Digit boundary check: cannot be preceded or followed by digits
    pat_target_in_cand = rf"(?<!\d){escaped_target}(?!\d)"
    pat_cand_in_target = rf"(?<!\d){escaped_cand}(?!\d)"

    if re.search(pat_target_in_cand, clean_cand) or re.search(pat_cand_in_target, clean_target):
        return True

    # Core pattern extraction (e.g. 114/2025 or 114/2025/ITD)
    match_core = re.search(r"(\d+)\s*/\s*(\d{4})(?:\s*/\s*([a-z&.-]+))?", clean_target)
    if match_core:
        num = match_core.group(1)
        year = match_core.group(2)
        dept = match_core.group(3)
        if dept:
            core_regex = rf"(?<!\d){num}\s*/\s*{year}\s*/\s*{re.escape(dept)}(?!\d)"
        else:
            core_regex = rf"(?<!\d){num}\s*/\s*{year}(?!\d)"
        if re.search(core_regex, clean_cand):
            return True

    return False


class RelationshipInvestigationMode(str, Enum):
    """Supported investigation modes for RelationshipAgent."""
    INVESTIGATE_ORDER = "INVESTIGATE_ORDER"
    EXTRACT_FROM_TEXT = "EXTRACT_FROM_TEXT"
    INVESTIGATE_PAIR = "INVESTIGATE_PAIR"
    CROSS_EXAMINE_DATABASE = "CROSS_EXAMINE_DATABASE"


class RelationshipAgentStatus(str, Enum):
    """Standardized response status for RelationshipAgent."""
    SUCCESS = "SUCCESS"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    EMPTY_INPUT = "EMPTY_INPUT"
    NO_RELATIONSHIPS_FOUND = "NO_RELATIONSHIPS_FOUND"
    ERROR = "ERROR"


class DecisionSource(str, Enum):
    """Decision provenance source for audit traces."""
    DETERMINISTIC_RULES = "DETERMINISTIC_RULES"
    LLM = "LLM"
    HEURISTIC_FALLBACK = "HEURISTIC_FALLBACK"


class RelationshipAgentRequest(BaseModel):
    """
    Validated request contract for RelationshipAgent.
    Enforces strict typing and mode-specific requirements.
    """
    model_config = ConfigDict(extra="forbid")

    mode: RelationshipInvestigationMode = Field(
        default=RelationshipInvestigationMode.INVESTIGATE_ORDER,
        description="Investigation strategy: INVESTIGATE_ORDER, EXTRACT_FROM_TEXT, INVESTIGATE_PAIR, or CROSS_EXAMINE_DATABASE"
    )
    go_number: Optional[str] = Field(
        default=None,
        description="Source Government Order number to investigate"
    )
    text: Optional[str] = Field(
        default=None,
        description="Raw document text for extraction"
    )
    target_go_number: Optional[str] = Field(
        default=None,
        description="Optional target GO number when investigating a specific order pair"
    )
    min_confidence: Optional[ExtractionConfidence] = Field(
        default=None,
        description="Minimum confidence threshold: HIGH, MEDIUM, or LOW"
    )

    @field_validator("go_number", "target_go_number")
    @classmethod
    def validate_identifier(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("GO identifier cannot be empty or whitespace.")
        return v.strip()

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Text cannot be empty or whitespace.")
        return v.strip()

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "RelationshipAgentRequest":
        if self.mode == RelationshipInvestigationMode.EXTRACT_FROM_TEXT:
            if not self.text:
                raise ValueError("Mode 'EXTRACT_FROM_TEXT' requires non-empty 'text'.")
        elif self.mode in {
            RelationshipInvestigationMode.INVESTIGATE_ORDER,
            RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
        }:
            if not self.go_number:
                raise ValueError(f"Mode '{self.mode.value}' requires a non-empty 'go_number'.")
        elif self.mode == RelationshipInvestigationMode.INVESTIGATE_PAIR:
            if not self.target_go_number:
                raise ValueError("Mode 'INVESTIGATE_PAIR' requires a non-empty 'target_go_number'.")
            if not self.go_number and not self.text:
                raise ValueError("Mode 'INVESTIGATE_PAIR' requires either 'go_number' or 'text'.")
        return self


class ExtractedReferenceItem(BaseModel):
    """Structured representation of an extracted reference paper ('Read' section)."""
    ordinal: int
    go_number: str
    issue_date: Optional[str] = None
    page_number: Optional[int] = None
    evidence_text: Optional[str] = None


class RelationshipCandidate(BaseModel):
    """
    Extracted relationship candidate with complete evidence attributes.
    Always tagged as UNVERIFIED_CANDIDATE and legal_effect_verified=False.
    """
    relationship_type: RelationshipType
    source_go_number: Optional[str] = None
    referenced_go_numbers: List[str] = Field(default_factory=list)
    reference_ordinals: List[int] = Field(default_factory=list)
    matched_phrase: str
    evidence_text: Optional[str] = None
    page_number: Optional[int] = None
    confidence: str
    extraction_method: str
    review_required: bool = False
    verification_status: str = "UNVERIFIED_CANDIDATE"
    legal_effect_verified: bool = False  # Candidate signal != legal verification
    potential_conflict: bool = False
    conflict_notes: Optional[str] = None


class DiscrepancyReport(BaseModel):
    """Report generated during database cross-examination."""
    referenced_go_number: str
    category: str  # "UNRECORDED_IN_DATABASE", "MISSING_FROM_TEXT", "TYPE_MISMATCH"
    text_relationship_type: Optional[str] = None
    db_relationship_type: Optional[str] = None
    evidence_text: Optional[str] = None
    description: str


class RelationshipAgentResponse(BaseModel):
    """Structured output contract returned by RelationshipAgent."""
    status: RelationshipAgentStatus
    mode: RelationshipInvestigationMode
    source_go_number: Optional[str] = None
    target_go_number: Optional[str] = None
    total_references: int = 0
    references: List[ExtractedReferenceItem] = Field(default_factory=list)
    total_relationships: int = 0
    relationships: List[RelationshipCandidate] = Field(default_factory=list)
    unrecognized_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    discrepancies: List[DiscrepancyReport] = Field(default_factory=list)
    discovered_go_numbers: List[str] = Field(default_factory=list)
    human_review_required: bool = False
    human_review_reason: Optional[str] = None
    is_complete: bool = True  # Set to False if tool returned malformed records or failed sub-steps
    malformed_records_discarded: int = 0
    errors: List[str] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    legal_effect_verified: bool = False  # Fixed invariant across relationship agent


class RelationshipAgent:
    """
    Specialized agent investigating relationship signals, references, and conflicts
    across Kerala Government Orders.
    """

    # Lifecycle-cancelling types that potentially clash with ongoing-modification types
    TERMINATING_TYPES: Set[RelationshipType] = {
        RelationshipType.CANCELS,
        RelationshipType.REVOKES,
        RelationshipType.SUPERSEDES,
    }
    CONTINUING_TYPES: Set[RelationshipType] = {
        RelationshipType.AMENDS,
        RelationshipType.PARTIALLY_AMENDS,
        RelationshipType.MODIFIES,
        RelationshipType.EXTENDS,
        RelationshipType.READ_AS,
    }

    # Narrow, explicitly approved bidirectional complementary pairs based on Kerala administrative practice
    EXPLICIT_COMPLEMENTARY_PAIRS: Set[FrozenSet[RelationshipType]] = {
        frozenset({RelationshipType.AMENDS, RelationshipType.PARTIALLY_AMENDS}),
        frozenset({RelationshipType.AMENDS, RelationshipType.MODIFIES}),
        frozenset({RelationshipType.PARTIALLY_AMENDS, RelationshipType.MODIFIES}),
        frozenset({RelationshipType.MODIFIES, RelationshipType.EXTENDS}),
        frozenset({RelationshipType.CANCELS, RelationshipType.REVOKES}),
    }

    @classmethod
    def _is_complementary_type(cls, type_a: str, type_b: str) -> bool:
        """
        Check if two relationship type strings belong to the narrow, explicitly
        approved bidirectional complementary pairs.
        Both strings MUST be validated against the existing RelationshipType enum
        before evaluating exact matches or complementary pairs.
        Two identical unsupported strings can NEVER be considered a match.
        All unsupported or unapproved combinations return False and are flagged for human review.
        Never implies legal equivalence or verified legal effect.
        """
        a_norm = type_a.strip().upper()
        b_norm = type_b.strip().upper()
        try:
            enum_a = RelationshipType(a_norm)
            enum_b = RelationshipType(b_norm)
        except ValueError:
            # If either string is not one of the 8 valid RelationshipType enums,
            # it is unsupported and can NEVER match or be complementary!
            return False

        if enum_a == enum_b:
            return True

        pair = frozenset({enum_a, enum_b})
        return pair in cls.EXPLICIT_COMPLEMENTARY_PAIRS

    def __init__(
        self,
        relationship_tool: Optional[RelationshipTool] = None,
        database_tool: Optional[DatabaseTool] = None,
    ) -> None:
        self.db_tool = database_tool or DatabaseTool()
        self.rel_tool = relationship_tool or RelationshipTool(database_tool=self.db_tool)

    def execute(self, request_data: Any) -> RelationshipAgentResponse:
        """
        Main entry point for executing relationship investigation.
        Validates input against RelationshipAgentRequest and routes to appropriate mode.
        """
        # Step 1: Validate request contract
        try:
            if isinstance(request_data, RelationshipAgentRequest):
                req = request_data
            elif isinstance(request_data, dict):
                req = RelationshipAgentRequest(**request_data)
            else:
                raise ValueError(f"Invalid request data type: {type(request_data).__name__}")
        except Exception as exc:
            logger.warning("RelationshipAgentRequest validation failed: %s", exc)
            query_str = None
            if isinstance(request_data, dict):
                query_str = request_data.get("go_number")

            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.ERROR,
                mode=RelationshipInvestigationMode.INVESTIGATE_ORDER,
                source_go_number=query_str,
                is_complete=False,
                errors=[f"Request validation error: {exc}"],
                execution_trace=[{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "relationship_agent",
                    "action": "validate_request",
                    "reasoning": "Validate input contract parameters and operational mode",
                    "tool_called": "none",
                    "tool_input": None,
                    "tool_output_summary": f"Validation failed: {exc}",
                    "decision": "terminate_with_error",
                    "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
                }],
            )

        # Step 2: Route by requested mode
        if req.mode == RelationshipInvestigationMode.INVESTIGATE_ORDER:
            return self._investigate_order(req)
        elif req.mode == RelationshipInvestigationMode.EXTRACT_FROM_TEXT:
            return self._extract_from_text(req)
        elif req.mode == RelationshipInvestigationMode.INVESTIGATE_PAIR:
            return self._investigate_pair(req)
        elif req.mode == RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE:
            return self._cross_examine_database(req)
        else:
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.ERROR,
                mode=req.mode,
                source_go_number=req.go_number,
                is_complete=False,
                errors=[f"Unsupported investigation mode: {req.mode}"],
            )

    def _investigate_order(self, req: RelationshipAgentRequest) -> RelationshipAgentResponse:
        """Fetch document text via DatabaseTool and extract references and relationships."""
        assert req.go_number is not None
        trace: List[Dict[str, Any]] = []

        try:
            tool_res = self.rel_tool.extract_for_order(go_number=req.go_number)
        except Exception as exc:
            logger.error("RelationshipTool exception in extract_for_order: %s", exc, exc_info=True)
            return self._error_response(
                req.mode, req.go_number,
                "Relationship extraction operation could not be completed.",
                action="extract_for_order", trace=trace
            )

        status_raw = tool_res.get("status", "ERROR") if isinstance(tool_res, dict) else "ERROR"
        summary = (
            f"Status={status_raw}, total_references={tool_res.get('total_references', 0) if isinstance(tool_res, dict) else 0}, "
            f"total_relationships={tool_res.get('total_relationships', 0) if isinstance(tool_res, dict) else 0}"
        )
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "relationship_agent",
            "action": "extract_for_order",
            "reasoning": f"Extract references and relationships for GO '{req.go_number}'",
            "tool_called": "relationship_tool",
            "tool_input": {"go_number": req.go_number},
            "tool_output_summary": summary,
            "decision": "evaluate_extraction_results",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        if status_raw == "AMBIGUOUS_SOURCE_ORDER":
            candidates = tool_res.get("candidates", []) if isinstance(tool_res, dict) else []
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.AMBIGUOUS_MATCH,
                mode=req.mode,
                source_go_number=req.go_number,
                errors=[tool_res.get("error") if isinstance(tool_res, dict) else f"Source order '{req.go_number}' is ambiguous: {candidates}"],
                execution_trace=trace,
            )

        if status_raw == "ORDER_NOT_FOUND":
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.NOT_FOUND,
                mode=req.mode,
                source_go_number=req.go_number,
                errors=[tool_res.get("error") if isinstance(tool_res, dict) else f"Source order '{req.go_number}' was not found in the database."],
                execution_trace=trace,
            )

        if status_raw == "ERROR":
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.ERROR,
                mode=req.mode,
                source_go_number=req.go_number,
                is_complete=False,
                errors=["Relationship extraction operation could not be completed."],
                execution_trace=trace,
            )

        if status_raw != "SUCCESS":
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.ERROR,
                mode=req.mode,
                source_go_number=req.go_number,
                is_complete=False,
                errors=[f"Relationship tool returned unknown operational status '{status_raw}'."],
                execution_trace=trace,
            )

        return self._format_and_analyze_results(
            tool_res=tool_res,
            mode=req.mode,
            source_go_number=req.go_number,
            min_confidence=req.min_confidence,
            trace=trace,
        )

    def _extract_from_text(self, req: RelationshipAgentRequest) -> RelationshipAgentResponse:
        """Extract references and relationships directly from user-supplied text."""
        assert req.text is not None
        trace: List[Dict[str, Any]] = []

        try:
            tool_res = self.rel_tool.extract_from_text(
                text=req.text,
                source_go_number=req.go_number,
            )
        except Exception as exc:
            logger.error("RelationshipTool exception in extract_from_text: %s", exc, exc_info=True)
            return self._error_response(
                req.mode, req.go_number,
                "Relationship extraction operation could not be completed.",
                action="extract_from_text", trace=trace
            )

        status_raw = tool_res.get("status", "ERROR") if isinstance(tool_res, dict) else "ERROR"
        summary = (
            f"Status={status_raw}, total_references={tool_res.get('total_references', 0) if isinstance(tool_res, dict) else 0}, "
            f"total_relationships={tool_res.get('total_relationships', 0) if isinstance(tool_res, dict) else 0}"
        )
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "relationship_agent",
            "action": "extract_from_text",
            "reasoning": "Extract references and relationships from raw document text",
            "tool_called": "relationship_tool",
            "tool_input": {
                "text_length": len(req.text),
                "source_go_number": req.go_number,
            },
            "tool_output_summary": summary,
            "decision": "evaluate_text_extraction_results",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        if status_raw == "EMPTY_TEXT":
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.EMPTY_INPUT,
                mode=req.mode,
                source_go_number=req.go_number,
                errors=["Input document text cannot be empty."],
                execution_trace=trace,
            )

        if status_raw == "ERROR":
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.ERROR,
                mode=req.mode,
                source_go_number=req.go_number,
                is_complete=False,
                errors=["Relationship extraction operation could not be completed."],
                execution_trace=trace,
            )

        if status_raw != "SUCCESS":
            return RelationshipAgentResponse(
                status=RelationshipAgentStatus.ERROR,
                mode=req.mode,
                source_go_number=req.go_number,
                is_complete=False,
                errors=[f"Relationship tool returned unknown operational status '{status_raw}'."],
                execution_trace=trace,
            )

        return self._format_and_analyze_results(
            tool_res=tool_res,
            mode=req.mode,
            source_go_number=req.go_number,
            min_confidence=req.min_confidence,
            trace=trace,
        )

    def _investigate_pair(self, req: RelationshipAgentRequest) -> RelationshipAgentResponse:
        """
        Investigate whether a Source GO specifically modifies, amends, or references
        a specific Target GO number using boundary-aware identifier matching.
        """
        assert req.target_go_number is not None

        # Step 1: Run standard extraction
        if req.text:
            base_res = self._extract_from_text(req)
        else:
            base_res = self._investigate_order(req)

        if base_res.status != RelationshipAgentStatus.SUCCESS:
            return base_res

        # Step 2: Filter relationships and references specifically mentioning the target GO
        # ONLY structured referenced_go_numbers establish a confirmed relationship match.
        # Mentions appearing solely in narrative evidence_text are preserved separately as unresolved mentions.
        filtered_rels: List[RelationshipCandidate] = []
        unresolved_mentions: List[Dict[str, Any]] = []

        for rel in base_res.relationships:
            mentions_target = any(
                _is_target_go_match(req.target_go_number, r_num)
                for r_num in rel.referenced_go_numbers
            )

            if mentions_target:
                filtered_rels.append(rel)
            elif rel.evidence_text and _is_target_go_match(req.target_go_number, rel.evidence_text):
                # Preserved separately as an unresolved narrative mention, NOT as a confirmed pair match
                unresolved_mentions.append({
                    "relationship_type": rel.relationship_type.value,
                    "target_query": req.target_go_number,
                    "evidence_text": rel.evidence_text,
                    "matched_phrase": rel.matched_phrase,
                    "page_number": rel.page_number,
                    "reason": "Target GO appears in narrative evidence_text only; not confirmed by structured referenced_go_numbers.",
                })

        filtered_refs: List[ExtractedReferenceItem] = []
        for ref in base_res.references:
            if _is_target_go_match(req.target_go_number, ref.go_number):
                filtered_refs.append(ref)

        base_res.mode = RelationshipInvestigationMode.INVESTIGATE_PAIR
        base_res.target_go_number = req.target_go_number
        base_res.relationships = filtered_rels
        base_res.total_relationships = len(filtered_rels)
        base_res.references = filtered_refs
        base_res.total_references = len(filtered_refs)
        base_res.unresolved_mentions = unresolved_mentions

        if unresolved_mentions and not base_res.human_review_required:
            base_res.human_review_required = True
            base_res.human_review_reason = (
                f"Detected {len(unresolved_mentions)} unconfirmed narrative mention(s) of target "
                f"'{req.target_go_number}' in evidence text. Preserved for review."
            )

        trace_item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "relationship_agent",
            "action": "filter_pair_relationships",
            "reasoning": f"Filter relationships connecting source '{req.go_number}' with target '{req.target_go_number}' via boundary-aware matching on structured references only",
            "tool_called": "none",
            "tool_input": {"target_go_number": req.target_go_number},
            "tool_output_summary": f"Retained {len(filtered_rels)} confirmed relationships, {len(filtered_refs)} references, and {len(unresolved_mentions)} narrative mentions",
            "decision": "pair_investigation_complete",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        }
        base_res.execution_trace.append(trace_item)

        if not filtered_rels and not filtered_refs:
            base_res.status = RelationshipAgentStatus.NO_RELATIONSHIPS_FOUND

        return base_res

    def _cross_examine_database(self, req: RelationshipAgentRequest) -> RelationshipAgentResponse:
        """
        Cross-examine text-extracted relationships against registered database records
        from DatabaseTool to detect unrecorded candidates, omissions, or type mismatches.
        If the database operation fails, flags an incomplete comparison rather than generating false discrepancies.
        """
        assert req.go_number is not None

        # Step 1: Run text extraction for order
        base_res = self._investigate_order(req)
        if base_res.status != RelationshipAgentStatus.SUCCESS:
            return base_res

        base_res.mode = RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE
        trace = base_res.execution_trace

        # Step 2: Retrieve registered outgoing relationships from PostgreSQL
        try:
            db_res = self.db_tool.get_outgoing_relationships(go_number=req.go_number)
        except Exception as exc:
            logger.error("DatabaseTool exception in cross_examine_database: %s", exc, exc_info=True)
            base_res.is_complete = False
            base_res.errors.append("Database query encountered an operational error during cross-examination.")
            trace.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": "relationship_agent",
                "action": "cross_examine_database",
                "reasoning": "Retrieve outgoing relationships from database",
                "tool_called": "database_tool",
                "tool_input": {"go_number": req.go_number, "direction": "OUTGOING"},
                "tool_output_summary": "Database query failed; aborting cross-examination to avoid false discrepancies",
                "decision": "comparison_incomplete",
                "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
            })
            return base_res

        if not isinstance(db_res, dict) or db_res.get("status") != "SUCCESS":
            err_msg = db_res.get("error") if isinstance(db_res, dict) else "Unknown database error"
            base_res.is_complete = False
            base_res.errors.append(f"Database query failed during cross-examination: {err_msg}")
            return base_res

        db_rels = db_res.get("relationships", [])
        if not isinstance(db_rels, list):
            db_rels = []

        db_items: List[Dict[str, Any]] = []
        malformed_db_count = 0
        unsupported_db_rels: List[Dict[str, Any]] = []

        for dbr in db_rels:
            if not isinstance(dbr, dict):
                malformed_db_count += 1
                continue
            orig_tgt = str(dbr.get("target_go_number") or dbr.get("referenced_go_number", "")).strip()
            dbr_type_raw = str(dbr.get("relationship_type", "")).strip().upper()
            if not orig_tgt or not dbr_type_raw:
                malformed_db_count += 1
                continue

            try:
                dbr_type_enum = RelationshipType(dbr_type_raw)
                dbr_type = dbr_type_enum.value
            except ValueError:
                # Structurally valid database record containing an unsupported relationship type:
                # Excluded from matching candidates, preserved as diagnostic in unrecognized_relationships,
                # marks comparison incomplete and requires human review, but NOT counted as a malformed record.
                diag_item = {
                    "source": "database",
                    "target_go_number": orig_tgt,
                    "raw_relationship_type": dbr_type_raw,
                    "reason": f"Database records unsupported relationship type '{dbr_type_raw}' which is not one of the 8 valid RelationshipType enums.",
                }
                unsupported_db_rels.append(diag_item)
                base_res.unrecognized_relationships.append(diag_item)
                base_res.errors.append(
                    f"Database record contains unsupported relationship type '{dbr_type_raw}' for target '{orig_tgt}'. "
                    f"Preserved as diagnostic for human review; excluded from matching."
                )
                continue

            db_items.append({
                "orig_target": orig_tgt,
                "norm_target": orig_tgt.lower(),
                "type": dbr_type,
                "dbr": dbr,
                "matched": False,
            })

        # Process structurally malformed database records
        if malformed_db_count > 0:
            base_res.is_complete = False
            base_res.malformed_records_discarded += malformed_db_count
            base_res.human_review_required = True
            base_res.errors.append(
                f"Skipped {malformed_db_count} malformed database relationship record(s) during cross-examination."
            )

        # Structurally valid records with unsupported types mark response incomplete and require review
        if unsupported_db_rels:
            base_res.is_complete = False
            base_res.human_review_required = True

        text_items: List[Dict[str, Any]] = []
        for cand in base_res.relationships:
            c_type = cand.relationship_type.value.upper()
            for r_num in cand.referenced_go_numbers:
                orig_r = r_num.strip()
                if orig_r:
                    text_items.append({
                        "orig_target": orig_r,
                        "norm_target": orig_r.lower(),
                        "type": c_type,
                        "cand": cand,
                        "matched": False,
                    })

        discrepancies: List[DiscrepancyReport] = []

        # Pass 1: Find EXACT target + EXACT relationship-type matches
        for t_item in text_items:
            if t_item["matched"]:
                continue
            for d_item in db_items:
                if d_item["matched"]:
                    continue
                if _is_target_go_match(t_item["norm_target"], d_item["norm_target"]):
                    if t_item["type"] == d_item["type"]:
                        t_item["matched"] = True
                        d_item["matched"] = True
                        break

        # Pass 2: Find EXACT target + COMPLEMENTARY relationship-type matches (e.g. MODIFIES vs EXTENDS)
        # Complementary types are compatible administrative classifications and do NOT produce TYPE_MISMATCH
        for t_item in text_items:
            if t_item["matched"]:
                continue
            for d_item in db_items:
                if d_item["matched"]:
                    continue
                if _is_target_go_match(t_item["norm_target"], d_item["norm_target"]):
                    if self._is_complementary_type(t_item["type"], d_item["type"]):
                        t_item["matched"] = True
                        d_item["matched"] = True
                        break

        # Pass 3: Check remaining items on the same target with NON-COMPLEMENTARY types (e.g. CANCELS vs EXTENDS)
        # Genuine differing classification: flag as TYPE_MISMATCH for review without claiming confirmed contradiction
        for t_item in text_items:
            if t_item["matched"]:
                continue
            for d_item in db_items:
                if d_item["matched"]:
                    continue
                if _is_target_go_match(t_item["norm_target"], d_item["norm_target"]):
                    discrepancies.append(
                        DiscrepancyReport(
                            referenced_go_number=t_item["orig_target"],
                            category="TYPE_MISMATCH",
                            text_relationship_type=t_item["type"],
                            db_relationship_type=d_item["type"],
                            evidence_text=t_item["cand"].evidence_text,
                            description=(
                                f"Potential relationship type discrepancy for target '{t_item['orig_target']}': "
                                f"text extracted '{t_item['type']}' versus database recorded '{d_item['type']}'. "
                                f"Preserved both for human review without claiming confirmed contradiction."
                            ),
                        )
                    )
                    t_item["matched"] = True
                    d_item["matched"] = True
                    break

        # Pass 4: Check remaining unmatched text items (UNRECORDED_IN_DATABASE)
        for t_item in text_items:
            if not t_item["matched"]:
                discrepancies.append(
                    DiscrepancyReport(
                        referenced_go_number=t_item["orig_target"],
                        category="UNRECORDED_IN_DATABASE",
                        text_relationship_type=t_item["type"],
                        db_relationship_type=None,
                        evidence_text=t_item["cand"].evidence_text,
                        description=f"Relationship '{t_item['type']}' for target '{t_item['orig_target']}' extracted from text but not recorded in database.",
                    )
                )

        # Pass 5: Check remaining unmatched database items (MISSING_FROM_TEXT)
        # Ensures that if DB had both EXTENDS and CANCELS, and text had EXTENDS only, CANCELS is not silently covered!
        for d_item in db_items:
            if not d_item["matched"]:
                discrepancies.append(
                    DiscrepancyReport(
                        referenced_go_number=d_item["orig_target"],
                        category="MISSING_FROM_TEXT",
                        text_relationship_type=None,
                        db_relationship_type=d_item["type"],
                        evidence_text=None,
                        description=f"Database records outgoing relationship '{d_item['type']}' to '{d_item['orig_target']}', but no corresponding text signal was extracted.",
                    )
                )

        base_res.discrepancies = discrepancies
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "relationship_agent",
            "action": "cross_examine_database",
            "reasoning": "Cross-examine text relationship candidates against PostgreSQL registered relationships",
            "tool_called": "database_tool",
            "tool_input": {"go_number": req.go_number, "direction": "OUTGOING"},
            "tool_output_summary": f"Found {len(discrepancies)} discrepancies between text extraction and database",
            "decision": "cross_examination_complete",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        return base_res

    def _format_and_analyze_results(
        self,
        tool_res: Dict[str, Any],
        mode: RelationshipInvestigationMode,
        source_go_number: Optional[str],
        min_confidence: Optional[ExtractionConfidence],
        trace: List[Dict[str, Any]],
    ) -> RelationshipAgentResponse:
        """
        Format raw tool results into structured items, perform conservative conflict
        detection, validate individual records, preserve unresolved ordinals, and extract discovered GO numbers.
        """
        raw_refs = tool_res.get("references", [])
        if not isinstance(raw_refs, list):
            raw_refs = []

        raw_rels = tool_res.get("relationships", [])
        if not isinstance(raw_rels, list):
            raw_rels = []

        actual_source = tool_res.get("source_go_number") or source_go_number

        ref_items: List[ExtractedReferenceItem] = []
        discovered_gos: List[str] = []
        malformed_count = 0
        errors: List[str] = []

        # Validate individual reference records
        for r in raw_refs:
            if not isinstance(r, dict) or "ordinal" not in r:
                malformed_count += 1
                continue

            ref_num = str(r.get("go_number") or "").strip()
            ref_items.append(
                ExtractedReferenceItem(
                    ordinal=r.get("ordinal", 0),
                    go_number=ref_num,
                    issue_date=r.get("issue_date"),
                    page_number=r.get("page_number"),
                    evidence_text=r.get("evidence_text"),
                )
            )
            if ref_num and ref_num not in discovered_gos and ref_num != actual_source:
                discovered_gos.append(ref_num)

        # Validate individual relationship candidates
        rel_candidates: List[RelationshipCandidate] = []
        unrecognized_rels: List[Dict[str, Any]] = []
        confidence_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_rank = confidence_rank.get(min_confidence.value, 1) if min_confidence else 1

        for r in raw_rels:
            if not isinstance(r, dict) or "relationship_type" not in r:
                malformed_count += 1
                continue

            raw_type_str = str(r.get("relationship_type", "")).strip().upper()
            try:
                rel_type = RelationshipType(raw_type_str)
            except ValueError:
                # Do NOT convert unknown types into MODIFIES or any other valid type!
                logger.warning("Unrecognized relationship type '%s'; preserving as diagnostic rejected candidate", raw_type_str)
                malformed_count += 1
                unrecognized_rels.append({
                    "raw_relationship_type": raw_type_str,
                    "referenced_go_numbers": r.get("referenced_go_numbers", []),
                    "matched_phrase": r.get("matched_phrase"),
                    "evidence_text": r.get("evidence_text"),
                    "page_number": r.get("page_number"),
                    "reason": f"Unknown relationship type '{raw_type_str}' is not one of the 8 standard RelationshipType enums.",
                })
                errors.append(
                    f"Unrecognized relationship type '{raw_type_str}' detected. "
                    f"Candidate preserved as unrecognized diagnostic for human review."
                )
                continue

            raw_conf = str(r.get("confidence", "HIGH")).upper()
            if confidence_rank.get(raw_conf, 1) < min_rank:
                continue

            ref_ordinals = r.get("reference_ordinals", [])
            if not isinstance(ref_ordinals, list):
                ref_ordinals = []

            ref_numbers = r.get("referenced_go_numbers", [])
            if not isinstance(ref_numbers, list):
                ref_numbers = []

            # Preserve unresolved ordinals: never invent a referenced GO number!
            clean_ref_numbers = [str(num).strip() for num in ref_numbers if num and str(num).strip()]

            cand = RelationshipCandidate(
                relationship_type=rel_type,
                source_go_number=actual_source,
                referenced_go_numbers=clean_ref_numbers,
                reference_ordinals=ref_ordinals,
                matched_phrase=str(r.get("matched_phrase", "")),
                evidence_text=r.get("evidence_text"),
                page_number=r.get("page_number"),
                confidence=raw_conf,
                extraction_method=str(r.get("extraction_method", "rule_based")),
                review_required=bool(r.get("review_required", False)),
                verification_status="UNVERIFIED_CANDIDATE",
                legal_effect_verified=False,
            )
            rel_candidates.append(cand)

            for num in clean_ref_numbers:
                if num and num not in discovered_gos and num != actual_source:
                    discovered_gos.append(num)

        # Conservative Conflict Detection:
        # Group candidates by referenced GO number and inspect lifecycle actions
        conflicts_flagged = self._evaluate_conservative_conflicts(rel_candidates)
        human_review_required = any(c.review_required or c.potential_conflict for c in rel_candidates) or bool(unrecognized_rels)
        human_review_reason = None
        if conflicts_flagged > 0:
            human_review_reason = (
                f"Flagged {conflicts_flagged} potentially conflicting relationship signals "
                f"for human review. (Preserved all candidate evidence without declaring confirmed legal contradiction)."
            )
        elif unrecognized_rels:
            human_review_reason = (
                f"Detected {len(unrecognized_rels)} unrecognized relationship type(s). "
                f"Preserved as diagnostic records for human review."
            )

        is_complete = True
        if malformed_count > 0:
            is_complete = False
            errors.append(f"Detected and skipped {malformed_count} malformed record(s) in tool output.")

        status = RelationshipAgentStatus.SUCCESS
        if not rel_candidates and not ref_items:
            status = RelationshipAgentStatus.NO_RELATIONSHIPS_FOUND

        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "relationship_agent",
            "action": "analyze_and_format_results",
            "reasoning": "Analyze extracted relationships, detect conservative conflicts, and format candidate evidence",
            "tool_called": "none",
            "tool_input": None,
            "tool_output_summary": (
                f"Formatted {len(rel_candidates)} relationships and {len(ref_items)} references. "
                f"Potential conflicts flagged: {conflicts_flagged}, malformed discarded: {malformed_count}."
            ),
            "decision": "investigation_complete",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })

        return RelationshipAgentResponse(
            status=status,
            mode=mode,
            source_go_number=actual_source,
            total_references=len(ref_items),
            references=ref_items,
            total_relationships=len(rel_candidates),
            relationships=rel_candidates,
            unrecognized_relationships=unrecognized_rels,
            discrepancies=[],
            discovered_go_numbers=discovered_gos,
            human_review_required=human_review_required,
            human_review_reason=human_review_reason,
            is_complete=is_complete,
            malformed_records_discarded=malformed_count,
            errors=errors,
            execution_trace=trace,
            legal_effect_verified=False,
        )

    def _evaluate_conservative_conflicts(self, candidates: List[RelationshipCandidate]) -> int:
        """
        Conservative conflict detection:
        - Layered or complementary modifications (e.g. EXTENDS + MODIFIES, AMENDS + READ_AS)
          are completely normal in Government Orders and are NOT considered conflicts.
        - Only contradictory lifecycle assertions (e.g. CANCELS/REVOKES occurring alongside
          EXTENDS/AMENDS/MODIFIES for the exact same target GO) are flagged for human review.
        - Preserves all candidates and evidence; never declares a confirmed legal contradiction.
        """
        target_to_candidates: Dict[str, List[RelationshipCandidate]] = {}
        for cand in candidates:
            for tgt in cand.referenced_go_numbers:
                norm_tgt = tgt.strip().lower()
                if norm_tgt:
                    target_to_candidates.setdefault(norm_tgt, []).append(cand)

        conflicts_count = 0
        for norm_tgt, cands in target_to_candidates.items():
            types_present = {c.relationship_type for c in cands}
            has_terminating = bool(types_present & self.TERMINATING_TYPES)
            has_continuing = bool(types_present & self.CONTINUING_TYPES)

            # If both terminating (CANCELS/REVOKES) and continuing (AMENDS/MODIFIES/EXTENDS)
            # are asserted on the same target, flag for human review
            if has_terminating and has_continuing:
                conflicts_count += 1
                for c in cands:
                    c.potential_conflict = True
                    c.review_required = True
                    c.conflict_notes = (
                        f"Potentially conflicting relationship signals detected for referenced GO '{norm_tgt}': "
                        f"terminating ({[t.value for t in types_present if t in self.TERMINATING_TYPES]}) "
                        f"vs continuing ({[t.value for t in types_present if t in self.CONTINUING_TYPES]}). "
                        f"Preserved all evidence for human review."
                    )

        return conflicts_count

    def _error_response(
        self,
        mode: RelationshipInvestigationMode,
        source_go_number: Optional[str],
        error_message: str,
        action: str,
        trace: List[Dict[str, Any]],
    ) -> RelationshipAgentResponse:
        """Helper to construct safe, sanitized error responses."""
        trace.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "relationship_agent",
            "action": action,
            "reasoning": "Execute relationship investigation",
            "tool_called": "relationship_tool",
            "tool_input": {"source_go_number": source_go_number},
            "tool_output_summary": f"Failed with sanitized message: {error_message}",
            "decision": "terminate_with_error",
            "decision_source": DecisionSource.DETERMINISTIC_RULES.value,
        })
        return RelationshipAgentResponse(
            status=RelationshipAgentStatus.ERROR,
            mode=mode,
            source_go_number=source_go_number,
            is_complete=False,
            errors=[error_message],
            execution_trace=trace,
            legal_effect_verified=False,
        )
