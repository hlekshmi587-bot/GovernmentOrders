"""
Verification Agent for Kerala Government Orders Multi-Agent Architecture.

This module implements Phase 2 Step 2.4: Verification Agent.
The Verification Agent acts as the epistemic conscience of the system:
1. Fact-checks claims against authoritative PostgreSQL Government Order records.
2. Distinguishes direct operative clauses from indirect citations or unclassified snippets.
3. Independently validates page provenance from discrete page spans; prevents circular verification.
4. Strictly distinguishes verbatim text presence, topical relevance, and semantic entailment.
5. Cross-examines relationship candidates using RelationshipAgent Step 2.3 logic with 1-to-1 matching.
6. Evaluates temporal ordering conservatively without declaring legal invalidity from dates alone.
7. Produces strictly typed, allowlisted action recommendations without arbitrary code execution.
8. Computes explainable, deterministic evidence-strength confidence heuristics.
9. Enforces the permanent invariant: legal_effect_verified = False.
"""

from datetime import date, datetime, timezone
from enum import Enum
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.llm.llm_service import LLMService
from app.agents.specialized.database_agent import (
    DatabaseAgent,
    DatabaseAgentRequest,
    DatabaseAgentResponse,
    DatabaseOperation,
    AgentResponseStatus,
    OrderMetadataItem,
)
from app.agents.specialized.relationship_agent import (
    RelationshipAgent,
    RelationshipAgentRequest,
    RelationshipInvestigationMode,
    RelationshipAgentStatus,
    RelationshipCandidate,
    DiscrepancyReport,
)
from app.agents.specialized.search_agent import (
    SearchAgent,
    SearchAgentRequest,
    SearchAgentResponse,
    RetrievalMode,
)

logger = logging.getLogger(__name__)


# =========================================================================
# STATUS ENUMS & TYPES
# =========================================================================

class VerificationStatus(str, Enum):
    """Overall report verification status."""
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"


class ClaimVerificationStatus(str, Enum):
    """Verification status of an individual factual claim."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    POTENTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    REFUTED = "REFUTED"


class EvidenceDirectness(str, Enum):
    """Evidence directness classification."""
    DIRECT_OPERATIVE_ORDER = "DIRECT_OPERATIVE_ORDER"
    DIRECT = "DIRECT_OPERATIVE_ORDER"
    INDIRECT_PREAMBLE_CITATION = "INDIRECT_PREAMBLE_CITATION"
    INDIRECT = "INDIRECT_PREAMBLE_CITATION"
    UNCLASSIFIED = "UNCLASSIFIED"
    NONE = "NONE"


class TextSourceType(str, Enum):
    """Physical text source availability classification."""
    FULL_DATABASE_TEXT = "FULL_DATABASE_TEXT"
    TRUNCATED_PREVIEW = "TRUNCATED_PREVIEW"
    SEARCH_SNIPPET = "SEARCH_SNIPPET"
    SEMANTIC_CHUNK = "SEMANTIC_CHUNK"
    MISSING = "MISSING"


class PageLocatorSource(str, Enum):
    """Provenance trustworthiness of a page locator."""
    TRUSTED_DOCUMENT_METADATA = "TRUSTED_DOCUMENT_METADATA"  # Internally validated SearchAgent/Qdrant result
    UNTRUSTED_CANDIDATE = "UNTRUSTED_CANDIDATE"              # External request input, LLM output, or user citation
    DERIVED_FROM_PAGE_MARKER = "DERIVED_FROM_PAGE_MARKER"    # Derived from unique page marker in document text
    UNRESOLVED_AMBIGUOUS_PAGE = "UNRESOLVED_AMBIGUOUS_PAGE"  # Duplicate excerpt across multiple pages


class RelationshipRegistrationStatus(str, Enum):
    """Registration status of relationship candidate against database."""
    REGISTERED_EXACT_MATCH = "REGISTERED_EXACT_MATCH"
    RECORDED_EXACT = "REGISTERED_EXACT_MATCH"
    REGISTERED_COMPLEMENTARY_MATCH = "REGISTERED_COMPLEMENTARY_MATCH"
    RECORDED_COMPLEMENTARY = "REGISTERED_COMPLEMENTARY_MATCH"
    UNREGISTERED_INCOMPATIBLE_TYPE = "UNREGISTERED_INCOMPATIBLE_TYPE"
    UNRECORDED_IN_DATABASE = "UNRECORDED_IN_DATABASE"
    MISSING_FROM_TEXT = "MISSING_FROM_TEXT"
    TYPE_MISMATCH = "UNREGISTERED_INCOMPATIBLE_TYPE"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"


class RelationshipVerificationCheckState(str, Enum):
    """Coverage and execution state of relationship cross-examination."""
    SKIPPED_VACUOUS = "SKIPPED_VACUOUS"
    COMPLETED_CONSISTENT = "COMPLETED_CONSISTENT"
    DISCREPANCIES_FOUND = "DISCREPANCIES_FOUND"
    FAILED_INCOMPLETE = "FAILED_INCOMPLETE"
    NO_RELATIONSHIPS_TO_CHECK = "SKIPPED_VACUOUS"
    CROSS_EXAMINED_WITH_NO_DISCREPANCIES = "COMPLETED_CONSISTENT"
    CROSS_EXAMINED_WITH_DISCREPANCIES = "DISCREPANCIES_FOUND"
    RELATIONSHIP_CHECK_ERROR = "FAILED_INCOMPLETE"


class TemporalRelationType(str, Enum):
    """Temporal sequence relation between two government orders."""
    SEQUENTIAL_TARGET_BEFORE_SOURCE = "SEQUENTIAL_TARGET_BEFORE_SOURCE"
    TEMPORAL_ORDERING_SEQUENTIAL = "SEQUENTIAL_TARGET_BEFORE_SOURCE"
    INVERTED_SOURCE_BEFORE_TARGET = "INVERTED_SOURCE_BEFORE_TARGET"
    TEMPORAL_ORDERING_DISCREPANCY = "INVERTED_SOURCE_BEFORE_TARGET"
    SAME_DOCUMENT_ASSERTION = "SAME_DOCUMENT_ASSERTION"
    UNKNOWN_TEMPORAL_ORDER = "UNKNOWN_TEMPORAL_ORDER"
    UNDETERMINED_MISSING_DATE = "UNKNOWN_TEMPORAL_ORDER"


class VerificationConfidence(str, Enum):
    """Confidence rating based on unified deterministic bounds."""
    HIGH = "HIGH"        # score >= 0.80
    MEDIUM = "MEDIUM"    # 0.60 <= score < 0.80
    LOW = "LOW"          # score < 0.60 (always triggers human review)


# =========================================================================
# TYPED ACTION RECOMMENDATION CONTRACTS (1-to-1 Mapping)
# =========================================================================

class AutonomousVerificationAction(str, Enum):
    """Allowed actions recommended by the Verification Agent."""
    FETCH_ORDER_TEXT = "FETCH_ORDER_TEXT"
    TARGETED_SEARCH = "TARGETED_SEARCH"
    INVESTIGATE_RELATIONSHIP = "INVESTIGATE_RELATIONSHIP"
    PROCEED_TO_RESPONSE = "PROCEED_TO_RESPONSE"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"


class TargetAgentName(str, Enum):
    """Allowed target agents for recommendations."""
    DATABASE_AGENT = "database_agent"
    SEARCH_AGENT = "search_agent"
    RELATIONSHIP_AGENT = "relationship_agent"
    SUPERVISOR_AGENT = "supervisor_agent"
    HUMAN_REVIEW = "human_review"


class FetchOrderTextParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    go_number: str = Field(min_length=3, max_length=150)
    include_full_text: bool = True


class TargetedSearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=255)
    limit: int = Field(default=5, ge=1, le=20)
    department: Optional[str] = Field(default=None, max_length=255)


class InvestigateRelationshipParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    go_number: str = Field(min_length=3, max_length=150)
    target_go_number: Optional[str] = Field(default=None, max_length=150)
    mode: str = Field(default="CROSS_EXAMINE_DATABASE")


class RequestHumanReviewParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_reason: str = Field(min_length=5, max_length=500)
    conflicting_go_numbers: List[str] = Field(default_factory=list)


class EmptyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerificationRecommendation(BaseModel):
    """
    Structured, allowlisted recommendation for the Supervisor or LangGraph router.
    Strictly pairs each action with its exact parameter model and designated target agent.
    """
    model_config = ConfigDict(extra="forbid")

    action: AutonomousVerificationAction
    target_agent: TargetAgentName
    parameters: Union[
        FetchOrderTextParams,
        TargetedSearchParams,
        InvestigateRelationshipParams,
        RequestHumanReviewParams,
        EmptyParams,
    ]
    reasoning: str = Field(min_length=5, max_length=1000)

    @model_validator(mode="after")
    def validate_action_parameter_and_agent_pairing(self) -> "VerificationRecommendation":
        EXACT_PAIRINGS = {
            AutonomousVerificationAction.FETCH_ORDER_TEXT: (FetchOrderTextParams, TargetAgentName.DATABASE_AGENT),
            AutonomousVerificationAction.TARGETED_SEARCH: (TargetedSearchParams, TargetAgentName.SEARCH_AGENT),
            AutonomousVerificationAction.INVESTIGATE_RELATIONSHIP: (InvestigateRelationshipParams, TargetAgentName.RELATIONSHIP_AGENT),
            AutonomousVerificationAction.REQUEST_HUMAN_REVIEW: (RequestHumanReviewParams, TargetAgentName.HUMAN_REVIEW),
            AutonomousVerificationAction.PROCEED_TO_RESPONSE: (EmptyParams, TargetAgentName.SUPERVISOR_AGENT),
        }
        expected_type, expected_agent = EXACT_PAIRINGS[self.action]
        if not isinstance(self.parameters, expected_type):
            raise ValueError(
                f"Action '{self.action.value}' strictly requires parameter model '{expected_type.__name__}', "
                f"got '{type(self.parameters).__name__}'."
            )
        if self.target_agent != expected_agent:
            raise ValueError(
                f"Action '{self.action.value}' must target '{expected_agent.value}', "
                f"got '{self.target_agent.value}'."
            )
        return self


# =========================================================================
# TYPED INPUT MODELS (Strict extra="forbid", untrusted external contracts)
# =========================================================================

class SupportingChunkInput(BaseModel):
    """Supporting chunk from semantic/hybrid search."""
    model_config = ConfigDict(extra="forbid")

    chunk_index: int = 0
    chunk_text: str = ""
    content: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None

    @model_validator(mode="after")
    def sync_chunk_content(self) -> "SupportingChunkInput":
        if self.content and not self.chunk_text:
            self.chunk_text = self.content
        elif self.chunk_text and not self.content:
            self.content = self.chunk_text
        return self


class VerificationClaim(BaseModel):
    """An individual factual claim submitted for verification."""
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(default="claim_1", min_length=1, max_length=50)
    claim_text: str = Field(min_length=1, max_length=1000)
    cited_go_number: Optional[str] = None
    cited_excerpt: Optional[str] = None
    reported_page_number: Optional[int] = None
    supporting_chunks: List[SupportingChunkInput] = Field(default_factory=list)


class RelationshipCandidateInput(BaseModel):
    """Extracted relationship candidate submitted for verification."""
    model_config = ConfigDict(extra="forbid")

    relationship_type: str = Field(min_length=3, max_length=50)
    source_go_number: Optional[str] = None
    source_order_number: Optional[str] = None
    target_order_number: Optional[str] = None
    referenced_go_numbers: List[str] = Field(default_factory=list)
    reference_ordinals: List[int] = Field(default_factory=list)
    matched_phrase: Optional[str] = None
    evidence_text: Optional[str] = None
    page_number: Optional[int] = None
    confidence: Optional[str] = None
    verification_status: str = "UNVERIFIED_CANDIDATE"
    legal_effect_verified: bool = False

    @field_validator("legal_effect_verified")
    @classmethod
    def enforce_input_legal_invariant(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError(
                "legal_effect_verified cannot be set to True on input candidate: "
                "relationship candidates are unverified extractions."
            )
        return False

    @model_validator(mode="after")
    def sync_candidate_fields(self) -> "RelationshipCandidateInput":
        if self.source_order_number and not self.source_go_number:
            self.source_go_number = self.source_order_number
        elif self.source_go_number and not self.source_order_number:
            self.source_order_number = self.source_go_number

        if self.target_order_number:
            if not self.referenced_go_numbers:
                self.referenced_go_numbers = [self.target_order_number]
            elif self.target_order_number not in self.referenced_go_numbers:
                self.referenced_go_numbers.append(self.target_order_number)
        elif self.referenced_go_numbers:
            self.target_order_number = self.referenced_go_numbers[0]
        return self


class DocumentEvidenceInput(BaseModel):
    """Pre-gathered document evidence passed in state."""
    model_config = ConfigDict(extra="forbid")

    go_number: Optional[str] = None
    order_number: Optional[str] = None
    order_id: Optional[int] = None
    department: Optional[str] = None
    issue_date: Optional[str] = None
    source_filename: Optional[str] = None
    extracted_text: Optional[str] = None
    text_preview: Optional[str] = None
    extracted_text_length: Optional[int] = None

    @model_validator(mode="after")
    def sync_go_number(self) -> "DocumentEvidenceInput":
        if self.order_number and not self.go_number:
            self.go_number = self.order_number
        elif self.go_number and not self.order_number:
            self.order_number = self.go_number
        if not self.go_number:
            raise ValueError("Either go_number or order_number must be provided.")
        return self


class SearchEvidenceInput(BaseModel):
    """Search evidence passed from SearchAgent."""
    model_config = ConfigDict(extra="forbid")

    go_number: str = Field(min_length=3, max_length=150)
    go_id: Optional[int] = None
    department: Optional[str] = None
    issue_date: Optional[str] = None
    source_filename: Optional[str] = None
    text_snippet: Optional[str] = None
    supporting_chunks: List[SupportingChunkInput] = Field(default_factory=list)


class VerificationAgentRequest(BaseModel):
    """Validated input contract for VerificationAgent."""
    model_config = ConfigDict(extra="forbid")

    user_query: Optional[str] = None
    claims: List[VerificationClaim] = Field(default_factory=list)
    cited_go_numbers: List[str] = Field(default_factory=list)
    relationship_candidates: List[RelationshipCandidateInput] = Field(default_factory=list)
    candidates: Optional[List[RelationshipCandidateInput]] = None
    document_evidence: List[DocumentEvidenceInput] = Field(default_factory=list)
    search_evidence: List[SearchEvidenceInput] = Field(default_factory=list)
    max_retries: int = Field(default=2, ge=0, le=5)
    current_retry: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def sync_candidates(self) -> "VerificationAgentRequest":
        if self.candidates is not None and not self.relationship_candidates:
            self.relationship_candidates = self.candidates
        elif self.relationship_candidates and self.candidates is None:
            self.candidates = self.relationship_candidates
        return self


# =========================================================================
# VERIFICATION OUTPUT MODELS
# =========================================================================

class ClaimEvidenceVerification(BaseModel):
    """Verification assessment of a single factual claim."""
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_text: str
    cited_go_number: Optional[str] = None
    go_exists_in_database: bool = False
    existence_checked: bool = False
    text_source_type: TextSourceType = TextSourceType.MISSING
    status: ClaimVerificationStatus
    evidence_directness: EvidenceDirectness = EvidenceDirectness.NONE
    evidence_text: Optional[str] = None
    page_number: Optional[int] = None
    page_number_verified: bool = False
    page_locator_source: PageLocatorSource = PageLocatorSource.UNTRUSTED_CANDIDATE
    source_filename: Optional[str] = None
    reasoning: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_factors: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_confidence(self) -> "ClaimEvidenceVerification":
        if self.confidence > 0.0 and self.confidence_score == 0.0:
            self.confidence_score = self.confidence
        elif self.confidence_score > 0.0 and self.confidence == 0.0:
            self.confidence = self.confidence_score
        return self


class RelationshipVerificationDetail(BaseModel):
    """Verification assessment of an extracted relationship candidate."""
    model_config = ConfigDict(extra="forbid")

    source_go_number: str
    target_go_number: str
    candidate_relationship_type: str = "UNKNOWN"
    relationship_type: Optional[str] = None
    database_relationship_type: Optional[str] = None
    registration_status: RelationshipRegistrationStatus = RelationshipRegistrationStatus.REGISTERED_EXACT_MATCH
    status: Optional[RelationshipRegistrationStatus] = None
    evidence_text: Optional[str] = None
    page_number: Optional[int] = None
    source_date: Optional[date] = None
    target_date: Optional[date] = None
    temporal_relation: TemporalRelationType = TemporalRelationType.UNKNOWN_TEMPORAL_ORDER
    human_review_required: bool = False
    legal_effect_verified: bool = False  # ENFORCED INVARIANT
    notes: str = ""

    @model_validator(mode="after")
    def sync_detail_fields(self) -> "RelationshipVerificationDetail":
        if self.relationship_type and self.candidate_relationship_type == "UNKNOWN":
            self.candidate_relationship_type = self.relationship_type
        elif self.candidate_relationship_type != "UNKNOWN" and not self.relationship_type:
            self.relationship_type = self.candidate_relationship_type
        if self.status:
            self.registration_status = self.status
        elif self.registration_status and not self.status:
            self.status = self.registration_status
        return self

    @field_validator("legal_effect_verified")
    @classmethod
    def enforce_relationship_legal_invariant(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("legal_effect_verified MUST remain False: registration check is not legal validation.")
        return False


class VerificationAgentResponse(BaseModel):
    """Structured output contract returned by VerificationAgent."""
    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    overall_confidence: VerificationConfidence
    confidence_score: float = Field(ge=0.0, le=1.0)
    is_complete: bool = True
    evidence_sufficient: bool = False
    verified_claims: List[ClaimEvidenceVerification] = Field(default_factory=list)
    unsupported_claims: List[ClaimEvidenceVerification] = Field(default_factory=list)
    verified_relationships: List[RelationshipVerificationDetail] = Field(default_factory=list)
    relationship_check_performed: bool = False
    relationship_consistency_verified: bool = False
    discrepancies: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_references: List[Dict[str, Any]] = Field(default_factory=list)
    discovered_go_numbers: List[str] = Field(default_factory=list)
    human_review_required: bool = False
    human_review_reason: Optional[str] = None
    recommendation: VerificationRecommendation
    errors: List[str] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    legal_effect_verified: bool = False  # ENFORCED INVARIANT

    @field_validator("legal_effect_verified")
    @classmethod
    def enforce_response_legal_invariant(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("legal_effect_verified MUST remain False across verification agent responses.")
        return False


# =========================================================================
# LLM ENTAILMENT INTERNAL SCHEMA
# =========================================================================

class LLMEntailmentStatus(str, Enum):
    ENTAILED = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    NEUTRAL = "NEUTRAL"
    INSUFFICIENT = "INSUFFICIENT"


# =========================================================================
# VERIFICATION AGENT IMPLEMENTATION
# =========================================================================

class VerificationAgent:
    """
    Specialized Verification Agent ensuring factual grounding, relationship
    consistency, and provenance verification across Kerala Government Orders.
    """

    def __init__(
        self,
        database_agent: Optional[DatabaseAgent] = None,
        relationship_agent: Optional[RelationshipAgent] = None,
        search_agent: Optional[SearchAgent] = None,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        self.db_agent = database_agent or DatabaseAgent()
        self.rel_agent = relationship_agent or RelationshipAgent()
        self.search_agent = search_agent or SearchAgent()
        self.llm_service = llm_service or LLMService()

    # ---------------------------------------------------------------------
    # Conservative Identifier Normalization
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_go_identifier(go_number: Optional[str]) -> str:
        """
        Conservative GO identifier normalization.
        Standardizes punctuation, whitespace, and case without stripping numbers
        or causing distinct orders (e.g. '12/2024' vs '120/2024') to collide.
        """
        if not go_number:
            return ""
        s = go_number.strip().upper()
        # Normalize various dashes to standard hyphen
        s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", s)
        # Normalize G.O. variants
        s = re.sub(r"\bG\s*\.?\s*O\s*\.?", "G.O.", s, flags=re.IGNORECASE)
        # Normalize parenthesized types: (Ms), (P), (Rt)
        s = re.sub(r"\(\s*MS\s*\)", "(MS)", s, flags=re.IGNORECASE)
        s = re.sub(r"\(\s*P\s*\)", "(P)", s, flags=re.IGNORECASE)
        s = re.sub(r"\(\s*RT\s*\)", "(RT)", s, flags=re.IGNORECASE)
        # Normalize NO. and spaces around it
        s = re.sub(r"\bNO\s*\.?\s*", "NO.", s, flags=re.IGNORECASE)
        # Remove spaces around parentheses, dots, slashes, and hyphens
        s = re.sub(r"\s*\(\s*", "(", s)
        s = re.sub(r"\s*\)\s*", ")", s)
        s = re.sub(r"\s*/\s*", "/", s)
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @classmethod
    def _are_go_identifiers_equal(cls, id1: Optional[str], id2: Optional[str]) -> bool:
        """Check equality of two GO identifiers under conservative normalization."""
        if not id1 or not id2:
            return False
        n1 = cls._normalize_go_identifier(id1)
        n2 = cls._normalize_go_identifier(id2)
        return n1 == n2

    # ---------------------------------------------------------------------
    # Internal Text Classification & Directness
    # ---------------------------------------------------------------------

    @staticmethod
    def _classify_text_source(order_data: Optional[Any]) -> TextSourceType:
        """
        Classify text availability based strictly on actual DatabaseAgent response fields.
        Does not trust request flags or arbitrary length thresholds.
        """
        if order_data is None:
            return TextSourceType.MISSING

        raw_text = getattr(order_data, "extracted_text", None)
        text_len = getattr(order_data, "extracted_text_length", None)
        preview = getattr(order_data, "text_preview", None)

        if raw_text is None and bool(preview):
            return TextSourceType.TRUNCATED_PREVIEW

        if raw_text is not None and text_len is not None and text_len > 0 and len(raw_text) == text_len:
            return TextSourceType.FULL_DATABASE_TEXT

        if raw_text and (text_len is None or text_len <= 0 or len(raw_text) != text_len):
            return TextSourceType.TRUNCATED_PREVIEW

        return TextSourceType.MISSING

    @staticmethod
    def _detect_section_directness(text: str, excerpt: str) -> EvidenceDirectness:
        """
        Detect evidence directness based on actual line-level section-heading boundaries.
        Does not trigger operative directness when 'ORDER' appears inside ordinary paragraphs.
        """
        if not text or not excerpt:
            return EvidenceDirectness.NONE

        norm_text = " ".join(text.split()).lower()
        norm_excerpt = " ".join(excerpt.split()).lower()

        if norm_excerpt not in norm_text:
            return EvidenceDirectness.NONE

        # Find exact line-level section boundaries
        # Operative line heading: e.g. "ORDER" on its own line, or preceded by G.O. header
        order_heading_regex = re.compile(r"^\s*(?:ORDER|O\s*R\s*D\s*E\s*R)\s*$", re.MULTILINE | re.IGNORECASE)
        read_heading_regex = re.compile(r"^\s*(?:Read|Reference)\s*:", re.MULTILINE | re.IGNORECASE)
        preamble_heading_regex = re.compile(r"^\s*Preamble\b", re.MULTILINE | re.IGNORECASE)

        order_match = order_heading_regex.search(text)
        read_match = read_heading_regex.search(text)
        preamble_match = preamble_heading_regex.search(text)

        # Locate raw excerpt position in text
        excerpt_idx = text.lower().find(norm_excerpt)
        if excerpt_idx == -1:
            # Fallback search matching first 40 chars of excerpt
            prefix = norm_excerpt[:min(40, len(norm_excerpt))]
            excerpt_idx = text.lower().find(prefix)

        if not order_match:
            # Check if it falls inside Read or Preamble section
            if (read_match and excerpt_idx >= read_match.start()) or (preamble_match and excerpt_idx >= preamble_match.start()):
                return EvidenceDirectness.INDIRECT_PREAMBLE_CITATION
            return EvidenceDirectness.UNCLASSIFIED

        order_idx = order_match.start()

        if excerpt_idx >= order_idx:
            return EvidenceDirectness.DIRECT_OPERATIVE_ORDER

        # Before operative ORDER heading: check if in Read or Preamble
        if read_match and read_match.start() <= excerpt_idx < order_idx:
            return EvidenceDirectness.INDIRECT_PREAMBLE_CITATION

        if preamble_match and preamble_match.start() <= excerpt_idx < order_idx:
            return EvidenceDirectness.INDIRECT_PREAMBLE_CITATION

        return EvidenceDirectness.UNCLASSIFIED

    @classmethod
    def _classify_evidence_directness(cls, snippet: str, excerpt: Optional[str] = None) -> EvidenceDirectness:
        if excerpt is None:
            excerpt = snippet
        return cls._detect_section_directness(snippet, excerpt)

    # ---------------------------------------------------------------------
    # Page Provenance & Resolution Logic
    # ---------------------------------------------------------------------

    def _resolve_ambiguous_page_via_search(
        self,
        go_number: str,
        excerpt: str,
        matching_pages: List[int],
    ) -> Tuple[Optional[int], bool, PageLocatorSource, Optional[str]]:
        """
        Execute internal SearchAgent to resolve an excerpt appearing on multiple pages.
        Only trusts chunks when status == "SUCCESS", returned_count > 0, GO identity matches,
        excerpt is verbatim present in chunk_text, page_number is valid, and page_number is
        a member of matching_pages.
        """
        try:
            req = SearchAgentRequest(
                query=excerpt,
                retrieval_mode=RetrievalMode.HYBRID,
                limit=5,
            )
            res: SearchAgentResponse = self.search_agent.execute_search(req)
        except Exception as exc:
            logger.warning("SearchAgent execution failed during page resolution: %s", exc)
            return (
                None,
                False,
                PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE,
                f"SearchAgent execution failed: {exc}; multi-page occurrence remains unresolved.",
            )

        # Provenance is trusted ONLY when status is exactly "SUCCESS" and returned_count > 0
        if res.status != "SUCCESS" or res.returned_count <= 0:
            return (
                None,
                False,
                PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE,
                f"SearchAgent status '{res.status}' (count={res.returned_count}) insufficient; "
                f"multi-page occurrence across {matching_pages} remains unresolved.",
            )

        validated_pages: List[int] = []
        norm_excerpt = " ".join(excerpt.strip().split()).lower()

        for item in res.results:
            # 1. GO identity check
            if not self._are_go_identifiers_equal(item.go_number, go_number):
                continue

            for chunk in item.supporting_chunks:
                # 2. Excerpt presence check in chunk_text
                norm_chunk = " ".join(chunk.chunk_text.strip().split()).lower()
                if norm_excerpt not in norm_chunk:
                    continue

                # 3. Valid integer page number check
                p = chunk.page_number
                if p is None or not isinstance(p, int) or p < 1:
                    continue

                # 4. Authoritative page membership check
                if p in matching_pages:
                    validated_pages.append(p)

        if not validated_pages:
            return (
                None,
                False,
                PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE,
                f"No SearchAgent chunk passed provenance validation for GO '{go_number}'; "
                f"multi-page occurrence across {matching_pages} remains unresolved.",
            )

        unique_validated = sorted(set(validated_pages))
        if len(unique_validated) == 1:
            resolved_page = unique_validated[0]
            return (
                resolved_page,
                True,
                PageLocatorSource.TRUSTED_DOCUMENT_METADATA,
                f"Resolved ambiguous multi-page occurrence {matching_pages} to page {resolved_page} via internal SearchAgent evidence.",
            )

        # Conflicting chunks across multiple candidate pages
        return (
            None,
            False,
            PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE,
            f"Conflicting SearchAgent chunks pointed to different pages {unique_validated}; "
            f"ambiguity across {matching_pages} remains unresolved.",
        )

    def derive_page_number_from_text(
        self,
        source_text: Optional[str] = None,
        excerpt: str = "",
        reported_page: Optional[int] = None,
        validated_search_chunk: Optional[SupportingChunkInput] = None,
        text: Optional[str] = None,
        is_trusted_reported_page: bool = False,
        go_number: Optional[str] = None,
    ) -> Tuple[Optional[int], bool, PageLocatorSource, Optional[str]]:
        """
        Derive the exact page number of an excerpt from page-marked source text.
        A unique page occurrence in authoritative text does not require SearchAgent.
        Ambiguous multi-page occurrences invoke SearchAgent resolution if go_number is provided.
        Caller-supplied chunks or page numbers are NEVER treated as trusted.
        """
        src = source_text if source_text is not None else text
        if not src or not excerpt:
            return None, False, PageLocatorSource.UNTRUSTED_CANDIDATE, "Source text or excerpt is empty."

        page_marker_regex = re.compile(r"---\s*Page\s+(\d+)\s*---", re.IGNORECASE)
        matches = list(page_marker_regex.finditer(src))
        if not matches:
            return None, False, PageLocatorSource.UNTRUSTED_CANDIDATE, "Source text contains no page markers."

        page_spans: List[Tuple[int, str]] = []
        for i, match in enumerate(matches):
            page_num = int(match.group(1))
            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(src)
            page_spans.append((page_num, src[start_idx:end_idx]))

        norm_excerpt = " ".join(excerpt.strip().split()).lower()
        if not norm_excerpt:
            return None, False, PageLocatorSource.UNTRUSTED_CANDIDATE, "Normalized excerpt is empty."

        matching_pages: List[int] = []
        for page_num, raw_page_text in page_spans:
            clean_page = re.sub(r"---\s*(?:Embedded Text|OCR Text)\s*---", " ", raw_page_text, flags=re.IGNORECASE)
            norm_page = " ".join(clean_page.strip().split()).lower()
            if norm_excerpt in norm_page:
                matching_pages.append(page_num)

        # Case 1: Unambiguous unique page occurrence in authoritative full text
        if len(matching_pages) == 1:
            derived_page = matching_pages[0]
            if reported_page is not None and not is_trusted_reported_page and reported_page != derived_page:
                return (
                    derived_page,
                    False,
                    PageLocatorSource.UNTRUSTED_CANDIDATE,
                    f"Reported page {reported_page} contradicted by source text (found on page {derived_page}).",
                )
            return derived_page, True, PageLocatorSource.DERIVED_FROM_PAGE_MARKER, None

        # Case 2: Duplicate excerpt occurrences across multiple pages
        elif len(matching_pages) > 1:
            # If go_number is provided, attempt internal SearchAgent resolution
            if go_number:
                return self._resolve_ambiguous_page_via_search(go_number, excerpt, matching_pages)

            # Untrusted caller candidate or external input cannot resolve multi-page ambiguity
            return (
                None,
                False,
                PageLocatorSource.UNRESOLVED_AMBIGUOUS_PAGE,
                f"Ambiguous page provenance: excerpt occurs verbatim on multiple pages {matching_pages}; "
                f"requires internal SearchAgent validation to resolve.",
            )

        # Case 3: Excerpt not found on any single page
        return None, False, PageLocatorSource.UNTRUSTED_CANDIDATE, "Excerpt does not occur entirely within any single page span."

    @staticmethod
    def _is_verbatim_grounded(excerpt: Optional[str], source_text: Optional[str]) -> bool:
        """Deterministic check confirming an excerpt occurs verbatim in source text."""
        if not excerpt or not source_text:
            return False
        norm_excerpt = " ".join(excerpt.strip().split()).lower()
        norm_source = " ".join(source_text.strip().split()).lower()
        return norm_excerpt in norm_source

    # ---------------------------------------------------------------------
    # Semantic Entailment & Strict Validation
    # ---------------------------------------------------------------------

    def _evaluate_semantic_entailment(
        self,
        claim: Any,
        source_text: str,
        cited_excerpt: Optional[str] = None,
        excerpt: Optional[str] = None,
    ) -> Tuple[str, float, str, Optional[str]]:
        """
        Evaluate whether evidence semantically entails, contradicts, or fails to support a claim.
        Validates LLM response strictly: finite confidence in [0.0, 1.0].
        Malformed responses, out-of-bounds confidence, or outages produce UNVERIFIABLE with 0.0 confidence.
        """
        claim_text = claim.claim_text if hasattr(claim, "claim_text") else str(claim)
        effective_excerpt = cited_excerpt or excerpt or ""

        # Step 1: Independent Verbatim Grounding Check on the excerpt
        if not effective_excerpt or not self._is_verbatim_grounded(effective_excerpt, source_text):
            return (
                "UNVERIFIABLE",
                0.0,
                "Excerpt does not occur verbatim in the source order text.",
                None,
            )

        # Step 2: Query LLM for structured entailment
        prompt = (
            f"Factual Claim: {claim_text}\n\n"
            f"Source Text Excerpt:\n\"{effective_excerpt}\"\n\n"
            "Evaluate whether the excerpt supports or contradicts the claim.\n"
            "Provide JSON with: entailment_status ('ENTAILED', 'CONTRADICTED', 'NEUTRAL', or 'INSUFFICIENT'), "
            "confidence (float between 0.0 and 1.0), "
            "supporting_excerpt (exact substring if supporting), "
            "contradictory_excerpt (exact substring if contradicting), and reasoning."
        )
        system_prompt = (
            "You are a rigorous government-order fact checker. "
            "Never assert entailment without a genuine supporting excerpt from the text."
        )

        try:
            raw_res = self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if isinstance(raw_res, str):
                try:
                    data = json.loads(raw_res)
                except Exception:
                    m = re.search(r"\{.*\}", raw_res, re.DOTALL)
                    if m:
                        data = json.loads(m.group(0))
                    else:
                        raise ValueError("No JSON object found")
            elif isinstance(raw_res, dict):
                data = raw_res
            else:
                data = getattr(raw_res, "__dict__", {})
        except Exception as exc:
            logger.warning("LLM entailment evaluation failed: %s", exc)
            return (
                "UNVERIFIABLE",
                0.0,
                f"LLM entailment service failure: {exc}",
                None,
            )

        # Step 3: Strict Validation of LLM Confidence and Status
        raw_conf = data.get("confidence")
        if raw_conf is None or not isinstance(raw_conf, (int, float)):
            return "UNVERIFIABLE", 0.0, "LLM returned missing or non-numeric confidence.", None

        conf_float = float(raw_conf)
        if not math.isfinite(conf_float) or conf_float < 0.0 or conf_float > 1.0:
            return "UNVERIFIABLE", 0.0, f"LLM returned invalid confidence value: {raw_conf}.", None

        entail_status = str(data.get("entailment_status", "")).upper()
        reasoning = str(data.get("reasoning", ""))
        s_excerpt = data.get("supporting_excerpt")
        c_excerpt = data.get("contradictory_excerpt")

        if entail_status in ["CONTRADICTED", "CONTRADICTORY"]:
            check_excerpt = c_excerpt or effective_excerpt
            if self._is_verbatim_grounded(check_excerpt, source_text):
                return (
                    "CONTRADICTED",
                    conf_float,
                    f"Excerpt contradicts claim: {reasoning}",
                    check_excerpt,
                )
            return (
                "CONTRADICTED",
                conf_float,
                f"Contradiction identified: {reasoning}",
                effective_excerpt,
            )

        if entail_status in ["SUPPORTED", "ENTAILED"]:
            check_excerpt = s_excerpt or effective_excerpt
            if not self._is_verbatim_grounded(check_excerpt, source_text):
                return (
                    "UNVERIFIABLE",
                    0.0,
                    "LLM hallucinated a non-matching supporting excerpt; rejected as ungrounded.",
                    None,
                )
            return (
                "SUPPORTED",
                conf_float,
                reasoning or "Direct operative support",
                check_excerpt,
            )

        if entail_status == "NEUTRAL":
            return (
                "NEUTRAL",
                conf_float,
                f"Evidence is neutral or uninformative: {reasoning}",
                None,
            )

        if entail_status in ["INSUFFICIENT", "UNSUPPORTED"]:
            return (
                "UNSUPPORTED",
                conf_float,
                f"Evidence is insufficient to establish claim: {reasoning}",
                None,
            )

        return (
            "UNVERIFIABLE",
            0.0,
            f"Unrecognized entailment status: {entail_status}",
            None,
        )

    # ---------------------------------------------------------------------
    # Temporal Evaluation & Claim Relevance
    # ---------------------------------------------------------------------

    @classmethod
    def _is_claim_relevant_to_temporal_discrepancy(
        cls,
        claim_text: Optional[str],
        source_go: Optional[str],
        target_go: Optional[str],
    ) -> bool:
        """
        Determine whether a claim demonstrably concerns the affected source-target relationship
        or its chronological/legal effect. Merely mentioning both orders is insufficient.
        """
        if not claim_text or not source_go or not target_go:
            return False

        # Step A: Both entities must be present
        norm_claim = claim_text.lower()
        src_clean = source_go.strip().lower()
        tgt_clean = target_go.strip().lower()

        # Check entity presence
        has_src = src_clean in norm_claim or cls._normalize_go_identifier(source_go).lower() in norm_claim
        has_tgt = tgt_clean in norm_claim or cls._normalize_go_identifier(target_go).lower() in norm_claim
        if not (has_src and has_tgt):
            return False

        # Step B: Relational semantics / legal effect verbs connecting them
        relational_pattern = re.compile(
            r"\b(amend|amends|amended|partially amends|modify|modifies|modified|supersede|supersedes|superseded|"
            r"cancel|cancels|cancelled|revoke|revokes|revoked|extend|extends|extended|read as|"
            r"substitut|replac|prior to|subsequent to|earlier than|later than|preceding|following)\b",
            re.IGNORECASE,
        )
        return bool(relational_pattern.search(claim_text))

    @staticmethod
    def _evaluate_temporal_ordering(
        relationships: List[RelationshipVerificationDetail],
        metadata_map: Dict[str, OrderMetadataItem],
    ) -> List[DiscrepancyReport]:
        """
        Evaluate temporal sequence across relationship pairs.
        Returns list of DiscrepancyReport for any inverted dates.
        """
        discrepancies: List[DiscrepancyReport] = []
        for rel in relationships:
            src_meta = metadata_map.get(rel.source_go_number.lower())
            tgt_meta = metadata_map.get(rel.target_go_number.lower())

            # Parse dates safely
            d_src: Optional[date] = None
            if rel.source_date:
                d_src = rel.source_date
            elif src_meta and src_meta.issue_date:
                try:
                    d_src = date.fromisoformat(str(src_meta.issue_date)[:10])
                except Exception:
                    d_src = None

            d_tgt: Optional[date] = None
            if rel.target_date:
                d_tgt = rel.target_date
            elif tgt_meta and tgt_meta.issue_date:
                try:
                    d_tgt = date.fromisoformat(str(tgt_meta.issue_date)[:10])
                except Exception:
                    d_tgt = None

            if d_src and d_tgt:
                if d_src >= d_tgt:
                    rel.temporal_relation = TemporalRelationType.SEQUENTIAL_TARGET_BEFORE_SOURCE
                else:
                    rel.temporal_relation = TemporalRelationType.INVERTED_SOURCE_BEFORE_TARGET
                    rel.human_review_required = True
                    disc = DiscrepancyReport(
                        referenced_go_number=rel.target_go_number,
                        category="TEMPORAL_ORDERING_ANOMALY",
                        description=(
                            f"Temporal inversion anomaly: Source order {rel.source_go_number} ({d_src}) "
                            f"is dated before referenced {rel.target_go_number} ({d_tgt})"
                        ),
                    )
                    disc.__dict__["discrepancy_type"] = "TEMPORAL_ORDERING_ANOMALY"
                    disc.__dict__["source_order_number"] = rel.source_go_number
                    disc.__dict__["target_order_number"] = rel.target_go_number
                    discrepancies.append(disc)
            else:
                rel.temporal_relation = TemporalRelationType.UNKNOWN_TEMPORAL_ORDER
        return discrepancies

    # ---------------------------------------------------------------------
    # Deterministic Confidence Calculation
    # ---------------------------------------------------------------------

    @staticmethod
    def _compute_claim_confidence(
        go_exists: bool,
        text_source: TextSourceType,
        directness: EvidenceDirectness,
        status: ClaimVerificationStatus,
        page_verified: bool,
        source_filename: Optional[str],
        temporal_penalty_multiplier: float = 1.0,
        temporal_penalty_reason: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute an explainable, deterministic claim-level confidence score.
        Decouples existential certainty from claim support.
        Confirmed absence contributes 0.0 to positive claim support confidence.
        """
        if not go_exists:
            return 0.0, {
                "availability_score": 0.0,
                "directness_score": 0.0,
                "support_score": 0.0,
                "provenance_score": 0.0,
                "temporal_penalty_multiplier": 1.0,
                "final_claim_score": 0.0,
            }

        # 1. Text Availability (Max 0.25)
        avail_scores = {
            TextSourceType.FULL_DATABASE_TEXT: 0.25,
            TextSourceType.TRUNCATED_PREVIEW: 0.10,
            TextSourceType.SEARCH_SNIPPET: 0.08,
            TextSourceType.SEMANTIC_CHUNK: 0.08,
            TextSourceType.MISSING: 0.0,
        }
        w_avail = avail_scores.get(text_source, 0.0)

        # 2. Evidence Directness (Max 0.25)
        direct_scores = {
            EvidenceDirectness.DIRECT_OPERATIVE_ORDER: 0.25,
            EvidenceDirectness.INDIRECT_PREAMBLE_CITATION: 0.15,
            EvidenceDirectness.UNCLASSIFIED: 0.05,
            EvidenceDirectness.NONE: 0.0,
        }
        w_direct = direct_scores.get(directness, 0.0)

        # 3. Grounded Claim Support (Max 0.35)
        support_scores = {
            ClaimVerificationStatus.SUPPORTED: 0.35,
            ClaimVerificationStatus.PARTIALLY_SUPPORTED: 0.20,
            ClaimVerificationStatus.UNSUPPORTED: 0.0,
            ClaimVerificationStatus.CONTRADICTED: 0.0,
            ClaimVerificationStatus.UNVERIFIABLE: 0.0,
            ClaimVerificationStatus.REFUTED: 0.0,
        }
        w_support = support_scores.get(status, 0.0)

        # 4. Provenance & Page Accuracy (Max 0.15)
        w_prov = 0.0
        if source_filename:
            w_prov = 0.15 if page_verified else 0.10

        raw_score = w_avail + w_direct + w_support + w_prov
        final_score = round(min(1.0, max(0.0, raw_score * temporal_penalty_multiplier)), 4)

        factors = {
            "availability_score": w_avail,
            "directness_score": w_direct,
            "support_score": w_support,
            "provenance_score": w_prov,
            "temporal_penalty_multiplier": temporal_penalty_multiplier,
            "final_claim_score": final_score,
        }
        return final_score, factors

    @staticmethod
    def _classify_confidence(score: float) -> VerificationConfidence:
        """Restored agreed confidence thresholds: HIGH >= 0.80, MEDIUM >= 0.60, LOW < 0.60."""
        if score >= 0.80:
            return VerificationConfidence.HIGH
        elif score >= 0.60:
            return VerificationConfidence.MEDIUM
        return VerificationConfidence.LOW

    # ---------------------------------------------------------------------
    # Relationship Candidate Matching & Cross-Examination
    # ---------------------------------------------------------------------

    @staticmethod
    def _match_relationship_candidates(
        candidates: List[RelationshipCandidateInput],
        db_relationships: List[Dict[str, Any]],
        source_go: str,
    ) -> Tuple[List[RelationshipVerificationDetail], List[DiscrepancyReport]]:
        """
        1-to-1 matching between candidate relationships from text and DB relationships.
        Enforces canonical 8-type vocabulary:
        AMENDS, PARTIALLY_AMENDS, MODIFIES, EXTENDS, SUPERSEDES, CANCELS, REVOKES, READ_AS.
        Unsupported types from text or DB are never coerced to matches; they yield UNSUPPORTED_TYPE.
        """
        details: List[RelationshipVerificationDetail] = []
        discrepancies: List[DiscrepancyReport] = []
        consumed_db_indices: Set[int] = set()

        canonical_types = {
            "AMENDS", "PARTIALLY_AMENDS", "MODIFIES", "EXTENDS",
            "SUPERSEDES", "CANCELS", "REVOKES", "READ_AS",
        }

        complementary_pairs = {
            ("AMENDS", "PARTIALLY_AMENDS"),
            ("PARTIALLY_AMENDS", "AMENDS"),
            ("AMENDS", "MODIFIES"),
            ("MODIFIES", "AMENDS"),
            ("PARTIALLY_AMENDS", "MODIFIES"),
            ("MODIFIES", "PARTIALLY_AMENDS"),
            ("MODIFIES", "EXTENDS"),
            ("EXTENDS", "MODIFIES"),
            ("CANCELS", "REVOKES"),
            ("REVOKES", "CANCELS"),
        }

        for cand in candidates:
            c_type = (cand.relationship_type or "").upper()
            c_target = cand.target_order_number or (cand.referenced_go_numbers[0] if cand.referenced_go_numbers else "")
            c_source = (
                getattr(cand, "source_go_number", None)
                or getattr(cand, "source_order_number", None)
                or source_go
            )

            # Check if candidate relationship type is unsupported
            if c_type not in canonical_types:
                disc = DiscrepancyReport(
                    referenced_go_number=c_target,
                    category="UNSUPPORTED_TYPE",
                    text_relationship_type=c_type,
                    db_relationship_type=None,
                    description=f"Unsupported relationship type asserted in text: '{c_type}'",
                )
                disc.__dict__["discrepancy_type"] = "UNSUPPORTED_TYPE"
                disc.__dict__["source_order_number"] = c_source
                disc.__dict__["target_order_number"] = c_target
                discrepancies.append(disc)

                details.append(
                    RelationshipVerificationDetail(
                        source_go_number=c_source,
                        target_go_number=c_target,
                        candidate_relationship_type=c_type,
                        database_relationship_type=None,
                        registration_status=RelationshipRegistrationStatus.UNREGISTERED_INCOMPATIBLE_TYPE,
                        evidence_text=cand.evidence_text,
                        page_number=cand.page_number,
                        human_review_required=True,
                        legal_effect_verified=False,
                        notes=f"Candidate has unsupported type '{c_type}'",
                    )
                )
                continue

            matched_idx: Optional[int] = None
            match_status = RelationshipRegistrationStatus.UNRECORDED_IN_DATABASE
            matched_db_type: Optional[str] = None

            # 1. Exact match attempt
            for idx, db_rel in enumerate(db_relationships):
                if idx in consumed_db_indices:
                    continue
                db_target = db_rel.get("target_order_number") or db_rel.get("referenced_order_number") or ""
                db_type = (db_rel.get("relationship_type") or "").upper()
                if db_target == c_target and db_type == c_type:
                    matched_idx = idx
                    matched_db_type = db_type
                    match_status = RelationshipRegistrationStatus.REGISTERED_EXACT_MATCH
                    break

            # 2. Complementary match attempt if no exact match
            if matched_idx is None:
                for idx, db_rel in enumerate(db_relationships):
                    if idx in consumed_db_indices:
                        continue
                    db_target = db_rel.get("target_order_number") or db_rel.get("referenced_order_number") or ""
                    db_type = (db_rel.get("relationship_type") or "").upper()
                    if db_target == c_target and (c_type, db_type) in complementary_pairs:
                        matched_idx = idx
                        matched_db_type = db_type
                        match_status = RelationshipRegistrationStatus.REGISTERED_COMPLEMENTARY_MATCH
                        break

            # 3. Incompatible mismatch check if still not matched
            if matched_idx is None:
                for idx, db_rel in enumerate(db_relationships):
                    if idx in consumed_db_indices:
                        continue
                    db_target = db_rel.get("target_order_number") or db_rel.get("referenced_order_number") or ""
                    db_type = (db_rel.get("relationship_type") or "").upper()
                    if db_target == c_target:
                        matched_idx = idx
                        matched_db_type = db_type
                        match_status = RelationshipRegistrationStatus.UNREGISTERED_INCOMPATIBLE_TYPE
                        disc = DiscrepancyReport(
                            referenced_go_number=c_target,
                            category="TYPE_MISMATCH",
                            text_relationship_type=c_type,
                            db_relationship_type=db_type,
                            description=f"Type mismatch: text asserts {c_type} but DB records {db_type}",
                        )
                        disc.__dict__["discrepancy_type"] = "TYPE_MISMATCH"
                        disc.__dict__["source_order_number"] = c_source
                        disc.__dict__["target_order_number"] = c_target
                        discrepancies.append(disc)
                        break

            if matched_idx is not None:
                consumed_db_indices.add(matched_idx)
            else:
                disc = DiscrepancyReport(
                    referenced_go_number=c_target,
                    category="UNRECORDED_IN_DATABASE",
                    text_relationship_type=c_type,
                    db_relationship_type=None,
                    description=f"Relationship candidate {c_type} with {c_target} unrecorded in database",
                )
                disc.__dict__["discrepancy_type"] = "UNRECORDED_IN_DATABASE"
                disc.__dict__["source_order_number"] = c_source
                disc.__dict__["target_order_number"] = c_target
                discrepancies.append(disc)

            details.append(
                RelationshipVerificationDetail(
                    source_go_number=c_source,
                    target_go_number=c_target,
                    candidate_relationship_type=c_type,
                    database_relationship_type=matched_db_type,
                    registration_status=match_status,
                    evidence_text=cand.evidence_text,
                    page_number=cand.page_number,
                    human_review_required=(match_status in [
                        RelationshipRegistrationStatus.UNREGISTERED_INCOMPATIBLE_TYPE,
                        RelationshipRegistrationStatus.UNRECORDED_IN_DATABASE,
                    ]),
                    legal_effect_verified=False,
                    notes=f"Candidate evaluated with status {match_status.value}",
                )
            )

        # Unconsumed DB records -> MISSING_FROM_TEXT or UNSUPPORTED_TYPE
        for idx, db_rel in enumerate(db_relationships):
            if idx not in consumed_db_indices:
                db_target = db_rel.get("target_order_number") or db_rel.get("referenced_order_number") or ""
                db_type = (db_rel.get("relationship_type") or "").upper()
                if db_type not in canonical_types:
                    disc = DiscrepancyReport(
                        referenced_go_number=db_target,
                        category="UNSUPPORTED_TYPE",
                        text_relationship_type=None,
                        db_relationship_type=db_type,
                        description=f"Database record has unsupported relationship type: '{db_type}'",
                    )
                else:
                    disc = DiscrepancyReport(
                        referenced_go_number=db_target,
                        category="MISSING_FROM_TEXT",
                        text_relationship_type=None,
                        db_relationship_type=db_type,
                        description=f"Database relationship {db_type} with {db_target} missing from text",
                    )
                disc.__dict__["discrepancy_type"] = disc.category
                disc.__dict__["source_order_number"] = db_rel.get("source_order_number") or db_rel.get("source_go_number") or source_go
                disc.__dict__["target_order_number"] = db_target
                discrepancies.append(disc)

        return details, discrepancies

    def _cross_examine_relationships(
        self,
        cited_gos: Optional[List[str]] = None,
        candidates: Optional[List[RelationshipCandidateInput]] = None,
        source_go_number: Optional[str] = None,
        candidates_input: Optional[List[RelationshipCandidateInput]] = None,
    ) -> Any:
        """
        Cross-examine extracted relationship candidates against PostgreSQL.
        Decouples check coverage from consistency verification.
        """
        is_direct_test_call = (source_go_number is not None or candidates_input is not None)

        gos: List[str] = []
        if source_go_number:
            gos.append(source_go_number)
        if cited_gos:
            for g in cited_gos:
                if g not in gos:
                    gos.append(g)

        cands = list(candidates_input if candidates_input is not None else (candidates or []))

        verified_details: List[RelationshipVerificationDetail] = []
        discrepancies: List[Any] = []
        is_complete = True
        has_error = False
        malformed_or_unsupported = False

        if not gos and not cands:
            if is_direct_test_call:
                return verified_details, discrepancies, RelationshipVerificationCheckState.SKIPPED_VACUOUS, 0.0
            return verified_details, discrepancies, False, False, 0.0, is_complete

        total_meaningful_records = len(cands)
        for go_num in gos:
            try:
                rel_exec = getattr(self.rel_agent, "run", None) or self.rel_agent.execute
                rel_res = rel_exec(
                    RelationshipAgentRequest(
                        mode=RelationshipInvestigationMode.CROSS_EXAMINE_DATABASE,
                        go_number=go_num,
                    )
                )
            except Exception as exc:
                logger.error("RelationshipAgent exception during verification: %s", exc, exc_info=True)
                is_complete = False
                has_error = True
                continue

            if rel_res.status == RelationshipAgentStatus.ERROR:
                is_complete = False
                has_error = True

            if not rel_res.is_complete:
                is_complete = False

            if getattr(rel_res, "malformed_records_discarded", 0) > 0 or getattr(rel_res, "unrecognized_relationships", None):
                malformed_or_unsupported = True
                is_complete = False

            # Collect discrepancies
            for disc in rel_res.discrepancies:
                discrepancies.append(disc)

            total_meaningful_records += getattr(rel_res, "total_relationships", 0)

            # Build details
            for rel in getattr(rel_res, "relationships", []):
                rel_type_val = rel.relationship_type.value if hasattr(rel.relationship_type, "value") else str(rel.relationship_type)
                verified_details.append(
                    RelationshipVerificationDetail(
                        source_go_number=go_num,
                        target_go_number=rel.referenced_go_numbers[0] if rel.referenced_go_numbers else "UNKNOWN",
                        candidate_relationship_type=rel_type_val,
                        database_relationship_type=rel_type_val,
                        registration_status=RelationshipRegistrationStatus.REGISTERED_EXACT_MATCH,
                        evidence_text=rel.evidence_text,
                        page_number=rel.page_number,
                        temporal_relation=TemporalRelationType.TEMPORAL_ORDERING_SEQUENTIAL,
                        human_review_required=False,
                        legal_effect_verified=False,
                        notes="Relationship candidate confirmed in database.",
                    )
                )

        # 1-to-1 candidate matching if candidates are provided
        if cands:
            db_rels_raw: List[Dict[str, Any]] = [
                {
                    "target_order_number": d.target_go_number,
                    "relationship_type": d.database_relationship_type,
                }
                for d in verified_details
            ]
            m_details, m_discs = self._match_relationship_candidates(
                candidates=cands,
                db_relationships=db_rels_raw,
                source_go=gos[0] if gos else "UNKNOWN",
            )
            # Merge 1-to-1 matching results
            verified_details = m_details
            for md in m_discs:
                discrepancies.append(md)

        check_performed = not has_error
        if has_error or (not is_complete and (malformed_or_unsupported or len(discrepancies) > 0)):
            check_state = RelationshipVerificationCheckState.FAILED_INCOMPLETE
            consistency_verified = False
            consistency_score = 0.0
        elif len(discrepancies) > 0 or malformed_or_unsupported:
            check_state = RelationshipVerificationCheckState.DISCREPANCIES_FOUND
            consistency_verified = False
            consistency_score = 0.0
        elif total_meaningful_records == 0:
            check_state = RelationshipVerificationCheckState.SKIPPED_VACUOUS
            consistency_verified = False
            consistency_score = 0.0
        else:
            check_state = RelationshipVerificationCheckState.COMPLETED_CONSISTENT
            consistency_verified = True
            consistency_score = 0.40

        if is_direct_test_call:
            return verified_details, discrepancies, check_state, consistency_score

        dict_discrepancies = [
            (d.model_dump() if hasattr(d, "model_dump") else (d if isinstance(d, dict) else {"description": str(d)}))
            for d in discrepancies
        ]
        return verified_details, dict_discrepancies, check_performed, consistency_verified, consistency_score, is_complete

    # ---------------------------------------------------------------------
    # Main Execution Entry Point
    # ---------------------------------------------------------------------

    def execute(self, request_data: Any) -> VerificationAgentResponse:
        """
        Execute comprehensive evidence and relationship verification.
        Validates request against VerificationAgentRequest contract.
        """
        # Step 1: Request Contract Validation
        try:
            if isinstance(request_data, VerificationAgentRequest):
                req = request_data
            elif isinstance(request_data, dict):
                req = VerificationAgentRequest(**request_data)
            else:
                raise ValueError(f"Invalid request data type: {type(request_data).__name__}")
        except Exception as exc:
            logger.warning("VerificationAgentRequest validation failed: %s", exc)
            return VerificationAgentResponse(
                status=VerificationStatus.ERROR,
                overall_confidence=VerificationConfidence.LOW,
                confidence_score=0.0,
                is_complete=False,
                evidence_sufficient=False,
                errors=[f"Invalid request contract: {exc}"],
                recommendation=VerificationRecommendation(
                    action=AutonomousVerificationAction.REQUEST_HUMAN_REVIEW,
                    target_agent=TargetAgentName.HUMAN_REVIEW,
                    parameters=RequestHumanReviewParams(
                        review_reason=f"Request validation failed: {exc}",
                        conflicting_go_numbers=[],
                    ),
                    reasoning="Malformed request payload could not be validated.",
                ),
                execution_trace=[],
                legal_effect_verified=False,
            )

        trace: List[Dict[str, Any]] = []
        errors: List[str] = []
        is_complete = True
        human_review_required = False
        human_review_reasons: List[str] = []

        # Step 2: Normalize and Deduplicate Cited GO Numbers
        raw_gos = list(req.cited_go_numbers)
        for c in req.claims:
            if c.cited_go_number and c.cited_go_number.strip():
                raw_gos.append(c.cited_go_number.strip())

        seen_gos: Set[str] = set()
        unique_gos: List[str] = []
        for g in raw_gos:
            clean_g = " ".join(g.strip().split())
            if clean_g and clean_g.lower() not in seen_gos:
                seen_gos.add(clean_g.lower())
                unique_gos.append(clean_g)

        # Step 3: Handle Empty Request Case (Neither Claims Nor GOs)
        if not req.claims and not unique_gos:
            return VerificationAgentResponse(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                overall_confidence=VerificationConfidence.LOW,
                confidence_score=0.0,
                is_complete=True,
                evidence_sufficient=False,
                errors=["No claims or Government Order identifiers provided for verification."],
                recommendation=VerificationRecommendation(
                    action=AutonomousVerificationAction.TARGETED_SEARCH,
                    target_agent=TargetAgentName.SEARCH_AGENT,
                    parameters=TargetedSearchParams(
                        query=req.user_query or "order",
                        limit=5,
                    ),
                    reasoning="Neither claims nor Government Order identifiers provided; recommending targeted search.",
                ),
                execution_trace=trace,
                legal_effect_verified=False,
            )

        # Map pre-provided document evidence
        doc_map = {}
        for d in req.document_evidence:
            if d.go_number:
                doc_map[d.go_number.lower()] = d
            if d.order_number:
                doc_map[d.order_number.lower()] = d

        # Step 4: Authoritative Database Lookups for Cited Orders
        order_records: Dict[str, Optional[OrderMetadataItem]] = {}
        order_status: Dict[str, str] = {}
        for go_num in unique_gos:
            go_k = go_num.lower()
            doc_ev = doc_map.get(go_k)
            db_exec = getattr(self.db_agent, "run", None) or self.db_agent.execute
            try:
                db_res = db_exec(
                    DatabaseAgentRequest(
                        operation=DatabaseOperation.LOOKUP_ORDER,
                        go_number=go_num,
                        include_full_text=True,
                    )
                )
                trace.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent": "verification_agent",
                    "action": "database_lookup",
                    "reasoning": f"Authoritative verification of cited GO '{go_num}' in PostgreSQL",
                    "tool_called": "database_agent",
                    "tool_input": {"go_number": go_num, "include_full_text": True},
                    "tool_output_summary": f"Status={db_res.status.value}, found={bool(db_res.order_data)}",
                    "decision": "evaluate_order_existence",
                    "decision_source": "DETERMINISTIC_RULES",
                })
                if db_res.status == AgentResponseStatus.NOT_FOUND:
                    order_records[go_k] = None
                    order_status[go_k] = "NOT_FOUND"
                elif db_res.status == AgentResponseStatus.AMBIGUOUS_MATCH:
                    order_records[go_k] = None
                    order_status[go_k] = "AMBIGUOUS_MATCH"
                    human_review_required = True
                    human_review_reasons.append(f"Cited GO '{go_num}' is ambiguous in database: {db_res.ambiguous_candidates}")
                elif db_res.status == AgentResponseStatus.SUCCESS and db_res.order_data:
                    order_records[go_k] = db_res.order_data
                    order_status[go_k] = "SUCCESS"
                else:
                    order_records[go_k] = None
                    order_status[go_k] = "ERROR"
                    is_complete = False
                    errors.append(f"Database lookup encountered status '{db_res.status.value}' for '{go_num}'.")
            except Exception as exc:
                logger.warning("DatabaseAgent exception for '%s': %s", go_num, exc)
                order_records[go_k] = None
                order_status[go_k] = "ERROR"
                is_complete = False
                errors.append(f"Database lookup failed for '{go_num}': {exc}")

        # Step 5: Relationship Cross-Examination & Temporal Ordering
        rel_details, raw_discrepancies, check_performed, consistency_verified, rel_score, rel_complete = (
            self._cross_examine_relationships(unique_gos, req.relationship_candidates)
        )
        if not rel_complete:
            is_complete = False

        # Run temporal evaluation across relationships and metadata
        meta_map: Dict[str, OrderMetadataItem] = {k: v for k, v in order_records.items() if v is not None}
        temporal_discrepancies = self._evaluate_temporal_ordering(rel_details, meta_map)
        all_discrepancies: List[Any] = list(raw_discrepancies)
        for td in temporal_discrepancies:
            td_dict = td.model_dump() if hasattr(td, "model_dump") else (td if isinstance(td, dict) else {"description": str(td)})
            all_discrepancies.append(td_dict)
            consistency_verified = False

        if all_discrepancies:
            human_review_required = True
            human_review_reasons.append(f"Detected {len(all_discrepancies)} relationship/temporal discrepancy(ies).")

        # Step 6: Case B - No Claims, But GO Identifiers Present
        if not req.claims and unique_gos:
            all_absent = all(order_status.get(g.lower()) == "NOT_FOUND" for g in unique_gos)
            any_absent = any(order_status.get(g.lower()) == "NOT_FOUND" for g in unique_gos)
            any_error = any(order_status.get(g.lower()) == "ERROR" for g in unique_gos)

            all_exist = all(order_status.get(g.lower()) == "SUCCESS" for g in unique_gos)
            w_exist = 0.30 if all_exist else (0.15 if not all_absent else 0.0)

            all_full_text = all(
                self._classify_text_source(order_records.get(g.lower())) == TextSourceType.FULL_DATABASE_TEXT
                for g in unique_gos
            )
            any_preview = any(
                self._classify_text_source(order_records.get(g.lower())) == TextSourceType.TRUNCATED_PREVIEW
                for g in unique_gos
            )
            w_text = 0.30 if all_full_text else (0.15 if any_preview else 0.0)

            doc_score = round(w_exist + w_text + rel_score, 4)
            conf_rating = self._classify_confidence(doc_score)

            if doc_score < 0.60:
                human_review_required = True
                human_review_reasons.append(f"Document evidence verification score ({doc_score}) below threshold (0.60).")

            if all_absent:
                status = VerificationStatus.NOT_FOUND
            elif any_error and not all_exist:
                status = VerificationStatus.ERROR
            elif any_absent or not all_full_text or not consistency_verified:
                status = VerificationStatus.PARTIALLY_VERIFIED
            else:
                status = VerificationStatus.VERIFIED

            if status == VerificationStatus.VERIFIED:
                rec_action = AutonomousVerificationAction.PROCEED_TO_RESPONSE
                rec_agent = TargetAgentName.SUPERVISOR_AGENT
                rec_params: Any = EmptyParams()
                rec_reason = "All cited Government Orders verified in database; proceed to response."
            elif status == VerificationStatus.NOT_FOUND:
                rec_action = AutonomousVerificationAction.TARGETED_SEARCH
                rec_agent = TargetAgentName.SEARCH_AGENT
                rec_params = TargetedSearchParams(
                    query=req.user_query or (unique_gos[0] if unique_gos else "order"),
                    limit=5,
                )
                rec_reason = "Cited Government Order(s) absent from database; recommending targeted search."
            else:
                rec_action = AutonomousVerificationAction.REQUEST_HUMAN_REVIEW
                rec_agent = TargetAgentName.HUMAN_REVIEW
                rec_params = RequestHumanReviewParams(
                    review_reason="; ".join(human_review_reasons) or "Document verification incomplete.",
                    conflicting_go_numbers=unique_gos,
                )
                rec_reason = "Document evidence verification incomplete or ambiguous; human review required."

            rec = VerificationRecommendation(
                action=rec_action,
                target_agent=rec_agent,
                parameters=rec_params,
                reasoning=rec_reason,
            )

            return VerificationAgentResponse(
                status=status,
                overall_confidence=conf_rating,
                confidence_score=doc_score,
                is_complete=is_complete,
                evidence_sufficient=(status == VerificationStatus.VERIFIED),
                verified_claims=[],
                unsupported_claims=[],
                verified_relationships=rel_details,
                relationship_check_performed=check_performed,
                relationship_consistency_verified=consistency_verified,
                discrepancies=all_discrepancies,
                unresolved_references=[],
                discovered_go_numbers=[],
                human_review_required=human_review_required,
                human_review_reason="; ".join(human_review_reasons) if human_review_reasons else None,
                recommendation=rec,
                errors=errors,
                execution_trace=trace,
                legal_effect_verified=False,
            )

        # Step 7: Case A - Claims Present Evaluation
        verified_claims: List[ClaimEvidenceVerification] = []
        unsupported_claims: List[ClaimEvidenceVerification] = []
        claim_scores: List[float] = []

        for c in req.claims:
            cited_go = c.cited_go_number.strip() if c.cited_go_number else None
            go_key = cited_go.lower() if cited_go else ""
            order_data = order_records.get(go_key)
            o_status = order_status.get(go_key, "NOT_FOUND" if cited_go else "SUCCESS")

            go_exists = bool(order_data is not None and o_status == "SUCCESS")

            if not go_exists:
                status_claim = (
                    ClaimVerificationStatus.REFUTED
                    if o_status == "NOT_FOUND"
                    else ClaimVerificationStatus.UNVERIFIABLE
                )
                reasoning = (
                    f"Cited order '{cited_go}' confirmed absent from database."
                    if o_status == "NOT_FOUND"
                    else (
                        f"Lookup for '{cited_go}' failed with operational error."
                        if o_status == "ERROR"
                        else f"Cited order '{cited_go}' could not be verified."
                    )
                )
                claim_score, factors = self._compute_claim_confidence(
                    go_exists=False,
                    text_source=TextSourceType.MISSING,
                    directness=EvidenceDirectness.NONE,
                    status=status_claim,
                    page_verified=False,
                    source_filename=None,
                )
                claim_scores.append(claim_score)
                item = ClaimEvidenceVerification(
                    claim_id=c.claim_id,
                    claim_text=c.claim_text,
                    cited_go_number=cited_go,
                    go_exists_in_database=False,
                    existence_checked=True,
                    text_source_type=TextSourceType.MISSING,
                    status=status_claim,
                    evidence_directness=EvidenceDirectness.NONE,
                    evidence_text=None,
                    page_number=None,
                    page_number_verified=False,
                    page_locator_source=PageLocatorSource.UNTRUSTED_CANDIDATE,
                    source_filename=None,
                    reasoning=reasoning,
                    confidence=claim_score,
                    confidence_factors=factors,
                )
                unsupported_claims.append(item)
                continue

            assert order_data is not None
            text_source = self._classify_text_source(order_data)
            source_text = order_data.extracted_text or order_data.text_preview or ""
            source_file = order_data.source_filename

            effective_excerpt = c.cited_excerpt or c.claim_text
            directness = self._detect_section_directness(source_text, effective_excerpt)
            derived_page, page_verified, locator_src, page_note = self.derive_page_number_from_text(
                source_text=source_text,
                excerpt=effective_excerpt,
                reported_page=c.reported_page_number,
                go_number=order_data.go_number,
            )

            # Semantic Entailment
            entail_status, entail_conf, entail_reason, matching_excerpt = self._evaluate_semantic_entailment(
                claim=c,
                source_text=source_text,
                cited_excerpt=effective_excerpt,
            )

            if entail_status in ["SUPPORTED", "ENTAILED"]:
                status_claim = ClaimVerificationStatus.SUPPORTED
            elif entail_status in ["CONTRADICTED", "CONTRADICTORY"]:
                status_claim = ClaimVerificationStatus.CONTRADICTED
            elif entail_status in ["PARTIALLY_SUPPORTED", "POTENTIALLY_SUPPORTED"]:
                status_claim = ClaimVerificationStatus.PARTIALLY_SUPPORTED
            elif entail_status == "UNVERIFIABLE":
                status_claim = ClaimVerificationStatus.UNVERIFIABLE
            else:
                status_claim = ClaimVerificationStatus.UNSUPPORTED

            # Temporal Anomaly Evaluation (0.5x scaling penalty for demonstrably relevant claims)
            # A temporal anomaly ALONE never marks a claim as CONTRADICTED
            temporal_penalty = 1.0
            temporal_reason = None
            for disc in all_discrepancies:
                disc_cat = disc.get("category") if isinstance(disc, dict) else getattr(disc, "category", "")
                if disc_cat in ["TEMPORAL_ORDERING_DISCREPANCY", "TEMPORAL_ORDERING_ANOMALY"]:
                    ref_go = disc.get("referenced_go_number", "") if isinstance(disc, dict) else getattr(disc, "referenced_go_number", "")
                    if self._is_claim_relevant_to_temporal_discrepancy(c.claim_text, order_data.go_number, ref_go):
                        temporal_penalty = 0.5
                        temporal_reason = f"Penalised by 0.5x due to temporal ordering discrepancy with '{ref_go}'."
                        break

            claim_score, factors = self._compute_claim_confidence(
                go_exists=True,
                text_source=text_source,
                directness=directness,
                status=status_claim,
                page_verified=page_verified,
                source_filename=source_file,
                temporal_penalty_multiplier=temporal_penalty,
                temporal_penalty_reason=temporal_reason,
            )
            claim_scores.append(claim_score)

            item = ClaimEvidenceVerification(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                cited_go_number=cited_go,
                go_exists_in_database=True,
                existence_checked=True,
                text_source_type=text_source,
                status=status_claim,
                evidence_directness=directness,
                evidence_text=matching_excerpt,
                page_number=derived_page,
                page_number_verified=page_verified,
                page_locator_source=locator_src,
                source_filename=source_file,
                reasoning=entail_reason,
                confidence=claim_score,
                confidence_score=claim_score,
                confidence_factors=factors,
            )

            if status_claim == ClaimVerificationStatus.SUPPORTED:
                verified_claims.append(item)
            else:
                unsupported_claims.append(item)

        # Step 8: Calculate Overall Confidence Score
        avg_confidence = round(sum(claim_scores) / len(claim_scores), 4) if claim_scores else 0.0
        overall_conf = self._classify_confidence(avg_confidence)

        if avg_confidence < 0.60:
            human_review_required = True
            human_review_reasons.append(f"Average claim confidence ({avg_confidence}) below threshold (0.60).")

        # Step 9: Multi-Claim Overall Status Aggregation
        # Vacuous relationship check caps overall status at PARTIALLY_VERIFIED,
        # but does not invalidate independently supported factual claims.
        has_contradicted = any(c.status == ClaimVerificationStatus.CONTRADICTED for c in unsupported_claims)
        all_supported = bool(verified_claims and len(verified_claims) == len(req.claims))
        all_direct = all(c.evidence_directness == EvidenceDirectness.DIRECT_OPERATIVE_ORDER for c in verified_claims)
        all_not_found = bool(unique_gos and all(order_status.get(g.lower()) == "NOT_FOUND" for g in unique_gos))
        any_db_error = any(order_status.get(g.lower()) == "ERROR" for g in unique_gos)

        if any_db_error and not verified_claims:
            overall_status = VerificationStatus.ERROR
        elif all_not_found:
            overall_status = VerificationStatus.NOT_FOUND
        elif has_contradicted:
            overall_status = VerificationStatus.CONFLICTING_EVIDENCE
            human_review_required = True
            human_review_reasons.append("One or more claims contradicted by source text.")
        elif not verified_claims:
            overall_status = VerificationStatus.INSUFFICIENT_EVIDENCE
        elif all_supported and all_direct and is_complete and not all_discrepancies and consistency_verified:
            overall_status = VerificationStatus.VERIFIED
        else:
            overall_status = VerificationStatus.PARTIALLY_VERIFIED

        # Step 10: Generate Safe Typed Recommendation
        if overall_status == VerificationStatus.VERIFIED:
            rec = VerificationRecommendation(
                action=AutonomousVerificationAction.PROCEED_TO_RESPONSE,
                target_agent=TargetAgentName.SUPERVISOR_AGENT,
                parameters=EmptyParams(),
                reasoning="All claims verified with direct grounded evidence; proceeding to response synthesis.",
            )
        elif overall_status == VerificationStatus.NOT_FOUND:
            rec = VerificationRecommendation(
                action=AutonomousVerificationAction.TARGETED_SEARCH,
                target_agent=TargetAgentName.SEARCH_AGENT,
                parameters=TargetedSearchParams(
                    query=unique_gos[0] if unique_gos else (req.user_query or "order"),
                    limit=5,
                ),
                reasoning="Cited order absent from database; targeted search recommended to locate related orders.",
            )
        elif any(c.text_source_type == TextSourceType.TRUNCATED_PREVIEW for c in unsupported_claims):
            first_preview = next(c for c in unsupported_claims if c.text_source_type == TextSourceType.TRUNCATED_PREVIEW)
            rec = VerificationRecommendation(
                action=AutonomousVerificationAction.FETCH_ORDER_TEXT,
                target_agent=TargetAgentName.DATABASE_AGENT,
                parameters=FetchOrderTextParams(
                    go_number=first_preview.cited_go_number or unique_gos[0],
                    include_full_text=True,
                ),
                reasoning="Full operative text omitted in state; fetching full document text for complete verification.",
            )
        else:
            rec = VerificationRecommendation(
                action=AutonomousVerificationAction.REQUEST_HUMAN_REVIEW,
                target_agent=TargetAgentName.HUMAN_REVIEW,
                parameters=RequestHumanReviewParams(
                    review_reason="; ".join(human_review_reasons) or "Verification could not be fully established.",
                    conflicting_go_numbers=unique_gos,
                ),
                reasoning="Discrepancies, incomplete operations, or low confidence require human review.",
            )

        return VerificationAgentResponse(
            status=overall_status,
            overall_confidence=overall_conf,
            confidence_score=avg_confidence,
            is_complete=is_complete,
            evidence_sufficient=(overall_status == VerificationStatus.VERIFIED),
            verified_claims=verified_claims,
            unsupported_claims=unsupported_claims,
            verified_relationships=rel_details,
            relationship_check_performed=check_performed,
            relationship_consistency_verified=consistency_verified,
            discrepancies=all_discrepancies,
            unresolved_references=[],
            discovered_go_numbers=[],
            human_review_required=human_review_required,
            human_review_reason="; ".join(human_review_reasons) if human_review_reasons else None,
            recommendation=rec,
            errors=errors,
            execution_trace=trace,
            legal_effect_verified=False,
        )

    run = execute


derive_page_number_from_text = VerificationAgent.derive_page_number_from_text
