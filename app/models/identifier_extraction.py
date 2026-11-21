
"""
Models for Government Order identifier and relationship extraction.
"""

from enum import Enum

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Supported Government Order relationship types."""

    AMENDS = "AMENDS"
    PARTIALLY_AMENDS = "PARTIALLY_AMENDS"
    MODIFIES = "MODIFIES"
    EXTENDS = "EXTENDS"
    SUPERSEDES = "SUPERSEDES"
    CANCELS = "CANCELS"
    REVOKES = "REVOKES"
    READ_AS = "READ_AS"


class ExtractionConfidence(str, Enum):
    """Confidence level for an extracted value."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ExtractedIdentifier(BaseModel):
    """An identifier extracted from a Government Order."""

    value: str
    confidence: ExtractionConfidence
    extraction_method: str

    page_number: int | None = None
    evidence_text: str | None = None


class ExtractedReference(BaseModel):
    """A Government Order referenced in another order."""

    ordinal: int
    go_number: str
    page_number: int | None = None
    evidence_text: str | None = None
    issue_date: str | None = None

class ExtractedRelationship(BaseModel):
    """A relationship detected in Government Order text."""

    relationship_type: RelationshipType

    # Phrase that indicates the relationship.
    matched_phrase: str

    # Exact sentence or text supporting the relationship.
    evidence_text: str | None = None

    page_number: int | None = None

    # References associated with this relationship.
    reference_ordinals: list[int] = Field(
        default_factory=list
    )

    # Actual Government Order numbers resolved from ordinals.
    referenced_go_numbers: list[str] = Field(
        default_factory=list
    )

    # Confidence in the extracted relationship.
    confidence: ExtractionConfidence = (
        ExtractionConfidence.MEDIUM
    )

    extraction_method: str = "RULE_BASED"

    # Uncertain relationships should be checked before acceptance.
    review_required: bool = True


class IdentifierExtractionResult(BaseModel):
    """Complete identifier and relationship extraction result."""

    document_id: str

    go_numbers: list[ExtractedIdentifier] = Field(
        default_factory=list
    )

    issue_dates: list[ExtractedIdentifier] = Field(
        default_factory=list
    )

    departments: list[ExtractedIdentifier] = Field(
        default_factory=list
    )

    references: list[ExtractedReference] = Field(
        default_factory=list
    )

    relationships: list[ExtractedRelationship] = Field(
        default_factory=list
    )

    review_flags: list[str] = Field(
        default_factory=list
    )
