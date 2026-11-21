
"""
Identifier extraction service for Government Orders.

Combines metadata, references and relationships
into one structured extraction result.
"""

from pathlib import Path

from app.config import EXTRACTED_DIR
from app.models.identifier_extraction import (
    ExtractedIdentifier,
    IdentifierExtractionResult,
    ExtractionConfidence,
)
from app.services.metadata_extraction_service import MetadataService
from app.services.relationship_extraction_service import (
    RelationshipExtractionService,
)


class IdentifierExtractionService:
    """Coordinate metadata, reference and relationship extraction."""

    def __init__(self) -> None:
        self.metadata_service = MetadataService()
        self.relationship_service = RelationshipExtractionService()

    @staticmethod
    def _get_confidence(value: str) -> ExtractionConfidence:
        """Assign basic confidence based on whether a value was extracted."""

        if not value or value == "Unknown":
            return ExtractionConfidence.LOW

        return ExtractionConfidence.MEDIUM

    def extract_from_text(
        self,
        text: str,
        filename: str,
    ) -> IdentifierExtractionResult:
        """Extract all identifiers and relationships from one document."""

        metadata = self.metadata_service.extract_metadata(
            text,
            filename,
        )

        references = self.relationship_service.extract_references(
            text
        )

        relationships = self.relationship_service.extract_relationships(
            text
        )

        relationships = (
            self.relationship_service.resolve_referenced_go_numbers(
                relationships,
                references,
            )
        )

        go_numbers = [
            ExtractedIdentifier(
                value=metadata["go_number"],
                confidence=self._get_confidence(
                    metadata["go_number"]
                ),
                extraction_method="METADATA_EXTRACTION",
            )
        ]

        issue_dates = [
            ExtractedIdentifier(
                value=metadata["date"],
                confidence=self._get_confidence(
                    metadata["date"]
                ),
                extraction_method="METADATA_EXTRACTION",
            )
        ]

        departments = [
            ExtractedIdentifier(
                value=metadata["department"],
                confidence=self._get_confidence(
                    metadata["department"]
                ),
                extraction_method="METADATA_EXTRACTION",
            )
        ]

        review_flags = []

        if metadata["verification_status"] != "Extracted - Verify":
            review_flags.append(
                "Document metadata requires manual verification."
            )


        for relationship in relationships:
            if relationship.review_required:
                review_flags.append(
                    f"Relationship {relationship.relationship_type.value} "
                    f"on page {relationship.page_number} requires review."
                )

            if (
                relationship.reference_ordinals
                and not relationship.referenced_go_numbers
            ):
                review_flags.append(
                    f"Could not resolve referenced GO numbers "
                    f"on page {relationship.page_number}."
                )

        return IdentifierExtractionResult(
            document_id=filename,
            go_numbers=go_numbers,
            issue_dates=issue_dates,
            departments=departments,
            references=references,
            relationships=relationships,
            review_flags=review_flags,
        )

    def extract_from_file(
        self,
        filename: str,
    ) -> IdentifierExtractionResult:
        """Extract identifiers from a previously extracted text file."""

        text_path = EXTRACTED_DIR / filename

        if not text_path.exists():
            raise FileNotFoundError(
                f"Extracted text file not found: {text_path}"
            )

        text = text_path.read_text(encoding="utf-8")

        return self.extract_from_text(
            text=text,
            filename=filename,
        )
