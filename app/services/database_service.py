"""
Database persistence service for Government Orders.

Saves extracted Government Order metadata, references
and relationships into PostgreSQL.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select

from app.config import DATASET_DIR, EXTRACTED_DIR
from app.database.connection import SessionLocal
from app.database.models import (
    GovernmentOrderDB,
    GOReferenceDB,
    GORelationshipDB,
    VerificationRecordDB,
)
from app.services.identifier_extraction_service import (
    IdentifierExtractionService,
)
from app.services.metadata_extraction_service import MetadataService


class DatabaseService:
    """Save extracted Government Order information in PostgreSQL."""

    def __init__(self) -> None:
        self.identifier_service = IdentifierExtractionService()
        self.metadata_service = MetadataService()

    @staticmethod
    def _parse_date(value: str | None):
        """Convert supported date formats into Python date objects."""

        if not value or value == "Unknown":
            return None

        formats = ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d")

        for date_format in formats:
            try:
                return datetime.strptime(value, date_format).date()
            except ValueError:
                continue

        raise ValueError(f"Unsupported date format: {value}")

    def save_document_record(
        self,
        filename: str,
        source_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract and save one Government Order, returning structured record data.

        Args:
            filename: Name of the extracted text file in EXTRACTED_DIR.
            source_filename: Optional original PDF filename in DATASET_DIR. If omitted,
                             defaults to '<filename.stem>.pdf'.

        Returns:
            Structured dictionary with insertion status, primary key ID,
            canonical GO number, department, dates, text, and counts.
        """
        text_path = EXTRACTED_DIR / filename

        if not text_path.exists():
            raise FileNotFoundError(
                f"Extracted text file not found: {text_path}"
            )

        if source_filename:
            safe_pdf_name = Path(source_filename).name
            pdf_path = DATASET_DIR / safe_pdf_name
        else:
            pdf_path = DATASET_DIR / f"{Path(filename).stem}.pdf"

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Original PDF not found: {pdf_path}"
            )

        # Read the extracted document.
        extracted_text = text_path.read_text(encoding="utf-8")

        # Extract structured identifiers and relationships.
        result = self.identifier_service.extract_from_text(
            extracted_text,
            filename,
        )

        # Extract metadata and verification status.
        metadata = self.metadata_service.extract_metadata(
            extracted_text,
            filename,
        )

        go_number = result.go_numbers[0].value

        if not go_number or go_number == "Unknown":
            raise ValueError(
                f"Cannot save {filename}: GO number is missing."
            )

        issue_date = self._parse_date(
            result.issue_dates[0].value
        )

        department = result.departments[0].value

        if department == "Unknown":
            department = None

        with SessionLocal() as session:
            with session.begin():

                # Avoid duplicate Government Orders.
                existing_order = session.scalar(
                    select(GovernmentOrderDB).where(
                        GovernmentOrderDB.go_number == go_number
                    )
                )

                if existing_order:
                    return {
                        "status": "ALREADY_EXISTS",
                        "go_id": existing_order.id,
                        "go_number": existing_order.go_number,
                        "department": existing_order.department,
                        "issue_date": (
                            existing_order.issue_date.isoformat()
                            if existing_order.issue_date
                            else None
                        ),
                        "source_filename": existing_order.source_filename,
                        "extracted_text_filename": existing_order.extracted_text_filename,
                        "extracted_text": existing_order.extracted_text,
                        "verification_status": existing_order.verification_status,
                        "references_count": 0,
                        "relationships_count": 0,
                        "message": (
                            f"Already exists: {go_number}. "
                            "No duplicate records inserted."
                        ),
                    }

                # Save the main Government Order.
                government_order = GovernmentOrderDB(
                    go_number=go_number,
                    issue_date=issue_date,
                    department=department,
                    source_filename=pdf_path.name,
                    extracted_text_filename=filename,
                    extracted_text=extracted_text,
                    verification_status=metadata[
                        "verification_status"
                    ],
                )

                session.add(government_order)
                session.flush()

                go_id = government_order.id

                # Save every numbered reference.
                for reference in result.references:
                    reference_record = GOReferenceDB(
                        source_go_id=go_id,
                        ordinal=reference.ordinal,
                        referenced_go_number=reference.go_number,
                        reference_date=self._parse_date(
                            reference.issue_date
                        ),
                        page_number=reference.page_number,
                        evidence_text=reference.evidence_text,
                    )

                    session.add(reference_record)

                # Save relationships.
                for relationship in result.relationships:

                    referenced_numbers = (
                        relationship.referenced_go_numbers
                    )

                    # Preserve evidence even when no GO number
                    # could be resolved.
                    if not referenced_numbers:
                        referenced_numbers = [None]

                    for referenced_number in referenced_numbers:

                        relationship_record = GORelationshipDB(
                            source_go_id=go_id,
                            relationship_type=(
                                relationship.relationship_type.value
                            ),
                            referenced_go_number=referenced_number,
                            matched_phrase=relationship.matched_phrase,
                            evidence_text=relationship.evidence_text,
                            page_number=relationship.page_number,
                            confidence=relationship.confidence.value,
                            extraction_method=(
                                relationship.extraction_method
                            ),
                            review_required=(
                                relationship.review_required
                            ),
                        )

                        session.add(relationship_record)
                        session.flush()

                        # Create a pending verification record.
                        verification_record = VerificationRecordDB(
                            go_id=go_id,
                            relationship_id=relationship_record.id,
                            verification_type="RELATIONSHIP",
                            verification_status="PENDING",
                        )

                        session.add(verification_record)

                return {
                    "status": "INSERTED",
                    "go_id": government_order.id,
                    "go_number": government_order.go_number,
                    "department": government_order.department,
                    "issue_date": (
                        government_order.issue_date.isoformat()
                        if government_order.issue_date
                        else None
                    ),
                    "source_filename": government_order.source_filename,
                    "extracted_text_filename": filename,
                    "extracted_text": extracted_text,
                    "verification_status": government_order.verification_status,
                    "references_count": len(result.references),
                    "relationships_count": len(result.relationships),
                    "message": (
                        f"Successfully saved: {go_number}\n"
                        f"References: {len(result.references)}\n"
                        f"Relationships: {len(result.relationships)}"
                    ),
                }

    def save_document(self, filename: str) -> str:
        """Extract and save one Government Order (backward-compatible)."""
        record = self.save_document_record(filename)
        return record["message"]