"""
SQLAlchemy models for the Government Orders database.
These models map to the existing PostgreSQL tables.
"""

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class GovernmentOrderDB(Base):
    """Database model for Government Orders."""

    __tablename__ = "government_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    go_number: Mapped[str] = mapped_column(
        String(150), unique=True, nullable=False
    )
    issue_date: Mapped[date | None] = mapped_column(Date)
    department: Mapped[str | None] = mapped_column(String(255))
    source_filename: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    extracted_text_filename: Mapped[str | None] = mapped_column(
        String(255)
    )
    extracted_text: Mapped[str | None] = mapped_column(Text)
    verification_status: Mapped[str | None] = mapped_column(
        String(50)
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.now()
    )


class GOReferenceDB(Base):
    """Database model for references to other Government Orders."""

    __tablename__ = "go_references"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_go_id: Mapped[int] = mapped_column(
        ForeignKey("government_orders.id"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    referenced_go_number: Mapped[str] = mapped_column(
        String(150), nullable=False
    )
    reference_date: Mapped[date | None] = mapped_column(Date)
    page_number: Mapped[int | None] = mapped_column(Integer)
    evidence_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.now()
    )


class GORelationshipDB(Base):
    """Database model for relationships between Government Orders."""

    __tablename__ = "go_relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_go_id: Mapped[int] = mapped_column(
        ForeignKey("government_orders.id"), nullable=False
    )
    relationship_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    referenced_go_number: Mapped[str | None] = mapped_column(
        String(150), nullable=True
    )
    matched_phrase: Mapped[str | None] = mapped_column(Text)
    evidence_text: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[str | None] = mapped_column(
        String(20), server_default="MEDIUM"
    )
    extraction_method: Mapped[str | None] = mapped_column(
        String(50), server_default="RULE_BASED"
    )
    review_required: Mapped[bool | None] = mapped_column(
        Boolean, server_default="true"
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.now()
    )


class VerificationRecordDB(Base):
    """Database model for manual verification records."""

    __tablename__ = "verification_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    go_id: Mapped[int] = mapped_column(
        ForeignKey("government_orders.id"), nullable=False
    )
    relationship_id: Mapped[int | None] = mapped_column(
        ForeignKey("go_relationships.id")
    )
    verification_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    verification_status: Mapped[str | None] = mapped_column(
        String(30), server_default="PENDING"
    )
    reviewer_notes: Mapped[str | None] = mapped_column(Text)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.now()
    )