"""
FastAPI Routes for Kerala Government Orders Multi-Agent System.
"""

import logging
import re
from pathlib import Path
from typing import Generator, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.api.auth import verify_admin_api_key
from app.api.schemas import (
    DocumentDetailResponse,
    DocumentRelationshipsResponse,
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReferenceItemSchema,
    RelationshipItemSchema,
    ResumeRequest,
    RetryIndexResponse,
    SearchRequest,
    SearchResponse,
    ThreadStatusResponse,
)
from app.api.service import WorkflowService
from app.agents.tools.database_tool import DatabaseTool
from app.config import DATASET_DIR
from app.database.connection import SessionLocal
from app.database.models import (
    GovernmentOrderDB,
    GOReferenceDB,
    GORelationshipDB,
    VerificationRecordDB,
)
from app.services.ingestion_service import IngestionService
from app.services.relationship_resolution import (
    resolve_incoming_relationships_for_order,
    resolve_relationship_verification_status as _resolve_relationship_verification_status,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_workflow_service() -> WorkflowService:
    """Dependency injection provider for WorkflowService."""
    return WorkflowService.get_instance()


def get_ingestion_service() -> IngestionService:
    """Dependency injection provider for IngestionService."""
    return IngestionService()


def get_db() -> Generator[Session, None, None]:
    """Dependency injection provider for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
    if ":" in fn:
        return False
    if Path(fn).is_absolute():
        return False
    if not fn.lower().endswith(".pdf"):
        return False
    return True


def _resolve_relationship_verification_status(
    records: List[VerificationRecordDB],
) -> str:
    """
    Resolve verification status for a relationship from its verification records.
    Resolution rule:
    - If no records exist, return "UNRECORDED".
    - Prefer the record with the latest effective timestamp:
      rec.verified_at if rec.verified_at is not None else rec.created_at.
    - Deterministic tie-breaker when timestamps are equal: rec.id descending.
    - An older VERIFIED record does not override a newer REJECTED record.
    """
    if not records:
        return "UNRECORDED"

    def sort_key(rec: VerificationRecordDB):
        effective_ts = (
            rec.verified_at.timestamp()
            if rec.verified_at
            else (rec.created_at.timestamp() if rec.created_at else 0.0)
        )
        rec_id = rec.id or 0
        return (effective_ts, rec_id)

    latest_record = max(records, key=sort_key)
    return latest_record.verification_status or "PENDING"


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness and Service Dependency Health Check",
)
def health_check(
    service: WorkflowService = Depends(get_workflow_service),
) -> HealthResponse:
    """
    Check the liveness of the API and availability of Ollama, PostgreSQL, and Qdrant.
    """
    try:
        return service.check_health()
    except Exception as exc:
        logger.exception("Health check encountered an unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed unexpectedly.",
        )


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Workflow"],
    summary="Submit Natural-Language Query to Multi-Agent Workflow",
)
def submit_query(
    request: QueryRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> QueryResponse:
    """
    Submit a natural-language question or Government Order inquiry.
    Dispatches to the LangGraph Supervisor Agent without bypassing the multi-agent architecture.
    """
    try:
        return service.execute_query(
            query=request.query,
            thread_id=request.thread_id,
            search_parameters={"limit": 10},
        )
    except Exception as exc:
        logger.exception("Query execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {type(exc).__name__}",
        )


@router.post(
    "/api/v1/resume",
    response_model=QueryResponse,
    tags=["Workflow"],
    summary="Resume Paused Human-Review Workflow",
)
def resume_workflow(
    request: ResumeRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> QueryResponse:
    """
    Provide clarification to a workflow interrupted at human_review_node.
    """
    try:
        return service.resume_thread(thread_id=request.thread_id, feedback=request.feedback)
    except KeyError as ke:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ke).strip("'\""),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        logger.exception("Resume execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow resume failed: {type(exc).__name__}",
        )


@router.get(
    "/api/v1/status/{thread_id}",
    response_model=ThreadStatusResponse,
    tags=["Workflow"],
    summary="Get Thread Execution and Checkpoint Status",
)
def get_thread_status(
    thread_id: str,
    service: WorkflowService = Depends(get_workflow_service),
) -> ThreadStatusResponse:
    """
    Inspect the checkpoint state of an existing workflow thread.
    Note: State is held in-memory and will not persist across server restarts.
    """
    if not thread_id or not thread_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id cannot be empty or whitespace.",
        )
    try:
        return service.get_thread_status(thread_id=thread_id.strip())
    except Exception as exc:
        logger.exception("Status check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve thread status.",
        )


@router.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search Government Orders by keyword, GO number, or semantics",
)
def search_documents(
    request: SearchRequest,
    service: WorkflowService = Depends(get_workflow_service),
) -> SearchResponse:
    """
    Public search endpoint for Government Orders.
    Supports KEYWORD, SEMANTIC, and HYBRID search modes, department filtering,
    and result pagination limits.
    """
    try:
        return service.search_documents(request)
    except Exception as exc:
        logger.exception("Search execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search execution failed: {type(exc).__name__}",
        )


@router.get(
    "/api/v1/documents/{go_id}",
    response_model=DocumentDetailResponse,
    tags=["Documents"],
    summary="Get details, references, and PDF availability for a Government Order",
)
def get_document_detail(
    go_id: int,
    db: Session = Depends(get_db),
) -> DocumentDetailResponse:
    """
    Public endpoint retrieving metadata, numbered references ('Read' section),
    and verified PDF availability for a Government Order.
    """
    if go_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="go_id must be a positive integer.",
        )

    order = db.scalar(
        select(GovernmentOrderDB).where(GovernmentOrderDB.id == go_id)
    )
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Government Order with ID {go_id} not found.",
        )

    # Validate PDF availability
    pdf_url: Optional[str] = None
    if _is_safe_filename(order.source_filename):
        resolved_dataset_dir = DATASET_DIR.resolve()
        target_path = (resolved_dataset_dir / order.source_filename).resolve()
        try:
            if target_path.is_relative_to(resolved_dataset_dir) and target_path.is_file():
                with target_path.open("rb") as f:
                    header = f.read(5)
                if len(header) >= 5 and header.startswith(b"%PDF-"):
                    pdf_url = f"/api/v1/documents/{go_id}/pdf"
        except Exception as exc:
            logger.warning("Error checking PDF availability for go_id=%d: %s", go_id, exc)
            pdf_url = None

    # Fetch references ordered by ordinal
    ref_stmt = (
        select(GOReferenceDB)
        .where(GOReferenceDB.source_go_id == go_id)
        .order_by(GOReferenceDB.ordinal)
    )
    references_db = db.scalars(ref_stmt).all()
    references_list = [
        ReferenceItemSchema(
            id=ref.id,
            ordinal=ref.ordinal,
            referenced_go_number=ref.referenced_go_number,
            reference_date=str(ref.reference_date) if ref.reference_date else None,
            page_number=ref.page_number,
            evidence_text=ref.evidence_text,
        )
        for ref in references_db
    ]

    # Calculate distinct relationships count:
    # 1. Outgoing distinct IDs (source_go_id == go_id)
    out_stmt = select(GORelationshipDB.id).where(GORelationshipDB.source_go_id == go_id)
    outgoing_ids = set(db.scalars(out_stmt).all())

    # 2. Incoming candidate matching (source_go_id != go_id)
    incoming_filters = [GORelationshipDB.referenced_go_number == order.go_number]
    parsed_target = DatabaseTool._parse_go_structure(order.go_number)
    if parsed_target:
        num, year, dept = parsed_target
        incoming_filters.append(GORelationshipDB.referenced_go_number.ilike(f"%{num}/{year}%"))
        if dept:
            incoming_filters.append(GORelationshipDB.referenced_go_number.ilike(f"%{num}/{year}/{dept}%"))

    core_match = re.search(r"\d+/\d{4}/[\w&.-]+", order.go_number)
    if core_match:
        core_ident = core_match.group(0)
        incoming_filters.append(GORelationshipDB.referenced_go_number.ilike(f"%{core_ident}%"))

    in_stmt = select(GORelationshipDB).where(
        GORelationshipDB.source_go_id != go_id,
        or_(*incoming_filters),
    )
    incoming_candidates = db.scalars(in_stmt).all()

    incoming_ids = set()
    for rel in incoming_candidates:
        ref_no = rel.referenced_go_number
        if ref_no == order.go_number or DatabaseTool._are_structurally_equivalent(ref_no, order.go_number):
            incoming_ids.add(rel.id)

    total_relationships_count = len(outgoing_ids) + len(incoming_ids)

    return DocumentDetailResponse(
        id=order.id,
        go_number=order.go_number,
        issue_date=str(order.issue_date) if order.issue_date else None,
        department=order.department,
        source_filename=order.source_filename,
        pdf_url=pdf_url,
        verification_status=order.verification_status,
        extracted_text_length=len(order.extracted_text or ""),
        references_count=len(references_list),
        relationships_count=total_relationships_count,
        references=references_list,
    )


@router.get(
    "/api/v1/documents/{go_id}/relationships",
    response_model=DocumentRelationshipsResponse,
    tags=["Documents"],
    summary="Get outgoing and incoming relationships for a Government Order",
)
def get_document_relationships(
    go_id: int,
    db: Session = Depends(get_db),
) -> DocumentRelationshipsResponse:
    """
    Public endpoint retrieving incoming and outgoing relationships for a Government Order.
    Resolves verification records per relationship and maintains legal_effect_verified=False.
    """
    if go_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="go_id must be a positive integer.",
        )

    order = db.scalar(
        select(GovernmentOrderDB).where(GovernmentOrderDB.id == go_id)
    )
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Government Order with ID {go_id} not found.",
        )

    # 1. Fetch outgoing relationships (source_go_id == go_id)
    out_stmt = (
        select(GORelationshipDB)
        .where(GORelationshipDB.source_go_id == go_id)
        .order_by(GORelationshipDB.id)
    )
    outgoing_rels = db.scalars(out_stmt).all()

    # Retrieve verification records for outgoing relationships
    outgoing_rel_ids = [r.id for r in outgoing_rels]
    ver_records_by_outgoing_rel: dict[int, list[VerificationRecordDB]] = {}
    if outgoing_rel_ids:
        ver_stmt = select(VerificationRecordDB).where(
            VerificationRecordDB.relationship_id.in_(outgoing_rel_ids)
        )
        for vrec in db.scalars(ver_stmt).all():
            if vrec.relationship_id is not None:
                ver_records_by_outgoing_rel.setdefault(vrec.relationship_id, []).append(vrec)

    # Map outgoing items
    outgoing_schemas = [
        RelationshipItemSchema(
            id=rel.id,
            direction="OUTGOING",
            source_go_id=order.id,
            source_go_number=order.go_number,
            target_go_number=rel.referenced_go_number,
            relationship_type=rel.relationship_type,
            matched_phrase=rel.matched_phrase,
            evidence_text=rel.evidence_text,
            page_number=rel.page_number,
            confidence=rel.confidence or "MEDIUM",
            extraction_method=rel.extraction_method or "RULE_BASED",
            review_required=rel.review_required if rel.review_required is not None else True,
            verification_status=_resolve_relationship_verification_status(
                ver_records_by_outgoing_rel.get(rel.id, [])
            ),
            legal_effect_verified=False,
        )
        for rel in outgoing_rels
    ]

    # 2. Fetch incoming relationships using the shared authoritative helper
    incoming_schemas, _ = resolve_incoming_relationships_for_order(
        order_id=order.id,
        go_number=order.go_number,
        db=db,
    )

    return DocumentRelationshipsResponse(
        go_id=order.id,
        go_number=order.go_number,
        outgoing_count=len(outgoing_schemas),
        incoming_count=len(incoming_schemas),
        outgoing_relationships=outgoing_schemas,
        incoming_relationships=incoming_schemas,
    )


@router.get(
    "/api/v1/documents/{go_id}/pdf",
    tags=["Documents"],
    summary="View or download Government Order PDF",
)
def get_document_pdf(
    go_id: int,
    download: bool = Query(
        False,
        description="If True, serve as attachment download; if False, serve inline for viewing.",
    ),
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Public endpoint providing secure PDF viewing or downloading.
    Enforces strict filename validation, path traversal prevention, containment
    within DATASET_DIR, and PDF header signature verification.
    """
    if go_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="go_id must be a positive integer.",
        )

    order = db.scalar(
        select(GovernmentOrderDB).where(GovernmentOrderDB.id == go_id)
    )
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Government Order with ID {go_id} not found.",
        )

    # 1. Strict filename check
    if not _is_safe_filename(order.source_filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid source filename recorded for this Government Order.",
        )

    # 2. Path containment check
    resolved_dataset_dir = DATASET_DIR.resolve()
    target_path = (resolved_dataset_dir / order.source_filename).resolve()
    if not target_path.is_relative_to(resolved_dataset_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal detected.",
        )

    # 3. File existence check
    if not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found on disk.",
        )

    # 4. Magic signature verification
    try:
        with target_path.open("rb") as f:
            header = f.read(5)
    except Exception as exc:
        logger.exception("Failed to read PDF file '%s': %s", target_path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the PDF file from disk.",
        )

    if len(header) < 5 or not header.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File on disk is not a valid PDF document.",
        )

    disposition_type = "attachment" if download else "inline"
    return FileResponse(
        path=str(target_path),
        media_type="application/pdf",
        filename=order.source_filename,
        content_disposition_type=disposition_type,
    )


@router.post(
    "/api/v1/admin/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Admin"],
    summary="Upload and ingest a new Government Order PDF",
)
async def upload_government_order_pdf(
    file: UploadFile = File(...),
    admin_auth: bool = Depends(verify_admin_api_key),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    """
    Protected administrator endpoint to upload, validate, extract, persist,
    and index a new Kerala Government Order PDF.
    """
    # 1. Filename extension validation
    raw_filename = file.filename or ""
    clean_filename = Path(raw_filename).name
    if not clean_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Uploaded file must have a .pdf extension.",
        )

    # 2. Safe chunked stream reading with 25 MB limit
    max_upload_size = 25 * 1024 * 1024  # 25 MB
    chunk_size = 64 * 1024  # 64 KB
    chunks = []
    total_bytes = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_upload_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Uploaded file exceeds the maximum allowed size of 25 MB.",
            )
        chunks.append(chunk)

    pdf_bytes = b"".join(chunks)

    # 3. Magic signature verification (do not trust client MIME type or extension alone)
    if len(pdf_bytes) < 5 or not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF document. File content does not match the PDF specification signature (%PDF-).",
        )

    # 4. Orchestrate ingestion
    try:
        result = ingestion_service.ingest_new_order_pdf(
            pdf_bytes=pdf_bytes,
            filename=clean_filename,
        )
        return DocumentUploadResponse(**result)
    except ValueError as val_err:
        logger.warning("Unprocessable document upload: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception(
            "Unexpected error during PDF ingestion for '%s': %s",
            clean_filename,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the uploaded Government Order PDF.",
        )


@router.post(
    "/api/v1/admin/documents/{go_id}/retry-index",
    response_model=RetryIndexResponse,
    tags=["Admin"],
    summary="Retry Qdrant vector indexing for an already-saved Government Order",
)
def retry_document_indexing(
    go_id: int,
    admin_auth: bool = Depends(verify_admin_api_key),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> RetryIndexResponse:
    """
    Protected administrator endpoint to re-index an existing Government Order
    using its committed database text and Qdrant safe-indexing mechanism.
    """
    if go_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="go_id must be a positive integer.",
        )

    try:
        result = ingestion_service.retry_index_document(go_id=go_id)
        return RetryIndexResponse(**result)
    except ValueError as val_err:
        logger.warning(
            "Retry indexing rejected for go_id=%d: %s", go_id, val_err
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception(
            "Unexpected error during retry indexing for go_id=%d: %s",
            go_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector re-indexing failed for Government Order ID {go_id}.",
        )

