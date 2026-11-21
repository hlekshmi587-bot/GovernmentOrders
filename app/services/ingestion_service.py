"""
Service for processing, ingesting, and indexing Government Order PDFs.
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.config import DATASET_DIR, EXTRACTED_DIR
from app.database.connection import SessionLocal
from app.database.models import GovernmentOrderDB
from app.services.chunking_service import chunk_document_text
from app.services.database_service import DatabaseService
from app.services.embedding_service import EmbeddingService
from app.services.pdf_classifier import PDFClassifier
from app.services.pdf_service import PDFService
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class IngestionService:
    """Classify, process, persist, and index Government Order PDFs."""

    def __init__(
        self,
        pdf_service: Optional[PDFService] = None,
        pdf_classifier: Optional[PDFClassifier] = None,
        database_service: Optional[DatabaseService] = None,
        qdrant_service: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
    ) -> None:
        self.pdf_service = pdf_service or PDFService()
        self.pdf_classifier = pdf_classifier or PDFClassifier()
        self.database_service = database_service or DatabaseService()
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service

    def process_pdf(self, pdf_path: Path) -> Path:
        """Classify a PDF, extract its text and save the result."""

        classification = self.pdf_classifier.classify_pdf(pdf_path)

        print(
            f"PDF type: {classification['overall_type']} "
            f"- {pdf_path.name}"
        )

        extracted_text = self.pdf_service.extract_text(pdf_path)

        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

        output_path = EXTRACTED_DIR / f"{pdf_path.stem}.txt"

        output_path.write_text(
            extracted_text,
            encoding="utf-8",
        )

        return output_path

    def process_all_pdfs(self) -> list[Path]:
        """Process every PDF in the dataset directory."""

        pdf_files = sorted(DATASET_DIR.glob("*.pdf"))
        processed_files: list[Path] = []

        for index, pdf_path in enumerate(pdf_files, start=1):
            print(
                f"\nProcessing {index}/{len(pdf_files)}: "
                f"{pdf_path.name}"
            )

            output_path = self.process_pdf(pdf_path)
            processed_files.append(output_path)

            print(f"Saved: {output_path.name}")

        return processed_files

    def ingest_new_order_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Orchestrate full ingestion of a validated Government Order PDF:
        1. Save original PDF with collision-safe name to dataset/.
        2. Classify document and extract text using PyMuPDF + Tesseract OCR.
        3. Validate that extracted text is non-empty and usable.
        4. Save extracted text to storage/extracted/.
        5. Atomically persist document, references, relationships in PostgreSQL.
        6. On duplicate: unlink newly created files, return ALREADY_EXISTS.
        7. Chunk text and index into local Qdrant collection with version staging.
        8. On Qdrant failure: preserve PostgreSQL record and disk files, return INDEXING_FAILED.
        """
        clean_name = Path(filename).name
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

        # Collision prevention: never overwrite an existing file on disk
        target_pdf_path = DATASET_DIR / clean_name
        if target_pdf_path.exists():
            unique_token = uuid.uuid4().hex[:8]
            clean_name = f"{Path(clean_name).stem}_{unique_token}.pdf"
            target_pdf_path = DATASET_DIR / clean_name

        created_files: List[Path] = []

        try:
            # 1. Save physical PDF to dataset directory
            target_pdf_path.write_bytes(pdf_bytes)
            created_files.append(target_pdf_path)

            # 2. Classify document
            classification = self.pdf_classifier.classify_pdf(target_pdf_path)
            if not classification.get("pages"):
                raise ValueError(
                    f"PDF document '{clean_name}' contains no readable pages."
                )

            # 3. Extract text (PyMuPDF with OCR fallback)
            extracted_text = self.pdf_service.extract_text(target_pdf_path)

            # Validate extracted text usability (must contain genuine content beyond headers)
            cleaned_content = re.sub(
                r"--- Page \d+ ---|--- Embedded Text ---|--- OCR Text ---",
                "",
                extracted_text,
            ).strip()
            if not cleaned_content:
                raise ValueError(
                    f"Extracted text from '{clean_name}' is empty or unusable."
                )

            # 4. Save extracted text to storage/extracted/
            txt_path = EXTRACTED_DIR / f"{target_pdf_path.stem}.txt"
            txt_path.write_text(extracted_text, encoding="utf-8")
            created_files.append(txt_path)

            # 5. Persist to PostgreSQL via DatabaseService
            db_record = self.database_service.save_document_record(
                filename=txt_path.name,
                source_filename=target_pdf_path.name,
            )

        except Exception as pre_commit_err:
            # Cleanup only files created during this request
            for f in created_files:
                f.unlink(missing_ok=True)
            logger.error(
                "Ingestion pre-commit phase failed for '%s': %s",
                clean_name,
                pre_commit_err,
                exc_info=True,
            )
            raise pre_commit_err

        # 6. Duplicate handling
        if db_record.get("status") == "ALREADY_EXISTS":
            for f in created_files:
                f.unlink(missing_ok=True)
            return {
                "status": "ALREADY_EXISTS",
                "message": db_record["message"],
                "go_id": db_record.get("go_id"),
                "go_number": db_record.get("go_number"),
                "department": db_record.get("department"),
                "issue_date": db_record.get("issue_date"),
                "source_filename": db_record.get("source_filename"),
                "references_count": 0,
                "relationships_count": 0,
                "chunks_indexed": 0,
                "qdrant_version": None,
                "retry_index_url": None,
            }

        # COMMIT BOUNDARY PASSED: PostgreSQL record has been committed.
        # Disk files (PDF & TXT) must NEVER be deleted after this point.

        # 7. Chunk and Index in Qdrant
        try:
            chunks = chunk_document_text(extracted_text)
            if not chunks:
                raise ValueError(
                    f"No semantic chunks generated for '{target_pdf_path.name}'."
                )

            qdrant_svc = self.qdrant_service or QdrantService()
            embedding_svc = self.embedding_service or EmbeddingService()

            indexing_res = qdrant_svc.safe_index_document(
                go_id=db_record["go_id"],
                go_number=db_record["go_number"],
                department=db_record["department"],
                source_filename=db_record["source_filename"],
                issue_date=db_record["issue_date"],
                chunks=chunks,
                embedding_service=embedding_svc,
            )

            return {
                "status": "SUCCESS",
                "message": f"Successfully ingested and indexed Government Order '{db_record['go_number']}'.",
                "go_id": db_record["go_id"],
                "go_number": db_record["go_number"],
                "department": db_record["department"],
                "issue_date": db_record["issue_date"],
                "source_filename": db_record["source_filename"],
                "references_count": db_record.get("references_count", 0),
                "relationships_count": db_record.get("relationships_count", 0),
                "chunks_indexed": indexing_res.get(
                    "chunks_indexed", len(chunks)
                ),
                "qdrant_version": indexing_res.get("version"),
                "retry_index_url": None,
            }

        except Exception as qdrant_err:
            logger.error(
                "Qdrant vector indexing failed for GO '%s' (id=%d): %s",
                db_record["go_number"],
                db_record["go_id"],
                qdrant_err,
                exc_info=True,
            )
            return {
                "status": "INDEXING_FAILED",
                "message": (
                    f"Government Order '{db_record['go_number']}' saved to database (id={db_record['go_id']}), "
                    f"but vector indexing failed: {qdrant_err}. Vector indexing retry is available."
                ),
                "go_id": db_record["go_id"],
                "go_number": db_record["go_number"],
                "department": db_record["department"],
                "issue_date": db_record["issue_date"],
                "source_filename": db_record["source_filename"],
                "references_count": db_record.get("references_count", 0),
                "relationships_count": db_record.get("relationships_count", 0),
                "chunks_indexed": 0,
                "qdrant_version": None,
                "retry_index_url": f"/api/v1/admin/documents/{db_record['go_id']}/retry-index",
            }

    def retry_index_document(self, go_id: int) -> Dict[str, Any]:
        """
        Reconstruct indexing arguments from stored PostgreSQL record and index into Qdrant.
        """
        with SessionLocal() as session:
            db_order = session.scalar(
                select(GovernmentOrderDB).where(GovernmentOrderDB.id == go_id)
            )
            if not db_order:
                raise ValueError(
                    f"Government Order with id={go_id} not found in database."
                )

            extracted_text = db_order.extracted_text
            if not extracted_text or not extracted_text.strip():
                txt_name = (
                    db_order.extracted_text_filename
                    or f"{Path(db_order.source_filename).stem}.txt"
                )
                txt_path = EXTRACTED_DIR / txt_name
                if txt_path.exists():
                    extracted_text = txt_path.read_text(encoding="utf-8")

            if not extracted_text or not extracted_text.strip():
                raise ValueError(
                    f"No extracted text available for Government Order with id={go_id}."
                )

            chunks = chunk_document_text(extracted_text)
            if not chunks:
                raise ValueError(
                    f"No text chunks could be generated for Government Order with id={go_id}."
                )

            qdrant_svc = self.qdrant_service or QdrantService()
            embedding_svc = self.embedding_service or EmbeddingService()

            indexing_res = qdrant_svc.safe_index_document(
                go_id=db_order.id,
                go_number=db_order.go_number,
                department=db_order.department,
                source_filename=db_order.source_filename,
                issue_date=(
                    db_order.issue_date.isoformat()
                    if db_order.issue_date
                    else None
                ),
                chunks=chunks,
                embedding_service=embedding_svc,
            )

            return {
                "status": (
                    "SUCCESS"
                    if indexing_res.get("status") in ["SUCCESS", "UNCHANGED"]
                    else indexing_res.get("status")
                ),
                "go_id": db_order.id,
                "go_number": db_order.go_number,
                "chunks_indexed": indexing_res.get(
                    "chunks_indexed", len(chunks)
                ),
                "qdrant_version": indexing_res.get("version"),
                "indexing_status": indexing_res.get("status"),
                "message": f"Successfully indexed Government Order '{db_order.go_number}' into Qdrant.",
            }
