"""
Unit tests for IngestionService.ingest_new_order_pdf and retry_index_document.

All tests in this suite run against completely ISOLATED MOCKS.
NO live PostgreSQL database connections or persistent Qdrant collection mutations occur.
"""

import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.database.models import GovernmentOrderDB
from app.services.ingestion_service import IngestionService


class TestIngestionOrchestration(unittest.TestCase):
    """Test suite using strictly ISOLATED MOCKS (zero live DB/Qdrant access)."""

    def setUp(self):
        self.mock_pdf_service = MagicMock()
        self.mock_pdf_classifier = MagicMock()
        self.mock_db_service = MagicMock()
        self.mock_qdrant_service = MagicMock()
        self.mock_embedding_service = MagicMock()

        self.service = IngestionService(
            pdf_service=self.mock_pdf_service,
            pdf_classifier=self.mock_pdf_classifier,
            database_service=self.mock_db_service,
            qdrant_service=self.mock_qdrant_service,
            embedding_service=self.mock_embedding_service,
        )

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_ingest_new_order_pdf_success_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify complete successful ingestion pipeline: PDF -> Text -> DB -> Qdrant -> Success."""
        mock_pdf_path = MagicMock(spec=Path)
        mock_pdf_path.exists.return_value = False
        mock_pdf_path.name = "new_order.pdf"
        mock_pdf_path.stem = "new_order"
        mock_dataset_dir.__truediv__.return_value = mock_pdf_path

        mock_txt_path = MagicMock(spec=Path)
        mock_txt_path.exists.return_value = False
        mock_txt_path.name = "new_order.txt"
        mock_extracted_dir.__truediv__.return_value = mock_txt_path

        # Classifier returns readable pages
        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "NATIVE",
            "pages": [{"page": 1, "type": "NATIVE"}],
        }

        # Extracted text is valid and usable
        self.mock_pdf_service.extract_text.return_value = (
            "--- Page 1 ---\nGovernment Order No. 10/2026/ITD\nDetailed sanctioned orders."
        )

        # Database service saves successfully
        self.mock_db_service.save_document_record.return_value = {
            "status": "INSERTED",
            "go_id": 15,
            "go_number": "G.O.(Rt)No.10/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-02-01",
            "source_filename": "new_order.pdf",
            "extracted_text_filename": "new_order.txt",
            "extracted_text": "Extracted text content",
            "references_count": 2,
            "relationships_count": 3,
            "message": "Successfully saved: G.O.(Rt)No.10/2026/ITD",
        }

        # Qdrant service safe_index_document succeeds
        self.mock_qdrant_service.safe_index_document.return_value = {
            "status": "SUCCESS",
            "go_id": 15,
            "go_number": "G.O.(Rt)No.10/2026/ITD",
            "version": 1,
            "chunks_indexed": 4,
            "point_ids": ["p1", "p2", "p3", "p4"],
        }

        # Execute
        result = self.service.ingest_new_order_pdf(
            pdf_bytes=b"%PDF-1.4 dummy binary content",
            filename="new_order.pdf",
        )

        # Assertions
        mock_pdf_path.write_bytes.assert_called_once_with(b"%PDF-1.4 dummy binary content")
        mock_txt_path.write_text.assert_called_once()
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["go_id"], 15)
        self.assertEqual(result["go_number"], "G.O.(Rt)No.10/2026/ITD")
        self.assertEqual(result["references_count"], 2)
        self.assertEqual(result["relationships_count"], 3)
        self.assertEqual(result["chunks_indexed"], 4)
        self.assertEqual(result["qdrant_version"], 1)
        self.assertIsNone(result["retry_index_url"])

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_ingest_duplicate_cleanup_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify duplicate detection unlinks newly created files and skips Qdrant indexing."""
        mock_pdf_path = MagicMock(spec=Path)
        mock_pdf_path.exists.return_value = False
        mock_pdf_path.name = "dup_order.pdf"
        mock_pdf_path.stem = "dup_order"
        mock_dataset_dir.__truediv__.return_value = mock_pdf_path

        mock_txt_path = MagicMock(spec=Path)
        mock_txt_path.name = "dup_order.txt"
        mock_extracted_dir.__truediv__.return_value = mock_txt_path

        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "NATIVE",
            "pages": [{"page": 1, "type": "NATIVE"}],
        }
        self.mock_pdf_service.extract_text.return_value = "--- Page 1 ---\nDuplicate content"

        # DB service signals duplicate
        self.mock_db_service.save_document_record.return_value = {
            "status": "ALREADY_EXISTS",
            "go_id": 1,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-01-23",
            "source_filename": "24_2026_ITD.pdf",
            "message": "Already exists: G.O.(Rt)No.24/2026/ITD. No duplicate records inserted.",
        }

        result = self.service.ingest_new_order_pdf(
            pdf_bytes=b"%PDF-1.4 dummy",
            filename="dup_order.pdf",
        )

        # Assertions
        self.assertEqual(result["status"], "ALREADY_EXISTS")
        self.assertEqual(result["go_id"], 1)
        self.assertEqual(result["chunks_indexed"], 0)

        # Both newly created files MUST be unlinked
        mock_pdf_path.unlink.assert_called_once_with(missing_ok=True)
        mock_txt_path.unlink.assert_called_once_with(missing_ok=True)

        # Qdrant indexing must NEVER be called
        self.mock_qdrant_service.safe_index_document.assert_not_called()

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_cleanup_on_extraction_failure_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify exception during extraction unlinks saved PDF and aborts cleanly."""
        mock_pdf_path = MagicMock(spec=Path)
        mock_pdf_path.exists.return_value = False
        mock_pdf_path.name = "corrupt.pdf"
        mock_pdf_path.stem = "corrupt"
        mock_dataset_dir.__truediv__.return_value = mock_pdf_path

        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "CORRUPTED",
            "pages": [{"page": 1, "type": "CORRUPTED"}],
        }
        self.mock_pdf_service.extract_text.side_effect = RuntimeError("PyMuPDF decompression bomb")

        with self.assertRaises(RuntimeError) as ctx:
            self.service.ingest_new_order_pdf(
                pdf_bytes=b"%PDF-1.4 corrupt",
                filename="corrupt.pdf",
            )

        self.assertIn("PyMuPDF decompression bomb", str(ctx.exception))
        # Saved PDF must be unlinked
        mock_pdf_path.unlink.assert_called_once_with(missing_ok=True)
        self.mock_db_service.save_document_record.assert_not_called()

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_empty_extracted_text_rejection_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify empty/unusable extracted text is rejected and unlinks saved PDF."""
        mock_pdf_path = MagicMock(spec=Path)
        mock_pdf_path.exists.return_value = False
        mock_pdf_path.name = "empty.pdf"
        mock_pdf_path.stem = "empty"
        mock_dataset_dir.__truediv__.return_value = mock_pdf_path

        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "SCANNED",
            "pages": [{"page": 1, "type": "SCANNED"}],
        }
        # Only page and OCR headers, zero actual text
        self.mock_pdf_service.extract_text.return_value = "--- Page 1 ---\n--- OCR Text ---\n   \n"

        with self.assertRaises(ValueError) as ctx:
            self.service.ingest_new_order_pdf(
                pdf_bytes=b"%PDF-1.4 empty",
                filename="empty.pdf",
            )

        self.assertIn("empty or unusable", str(ctx.exception))
        mock_pdf_path.unlink.assert_called_once_with(missing_ok=True)
        self.mock_db_service.save_document_record.assert_not_called()

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_invalid_classification_rejection_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify document with 0 readable pages is rejected and unlinks saved PDF."""
        mock_pdf_path = MagicMock(spec=Path)
        mock_pdf_path.exists.return_value = False
        mock_pdf_path.name = "zero_page.pdf"
        mock_pdf_path.stem = "zero_page"
        mock_dataset_dir.__truediv__.return_value = mock_pdf_path

        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "CORRUPTED",
            "pages": [],  # zero pages
        }

        with self.assertRaises(ValueError) as ctx:
            self.service.ingest_new_order_pdf(
                pdf_bytes=b"%PDF-1.4 zero",
                filename="zero_page.pdf",
            )

        self.assertIn("contains no readable pages", str(ctx.exception))
        mock_pdf_path.unlink.assert_called_once_with(missing_ok=True)
        self.mock_pdf_service.extract_text.assert_not_called()

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_database_failure_cleans_up_files_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify database exception triggers unlinking of both PDF and TXT files."""
        mock_pdf_path = MagicMock(spec=Path)
        mock_pdf_path.exists.return_value = False
        mock_pdf_path.name = "db_fail.pdf"
        mock_pdf_path.stem = "db_fail"
        mock_dataset_dir.__truediv__.return_value = mock_pdf_path

        mock_txt_path = MagicMock(spec=Path)
        mock_txt_path.name = "db_fail.txt"
        mock_extracted_dir.__truediv__.return_value = mock_txt_path

        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "NATIVE",
            "pages": [{"page": 1, "type": "NATIVE"}],
        }
        self.mock_pdf_service.extract_text.return_value = "--- Page 1 ---\nSome valid content"
        self.mock_db_service.save_document_record.side_effect = RuntimeError("PostgreSQL IO Error")

        with self.assertRaises(RuntimeError) as ctx:
            self.service.ingest_new_order_pdf(
                pdf_bytes=b"%PDF-1.4 fail",
                filename="db_fail.pdf",
            )

        self.assertIn("PostgreSQL IO Error", str(ctx.exception))
        mock_pdf_path.unlink.assert_called_once_with(missing_ok=True)
        mock_txt_path.unlink.assert_called_once_with(missing_ok=True)

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_qdrant_failure_preserves_db_and_files_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify Qdrant failure AFTER database commit PRESERVES disk files and returns INDEXING_FAILED."""
        mock_pdf_path = MagicMock(spec=Path)
        mock_pdf_path.exists.return_value = False
        mock_pdf_path.name = "qdrant_fail.pdf"
        mock_pdf_path.stem = "qdrant_fail"
        mock_dataset_dir.__truediv__.return_value = mock_pdf_path

        mock_txt_path = MagicMock(spec=Path)
        mock_txt_path.name = "qdrant_fail.txt"
        mock_extracted_dir.__truediv__.return_value = mock_txt_path

        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "NATIVE",
            "pages": [{"page": 1, "type": "NATIVE"}],
        }
        self.mock_pdf_service.extract_text.return_value = "--- Page 1 ---\nSanction accorded."

        # DB succeeds
        self.mock_db_service.save_document_record.return_value = {
            "status": "INSERTED",
            "go_id": 99,
            "go_number": "G.O.(Rt)No.99/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-02-15",
            "source_filename": "qdrant_fail.pdf",
            "references_count": 1,
            "relationships_count": 1,
            "message": "Saved",
        }

        # Qdrant fails
        self.mock_qdrant_service.safe_index_document.side_effect = RuntimeError("Qdrant Lock Error")

        result = self.service.ingest_new_order_pdf(
            pdf_bytes=b"%PDF-1.4 qfail",
            filename="qdrant_fail.pdf",
        )

        # Assertions
        self.assertEqual(result["status"], "INDEXING_FAILED")
        self.assertEqual(result["go_id"], 99)
        self.assertEqual(result["chunks_indexed"], 0)
        self.assertIn("retry is available", result["message"])
        self.assertEqual(result["retry_index_url"], "/api/v1/admin/documents/99/retry-index")

        # Crucial check: files must NOT be unlinked!
        mock_pdf_path.unlink.assert_not_called()
        mock_txt_path.unlink.assert_not_called()

    @patch("app.services.ingestion_service.EXTRACTED_DIR")
    @patch("app.services.ingestion_service.DATASET_DIR")
    def test_collision_safe_filename_generation_mocked(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED] Verify that if a file with same name exists in dataset/, a unique suffix is generated."""
        existing_pdf = MagicMock(spec=Path)
        existing_pdf.exists.side_effect = [True, False]  # first check exists, suffixed check does not
        mock_dataset_dir.__truediv__.return_value = existing_pdf

        mock_txt_path = MagicMock(spec=Path)
        mock_extracted_dir.__truediv__.return_value = mock_txt_path

        self.mock_pdf_classifier.classify_pdf.return_value = {
            "overall_type": "NATIVE",
            "pages": [{"page": 1, "type": "NATIVE"}],
        }
        self.mock_pdf_service.extract_text.return_value = "--- Page 1 ---\nSome valid content"
        self.mock_db_service.save_document_record.return_value = {
            "status": "INSERTED",
            "go_id": 55,
            "go_number": "G.O.(Rt)No.55/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-02-10",
            "source_filename": "collision_order.pdf",
            "references_count": 0,
            "relationships_count": 0,
        }
        self.mock_qdrant_service.safe_index_document.return_value = {
            "status": "SUCCESS", "version": 1, "chunks_indexed": 1
        }

        result = self.service.ingest_new_order_pdf(
            pdf_bytes=b"%PDF-1.4 collision",
            filename="collision_order.pdf",
        )

        self.assertEqual(result["status"], "SUCCESS")
        # Confirms collision check branch was taken
        self.assertGreater(mock_dataset_dir.__truediv__.call_count, 1)

    @patch("app.services.ingestion_service.SessionLocal")
    def test_retry_index_document_success_mocked(self, mock_session_cls):
        """[MOCKED] Verify retry_index_document reconstructs arguments and safely indexes document."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        db_order = GovernmentOrderDB(
            id=77,
            go_number="G.O.(Rt)No.77/2026/ITD",
            department="ITD",
            issue_date=date(2026, 1, 10),
            source_filename="order_77.pdf",
            extracted_text_filename="order_77.txt",
            extracted_text="--- Page 1 ---\nSanction accorded for project 77.",
            verification_status="VERIFIED",
        )
        mock_session.scalar.return_value = db_order

        self.mock_qdrant_service.safe_index_document.return_value = {
            "status": "SUCCESS",
            "version": 2,
            "chunks_indexed": 3,
        }

        result = self.service.retry_index_document(go_id=77)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["go_id"], 77)
        self.assertEqual(result["qdrant_version"], 2)
        self.assertEqual(result["chunks_indexed"], 3)
        self.mock_qdrant_service.safe_index_document.assert_called_once()

    @patch("app.services.ingestion_service.SessionLocal")
    def test_retry_index_document_not_found_mocked(self, mock_session_cls):
        """[MOCKED] Verify retry_index_document raises ValueError when go_id does not exist in DB."""
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.scalar.return_value = None

        with self.assertRaises(ValueError) as ctx:
            self.service.retry_index_document(go_id=999)

        self.assertIn("not found in database", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
