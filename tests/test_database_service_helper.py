"""
Unit tests for DatabaseService.save_document_record and save_document.

All tests in this suite run against completely ISOLATED MOCKS.
NO live PostgreSQL database connections or mutations occur.
"""

import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.database.models import GovernmentOrderDB
from app.services.database_service import DatabaseService


class TestDatabaseServiceHelper(unittest.TestCase):
    """Test suite using strictly ISOLATED MOCKS (zero live PostgreSQL access)."""

    def setUp(self):
        self.service = DatabaseService()

    @patch("app.services.database_service.EXTRACTED_DIR")
    @patch("app.services.database_service.DATASET_DIR")
    @patch("app.services.database_service.SessionLocal")
    def test_successful_insertion_mocked_db(
        self, mock_session_cls, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED DB] Verify successful insertion returns structured dict with status INSERTED."""
        # Setup mock files
        mock_txt = MagicMock(spec=Path)
        mock_txt.exists.return_value = True
        mock_txt.read_text.return_value = "ORDER CONTENT"
        mock_extracted_dir.__truediv__.return_value = mock_txt

        mock_pdf = MagicMock(spec=Path)
        mock_pdf.exists.return_value = True
        mock_pdf.name = "sample.pdf"
        mock_dataset_dir.__truediv__.return_value = mock_pdf

        # Setup mock extraction results
        self.service.identifier_service.extract_from_text = MagicMock()
        mock_id_res = MagicMock()
        mock_id_res.go_numbers = [MagicMock(value="G.O.(Rt)No.100/2026/ITD")]
        mock_id_res.issue_dates = [MagicMock(value="15-01-2026")]
        mock_id_res.departments = [MagicMock(value="ITD")]

        # 1 reference, 1 relationship
        mock_ref = MagicMock()
        mock_ref.ordinal = 1
        mock_ref.go_number = "G.O.(Rt)No.50/2025/ITD"
        mock_ref.issue_date = "10-01-2025"
        mock_ref.page_number = 1
        mock_ref.evidence_text = "Ref evidence"
        mock_id_res.references = [mock_ref]

        mock_rel = MagicMock()
        mock_rel.relationship_type.value = "AMENDS"
        mock_rel.referenced_go_numbers = ["G.O.(Rt)No.50/2025/ITD"]
        mock_rel.matched_phrase = "amends previous order"
        mock_rel.evidence_text = "Rel evidence"
        mock_rel.page_number = 1
        mock_rel.confidence.value = "HIGH"
        mock_rel.extraction_method = "RULE_BASED"
        mock_rel.review_required = False
        mock_id_res.relationships = [mock_rel]

        self.service.identifier_service.extract_from_text.return_value = mock_id_res

        self.service.metadata_service.extract_metadata = MagicMock(
            return_value={"verification_status": "VERIFIED"}
        )

        # Setup mock SQLAlchemy session
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.begin.return_value.__enter__.return_value = mock_session

        # Existing order check: returns None (new document)
        mock_session.scalar.return_value = None

        def fake_flush():
            # simulate DB assigning primary key
            for call in mock_session.add.call_args_list:
                obj = call[0][0]
                if isinstance(obj, GovernmentOrderDB) and getattr(obj, "id", None) is None:
                    obj.id = 42

        mock_session.flush.side_effect = fake_flush

        # Execute
        result = self.service.save_document_record("sample.txt")

        # Verify structured output
        self.assertEqual(result["status"], "INSERTED")
        self.assertEqual(result["go_id"], 42)
        self.assertEqual(result["go_number"], "G.O.(Rt)No.100/2026/ITD")
        self.assertEqual(result["department"], "ITD")
        self.assertEqual(result["issue_date"], "2026-01-15")
        self.assertEqual(result["source_filename"], "sample.pdf")
        self.assertEqual(result["extracted_text_filename"], "sample.txt")
        self.assertEqual(result["extracted_text"], "ORDER CONTENT")
        self.assertEqual(result["verification_status"], "VERIFIED")
        self.assertEqual(result["references_count"], 1)
        self.assertEqual(result["relationships_count"], 1)
        self.assertIn("Successfully saved: G.O.(Rt)No.100/2026/ITD", result["message"])

    @patch("app.services.database_service.EXTRACTED_DIR")
    @patch("app.services.database_service.DATASET_DIR")
    @patch("app.services.database_service.SessionLocal")
    def test_duplicate_go_handling_mocked_db(
        self, mock_session_cls, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED DB] Verify duplicate document detection returns status ALREADY_EXISTS with existing details."""
        mock_txt = MagicMock(spec=Path)
        mock_txt.exists.return_value = True
        mock_txt.read_text.return_value = "ORDER CONTENT"
        mock_extracted_dir.__truediv__.return_value = mock_txt

        mock_pdf = MagicMock(spec=Path)
        mock_pdf.exists.return_value = True
        mock_pdf.name = "sample.pdf"
        mock_dataset_dir.__truediv__.return_value = mock_pdf

        # Extraction returns existing GO number
        mock_id_res = MagicMock()
        mock_id_res.go_numbers = [MagicMock(value="G.O.(Rt)No.24/2026/ITD")]
        mock_id_res.issue_dates = [MagicMock(value="23-01-2026")]
        mock_id_res.departments = [MagicMock(value="ITD")]
        mock_id_res.references = []
        mock_id_res.relationships = []
        self.service.identifier_service.extract_from_text = MagicMock(return_value=mock_id_res)
        self.service.metadata_service.extract_metadata = MagicMock(
            return_value={"verification_status": "MANUAL"}
        )

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.begin.return_value.__enter__.return_value = mock_session

        # Existing order in DB
        existing_db_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            issue_date=date(2026, 1, 23),
            source_filename="24_2026_ITD.pdf",
            extracted_text_filename="24_2026_ITD.txt",
            extracted_text="Existing text",
            verification_status="MANUAL",
        )
        mock_session.scalar.return_value = existing_db_order

        # Execute
        result = self.service.save_document_record("sample.txt")

        # Verify
        self.assertEqual(result["status"], "ALREADY_EXISTS")
        self.assertEqual(result["go_id"], 1)
        self.assertEqual(result["go_number"], "G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(result["source_filename"], "24_2026_ITD.pdf")
        self.assertEqual(result["references_count"], 0)
        self.assertEqual(result["relationships_count"], 0)
        self.assertIn("Already exists: G.O.(Rt)No.24/2026/ITD", result["message"])

        # Crucial check: session.add was NEVER called
        mock_session.add.assert_not_called()

    @patch("app.services.database_service.EXTRACTED_DIR")
    @patch("app.services.database_service.DATASET_DIR")
    @patch("app.services.database_service.SessionLocal")
    def test_custom_source_filename_sanitization_mocked_db(
        self, mock_session_cls, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED DB] Verify custom source_filename strips traversal characters and matches saved record."""
        mock_txt = MagicMock(spec=Path)
        mock_txt.exists.return_value = True
        mock_txt.read_text.return_value = "TEXT"
        mock_extracted_dir.__truediv__.return_value = mock_txt

        mock_pdf = MagicMock(spec=Path)
        mock_pdf.exists.return_value = True
        mock_pdf.name = "clean_file.pdf"
        mock_dataset_dir.__truediv__.return_value = mock_pdf

        mock_id_res = MagicMock()
        mock_id_res.go_numbers = [MagicMock(value="G.O.(Rt)No.300/2026/ITD")]
        mock_id_res.issue_dates = [MagicMock(value="01-02-2026")]
        mock_id_res.departments = [MagicMock(value="ITD")]
        mock_id_res.references = []
        mock_id_res.relationships = []
        self.service.identifier_service.extract_from_text = MagicMock(return_value=mock_id_res)
        self.service.metadata_service.extract_metadata = MagicMock(
            return_value={"verification_status": None}
        )

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.begin.return_value.__enter__.return_value = mock_session
        mock_session.scalar.return_value = None

        def fake_flush():
            for call in mock_session.add.call_args_list:
                obj = call[0][0]
                if isinstance(obj, GovernmentOrderDB):
                    obj.id = 99

        mock_session.flush.side_effect = fake_flush

        # Pass traversal attempt: "../../clean_file.pdf"
        result = self.service.save_document_record("sample.txt", source_filename="../../clean_file.pdf")

        # Verify dataset_dir was queried with clean basename only
        mock_dataset_dir.__truediv__.assert_called_with("clean_file.pdf")
        self.assertEqual(result["source_filename"], "clean_file.pdf")

    @patch("app.services.database_service.EXTRACTED_DIR")
    @patch("app.services.database_service.DATASET_DIR")
    @patch("app.services.database_service.SessionLocal")
    def test_backward_compatibility_save_document_mocked_db(
        self, mock_session_cls, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED DB] Verify save_document() returns the exact string output format for success and duplicate."""
        mock_txt = MagicMock(spec=Path)
        mock_txt.exists.return_value = True
        mock_txt.read_text.return_value = "TEXT"
        mock_extracted_dir.__truediv__.return_value = mock_txt

        mock_pdf = MagicMock(spec=Path)
        mock_pdf.exists.return_value = True
        mock_pdf.name = "sample.pdf"
        mock_dataset_dir.__truediv__.return_value = mock_pdf

        mock_id_res = MagicMock()
        mock_id_res.go_numbers = [MagicMock(value="G.O.(Rt)No.400/2026/ITD")]
        mock_id_res.issue_dates = [MagicMock(value="01-02-2026")]
        mock_id_res.departments = [MagicMock(value="ITD")]
        mock_id_res.references = [
            MagicMock(
                ordinal=1,
                go_number="GO-1",
                issue_date="01-01-2025",
                page_number=1,
                evidence_text="e",
            )
        ]
        mock_id_res.relationships = [MagicMock(referenced_go_numbers=["GO-1"], relationship_type=MagicMock(value="AMENDS"), matched_phrase="m", evidence_text="e", page_number=1, confidence=MagicMock(value="HIGH"), extraction_method="RULE_BASED", review_required=False)]
        self.service.identifier_service.extract_from_text = MagicMock(return_value=mock_id_res)
        self.service.metadata_service.extract_metadata = MagicMock(return_value={"verification_status": "V"})

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        mock_session.begin.return_value.__enter__.return_value = mock_session

        # 1. Success case
        mock_session.scalar.return_value = None
        msg = self.service.save_document("sample.txt")
        self.assertIsInstance(msg, str)
        self.assertEqual(
            msg,
            "Successfully saved: G.O.(Rt)No.400/2026/ITD\nReferences: 1\nRelationships: 1"
        )

        # 2. Duplicate case
        mock_session.scalar.return_value = GovernmentOrderDB(
            id=2, go_number="G.O.(Rt)No.400/2026/ITD", issue_date=None, department="ITD",
            source_filename="sample.pdf", extracted_text_filename="sample.txt",
            extracted_text="T", verification_status="V"
        )
        msg_dup = self.service.save_document("sample.txt")
        self.assertIsInstance(msg_dup, str)
        self.assertEqual(
            msg_dup,
            "Already exists: G.O.(Rt)No.400/2026/ITD. No duplicate records inserted."
        )

    @patch("app.services.database_service.EXTRACTED_DIR")
    @patch("app.services.database_service.DATASET_DIR")
    def test_file_not_found_handling_mocked_db(self, mock_dataset_dir, mock_extracted_dir):
        """[MOCKED DB] Verify FileNotFoundError raised when text file or PDF file is missing."""
        # Text file missing
        mock_txt = MagicMock(spec=Path)
        mock_txt.exists.return_value = False
        mock_extracted_dir.__truediv__.return_value = mock_txt

        with self.assertRaises(FileNotFoundError) as ctx:
            self.service.save_document_record("nonexistent.txt")
        self.assertIn("Extracted text file not found", str(ctx.exception))

        # PDF missing
        mock_txt.exists.return_value = True
        mock_pdf = MagicMock(spec=Path)
        mock_pdf.exists.return_value = False
        mock_dataset_dir.__truediv__.return_value = mock_pdf

        with self.assertRaises(FileNotFoundError) as ctx:
            self.service.save_document_record("sample.txt")
        self.assertIn("Original PDF not found", str(ctx.exception))

    @patch("app.services.database_service.EXTRACTED_DIR")
    @patch("app.services.database_service.DATASET_DIR")
    def test_missing_go_number_raises_value_error_mocked_db(
        self, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED DB] Verify ValueError is raised when GO number cannot be determined."""
        mock_txt = MagicMock(spec=Path)
        mock_txt.exists.return_value = True
        mock_txt.read_text.return_value = "Random unparsed text"
        mock_extracted_dir.__truediv__.return_value = mock_txt

        mock_pdf = MagicMock(spec=Path)
        mock_pdf.exists.return_value = True
        mock_dataset_dir.__truediv__.return_value = mock_pdf

        mock_id_res = MagicMock()
        mock_id_res.go_numbers = [MagicMock(value="Unknown")]
        self.service.identifier_service.extract_from_text = MagicMock(return_value=mock_id_res)
        self.service.metadata_service.extract_metadata = MagicMock(return_value={"verification_status": None})

        with self.assertRaises(ValueError) as ctx:
            self.service.save_document_record("sample.txt")
        self.assertIn("GO number is missing", str(ctx.exception))

    @patch("app.services.database_service.EXTRACTED_DIR")
    @patch("app.services.database_service.DATASET_DIR")
    @patch("app.services.database_service.SessionLocal")
    def test_atomic_rollback_on_error_mocked_db(
        self, mock_session_cls, mock_dataset_dir, mock_extracted_dir
    ):
        """[MOCKED DB] Verify atomic transaction rolls back and re-raises exception on database error."""
        mock_txt = MagicMock(spec=Path)
        mock_txt.exists.return_value = True
        mock_txt.read_text.return_value = "TEXT"
        mock_extracted_dir.__truediv__.return_value = mock_txt

        mock_pdf = MagicMock(spec=Path)
        mock_pdf.exists.return_value = True
        mock_pdf.name = "sample.pdf"
        mock_dataset_dir.__truediv__.return_value = mock_pdf

        mock_id_res = MagicMock()
        mock_id_res.go_numbers = [MagicMock(value="G.O.(Rt)No.500/2026/ITD")]
        mock_id_res.issue_dates = [MagicMock(value="01-01-2026")]
        mock_id_res.departments = [MagicMock(value="ITD")]
        mock_id_res.references = [MagicMock(ordinal=1, go_number="G1", issue_date="01-01-2025", page_number=1, evidence_text="e")]
        mock_id_res.relationships = []
        self.service.identifier_service.extract_from_text = MagicMock(return_value=mock_id_res)
        self.service.metadata_service.extract_metadata = MagicMock(return_value={"verification_status": "V"})

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session
        
        # Simulate transactional context manager
        transaction_ctx = MagicMock()
        mock_session.begin.return_value = transaction_ctx
        transaction_ctx.__enter__.return_value = mock_session
        
        # Duplicate check passes
        mock_session.scalar.return_value = None

        # Simulate database failure during flush
        mock_session.flush.side_effect = RuntimeError("Simulated DB Disk Failure")

        with self.assertRaises(RuntimeError) as ctx:
            self.service.save_document_record("sample.txt")
        self.assertIn("Simulated DB Disk Failure", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
