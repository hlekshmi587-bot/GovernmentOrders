"""
Tests for Phase 4.1 - Step 3: FastAPI Dashboard API Routes.

Covers:
1. POST /api/v1/search (HYBRID, KEYWORD, department filter, validation errors)
2. GET /api/v1/documents/{go_id} (details, references, pdf_url availability, missing PDF)
3. GET /api/v1/documents/{go_id}/relationships (outgoing/incoming, verification resolution, legal_effect_verified=False)
4. GET /api/v1/documents/{go_id}/pdf (inline viewing, attachment download, traversal defense, signature verification)
5. Scenario 2 direction separation (24/2026/ITD outgoing vs incoming)
6. Security checks (path traversal rejection, missing file, non-PDF signature rejection)
"""

from datetime import date, datetime
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from app.api.main import create_app
from app.api.routes import get_db, get_workflow_service
from app.api.schemas import (
    DocumentDetailResponse,
    DocumentRelationshipsResponse,
    ReferenceItemSchema,
    RelationshipItemSchema,
    SearchRequest,
    SearchResponse,
    SearchResultItemSchema,
)
from app.database.models import (
    GovernmentOrderDB,
    GOReferenceDB,
    GORelationshipDB,
    VerificationRecordDB,
)


def _mock_scalar_result(items: list):
    """Helper creating a mock SQLAlchemy ScalarResult supporting .all()."""
    mock_res = MagicMock()
    mock_res.all.return_value = items
    return mock_res


class TestDashboardRoutes(unittest.TestCase):
    """Test suite covering Phase 4.1 Step 3 FastAPI endpoints."""

    def setUp(self):
        self.mock_workflow_service = MagicMock()
        self.mock_db = MagicMock()

        self.app = create_app()
        self.app.dependency_overrides[get_workflow_service] = (
            lambda: self.mock_workflow_service
        )
        self.app.dependency_overrides[get_db] = lambda: self.mock_db

        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    # =========================================================================
    # 1. POST /api/v1/search
    # =========================================================================

    def test_search_endpoint_hybrid(self):
        """POST /api/v1/search in HYBRID mode returns 200 with SearchResponse."""
        mock_response = SearchResponse(
            status="SUCCESS",
            query="finance department pension",
            returned_count=1,
            limit_applied=10,
            has_more=False,
            is_exhaustive=False,
            results=[
                SearchResultItemSchema(
                    go_id=1,
                    go_number="G.O.(P)No.100/2026/FIN",
                    issue_date="2026-01-10",
                    department="Finance Department",
                    relevance_score=0.92,
                    text_snippet="Pension revision rules...",
                    source_filename="order100.pdf",
                    pdf_url="/api/v1/documents/1/pdf",
                    has_pdf=True,
                )
            ],
        )
        self.mock_workflow_service.search_documents.return_value = mock_response

        response = self.client.post(
            "/api/v1/search",
            json={"query": "finance department pension", "retrieval_mode": "HYBRID", "limit": 10},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["query"], "finance department pension")
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["go_number"], "G.O.(P)No.100/2026/FIN")
        self.assertEqual(data["results"][0]["issue_date"], "2026-01-10")
        self.assertFalse(data["is_exhaustive"])

    def test_search_endpoint_keyword_department_filter(self):
        """POST /api/v1/search with KEYWORD mode and department filter."""
        mock_response = SearchResponse(
            status="SUCCESS",
            query="pension",
            returned_count=0,
            limit_applied=5,
            has_more=False,
            is_exhaustive=True,
            results=[],
        )
        self.mock_workflow_service.search_documents.return_value = mock_response

        response = self.client.post(
            "/api/v1/search",
            json={
                "query": "pension",
                "retrieval_mode": "KEYWORD",
                "department": "Finance Department",
                "limit": 5,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["query"], "pension")
        self.assertEqual(data["returned_count"], 0)
        self.assertTrue(data["is_exhaustive"])

    def test_search_endpoint_invalid_body(self):
        """POST /api/v1/search with invalid body returns 422."""
        response = self.client.post("/api/v1/search", json={})
        self.assertEqual(response.status_code, 422)

    # =========================================================================
    # 2. GET /api/v1/documents/{go_id}
    # =========================================================================

    def test_document_detail_success(self):
        """GET /api/v1/documents/{go_id} returns 200 with full details and valid pdf_url."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_file = tmp_path / "order24.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

            mock_order = GovernmentOrderDB(
                id=1,
                go_number="G.O.(Rt)No.24/2026/ITD",
                issue_date=date(2026, 1, 15),
                department="Electronics & Information Technology",
                source_filename="order24.pdf",
                extracted_text="Full text of GO 24/2026/ITD",
                verification_status="VERIFIED",
            )
            mock_ref = GOReferenceDB(
                id=10,
                source_go_id=1,
                ordinal=1,
                referenced_go_number="G.O.(Ms)No.114/2025/ITD",
                reference_date=date(2025, 11, 20),
                page_number=1,
                evidence_text="Read: G.O.(Ms)No.114/2025/ITD dated 20-11-2025",
            )

            self.mock_db.scalar.return_value = mock_order
            self.mock_db.scalars.side_effect = [
                _mock_scalar_result([mock_ref]),  # references
                _mock_scalar_result([101]),       # outgoing rel IDs
                _mock_scalar_result([]),          # incoming rel candidates
            ]

            with patch("app.api.routes.DATASET_DIR", tmp_path):
                response = self.client.get("/api/v1/documents/1")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["id"], 1)
            self.assertEqual(data["go_number"], "G.O.(Rt)No.24/2026/ITD")
            self.assertEqual(data["pdf_url"], "/api/v1/documents/1/pdf")
            self.assertEqual(data["references_count"], 1)
            self.assertEqual(data["relationships_count"], 1)
            self.assertEqual(len(data["references"]), 1)
            self.assertEqual(data["references"][0]["ordinal"], 1)

    def test_document_detail_not_found(self):
        """GET /api/v1/documents/99999 returns 404 when order does not exist."""
        self.mock_db.scalar.return_value = None
        response = self.client.get("/api/v1/documents/99999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_document_detail_invalid_id(self):
        """GET /api/v1/documents/{go_id} returns 400 for non-positive IDs."""
        res_zero = self.client.get("/api/v1/documents/0")
        self.assertEqual(res_zero.status_code, 400)

        res_neg = self.client.get("/api/v1/documents/-5")
        self.assertEqual(res_neg.status_code, 400)

    def test_document_detail_missing_pdf(self):
        """GET /api/v1/documents/{go_id} returns pdf_url=None when PDF file is missing on disk."""
        mock_order = GovernmentOrderDB(
            id=2,
            go_number="G.O.(Ms)No.50/2026/GAD",
            issue_date=date(2026, 2, 1),
            department="General Administration",
            source_filename="missing_order50.pdf",
            extracted_text="Order text",
            verification_status="PENDING",
        )
        self.mock_db.scalar.return_value = mock_order
        self.mock_db.scalars.side_effect = [
            _mock_scalar_result([]),   # references
            _mock_scalar_result([]),   # outgoing rel IDs
            _mock_scalar_result([]),   # incoming rel candidates
        ]

        with tempfile.TemporaryDirectory() as empty_dir:
            with patch("app.api.routes.DATASET_DIR", Path(empty_dir)):
                response = self.client.get("/api/v1/documents/2")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["pdf_url"])

    # =========================================================================
    # 3. GET /api/v1/documents/{go_id}/relationships
    # =========================================================================

    def test_document_relationships_success(self):
        """GET /api/v1/documents/{go_id}/relationships returns outgoing and incoming correctly."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            source_filename="order24.pdf",
        )
        outgoing_rel = GORelationshipDB(
            id=101,
            source_go_id=1,
            relationship_type="MODIFIES",
            referenced_go_number="G.O.(Ms)No.114/2025/ITD",
            matched_phrase="in partial modification of",
            evidence_text="In partial modification of G.O.(Ms)No.114/2025/ITD",
            page_number=1,
            confidence="HIGH",
            extraction_method="RULE_BASED",
            review_required=True,
        )

        self.mock_db.scalar.return_value = mock_order
        self.mock_db.scalars.side_effect = [
            _mock_scalar_result([outgoing_rel]),   # outgoing_rels
            _mock_scalar_result([]),               # ver_records
        ]
        self.mock_db.execute.return_value.all.return_value = []

        response = self.client.get("/api/v1/documents/1/relationships")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["go_id"], 1)
        self.assertEqual(data["go_number"], "G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(data["outgoing_count"], 1)
        self.assertEqual(data["incoming_count"], 0)
        self.assertEqual(len(data["outgoing_relationships"]), 1)
        self.assertEqual(data["outgoing_relationships"][0]["verification_status"], "UNRECORDED")
        self.assertFalse(data["legal_effect_verified"])

    def test_document_relationships_verification_resolution(self):
        """Verification status prefers latest effective timestamp; tie breaks on id desc."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            source_filename="order24.pdf",
        )
        outgoing_rel = GORelationshipDB(
            id=101,
            source_go_id=1,
            relationship_type="MODIFIES",
            referenced_go_number="114/2025/ITD",
            confidence="HIGH",
        )

        # Older VERIFIED record and newer REJECTED record
        rec_old = VerificationRecordDB(
            id=1,
            relationship_id=101,
            go_id=1,
            verification_type="RELATIONSHIP",
            verification_status="VERIFIED",
            created_at=datetime(2026, 1, 1, 10, 0),
            verified_at=datetime(2026, 1, 1, 10, 5),
        )
        rec_new = VerificationRecordDB(
            id=2,
            relationship_id=101,
            go_id=1,
            verification_type="RELATIONSHIP",
            verification_status="REJECTED",
            created_at=datetime(2026, 1, 2, 11, 0),
            verified_at=datetime(2026, 1, 2, 11, 5),
        )

        self.mock_db.scalar.return_value = mock_order
        self.mock_db.scalars.side_effect = [
            _mock_scalar_result([outgoing_rel]),        # outgoing_rels
            _mock_scalar_result([rec_old, rec_new]),    # verification records
        ]
        self.mock_db.execute.return_value.all.return_value = []

        response = self.client.get("/api/v1/documents/1/relationships")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["outgoing_relationships"][0]["verification_status"], "REJECTED")

    def test_document_relationships_null_target_go_number_handled(self):
        """Relationship with referenced_go_number=None is serialized without HTTP 500."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            source_filename="order24.pdf",
        )
        outgoing_rel = GORelationshipDB(
            id=102,
            source_go_id=1,
            relationship_type="REFERENCES",
            referenced_go_number=None,
            confidence="MEDIUM",
        )

        self.mock_db.scalar.return_value = mock_order
        self.mock_db.scalars.side_effect = [
            _mock_scalar_result([outgoing_rel]),
            _mock_scalar_result([]),
        ]
        self.mock_db.execute.return_value.all.return_value = []

        response = self.client.get("/api/v1/documents/1/relationships")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["outgoing_relationships"]), 1)
        self.assertIsNone(data["outgoing_relationships"][0]["target_go_number"])

    def test_document_relationships_tie_breaker(self):
        """Tie-breaker with identical timestamps uses highest id."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            source_filename="order24.pdf",
        )
        outgoing_rel = GORelationshipDB(
            id=101,
            source_go_id=1,
            relationship_type="MODIFIES",
            referenced_go_number="114/2025/ITD",
        )

        rec_low_id = VerificationRecordDB(
            id=5,
            relationship_id=101,
            go_id=1,
            verification_type="RELATIONSHIP",
            verification_status="PENDING",
            created_at=datetime(2026, 1, 1, 10, 0),
            verified_at=datetime(2026, 1, 1, 10, 0),
        )
        rec_high_id = VerificationRecordDB(
            id=25,
            relationship_id=101,
            go_id=1,
            verification_type="RELATIONSHIP",
            verification_status="VERIFIED",
            created_at=datetime(2026, 1, 1, 10, 0),
            verified_at=datetime(2026, 1, 1, 10, 0),
        )

        self.mock_db.scalar.return_value = mock_order
        self.mock_db.scalars.side_effect = [
            _mock_scalar_result([outgoing_rel]),
            _mock_scalar_result([rec_low_id, rec_high_id]),
        ]
        self.mock_db.execute.return_value.all.return_value = []

        response = self.client.get("/api/v1/documents/1/relationships")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["outgoing_relationships"][0]["verification_status"], "VERIFIED")

    def test_document_relationships_legal_effect_verified_strictly_false(self):
        """Regardless of database records, legal_effect_verified is ALWAYS False."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            source_filename="order24.pdf",
        )
        outgoing_rel = GORelationshipDB(
            id=101,
            source_go_id=1,
            relationship_type="MODIFIES",
            referenced_go_number="114/2025/ITD",
        )
        rec = VerificationRecordDB(
            id=1,
            relationship_id=101,
            go_id=1,
            verification_type="RELATIONSHIP",
            verification_status="VERIFIED",
            created_at=datetime(2026, 1, 1, 10, 0),
            verified_at=datetime(2026, 1, 1, 10, 5),
        )

        self.mock_db.scalar.return_value = mock_order
        self.mock_db.scalars.side_effect = [
            _mock_scalar_result([outgoing_rel]),
            _mock_scalar_result([rec]),
        ]
        self.mock_db.execute.return_value.all.return_value = []

        response = self.client.get("/api/v1/documents/1/relationships")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["legal_effect_verified"])

    def test_scenario_2_direction_separation(self):
        """Scenario 2 (24/2026/ITD): exactly one outgoing candidate, zero incoming candidates."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            source_filename="order24.pdf",
        )
        outgoing_rel = GORelationshipDB(
            id=101,
            source_go_id=1,
            relationship_type="MODIFIES",
            referenced_go_number="G.O.(Ms)No.114/2025/ITD",
            matched_phrase="in partial modification of",
            evidence_text="In partial modification of G.O.(Ms)No.114/2025/ITD",
        )

        self.mock_db.scalar.return_value = mock_order
        self.mock_db.scalars.side_effect = [
            _mock_scalar_result([outgoing_rel]),   # outgoing
            _mock_scalar_result([]),               # ver records
        ]
        self.mock_db.execute.return_value.all.return_value = []

        response = self.client.get("/api/v1/documents/1/relationships")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["outgoing_count"], 1)
        self.assertEqual(data["incoming_count"], 0)
        self.assertEqual(data["incoming_relationships"], [])

    # =========================================================================
    # 4. GET /api/v1/documents/{go_id}/pdf
    # =========================================================================

    def test_document_pdf_view_inline(self):
        """GET /api/v1/documents/{go_id}/pdf without download flag serves inline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_file = tmp_path / "order24.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

            mock_order = GovernmentOrderDB(
                id=1,
                go_number="G.O.(Rt)No.24/2026/ITD",
                source_filename="order24.pdf",
            )
            self.mock_db.scalar.return_value = mock_order

            with patch("app.api.routes.DATASET_DIR", tmp_path):
                response = self.client.get("/api/v1/documents/1/pdf")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "application/pdf")
            self.assertIn("inline", response.headers["content-disposition"])

    def test_document_pdf_download_attachment(self):
        """GET /api/v1/documents/{go_id}/pdf?download=true serves as attachment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_file = tmp_path / "order24.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

            mock_order = GovernmentOrderDB(
                id=1,
                go_number="G.O.(Rt)No.24/2026/ITD",
                source_filename="order24.pdf",
            )
            self.mock_db.scalar.return_value = mock_order

            with patch("app.api.routes.DATASET_DIR", tmp_path):
                response = self.client.get("/api/v1/documents/1/pdf?download=true")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "application/pdf")
            self.assertIn("attachment", response.headers["content-disposition"])
            self.assertIn('filename="order24.pdf"', response.headers["content-disposition"])

    def test_document_pdf_traversal_rejected(self):
        """Filename with directory traversal is rejected with 400."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            source_filename="../../etc/passwd.pdf",
        )
        self.mock_db.scalar.return_value = mock_order

        response = self.client.get("/api/v1/documents/1/pdf")
        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid source filename", response.json()["detail"].lower())

    def test_document_pdf_missing_file_404(self):
        """Valid filename but file not on disk returns 404."""
        mock_order = GovernmentOrderDB(
            id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            source_filename="nonexistent.pdf",
        )
        self.mock_db.scalar.return_value = mock_order

        with tempfile.TemporaryDirectory() as empty_dir:
            with patch("app.api.routes.DATASET_DIR", Path(empty_dir)):
                response = self.client.get("/api/v1/documents/1/pdf")
                self.assertEqual(response.status_code, 404)
                self.assertIn("not found on disk", response.json()["detail"].lower())

    def test_document_pdf_invalid_signature_400(self):
        """File on disk lacking %PDF- signature returns 400."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_file = tmp_path / "fake.pdf"
            fake_file.write_bytes(b"NOT_A_PDF_FILE")

            mock_order = GovernmentOrderDB(
                id=1,
                go_number="G.O.(Rt)No.24/2026/ITD",
                source_filename="fake.pdf",
            )
            self.mock_db.scalar.return_value = mock_order

            with patch("app.api.routes.DATASET_DIR", tmp_path):
                response = self.client.get("/api/v1/documents/1/pdf")

            self.assertEqual(response.status_code, 400)
            self.assertIn("not a valid pdf", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
