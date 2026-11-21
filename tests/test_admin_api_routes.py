"""
Integration and security tests for Protected Admin API Routes and CORS (Step 6).

All tests run using FastAPI TestClient with mocked ingestion and workflow services.
NO live database queries, Qdrant mutations, or filesystem modifications occur.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from app.api.main import create_app
from app.api.routes import get_ingestion_service, get_workflow_service
from app.api.schemas import HealthResponse, QueryResponse


class TestAdminAPIRoutes(unittest.TestCase):
    """Test suite covering the 12 administrative endpoint, security, and CORS scenarios."""

    TEST_ADMIN_KEY = "test-admin-secret-key-12345"

    def setUp(self):
        self.mock_ingestion_service = MagicMock()
        self.mock_workflow_service = MagicMock()

        self.env_patcher = patch.dict(
            os.environ,
            {
                "ADMIN_API_KEY": self.TEST_ADMIN_KEY,
                "ALLOWED_ORIGINS": "http://localhost:5173",
            },
            clear=True,
        )
        self.env_patcher.start()

        # Build FastAPI app with mocked dependency providers
        self.app = create_app()
        self.app.dependency_overrides[get_ingestion_service] = (
            lambda: self.mock_ingestion_service
        )
        self.app.dependency_overrides[get_workflow_service] = (
            lambda: self.mock_workflow_service
        )

        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        self.env_patcher.stop()

    # 1. Admin upload without API key returns 403
    def test_01_admin_upload_missing_key_returns_403(self):
        files = {"file": ("test.pdf", b"%PDF-1.4 sample content", "application/pdf")}
        response = self.client.post("/api/v1/admin/documents/upload", files=files)
        self.assertEqual(response.status_code, 403)
        self.assertIn("Missing required administrative header", response.json()["detail"])

    # 2. Admin upload with invalid API key returns 403
    def test_02_admin_upload_invalid_key_returns_403(self):
        files = {"file": ("test.pdf", b"%PDF-1.4 sample content", "application/pdf")}
        headers = {"X-Admin-API-Key": "wrong-secret-key"}
        response = self.client.post(
            "/api/v1/admin/documents/upload", files=files, headers=headers
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Invalid administrative API key", response.json()["detail"])

    # 3. Admin upload with valid API key reaches the ingestion service (200 OK)
    def test_03_admin_upload_valid_key_success(self):
        self.mock_ingestion_service.ingest_new_order_pdf.return_value = {
            "status": "SUCCESS",
            "message": "Successfully ingested and indexed Government Order 'G.O.(Rt)No.24/2026/ITD'.",
            "go_id": 12,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-01-23",
            "source_filename": "test.pdf",
            "references_count": 4,
            "relationships_count": 6,
            "chunks_indexed": 10,
            "qdrant_version": 1,
            "retry_index_url": None,
        }

        files = {"file": ("test.pdf", b"%PDF-1.4 valid content", "application/pdf")}
        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/upload", files=files, headers=headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["go_id"], 12)
        self.assertEqual(data["go_number"], "G.O.(Rt)No.24/2026/ITD")
        self.mock_ingestion_service.ingest_new_order_pdf.assert_called_once()

    # 4. Non-PDF upload is rejected (400 Bad Request)
    def test_04_admin_upload_non_pdf_rejected(self):
        files = {"file": ("malicious.docx", b"%PDF-1.4 spoofed content", "application/pdf")}
        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/upload", files=files, headers=headers
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("must have a .pdf extension", response.json()["detail"])

    # 5. Invalid signature rejected (400 Bad Request)
    def test_05_admin_upload_invalid_signature_rejected(self):
        files = {"file": ("fake.pdf", b"NOT_A_REAL_PDF_HEADER", "application/pdf")}
        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/upload", files=files, headers=headers
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("PDF specification signature", response.json()["detail"])

    # 6. Oversized PDF is rejected (413 Payload Too Large)
    def test_06_admin_upload_oversized_file_rejected(self):
        # 25 MB + 100 bytes stream
        oversized_data = b"%PDF-" + (b"0" * (25 * 1024 * 1024 + 100))
        files = {"file": ("oversized.pdf", oversized_data, "application/pdf")}
        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/upload", files=files, headers=headers
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds the maximum allowed size of 25 MB", response.json()["detail"])

    # 7. Duplicate GO response is returned correctly (200 OK with status="ALREADY_EXISTS")
    def test_07_admin_upload_duplicate_response(self):
        self.mock_ingestion_service.ingest_new_order_pdf.return_value = {
            "status": "ALREADY_EXISTS",
            "message": "Already exists: G.O.(Rt)No.24/2026/ITD. No duplicate records inserted.",
            "go_id": 1,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-01-23",
            "source_filename": "24_2026_ITD.pdf",
            "references_count": 0,
            "relationships_count": 0,
            "chunks_indexed": 0,
            "qdrant_version": None,
            "retry_index_url": None,
        }

        files = {"file": ("dup.pdf", b"%PDF-1.4 duplicate", "application/pdf")}
        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/upload", files=files, headers=headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ALREADY_EXISTS")
        self.assertEqual(data["go_id"], 1)

    # 8. Qdrant indexing failure response is returned correctly (200 OK with status="INDEXING_FAILED" & retry_index_url)
    def test_08_admin_upload_indexing_failed_response(self):
        self.mock_ingestion_service.ingest_new_order_pdf.return_value = {
            "status": "INDEXING_FAILED",
            "message": "Saved to DB (id=15), but vector indexing failed. Retry available.",
            "go_id": 15,
            "go_number": "G.O.(Rt)No.15/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-02-01",
            "source_filename": "15.pdf",
            "references_count": 1,
            "relationships_count": 1,
            "chunks_indexed": 0,
            "qdrant_version": None,
            "retry_index_url": "/api/v1/admin/documents/15/retry-index",
        }

        files = {"file": ("qfail.pdf", b"%PDF-1.4 content", "application/pdf")}
        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/upload", files=files, headers=headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "INDEXING_FAILED")
        self.assertEqual(data["retry_index_url"], "/api/v1/admin/documents/15/retry-index")

    # 9. Retry without API key returns 403
    def test_09_retry_index_missing_key_returns_403(self):
        response = self.client.post("/api/v1/admin/documents/12/retry-index")
        self.assertEqual(response.status_code, 403)
        self.assertIn("Missing required administrative header", response.json()["detail"])

    # 10. Retry with valid key calls the ingestion service (200 OK)
    def test_10_retry_index_valid_key_success(self):
        self.mock_ingestion_service.retry_index_document.return_value = {
            "status": "SUCCESS",
            "message": "Successfully indexed Government Order into Qdrant.",
            "go_id": 12,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "chunks_indexed": 10,
            "qdrant_version": 2,
            "indexing_status": "SUCCESS",
        }

        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/12/retry-index", headers=headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["go_id"], 12)
        self.assertEqual(data["chunks_indexed"], 10)
        self.mock_ingestion_service.retry_index_document.assert_called_once_with(go_id=12)

    # 11. Unknown GO ID is handled correctly (404 Not Found)
    def test_11_retry_index_unknown_go_id_returns_404(self):
        self.mock_ingestion_service.retry_index_document.side_effect = ValueError(
            "Government Order with id=999 not found in database."
        )

        headers = {"X-Admin-API-Key": self.TEST_ADMIN_KEY}
        response = self.client.post(
            "/api/v1/admin/documents/999/retry-index", headers=headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found in database", response.json()["detail"])

    # 12. Public endpoints unaffected & CORS allows configured origin
    def test_12_public_endpoints_unaffected_and_cors(self):
        # A. Public /health works with NO auth header
        self.mock_workflow_service.check_health.return_value = HealthResponse(
            status="healthy",
            service="kerala-government-orders-multi-agent-system",
            version="1.0.0",
            ollama_online=True,
            postgres_online=True,
            qdrant_online=True,
        )
        health_res = self.client.get("/health")
        self.assertEqual(health_res.status_code, 200)

        # B. Public /api/v1/query works with NO auth header
        self.mock_workflow_service.execute_query.return_value = QueryResponse(
            thread_id="t1",
            workflow_status="COMPLETED",
            final_answer="Answer for public user",
            legal_effect_verified=False,
        )
        query_res = self.client.post(
            "/api/v1/query",
            json={"query": "Show details for G.O.(Rt)No.24/2026/ITD"},
        )
        self.assertEqual(query_res.status_code, 200)

        # C. CORS Preflight from ALLOWED origin (http://localhost:5173)
        cors_headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Admin-API-Key, Content-Type",
        }
        preflight_res = self.client.options(
            "/api/v1/admin/documents/upload", headers=cors_headers
        )
        self.assertEqual(preflight_res.status_code, 200)
        self.assertEqual(
            preflight_res.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

        # D. CORS Request from DISALLOWED origin (http://evil-site.com)
        disallowed_cors = {
            "Origin": "http://evil-site.com",
            "Access-Control-Request-Method": "POST",
        }
        disallowed_res = self.client.options(
            "/api/v1/admin/documents/upload", headers=disallowed_cors
        )
        self.assertIsNone(disallowed_res.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()
