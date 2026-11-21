"""
Unit tests for Admin Authentication and API Schemas (Step 5).

Runs against isolated unit tests (no live DB, Ollama, or Qdrant connections).
"""

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.api.auth import verify_admin_api_key
from app.api.schemas import DocumentUploadResponse, RetryIndexResponse


class TestAdminAuthAndSchemas(unittest.TestCase):
    """Test suite for Step 5 administrative authentication and schemas."""

    def test_admin_auth_fails_closed_when_env_var_missing(self):
        """[ISOLATED] Verify verify_admin_api_key raises HTTP 503 when ADMIN_API_KEY is unset."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as ctx:
                verify_admin_api_key(provided_key="any-client-key")
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("Administrative services are unavailable", ctx.exception.detail)

    def test_admin_auth_fails_closed_when_env_var_whitespace(self):
        """[ISOLATED] Verify verify_admin_api_key raises HTTP 503 when ADMIN_API_KEY is whitespace."""
        with patch.dict(os.environ, {"ADMIN_API_KEY": "   \t\n  "}, clear=True):
            with self.assertRaises(HTTPException) as ctx:
                verify_admin_api_key(provided_key="any-client-key")
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("Administrative services are unavailable", ctx.exception.detail)

    def test_admin_auth_missing_header_raises_403(self):
        """[ISOLATED] Verify verify_admin_api_key raises HTTP 403 when X-Admin-API-Key is missing or empty."""
        with patch.dict(os.environ, {"ADMIN_API_KEY": "secure-test-admin-key-789"}, clear=True):
            # None / omitted header
            with self.assertRaises(HTTPException) as ctx:
                verify_admin_api_key(provided_key=None)
            self.assertEqual(ctx.exception.status_code, 403)
            self.assertIn("Missing required administrative header", ctx.exception.detail)

            # Empty string header
            with self.assertRaises(HTTPException) as ctx_empty:
                verify_admin_api_key(provided_key="   ")
            self.assertEqual(ctx_empty.exception.status_code, 403)
            self.assertIn("Missing required administrative header", ctx_empty.exception.detail)

    def test_admin_auth_invalid_key_raises_403(self):
        """[ISOLATED] Verify verify_admin_api_key raises HTTP 403 when constant-time check fails."""
        with patch.dict(os.environ, {"ADMIN_API_KEY": "secure-test-admin-key-789"}, clear=True):
            with self.assertRaises(HTTPException) as ctx:
                verify_admin_api_key(provided_key="wrong-admin-key-value")
            self.assertEqual(ctx.exception.status_code, 403)
            self.assertIn("Invalid administrative API key", ctx.exception.detail)

    def test_admin_auth_valid_key_returns_true(self):
        """[ISOLATED] Verify verify_admin_api_key returns True when key matches exactly."""
        with patch.dict(os.environ, {"ADMIN_API_KEY": "secure-test-admin-key-789"}, clear=True):
            result = verify_admin_api_key(provided_key="secure-test-admin-key-789")
            self.assertTrue(result)

    def test_document_upload_response_schema_validation(self):
        """[ISOLATED] Verify DocumentUploadResponse schema serialization and constraints."""
        # 1. SUCCESS case
        success_data = {
            "status": "SUCCESS",
            "message": "Successfully ingested and indexed.",
            "go_id": 10,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-01-23",
            "source_filename": "24_2026_ITD.pdf",
            "references_count": 4,
            "relationships_count": 6,
            "chunks_indexed": 12,
            "qdrant_version": 1,
            "retry_index_url": None,
        }
        res_success = DocumentUploadResponse(**success_data)
        self.assertEqual(res_success.status, "SUCCESS")
        self.assertEqual(res_success.go_id, 10)
        self.assertEqual(res_success.chunks_indexed, 12)

        # 2. ALREADY_EXISTS case
        dup_data = {
            "status": "ALREADY_EXISTS",
            "message": "Already exists.",
            "go_id": 1,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-01-23",
            "source_filename": "24_2026_ITD.pdf",
        }
        res_dup = DocumentUploadResponse(**dup_data)
        self.assertEqual(res_dup.status, "ALREADY_EXISTS")
        self.assertEqual(res_dup.chunks_indexed, 0)
        self.assertIsNone(res_dup.qdrant_version)

        # 3. INDEXING_FAILED case
        failed_data = {
            "status": "INDEXING_FAILED",
            "message": "Vector indexing failed. Retry available.",
            "go_id": 10,
            "go_number": "G.O.(Rt)No.24/2026/ITD",
            "department": "ITD",
            "issue_date": "2026-01-23",
            "source_filename": "24_2026_ITD.pdf",
            "references_count": 4,
            "relationships_count": 6,
            "retry_index_url": "/api/v1/admin/documents/10/retry-index",
        }
        res_failed = DocumentUploadResponse(**failed_data)
        self.assertEqual(res_failed.status, "INDEXING_FAILED")
        self.assertEqual(res_failed.retry_index_url, "/api/v1/admin/documents/10/retry-index")

        # 4. Invalid status should raise ValidationError
        invalid_data = {**success_data, "status": "UNKNOWN_STATUS"}
        with self.assertRaises(ValidationError):
            DocumentUploadResponse(**invalid_data)

    def test_retry_index_response_schema_validation(self):
        """[ISOLATED] Verify RetryIndexResponse schema serialization and constraints."""
        retry_data = {
            "status": "SUCCESS",
            "message": "Successfully indexed Government Order into Qdrant.",
            "go_id": 42,
            "go_number": "G.O.(Rt)No.42/2026/ITD",
            "chunks_indexed": 8,
            "qdrant_version": 2,
            "indexing_status": "SUCCESS",
        }
        res = RetryIndexResponse(**retry_data)
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.go_id, 42)
        self.assertEqual(res.chunks_indexed, 8)
        self.assertEqual(res.qdrant_version, 2)
        self.assertEqual(res.indexing_status, "SUCCESS")

        # Missing required field go_id should raise ValidationError
        with self.assertRaises(ValidationError):
            RetryIndexResponse(
                status="SUCCESS",
                message="Missing id",
                go_number="G.O.1",
            )


if __name__ == "__main__":
    unittest.main()
