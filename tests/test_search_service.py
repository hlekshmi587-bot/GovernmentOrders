"""
Focused Unit Tests for WorkflowService.search_documents and Search Result Mapping.

Covers:
1. Keyword search mode forwarding.
2. Semantic search mode forwarding.
3. Hybrid search mode forwarding.
4. Exact GO number query forwarding.
5. Department filter forwarding.
6. Multiple result mapping and relevance scores.
7. Empty result (NO_MATCH) handling.
8. has_more is True ONLY when candidates_retained > limit (not on ceiling alone).
9. is_exhaustive is False when candidate ceiling is reached or error occurred.
10. Missing vs present PDF disk verification and url generation.
11. Unsafe filename rejection (no silent basename sanitization).
12. SearchAgent failure exception handling.
13. QueryResponse backward compatibility in _format_response.
"""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from app.api.schemas import (
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.api.service import WorkflowService
from app.agents.specialized.search_agent import (
    RetrievalMode,
    SearchAgentResponse,
    SearchResultItem,
)


class TestWorkflowSearchService(unittest.TestCase):
    """Test suite for WorkflowService search integration."""

    def setUp(self) -> None:
        WorkflowService.reset_instance()
        self.mock_search_agent = MagicMock()
        with patch("app.api.service.build_government_orders_graph"):
            self.service = WorkflowService(
                checkpointer=MagicMock(),
                search_agent=self.mock_search_agent,
            )

    def tearDown(self) -> None:
        WorkflowService.reset_instance()

    def test_01_keyword_search_mode_forwarding(self) -> None:
        """Verify KEYWORD retrieval mode is correctly mapped to SearchAgentRequest."""
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="electricity",
            requested_mode=RetrievalMode.KEYWORD,
            returned_count=0,
            candidates_retained=0,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            results=[],
        )

        req = SearchRequest(query="electricity", retrieval_mode="KEYWORD", limit=10)
        resp = self.service.search_documents(req)

        self.mock_search_agent.execute_search.assert_called_once()
        agent_req = self.mock_search_agent.execute_search.call_args[0][0]
        self.assertEqual(agent_req.query, "electricity")
        self.assertEqual(agent_req.retrieval_mode, RetrievalMode.KEYWORD)
        self.assertEqual(resp.status, "SUCCESS")

    def test_02_semantic_search_mode_forwarding(self) -> None:
        """Verify SEMANTIC retrieval mode is correctly mapped to SearchAgentRequest."""
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="solar power tariff",
            requested_mode=RetrievalMode.SEMANTIC,
            returned_count=0,
            candidates_retained=0,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            results=[],
        )

        req = SearchRequest(query="solar power tariff", retrieval_mode="SEMANTIC", limit=5)
        resp = self.service.search_documents(req)

        agent_req = self.mock_search_agent.execute_search.call_args[0][0]
        self.assertEqual(agent_req.retrieval_mode, RetrievalMode.SEMANTIC)
        self.assertEqual(resp.limit_applied, 5)

    def test_03_hybrid_search_mode_forwarding(self) -> None:
        """Verify HYBRID retrieval mode is default and passed to SearchAgentRequest."""
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="pension scheme",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=0,
            candidates_retained=0,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            results=[],
        )

        req = SearchRequest(query="pension scheme")
        self.assertEqual(req.retrieval_mode, "HYBRID")
        resp = self.service.search_documents(req)

        agent_req = self.mock_search_agent.execute_search.call_args[0][0]
        self.assertEqual(agent_req.retrieval_mode, RetrievalMode.HYBRID)
        self.assertEqual(resp.status, "SUCCESS")

    def test_04_exact_go_number_forwarding(self) -> None:
        """Verify exact Government Order numbers are passed unaltered to SearchAgent."""
        exact_go = "G.O.(Rt)No.24/2026/ITD"
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query=exact_go,
            requested_mode=RetrievalMode.HYBRID,
            returned_count=0,
            candidates_retained=0,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            results=[],
        )

        req = SearchRequest(query=exact_go)
        self.service.search_documents(req)

        agent_req = self.mock_search_agent.execute_search.call_args[0][0]
        self.assertEqual(agent_req.query, exact_go)

    def test_05_department_filtering_forwarding(self) -> None:
        """Verify department filter is properly mapped and trimmed."""
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="budget",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=0,
            candidates_retained=0,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            results=[],
        )

        req = SearchRequest(query="budget", department="  Finance Department  ")
        self.assertEqual(req.department, "Finance Department")
        self.service.search_documents(req)

        agent_req = self.mock_search_agent.execute_search.call_args[0][0]
        self.assertEqual(agent_req.department_filter, "Finance Department")

    def test_06_multiple_results_mapping_and_relevance_scores(self) -> None:
        """Verify multiple SearchResultItems are mapped to SearchResultItemSchema accurately."""
        item1 = SearchResultItem(
            go_id=1,
            go_number="G.O.(Rt)No.24/2026/ITD",
            department="ITD",
            issue_date="2026-01-15",
            source_filename="go_24.pdf",
            matched_sources=["KEYWORD", "SEMANTIC"],
            rrf_score=0.0328,
            text_snippet="Snippet for GO 24",
        )
        item2 = SearchResultItem(
            go_id=2,
            go_number="G.O.(Ms)No.114/2025/ITD",
            department="ITD",
            issue_date="2025-10-10",
            source_filename="go_114.pdf",
            matched_sources=["SEMANTIC"],
            semantic_score=0.884,
            text_snippet="Snippet for GO 114",
        )
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="it policy",
            requested_mode=RetrievalMode.HYBRID,
            candidates_retained=2,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            returned_count=2,
            results=[item1, item2],
        )

        req = SearchRequest(query="it policy", limit=10)
        resp = self.service.search_documents(req)

        self.assertEqual(resp.status, "SUCCESS")
        self.assertEqual(resp.returned_count, 2)
        self.assertFalse(resp.has_more)
        self.assertTrue(resp.is_exhaustive)
        self.assertEqual(len(resp.results), 2)

        # Check item 1 mapping
        r1 = resp.results[0]
        self.assertEqual(r1.go_id, 1)
        self.assertEqual(r1.go_number, "G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(r1.relevance_score, 0.0328)
        self.assertEqual(r1.matched_sources, ["KEYWORD", "SEMANTIC"])

        # Check item 2 mapping (semantic fallback for relevance_score)
        r2 = resp.results[1]
        self.assertEqual(r2.go_id, 2)
        self.assertEqual(r2.relevance_score, 0.884)
        self.assertEqual(r2.matched_sources, ["SEMANTIC"])

    def test_07_empty_result_handling(self) -> None:
        """Verify empty result from SearchAgent yields NO_MATCH status and empty list."""
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="NO_MATCH",
            query="nonexistent term",
            requested_mode=RetrievalMode.HYBRID,
            candidates_retained=0,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            returned_count=0,
            results=[],
        )

        req = SearchRequest(query="nonexistent term")
        resp = self.service.search_documents(req)

        self.assertEqual(resp.status, "NO_MATCH")
        self.assertEqual(resp.returned_count, 0)
        self.assertEqual(resp.results, [])
        self.assertFalse(resp.has_more)
        self.assertTrue(resp.is_exhaustive)

    def test_08_has_more_only_when_retained_exceeds_limit(self) -> None:
        """
        Verify has_more is True ONLY when candidates_retained > returned count.
        Reaching a candidate ceiling alone must NOT set has_more=True.
        """
        item = SearchResultItem(
            go_id=1,
            go_number="G.O.1",
            source_filename="f1.pdf",
            matched_sources=["KEYWORD"],
        )

        # Sub-case A: Ceiling reached, but retained equals returned (no confirmed additional orders)
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="PARTIAL_SUCCESS",
            query="common topic",
            requested_mode=RetrievalMode.HYBRID,
            candidates_retained=1,
            candidate_ceiling_reached=True,  # Ceiling reached alone
            is_exhaustive=False,
            returned_count=1,
            results=[item],
        )

        req_a = SearchRequest(query="common topic", limit=1)
        resp_a = self.service.search_documents(req_a)
        self.assertFalse(resp_a.has_more, "Ceiling reached alone must NOT set has_more=True")

        # Sub-case B: Retained (5) strictly exceeds returned count (1) -> confirmed more exist
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="PARTIAL_SUCCESS",
            query="common topic",
            requested_mode=RetrievalMode.HYBRID,
            candidates_retained=5,
            candidate_ceiling_reached=False,
            is_exhaustive=False,
            returned_count=1,
            results=[item],
        )

        req_b = SearchRequest(query="common topic", limit=1)
        resp_b = self.service.search_documents(req_b)
        self.assertTrue(resp_b.has_more, "has_more must be True when candidates_retained > returned count")

    def test_09_is_exhaustive_false_when_retrieval_incomplete(self) -> None:
        """Verify is_exhaustive is False whenever ceiling is reached or retrieval is incomplete."""
        item = SearchResultItem(
            go_id=1,
            go_number="G.O.1",
            source_filename="f1.pdf",
            matched_sources=["KEYWORD"],
        )
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="PARTIAL_SUCCESS",
            query="common topic",
            requested_mode=RetrievalMode.HYBRID,
            candidates_retained=1,
            candidate_ceiling_reached=True,
            is_exhaustive=False,
            returned_count=1,
            results=[item],
        )

        req = SearchRequest(query="common topic")
        resp = self.service.search_documents(req)
        self.assertFalse(resp.is_exhaustive)

    @patch("pathlib.Path.is_file")
    def test_10_missing_and_present_pdf_handling(self, mock_is_file: MagicMock) -> None:
        """Verify safe PDF existence checking on disk without failing search."""
        item1 = SearchResultItem(
            go_id=10,
            go_number="G.O. 10",
            source_filename="present.pdf",
            matched_sources=["KEYWORD"],
        )
        item2 = SearchResultItem(
            go_id=20,
            go_number="G.O. 20",
            source_filename="missing.pdf",
            matched_sources=["KEYWORD"],
        )
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="test",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=2,
            candidates_retained=2,
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            results=[item1, item2],
        )

        # Mock: first check (present.pdf) is True, second check (missing.pdf) is False
        mock_is_file.side_effect = [True, False]

        req = SearchRequest(query="test")
        resp = self.service.search_documents(req)

        self.assertEqual(len(resp.results), 2)
        # Item 1 has PDF
        self.assertTrue(resp.results[0].has_pdf)
        self.assertEqual(resp.results[0].pdf_url, "/api/v1/documents/10/pdf")

        # Item 2 missing PDF: should not crash search
        self.assertFalse(resp.results[1].has_pdf)
        self.assertIsNone(resp.results[1].pdf_url)

    def test_11_unsafe_filename_rejection(self) -> None:
        """
        Verify unsafe filenames (absolute paths, traversals, directory components)
        are strictly rejected without silent sanitization, setting has_pdf=False and pdf_url=None.
        """
        unsafe_items = [
            SearchResultItem(go_id=1, go_number="G.O.1", source_filename="../../etc/passwd.pdf", matched_sources=["KEYWORD"]),
            SearchResultItem(go_id=2, go_number="G.O.2", source_filename="C:\\dataset\\file.pdf", matched_sources=["KEYWORD"]),
            SearchResultItem(go_id=3, go_number="G.O.3", source_filename="/var/dataset/file.pdf", matched_sources=["KEYWORD"]),
            SearchResultItem(go_id=4, go_number="G.O.4", source_filename="subdir/file.pdf", matched_sources=["KEYWORD"]),
            SearchResultItem(go_id=5, go_number="G.O.5", source_filename="invalid_ext.txt", matched_sources=["KEYWORD"]),
        ]
        self.mock_search_agent.execute_search.return_value = SearchAgentResponse(
            status="SUCCESS",
            query="test",
            requested_mode=RetrievalMode.HYBRID,
            returned_count=len(unsafe_items),
            candidates_retained=len(unsafe_items),
            candidate_ceiling_reached=False,
            is_exhaustive=True,
            results=unsafe_items,
        )

        req = SearchRequest(query="test")
        resp = self.service.search_documents(req)

        self.assertEqual(len(resp.results), 5)
        for r in resp.results:
            self.assertFalse(r.has_pdf, f"Unsafe filename '{r.source_filename}' must have has_pdf=False")
            self.assertIsNone(r.pdf_url, f"Unsafe filename '{r.source_filename}' must have pdf_url=None")

    def test_12_search_agent_failure_handling(self) -> None:
        """Verify unexpected SearchAgent exception returns ERROR response cleanly."""
        self.mock_search_agent.execute_search.side_effect = RuntimeError("Backend down")

        req = SearchRequest(query="test error")
        resp = self.service.search_documents(req)

        self.assertEqual(resp.status, "ERROR")
        self.assertEqual(resp.returned_count, 0)
        self.assertFalse(resp.has_more)
        self.assertFalse(resp.is_exhaustive)
        self.assertEqual(resp.results, [])

    def test_13_query_response_format_compatibility(self) -> None:
        """Verify _format_response maps search_results and preserves legal_effect_verified=False."""
        state = {
            "workflow_status": "COMPLETED",
            "final_answer": "Final analysis prose",
            "intent": "KEYWORD_SEARCH",
            "search_results": [
                {
                    "go_id": 1,
                    "go_number": "G.O.(Rt)No.24/2026/ITD",
                    "department": "ITD",
                    "source_filename": "go_24.pdf",
                    "matched_sources": ["KEYWORD"],
                }
            ],
            "relationship_results": [
                {
                    "id": 101,
                    "direction": "OUTGOING",
                    "source_go_id": 1,
                    "source_go_number": "G.O.(Rt)No.24/2026/ITD",
                    "target_go_number": "G.O.(Ms)No.114/2025/ITD",
                    "relationship_type": "PARTIALLY_AMENDS",
                    "verification_status": "PENDING",
                }
            ],
            "evidence_sufficient": True,
        }

        resp = self.service._format_response(thread_id="test_thread", state=state)
        self.assertIsInstance(resp, QueryResponse)
        self.assertEqual(resp.thread_id, "test_thread")
        self.assertEqual(resp.workflow_status, "COMPLETED")
        self.assertFalse(resp.legal_effect_verified)
        self.assertEqual(len(resp.search_results), 1)
        self.assertEqual(resp.search_results[0].go_number, "G.O.(Rt)No.24/2026/ITD")
        self.assertEqual(len(resp.relationships), 1)
        self.assertEqual(resp.relationships[0].relationship_type, "PARTIALLY_AMENDS")


if __name__ == "__main__":
    unittest.main()
