"""
I approve preparation of a Phase 4.2 LLM latency optimization plan.

This is PLANNING ONLY. Do not implement anything yet.

Scope:
1. `app/agents/specialized/supervisor_agent.py`
2. `app/agents/llm/ollama_client.py`
3. `app/agents/llm/llm_service.py`, only if necessary.

Prepare a minimal implementation plan addressing:

A. Supervisor prompt optimization:
- Remove duplicated instructions.
- Preserve all six query intents.
- Preserve exact Government Order extraction.
- Preserve incoming/outgoing relationship distinctions.
- Preserve agent/tool mappings and parameter schemas.
- Preserve multi-step planning and replanning.
- Preserve genuine LLM planning and existing heuristic fallback.

B. Ollama generation configuration:
- Propose configurable `num_predict`.
- Propose an appropriate context-window setting only after checking prompt and output requirements.
- Avoid affecting unrelated LLM operations.
- Preserve structured JSON validation and error handling.

C. Safety:
- Keep the existing 120-second client timeout unchanged.
- Keep the 180-second Test 14 parent timeout unchanged.
- Do not weaken assertions.
- Do not remove fallbacks.
- Do not introduce hardcoded intent classification for the relationship query.

D. Validation:
- Identify exact offline tests to run.
- Identify the strict live test to rerun after offline validation.
- Explain how to measure whether prompt optimization actually reduces latency.

Return:
1. Exact files and functions to modify.
2. Proposed changes.
3. Risks and mitigations.
4. Expected validation results.
5. Exact diff scope anticipated.

Do not modify files, execute tests, invoke Ollama, or access databases.

Stop and wait for implementation approval."""
"""
FastAPI Test Suite for Kerala Government Orders Multi-Agent Architecture:
Phase 4.1 Integration Tests

Covers:
1. Health endpoint.
2. Valid query.
3. Empty and whitespace-only query.
4. Query exceeding 1000 characters.
5. Invalid request payload.
6. Workflow execution failure.
7. Human-review interruption.
8. Valid resume.
9. Invalid resume.
10. Status endpoint.
11. legal_effect_verified=False invariant.
12. Preservation of evidence and mandatory disclaimer.
13. Live end-to-end integration test through the API (Excluded from mocked runs).
"""

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport, TimeoutException

from app.api.main import create_app
from app.api.service import WorkflowService
from app.api.schemas import QueryResponse, HealthResponse, ThreadStatusResponse
from app.agents.llm.ollama_client import OllamaClient
from app.database.connection import SessionLocal
from app.services.qdrant_service import QdrantService


class TestFastAPIInvariants(unittest.TestCase):
    """Synchronous test suite for pure schema contracts and legal invariants."""

    # 11. legal_effect_verified=False invariant (Pure synchronous Pydantic schema test)
    def test_11_legal_effect_verified_invariant(self):
        print("\n--- Test 11: legal_effect_verified=False Invariant Enforcement ---")
        # Ensure schema itself rejects True
        from pydantic import ValidationError
        try:
            QueryResponse(
                thread_id="t1",
                workflow_status="COMPLETED",
                legal_effect_verified=True,  # Must be rejected!
            )
            self.fail("QueryResponse should have rejected legal_effect_verified=True")
        except ValidationError:
            print("PASS: Pydantic schema strictly rejects legal_effect_verified=True.")


class TestFastAPIBackend(unittest.IsolatedAsyncioTestCase):
    """Test suite for FastAPI endpoints using asynchronous ASGI transport."""

    async def asyncSetUp(self):
        WorkflowService.reset_instance()
        self.app = create_app()
        self.transport = ASGITransport(app=self.app)
        self.client = AsyncClient(transport=self.transport, base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()
        WorkflowService.reset_instance()

    # 1. Health endpoint test (Mocked)
    async def test_01_health_endpoint_mocked(self):
        print("\n--- Test 1: Health Endpoint (Mocked) ---")
        with patch.object(WorkflowService, "check_health") as mock_health:
            mock_health.return_value = HealthResponse(
                status="healthy",
                service="kerala-government-orders-multi-agent-system",
                version="1.0.0",
                ollama_online=True,
                postgres_online=True,
                qdrant_online=True,
            )
            response = await self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["service"], "kerala-government-orders-multi-agent-system")
            self.assertTrue(data["ollama_online"])
            self.assertTrue(data["postgres_online"])
            self.assertTrue(data["qdrant_online"])
            print("PASS: /health returned 200 OK with verified service status flags.")

    # 2. Valid query request (Mocked workflow)
    async def test_02_valid_query_request(self):
        print("\n--- Test 2: Valid Query Request (Mocked) ---")
        mock_response = QueryResponse(
            thread_id="test_thread_001",
            workflow_status="COMPLETED",
            final_answer="## Government Orders Analysis Report\n\nVerified finding.\n> [!IMPORTANT]\n> **Official Notice:** Legal effect remains unverified.",
            verification_status="VERIFIED",
            confidence="HIGH",
            evidence_sufficient=True,
            legal_effect_verified=False,
            intent="GO_LOOKUP",
            identified_go_numbers=["G.O.(Rt)No.24/2026/ITD"],
            discovered_go_numbers=[],
            human_review_required=False,
            human_review_prompt=None,
            execution_trace_count=5,
            errors=[],
        )
        with patch.object(WorkflowService, "execute_query", return_value=mock_response):
            payload = {
                "query": "Show details for G.O.(Rt)No.24/2026/ITD",
                "thread_id": "test_thread_001",
            }
            response = await self.client.post("/api/v1/query", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["thread_id"], "test_thread_001")
            self.assertEqual(data["workflow_status"], "COMPLETED")
            self.assertEqual(data["verification_status"], "VERIFIED")
            self.assertFalse(data["legal_effect_verified"])
            self.assertIn("Official Notice", data["final_answer"])
            print("PASS: Valid query executed and returned full QueryResponse schema.")

    # 3. Empty and whitespace-only query rejection
    async def test_03_empty_and_whitespace_query_rejection(self):
        print("\n--- Test 3: Empty and Whitespace Query Rejection ---")
        # Empty string
        resp1 = await self.client.post("/api/v1/query", json={"query": ""})
        self.assertEqual(resp1.status_code, 422)

        # Whitespace string
        resp2 = await self.client.post("/api/v1/query", json={"query": "   \n\t   "})
        self.assertEqual(resp2.status_code, 422)
        print("PASS: Empty and whitespace queries strictly rejected with 422 Unprocessable Entity.")

    # 4. Query exceeding 1000 characters
    async def test_04_excessive_length_query_rejection(self):
        print("\n--- Test 4: Query Exceeding 1000 Characters ---")
        long_query = "A" * 1001
        response = await self.client.post("/api/v1/query", json={"query": long_query})
        self.assertEqual(response.status_code, 422)
        print("PASS: Query exceeding 1000 characters strictly rejected with 422.")

    # 5. Invalid request payload
    async def test_05_invalid_request_payload(self):
        print("\n--- Test 5: Invalid Request Payload ---")
        # Missing required 'query' field
        response = await self.client.post("/api/v1/query", json={"thread_id": "only_thread"})
        self.assertEqual(response.status_code, 422)
        # Malformed JSON payload
        response2 = await self.client.post(
            "/api/v1/query",
            content="not a json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response2.status_code, 422)
        print("PASS: Malformed payloads and missing required fields return 422.")

    # 6. Workflow execution failure
    async def test_06_workflow_execution_failure(self):
        print("\n--- Test 6: Workflow Execution Failure Handling ---")
        with patch.object(WorkflowService, "execute_query", side_effect=RuntimeError("Internal engine error")):
            response = await self.client.post("/api/v1/query", json={"query": "Valid query"})
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("Workflow execution failed", data["detail"])
            # Ensure sensitive internals are not leaked
            self.assertNotIn("password", str(data).lower())
            print("PASS: Workflow execution error handled safely with sanitized 500 response.")

    # 7. Human-review interruption
    async def test_07_human_review_interruption(self):
        print("\n--- Test 7: Human-Review Interruption (Mocked) ---")
        mock_interrupted = QueryResponse(
            thread_id="interrupted_thread_123",
            workflow_status="WAITING_FOR_HUMAN_REVIEW",
            final_answer=None,
            verification_status=None,
            confidence="LOW",
            evidence_sufficient=False,
            legal_effect_verified=False,
            intent="AMBIGUOUS_CLARIFICATION",
            identified_go_numbers=[],
            discovered_go_numbers=[],
            human_review_required=True,
            human_review_prompt={
                "prompt": "Human review required",
                "reason": "Ambiguous query: 9 candidate orders match '2026'",
                "user_query": "check 2026",
            },
            execution_trace_count=2,
            errors=[],
        )
        with patch.object(WorkflowService, "execute_query", return_value=mock_interrupted):
            response = await self.client.post("/api/v1/query", json={"query": "check 2026"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["workflow_status"], "WAITING_FOR_HUMAN_REVIEW")
            self.assertTrue(data["human_review_required"])
            self.assertIsNotNone(data["human_review_prompt"])
            print("PASS: Interrupted workflow returned WAITING_FOR_HUMAN_REVIEW with interrupt payload.")

    # 8. Valid resume
    async def test_08_valid_resume(self):
        print("\n--- Test 8: Valid Resume (Mocked) ---")
        mock_resumed = QueryResponse(
            thread_id="interrupted_thread_123",
            workflow_status="COMPLETED",
            final_answer="## Government Orders Analysis Report\n\nResolved via clarification for G.O.(Rt)No.24/2026/ITD.",
            verification_status="VERIFIED",
            confidence="HIGH",
            evidence_sufficient=True,
            legal_effect_verified=False,
            intent="GO_LOOKUP",
            identified_go_numbers=["G.O.(Rt)No.24/2026/ITD"],
            discovered_go_numbers=[],
            human_review_required=False,
            human_review_prompt=None,
            execution_trace_count=6,
            errors=[],
        )
        with patch.object(WorkflowService, "resume_thread", return_value=mock_resumed):
            payload = {
                "thread_id": "interrupted_thread_123",
                "feedback": "I meant G.O.(Rt)No.24/2026/ITD specifically",
            }
            response = await self.client.post("/api/v1/resume", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["workflow_status"], "COMPLETED")
            self.assertFalse(data["human_review_required"])
            self.assertIn("G.O.(Rt)No.24/2026/ITD", data["final_answer"])
            print("PASS: Valid resume completed successfully and returned updated response.")

    # 9. Invalid resume (thread not found or not interrupted)
    async def test_09_invalid_resume(self):
        print("\n--- Test 9: Invalid Resume Handling ---")
        # Non-existent thread -> 404
        with patch.object(WorkflowService, "resume_thread", side_effect=KeyError("Thread not found")):
            resp1 = await self.client.post(
                "/api/v1/resume",
                json={"thread_id": "missing_thread", "feedback": "clarification"},
            )
            self.assertEqual(resp1.status_code, 404)

        # Thread not in paused state -> 400
        with patch.object(WorkflowService, "resume_thread", side_effect=ValueError("Thread is not paused")):
            resp2 = await self.client.post(
                "/api/v1/resume",
                json={"thread_id": "running_thread", "feedback": "clarification"},
            )
            self.assertEqual(resp2.status_code, 400)
        print("PASS: Invalid resume requests rejected with appropriate 404 and 400 statuses.")

    # 10. Status endpoint
    async def test_10_status_endpoint(self):
        print("\n--- Test 10: Status Endpoint (Mocked) ---")
        mock_status = ThreadStatusResponse(
            thread_id="status_thread_123",
            workflow_status="WAITING_FOR_HUMAN_REVIEW",
            next_nodes=["human_review_node"],
            human_review_required=True,
            human_review_prompt={"prompt": "Human review required"},
            is_persisted_across_restarts=False,
        )
        with patch.object(WorkflowService, "get_thread_status", return_value=mock_status):
            response = await self.client.get("/api/v1/status/status_thread_123")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["thread_id"], "status_thread_123")
            self.assertEqual(data["workflow_status"], "WAITING_FOR_HUMAN_REVIEW")
            self.assertFalse(data["is_persisted_across_restarts"])
            self.assertTrue(data["human_review_required"])
            print("PASS: Status endpoint returned thread status and acknowledged volatility.")

    # 12. Evidence and mandatory disclaimer preservation
    async def test_12_evidence_and_disclaimer_preservation(self):
        print("\n--- Test 12: Evidence and Mandatory Disclaimer Preservation ---")
        from app.agents.specialized.response_agent import ResponseAgent
        mock_resp = QueryResponse(
            thread_id="t_disclaimer",
            workflow_status="COMPLETED",
            final_answer=f"Analysis text\n\n{ResponseAgent.MANDATORY_DISCLAIMER}",
            verification_status="VERIFIED",
            evidence_sufficient=True,
            legal_effect_verified=False,
        )
        with patch.object(WorkflowService, "execute_query", return_value=mock_resp):
            response = await self.client.post("/api/v1/query", json={"query": "test query"})
            data = response.json()
            self.assertIn("Official Notice", data["final_answer"])
            self.assertIn("Legal effect remains unverified", data["final_answer"])
            print("PASS: Mandatory legal disclaimer preserved verbatim in final response.")

    # 13. Genuine Live Workflow Integration Test through FastAPI (Excluded from mocked suite)
    def test_13_live_workflow_integration_through_api(self):
        print("\n==========================================")
        print("GENUINE LIVE WORKFLOW INTEGRATION TEST")
        print("==========================================")
        ollama_client = OllamaClient()
        if not ollama_client.is_server_running():
            print("SKIP: Ollama server not reachable on localhost:11434. Skipping live test.")
            return

        from fastapi.testclient import TestClient
        live_app = create_app()
        live_client = TestClient(live_app)

        live_query = "Show details for G.O.(Rt)No.24/2026/ITD"
        print(f"Submitting live query through API: '{live_query}'")

        response = live_client.post(
            "/api/v1/query",
            json={"query": live_query, "thread_id": "live_api_test_thread"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        print(f"API Returned Status: {data.get('workflow_status')}")
        print(f"API Returned Verification Status: {data.get('verification_status')}")
        print(f"API Returned Evidence Sufficient: {data.get('evidence_sufficient')}")
        print(f"API Returned Legal Effect Verified: {data.get('legal_effect_verified')}")
        print(f"API Returned Final Answer Snippet:\n{str(data.get('final_answer'))[:250]}...")

        self.assertEqual(data["workflow_status"], "COMPLETED")
        self.assertEqual(data["verification_status"], "VERIFIED")
        self.assertFalse(data["legal_effect_verified"])
        self.assertIn("24/2026/ITD", str(data["final_answer"]))
        self.assertIn("Official Notice", str(data["final_answer"]))
        print("PASS: Genuine live query executed through FastAPI against PostgreSQL, Qdrant, and Ollama!")

    # 14. Genuine Live Semantic Retrieval & Qdrant Integration Test through FastAPI
    TEST_14_SUCCESS_SIGNAL = "=== TEST 14 VERIFICATION COMPLETE: ALL ASSERTIONS PASSED ==="
    TEST_14_SKIP_SIGNAL = "=== TEST 14 SKIPPED: OLLAMA SERVER NOT REACHABLE ==="

    def test_14_live_semantic_retrieval_through_api(self):
        """
        Process-isolated Test 14 with hard parent timeout enforcement and two-factor signal validation.
        - Parent mode (default): Spawns child in separate process, streams output live via Queue,
          and enforces a strict 180s monotonic deadline with Windows process-tree cleanup.
        - Child mode (AGY_TEST14_CHILD=1): Executes genuine live FastAPI -> LangGraph -> Ollama ->
          Qdrant -> PostgreSQL workflow with all 7 invariant/provenance assertions.
        """
        if os.environ.get("AGY_TEST14_CHILD") != "1":
            self._run_test_14_as_isolated_parent()
        else:
            self._run_test_14_as_child_worker()

    def _run_test_14_as_isolated_parent(self):
        """Parent runner: spawns child subprocess, streams output, and enforces hard deadline."""
        print("\n========================================================")
        print("PARENT RUNNER: LAUNCHING TEST 14 IN ISOLATED PROCESS")
        print("========================================================")
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "unittest",
            "-v",
            "tests.test_api.TestFastAPIBackend.test_14_live_semantic_retrieval_through_api",
        ]
        child_env = os.environ.copy()
        child_env["AGY_TEST14_CHILD"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )

        output_queue: queue.Queue = queue.Queue()
        captured_lines: list[str] = []

        def reader_thread_func():
            try:
                for line in iter(proc.stdout.readline, ""):
                    output_queue.put(line)
            except Exception:
                pass
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

        reader_thread = threading.Thread(target=reader_thread_func, daemon=True)
        reader_thread.start()

        timeout_seconds = 180.0
        start_mono = time.monotonic()
        timed_out = False

        while True:
            # Drain captured output queue in real-time
            while not output_queue.empty():
                try:
                    line = output_queue.get_nowait()
                    captured_lines.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except queue.Empty:
                    break

            # Check if child process has completed
            ret = proc.poll()
            if ret is not None:
                break

            # Enforce hard deadline via monotonic clock independently of child output
            elapsed = time.monotonic() - start_mono
            if elapsed > timeout_seconds:
                timed_out = True
                break

            time.sleep(0.1)

        if timed_out:
            elapsed = time.monotonic() - start_mono
            print(f"\n[TIMEOUT] Child process exceeded hard {timeout_seconds:.1f}s limit (elapsed: {elapsed:.1f}s). Terminating process tree...")

            # 1. Invoke Windows taskkill to terminate child process tree
            try:
                tk_res = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                )
                if tk_res.returncode != 0:
                    err_msg = tk_res.stderr.strip() if tk_res.stderr else "No stderr output"
                    print(f"[CLEANUP WARNING] taskkill returned non-zero code {tk_res.returncode}: {err_msg}")
            except Exception as e:
                print(f"[CLEANUP ERROR] taskkill failed: {e}")

            # 2. proc.kill() fallback
            try:
                proc.kill()
            except Exception:
                pass

            # 3. Reap child process
            try:
                proc.wait(timeout=5.0)
            except Exception:
                pass

            reader_thread.join(timeout=2.0)
            try:
                if proc.stdout and not proc.stdout.closed:
                    proc.stdout.close()
            except Exception:
                pass

            self.fail(
                f"Test 14 TIMEOUT: Child process exceeded hard {timeout_seconds:.1f}s deadline (elapsed: {elapsed:.1f}s). "
                f"Child process (PID {proc.pid}) was terminated via taskkill and its network connection closed. "
                f"Note: Terminating the child process and closing its connection stops all client-side LangGraph "
                f"execution, but immediate cancellation of in-flight Ollama server-side generation depends on Ollama."
            )

        # Normal completion: ensure child has completely exited
        try:
            proc.wait(timeout=5.0)
        except Exception:
            pass

        # Join the reader thread with a bounded timeout to guarantee zero output loss
        reader_thread.join(timeout=3.0)
        if reader_thread.is_alive():
            self.fail("Test 14 output-capture failure: reader thread did not terminate within bounded join timeout.")

        # Drain any remaining lines in output queue
        while not output_queue.empty():
            try:
                line = output_queue.get_nowait()
                captured_lines.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            except queue.Empty:
                break

        # Safely close child stdout pipe
        try:
            if proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
        except Exception:
            pass

        full_output = "".join(captured_lines)
        has_success_signal = self.TEST_14_SUCCESS_SIGNAL in full_output
        has_skip_signal = self.TEST_14_SKIP_SIGNAL in full_output

        # Classification Rule 6: Both signals -> FAIL (invalid outcome)
        if has_success_signal and has_skip_signal:
            self.fail("Test 14 INVALID OUTCOME: Child emitted both SUCCESS and SKIP signals.")

        # Classification Rule 4: Non-zero exit code -> FAIL
        if proc.returncode != 0:
            self.fail(f"Test 14 child process failed with non-zero exit code: {proc.returncode}.")

        # Classification Rule 2: Exit code 0 + skip signal -> SKIP
        if has_skip_signal:
            self.skipTest("BLOCKED: Ollama server not reachable on localhost:11434 (propagated from child worker).")

        # Classification Rule 1: Exit code 0 + success signal -> PASS
        if has_success_signal:
            return

        # Classification Rule 3: Exit code 0 + neither signal -> FAIL (incomplete)
        self.fail(
            "Test 14 INCOMPLETE: Child process exited with return code 0 but emitted neither SUCCESS nor SKIP signal. "
            "Workflow or assertions terminated prematurely."
        )

    def _run_test_14_as_child_worker(self):
        """Child worker: executes genuine live FastAPI, LangGraph, Ollama, Qdrant, and PostgreSQL."""
        print("\n==========================================")
        print("GENUINE LIVE SEMANTIC & QDRANT INTEGRATION TEST (CHILD WORKER)")
        print("==========================================")
        ollama_client = OllamaClient()
        if not ollama_client.is_server_running():
            print(self.TEST_14_SKIP_SIGNAL)
            sys.stdout.flush()
            self.skipTest("BLOCKED: Ollama server not reachable on localhost:11434. Live test cannot proceed.")

        # Configure real-time progress logging to stdout for full intermediate visibility
        log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout, force=True)

        from fastapi.testclient import TestClient
        from app.agents.specialized.response_agent import ResponseAgent
        live_app = create_app()
        live_client = TestClient(live_app)

        query = "Find orders regarding dearness relief and pension revision allowances"
        thread_id = "live_semantic_test_thread"
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Submitting semantic query through API: '{query}'")

        # 1. Dispatch live request through FastAPI endpoint
        response = live_client.post(
            "/api/v1/query",
            json={"query": query, "thread_id": thread_id},
        )

        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] API request completed in {elapsed:.2f}s with HTTP {response.status_code}")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        print(f"API Returned Status: {data.get('workflow_status')}")
        print(f"API Returned Intent: {data.get('intent')}")
        print(f"API Returned Legal Effect Verified: {data.get('legal_effect_verified')}")
        print(f"API Returned Final Answer Snippet:\n{str(data.get('final_answer'))[:250]}...")

        # 2. Strict Invariant Checks on API Response
        self.assertEqual(data["workflow_status"], "COMPLETED")
        self.assertFalse(data["legal_effect_verified"], "legal_effect_verified must strictly remain False")
        self.assertEqual(data["intent"], "KEYWORD_SEARCH")
        self.assertIn("Official Notice", str(data["final_answer"]))
        self.assertIn(ResponseAgent.MANDATORY_DISCLAIMER, data["final_answer"])

        # 3. Inspect In-Memory Checkpoint State for Execution Provenance
        service = WorkflowService.get_instance()
        state = service.graph.get_state({"configurable": {"thread_id": thread_id}}).values
        self.assertIsNotNone(state, "Thread state checkpoint must exist in MemorySaver.")

        # 4. Prove Genuine LLM Planning
        plan = state.get("execution_plan") or {}
        plan_source = plan.get("plan_source")
        raw_intent = plan.get("intent")
        plan_intent = raw_intent.value if hasattr(raw_intent, "value") else str(raw_intent)
        print(f"Supervisor Plan Source: {plan_source} | Intent: {plan_intent}")
        self.assertEqual(plan_source, "LLM", f"Expected genuine LLM planning, got {plan_source}")
        self.assertEqual(plan_intent, "KEYWORD_SEARCH")

        # 5. Direct Evidence of SearchAgent and Qdrant Tool Invocation in Trace
        execution_trace = state.get("execution_trace", [])
        search_trace = [t for t in execution_trace if t.get("agent") == "search_agent"]
        self.assertGreater(len(search_trace), 0, "Expected execution trace record for search_agent.")
        tool_called = search_trace[0].get("tool_called")
        print(f"SearchAgent Trace Tool Called: {tool_called}")
        self.assertEqual(tool_called, "search_tool")

        # 6. Prove Qdrant Semantic Retrieval & Provenance
        search_results = state.get("search_results", [])
        print(f"Total search results retrieved: {len(search_results)}")
        self.assertGreater(len(search_results), 0, "SearchAgent must return at least one search result.")

        # Verify semantic provenance across retrieved results
        semantic_hits = [r for r in search_results if "SEMANTIC" in r.get("matched_sources", [])]
        print(f"Results with confirmed SEMANTIC matched_source: {len(semantic_hits)}")
        self.assertGreater(
            len(semantic_hits), 0,
            "Direct proof failure: zero results have 'SEMANTIC' in matched_sources; Qdrant was not queried."
        )

        top_semantic = semantic_hits[0]
        sem_rank = top_semantic.get("semantic_rank")
        sem_score = top_semantic.get("semantic_score")
        supporting_chunks = top_semantic.get("supporting_chunks", [])
        print(f"Top Semantic Hit GO: {top_semantic.get('go_number')} | Rank: {sem_rank} | Score: {sem_score}")

        self.assertIsNotNone(sem_rank, "Semantic hit must have a valid semantic_rank.")
        self.assertGreaterEqual(sem_rank, 1)
        self.assertIsNotNone(sem_score, "Semantic hit must have a valid cosine similarity score.")
        self.assertGreater(sem_score, 0.0, "Semantic cosine similarity score must be greater than 0.0.")
        self.assertGreater(len(supporting_chunks), 0, "Semantic hit must retain supporting chunks with text evidence.")

        # 7. Report Actual Retrieved Orders
        retrieved_gos = [r.get("go_number") for r in search_results]
        print(f"Actual Retrieved Orders: {retrieved_gos}")
        target_found = any("27/2026/FIN" in str(go) for go in retrieved_gos)
        if target_found:
            print("Target Order G.O.(P)No.27/2026/FIN successfully matched and retrieved.")
        else:
            print(f"Informational notice: Target 27/2026/FIN was not in top results; retrieved orders: {retrieved_gos}")

        print("PASS: Genuine semantic retrieval through Qdrant, PostgreSQL, and LLM verified!")
        print(self.TEST_14_SUCCESS_SIGNAL)
        sys.stdout.flush()


if __name__ == "__main__":
    unittest.main()
