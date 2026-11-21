"""
Focused Tests for Phase 2 Step 2.1 Corrections:
1. BGE Query Prefix Validation (embed_query vs embed_text).
2. Page-number handling (preserving real page numbers, None when unavailable).
3. Crash Recovery Scenarios in QdrantService:
   - Crash before activation (incomplete staged chunks).
   - Crash during activation (partial active chunks).
   - Crash after activation before cleanup (multiple complete active versions).
   - Reopening Qdrant and running reconciliation on startup.
   - Idempotency of reconciliation (running multiple times safely).
   - Ensuring only the correct complete version is searchable.
"""

import shutil
from typing import Any, Dict, List
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config import PROJECT_ROOT, EMBEDDING_DIMENSION
from app.services.embedding_service import EmbeddingService, BGE_QUERY_PREFIX
from app.services.qdrant_service import QdrantService
from scripts.index_qdrant import chunk_document_text


def test_bge_query_prefix_behavior() -> None:
    print("\n--- 1. BGE Query Prefix Validation ---")
    service = EmbeddingService()

    # 1. Verify query prefix is applied
    raw_query = "technology incubators startup grant"
    vec_query = service.embed_query(raw_query)
    assert len(vec_query) == 384, f"Expected 384 dimensions, got {len(vec_query)}"

    # 2. Verify that manual pre-prefixed query is not double-prefixed
    prefixed_input = f"{BGE_QUERY_PREFIX}{raw_query}"
    vec_prefixed = service.embed_query(prefixed_input)
    assert vec_query == vec_prefixed, "Double prefix was incorrectly added"

    # 3. Verify document embedding does NOT receive the query prefix
    vec_doc = service.embed_text(raw_query)
    assert len(vec_doc) == 384
    # Query vector and document vector for the same text MUST differ because query vector has BGE prefix
    assert vec_query != vec_doc, "Query embedding and document embedding should differ due to BGE instruction prefix"

    # 4. Verify batch document embedding works without prefix
    batch = ["First passage text", "Second passage text"]
    vec_batch = service.embed_batch(batch)
    assert len(vec_batch) == 2
    assert len(vec_batch[0]) == 384
    assert len(vec_batch[1]) == 384

    # 5. Verify empty and whitespace query handling raises ValueError
    try:
        service.embed_query("")
        assert False, "Expected ValueError on empty query string"
    except ValueError as exc:
        assert "cannot be empty" in str(exc).lower()

    try:
        service.embed_query("   ")
        assert False, "Expected ValueError on whitespace query string"
    except ValueError as exc:
        assert "cannot be empty" in str(exc).lower()

    print("PASS: embed_query correctly applies BGE prefix once and raises ValueError on empty/whitespace; embed_text/embed_batch remain clean; 384 dimensions verified.")


def test_page_number_extraction_and_null_handling() -> None:
    print("\n--- 2. Page-Number Extraction & Null Handling ---")

    # Document WITH explicit page markers
    doc_with_pages = """--- Page 1 ---
G.O.(Rt) No. 24/2026/ITD
Read 1 G.O (Rt) No. 114/2025/ITD

ORDER
Sanction accorded for project.
--- Page 2 ---
Clause 2: Financial disbursement details.
Clause 3: Monitoring mechanism.
"""
    chunks_with_pages = chunk_document_text(doc_with_pages, target_chunk_size=200)
    assert len(chunks_with_pages) >= 2
    assert chunks_with_pages[0]["page_number"] == 1, f"Expected page 1, got {chunks_with_pages[0]['page_number']}"
    assert any(c["page_number"] == 2 for c in chunks_with_pages), "Failed to capture page 2"
    print("PASS: Documents with page markers accurately preserve true page numbers.")

    # Document WITHOUT page markers
    doc_without_pages = """GOVERNMENT OF KERALA
Abstract: Electronics & Information Technology Department.
ORDER
Administrative Sanction accorded for digital transformation schemes.
Paragraph 2: Implementation through KSITM.
"""
    chunks_without_pages = chunk_document_text(doc_without_pages, target_chunk_size=200)
    assert len(chunks_without_pages) >= 1
    for c in chunks_without_pages:
        assert c["page_number"] is None, f"Expected page_number to be None when markers absent, got {c['page_number']}"
    print("PASS: Documents without page markers store page_number as None (no artificial page 1 assigned).")


def test_qdrant_crash_recovery_scenarios() -> None:
    print("\n--- 3. Qdrant Crash Recovery Scenarios ---")
    test_storage = PROJECT_ROOT / "storage" / "qdrant_test_crash_recovery"
    if test_storage.exists():
        shutil.rmtree(test_storage)

    embedding_service = EmbeddingService()
    qdrant = QdrantService(
        storage_path=test_storage,
        collection_name="test_crash_collection",
        dimension=EMBEDDING_DIMENSION,
    )

    test_go_id = 777
    test_go_num = "G.O.(Test)No.777/2026/RECOVERY"

    # Step 1: Base State - Version 1 indexed completely and actively
    v1_chunks = [
        {"chunk_index": 1, "chunk_text": "Clause 1 of Version 1", "page_number": None},
        {"chunk_index": 2, "chunk_text": "Clause 2 of Version 1", "page_number": None},
    ]
    res_v1 = qdrant.safe_index_document(
        go_id=test_go_id,
        go_number=test_go_num,
        department="Test Dept",
        source_filename="test.pdf",
        issue_date="2026-01-01",
        chunks=v1_chunks,
        embedding_service=embedding_service,
    )
    assert res_v1["status"] == "SUCCESS"
    assert res_v1["version"] == 1
    assert qdrant.get_order_active_version(test_go_id) == 1

    # SCENARIO A: Crash before activation (incomplete staged chunks left in collection)
    print("  Testing Scenario A: Crash before activation (staged is_active=False)...")
    staged_pt_id = qdrant.generate_point_id(test_go_id, version=2, chunk_index=1)
    qdrant.client.upsert(
        collection_name=qdrant.collection_name,
        points=[
            PointStruct(
                id=staged_pt_id,
                vector=embedding_service.embed_text("Orphaned staged chunk"),
                payload={
                    "go_id": test_go_id,
                    "go_number": test_go_num,
                    "version": 2,
                    "chunk_index": 1,
                    "total_chunks": 3,  # Missing chunks 2 and 3!
                    "is_active": False,
                },
            )
        ],
    )
    # Search should NOT see the staged point
    query_vec = embedding_service.embed_query("Orphaned staged chunk")
    hits = qdrant.search_similar(query_vec, limit=5)
    for h in hits:
        assert h["version"] == 1, "Inactive staged chunk leaked into search results"

    # Reconciliation should prune the incomplete staged point
    rec_res = qdrant.reconcile_versions(target_go_id=test_go_id)
    assert rec_res["pruned_incomplete_versions"] >= 1
    assert qdrant.get_order_active_version(test_go_id) == 1
    print("  PASS: Inactive staged chunks excluded from search and cleanly pruned by reconciliation.")

    # SCENARIO B1: Crash during upload with partial active chunk (incomplete chunks)
    print("  Testing Scenario B1: Crash during upload with partial active chunk...")
    partial_pt_id = qdrant.generate_point_id(test_go_id, version=2, chunk_index=1)
    qdrant.client.upsert(
        collection_name=qdrant.collection_name,
        points=[
            PointStruct(
                id=partial_pt_id,
                vector=embedding_service.embed_text("Partial chunk activated before crash"),
                payload={
                    "go_id": test_go_id,
                    "go_number": test_go_num,
                    "version": 2,
                    "chunk_index": 1,
                    "total_chunks": 4,  # Says total 4, but only 1 exists!
                    "is_active": True,  # Leaked active flag
                },
            )
        ],
    )
    # Reconciliation must detect that Version 2 is INCOMPLETE despite is_active=True
    rec_b = qdrant.reconcile_versions(target_go_id=test_go_id)
    assert rec_b["pruned_incomplete_versions"] >= 1
    # Version 1 must remain the intact active version!
    assert qdrant.get_order_active_version(test_go_id) == 1
    print("  PASS: Incomplete version with partial is_active=True detected and pruned; Version 1 preserved.")

    # SCENARIO B2: Crash during activation (ALL chunks uploaded, but only some marked is_active=True)
    print("  Testing Scenario B2: Crash during activation (mixed active/inactive on complete chunks)...")
    v2_p1 = qdrant.generate_point_id(test_go_id, version=2, chunk_index=1)
    v2_p2 = qdrant.generate_point_id(test_go_id, version=2, chunk_index=2)
    qdrant.client.upsert(
        collection_name=qdrant.collection_name,
        points=[
            PointStruct(
                id=v2_p1,
                vector=embedding_service.embed_text("Mixed Version 2 chunk 1 - activated"),
                payload={
                    "go_id": test_go_id,
                    "go_number": test_go_num,
                    "version": 2,
                    "chunk_index": 1,
                    "total_chunks": 2,
                    "is_active": True,  # Already activated before crash
                },
            ),
            PointStruct(
                id=v2_p2,
                vector=embedding_service.embed_text("Mixed Version 2 chunk 2 - still inactive"),
                payload={
                    "go_id": test_go_id,
                    "go_number": test_go_num,
                    "version": 2,
                    "chunk_index": 2,
                    "total_chunks": 2,
                    "is_active": False,  # Not yet activated when crash occurred
                },
            ),
        ],
    )

    # Before reconciliation: Version 2 is incomplete in terms of active points (only 1 of 2 is active)
    # Therefore get_order_active_version must reject Version 2 and continue returning Version 1
    assert qdrant.get_order_active_version(test_go_id) == 1, (
        "Partially activated version must not be recognized as active version before reconciliation"
    )

    # Reconciliation runs: detects Version 2 is fully complete on disk, completes activation of chunk 2,
    # and prunes obsolete Version 1
    rec_b2 = qdrant.reconcile_versions(target_go_id=test_go_id)
    assert rec_b2["pruned_obsolete_versions"] >= 1
    assert qdrant.get_order_active_version(test_go_id) == 2

    # Verify both chunks of Version 2 are now active in Qdrant
    active_v2_pts = qdrant.get_active_points_for_order(test_go_id)
    assert len(active_v2_pts) == 2
    assert all((p.payload or {}).get("is_active") is True for p in active_v2_pts)
    assert all((p.payload or {}).get("version") == 2 for p in active_v2_pts)
    print("  PASS: Mixed-activation crash safely detected, completed to is_active=True, and obsolete V1 pruned.")

    # SCENARIO C: Crash after activation before cleanup (two complete active versions)
    print("  Testing Scenario C: Crash after activation before cleanup (two complete versions)...")
    # Manually insert complete Version 3 points with is_active=True (simulating crash right after Step 5)
    v3_p1 = qdrant.generate_point_id(test_go_id, version=3, chunk_index=1)
    v3_p2 = qdrant.generate_point_id(test_go_id, version=3, chunk_index=2)
    qdrant.client.upsert(
        collection_name=qdrant.collection_name,
        points=[
            PointStruct(
                id=v3_p1,
                vector=embedding_service.embed_text("Complete Version 3 chunk 1"),
                payload={
                    "go_id": test_go_id,
                    "go_number": test_go_num,
                    "version": 3,
                    "chunk_index": 1,
                    "total_chunks": 2,
                    "is_active": True,
                },
            ),
            PointStruct(
                id=v3_p2,
                vector=embedding_service.embed_text("Complete Version 3 chunk 2"),
                payload={
                    "go_id": test_go_id,
                    "go_number": test_go_num,
                    "version": 3,
                    "chunk_index": 2,
                    "total_chunks": 2,
                    "is_active": True,
                },
            ),
        ],
    )
    # Now both Version 2 and Version 3 are active!
    # Reconciliation should detect this, keep highest complete version (V3), and prune obsolete V2
    rec_c = qdrant.reconcile_versions(target_go_id=test_go_id)
    assert rec_c["pruned_obsolete_versions"] >= 1
    assert qdrant.get_order_active_version(test_go_id) == 3

    # Verify search only sees Version 3
    hits_v3 = qdrant.search_similar(embedding_service.embed_query("Version"), limit=10)
    for h in hits_v3:
        assert h["version"] == 3, f"Old version leaked into search results: {h}"
    print("  PASS: Multiple active complete versions resolved to highest version (V3); obsolete V2 pruned.")

    # SCENARIO D: Reconciliation Idempotency
    print("  Testing Scenario D: Reconciliation idempotency...")
    rec_repeat = qdrant.reconcile_versions(target_go_id=test_go_id)
    assert rec_repeat["pruned_incomplete_versions"] == 0
    assert rec_repeat["pruned_obsolete_versions"] == 0
    assert qdrant.get_order_active_version(test_go_id) == 3
    print("  PASS: Second reconciliation run made 0 changes; state remains perfectly stable.")

    # SCENARIO E: Reopen Qdrant and verify startup reconciliation
    print("  Testing Scenario E: Client close, reopen, and startup recovery...")
    qdrant.close()

    # Reopen from disk
    qdrant_reopened = QdrantService(
        storage_path=test_storage,
        collection_name="test_crash_collection",
        dimension=EMBEDDING_DIMENSION,
    )
    assert qdrant_reopened.get_order_active_version(test_go_id) == 3
    search_reopened = qdrant_reopened.search_similar(embedding_service.embed_query("Complete Version 3"), limit=2)
    assert len(search_reopened) == 2
    assert search_reopened[0]["version"] == 3
    print("  PASS: Storage persisted across client close/reopen; version 3 intact and searchable.")

    qdrant_reopened.close()
    if test_storage.exists():
        shutil.rmtree(test_storage)


if __name__ == "__main__":
    try:
        test_bge_query_prefix_behavior()
        test_page_number_extraction_and_null_handling()
        test_qdrant_crash_recovery_scenarios()
        print("\n=======================================================")
        print("ALL STEP 2.1 CORRECTION TESTS PASSED SUCCESSFULLY!")
        print("=======================================================")
    except Exception as exc:
        print(f"\nTEST SUITE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
