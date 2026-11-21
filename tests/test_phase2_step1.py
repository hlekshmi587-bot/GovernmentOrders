"""
Comprehensive Tests for Phase 2 Step 2.1: Semantic Vector Infrastructure.

Verifies:
1. EmbeddingService: FastEmbed ONNX generation and actual dimension (384).
2. QdrantService: Local persistent storage, collection initialization, and persistent client handling.
3. Live Indexing of 11 Kerala Government Orders from PostgreSQL.
4. Metadata and document linkage verification.
5. Duplicate indexing prevention via SHA-256 content hashes.
6. Safe re-indexing with atomic versioning.
7. Simulated indexing failure: confirms previous active index remains working.
8. Stale or missing PostgreSQL record handling.
9. Persistent index availability after closing and reopening the Qdrant client.
10. Database baseline integrity check (PostgreSQL remains completely untouched).
"""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import psutil

from app.config import (
    QDRANT_STORAGE_PATH,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    PROJECT_ROOT,
)
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from scripts.index_qdrant import run_indexing, chunk_document_text
from app.database.connection import SessionLocal
from app.database.models import GovernmentOrderDB
from sqlalchemy import text


def test_embedding_generation_and_dimension() -> None:
    print("\n--- 1. Embedding Generation & Dimension Validation ---")
    start = time.time()
    service = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)
    dim = service.get_dimension()
    assert dim == 384, f"Expected 384 dimensions, got {dim}"

    sample_text = "Government Order regarding sanction of grant-in-aid to technology incubators."
    vector = service.embed_text(sample_text)
    assert len(vector) == 384, f"Single embedding vector dimension mismatch: {len(vector)}"
    assert any(v != 0.0 for v in vector), "Vector contains all zeros"

    batch = [
        "First clause of the administrative order.",
        "Second paragraph with financial sanctions and conditions.",
    ]
    vectors = service.embed_batch(batch)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    elapsed = round(time.time() - start, 3)
    print(f"PASS: FastEmbed ONNX embedding verified. Dimension={dim}. Single & batch generation working ({elapsed}s).")


def test_live_indexing_of_11_orders() -> Dict[str, Any]:
    print("\n--- 2. Live Indexing of 11 Government Orders into Qdrant ---")
    stats = run_indexing()

    assert stats["orders_processed"] == 11, f"Expected 11 orders, got {stats['orders_processed']}"
    assert stats["active_points_in_qdrant"] > 0, "No active points in Qdrant after indexing"
    assert stats["embedding_dimension"] == 384
    print(f"PASS: Successfully indexed 11 orders into Qdrant.")
    print(f"      Active points: {stats['active_points_in_qdrant']}, Peak RSS: {stats['peak_rss_mb']} MB, Time: {stats['elapsed_seconds']}s")
    return stats


def test_metadata_and_document_linkage() -> None:
    print("\n--- 3. Metadata & PostgreSQL Document Linkage Verification ---")
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService(
        storage_path=QDRANT_STORAGE_PATH,
        collection_name=QDRANT_COLLECTION_NAME,
        dimension=embedding_service.get_dimension(),
    )

    # Search for an ITD order topic
    query_vec = embedding_service.embed_text("technology incubators startup grant")
    results = qdrant_service.search_similar(query_vec, limit=3)

    assert len(results) > 0, "Vector search returned zero results"
    top = results[0]

    # Verify all essential metadata fields are present and correctly linked
    assert top["go_id"] is not None, "go_id is missing"
    assert top["go_number"] is not None, "go_number is missing"
    assert top["source_filename"] is not None, "source_filename is missing"
    assert top["chunk_index"] is not None, "chunk_index is missing"
    assert top["chunk_text"] is not None, "chunk_text is missing"
    assert top["version"] is not None, "version is missing"

    # Verify document linkage against PostgreSQL
    with SessionLocal() as session:
        db_order = session.query(GovernmentOrderDB).filter(GovernmentOrderDB.id == top["go_id"]).first()
        assert db_order is not None, f"Linked PostgreSQL order with id={top['go_id']} not found in database!"
        assert db_order.go_number == top["go_number"], f"GO number mismatch: DB={db_order.go_number}, Vector={top['go_number']}"

    print(f"PASS: Metadata and PostgreSQL linkage verified for top match: {top['go_number']} (go_id={top['go_id']}, score={top['score']:.4f}).")


def test_duplicate_indexing_prevention() -> None:
    print("\n--- 4. Duplicate Indexing Prevention (Idempotency) ---")
    # Run indexing a second time without text changes
    stats_repeat = run_indexing()
    assert stats_repeat["unchanged_orders"] == 11, f"Expected 11 unchanged orders on repeat, got {stats_repeat['unchanged_orders']}"
    assert stats_repeat["indexed_orders"] == 0, f"Expected 0 re-indexed orders, got {stats_repeat['indexed_orders']}"
    print(f"PASS: Idempotency confirmed. All 11 orders identified as UNCHANGED via content hash; zero duplicate vectors created.")


def test_safe_reindexing_and_simulated_failure() -> None:
    print("\n--- 5. Safe Re-Indexing & Simulated Failure Rollback ---")
    test_storage = PROJECT_ROOT / "storage" / "qdrant_test_rollback"
    if test_storage.exists():
        shutil.rmtree(test_storage)

    embedding_service = EmbeddingService()
    test_qdrant = QdrantService(
        storage_path=test_storage,
        collection_name="test_rollback_collection",
        dimension=embedding_service.get_dimension(),
    )

    test_go_id = 901
    test_go_num = "G.O.(Test)No.901/2026/TEST"

    # Step 1: Index Version 1
    v1_chunks = [
        {"chunk_index": 1, "chunk_text": "Original version 1 clause A."},
        {"chunk_index": 2, "chunk_text": "Original version 1 clause B."},
    ]
    res_v1 = test_qdrant.safe_index_document(
        go_id=test_go_id,
        go_number=test_go_num,
        department="Test Department",
        source_filename="test.pdf",
        issue_date="2026-01-01",
        chunks=v1_chunks,
        embedding_service=embedding_service,
    )
    assert res_v1["status"] == "SUCCESS"
    assert res_v1["version"] == 1

    # Verify version 1 is active and searchable
    vec = embedding_service.embed_text("Original version 1 clause A")
    search_v1 = test_qdrant.search_similar(vec, limit=2)
    assert len(search_v1) == 2
    assert search_v1[0]["version"] == 1

    # Step 2: Simulate Failure during Version 2 indexing
    class FailingEmbeddingService:
        def embed_batch(self, texts: List[str]) -> List[List[float]]:
            raise RuntimeError("Simulated network/hardware failure during embedding generation!")

    try:
        test_qdrant.safe_index_document(
            go_id=test_go_id,
            go_number=test_go_num,
            department="Test Department",
            source_filename="test.pdf",
            issue_date="2026-01-01",
            chunks=[{"chunk_index": 1, "chunk_text": "Updated version 2 text."}],
            embedding_service=FailingEmbeddingService(),
        )
        assert False, "Failed to raise simulated failure"
    except RuntimeError:
        print("  (Simulated failure triggered as expected)")

    # Confirm that Version 1 remains completely intact and active!
    search_after_fail = test_qdrant.search_similar(vec, limit=2)
    assert len(search_after_fail) == 2
    assert search_after_fail[0]["version"] == 1
    assert "Original version 1" in search_after_fail[0]["chunk_text"]
    print("PASS: Simulated indexing failure did NOT destroy active index. Version 1 remained completely usable.")

    # Step 3: Now successfully index Version 2
    v2_chunks = [
        {"chunk_index": 1, "chunk_text": "Updated replacement version 2 clause A."},
        {"chunk_index": 2, "chunk_text": "Updated replacement version 2 clause B."},
    ]
    res_v2 = test_qdrant.safe_index_document(
        go_id=test_go_id,
        go_number=test_go_num,
        department="Test Department",
        source_filename="test.pdf",
        issue_date="2026-01-01",
        chunks=v2_chunks,
        embedding_service=embedding_service,
    )
    assert res_v2["status"] == "SUCCESS"
    assert res_v2["version"] == 2

    # Verify Version 2 is active and old Version 1 was pruned
    v2_vec = embedding_service.embed_text("Updated replacement version 2")
    search_v2 = test_qdrant.search_similar(v2_vec, limit=5)
    assert len(search_v2) == 2
    for item in search_v2:
        assert item["version"] == 2
        assert "version 1" not in item["chunk_text"]
    print("PASS: Successful version 2 indexing activated cleanly; obsolete version 1 vectors pruned.")

    test_qdrant.close()
    if test_storage.exists():
        shutil.rmtree(test_storage)


def test_stale_or_missing_postgresql_record_handling() -> None:
    print("\n--- 6. Stale or Missing PostgreSQL Record Handling ---")
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService(
        storage_path=QDRANT_STORAGE_PATH,
        collection_name=QDRANT_COLLECTION_NAME,
        dimension=embedding_service.get_dimension(),
    )

    query_vec = embedding_service.embed_text("general administration pension rules")
    raw_hits = qdrant_service.search_similar(query_vec, limit=5)

    # Filter against PostgreSQL to check for stale/deleted orders
    with SessionLocal() as session:
        existing_ids = {row[0] for row in session.execute(text("SELECT id FROM government_orders")).fetchall()}

    valid_hits = []
    stale_hits = []
    for hit in raw_hits:
        if hit["go_id"] in existing_ids:
            valid_hits.append(hit)
        else:
            stale_hits.append(hit)

    assert len(valid_hits) == len(raw_hits), f"Found unexpected stale hits in clean corpus: {stale_hits}"
    print(f"PASS: All {len(valid_hits)} vector hits confirmed backed by existing PostgreSQL records.")


def test_persistent_index_after_closing_and_reopening() -> None:
    print("\n--- 7. Persistent Index Availability After Client Reopen ---")
    embedding_service = EmbeddingService()
    
    # 1. Close the current QdrantService client
    qdrant_service = QdrantService(
        storage_path=QDRANT_STORAGE_PATH,
        collection_name=QDRANT_COLLECTION_NAME,
        dimension=embedding_service.get_dimension(),
    )
    initial_count = qdrant_service.count_active_points()
    assert initial_count > 0, "No active points before closing"
    qdrant_service.close()

    # 2. Re-instantiate from the persistent disk storage
    reopened_service = QdrantService(
        storage_path=QDRANT_STORAGE_PATH,
        collection_name=QDRANT_COLLECTION_NAME,
        dimension=embedding_service.get_dimension(),
    )
    reopened_count = reopened_service.count_active_points()
    assert reopened_count == initial_count, f"Count mismatch after reopen: before={initial_count}, after={reopened_count}"

    # 3. Perform a search to confirm vector index works immediately
    query_vec = embedding_service.embed_text("information technology department order")
    results = reopened_service.search_similar(query_vec, limit=2)
    assert len(results) > 0
    print(f"PASS: Persistent storage verified. Client reopened cleanly; all {reopened_count} active points immediately available.")


def test_postgresql_baseline_integrity() -> None:
    print("\n--- 8. PostgreSQL Baseline Integrity Check ---")
    with SessionLocal() as session:
        orders_count = session.execute(text("SELECT count(*) FROM government_orders")).scalar()
        refs_count = session.execute(text("SELECT count(*) FROM go_references")).scalar()
        rels_count = session.execute(text("SELECT count(*) FROM go_relationships")).scalar()
        verif_count = session.execute(text("SELECT count(*) FROM verification_records")).scalar()

    assert orders_count == 11, f"Expected 11 orders, found {orders_count}"
    assert refs_count == 4, f"Expected 4 references, found {refs_count}"
    assert rels_count == 6, f"Expected 6 relationships, found {rels_count}"
    assert verif_count == 6, f"Expected 6 verification records, found {verif_count}"
    print(f"PASS: PostgreSQL database remains 100% intact: orders={orders_count}, refs={refs_count}, rels={rels_count}, verif={verif_count}.")


if __name__ == "__main__":
    try:
        test_embedding_generation_and_dimension()
        test_live_indexing_of_11_orders()
        test_metadata_and_document_linkage()
        test_duplicate_indexing_prevention()
        test_safe_reindexing_and_simulated_failure()
        test_stale_or_missing_postgresql_record_handling()
        test_persistent_index_after_closing_and_reopening()
        test_postgresql_baseline_integrity()
        print("\n=======================================================")
        print("ALL PHASE 2 STEP 2.1 TESTS PASSED SUCCESSFULLY!")
        print("=======================================================")
    except Exception as exc:
        print(f"\nTEST SUITE FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
