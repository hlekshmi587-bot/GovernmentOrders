"""
Safe Qdrant Vector Indexing Script for Kerala Government Orders.

Performs read-only extraction of document texts from PostgreSQL,
generates semantic chunks respecting paragraph and clause boundaries,
computes dense vector embeddings via FastEmbed (ONNX Runtime),
and safely indexes vectors into local persistent Qdrant storage.
"""

import logging
import os
import re
import sys
import time
from typing import Any, Dict, List
import psutil

from app.database.connection import SessionLocal
from app.database.models import GovernmentOrderDB
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.config import QDRANT_STORAGE_PATH, QDRANT_COLLECTION_NAME
from app.services.chunking_service import chunk_document_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("index_qdrant")


def run_indexing() -> Dict[str, Any]:
    """
    Main indexing runner.
    Reads 11 orders from PostgreSQL and indexes them into Qdrant.
    """
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    logger.info("Initializing EmbeddingService and QdrantService...")
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService(
        storage_path=QDRANT_STORAGE_PATH,
        collection_name=QDRANT_COLLECTION_NAME,
        dimension=embedding_service.get_dimension(),
    )

    logger.info("Connecting to PostgreSQL to fetch Government Orders (read-only)...")
    orders_data = []
    with SessionLocal() as session:
        orders = session.query(GovernmentOrderDB).order_by(GovernmentOrderDB.id).all()
        for o in orders:
            orders_data.append(
                {
                    "id": o.id,
                    "go_number": o.go_number,
                    "department": o.department,
                    "source_filename": o.source_filename,
                    "issue_date": o.issue_date.isoformat() if o.issue_date else None,
                    "text": o.extracted_text or "",
                }
            )

    logger.info("Fetched %d orders from PostgreSQL for indexing.", len(orders_data))

    total_chunks = 0
    results_summary = []
    unchanged_count = 0
    indexed_count = 0

    for item in orders_data:
        go_id = item["id"]
        go_number = item["go_number"]
        chunks = chunk_document_text(item["text"])
        total_chunks += len(chunks)

        logger.info(
            "Indexing GO '%s' (id=%d) - %d chunks generated.",
            go_number,
            go_id,
            len(chunks),
        )

        res = qdrant_service.safe_index_document(
            go_id=go_id,
            go_number=go_number,
            department=item["department"],
            source_filename=item["source_filename"],
            issue_date=item["issue_date"],
            chunks=chunks,
            embedding_service=embedding_service,
        )

        results_summary.append(res)
        if res["status"] == "UNCHANGED":
            unchanged_count += 1
        elif res["status"] == "SUCCESS":
            indexed_count += 1

    elapsed_sec = time.time() - start_time
    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    active_points = qdrant_service.count_active_points()
    total_points = qdrant_service.count_total_points()

    stats = {
        "orders_processed": len(orders_data),
        "indexed_orders": indexed_count,
        "unchanged_orders": unchanged_count,
        "total_chunks": total_chunks,
        "active_points_in_qdrant": active_points,
        "total_points_in_qdrant": total_points,
        "embedding_dimension": embedding_service.get_dimension(),
        "elapsed_seconds": round(elapsed_sec, 2),
        "memory_used_mb": round(mem_after_mb - mem_before_mb, 2),
        "peak_rss_mb": round(mem_after_mb, 2),
        "storage_path": str(QDRANT_STORAGE_PATH),
    }

    logger.info("=== Indexing Completed Successfully ===")
    logger.info("Stats: %s", stats)

    return stats


if __name__ == "__main__":
    try:
        run_indexing()
    except Exception as exc:
        logger.error("Indexing failed: %s", exc, exc_info=True)
        sys.exit(1)
