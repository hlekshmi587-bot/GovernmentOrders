"""
Qdrant Local Persistent Vector Storage Service.

Manages collection lifecycle, deterministic point indexing, versioned safe replacements,
and similarity searches using local persistent file storage on disk.
Guards against concurrent multi-client locks on the same storage directory.
"""

import hashlib
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import (
    QDRANT_STORAGE_PATH,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)

# Deterministic namespace for reproducible point UUIDs
POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace OID


class QdrantService:
    """
    Local persistent vector storage service using Qdrant.
    Manages collection creation, versioned document indexing, and semantic search.
    Enforces a singleton client per storage path to prevent RocksDB lock contention.
    """

    _instances: Dict[str, "QdrantService"] = {}
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        storage_path: Optional[Path] = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> "QdrantService":
        path_str = str((storage_path or QDRANT_STORAGE_PATH).resolve())
        with cls._lock:
            if path_str not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[path_str] = instance
            return cls._instances[path_str]

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self.storage_path = (storage_path or QDRANT_STORAGE_PATH).resolve()
        self.collection_name = collection_name
        self.dimension = dimension

        # Ensure parent directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing QdrantService with local storage at: %s", self.storage_path)
        try:
            self.client = QdrantClient(path=str(self.storage_path))
            self._ensure_collection()
            # Startup recovery: reconcile versions to prune incomplete or orphaned states
            self.reconcile_versions()
            self._initialized = True
            logger.info("QdrantService successfully initialized on collection '%s'.", self.collection_name)
        except Exception as exc:
            logger.error("Failed to initialize QdrantClient at '%s': %s", self.storage_path, exc, exc_info=True)
            raise RuntimeError(f"Could not initialize local Qdrant at {self.storage_path}: {exc}") from exc

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if self.collection_name not in existing_names:
                logger.info("Creating Qdrant collection '%s' with dimension %d...", self.collection_name, self.dimension)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
                logger.info("Collection '%s' created successfully.", self.collection_name)
            else:
                logger.debug("Collection '%s' already exists.", self.collection_name)
        except Exception as exc:
            logger.error("Error ensuring collection '%s': %s", self.collection_name, exc, exc_info=True)
            raise

    @staticmethod
    def generate_point_id(go_id: int, version: int, chunk_index: int) -> str:
        """
        Generate a deterministic UUID for a vector point based on go_id, version, and chunk_index.
        Ensures idempotency and exact traceability.
        """
        ident_str = f"go_{go_id}_v{version}_chunk_{chunk_index}"
        return str(uuid.uuid5(POINT_NAMESPACE, ident_str))

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute SHA-256 hash of chunk text to detect content changes."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def validate_version_completeness(self, points: List[Any]) -> bool:
        """
        Check if a list of points represents a fully complete indexed version.
        A version is complete if and only if:
        1. points list is not empty.
        2. Every chunk index from 1 to expected_total is present with no duplicates or gaps.
        """
        if not points:
            return False

        chunk_indices = set()
        total_chunks_values = set()

        for p in points:
            payload = p.payload or {}
            c_idx = payload.get("chunk_index")
            if c_idx is not None:
                chunk_indices.add(c_idx)
            t_chunks = payload.get("total_chunks")
            if t_chunks is not None:
                total_chunks_values.add(t_chunks)

        if not chunk_indices:
            return False

        if len(total_chunks_values) == 1:
            expected_total = total_chunks_values.pop()
        else:
            # Fallback if total_chunks was omitted in earlier indexing
            expected_total = max(chunk_indices)

        expected_set = set(range(1, expected_total + 1))
        return (len(points) == expected_total) and (chunk_indices == expected_set)

    def reconcile_versions(self, target_go_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Startup and recovery reconciliation:
        1. Scans points (for target_go_id, or all points in the collection).
        2. Groups points by go_id and version.
        3. Validates completeness of every version.
        4. Prunes incomplete staged/partial versions (whether active or inactive).
        5. If multiple complete versions exist (e.g. process died after activation before cleanup),
           activates the highest complete version and prunes older complete versions.
        6. Preserves the last known complete version if a newer version is incomplete.
        7. Fully idempotent.
        """
        scroll_filter = None
        if target_go_id is not None:
            scroll_filter = Filter(must=[FieldCondition(key="go_id", match=MatchValue(value=target_go_id))])

        all_points: List[Any] = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=200,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            return {
                "status": "SUCCESS",
                "reconciled_orders": 0,
                "pruned_incomplete_versions": 0,
                "pruned_obsolete_versions": 0,
            }

        grouped: Dict[int, Dict[int, List[Any]]] = {}
        for p in all_points:
            payload = p.payload or {}
            gid = payload.get("go_id")
            ver = payload.get("version")
            if gid is None or ver is None:
                continue
            grouped.setdefault(gid, {}).setdefault(ver, []).append(p)

        pruned_incomplete = 0
        pruned_obsolete = 0

        for gid, versions_map in grouped.items():
            complete_versions: Dict[int, List[Any]] = {}
            incomplete_versions: Dict[int, List[Any]] = {}

            for ver, pts in versions_map.items():
                if self.validate_version_completeness(pts):
                    complete_versions[ver] = pts
                else:
                    incomplete_versions[ver] = pts

            # 1. Prune incomplete versions (crashes during upload or partial activation)
            for inc_ver, inc_pts in incomplete_versions.items():
                logger.warning(
                    "Reconciliation: Pruning incomplete version %d (%d chunks) for go_id=%d.",
                    inc_ver, len(inc_pts), gid
                )
                del_filter = Filter(
                    must=[
                        FieldCondition(key="go_id", match=MatchValue(value=gid)),
                        FieldCondition(key="version", match=MatchValue(value=inc_ver)),
                    ]
                )
                self.client.delete(collection_name=self.collection_name, points_selector=del_filter)
                pruned_incomplete += 1

            # 2. Resolve complete versions: keep the highest, prune older ones
            if complete_versions:
                highest_complete_ver = max(complete_versions.keys())
                highest_pts = complete_versions[highest_complete_ver]

                # Ensure all points of highest version are active
                inactive_point_ids = [
                    p.id for p in highest_pts if not (p.payload or {}).get("is_active")
                ]
                if inactive_point_ids:
                    logger.info(
                        "Reconciliation: Activating complete version %d for go_id=%d (%d points).",
                        highest_complete_ver, gid, len(inactive_point_ids)
                    )
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"is_active": True},
                        points=inactive_point_ids,
                    )

                # Prune older complete versions
                for old_ver in complete_versions.keys():
                    if old_ver < highest_complete_ver:
                        logger.info(
                            "Reconciliation: Pruning obsolete version %d for go_id=%d in favor of version %d.",
                            old_ver, gid, highest_complete_ver
                        )
                        del_filter = Filter(
                            must=[
                                FieldCondition(key="go_id", match=MatchValue(value=gid)),
                                FieldCondition(key="version", match=MatchValue(value=old_ver)),
                            ]
                        )
                        self.client.delete(collection_name=self.collection_name, points_selector=del_filter)
                        pruned_obsolete += 1

        return {
            "status": "SUCCESS",
            "reconciled_orders": len(grouped),
            "pruned_incomplete_versions": pruned_incomplete,
            "pruned_obsolete_versions": pruned_obsolete,
        }

    def get_order_active_version(self, go_id: int) -> Optional[int]:
        """
        Find the current active version number for a given Government Order.
        Verifies that the version is genuinely complete and active.
        """
        filter_condition = Filter(
            must=[
                FieldCondition(key="go_id", match=MatchValue(value=go_id)),
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]
        )
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None

        # Group active points by version
        versions_map: Dict[int, List[Any]] = {}
        for r in records:
            ver = (r.payload or {}).get("version")
            if ver is not None:
                versions_map.setdefault(ver, []).append(r)

        # Filter only complete versions
        complete_active = [
            ver for ver, pts in versions_map.items()
            if self.validate_version_completeness(pts)
        ]
        if complete_active:
            return max(complete_active)

        return None

    def get_active_points_for_order(self, go_id: int) -> List[Any]:
        """
        Retrieve all active points currently stored for a specific Government Order.
        """
        filter_condition = Filter(
            must=[
                FieldCondition(key="go_id", match=MatchValue(value=go_id)),
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]
        )
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        return records

    def safe_index_document(
        self,
        go_id: int,
        go_number: str,
        department: Optional[str],
        source_filename: str,
        issue_date: Optional[str],
        chunks: List[Dict[str, Any]],
        embedding_service: Any,
    ) -> Dict[str, Any]:
        """
        Safely index or re-index a Government Order with atomic-style versioning:
        1. Determine next version (e.g. current + 1).
        2. Generate embeddings and create PointStruct objects with `is_active=False`.
        3. Upload new points under the new version.
        4. Verify upload integrity by scrolling the uploaded points.
        5. Mark new points as `is_active=True`.
        6. Delete old version points only after successful activation.
        7. If any step fails, the previous active points remain intact and active!

        Args:
            go_id: PostgreSQL primary key.
            go_number: Canonical GO number.
            department: Department name.
            source_filename: Source PDF or text filename.
            issue_date: ISO date string.
            chunks: List of chunk dicts containing 'chunk_index', 'chunk_text', and optional 'page_number'.
            embedding_service: EmbeddingService instance.

        Returns:
            Dictionary with indexing status, version, chunk count, and point IDs.
        """
        if not chunks:
            return {
                "status": "SKIPPED",
                "go_id": go_id,
                "go_number": go_number,
                "reason": "No text chunks provided",
                "chunks_indexed": 0,
            }

        current_version = self.get_order_active_version(go_id)
        new_version = (current_version + 1) if current_version is not None else 1

        logger.info(
            "Preparing safe indexing for GO '%s' (id=%d) with new version %d (current: %s)...",
            go_number,
            go_id,
            new_version,
            current_version,
        )

        # Check if identical chunks are already indexed under current_version (Duplicate prevention)
        if current_version is not None:
            existing_points = self.get_active_points_for_order(go_id)
            if len(existing_points) == len(chunks):
                existing_hashes = {p.payload.get("chunk_index"): p.payload.get("content_hash") for p in existing_points if p.payload}
                all_match = True
                for c in chunks:
                    c_idx = c["chunk_index"]
                    c_hash = self.compute_content_hash(c["chunk_text"])
                    if existing_hashes.get(c_idx) != c_hash:
                        all_match = False
                        break
                if all_match:
                    logger.info("GO '%s' (id=%d) content is unchanged. Skipping duplicate indexing.", go_number, go_id)
                    return {
                        "status": "UNCHANGED",
                        "go_id": go_id,
                        "go_number": go_number,
                        "version": current_version,
                        "chunks_indexed": len(chunks),
                        "reason": "Content matches active version hash",
                    }

        # Step 1: Embed chunks
        texts_to_embed = [c["chunk_text"] for c in chunks]
        vectors = embedding_service.embed_batch(texts_to_embed)

        # Step 2: Prepare new points with `is_active=False` (staged)
        staged_points: List[PointStruct] = []
        new_point_ids: List[str] = []

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            c_index = chunk["chunk_index"]
            c_text = chunk["chunk_text"]
            point_id = self.generate_point_id(go_id=go_id, version=new_version, chunk_index=c_index)
            new_point_ids.append(point_id)
            content_hash = self.compute_content_hash(c_text)

            payload = {
                "go_id": go_id,
                "go_number": go_number,
                "department": department,
                "source_filename": source_filename,
                "issue_date": issue_date,
                "chunk_index": c_index,
                "total_chunks": len(chunks),
                "chunk_text": c_text,
                "page_number": chunk.get("page_number"),
                "content_hash": content_hash,
                "version": new_version,
                "is_active": False,  # Staged: not yet visible to searches
            }

            staged_points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        try:
            # Step 3: Upload staged points
            self.client.upsert(
                collection_name=self.collection_name,
                points=staged_points,
            )

            # Step 4: Verify uploaded points exist and form a complete version
            uploaded_records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=new_point_ids,
                with_payload=True,
            )
            if len(uploaded_records) != len(new_point_ids):
                raise RuntimeError(
                    f"Verification failed: Uploaded {len(new_point_ids)} points but retrieved only {len(uploaded_records)}."
                )
            if not self.validate_version_completeness(uploaded_records):
                raise RuntimeError(
                    f"Verification failed: Uploaded points for GO '{go_number}' version {new_version} do not form a complete version sequence."
                )

            # Step 5: Activate new points
            activated_points: List[PointStruct] = []
            for p in staged_points:
                p.payload["is_active"] = True
                activated_points.append(p)

            self.client.upsert(
                collection_name=self.collection_name,
                points=activated_points,
            )

            # Step 6: Now that new version is fully active, delete old version points
            if current_version is not None:
                old_filter = Filter(
                    must=[
                        FieldCondition(key="go_id", match=MatchValue(value=go_id)),
                        FieldCondition(key="version", match=MatchValue(value=current_version)),
                    ]
                )
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=old_filter,
                )
                logger.info("Pruned old version %d vectors for GO '%s'.", current_version, go_number)

            logger.info("Successfully indexed GO '%s' (id=%d) under active version %d (%d chunks).", go_number, go_id, new_version, len(chunks))

            return {
                "status": "SUCCESS",
                "go_id": go_id,
                "go_number": go_number,
                "version": new_version,
                "chunks_indexed": len(chunks),
                "point_ids": new_point_ids,
            }

        except Exception as exc:
            # Step 7: Rollback / Cleanup staged points so previous active version remains working
            logger.error("Indexing failed for GO '%s' (id=%d). Preserving previous active version: %s", go_number, go_id, exc, exc_info=True)
            try:
                staged_filter = Filter(
                    must=[
                        FieldCondition(key="go_id", match=MatchValue(value=go_id)),
                        FieldCondition(key="version", match=MatchValue(value=new_version)),
                    ]
                )
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=staged_filter,
                )
            except Exception as cleanup_err:
                logger.warning("Failed to cleanup staged points for failed indexing: %s", cleanup_err)

            raise RuntimeError(f"Safe indexing failed for GO {go_number}: {exc}") from exc

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        department_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar text chunks using cosine similarity.
        Only returns active points (`is_active=True`).

        Args:
            query_vector: Dense embedding vector for query.
            limit: Maximum points to return.
            department_filter: Optional department name filter.
            min_score: Minimum cosine similarity threshold.

        Returns:
            List of matching records with payload and score.
        """
        must_conditions: List[FieldCondition] = [
            FieldCondition(key="is_active", match=MatchValue(value=True))
        ]
        if department_filter:
            must_conditions.append(
                FieldCondition(key="department", match=MatchValue(value=department_filter))
            )

        query_filter = Filter(must=must_conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

        formatted = []
        for r in results:
            if r.score < min_score:
                continue
            payload = r.payload or {}
            formatted.append(
                {
                    "point_id": str(r.id),
                    "score": float(r.score),
                    "go_id": payload.get("go_id"),
                    "go_number": payload.get("go_number"),
                    "department": payload.get("department"),
                    "source_filename": payload.get("source_filename"),
                    "issue_date": payload.get("issue_date"),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_text": payload.get("chunk_text"),
                    "page_number": payload.get("page_number"),
                    "version": payload.get("version"),
                }
            )

        return formatted

    def count_active_points(self) -> int:
        """Count total active points in the collection."""
        active_filter = Filter(
            must=[FieldCondition(key="is_active", match=MatchValue(value=True))]
        )
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=active_filter,
            exact=True,
        ).count

    def count_total_points(self) -> int:
        """Count all points in the collection regardless of status."""
        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    def close(self) -> None:
        """Release client locks and close connection."""
        try:
            self.client.close()
            path_str = str(self.storage_path)
            with self._lock:
                if path_str in self._instances:
                    del self._instances[path_str]
            logger.info("Closed QdrantClient for '%s'.", self.storage_path)
        except Exception as exc:
            logger.warning("Error closing QdrantClient: %s", exc)
