"""
Embedding Service for Government Orders System.

Generates dense vector embeddings using FastEmbed and ONNX Runtime.
Uses the lightweight BAAI/bge-small-en-v1.5 model (384 dimensions)
for fast, memory-efficient CPU inference on local hardware.
"""

import logging
import threading
from typing import List, Optional
from fastembed import TextEmbedding

from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

# Official BGE search query instruction prefix
BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    """
    Singleton-wrapped Embedding Service using FastEmbed.
    Provides fast, PyTorch-free embeddings running via ONNX Runtime.
    """

    _instance: Optional["EmbeddingService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        logger.info("Initializing EmbeddingService with model: %s", self.model_name)
        try:
            self.model = TextEmbedding(model_name=self.model_name)
            self._initialized = True
            logger.info("EmbeddingService initialized successfully (expected dimension: %d).", self.dimension)
        except Exception as exc:
            logger.error("Failed to load FastEmbed model '%s': %s", self.model_name, exc, exc_info=True)
            raise RuntimeError(f"Could not initialize EmbeddingService: {exc}") from exc

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string (document passage or raw text).
        Does NOT apply query search prefixes.

        Args:
            text: Input string.

        Returns:
            List of floats representing the embedding vector.
        """
        if not text or not text.strip():
            # Return zero vector if text is empty
            return [0.0] * self.dimension

        embeddings = list(self.model.embed([text]))
        vector = embeddings[0].tolist()
        if len(vector) != self.dimension:
            raise ValueError(f"Generated embedding dimension {len(vector)} does not match expected {self.dimension}")
        return vector

    def embed_query(self, query_text: str) -> List[float]:
        """
        Generate embedding for a search query using the recommended BGE instruction prefix.
        Document passages must NOT use this method.

        Args:
            query_text: Natural-language search query string.

        Returns:
            List of floats representing the query embedding vector (384 dimensions).

        Raises:
            ValueError: If query_text is empty, None, or whitespace-only.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Search query cannot be empty or whitespace.")

        clean_query = query_text.strip()
        if clean_query.startswith(BGE_QUERY_PREFIX):
            prefixed = clean_query
        else:
            prefixed = f"{BGE_QUERY_PREFIX}{clean_query}"

        return self.embed_text(prefixed)

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.

        Args:
            texts: List of input strings.
            batch_size: Batch size for FastEmbed inference.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Replace empty strings with a single space to avoid tokenizer errors
        cleaned = [t if t and t.strip() else " " for t in texts]
        embeddings_iter = self.model.embed(cleaned, batch_size=batch_size)
        results = [emb.tolist() for emb in embeddings_iter]

        for vec in results:
            if len(vec) != self.dimension:
                raise ValueError(f"Generated embedding dimension {len(vec)} does not match expected {self.dimension}")

        return results

    def get_dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        return self.dimension
