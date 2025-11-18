import json
import logging
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from django.conf import settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton service for managing embeddings.

    This service loads the embedding model once and provides methods
    for embedding text and computing similarities.
    """

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @lru_cache(maxsize=1000)
    def _embed_cached(self, text: str, query_id: Optional[str] = None) -> tuple:
        """
        Generate sentence embedding with caching.

        Caches by both text and query_id. If query_id is provided, it becomes
        part of the cache key, allowing the same text with different query_ids
        to be cached separately.

        Args:
            text: Input text to embed
            query_id: Optional query identifier for cache key

        Returns:
            tuple of embedding values
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            # tuples are hashable, numpy arrays are not
            return tuple(embedding.tolist())
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_text(self, text: str, query_id: Optional[str] = None) -> np.ndarray:
        """
        Generate sentence embedding for a single text.

        Args:
            text: Input text to embed
            query_id: Optional query identifier for caching by query_id

        Returns:
            numpy array of shape (embedding_dim,)
        """
        # get cached tuple and convert back to numpy array
        embedding_tuple = self._embed_cached(text, query_id)
        return np.array(embedding_tuple)

    def embed_query(self, query_text: str, query_id: str) -> np.ndarray:
        """
        Generate embedding for a query with query_id-based caching.

        This method is specifically for queries where you want to cache
        by query_id rather than just text content.

        Args:
            query_text: Query text to embed
            query_id: Query identifier (becomes part of cache key)

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed_text(query_text, query_id=query_id)

    def embed_batch(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Generate sentence embeddings for a batch of texts.

        Notes:
            - Automatically truncates texts longer than `max_length` tokens to avoid model context overflow.
            - Batch embedding bypasses cache for efficiency.
            - Use embed_text() or embed_query() for cached individual embeddings.

        Args:
            texts: List of texts to embed.
            max_length: Maximum allowed number of tokens (depends on model).

        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Text list cannot be empty")

        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")

        try:
            tokenizer = self._model.tokenizer
        except AttributeError:
            raise RuntimeError("The model does not expose a tokenizer for token-level truncation.")

        truncated_texts = []
        num_truncated = 0

        for text in valid_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_length:
                num_truncated += 1
                tokens = tokens[:max_length]
                truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                truncated_text = text
            truncated_texts.append(truncated_text)

        if num_truncated:
            logger.warning(f"Truncated {num_truncated} texts exceeding {max_length} tokens.")

        try:
            embeddings = self._model.encode(
                truncated_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32,
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed batch of {len(valid_texts)} texts: {e}")
            raise

    def compute_cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            doc_embeddings: Document embeddings of shape (num_docs, embedding_dim)

        Returns:
            Similarity scores of shape (num_docs,)
        """
        # Reshape query embedding to (1, embedding_dim)
        query_embedding = query_embedding.reshape(1, -1)  # 384

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)

        # Return as 1D array
        return similarities[0]

    def get_cache_info(self) -> Dict:
        """Get cache statistics from lru_cache."""
        cache_info = self._embed_cached.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "size": cache_info.currsize,
            "maxsize": cache_info.maxsize,
            "hit_rate": (
                cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
            ),
        }

    def clear_cache(self):
        """Clear the LRU cache."""
        self._embed_cached.cache_clear()
        logger.info("Embedding cache cleared")

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> str:
        """
        Serialize numpy array to JSON string for database storage.

        Args:
            embedding: Numpy array

        Returns:
            JSON string representation
        """
        return json.dumps(embedding.tolist())

    @staticmethod
    def deserialize_embedding(embedding_str: str) -> np.ndarray:
        """
        Deserialize JSON string back to numpy array.

        Args:
            embedding_str: JSON string

        Returns:
            Numpy array
        """
        return np.array(json.loads(embedding_str))
