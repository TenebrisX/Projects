import logging
from typing import Dict, List, Tuple

import numpy as np

from .embedding_service import EmbeddingService
from .models import Document, QueryRelevance

logger = logging.getLogger(__name__)


class SearchService:
    """Service for performing search operations and evaluation."""

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def search(self, query_text: str, query_id: str = None, top_k: int = 10) -> Tuple[List[str], Dict[str, float]]:
        """
        Search for documents similar to the query.

        Args:
            query_text: The query text to search for
            query_id: Optional query ID for caching and evaluation
            top_k: Number of top documents to return

        Returns:
            Tuple of (top_doc_ids, metadata)
        """
        if query_id:
            query_embedding = self.embedding_service.embed_query(query_text, query_id)
        else:
            query_embedding = self.embedding_service.embed_text(query_text)

        documents = Document.objects.all()  # cuz the amount of documents is small

        if not documents.exists():
            logger.warning("No documents found in database")
            return [], {}

        doc_ids = []
        doc_embeddings = []

        for doc in documents:
            doc_ids.append(doc.doc_id)
            embedding = self.embedding_service.deserialize_embedding(doc.embedding)
            doc_embeddings.append(embedding)

        doc_embeddings = np.array(doc_embeddings)

        similarities = self.embedding_service.compute_cosine_similarity(query_embedding, doc_embeddings)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_doc_ids = [doc_ids[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]

        metadata = {
            "scores": dict(zip(top_doc_ids, top_scores)),
            "total_documents": len(doc_ids),
        }

        return top_doc_ids, metadata

    def calculate_precision_at_k(self, query_id: str, retrieved_docs: List[str], k: int = 5) -> float:
        """Calculate Precision@K for a query."""
        if not query_id:
            logger.warning("No query_id provided for P@K calculation")
            return 0.0

        relevant_docs = set(QueryRelevance.objects.filter(query_id=query_id).values_list("doc_id", flat=True))

        if not relevant_docs:
            logger.warning(f"No relevant documents found for query_id: {query_id}")
            return 0.0

        top_k_docs = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)

        return relevant_retrieved / k if k > 0 else 0.0

    def search_and_evaluate(self, query_text: str, query_id: str, top_k: int = 10) -> Dict:
        """Perform search and calculate evaluation metrics."""
        top_docs, metadata = self.search(query_text, query_id, top_k)
        p5 = self.calculate_precision_at_k(query_id, top_docs, k=5)

        return {
            "top_docs": top_docs,
            "p5": round(p5, 3),
            "scores": metadata.get("scores", {}),
            "total_documents": metadata.get("total_documents", 0),
        }

    def get_relevant_docs(self, query_id: str) -> List[str]:
        """Get all relevant documents for a query from qrels."""
        return list(QueryRelevance.objects.filter(query_id=query_id).values_list("doc_id", flat=True))

    def get_cache_statistics(self) -> Dict:
        """Get embedding cache statistics."""
        return self.embedding_service.get_cache_info()

    def clear_embedding_cache(self):
        """Clear the embedding cache."""
        self.embedding_service.clear_cache()
        logger.info("Embedding cache cleared")
