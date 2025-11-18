"""
Unit tests for the search engine API.

Tests cover:
- Embedding service functionality
- Search service functionality
- API endpoints
- Model operations
"""

import numpy as np
from django.test import TestCase, TransactionTestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from search_api.embedding_service import EmbeddingService
from search_api.models import Document, Query, QueryRelevance
from search_api.search_service import SearchService


class EmbeddingServiceTests(TestCase):
    """Tests for the EmbeddingService."""

    def setUp(self):
        self.service = EmbeddingService()
        # Clear cache before each test
        self.service.clear_cache()

    def test_embed_text_returns_numpy_array(self):
        """Test that embedding returns a numpy array."""
        text = "cardiovascular disease"
        embedding = self.service.embed_text(text)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding.shape), 1)  # 1D array
        self.assertGreater(embedding.shape[0], 0)  # Has dimensions

    def test_embed_empty_text_raises_error(self):
        """Test that embedding empty text raises ValueError."""
        with self.assertRaises(ValueError):
            self.service.embed_text("")

    def test_embed_whitespace_only_raises_error(self):
        """Test that embedding whitespace-only text raises ValueError."""
        with self.assertRaises(ValueError):
            self.service.embed_text("   ")

    def test_embed_batch(self):
        """Test batch embedding."""
        texts = ["heart disease", "diabetes", "cancer"]
        embeddings = self.service.embed_batch(texts)

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(texts))

    def test_embed_batch_filters_empty_texts(self):
        """Test that batch embedding filters out empty texts."""
        texts = ["heart disease", "", "diabetes", "   ", "cancer"]
        embeddings = self.service.embed_batch(texts)

        # Should only embed 3 valid texts
        self.assertEqual(embeddings.shape[0], 3)

    def test_embed_batch_all_empty_raises_error(self):
        """Test that batch embedding all empty texts raises ValueError."""
        texts = ["", "   ", ""]
        with self.assertRaises(ValueError):
            self.service.embed_batch(texts)

    def test_serialize_deserialize_embedding(self):
        """Test serialization and deserialization of embeddings."""
        original = np.array([0.1, 0.2, 0.3, 0.4])
        serialized = EmbeddingService.serialize_embedding(original)
        deserialized = EmbeddingService.deserialize_embedding(serialized)

        np.testing.assert_array_almost_equal(original, deserialized)

    def test_embed_query_caching(self):
        """Test that embed_query caches by query_id."""
        # First call - cache miss
        emb1 = self.service.embed_query("heart disease", "Q1")
        stats1 = self.service.get_cache_info()
        self.assertEqual(stats1["misses"], 1)
        self.assertEqual(stats1["hits"], 0)

        # Second call with same query_id and text - cache hit
        emb2 = self.service.embed_query("heart disease", "Q1")
        stats2 = self.service.get_cache_info()
        self.assertEqual(stats2["hits"], 1)

        # Embeddings should be identical
        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_query_different_query_ids(self):
        """Test that different query_ids create separate cache entries."""
        # Same text, different query_ids
        emb1 = self.service.embed_query("heart disease", "Q1")
        emb2 = self.service.embed_query("heart disease", "Q2")

        stats = self.service.get_cache_info()
        self.assertEqual(stats["misses"], 2)  # Both were misses
        self.assertEqual(stats["size"], 2)  # Two cache entries

        # Embeddings should be identical (same text)
        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_text_caching(self):
        """Test that embed_text caches by text content."""
        # First call - cache miss
        emb1 = self.service.embed_text("diabetes")
        stats1 = self.service.get_cache_info()
        self.assertEqual(stats1["misses"], 1)

        # Second call with same text - cache hit
        emb2 = self.service.embed_text("diabetes")
        stats2 = self.service.get_cache_info()
        self.assertEqual(stats2["hits"], 1)

        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        self.service.clear_cache()

        # Generate some cache activity
        self.service.embed_query("heart disease", "Q1")  # Miss
        self.service.embed_query("heart disease", "Q1")  # Hit
        self.service.embed_query("diabetes", "Q2")  # Miss

        stats = self.service.get_cache_info()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 2)
        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["maxsize"], 1000)
        self.assertAlmostEqual(stats["hit_rate"], 1 / 3, places=2)

    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add some entries
        self.service.embed_query("heart disease", "Q1")
        self.service.embed_query("diabetes", "Q2")

        stats_before = self.service.get_cache_info()
        self.assertEqual(stats_before["size"], 2)

        # Clear cache
        self.service.clear_cache()

        stats_after = self.service.get_cache_info()
        self.assertEqual(stats_after["size"], 0)
        self.assertEqual(stats_after["hits"], 0)
        self.assertEqual(stats_after["misses"], 0)


class ModelTests(TransactionTestCase):
    """Tests for Django models."""

    def test_create_document(self):
        """Test creating a Document."""
        embedding = np.array([0.1, 0.2, 0.3])
        embedding_str = EmbeddingService.serialize_embedding(embedding)

        doc = Document.objects.create(doc_id="MED-1", text="This is a test document", embedding=embedding_str)

        self.assertEqual(doc.doc_id, "MED-1")
        self.assertEqual(doc.text, "This is a test document")

    def test_create_query_relevance(self):
        """Test creating QueryRelevance."""
        qrel = QueryRelevance.objects.create(query_id="PLAIN-1", doc_id="MED-1", relevance_score=2)

        self.assertEqual(qrel.query_id, "PLAIN-1")
        self.assertEqual(qrel.doc_id, "MED-1")
        self.assertEqual(qrel.relevance_score, 2)

    def test_create_query(self):
        """Test creating Query."""
        query = Query.objects.create(query_id="PLAIN-1", query_text="test query")

        self.assertEqual(query.query_id, "PLAIN-1")
        self.assertEqual(query.query_text, "test query")


class SearchServiceTests(TransactionTestCase):
    """Tests for the SearchService."""

    def setUp(self):
        """Set up test data."""
        self.service = SearchService()
        self.embedding_service = EmbeddingService()

        self.embedding_service.clear_cache()

        texts = [
            "cardiovascular disease and heart health",
            "diabetes management and treatment",
            "cancer research and therapy",
        ]

        for i, text in enumerate(texts):
            embedding = self.embedding_service.embed_text(text)
            embedding_str = self.embedding_service.serialize_embedding(embedding)

            Document.objects.create(doc_id=f"MED-{i+1}", text=text, embedding=embedding_str)

    def test_search_returns_results(self):
        """Test that search returns results."""
        top_docs, metadata = self.service.search(query_text="heart disease", top_k=3)

        self.assertEqual(len(top_docs), 3)
        self.assertIn("scores", metadata)
        self.assertIn("total_documents", metadata)

    def test_search_with_query_id(self):
        """Test search with query_id for caching."""
        # First search
        top_docs1, metadata1 = self.service.search(query_text="heart disease", query_id="Q1", top_k=3)

        # Second search with same query_id (should use cache)
        top_docs2, metadata2 = self.service.search(query_text="heart disease", query_id="Q1", top_k=3)

        # Results should be identical
        self.assertEqual(top_docs1, top_docs2)

        # Check cache was used
        cache_stats = self.embedding_service.get_cache_info()
        self.assertGreater(cache_stats["hits"], 0)

    def test_search_ranking(self):
        """Test that search returns documents in order of similarity."""
        top_docs, metadata = self.service.search(query_text="cardiovascular", top_k=3)

        # First document should be most similar
        scores = metadata["scores"]
        self.assertGreater(scores[top_docs[0]], scores[top_docs[1]])

    def test_calculate_precision_at_k(self):
        """Test P@K calculation."""
        # Create qrels
        QueryRelevance.objects.create(query_id="TEST-1", doc_id="MED-1", relevance_score=2)
        QueryRelevance.objects.create(query_id="TEST-1", doc_id="MED-2", relevance_score=1)

        # Retrieved docs: 2 relevant in top 5
        retrieved = ["MED-1", "MED-3", "MED-2", "MED-4", "MED-5"]
        p5 = self.service.calculate_precision_at_k("TEST-1", retrieved, k=5)

        # 2 relevant out of 5 = 0.4
        self.assertAlmostEqual(p5, 0.4, places=2)

    def test_precision_at_k_no_relevant(self):
        """Test P@K when there are no relevant documents."""
        retrieved = ["MED-1", "MED-2", "MED-3"]
        p5 = self.service.calculate_precision_at_k("NONEXISTENT", retrieved, k=5)

        self.assertEqual(p5, 0.0)

    def test_search_and_evaluate(self):
        """Test search_and_evaluate method."""
        # Create qrels
        QueryRelevance.objects.create(query_id="TEST-1", doc_id="MED-1", relevance_score=2)

        result = self.service.search_and_evaluate(query_text="heart disease", query_id="TEST-1", top_k=3)

        self.assertIn("top_docs", result)
        self.assertIn("p5", result)
        self.assertIn("scores", result)
        self.assertEqual(len(result["top_docs"]), 3)
        self.assertIsInstance(result["p5"], float)

    def test_get_cache_statistics(self):
        """Test getting cache statistics from SearchService."""
        # Generate some searches
        self.service.search("heart disease", query_id="Q1", top_k=5)
        self.service.search("heart disease", query_id="Q1", top_k=5)

        stats = self.service.get_cache_statistics()
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertGreater(stats["hits"], 0)

    def test_clear_embedding_cache(self):
        """Test clearing cache via SearchService."""
        # Add cache entry
        self.service.search("diabetes", query_id="Q1", top_k=5)

        stats_before = self.service.get_cache_statistics()
        self.assertGreater(stats_before["size"], 0)

        # Clear cache
        self.service.clear_embedding_cache()

        stats_after = self.service.get_cache_statistics()
        self.assertEqual(stats_after["size"], 0)


class StatusAPITests(APITestCase):
    """Tests for the /status/ endpoint."""

    def setUp(self):
        """Set up test data."""
        embedding_service = EmbeddingService()
        embedding_service.clear_cache()

        # Create test documents
        for i in range(5):
            embedding = embedding_service.embed_text(f"test doc {i}")
            embedding_str = embedding_service.serialize_embedding(embedding)

            Document.objects.create(doc_id=f"MED-{i+1}", text=f"test document {i}", embedding=embedding_str)

        # Create test qrels
        for i in range(3):
            QueryRelevance.objects.create(query_id=f"PLAIN-{i+1}", doc_id=f"MED-{i+1}", relevance_score=2)

    def test_status_endpoint(self):
        """Test the /status/ endpoint."""
        url = reverse("search_api:status")
        response = self.client.post(url, {}, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("num_of_indexed_items", response.data)
        self.assertIn("num_of_queries_in_qrels", response.data)
        self.assertEqual(response.data["num_of_indexed_items"], 5)
        self.assertEqual(response.data["num_of_queries_in_qrels"], 3)


class QueryAPITests(APITestCase):
    """Tests for the /query/ endpoint."""

    def setUp(self):
        """Set up test data."""
        embedding_service = EmbeddingService()
        embedding_service.clear_cache()

        # Create test documents
        texts = ["cardiovascular disease", "diabetes treatment", "cancer therapy"]

        for i, text in enumerate(texts):
            embedding = embedding_service.embed_text(text)
            embedding_str = embedding_service.serialize_embedding(embedding)

            Document.objects.create(doc_id=f"MED-{i+1}", text=text, embedding=embedding_str)

        # Create qrels
        QueryRelevance.objects.create(query_id="PLAIN-1", doc_id="MED-1", relevance_score=2)

    def test_query_endpoint_success(self):
        """Test successful query."""
        url = reverse("search_api:query")
        data = {"query_id": "PLAIN-1", "query_text": "heart disease"}

        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("top_docs", response.data)
        self.assertIn("p5", response.data)
        self.assertEqual(len(response.data["top_docs"]), 3)

    def test_query_endpoint_caching(self):
        """Test that repeated queries use caching."""
        url = reverse("search_api:query")
        data = {"query_id": "PLAIN-1", "query_text": "heart disease"}

        # First request
        response1 = self.client.post(url, data, format="json")
        self.assertEqual(response1.status_code, status.HTTP_200_OK)

        # Second request (should use cache)
        response2 = self.client.post(url, data, format="json")
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

        # Results should be identical
        self.assertEqual(response1.data["top_docs"], response2.data["top_docs"])

    def test_query_endpoint_missing_field(self):
        """Test query with missing required field."""
        url = reverse("search_api:query")
        data = {"query_id": "PLAIN-1"}  # Missing query_text

        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_query_endpoint_empty_text(self):
        """Test query with empty text."""
        url = reverse("search_api:query")
        data = {"query_id": "PLAIN-1", "query_text": ""}

        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_query_endpoint_whitespace_text(self):
        """Test query with whitespace-only text."""
        url = reverse("search_api:query")
        data = {"query_id": "PLAIN-1", "query_text": "   "}

        response = self.client.post(url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class HealthCheckTests(APITestCase):
    """Tests for the /health/ endpoint."""

    def test_health_check(self):
        """Test the health check endpoint."""
        url = reverse("search_api:health")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], "healthy")
