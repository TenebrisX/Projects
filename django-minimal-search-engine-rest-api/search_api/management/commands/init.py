"""
Django management command to index documents from the dataset.

Usage:
    python manage.py init
    python manage.py init --docs-file custom.docs
    python manage.py init --clear

This command:
1. Loads documents from train.docs (or specified file)
2. Loads queries from train.titles.queries (or specified file)
3. Loads qrels from train.3-2-1.qrel (or specified file)
4. Generates embeddings for all documents
5. Stores everything in the database
"""

import logging
import os
from typing import List, Tuple

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from search_api.embedding_service import EmbeddingService
from search_api.models import Document, Query, QueryRelevance

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Management command to index documents and load qrels."""

    help = "Index documents from the dataset and load qrels"

    def __init__(self):
        super().__init__()
        self.embedding_service = EmbeddingService()

    def add_arguments(self, parser):
        """Add command-line arguments."""
        parser.add_argument(
            "--data-path",
            type=str,
            default=settings.DATASET_PATH,
            help="Path to the dataset directory",
        )
        parser.add_argument(
            "--docs-file",
            type=str,
            default="train.docs",
            help="Name of the documents file (default: train.docs)",
        )
        parser.add_argument(
            "--queries-file",
            type=str,
            default="train.titles.queries",
            help="Name of the queries file (default: train.titles.queries)",
        )
        parser.add_argument(
            "--qrels-file",
            type=str,
            default="train.3-2-1.qrel",
            help="Name of the qrels file (default: train.3-2-1.qrel)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Batch size for embedding generation",
        )
        parser.add_argument("--clear", action="store_true", help="Clear existing data before indexing")

    def handle(self, *args, **options):
        """Execute the command."""
        data_path = options["data_path"]
        docs_file = options["docs_file"]
        queries_file = options["queries_file"]
        qrels_file = options["qrels_file"]
        batch_size = options["batch_size"]
        clear_data = options["clear"]

        self.stdout.write(self.style.SUCCESS("Starting indexing process..."))

        # Verify data directory exists
        if not os.path.exists(data_path):
            raise CommandError(
                f"Data directory not found: {data_path}\n" f"Please download and extract the dataset to this location."
            )

        # Clear existing data if requested
        if clear_data:
            self.stdout.write("Clearing existing data...")
            Document.objects.all().delete()
            QueryRelevance.objects.all().delete()
            Query.objects.all().delete()
            self.stdout.write(self.style.SUCCESS("Data cleared"))

        try:
            # Step 1: Load and index documents
            self.stdout.write("Step 1: Loading documents...")
            docs = self.load_documents(data_path, docs_file)
            self.stdout.write(f"Loaded {len(docs)} documents from {docs_file}")

            self.stdout.write("Step 2: Generating embeddings...")
            self.index_documents(docs, batch_size)
            self.stdout.write(self.style.SUCCESS(f"Indexed {len(docs)} documents"))

            # Step 2: Load queries
            self.stdout.write("Step 3: Loading queries...")
            queries = self.load_queries(data_path, queries_file)
            self.save_queries(queries)
            self.stdout.write(self.style.SUCCESS(f"Loaded {len(queries)} queries from {queries_file}"))

            # Step 3: Load qrels
            self.stdout.write("Step 4: Loading qrels...")
            qrels = self.load_qrels(data_path, qrels_file)
            self.save_qrels(qrels)
            self.stdout.write(self.style.SUCCESS(f"Loaded {len(qrels)} relevance judgments from {qrels_file}"))

            # Summary
            self.stdout.write(self.style.SUCCESS("\n=== Indexing Complete ==="))
            self.stdout.write(f"Documents indexed: {Document.objects.count()}")
            self.stdout.write(f"Queries loaded: {Query.objects.count()}")
            self.stdout.write(f"Qrels loaded: {QueryRelevance.objects.count()}")

        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            raise CommandError(f"Indexing failed: {e}")

    def load_documents(self, data_path: str, filename: str) -> List[Tuple[str, str]]:
        """
        Load documents from file.

        Format: Each line is a document with format "doc_id\tdocument_text"

        Args:
            data_path: Directory containing the file
            filename: Name of the documents file

        Returns:
            List of (doc_id, text) tuples
        """
        filepath = os.path.join(data_path, filename)

        if not os.path.exists(filepath):
            raise CommandError(f"Documents file not found: {filepath}")

        documents = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t", 1)
                if len(parts) != 2:
                    logger.warning(f"Skipping malformed line {line_num} in {filename}")
                    continue

                doc_id, text = parts

                if not text or not text.strip():
                    logger.warning(f"Skipping empty document: {doc_id}")
                    continue

                documents.append((doc_id, text))

        return documents

    def index_documents(self, documents: List[Tuple[str, str]], batch_size: int):
        """
        Generate embeddings and save documents to database.

        Args:
            documents: List of (doc_id, text) tuples
            batch_size: Number of documents to process at once
        """
        total = len(documents)
        total_batches = (total + batch_size - 1) // batch_size

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1

            doc_ids = [doc_id for doc_id, _ in batch]
            texts = [text for _, text in batch]

            progress = (batch_num / total_batches) * 100
            self.stdout.write(
                f"Processing batch {batch_num}/{total_batches} ({progress:.1f}%) " f"[{doc_ids[0]} ... {doc_ids[-1]}]"
            )

            try:
                embeddings = self.embedding_service.embed_batch(texts)

                with transaction.atomic():
                    for j, (doc_id, text) in enumerate(batch):
                        embedding_str = self.embedding_service.serialize_embedding(embeddings[j])

                        Document.objects.update_or_create(
                            doc_id=doc_id,
                            defaults={"text": text, "embedding": embedding_str},
                        )

                self.stdout.write(self.style.SUCCESS(f"✓ Saved batch {batch_num}"))

            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}", exc_info=True)
                self.stdout.write(self.style.ERROR(f"✗ Failed to process batch {batch_num}: {e}"))
                raise

    def load_queries(self, data_path: str, filename: str) -> List[Tuple[str, str]]:
        """
        Load queries from file.

        Format: Each line is "query_id\tquery_text"

        Args:
            data_path: Directory containing the file
            filename: Name of the queries file

        Returns:
            List of (query_id, query_text) tuples
        """
        filepath = os.path.join(data_path, filename)

        if not os.path.exists(filepath):
            raise CommandError(f"Queries file not found: {filepath}")

        queries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t", 1)
                if len(parts) != 2:
                    logger.warning(f"Skipping malformed line {line_num} in {filename}")
                    continue

                query_id, query_text = parts
                queries.append((query_id, query_text))

        return queries

    def save_queries(self, queries: List[Tuple[str, str]]):
        """Save queries to database."""
        with transaction.atomic():
            for query_id, query_text in queries:
                Query.objects.update_or_create(query_id=query_id, defaults={"query_text": query_text})

    def load_qrels(self, data_path: str, filename: str) -> List[Tuple[str, str, int]]:
        """
        Load query relevance judgments from file.

        Format: "query_id\tdoc_id\tunused\trelevance_score"

        Args:
            data_path: Directory containing the file
            filename: Name of the qrels file

        Returns:
            List of (query_id, doc_id, relevance_score) tuples
        """
        filepath = os.path.join(data_path, filename)

        if not os.path.exists(filepath):
            raise CommandError(f"Qrels file not found: {filepath}")

        qrels = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 4:
                    logger.warning(f"Skipping malformed line {line_num} in {filename}")
                    continue

                query_id = parts[0]
                doc_id = parts[2]
                relevance_score = int(parts[3])

                qrels.append((query_id, doc_id, relevance_score))

        return qrels

    def save_qrels(self, qrels: List[Tuple[str, str, int]]):
        """Save query relevance judgments to database."""
        with transaction.atomic():
            for query_id, doc_id, relevance_score in qrels:
                QueryRelevance.objects.update_or_create(
                    query_id=query_id,
                    doc_id=doc_id,
                    defaults={"relevance_score": relevance_score},
                )
