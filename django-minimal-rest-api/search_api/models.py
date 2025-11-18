from django.db import models


class Document(models.Model):
    """
    Represents a document in the corpus.

    Attributes:
        doc_id (str): Unique identifier for the document (e.g., "MED-1")
        text (str): Full text content of the document
        embedding (str): JSON-serialized numpy array of the document embedding
        created_at (datetime): Timestamp when document was indexed
    """

    doc_id = models.CharField(max_length=100, unique=True, db_index=True)
    text = models.TextField()
    embedding = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["doc_id"]
        indexes = [
            models.Index(fields=["doc_id"]),
        ]

    def __str__(self):
        return f"Document({self.doc_id})"


class QueryRelevance(models.Model):
    """
    Represents query-document relevance judgments (qrels).

    Attributes:
        query_id (str): Unique identifier for the query (e.g., "PLAIN-831")
        doc_id (str): Document identifier that is relevant to this query
        relevance_score (int): Relevance score (typically 1, 2, or 3)
    """

    query_id = models.CharField(max_length=100, db_index=True)
    doc_id = models.CharField(max_length=100, db_index=True)
    relevance_score = models.IntegerField()

    class Meta:
        unique_together = ["query_id", "doc_id"]
        indexes = [
            models.Index(fields=["query_id"]),
            models.Index(fields=["doc_id"]),
        ]

    def __str__(self):
        return f"QueryRel({self.query_id} -> {self.doc_id}, score={self.relevance_score})"


class Query(models.Model):
    """
    Represents a query from the dataset.

    Attributes:
        query_id (str): Unique identifier for the query
        query_text (str): Text of the query
    """

    query_id = models.CharField(max_length=100, unique=True, db_index=True)
    query_text = models.TextField()

    class Meta:
        ordering = ["query_id"]
        indexes = [
            models.Index(fields=["query_id"]),
        ]

    def __str__(self):
        return f"Query({self.query_id}: {self.query_text[:50]})"
