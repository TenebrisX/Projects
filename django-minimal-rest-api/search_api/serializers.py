from rest_framework import serializers


class StatusResponseSerializer(serializers.Serializer):
    """
    Serializer for the /status/ endpoint response.

    Returns information about indexed documents and queries.
    """

    num_of_indexed_items = serializers.IntegerField(help_text="Number of documents indexed in the system")
    num_of_queries_in_qrels = serializers.IntegerField(help_text="Number of unique queries in the relevance judgments")


class QueryRequestSerializer(serializers.Serializer):
    """
    Serializer for the /query/ endpoint request.

    Validates incoming query requests.
    """

    query_id = serializers.CharField(
        max_length=100,
        required=True,
        help_text="Unique identifier for the query (e.g., 'PLAIN-***')",
    )
    query_text = serializers.CharField(
        required=True,
        allow_blank=False,
        help_text="The query text to search for (e.g., 'cardiovascular disease')",
    )

    def validate_query_text(self, value):
        """Ensure query text is not empty after stripping whitespace."""
        if not value or not value.strip():
            raise serializers.ValidationError("Query text cannot be empty")
        return value.strip()

    def validate_query_id(self, value):
        """Ensure query_id is not empty after stripping whitespace."""
        if not value or not value.strip():
            raise serializers.ValidationError("Query ID cannot be empty")
        return value.strip()


class QueryResponseSerializer(serializers.Serializer):
    """
    Serializer for the /query/ endpoint response.

    Returns search results and evaluation metrics.
    """

    top_docs = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of top 10 document IDs ranked by cosine similarity",
    )
    p5 = serializers.FloatField(help_text="Precision@5 metric for the query")


class ErrorResponseSerializer(serializers.Serializer):
    """
    Serializer for error responses.
    """

    error = serializers.CharField(help_text="Error message")
    details = serializers.DictField(required=False, help_text="Additional error details")
