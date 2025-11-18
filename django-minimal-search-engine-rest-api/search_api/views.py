import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document, QueryRelevance
from .search_service import SearchService
from .serializers import (
    ErrorResponseSerializer,
    QueryRequestSerializer,
    QueryResponseSerializer,
    StatusResponseSerializer,
)

logger = logging.getLogger(__name__)


class StatusView(APIView):
    """
    POST /status/

    Returns information about the indexed documents and queries.

    Request: Empty JSON object {}

    Response:
    {
        "num_of_indexed_items": N,
        "num_of_queries_in_qrels": N
    }
    """

    def post(self, request):
        """
        Handle POST requests to /status/

        Returns counts of indexed documents and queries in qrels.
        """
        try:
            num_indexed = Document.objects.count()
            num_queries = QueryRelevance.objects.values("query_id").distinct().count()

            response_data = {
                "num_of_indexed_items": num_indexed,
                "num_of_queries_in_qrels": num_queries,
            }

            serializer = StatusResponseSerializer(data=response_data)
            serializer.is_valid(raise_exception=True)

            logger.info(f"Status check: {num_indexed} documents, {num_queries} queries")

            return Response(serializer.validated_data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in StatusView: {e}", exc_info=True)
            error_serializer = ErrorResponseSerializer(data={"error": "Internal server error", "details": {"message": str(e)}})
            error_serializer.is_valid()
            return Response(
                error_serializer.validated_data,
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class QueryView(APIView):
    """
    POST /query/

    Processes a search query and returns ranked results with evaluation metrics.

    Request:
    {
        "query_id": "PLAIN-831",
        "query_text": "cardiovascular disease"
    }

    Response:
    {
        "top_docs": ["MED-1", "MED-2", ..., "MED-10"],
        "p5": 0.732
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_service = SearchService()

    def post(self, request):
        """
        Handle POST requests to /query/

        Validates input, performs search, and calculates P@5.
        """
        try:
            input_serializer = QueryRequestSerializer(data=request.data)
            if not input_serializer.is_valid():
                return Response(
                    {"error": "Invalid input", "details": input_serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            query_id = input_serializer.validated_data["query_id"]
            query_text = input_serializer.validated_data["query_text"]

            logger.info(f"Processing query: {query_id} - {query_text}")

            if Document.objects.count() == 0:
                return Response(
                    {
                        "error": "No documents indexed",
                        "details": {"message": "Please run the indexing command first"},
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            result = self.search_service.search_and_evaluate(query_text=query_text, query_id=query_id, top_k=10)

            response_data = {"top_docs": result["top_docs"], "p5": result["p5"]}

            output_serializer = QueryResponseSerializer(data=response_data)
            output_serializer.is_valid(raise_exception=True)

            logger.info(
                f"Query {query_id} completed: P@5={result['p5']}, "
                f"top doc={result['top_docs'][0] if result['top_docs'] else 'none'}"
            )

            return Response(output_serializer.validated_data, status=status.HTTP_200_OK)

        except ValueError as e:
            logger.warning(f"Validation error in QueryView: {e}")
            return Response(
                {"error": "Invalid query", "details": {"message": str(e)}},
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Error in QueryView: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error", "details": {"message": str(e)}},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class HealthCheckView(APIView):
    """
    GET /health/

    Simple health check endpoint to verify the service is running.
    """

    def get(self, request):
        """Return a simple health status."""
        return Response({"status": "healthy"}, status=status.HTTP_200_OK)
