"""
URL configuration for search_api.
"""

from django.urls import path

from .views import HealthCheckView, QueryView, StatusView

app_name = "search_api"

urlpatterns = [
    path("api/status/", StatusView.as_view(), name="status"),
    path("api/query/", QueryView.as_view(), name="query"),
    path("api/health/", HealthCheckView.as_view(), name="health"),
]
