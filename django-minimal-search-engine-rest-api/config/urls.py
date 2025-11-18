from django.urls import include, path

urlpatterns = [
    path("", include("search_api.urls")),
]
