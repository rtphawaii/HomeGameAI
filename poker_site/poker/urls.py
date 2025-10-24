from django.urls import path
from . import views

urlpatterns = [
    # ... your existing routes
    path("rooms/clear/", views.clear_rooms_view, name="rooms_clear"),
]
