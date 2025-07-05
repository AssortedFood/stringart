# stringart_app/urls.py

from django.urls import path
from .views import home, stream_logs

urlpatterns = [
    path('', home, name='home'),
    path('stream-logs/', stream_logs, name='stream_logs'),
]
