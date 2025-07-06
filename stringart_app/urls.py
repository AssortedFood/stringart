# stringart_app/urls.py

from django.urls import path
from .views import home, stream_logs, stream_results, stop_job

urlpatterns = [
    path('', home, name='home'),
    path('stream-logs/', stream_logs, name='stream_logs'),
    path('stream-results/', stream_results, name='stream_results'),
    path('stop-job/<uuid:job_id>/', stop_job, name='stop_job'),
]
