# stringart_project/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('stringart_app.urls')),  # ← Route the root URL to core.urls
]
