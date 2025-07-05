# 1. Use an official Python runtime as a parent image
FROM python:3.12.11-slim-bookworm

# 2. Prevent Python from buffering stdout/stderr (so logs appear immediately)
ENV PYTHONUNBUFFERED=1

# 3. Set workdir
WORKDIR /app

# 4. Install system deps (Pillow etc) and clean up
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      libjpeg62-turbo-dev \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# 5. Copy your requirements.txt and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your code
COPY . .

# 7. Collect static files into STATIC_ROOT
RUN python manage.py collectstatic --noinput

# 8. Expose port 8000
EXPOSE 8000

# 9. Run Gunicorn with your working config
CMD ["gunicorn", "stringart_project.wsgi:application", \
     "--worker-class", "gthread", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--keep-alive", "2", \
     "--access-logfile", "-", \
     "--log-level", "info", \
     "--bind", "0.0.0.0:8000"]
