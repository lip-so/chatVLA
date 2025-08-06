# Production Dockerfile - Optimized for Railway/Render deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements-minimal.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY simple_deploy.py wsgi.py ./
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY index.html 404.html ./

# Optional files (ignore if missing)
COPY robots.txt* sitemap.xml* CNAME* ./

# Set environment variables
ENV PYTHONPATH=/app:/app/backend
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Default port (will be overridden by platform)
ENV PORT=5000

# Expose the port
EXPOSE $PORT

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the application
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --keep-alive 5 wsgi:app