FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files first
COPY requirements-simple.txt .
RUN pip install --no-cache-dir -r requirements-simple.txt

# Copy backend code
COPY backend/ ./backend/
COPY start.py .
COPY railway_start.py .
COPY railway_debug.py .
COPY minimal_app.py .
COPY ultra_minimal.py .
COPY app.py .
COPY Procfile .

# Copy static files for Flask to serve
COPY index.html .
COPY 404.html .
COPY pages/ ./pages/
COPY css/ ./css/
COPY js/ ./js/
COPY assets/ ./assets/
COPY robots.txt .
COPY sitemap.xml .
COPY CNAME .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONPATH=/app

# Run using gunicorn (Railway standard)
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000}"]
