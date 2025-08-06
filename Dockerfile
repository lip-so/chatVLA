FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files first
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy backend code
COPY backend/ ./backend/
COPY simple_deploy.py .

COPY wsgi.py .
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

# Run the application with Gunicorn for production
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT wsgi:app
