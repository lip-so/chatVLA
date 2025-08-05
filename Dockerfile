FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/
COPY start.py .

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

# Run the application using gunicorn with eventlet workers for SocketIO support
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:${PORT:-5000} backend.plug_and_play.wsgi:application
