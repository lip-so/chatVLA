# Simple Dockerfile that works on Railway
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app:/app/backend
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Railway sets PORT dynamically, don't hardcode it
# Remove EXPOSE as Railway handles this

# Start command - Railway will override this with Procfile
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "wsgi:app"]