FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies first
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy app files
COPY . .

# Expose the port Railway expects
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Start the simplified backend (Railway will handle the real backend later)
CMD ["python", "start.py"]
