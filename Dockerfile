# Multi-stage build for optimized production deployment
# Stage 1: Builder stage with compilation dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies in smaller chunks to reduce memory usage
# Split package installation to avoid memory issues
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install additional build tools if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make \
    cmake \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel for faster installations
RUN pip install --upgrade pip wheel setuptools

# Copy requirements and install Python dependencies
# Install in stages to manage memory better
COPY requirements-deploy.txt .

# Install core dependencies first (lightweight)
RUN echo "Flask>=2.3.0" > requirements-core.txt && \
    echo "flask-cors>=4.0.0" >> requirements-core.txt && \
    echo "flask-socketio>=5.3.6" >> requirements-core.txt && \
    echo "python-dotenv>=1.0.0" >> requirements-core.txt && \
    echo "gunicorn>=21.2.0" >> requirements-core.txt && \
    echo "eventlet>=0.33.3" >> requirements-core.txt && \
    echo "PyJWT>=2.8.0" >> requirements-core.txt && \
    echo "PyYAML>=6.0.1" >> requirements-core.txt && \
    echo "requests>=2.31.0" >> requirements-core.txt && \
    pip install --no-cache-dir -r requirements-core.txt

# Install medium-weight dependencies
RUN echo "bcrypt>=4.1.2" > requirements-medium.txt && \
    echo "Pillow>=10.0.0" >> requirements-medium.txt && \
    echo "pyserial>=3.5" >> requirements-medium.txt && \
    echo "psutil>=5.9.0" >> requirements-medium.txt && \
    echo "numpy>=1.24.0,<2.0" >> requirements-medium.txt && \
    pip install --no-cache-dir -r requirements-medium.txt

# Install heavy dependencies one by one to manage memory
# Install scipy separately
RUN pip install --no-cache-dir "scipy>=1.10.0"

# Install pandas separately
RUN pip install --no-cache-dir "pandas>=2.0.0"

# Install matplotlib separately
RUN pip install --no-cache-dir "matplotlib>=3.7.0"

# Install scikit-learn separately
RUN pip install --no-cache-dir "scikit-learn>=1.3.0"

# Install opencv-python (headless version for server)
RUN pip install --no-cache-dir opencv-python-headless

# Install huggingface-hub
RUN pip install --no-cache-dir "huggingface-hub>=0.16.0"

# Install Firebase admin (can be heavy)
RUN pip install --no-cache-dir "firebase-admin>=6.4.0" || echo "Firebase installation failed, continuing..."

# Note: Skipping torch, torchvision, transformers, datasets, clip-by-openai 
# as they are VERY heavy and might not be needed for basic deployment
# Uncomment these lines if you absolutely need them:
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torchvision
# RUN pip install --no-cache-dir transformers datasets

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (much smaller)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY backend/ ./backend/
COPY simple_deploy.py .
COPY wsgi.py .
COPY Procfile .

# Copy static files for Flask to serve
COPY index.html .
COPY 404.html .

# Create directories and copy static content
RUN mkdir -p frontend/pages frontend/css frontend/js frontend/assets

# Copy frontend files (handle missing directories gracefully)
COPY --chown=root:root frontend/pages/ ./frontend/pages/ 2>/dev/null || true
COPY --chown=root:root frontend/css/ ./frontend/css/ 2>/dev/null || true
COPY --chown=root:root frontend/js/ ./frontend/js/ 2>/dev/null || true
COPY --chown=root:root frontend/assets/ ./frontend/assets/ 2>/dev/null || true

# Copy other static files
COPY robots.txt sitemap.xml CNAME* ./

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=5000
ENV PYTHONPATH=/app:/app/backend
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with Gunicorn for production
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "120", "--graceful-timeout", "120", "wsgi:app"]