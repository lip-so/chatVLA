FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for real installations
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for conda environments
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Install minimal dependencies
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy app files
COPY . .

# Expose the port Railway expects
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Start the REAL backend
CMD ["python", "real_backend.py"]
