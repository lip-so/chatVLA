FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for LeRobot
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Install LeRobot dependencies
RUN pip install --no-cache-dir \
    lerobot \
    opencv-python-headless \
    pyserial \
    numpy \
    torch \
    torchvision \
    huggingface-hub

# Copy the entire application
COPY . .

# Expose the port Railway expects
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Start the LeRobot backend
CMD ["python", "-m", "backend.api.main"]
