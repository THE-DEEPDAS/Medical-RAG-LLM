# Docker file ma base image of a container kevi rite banavani enu config hoy
FROM python:3.10-slim

# container ni andar working directory
WORKDIR /app

# Install system dependencies
# build-essential: C/C++ compiler
# curl: for downloading files
# software-properties-common: for mnaging software repositories
# git: for version control and cloning repositories
# last line cleans up apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories for storing model and data
RUN mkdir -p /app/model /app/Data

# Run with uvicorn
CMD ["uvicorn", "rag:app", "--host", "0.0.0.0", "--port", "8000"]
