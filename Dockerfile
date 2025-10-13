# =========================================
# ✅ Aeiron Ingestion Worker - Dockerfile
# =========================================

# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Ensure logs flush immediately
ENV PYTHONUNBUFFERED=1

# Default command for Render Background Worker
CMD ["python", "main.py"]
