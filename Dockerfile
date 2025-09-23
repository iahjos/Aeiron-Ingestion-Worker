# Use a slim Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyMuPDF, pandas, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose port (for local dev; Render sets $PORT dynamically)
EXPOSE 8000

# Use shell form so $PORT gets expanded correctly
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
