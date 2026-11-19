FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY gradeforge ./gradeforge
COPY tests ./tests
COPY CLAUDE.md PAYWALL_SETUP.md README.md ./

# Create data directory
RUN mkdir -p /data

# Set data directory
ENV GRADEFORGE_DATA_DIR=/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()" || exit 1

# Run app
CMD ["python", "-m", "uvicorn", "gradeforge.app:app", "--host", "0.0.0.0", "--port", "8000"]
