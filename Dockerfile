# ============================================================
#  Quant & Data Science Master Portfolio — Dockerfile
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib, lightgbm, xgboost, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.headless", "true", \
            "--server.port", "8501", \
            "--server.address", "0.0.0.0"]
