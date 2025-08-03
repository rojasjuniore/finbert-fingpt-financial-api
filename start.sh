#!/bin/bash

# FinBERT API Startup Script

set -e

echo "Starting FinBERT API..."
echo "Environment: ${ENVIRONMENT:-development}"
echo "Model: ${MODEL_NAME:-ProsusAI/finbert}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p .cache/transformers
mkdir -p .cache/huggingface

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your configuration."
fi

# Run tests (optional)
if [ "${RUN_TESTS:-false}" = "true" ]; then
    echo "Running tests..."
    python -m pytest tests/ -v
fi

# Start the API
echo "Starting API server..."
if [ "${ENVIRONMENT:-development}" = "development" ]; then
    python main.py
else
    # Production mode with gunicorn
    gunicorn app.main:app \
        --bind ${API_HOST:-0.0.0.0}:${API_PORT:-8000} \
        --workers ${API_WORKERS:-1} \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 120 \
        --keepalive 2 \
        --access-logfile - \
        --error-logfile -
fi