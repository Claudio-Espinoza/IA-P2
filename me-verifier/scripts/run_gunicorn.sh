#!/bin/bash
# Run Flask API with Gunicorn

cd "$(dirname "$0")/.."

gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    api.app:app
