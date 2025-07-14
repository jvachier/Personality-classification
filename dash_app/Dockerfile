# Dockerfile for Personality Classification Dash Application

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Set UV to not create virtual environments
ENV UV_SYSTEM_PYTHON=1

# Copy pyproject.toml and source code for dependency installation
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Install dependencies using uv (no virtual environment)
RUN uv pip install --system --no-cache-dir .

# Copy dash application code
COPY dash_app/src/ ./dash_app/src/
COPY dash_app/main.py ./dash_app/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Command to run the application
CMD ["python", "dash_app/main.py", "--model-name", "personality_model", "--host", "0.0.0.0"]
