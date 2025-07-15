# Makefile for Six-Stack Personality Classification Pipeline
# Author: AI Assistant
# Date: 2025-07-14

.PHONY: help install format lint typecheck security check-all test run train-models dash stop-dash

# Default target
help:
	@echo "Six-Stack Personality Classification Pipeline"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install dependencies using uv"
	@echo "  format        - Format code with ruff"
	@echo "  lint          - Lint code with ruff (includes format check)"
	@echo "  typecheck     - Type check with mypy"
	@echo "  security      - Security check with bandit"
	@echo "  check-all     - Run all code quality checks (lint, typecheck, security)"
	@echo "  test          - Run tests"
	@echo "  run           - Run the modular pipeline"
	@echo "  train-models  - Train and save ML models"
	@echo "  dash          - Run Dash application"
	@echo "  stop-dash     - Stop Dash application"
	@echo ""

# Dependency management
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync --all-extras

# Code quality with Ruff
format:
	@echo "🎨 Formatting code with ruff..."
	uv run ruff format src/ dash_app/ tests/ scripts/

lint:
	@echo "🔍 Linting code with ruff..."
	uv run ruff check .
	uv run ruff format --check .

# Type checking
typecheck:
	@echo "🔎 Type checking with mypy..."
	uv run mypy src/ --ignore-missing-imports

# Security checking
security:
	@echo "🔒 Security checking with bandit..."
	uv run bandit -r src/ -f json

# Run all quality checks
check-all: lint typecheck security
	@echo "✅ All code quality checks completed!"

# Testing
test:
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v

# Pipeline execution
run:
	@echo "🚀 Running modular pipeline..."
	uv run python src/main_modular.py

# Model training
train-models:
	@echo "🤖 Training and saving ML models..."
	uv run python scripts/train_and_save_models.py

# Dash application
dash:
	@echo "📊 Starting Dash application..."
	uv run python dash_app/main.py --model-name ensemble

stop-dash:
	@echo "🛑 Stopping Dash application..."
	@lsof -ti:8050 | xargs kill -9 2>/dev/null || echo "No process found on port 8050"
