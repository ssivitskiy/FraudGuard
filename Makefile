.PHONY: help install install-dev test lint format type-check clean train predict app docker-build docker-run

# Default target
help:
	@echo "FraudGuard - ML Fraud Detection"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install all dependencies (including dev)"
	@echo "  pre-commit    Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  test          Run tests with coverage"
	@echo "  test-fast     Run tests without coverage"
	@echo "  lint          Run linter (ruff)"
	@echo "  format        Format code (ruff)"
	@echo "  type-check    Run type checker (mypy)"
	@echo "  check         Run all checks (lint + type-check + test)"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  train         Train the fraud detection model"
	@echo "  predict       Run prediction (use ARGS for parameters)"
	@echo "  app           Run Streamlit application"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         Remove build artifacts and cache"

# ========================
# Setup
# ========================
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"

pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

# ========================
# Development
# ========================
test:
	pytest --cov=fraudguard --cov-report=term-missing --cov-report=html

test-fast:
	pytest -x -q

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

type-check:
	mypy fraudguard/ --ignore-missing-imports

check: lint type-check test

# ========================
# ML Pipeline
# ========================
train:
	python -m scripts.train

predict:
	python -m scripts.predict $(ARGS)

app:
	streamlit run app/app.py

# ========================
# Docker
# ========================
docker-build:
	docker build -t fraudguard:latest .

docker-run:
	docker run -p 8501:8501 fraudguard:latest

# ========================
# Cleanup
# ========================
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete