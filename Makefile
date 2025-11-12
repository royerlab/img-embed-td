.PHONY: test test-cov test-cov-xml help install clean

# Default target
help:
	@echo "Available targets:"
	@echo "  test         - Run tests without coverage"
	@echo "  test-cov     - Run tests with coverage and generate HTML report"
	@echo "  test-cov-xml - Run tests with coverage and generate XML report"
	@echo "  install      - Install dependencies with test extras"
	@echo "  clean        - Clean coverage files and cache"

# Install dependencies
install:
	uv sync --extra test

# Run tests without coverage
test: install
	@echo "Running tests..."
	uv run pytest -v

# Run tests with coverage and generate HTML report
test-cov: install
	@echo "Running tests with coverage..."
	uv run pytest --cov=src/img_embed_td --cov-report=html --cov-report=term-missing -v
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"
	@echo "You can open it with: open htmlcov/index.html (macOS) or xdg-open htmlcov/index.html (Linux)"

# Run tests with coverage and generate XML report
test-cov-xml: install
	@echo "Running tests with coverage (XML output)..."
	uv run pytest --cov=src/img_embed_td --cov-report=xml --cov-report=term-missing -v
	@echo ""
	@echo "Coverage report generated in coverage.xml"

# Clean coverage files and cache
clean:
	@echo "Cleaning coverage files and cache..."
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f .coverage
	rm -rf .pytest_cache/
	rm -rf src/**/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
