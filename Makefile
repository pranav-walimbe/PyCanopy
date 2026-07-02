.DEFAULT_GOAL := check

sources = python/ tests/python/ bench/

# Preserve color in cargo output when running from a tty
export CARGO_TERM_COLOR=$(shell (test -t 0 && echo "always") || echo "auto")

.PHONY: setup ## Create .venv and install dev dependencies from uv.lock
setup:
	uv sync --group dev

.PHONY: format
format:
	cargo fmt
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)

.PHONY: lint-python
lint-python:
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)

.PHONY: lint-rust
lint-rust:
	cargo fmt --all -- --check
	cargo clippy --tests -- -D warnings

.PHONY: lint
lint: lint-python lint-rust

.PHONY: build ## Debug build
build:
	@rm -f python/pycanopy/*.so
	uv run maturin develop

.PHONY: build-prod ## Optimised build
build-prod:
	@rm -f python/pycanopy/*.so
	uv run maturin develop --release

# Build first so clippy and cargo nextest reuse compiled objects from maturin
.PHONY: check
check: format build lint
	cargo nextest run
	uv run pytest tests/python/ --durations=5

.PHONY: test
test: build
	cargo nextest run
	uv run pytest tests/python/

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -rf .pytest_cache .ruff_cache
	rm -f python/pycanopy/*.so

.PHONY: help
help:
	@grep -E '^\.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-15s\033[0m %s\n", $$2, $$3}'
