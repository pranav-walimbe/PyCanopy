.DEFAULT_GOAL := check

sources = python/ tests/python/ bench/

# Preserve colour in cargo output when running from a tty
export CARGO_TERM_COLOR=$(shell (test -t 0 && echo "always") || echo "auto")

.PHONY: setup ## Create .venv and install dev dependencies from uv.lock
setup:
	uv sync --group dev

.PHONY: format ## Auto-format Rust and Python source files
format:
	cargo fmt
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)

.PHONY: lint-python ## Lint Python source files
lint-python:
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)

.PHONY: lint-rust ## Lint Rust source files (fmt check + clippy over all code incl. tests)
lint-rust:
	cargo fmt --all -- --check
	cargo clippy --tests -- -D warnings

.PHONY: lint ## Lint Rust and Python source files
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
# instead of each triggering a second Rust compile pass.
# sccache (via RUSTC_WRAPPER in .cargo/config.toml) caches across runs.
.PHONY: check ## Format, build, lint, and test — run before every commit
check: format build lint
	cargo nextest run
	uv run pytest tests/python/ --durations=5

.PHONY: test ## Build and run all tests without formatting or linting
test: build
	cargo nextest run
	uv run pytest tests/python/

.PHONY: clean ## Remove build artifacts and caches
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -rf .pytest_cache .ruff_cache
	rm -f python/pycanopy/*.so

.PHONY: help ## Display this help message
help:
	@grep -E '^\.PHONY: .*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ".PHONY: |## "}; {printf "\033[36m%-15s\033[0m %s\n", $$2, $$3}'
