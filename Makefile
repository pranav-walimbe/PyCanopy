.DEFAULT_GOAL := check

sources = python/ tests/python/ benchmarks/

# Preserve colour in cargo output when running from a tty.
export CARGO_TERM_COLOR=$(shell (test -t 0 && echo "always") || echo "auto")

.PHONY: format ## Auto-format Rust and Python source files
format:
	cargo fmt
	.venv/bin/ruff check --fix $(sources)
	.venv/bin/ruff format $(sources)

.PHONY: lint-python ## Lint Python source files
lint-python:
	.venv/bin/ruff check $(sources)
	.venv/bin/ruff format --check $(sources)

.PHONY: lint-rust ## Lint Rust source files (fmt check + clippy over all code incl. tests)
lint-rust:
	cargo fmt --all -- --check
	cargo clippy --tests -- -D warnings

.PHONY: lint ## Lint Rust and Python source files
lint: lint-python lint-rust

.PHONY: build ## Debug build — fast compile, use for local iteration
build:
	@rm -f python/pycanopy/*.so
	maturin develop

.PHONY: build-prod ## Optimised build — use for benchmarks and profiling
build-prod:
	@rm -f python/pycanopy/*.so
	maturin develop --release

# Build first so clippy and cargo nextest reuse compiled objects from maturin
# instead of each triggering a second Rust compile pass.
# sccache (via RUSTC_WRAPPER in .cargo/config.toml) caches across runs.
.PHONY: check ## Format, build, lint, and test — run before every commit
check: format build lint
	cargo nextest run
	.venv/bin/pytest tests/python/ --durations=5

.PHONY: test ## Build and run all tests without formatting or linting
test: build
	cargo nextest run
	.venv/bin/pytest tests/python/

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
