.PHONY: fmt lint test build check

# Auto-fix formatting for Rust and Python
fmt:
	cargo fmt
	ruff format python/ tests/python/ benchmarks/

# Lint without modifying files
lint:
	cargo clippy -- -D warnings
	ruff check python/ tests/python/

# Build the release extension then run all tests
test: build
	cargo test
	pytest tests/python/

# Build the Python extension in release mode
build:
	maturin develop --release

# Full pre-commit check: format, lint, test
check: fmt lint test
