.PHONY: fmt lint test build release check

# Auto-fix formatting for Rust and Python
fmt:
	cargo fmt
	ruff format python/ tests/python/ benchmarks/

# Lint without modifying files
lint:
	cargo clippy -- -D warnings
	ruff check python/ tests/python/

# Build the Python extension (fast: no LTO, parallel codegen)
build:
	maturin develop --profile dev-release

# Build with full release optimisations (slow: LTO + single codegen unit)
release:
	maturin develop --release

# Build the fast extension then run all tests
test: build
	cargo test
	.venv/bin/pytest tests/python/

# Full pre-commit check: format, lint, test
check: fmt lint test
