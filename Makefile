.DEFAULT_GOAL := check

sources = python/ tests/python/ bench/

# Engines the sf1 and sf10 targets launch. Override with `make sf1 engines=pycanopy`
engines = pycanopy duckdb sedonadb geopandas

# Preserve color in cargo output when running from a tty
export CARGO_TERM_COLOR=$(shell (test -t 0 && echo "always") || echo "auto")

.PHONY: setup ## Create .venv and install dev dependencies from uv.lock
setup:
	uv sync --group dev

# Build before linting and testing so clippy and nextest reuse maturin's compiled objects
.PHONY: check ## Format, build, lint and run every test
check:
	cargo fmt
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)
	@rm -f python/pycanopy/*.so
	uv run maturin develop
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)
	cargo fmt --all -- --check
	cargo clippy --tests -- -D warnings
	cargo nextest run
	uv run pytest tests/python/ --durations=5

.PHONY: build ## Debug build
build:
	@rm -f python/pycanopy/*.so
	uv run maturin develop

.PHONY: build-prod ## Optimised build
build-prod:
	@rm -f python/pycanopy/*.so
	uv run maturin develop --release

.PHONY: tune-engine ## Calibrate planner costs and update the bundled profile
tune-engine: build-prod
	uv run python -m bench.ops

.PHONY: profile ## Two-build SF1 profile on EC2, writes assets/profile.txt
profile:
	uv run --group bench python -m bench.spatial_bench --profile

.PHONY: sf1 ## SpatialBench SF1 on EC2, all four engines unless engines= is set
sf1:
	uv run --group bench python -m bench.spatial_bench --scale-factor 1 --engine $(engines)

.PHONY: sf10 ## SpatialBench SF10 on EC2, all four engines unless engines= is set
sf10:
	uv run --group bench python -m bench.spatial_bench --scale-factor 10 --engine $(engines)

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
