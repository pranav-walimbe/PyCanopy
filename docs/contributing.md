# Contributing

## Build setup

You need Rust (stable), `cargo-nextest`, and Python 3.10–3.12. Install the Rust test runner with `cargo install cargo-nextest` if it is not already available.

```bash
# Clone and set up
git clone https://github.com/pranav-walimbe/PyCanopy
cd PyCanopy

# Install dev dependencies (uv recommended)
uv sync --group dev

# Build the Rust extension and install in editable mode
uv run maturin develop

# Full check: format + build + lint + test
make check
```

For a release build (needed for accurate benchmark numbers):

```bash
uv run maturin develop --release
```

## Make targets

| Command | What it does |
|:--------|:-------------|
| `make check` | fmt + build + lint + test |
| `make test` | Run the Python test suite |
| `make build` | Debug build |
| `make build-prod` | Release build |
| `make clean` | Remove build artifacts |

## Running tests

```bash
make test
# or directly
uv run pytest tests/python -x -q
```

## Style

After every code change, run:

```bash
uv run ruff format && uv run ruff check
cargo fmt && cargo clippy
```

To avoid a slopocolypse, I recommend using these guidelines:

- No em dashes, no semicolons in comments or docstrings.
- All Python imports at module level.
- Public Python functions use Google-style docstrings (`Args:`, `Returns:`).
- Private Python functions use a `#` comment as the first line in the body.
- Rust `pub` items require `///` doc comments; every module file requires `//!`.
- Single-line comments have no trailing period, multi-line comment blocks end each sentence with a period.
