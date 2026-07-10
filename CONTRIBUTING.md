# Contributing

## Build setup

You need Rust (stable) and Python 3.10–3.12.

```bash
# Clone and set up
git clone https://github.com/pranav-walimbe/PyCanopy
cd PyCanopy

# Install dev dependencies (uv recommended)
uv sync --group dev

# Build the Rust extension and install in editable mode
maturin develop

# Full check: format + build + lint + test
make check
```

For a release build (needed for accurate benchmark numbers):

```bash
maturin develop --release
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
pytest tests/python -x -q
```

## Style

After every code change, run:

```bash
ruff format && ruff check
cargo fmt && cargo clippy
```

To avoid a slopocolypse, I like these guidelines:

- No em dashes, no semicolons in comments or docstrings.
- All Python imports at module level.
- Public Python functions use Google-style docstrings (`Args:`, `Returns:`).
- Private Python functions use a `#` comment as the first line in the body.
- Rust `pub` items require `///` doc comments; every module file requires `//!`.
- Single-line comments have no trailing period, multi-line comment blocks end each sentence with a period.
