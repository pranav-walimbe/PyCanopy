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
| `make check` | Format, build, lint and run every test. The default target |
| `make build` | Debug build |
| `make build-prod` | Release build, needed for accurate benchmark numbers |
| `make tune-engine` | Calibrate planner costs and update the bundled profile |
| `make profile` | Two-build SF1 profile, writes `assets/profile.txt` |
| `make sf1`, `make sf10` | SpatialBench at that scale factor, all four engines |
| `make clean` | Remove build artifacts |

`profile`, `sf1` and `sf10` launch EC2 instances and need AWS credentials. Narrow the engine
list with `make sf1 engines=pycanopy`.

## Running tests

```bash
make check
# or directly
cargo nextest run
uv run pytest tests/python -x -q
```

## Style

After every code change, run:

```bash
uv run ruff format && uv run ruff check
cargo fmt && cargo clippy
```

`scripts/check_comments.py` enforces the rules below that ruff and clippy cannot express. Run
`uv run python scripts/check_comments.py` to list violations, or with `--fix` to strip trailing
periods from single-line comments.

For coding style, I like these guidelines:

**Comments**

- Comments annotate code in one line. Use a multi-line block only when one line truly cannot carry it.
- Comments should use near-zero commas. Say the one thing the reader needs and stop.
- A comment states its fact and stops. Never trail a justification clause off a comma, such as "so this is exact" or "the way the old code did".
- No em dashes, no semicolons in comments or docstrings.
- Single-line comments have no trailing period, multi-line comment blocks end each sentence with a period.
- Write a TODO as `// TODO(name):` or `# TODO(name):`.

**Python**

- All imports at module level.
- Public functions use Google-style docstrings with `Args:`, `Returns:` and `Yields:` as applicable.
- Docstrings carry no `Raises:` section and no line for a `None` return or input.
- Private functions carry no docstring. They use a `#` comment as the first line in the body.

**Rust**

- `pub` items require a one-line `///` doc comment. Private `fn` usually carry none.
- `///` docs are free prose. `Args:` and `Returns:` headings are Python-only.
- Every module file requires a single-line `//!` module doc.
- Every `unsafe` block requires a `// SAFETY:` comment.
