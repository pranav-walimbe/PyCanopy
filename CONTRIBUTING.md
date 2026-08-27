# Contributing

## Build setup

You need Rust (stable), `cargo-nextest`, and Python 3.10–3.12. Install the Rust test runner with `cargo install cargo-nextest` if it is not already available.

```bash
# Clone and set up
git clone https://github.com/pranav-walimbe/PyCanopy
cd PyCanopy

# Install dev dependencies
make setup

# Build the Rust extension and install in editable mode
make build

# Full check: format + build + lint + test
make check
```

For a release build (needed for accurate benchmark numbers):

```bash
make build-prod
```

## Make targets

| Command | What it does |
|:--------|:-------------|
| `make setup` | Create the virtual environment and install development dependencies |
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

Run the complete local check before submitting a change:

```bash
make check
```

Run either test suite directly while iterating:

```bash
cargo nextest run
uv run pytest tests/python -x -q
```

## Style

Use these commands for focused formatting and linting while iterating:

```bash
uv run ruff check --fix python/ tests/python/ bench/
uv run ruff format python/ tests/python/ bench/
cargo fmt
cargo clippy --tests -- -D warnings
```

Then run `make check` before submitting the change.

### Comments

- Comments annotate code in one line. Use a multi-line block only when one line truly cannot carry it.
- Comments should use near-zero commas. Say the one thing the reader needs and stop.
- A comment states its fact and stops. Never trail a justification clause off a comma, such as "so this is exact" or "the way the old code did".
- No em dashes, no semicolons in comments or docstrings.
- Single-line comments have no trailing period, multi-line comment blocks end each sentence with a period.
- Write a TODO as `// TODO(name):` or `# TODO(name):`.

### Python

- All imports at module level.
- Public functions use Google-style docstrings with `Args:`, `Returns:` and `Yields:` as applicable.
- Docstrings carry no `Raises:` section and no line for a `None` return or input.
- Private functions carry no docstring. They use a `#` comment as the first line in the body.

### Rust

- `pub` items require a one-line `///` doc comment. Private `fn` usually carry none.
- `///` docs are free prose. `Args:` and `Returns:` headings are Python-only.
- Every module file requires a single-line `//!` module doc.
- Every `unsafe` block requires a `// SAFETY:` comment.
