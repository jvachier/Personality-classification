# Pre-commit Setup Guide

This project uses [pre-commit](https://pre-commit.com/) to ensure code quality and consistency.

## Installation

Pre-commit is automatically installed when you run:

```bash
make setup-env
# or
make install-dev
```

To manually install pre-commit hooks:

```bash
make pre-commit-install
# or
uv run pre-commit install
```

## Usage

### Automatic (Recommended)

Pre-commit will automatically run on every `git commit`. If any checks fail, the commit will be blocked until issues are fixed.

### Manual Execution

Run on staged files only:

```bash
make pre-commit-run
# or
uv run pre-commit run
```

Run on all files:

```bash
make pre-commit-all
# or
uv run pre-commit run --all-files
```

## Configured Hooks

### Code Formatting

- **Black**: Python code formatter
- **isort**: Import sorting
- **Ruff**: Fast Python linter and formatter
- **Prettier**: Markdown, YAML, JSON formatting

### Code Quality

- **Ruff**: Comprehensive Python linting
- **Bandit**: Security vulnerability scanner
- **MyPy**: Static type checking (optional)

### Documentation

- **Pydocstyle**: Docstring style checking (Google convention)

### General

- **Trailing whitespace removal**
- **End-of-file fixing**
- **Large file detection**
- **Merge conflict detection**
- **YAML/TOML/JSON validation**

### Jupyter Notebooks

- **nbstripout**: Remove notebook outputs
- **nbqa**: Apply formatters to notebooks

## Configuration

Pre-commit configuration is in `.pre-commit-config.yaml`.

Tool-specific configurations are in `pyproject.toml`:

- `[tool.black]`
- `[tool.isort]`
- `[tool.ruff]`
- `[tool.bandit]`
- `[tool.pydocstyle]`
- `[tool.mypy]`

## Bypassing Hooks

In emergency situations, you can bypass pre-commit:

```bash
git commit --no-verify -m "Emergency fix"
```

**Note**: This should be used sparingly and issues should be fixed in follow-up commits.

## Troubleshooting

### Hook Installation Issues

```bash
# Reinstall hooks
uv run pre-commit clean
uv run pre-commit install
```

### Update Hook Versions

```bash
uv run pre-commit autoupdate
```

### Skip Specific Hooks

```bash
SKIP=mypy git commit -m "Skip MyPy for this commit"
```

## IDE Integration

Most IDEs can be configured to run these tools automatically:

### VS Code

Install extensions:

- Python
- Black Formatter
- isort
- Ruff
- Prettier

### PyCharm

Enable:

- Black integration
- isort integration
- Pre-commit integration plugin

## Makefile Targets

The following Makefile targets are available for code quality:

```bash
make format          # Format code with ruff
make lint            # Lint code with ruff
make check           # Run linting and formatting checks
make fix             # Auto-fix issues
make pre-commit-install  # Install pre-commit hooks
make pre-commit-run     # Run on staged files
make pre-commit-all     # Run on all files
```

## CI/CD Integration

Pre-commit runs automatically on GitHub Actions and other CI platforms. Some intensive hooks (like MyPy) are skipped on CI for performance.
