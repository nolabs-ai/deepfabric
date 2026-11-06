## Contributing

Thank you for considering contributing to [deepfabric](https://github.com/nolabs-ai/deepfabric)!

When contributing, please first discuss the change you wish to make via [issue](https://github.com/nolabs-ai/deepfabric/issues), [email](mailto:hello@alwaysfurther.ai), or any other method with the owners of this repository before making a change.

Note that we have a [Code of Conduct](./CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

## Setup

### Prerequisites

- **Python 3.10 or later**

## Development

### 1. Fork and clone the repository

Fork this repository and create your branch from `main`.

```sh
git clone https://github.com/<YOUR-USERNAME>/deepfabric
cd deepfabric
```

### 2. Create and activate a virtual environment

We strongly recommend using an isolated virtual environment.

#### Option A: Using `venv` (standard Python)

```sh
python -m venv .venv
source .venv/bin/activate
```

#### Option B: Using [`uv`](https://docs.astral.sh/uv/) (recommended)

DeepFabric supports running all development tasks via [`uv`](https://docs.astral.sh/uv/), which provides fast and reproducible environments without manual virtual environment activation.

### 3. Install the project and development dependencies

DeepFabric uses [PEP 621 (pyproject.toml)](https://peps.python.org/pep-0621/) with [Hatch](https://hatch.pypa.io/latest/) as the build backend.

<details>
<summary>Option A: Standard Python (`venv` + `pip`)</summary>

```sh
# Make sure your virtual environment is activated.
pip install -U pip
pip install -e '.[dev]'
```
</details>

<details>
<summary>Option B: Using `uv` (recommended)</summary>

```sh
uv sync --extra dev
```
</details>

This installs:
- test dependencies (`pytest`, `pytest-cov`, etc.)
- linting and security tools (`ruff`, `bandit`)

### 4. Run the test suite

#### Test categories

| Directory | Scope | External services |
| --- | --- | --- |
| `tests/unit/` | Focused behavior of a function or class, such as validation, schema conversion, or prompt formatting. | Mock external I/O. |
| `tests/component/` | Internal workflows with several real components working together, such as schedulers, builders, checkpoint storage, and event consumers. | Mock provider, Spin, and other external-service I/O; no live credentials required. |
| `tests/integration/` | Compatibility with real external systems, such as LLM providers, Spin, and dataset hubs. | May require credentials, endpoints, and network access. |

Offline acceptance tests of generation workflows belong in `tests/component/`.

#### Offline tests (required)

Run **both unit and component tests** before submitting changes. CI and the build
target run this combined offline suite.

<details>
<summary>Option A: Standard Python (`venv` + `pip`)</summary>

```sh
pytest tests/unit/ tests/component/
```
</details>

<details>
<summary>Option B: Using `uv` (recommended)</summary>

```sh
uv run pytest tests/unit/ tests/component/
```
</details>

The equivalent Makefile target is `make test-offline`. For a focused run, use
`make test-unit` or `make test-component`.

#### Integration tests (optional, but recommended)

See the [Makefile](./Makefile) for integration test targets, including provider-specific runs.

#### Security checks (optional)

We recommend running security checks before submitting larger or user-facing changes.

<details>
<summary>Option A: Standard Python (`venv` + `pip`)</summary>

```sh
# Make sure your virtual environment is activated.
bandit -r deepfabric/
```
</details>

<details>
<summary>Option B: Using `uv` (recommended)</summary>

```sh
uv run bandit -r deepfabric/
```
</details>

### 5. Lint and format the code

Ensure the codebase is properly linted and formatted before submitting changes.

<details>
<summary>Option A: Standard Python (`venv` + `pip`)</summary>

```sh
# Make sure your virtual environment is activated.
ruff format deepfabric/ tests/
ruff check . --exclude notebooks/
```
</details>

<details>
<summary>Option B: Using `uv` (recommended)</summary>

```sh
uv run ruff format deepfabric/ tests/
uv run ruff check . --exclude notebooks/
```
</details>

### 6. Commit your changes

We strongly recommend following the [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/) when committing:

- **feat:** new features
- **fix:** bug fixes
- **docs:** documentation-only changes
- **refactor:** code refactoring
- **test:** adding or updating tests
- **chore:** tooling and maintenance

Example:

```sh
git commit -m "feat: add dataset validation pipeline"
```

## Documentation

### 1. Install documentation dependencies

DeepFabric uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) for documentation.

<details>
<summary>Option A: Standard Python (`venv` + `pip`)</summary>

```sh
# Make sure your virtual environment is activated.
pip install -U pip
pip install -e '.[docs]'
```
</details>

<details>
<summary>Option B: Using `uv` (recommended)</summary>

```sh
uv sync --extra docs
```
</details>

This installs:
- MkDocs and the Material theme (`mkdocs-material`)
- Python API documentation support via `mkdocstrings[python]`

### 2. Writing documentation

- User-facing documentation lives in the [docs](./docs) directory.
- API documentation is generated from [Python docstrings](https://peps.python.org/pep-0257/) via `mkdocstrings`.
- Please follow the existing style and structure when adding new pages.

### 3. Docstring conventions

DeepFabric enforces [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), consistent with the `ruff` configuration:

```toml
[tool.ruff.lint.pydocstyle]
convention = "google"
```

Example:

```python
def load_dataset(path: str) -> Dataset:
    """Load a dataset from disk.

    Args:
        path: Path to the dataset directory.

    Returns:
        The loaded dataset.
    """
    ...
```

### 4. Serve the documentation locally

Ensure you can preview the documentation site locally before submitting changes.

<details>
<summary>Option A: Standard Python (`venv` + `pip`)</summary>

```sh
# Make sure your virtual environment is activated.
mkdocs serve
```
</details>

<details>
<summary>Option B: Using `uv` (recommended)</summary>

```sh
uv run mkdocs serve
```
</details>

### 5. Build the documentation

Ensure you generate the static documentation site before publishing or submitting changes.

<details>
<summary>Option A: Standard Python (`venv` + `pip`)</summary>

```sh
# Make sure your virtual environment is activated.
mkdocs build
```
</details>

<details>
<summary>Option B: Using `uv` (recommended)</summary>

```sh
uv run mkdocs build
```
</details>
