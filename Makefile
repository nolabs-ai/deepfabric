.PHONY: clean install format lint test-unit test-component test-offline
.PHONY: test-integration test-integration-verbose security build all
.PHONY: test-integration-openai test-integration-gemini test-integration-llm
.PHONY: test-integration-hubs test-integration-spin test-integration-quick
.PHONY: test-integration-graph test-integration-generator

# Base command for integration tests
PYTEST_INTEGRATION = uv run pytest tests/integration --tb=short -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

install:
	uv sync --all-extras

format: ## Format code with ruff (parallel)
	uv run ruff format deepfabric/ tests/

lint:
	uv run ruff check . --exclude notebooks/

test-unit:
	uv run pytest tests/unit/

test-component:
	uv run pytest tests/component/

test-offline:
	uv run pytest tests/unit/ tests/component/

test-integration:
	$(PYTEST_INTEGRATION) --maxfail=1

test-integration-verbose:
	$(PYTEST_INTEGRATION) -s -rA --durations=10 --tb=long

test-integration-openai:
	$(PYTEST_INTEGRATION) -m openai

test-integration-gemini:
	$(PYTEST_INTEGRATION) -m gemini

test-integration-llm:
	$(PYTEST_INTEGRATION) -m "openai or gemini"

test-integration-hubs:
	$(PYTEST_INTEGRATION) -m huggingface

test-integration-spin:
	$(PYTEST_INTEGRATION) -m spin

test-integration-quick:
	$(PYTEST_INTEGRATION) -m "not huggingface"

test-integration-graph:
	$(PYTEST_INTEGRATION) tests/integration/test_graph_integration.py

test-integration-generator:
	$(PYTEST_INTEGRATION) tests/integration/test_generator_integration.py

security:
	uv run bandit -r deepfabric/

build: clean test-offline
	uv build

all: clean install format lint test-offline test-integration security build
