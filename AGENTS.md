# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Overview

DeepFabric is a Python library for generating synthetic datasets using LLM providers. It consists of three main components that work together in a pipeline:

1. **Tree / Graph**: Generate hierarchical topic structures from a root prompt
2. **DataSetGenerator**: Creates training examples based on topics
3. **Dataset handling**: Stores and exports generated records

## Development Commands

### Package Management

- `uv sync --all-extras` - Install all dependencies including dev tools
- `make install` - Same as above via Makefile

### Code Quality

- `make format` - Format code with Ruff
- `make lint` - Run ruff linting checks
- `make security` - Run bandit security analysis on deepfabric/ directory

### Testing

- `make test-offline` - Run unit and component tests; this is the CI test target
- `make test-unit` - Run only `tests/unit/`
- `make test-component` - Run only `tests/component/`
- `make test-integration` - Run integration tests requiring external providers/services
- `uv run pytest tests/unit/ tests/component/` - Direct offline pytest execution
- Test files are grouped under `tests/unit/`, `tests/component/`, and `tests/integration/`
- GitHub PR integration checks run on Python 3.10, 3.11, and 3.12; the offline CI workflow runs on Python 3.11.

### Build and Release

- `make build` - Clean, test, and build package
- `make clean` - Remove build artifacts and cache files
- `make all` - Clean, install, format, lint, run offline/integration tests, scan security, and build

## Core Architecture

### Configuration System

- YAML-based configuration with direct system prompt specification
- CLI supports extensive parameter overrides

### Data Flow

1. Tree or Graph generates a topic hierarchy from the root prompt
2. DataSetGenerator uses topics to generate and validate conversations or custom-schema records
3. Generated records are collected in a Hugging Face dataset and exported by the dataset manager
4. Optional HuggingFace Hub upload with auto-generated dataset cards

### Key Classes and Their Relationships

- `DeepFabricConfig`: Loads and validates YAML and provides parameter dictionaries
- `TreeConfig`/`GraphConfig`/`DataSetGeneratorConfig`: Pydantic models for parameters
- `datasets.Dataset`: Generation result used by the generator and CLI exporters
- `deepfabric.Dataset`: Native container for loaded data, with JSONL import/export
- `HFUploader`: Manages Hugging Face Hub integration

## Important Implementation Details

### JSON Validation

The engine includes robust JSON parsing with regex extraction and retry logic for handling LLM response inconsistencies.

### Error Handling

- Max retries configurable for failed LLM calls
- Failed samples tracked separately from successful ones
- Comprehensive error reporting in final summary

### Topic Path Context in Generation

When a topic model is supplied, dataset generation must pass the **full hierarchical path** to the target topic (root → ... → topic), including when the target is an interior graph node. Exceptions are response generation in tool conversations and default bulk/seed completion, which rely on the generated or pinned user request/seed. For non-tool bulk/seed completion, include the path by setting `generation.completion_includes_topic_path: true`. This setting does not add the path to agent decision or final-response prompts. See [Sample Generation Modes](docs/dataset-generation/sampling-modes.md#tools-and-call-counts) for tool-conversation context.

Dataset generation from a Graph targets every non-root node, including interior nodes. Tree targets correspond to its recorded leaf paths.

Topic-based single-pass generation builds an ordered allocation plan from unique topics, assigns each topic its quota, and resolves its path through `get_path_by_id()`. In this mode, `output.batch_size` limits concurrently active topics. Python generation without a topic model instead uses sample batches.

Two distinct questions, two distinct methods - do not substitute one for the other:

- `Graph.get_path_by_id(uuid)` - "what is this node's ancestry?" Resolves any graph node, including interior nodes, by walking first parents toward the root. A node with several parents gets one canonical ancestry via its first parent (the primary expansion parent; cross-connections are appended after it). Tree inherits `TopicModel.get_path_by_id()`, which searches enumerated leaf paths.
- `get_all_paths_with_ids()` - "what are the root-to-leaf routes?" Enumerates complete routes, keyed by leaf UUID. It cannot answer ancestry for an interior node, and `Graph.get_path_by_id()` does not call it.

For generation stages that receive topic context, verify that the prompt builder receives the complete path, never the target topic text alone. Calls to `build_prompt()` must supply that path as `subtopics_list`. Preserve the completion exceptions above.

### System Message Control

For conversation output, Python's `sys_msg` parameter and YAML's `output.include_system_message` control whether exported conversations include a system message. Custom-schema output follows its configured schema instead.

## Configuration Patterns

### YAML Structure

```yaml
llm:
  provider: "ollama"
  model: "mistral:latest"
topics:
  prompt: "Topics for the dataset"
  mode: "tree"
  save_as: "topics.jsonl"
generation:
  system_prompt: "Generate helpful question-and-answer samples."
output:
  num_samples: 10
  save_as: "dataset.jsonl"
```

### Provider Configuration

Specify provider and model separately:

- YAML (`llm`): `provider: "ollama"`, `model: "mistral:latest"`
- Python (`DataSetGenerator`): `provider="ollama"`, `model_name="mistral:latest"`

For conversation generation, an omitted or empty `generation.tools` block counts as no tools. Agent mode requires configured tool components or custom tools; endpoint settings alone do not activate it. YAML and direct Python configuration must use the same rule.

## CLI Usage Patterns

The CLI supports both YAML configuration and parameter overrides:

```bash
deepfabric generate config.yaml --provider ollama --model mistral:latest --temperature 0.8
```

## Code Style Notes

- Package supports Python 3.10+; keep changes compatible with the Python 3.10–3.12 integration test matrix.
- Google-style docstrings preferred
- Place imports at the top of the file.

## Documentation Guidelines

These guidelines apply to user-facing documentation under `docs/`; this agent guide may include implementation guidance.

- Explain user-visible behavior, configuration, and outcomes; omit internal implementation details.
- Before adding an explanation, check whether it already exists elsewhere. Link to existing guidance instead of repeating it.
- Keep wording concise and read the surrounding section to check its flow after editing.
- When adding or removing Markdown files under `docs/`, check whether `mkdocs.yml` needs updating.
