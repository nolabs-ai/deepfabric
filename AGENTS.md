# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Overview

DeepFabric is a Python library for generating synthetic datasets using LLM providers. It consists of three main components that work together in a pipeline:

1. **Tree**: Generates hierarchical topic structures from a root prompt
2. **DataSetGenerator**: Creates training examples based on topics 
3. **Dataset**: Manages and exports the final dataset

## Development Commands

### Package Management
- `uv sync --all-extras` - Install all dependencies including dev tools
- `make install` - Same as above via Makefile

### Code Quality
- `make format` - Format code with black and ruff auto-fix
- `make lint` - Run ruff linting checks
- `make security` - Run bandit security analysis on deepfabric/ directory

### Testing
- `make test` - Run pytest test suite
- `uv run pytest` - Direct pytest execution
- Test files are in `tests/` directory

### Build and Release
- `make build` - Clean, test, and build package
- `make clean` - Remove build artifacts and cache files
- `make all` - Complete workflow: install, format, lint, test, security, build

## Core Architecture

### Configuration System
- YAML-based configuration with direct system prompt specification
- CLI supports extensive parameter overrides

### Data Flow
1. Tree generates topic hierarchy from root prompt
2. DataSetGenerator uses topics to create question/answer pairs
3. Dataset validates, stores, and exports training examples
4. Optional HuggingFace Hub upload with auto-generated dataset cards

### Key Classes and Their Relationships
- `DeepFabricConfig`: Loads YAML and provides argument objects
- `TreeArguments`/`DataSetGeneratorArguments`: Pydantic-style dataclasses for parameters
- `Dataset`: Handles JSONL import/export and validation
- `HFUploader`: Manages Hugging Face Hub integration

## Important Implementation Details

### JSON Validation
The engine includes robust JSON parsing with regex extraction and retry logic for handling LLM response inconsistencies.

### Error Handling
- Max retries configurable for failed LLM calls
- Failed samples tracked separately from successful ones
- Comprehensive error reporting in final summary

### Topic Path Context in Generation
Dataset generation must pass the **full hierarchical path** to the target topic (root → ... → topic) to `build_prompt()`, including when the target is an interior graph node. The full path provides essential context for the LLM to generate domain-specific samples.

Graph generation targets every non-root node, so interior nodes are generation targets in their own right, not merely waypoints on a route to a leaf.

Cycle-based generation resolves each unique topic through `TopicModel.get_path_by_id()`. Graph overrides this method to follow first-parent ancestry; Tree uses the inherited lookup over enumerated leaf paths.

Two distinct questions, two distinct methods - do not substitute one for the other:

- `get_path_by_id(uuid)` - "what is this node's ancestry?" Resolves any node, interior ones included, by walking first parents to the root. A node with several parents gets one canonical ancestry via its first parent (the primary expansion parent; cross-connections are appended after it).
- `get_all_paths_with_ids()` - "what are the root-to-leaf routes?" Enumerates complete routes, keyed by leaf UUID. It cannot answer ancestry for an interior node, and `Graph.get_path_by_id()` does not call it.

Verify that `subtopics_list` receives the complete path, never the target topic text alone when hierarchical context is available.

### System Message Control
The `sys_msg` parameter controls whether system messages are included in the final dataset format - this affects training data structure.

## Configuration Patterns

### YAML Structure
```yaml
dataset_system_prompt: "..."
topic_tree:
  args: {...}
  save_as: "file.jsonl"
data_engine:
  args: {...}
dataset:
  creation: {...}
  save_as: "file.jsonl"
huggingface: {...}  # optional
```

### Provider Configuration
Specify provider and model separately in config, or use combined format in code:
- Config: `provider: "ollama"`, `model: "mistral:latest"`
- Code: `model_name: "ollama/mistral:latest"`

## CLI Usage Patterns

The CLI supports both YAML configuration and parameter overrides:
```bash
deepfabric start config.yaml --model gpt-4 --temperature 0.8 --hf-repo user/dataset
```

## Code Style Notes
- Uses uv for dependency management
- Ruff for linting with extensive rule set
- Black for formatting
- Bandit for security analysis
- Python 3.11+ required
- Google-style docstrings preferred
- do not place imports anywhere but the top of the file
- When updating `docs/` documentation, if new Markdown files are added or removed, consider updating `mkdocs.yml`.