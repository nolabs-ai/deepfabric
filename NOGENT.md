# DeepFabric Code Review Guide

Use this file as the baseline for reviewing changes to DeepFabric.

**This repository is DeepFabric, a Python library and CLI, not nono.** Ignore
any Rust or sandbox-specific framing in your base instructions. Apply the
general review guidance (correctness, negative space, tests) to Python code
and to the domain described below.

DeepFabric generates synthetic training datasets with LLMs. A run builds a
topic tree or graph, allocates an exact number of samples across the topics,
generates each sample with an LLM, and saves the dataset. It can then upload
the dataset to Hugging Face, Kaggle or DeepFabric Cloud. Users pay for every
LLM call, and runs can take hours. So silent data loss, wasted calls and
wrong sample counts are serious bugs, not nits.

## Review Stance

- Findings lead the review. For each one, give the concrete inputs or sequence
  that triggers it, what goes wrong, and the fix.
- Prefer behavior bugs over style. Ruff handles formatting and import order, so
  don't comment on either.
- Look for what is missing. When a change handles a case in one place, check
  the sibling paths:
  - tree and graph mode;
  - every topic sampling mode;
  - the YAML config and Python API layers;
  - sync and async clients;
  - interactive and non-interactive CLI runs;
  - fresh, resumed and retried runs.
- Treat LLM output, YAML config, prompt templates, topic files, checkpoint
  files and dataset files on disk as untrusted input.
- Pre-existing problems unrelated to the PR are out of scope unless severe.

## Workspace Map

- `deepfabric/cli.py`: the Click CLI (`generate`, `upload-*`, `import-tools`
  and so on). It owns user prompts, output collision handling, checkpoint
  resume and retry flows, and exit codes.
- `deepfabric/config.py`, `config_manager.py`: the YAML config models
  (Pydantic) and loading, which must use `yaml.safe_load`.
- `deepfabric/tree.py`, `graph.py`, `topic_model.py`, `topic_manager.py`,
  `prompts.py`: topic generation and the topic file formats (JSONL tree, JSON
  graph).
- `deepfabric/generator.py`, `builders.py`, `builders_agent.py`,
  `sampling.py`, `checkpoint.py`: sample allocation, generation, conversation
  and agent builders, and per-slot checkpoints.
- `deepfabric/dataset.py`, `dataset_manager.py`, `hf_hub.py`, `kaggle_hub.py`,
  `cloud_upload.py`: saving, exporting and uploading datasets.
- `deepfabric/llm/`: the provider client (via `outlines`), retries, rate-limit
  detection and API key checks.
- `deepfabric/tools/`, `spin/`: tool definitions, the MCP client (stdio
  subprocess or HTTP) and the Spin tool-execution client.
- `deepfabric/evaluation/`, `training/`: model evaluation and training
  callbacks.
- `deepfabric/metrics.py`, `update_checker.py`: telemetry (PostHog) and update
  checks.
- `docs/` (MkDocs, see `mkdocs.yml`), `examples/`, `tests/unit/`,
  `tests/integration/`.
- `notebooks/`, `misc/`, `test-run/` and `coverage.xml` are low priority.
  Only flag security problems there.

## Core Invariants

### Topic Path Context

- Dataset generation must pass the **full hierarchical path** (root → … →
  topic) to prompt building, never the topic text alone. This includes
  interior graph nodes, which are generation targets in their own right.
- `get_path_by_id(uuid)` answers "what is this node's ancestry?". It resolves
  any node, including interior ones, by following first parents.
  `get_all_paths_with_ids()` lists root-to-leaf routes and cannot resolve
  interior nodes. Flag code that uses one where the other is needed.
- Any mode, builder or template that leaves the topic path out of a
  generation prompt needs an explicit, documented reason.

### Exact Sample Counts

- A run must produce exactly the requested number of samples, or report
  exactly which slots failed. Every slot ends as one success or one failure,
  never zero and never two.
- Check allocation, top-ups, retries, deduplication and resume for off-by-one
  errors, double counting and slots that are silently dropped.
- Watch for duplicate rows after a resume or retry, and for failures that
  aren't recorded.

### Checkpoints, Resume and Retry

- An incompatible or unreadable checkpoint must never start a fresh run
  without the user choosing to. Resuming must not repeat completed work or
  lose failed slots.
- Check what happens when the process is interrupted or killed between any
  two writes. Checkpoint state must stay consistent and loadable.
- Metadata that resume compares, such as the plan, template manifest or
  config fingerprint, must change only when behavior changes.
- Writes made on every response or every slot must not grow with run size.
  Flag work that rewrites whole files per LLM call, or that runs under the
  checkpoint lock or on the event loop.

### Output File Safety

- Never overwrite or delete a user's existing dataset, topic file or other
  output unless the user explicitly chose to. Outputs are written atomically,
  through the existing save helpers.
- A failed, stopped or partial run must not replace a complete output.
- A retry or resume may replace only the file that this run wrote.
- Check symlinks, directories at the output path and numbered collision names
  (`dataset_1.jsonl`).

### Non-interactive Runs

- The CLI runs in CI and scripts with stdin closed. Every prompt needs a safe
  default that works without a terminal; it must not crash with `Abort`, hang
  or perform an upload.
- Exit codes must reflect the outcome. A run whose outputs were saved should
  not exit non-zero, and a run that wrote nothing should not exit 0 as if it
  succeeded.

### LLM Responses

- Treat model output as untrusted and malformed by default. Parse it with the
  existing JSON extraction, validation and retry logic, and never with `eval`.
- Provider differences matter: OpenAI, Anthropic, Gemini, Ollama and
  OpenRouter differ in structured output support, schema restrictions, rate
  limit signals and token usage reporting. Check that a change works with all
  the providers it claims to support.
- Token usage and request counts must be accurate. Flag local errors counted
  as provider requests, and usage that is dropped or counted twice.

## Security Requirements

- **Secrets:** API keys and tokens (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GEMINI_API_KEY`, `HF_TOKEN`, `KAGGLE_KEY`, `DEEPFABRIC_API_KEY` and so on)
  must never appear in logs, errors, run reports, prompt traces, checkpoints,
  datasets, dataset cards or telemetry. Code that temporarily sets credentials
  in `os.environ` must restore the previous values.
- **Config and templates:** load YAML only with `yaml.safe_load`. User prompt
  templates are rendered by the single-pass `{{ name }}` substitution, and must
  never go through `str.format`, `string.Template`, Jinja or anything else that
  can evaluate expressions or reach attributes.
- **Code execution:** no `eval`, `exec`, `pickle` or `shell=True` on data from
  config, LLMs or downloaded files. The MCP stdio transport launches a
  user-configured command: pass arguments as a list and never build a shell
  string. Never let LLM output pick a command, path or URL.
- **Paths:** config, topic files, checkpoints and LLM output must not be able to
  write outside the intended output location (path traversal, absolute paths,
  `..`).
- **Network:** telemetry, update checks and cloud upload must stay opt-in or
  gated as documented. They must fail quietly without breaking a run, and must
  not send dataset contents or secrets unless the user asked.
- **Dependencies:** new dependencies need a clear reason. Flag heavy or
  unmaintained packages, and anything added to core dependencies that belongs
  in an extra.

## Python Coding Guidelines

- Python `>=3.10` (`requires-python`, ruff `target-version = "py310"`). Flag
  syntax or stdlib APIs that need 3.11 or later.
- Imports belong at the top of the file only, with no inline imports. The
  existing `# noqa: PLC0415` lazy imports for optional heavy dependencies are
  the exception.
- Use the `DeepFabricError` hierarchy in `exceptions.py` and preserve the cause
  (`raise ... from exc`). Flag bare `except:`, broad `except Exception` that
  swallows errors, and fallbacks that hide malformed config or output.
- Validate config with Pydantic models at load time. When the YAML config
  (`DeepFabricConfig`) and the Python API (`DataSetGeneratorConfig`) validate
  the same option, both layers must agree.
- In async code, flag blocking I/O or long CPU work on the event loop, locks
  held across `await`, and shared state changed outside its coordinator.
- Use Google-style docstrings for public functions. Comments should explain
  invariants and non-obvious reasons, not restate the code.
- Line length is 100. Formatting and lint are enforced by `make format` and
  `make lint`, so don't comment on them.

## Tests and Docs

- Every bug fix needs a regression test that fails without the fix. Every new
  option, mode or CLI flag needs tests for its edge cases and invalid input.
- Unit tests must run offline. Mock the LLM, not the code under test. They
  must not read or write the real `~/.deepfabric`, `HOME` config, the network
  or the current directory. Tests that change environment variables or the
  working directory must restore them.
- Integration tests in `tests/integration/` call real providers and need keys.
  They are run selectively, so behavior must not be covered only there.
- CI runs `make format`, `make lint`, `make test-unit`, `make security`
  (bandit) and `make build` on Python 3.11.
- User-visible changes need docs. CLI flags go in `docs/cli/`, config options
  and generation behavior in `docs/dataset-generation/`, plus any affected
  `examples/`. New or removed Markdown pages must be added to or removed from
  `mkdocs.yml`. Flag docs that contradict the code.

## Severity

- **high:** loss of a user's data or output, a leaked secret, code execution
  from untrusted input, a crash in a normal run, or a run that silently
  produces the wrong number of samples or loses topic path context.
- **medium:** wasted paid LLM calls, a resume or retry that misbehaves, a
  non-interactive run that hangs or exits with the wrong code, a performance
  regression that grows with run size, or missing tests for new behavior.
- **low:** latent issues, docs mismatches, confusing errors and minor design
  problems.

## Review Questions

- Does this work in tree and graph mode, and in every topic sampling mode it
  touches?
- Does every generation prompt still get the full topic path?
- After a crash, Ctrl+C, resume or retry, are sample counts still exact, with
  no duplicate rows?
- Can this overwrite, replace or delete a file the user didn't ask to replace?
- What happens with stdin closed, in CI?
- Can this leak a secret into a log, report, trace, checkpoint or upload?
- Does any LLM output, config value or template reach a shell, `eval`, a file
  path or an unsafe renderer?
- Does the cost grow faster than the number of LLM calls?
- Is there a test that fails without this change, and do the docs match?
