# Configuration Reference

DeepFabric uses YAML configuration with four main sections: `llm`, `topics`, `generation`, and `output`.

## Complete Example

```yaml title="config.yaml"
# Shared LLM defaults (optional)
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7
  # gemini_safety_settings: [...]  # Gemini only — see Section Reference below

# Topic generation
topics:
  prompt: "Python programming fundamentals"
  mode: graph             # tree | graph
  prompt_style: anchored  # default | isolated | anchored (graph mode only)
  depth: 2
  degree: 3
  save_as: "topics.json"
  llm:                    # Override shared LLM
    model: "gpt-4o-mini"

# Sample generation
generation:
  system_prompt: |
    Generate clear, educational examples.
  instructions: "Create diverse, practical scenarios."

  # Agent mode is implicit when tools are configured
  conversation:
    type: cot
    reasoning_style: agent

  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:
        - read_file
        - write_file

  max_retries: 3
  llm:
    temperature: 0.5

# Output configuration
output:
  system_prompt: |
    You are a helpful assistant with tool access.
  include_system_message: true
  num_samples: 4
  batch_size: 2
  save_as: "dataset.jsonl"

  # Optional: Checkpoint for resumable generation
  checkpoint:
    interval: 500       # Save every 500 samples
    retry_failed: false

# Optional: Upload to HuggingFace
huggingface:
  repository: "org/dataset-name"
  tags: ["python", "agents"]
```

!!! note "HuggingFace Upload"
    The `huggingface` section is optional and used to upload the dataset after generation. It requires a token exported as `HF_TOKEN` or pre-authentication via `huggingface-cli`.

## Section Reference

### llm (Optional)

Shared LLM defaults inherited by `topics` and `generation`.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | string | LLM provider: openai, anthropic, gemini, ollama |
| `model` | string | Model name |
| `temperature` | float | Sampling temperature (0.0-2.0) |
| `base_url` | string | Custom API endpoint |
| `gemini_safety_settings` | list | Safety filter overrides for Gemini models (see below) |

#### Gemini Safety Settings

Gemini models apply content safety filters by default. For synthetic dataset generation — especially adversarial, security, or red-teaming datasets — these filters can block legitimate content. Use `gemini_safety_settings` to control the threshold for each category.

Each entry in the list requires two fields:

| Field | Values |
|-------|--------|
| `category` | `HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_HARASSMENT`, `HARM_CATEGORY_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_DANGEROUS_CONTENT`, `HARM_CATEGORY_CIVIC_INTEGRITY` |
| `threshold` | `OFF`, `BLOCK_NONE`, `BLOCK_ONLY_HIGH`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_LOW_AND_ABOVE` |

!!! info "Default behaviour"
    Gemini 2.5 and later models default to `OFF` for all categories. For earlier models the default is `BLOCK_MEDIUM_AND_ABOVE`. Setting `OFF` disables the filter entirely; `BLOCK_NONE` allows all content regardless of the model's safety probability score.

!!! warning "YAML quoting"
    YAML 1.1 (used by PyYAML) parses bare `OFF` and `ON` as booleans. DeepFabric handles this automatically, but if you prefer to be explicit you can quote the value: `threshold: "OFF"`.

```yaml title="Disable all safety filters"
llm:
  provider: gemini
  model: gemini-2.0-flash
  gemini_safety_settings:
    - category: HARM_CATEGORY_HATE_SPEECH
      threshold: OFF
    - category: HARM_CATEGORY_HARASSMENT
      threshold: OFF
    - category: HARM_CATEGORY_SEXUALLY_EXPLICIT
      threshold: OFF
    - category: HARM_CATEGORY_DANGEROUS_CONTENT
      threshold: OFF
    - category: HARM_CATEGORY_CIVIC_INTEGRITY
      threshold: OFF
```

```yaml title="Relax only the dangerous-content filter"
llm:
  provider: gemini
  model: gemini-2.0-flash
  gemini_safety_settings:
    - category: HARM_CATEGORY_DANGEROUS_CONTENT
      threshold: BLOCK_ONLY_HIGH
```

Safety settings can be placed on the top-level `llm` block (applies to both topics and generation) or on a section-specific `llm` block to target only that stage:

```yaml title="Per-section safety settings"
llm:
  provider: gemini
  model: gemini-2.0-flash

topics:
  prompt: "Adversarial AI attack scenarios"
  llm:
    gemini_safety_settings:
      - category: HARM_CATEGORY_DANGEROUS_CONTENT
        threshold: OFF

generation:
  system_prompt: "Generate adversarial test cases."
  llm:
    gemini_safety_settings:
      - category: HARM_CATEGORY_DANGEROUS_CONTENT
        threshold: OFF
      - category: HARM_CATEGORY_HARASSMENT
        threshold: OFF
```

### topics

Controls topic tree/graph generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Root topic for generation |
| `mode` | string | "graph" | Generation mode: tree or graph |
| `depth` | int | 2 | Hierarchy depth (1-10) |
| `degree` | int | 3 | Subtopics per node (1-50) |
| `max_concurrent` | int | 4 | Max concurrent LLM calls (graph mode only, 1-20) |
| `prompt_style` | string | "default" | Graph expansion prompt style (graph mode only, see below) |
| `system_prompt` | string | "" | Custom instructions for topic LLM |
| `max_tokens` | int | 1000 | Max tokens for topic generation LLM calls |
| `save_as` | string | - | Path to save topics JSONL |
| `llm` | object | - | Override shared LLM settings |

#### topics.prompt_style (Graph Mode Only)

Controls how subtopics are generated during graph expansion:

| Style | Cross-connections | Examples | Use Case |
|-------|-------------------|----------|----------|
| `default` | Yes | Generic | General-purpose topic graphs with cross-links |
| `isolated` | No | Generic | Independent subtopics without cross-connections |
| `anchored` | No | Domain-aware | Focused generation with domain-specific examples |

**`anchored`** is recommended for specialized domains (security, technical) where you want subtopics to stay tightly focused on the parent topic. It automatically detects the domain from your `system_prompt` and provides relevant examples to guide generation.

```yaml title="Example: Security-focused topic generation"
topics:
  prompt: "Credential access attack scenarios"
  mode: graph
  prompt_style: anchored   # Uses security-domain examples
  depth: 3
  degree: 8
  system_prompt: |
    Generate adversarial security test cases for AI assistant hardening.
```

#### Failed Generation Tracking

When topic generation encounters failures, sidecar files are created automatically:

- **Tree mode**: `{save_as}_failed.jsonl` alongside the JSONL output
- **Graph mode**: `{save_as}_failed.jsonl` alongside the JSON output

These files contain error details for each failed generation attempt, useful for debugging provider issues or prompt problems.

!!! tip "Truncation Detection"
    DeepFabric detects truncated LLM responses ("EOF while parsing" errors) and suggests increasing `max_tokens` in the topic configuration.

### generation

Controls sample generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string | - | Instructions for generation LLM |
| `instructions` | string | - | Additional guidance |
| `conversation` | object | - | Conversation type settings |
| `tools` | object | - | Tool configuration |
| `max_retries` | int | 3 | Retries on API failures |
| `sample_retries` | int | 2 | Retries on validation failures |
| `max_tokens` | int | 2000 | Max tokens per generation |
| `llm` | object | - | Override shared LLM settings |

#### generation.conversation

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `type` | string | basic, cot | Conversation format |
| `reasoning_style` | string | freetext, agent | For cot only |

!!! note "Agent Mode"
    Agent mode is automatically enabled when tools are configured. No explicit `agent_mode` setting is required.

#### generation.tools

| Field | Type | Description |
|-------|------|-------------|
| `spin_endpoint` | string | Spin service URL |
| `tools_endpoint` | string | Endpoint to load tool definitions (for non-builtin components) |
| `components` | object | Map of component name to tool names (see below) |
| `custom` | list | Inline tool definitions |
| `max_per_query` | int | Max tools per sample |
| `max_agent_steps` | int | Max ReAct iterations |
| `scenario_seed` | object | Initial file state |

##### components

The `components` field maps component names to lists of tool names. Each component routes to `/{component}/execute`:

```yaml title="Component routing"
components:
  builtin:              # Routes to /vfs/execute (built-in tools)
    - read_file
    - write_file
  mock:                 # Routes to /mock/execute
    - get_weather
  github:               # Routes to /github/execute
    - list_issues
```

!!! info "Component Types"
    - `builtin`: Uses built-in VFS tools (read_file, write_file, list_files, delete_file)
    - Other components: Load tool definitions from `tools_endpoint`

### output

Controls final dataset.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | string | - | System message in training data |
| `include_system_message` | bool | true | Include system message |
| `num_samples` | int \| string | required | Total samples: integer, `"auto"`, or percentage like `"50%"` |
| `batch_size` | int | 1 | Parallel generation concurrency (number of simultaneous LLM calls) |
| `save_as` | string | required | Output file path |
| `checkpoint` | object | - | Checkpoint configuration (see below) |

!!! tip "Auto and Percentage Samples"
    `num_samples` supports special values:

    - **Integer** (e.g., `100`): Generate exactly this many samples
    - **`"auto"`**: Generate one sample per unique topic (100% coverage)
    - **Percentage** (e.g., `"50%"`, `"200%"`): Generate samples relative to unique topic count

    When `num_samples` exceeds the number of unique topics, DeepFabric iterates through multiple **cycles**. Each cycle processes all unique topics once. For example, with 50 unique topics and `num_samples: 120`:

    - Cycle 1: 50 samples (topics 1-50)
    - Cycle 2: 50 samples (topics 1-50)
    - Cycle 3: 20 samples (topics 1-20, partial)

#### output.checkpoint (Optional)

Configuration for checkpoint-based resume capability. Checkpoints allow pausing and resuming long-running dataset generation without losing progress.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interval` | int | required | Save checkpoint every N samples |
| `path` | string | XDG data dir | Directory to store checkpoint files (auto-managed) |
| `retry_failed` | bool | false | When resuming, retry previously failed samples |

```yaml title="Checkpoint configuration"
output:
  save_as: "dataset.jsonl"
  num_samples: 5000
  batch_size: 5
  checkpoint:
    interval: 500     # Save every 500 samples
    retry_failed: false
```

Checkpointing creates three files in the checkpoint directory:

- `{name}.checkpoint.json` - Metadata (progress, IDs processed)
- `{name}.checkpoint.jsonl` - Samples saved so far
- `{name}.checkpoint.failures.jsonl` - Failed samples for debugging

!!! tip "Choosing Checkpoint Interval"
    The checkpoint `interval` specifies how many samples to generate between saves.

    Choose an interval that balances recovery granularity (smaller = less work lost) against I/O overhead (larger = fewer disk writes). For example, `interval: 100` saves progress every 100 samples.

!!! info "Cycle-Based Generation Model"
    DeepFabric uses a cycle-based generation model:

    - **Unique topics**: Deduplicated topics from your tree/graph (by UUID)
    - **Cycles**: Number of times to iterate through all unique topics
    - **Concurrency**: Maximum parallel LLM calls (`batch_size`)

    For example, with 100 unique topics and `num_samples: 250`:

    - Cycles needed: 3 (ceil(250/100))
    - Cycle 1: 100 samples, Cycle 2: 100 samples, Cycle 3: 50 samples (partial)

    Checkpoints track progress as `(topic_uuid, cycle)` tuples, enabling precise resume from any point.

!!! tip "Memory Optimization"
    When checkpointing is enabled, samples are flushed to disk periodically, keeping memory usage constant regardless of dataset size.

### huggingface (Optional)

| Field | Type | Description |
|-------|------|-------------|
| `repository` | string | HuggingFace repo (user/name) |
| `tags` | list | Dataset tags |

## CLI Overrides

Most config options can be overridden via CLI:

```bash title="CLI overrides"
deepfabric generate config.yaml \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --num-samples 100 \
  --batch-size 10 \
  --temperature 0.5
```

!!! tip "Full Options"
    Run `deepfabric generate --help` for all available options.
