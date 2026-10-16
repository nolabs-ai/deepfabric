# Configuration Reference

DeepFabric uses YAML configuration with four main sections: `llm`, `topics`, `generation`, and `output`.

## Complete Example

```yaml title="config.yaml"
# Shared LLM defaults (optional)
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7

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
| `output_schema` | object | - | Custom JSON Schema for structured output (see below) |
| `llm` | object | - | Override shared LLM settings |

#### generation.output_schema (Custom Structured Output)

By default, DeepFabric generates samples in a conversation format (`basic` or `cot`, made up of messages). Setting `output_schema` bypasses the conversation format entirely: DeepFabric generates directly into your JSON Schema via constrained decoding, and each output record matches that schema instead of the OpenAI messages format.

```yaml title="Example: Custom schema output"
generation:
  system_prompt: "Extract structured product information from the topic."
  instructions: "Populate every field with realistic, specific values."
  output_schema:
    type: object
    properties:
      product_name: {type: string}
      category: {type: string}
      price_usd: {type: number}
      in_stock: {type: boolean}
    required: [product_name, category, price_usd, in_stock]

output:
  format: custom   # documents intent; the switch is driven by output_schema
  num_samples: 100
  batch_size: 5
  save_as: "products.jsonl"
```

Set `output.format: custom` alongside it to document the intent in your config — see the `output` section below for how `format` is used.

!!! warning "Limitations"
    - **No field-level validation.** DeepFabric confirms the model's response is valid JSON but does not check it against `required`, types, `enum`, or nested `properties` in your schema. Malformed or incomplete-but-valid JSON will pass through into the dataset.
    - **No example demonstrations.** Unlike the standard conversation prompts, the custom-schema prompt does not support `generation.example_data`; example demonstrations have no effect in this mode.
    - **Constrained decoding depends on the provider.** OpenAI, OpenRouter, Ollama, and local providers enforce the schema via [Outlines](https://github.com/dottxt-ai/outlines); Gemini and Anthropic use their native structured-output APIs. Enforcement quality varies by provider and model.
    - **No CLI override.** `output_schema` can only be set via YAML config or the Python API (`DataSetGenerator(output_schema={...}, ...)`); there is no `--output-schema` flag.
    - **Retries follow the same validation-error heuristics** as normal generation (`sample_retries`). If a schema-related error isn't recognized as a validation error, it fails immediately instead of retrying.

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
| `format` | string | "messages" | `messages` (OpenAI chat format) or `custom` (matches `generation.output_schema`) |
| `checkpoint` | object | - | Checkpoint configuration (see below) |

!!! tip "Auto and Percentage Samples"
    `num_samples` supports the following values:

    - **Integer** (e.g., `100`): Generate exactly this many samples
    - **`"auto"`**: One sample per unique topic (one full cycle)
    - **Percentage** (e.g., `"50%"`, `"200%"`): Sample count relative to the number of unique topics

    For trees, only the final topic on each branch is counted. 
    
    For graphs, every non-root topic is counted, **includes intermediate topics**.
    Connecting a graph topic to additional parents does not increase the count.

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

    - **Unique topics**: One per leaf path (tree) or per non-root node (graph)
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
