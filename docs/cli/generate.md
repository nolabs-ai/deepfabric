# generate

The `generate` command executes the complete synthetic data generation pipeline from YAML configuration to finished dataset. This command represents the primary interface for transforming domain concepts into structured training data through topic modeling and content generation.

```mermaid
graph LR
    A[YAML Config] --> B[Topic Generation]
    B --> C[Dataset Creation]
    C --> D[Output Files]
```

The generation process operates through multiple stages that can be monitored in real-time, providing visibility into topic expansion, content creation, and quality control measures.

## Basic Usage

Generate a complete dataset from a configuration file:

```bash title="Basic generation"
deepfabric generate config.yaml
```

This command reads your configuration, generates the topic structure, creates training examples, and saves all outputs to the specified locations.

## Configuration Override

Override specific configuration parameters without modifying the configuration file:

```bash title="Override parameters"
deepfabric generate config.yaml \
  --provider anthropic \
  --model claude-sonnet-4-5 \
  --temperature 0.8 \
  --num-samples 100 \
  --batch-size 5
```

!!! tip "Experimentation"
    Configuration overrides apply to all stages, enabling experimentation with different settings while maintaining the base configuration.

## File Management Options

Control where intermediate and final outputs are saved:

```bash title="Custom output paths"
deepfabric generate config.yaml \
  --topics-save-as custom_topics.jsonl \
  --output-save-as custom_dataset.jsonl
```

!!! note "Output Organization"
    These options override the file paths specified in your configuration, useful for organizing outputs by experiment or preventing accidental overwrites.

## Loading Existing Topic Structures

Skip topic generation by loading previously generated topic trees or graphs:

```bash title="Load existing topics"
deepfabric generate config.yaml --topics-load existing_topics.jsonl
```

!!! tip "Faster Iteration"
    This approach accelerates iteration when experimenting with dataset generation parameters while keeping the topic structure constant.

## Topic-Only Generation

Generate and save only the topic structure without proceeding to dataset creation:

=== "Graph Mode (Default)"

    ```bash
    deepfabric generate config.yaml --topic-only
    ```

=== "Tree Mode"

    ```bash
    deepfabric generate config.yaml --mode tree --topic-only
    ```

The `--topic-only` flag stops the pipeline after topic generation and saves the topic structure to the configured location.

## Topic Modeling Parameters

Fine-tune topic generation behavior through command-line parameters:

```bash title="Topic parameters"
deepfabric generate config.yaml \
  --degree 5 \
  --depth 4 \
  --temperature 0.7
```

| Parameter | Effect |
|-----------|--------|
| `--degree` | More subtopics per node (broader exploration) |
| `--depth` | More levels of detail (deeper exploration) |
| `--temperature` | Higher values create more diverse topics |

## Dataset Generation Controls

Adjust dataset creation parameters for different scales and quality requirements:

```bash title="Generation controls"
deepfabric generate config.yaml \
  --num-samples 500 \
  --batch-size 10 \
  --no-system-message
```

| Parameter | Description |
|-----------|-------------|
| `--num-samples` | Total samples: integer, `auto`, or percentage (e.g., `50%`, `200%`) |
| `--batch-size` | Maximum concurrent topic tasks; samples within a topic are sequential |
| `--include-system-message` | Include system prompts in training examples |
| `--no-system-message` | Exclude system prompts from training examples |

!!! tip "Automatic Sample Count"
    Use `--num-samples auto` to allocate one sample per unique topic (tree leaf or non-root graph node). Use `--num-samples 50%` to select the existing ordered prefix, or `--num-samples 200%` for two slots per topic in one pass. Terminal failures can reduce successful output rows.

    Integer targets use the same allocation: `--num-samples 100` with 50 unique topics is equivalent to `--num-samples 200%`. Targets are never rounded to a batch multiple.

## Conversation Type Options

Control the type of conversations generated:

```bash title="Conversation options"
deepfabric generate config.yaml \
  --conversation-type cot \
  --reasoning-style freetext
```

| Option | Values | Description |
|--------|--------|-------------|
| `--conversation-type` | `basic`, `cot` | Base conversation type |
| `--reasoning-style` | `freetext`, `agent` | Reasoning style for cot |

!!! note "Agent Mode"
    Agent mode is automatically enabled when tools are configured. No explicit `--agent-mode` flag is required. The `--agent-mode` flag is deprecated.

## TUI Options

Control the terminal user interface:

=== "Rich TUI"

    ```bash
    deepfabric generate config.yaml --tui rich
    ```

    Two-pane interface with real-time progress (default).

=== "Simple Output"

    ```bash
    deepfabric generate config.yaml --tui simple
    ```

    Headless-friendly plain text output.

## Provider and Model Selection

Use different providers or models for different components:

```bash title="Provider selection"
deepfabric generate config.yaml \
  --provider openai \
  --model gpt-4 \
  --temperature 0.9
```

!!! info "Provider Scope"
    Provider changes apply to all components unless overridden in the configuration file.

## Complete Example

??? example "Comprehensive generation command"

    ```bash title="Full example"
    deepfabric generate research-dataset.yaml \
      --topics-save-as research_topics.jsonl \
      --output-save-as research_examples.jsonl \
      --provider anthropic \
      --model claude-sonnet-4-5 \
      --degree 4 \
      --depth 3 \
      --num-samples 200 \
      --batch-size 8 \
      --temperature 0.8 \
      --include-system-message
    ```

    This creates a research dataset with comprehensive topic coverage and high-quality content generation.

## Progress Monitoring

The generation process provides real-time feedback:

- :material-file-tree: Topic tree construction progress with node counts
- :material-percent: Dataset generation status with completion percentages
- :material-alert: Error reporting with retry attempts and failure categorization
- :material-chart-bar: Final statistics including success rates and output file locations

## Checkpoint and Resume

For long-running generation jobs, enable checkpointing to save progress and allow resuming after interruptions:

```bash title="Enable checkpointing"
deepfabric generate config.yaml --checkpoint-interval 500
```

If generation is interrupted, resume from the checkpoint:

```bash title="Resume from checkpoint"
deepfabric generate config.yaml --resume
```

To retry previously failed samples when resuming:

```bash title="Resume with retry"
deepfabric generate config.yaml --resume --retry-failed
```

| Parameter | Description |
|-----------|-------------|
| `--checkpoint-interval N` | Save checkpoint every N samples |
| `--checkpoint-path PATH` | Override checkpoint directory (rarely needed) |
| `--resume` | Resume from existing checkpoint |
| `--retry-failed` | When resuming, retry previously failed samples |

!!! info "Checkpoint Location"
    Checkpoints are stored in an XDG-compliant location: `~/.local/share/deepfabric/checkpoints/` (or `$XDG_DATA_HOME/deepfabric/checkpoints/`). Each config file gets its own subdirectory based on a hash of its path, so different projects don't interfere with each other.

!!! tip "Checkpoint Status"
    Use `deepfabric checkpoint-status config.yaml` to inspect checkpoint progress without resuming.

### How Checkpointing Works

Checkpointing saves progress at regular intervals:

1. **Samples file** - Incrementally appends successful samples (no write amplification)
2. **Failures file** - Records failed samples with error details
3. **Metadata file** - Tracks progress, processed paths, and generation config

On resume, the generator:

- Loads previously processed topic paths
- Skips already-completed work
- Continues from where it left off
- Optionally retries failed samples with `--retry-failed`

!!! info "Memory Optimization"
    Checkpointing periodically flushes samples to disk, reducing sample buffering during generation. Completion identities remain in memory, and final dataset assembly loads the saved rows.

### Migrating Older Checkpoints

Version 4 checkpoints require migration before resuming. See [Migrating Version 4 Checkpoints](../dataset-generation/configuration.md#migrating-version-4-checkpoints) for the helper command and required inputs.

### Graceful Stop

Press ++ctrl+c++ once during generation to request a graceful stop. The generator will:

1. Complete the current batch
2. Save a final checkpoint
3. Exit cleanly

The TUI status panel shows "Stopping: at next checkpoint" to confirm the stop is pending. Press ++ctrl+c++ a second time to force quit immediately (no checkpoint saved).

### Checkpoint Conflicts

If you start generation without `--resume` but a checkpoint already exists for that configuration, you'll be prompted:

```
⚠ Existing checkpoint found for this configuration

  1) Resume from checkpoint
  2) Clear checkpoint and start fresh
  3) Abort

Choose an option [1/2/3] (1):
```

This prevents accidentally overwriting or corrupting existing checkpoint data.

## Error Recovery

!!! warning "Partial Failures"
    When generation fails partway through, the system saves intermediate results where possible. Topic trees are saved incrementally, enabling recovery by loading partial results and continuing from the dataset generation stage. With checkpointing enabled, dataset generation progress is also preserved.

Generation errors and handled interruptions can also save partial dataset exports. See [Partial Results After Errors or Interruptions](../dataset-generation/configuration.md#partial-results-after-errors-or-interruptions) for filenames and recovery behavior.

??? tip "Optimizing Generation Performance"
    Balance `batch-size` with your API rate limits and system resources. Larger batches increase throughput but consume more memory and may trigger rate limiting. Start with smaller batches and increase based on your provider's capabilities.

## Output Validation

After generation completes, verify your outputs:

```bash title="Validation commands"
# Check dataset format
head -n 5 your_dataset.jsonl

# Validate JSON structure
python -m json.tool your_dataset.jsonl > /dev/null

# Count generated examples
wc -l your_dataset.jsonl
```

## Topic Inspection

After generating topics, explore and refine the structure before dataset generation:

```bash title="Inspect and prune workflow"
# Explore the topic structure
deepfabric topic inspect topics.json --level 1 --expand

# Preview what pruning would remove
deepfabric topic prune topics.json --level 2 --dry-run

# Prune and save to new file
deepfabric topic prune topics.json --level 2 -o pruned_topics.json
```

See [topic inspect](topic-inspect.md) and [topic prune](topic-prune.md) for full details.
