# Basic Datasets

Basic datasets generate simple question-answer pairs without reasoning traces or tool calls.

## When to Use

- General instruction-following tasks
- Domain-specific Q&A (e.g., customer support, FAQs)
- Models that don't need to show reasoning
- Quick dataset generation with minimal configuration

## Configuration

```yaml title="config.yaml"
topics:
  prompt: "Python programming fundamentals"
  mode: graph             # tree | graph
  depth: 2
  degree: 2

generation:
  system_prompt: "Generate clear, educational Q&A pairs."
  instructions: "Create diverse questions with detailed answers."

  conversation:
    type: basic

  llm:
    provider: "openai"
    model: "gpt-4o"

output:
  system_prompt: |
    You are a helpful assistant.
  num_samples: 2
  batch_size: 1
  save_as: "dataset.jsonl"
```

!!! note "Key Setting"
    The key setting is `conversation.type: basic`.

## Output Format

Basic datasets produce standard chat-format JSONL:

```json title="dataset.jsonl"
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What are Python's numeric data types?"
    },
    {
      "role": "assistant",
      "content": "Python has three built-in numeric types: integers (int), floating-point numbers (float), and complex numbers (complex)..."
    }
  ]
}
```

## CLI Usage

Generate a basic dataset from the command line:

```bash
deepfabric generate config.yaml
```

Or with inline options:

```bash title="CLI generation"
deepfabric generate \
  --topic-prompt "Machine learning basics" \
  --conversation-type basic \
  --num-samples 2 \
  --batch-size 1 \
  --provider openai \
  --model gpt-4o \
  --output-save-as ml-dataset.jsonl
```

## Tips

!!! tip "Topic Depth and Degree"
    Topic depth and degree control dataset diversity. A tree with `depth: 3` and `degree: 3` produces 27 unique paths (`3^3 = 27` leaf nodes).

!!! warning "System Prompt Confusion"
    System prompts differ between generation and output:

    - `generation.system_prompt` - Instructions for the LLM generating examples
    - `output.system_prompt` - The system message included in training data

!!! info "Sample Size and Generation Model"
    DeepFabric uses a cycle-based generation model:

    - **Unique topics**: One per leaf path (tree) or per non-root node (graph)
    - **Cycles**: Number of iterations through all unique topics
    - **Concurrency**: `batch_size` controls parallel LLM calls

    For example, with 4 unique topics and `num_samples: 10`:

    - Cycles needed: 3 (ceil(10/4))
    - Cycle 1: 4 samples, Cycle 2: 4 samples, Cycle 3: 2 samples (partial)

    **Special values for `num_samples`:**

    - `"auto"` - One sample per unique topic (one full cycle)
    - Percentage (e.g., `"50%"`, `"200%"`) - Sample count relative to the number of unique topics

## Topic Count and Cycles

When configuring topic generation with a tree or graph, the total number of unique topics is determined by the structure:

- **Tree**: Unique topics = degree^depth (each leaf node has a unique UUID)
- **Graph**: Unique topics = every non-root node, interior nodes included. Interior nodes are also generation targets, not just waypoints, so the count is typically larger than `degree^depth`.

For example, a tree with `depth: 2` and `degree: 2` yields 4 unique topics (`2^2 = 4`).

!!! note "Graph counts are structural, not formulaic"
    A graph's topic count depends on how many nodes the build actually produced. Expansion can fall short of the requested depth or degree, so treat `degree^depth` as a rough estimate and read the real count from `deepfabric topic inspect`.

!!! info "Cycle-Based Generation"
    When `num_samples` exceeds the number of unique topics, DeepFabric iterates through multiple **cycles**:

    - Each unique topic (identified by UUID) is processed once per cycle
    - Cycles continue until `num_samples` is reached
    - The final cycle may be partial (fewer topics than a full cycle)

    For example, with 4 unique topics and `num_samples: 10`:

    - Cycle 1: Topics 1-4 (4 samples)
    - Cycle 2: Topics 1-4 (4 samples)
    - Cycle 3: Topics 1-2 (2 samples, partial)

    Checkpoints track progress as `(topic_uuid, cycle)` tuples, allowing precise resume from any point in any cycle.

!!! tip "Graphs with Shared Nodes"
    In graph mode, generation iterates over **unique topics (nodes)**, not over root-to-leaf routes. Each unique topic produces exactly one sample per complete cycle, no matter how many routes pass through it; the final cycle may be partial and cover only some topics.

    Each topic's prompt receives its full ancestry (root → ... → topic). When a node has several parents, its **first parent** supplies that ancestry — the primary expansion parent, since cross-connections are appended after it. Generating separate samples from a node's alternative routes is out of scope here.
