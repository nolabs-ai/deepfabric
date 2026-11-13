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

!!! info "Exact Sample Allocation"
    DeepFabric assigns every unique topic its full sample quota in one pass. For 10 requested samples and 4 topics, the quotas are **3, 3, 2, 2** in the existing topic order. If the target is smaller than the topic count, only the ordered prefix is selected.

    `batch_size` limits concurrent **topic tasks**. Each topic generates its samples one at a time, using independent requests without previous sibling samples as context. The next topic group starts after the current group settles. This scheduling does not guarantee content diversity.

    The target is never rounded to a batch multiple. All-success runs return exactly the target; terminal failures are reported separately and do not create replacement slots.

    In YAML configuration, `output.num_samples` also accepts these special values when using a topic model:

    - `"auto"`: One sample per actual unique topic.
    - Percentages such as `"50%"` or `"200%"`: Resolve against actual unique topics, rounding down with a minimum of one.

## Topic Count and Quotas

When configuring topic generation with a tree or graph, the total number of unique topics is determined by the structure:

- **Tree**: Unique topics = degree^depth (each leaf node has a unique UUID)
- **Graph**: Unique topics = every non-root node, interior nodes included. Interior nodes are also generation targets, not just waypoints, so the count is typically larger than `degree^depth`.

For example, a tree with `depth: 2` and `degree: 2` yields 4 unique topics (`2^2 = 4`).

!!! note "Graph counts are structural, not formulaic"
    A graph's topic count depends on how many nodes the build actually produced. Expansion can fall short of the requested depth or degree, so treat `degree^depth` as a rough estimate and read the real count from `deepfabric topic inspect`.

!!! info "Checkpointing Individual Slots"
    Checkpoints track `(topic_uuid, sample_ordinal)` identities, with zero-based ordinals within each topic. They can save progress partway through a topic's quota while other topic tasks continue. Resume preserves the original allocation and skips settled slots; `--retry-failed` reopens only failed slots.

!!! tip "Graphs with Shared Nodes"
    In graph mode, generation iterates over **unique topics (nodes)**, not over root-to-leaf routes. Each UUID receives one quota, regardless of how many routes pass through it. Different UUIDs with equal topic text remain distinct.

    Each topic's prompt receives its full ancestry (root → ... → topic). When a node has several parents, its **first parent** supplies that ancestry — the primary expansion parent, since cross-connections are appended after it. Generating separate samples from a node's alternative routes is out of scope here.
