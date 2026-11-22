# Dataset Generation

DeepFabric generates three types of synthetic training datasets, each designed for different model capabilities.

## Dataset Types

<div class="grid cards" markdown>

-   :material-chat-question-outline: **Basic**

    ---

    Simple Q&A pairs for general instruction following

    [:octicons-arrow-right-24: Learn more](basic.md)

-   :material-head-cog-outline: **Reasoning**

    ---

    Chain-of-thought traces for step-by-step problem solving

    [:octicons-arrow-right-24: Learn more](reasoning.md)

-   :material-robot-outline: **Agent**

    ---

    Tool-calling datasets with ReAct-style reasoning

    [:octicons-arrow-right-24: Learn more](agent.md)

</div>

## Generation Pipeline

All dataset types follow the same three-stage pipeline:

```mermaid
graph LR
    A[Topic Generation] --> B[Sample Generation] --> C[Output]
    A --> |"Tree or Graph"| A1[Subtopics]
    B --> |"Per Topic"| B1[Training Examples]
    C --> |"JSONL"| C1[HuggingFace Upload]
```

1. **Topic Generation** - Creates a tree or graph of subtopics from your root prompt
2. **Sample Generation** - Produces training examples for each topic
3. **Output** - Saves to JSONL with optional HuggingFace upload

## Quick Comparison

=== "Basic"

    ```yaml title="config.yaml"
    generation:
      conversation:
        type: basic
    ```

    Simple Q&A without explicit reasoning.

=== "Reasoning"

    ```yaml title="config.yaml"
    generation:
      conversation:
        type: cot
        reasoning_style: freetext
    ```

    Includes step-by-step reasoning traces.

=== "Agent"

    ```yaml title="config.yaml"
    generation:
      conversation:
        type: cot
        reasoning_style: agent
      tools:
        spin_endpoint: http://localhost:3000
        components:
          builtin: [read_file]
    ```

    Tool-calling with ReAct-style reasoning.

## Choosing a Dataset Type

!!! tip "Quick Selection Guide"
    - **Basic**: General instruction-following without explicit reasoning
    - **Reasoning**: Models that need to explain their logic
    - **Agent**: Tool-calling capabilities

**Basic datasets** work for general instruction-following tasks. The model learns to answer questions directly without explicit reasoning.

**Reasoning datasets** teach models to think before answering. The output includes a `reasoning` field with the model's thought process, useful for training models that explain their logic.

**Agent datasets** train tool-calling capabilities. When tools are configured, agent mode is automatically enabled and generates complete tool workflows with ReAct-style reasoning.

## Next Steps

- [Sample Generation Modes](sampling-modes.md) - Independent, sequential and bulk generation
- [Basic Datasets](basic.md) - Simple Q&A generation
- [Reasoning Datasets](reasoning.md) - Chain-of-thought training data
- [Agent Datasets](agent.md) - Tool-calling datasets
- [Configuration Reference](configuration.md) - Full YAML options
