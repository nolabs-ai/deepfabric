# Reasoning Datasets

Reasoning datasets include chain-of-thought traces that show how the model arrives at its answer.

## When to Use

- Training models to explain their thinking
- Math, logic, or multi-step problems
- Tasks requiring transparent decision-making
- Improving model reliability through explicit reasoning

## Configuration

```yaml title="config.yaml"
topics:
  prompt: "Mathematical problem solving"
  mode: graph             # tree | graph
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate problems with step-by-step solutions."
  instructions: "Show clear reasoning before the final answer."

  conversation:
    type: cot
    reasoning_style: freetext

  llm:
    provider: "openai"
    model: "gpt-4o"

output:
  system_prompt: |
    You are an AI that explains its reasoning step-by-step.
  num_samples: 7
  batch_size: 1
  save_as: "reasoning-dataset.jsonl"
```

!!! note "Key Settings"
    - `conversation.type: cot`
    - `conversation.reasoning_style: freetext`

## Output Format

Reasoning datasets include a `reasoning` field alongside messages:

??? example "Sample Output"

    ```json title="reasoning-dataset.jsonl"
    {
        "messages": [
            {
                "content": "You are an AI that explains its reasoning step-by-step.\n",
                "role": "system"
            },
            {
                "content": "Can you create a reasoning problem that involves mathematical problem-solving and logical reasoning?",
                "role": "user"
            },
            {
                "content": "Sure, let's dive into a problem that requires a bit of logical reasoning and mathematical problem-solving.\n\n**Problem:**\n\nImagine you have a basket containing a mix of apples and oranges. The total number of fruits in the basket is 30. If the number of apples is twice the number of oranges, how many apples and oranges are there in the basket?\n\n**Reasoning:**\n\nHmm, let's think about this. We know the total number of fruits is 30...",
                "role": "assistant"
            }
        ],
        "reasoning": {
            "content": "Hmm, let's think about this. We know the total number of fruits is 30. So, if we let the number of oranges be x, then the number of apples would be 2x...",
            "style": "freetext"
        },
        "question": "Can you create a reasoning problem that involves mathematical problem-solving and logical reasoning?",
        "final_answer": "There are 20 apples and 10 oranges in the basket."
    }
    ```

## Reasoning Styles

DeepFabric supports two reasoning styles:

| Style | Description |
|-------|-------------|
| `freetext` | Natural language explanation of thought process |
| `agent` | Structured reasoning steps; tools are enabled separately through `generation.tools` |

The `freetext` style produces readable, conversational reasoning traces.

## CLI Usage

```bash title="Generate reasoning dataset"
deepfabric generate \
  --topic-prompt "Logic puzzles" \
  --conversation-type cot \
  --reasoning-style freetext \
  --num-samples 1 \
  --batch-size 1 \
  --provider openai \
  --model gpt-4o \
  --output-save-as logic-dataset.jsonl
```

## Training Considerations

!!! tip "Reasoning Placement"
    The `reasoning` field is stored separately from messages. During training, you can:

    - Include reasoning in the assistant's response (visible to users)
    - Use it as auxiliary training signal only
    - Format it as a special token section (e.g., `<thinking>...</thinking>`)
