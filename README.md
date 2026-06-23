<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-light.png" />
    <img alt="DeepFabric logo" src="./assets/logo-light.png" style="width:40%;max-width:40%;height:auto;display:block;margin:0 auto;" />
  </picture>
  <h3>Training Model Behavior in Agentic Systems</h3>

  <!-- CTA Buttons -->
  <p>
    <a href="https://github.com/nolabs-ai/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
      <img src="https://img.shields.io/badge/Contribute-Good%20First%20Issues-green?style=for-the-badge&logo=github" alt="Good First Issues"/>
    </a>
    &nbsp;
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/badge/Chat-Join%20Discord-7289da?style=for-the-badge&logo=discord&logoColor=white" alt="Join Discord"/>
    </a>
  </p>

  <!-- Badges -->
  <p>
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
    </a>
    <a href="https://github.com/nolabs-ai/deepfabric/actions/workflows/test.yml">
      <img src="https://github.com/nolabs-ai/deepfabric/actions/workflows/test.yml/badge.svg" alt="CI Status"/>
    </a>
    <a href="https://pypi.org/project/deepfabric/">
      <img src="https://img.shields.io/pypi/v/deepfabric.svg" alt="PyPI Version"/>
    </a>
    <a href="https://pepy.tech/project/deepfabric">
      <img src="https://static.pepy.tech/badge/deepfabric" alt="Downloads"/>
    </a>
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/discord/1384081906773131274?color=7289da&label=Discord&logo=discord&logoColor=white" alt="Discord"/>
    </a>
    <a href="https://www.reddit.com/r/deepfabric/">
      <img src="https://img.shields.io/badge/Reddit-r%2Fdeepfabric-FF4500?logo=reddit&logoColor=white" alt="Reddit"/>
    </a>
  </p>
</div>

**DeepFabric** generates synthetic training data for language models and agent evaluations. By combining reasoning traces with tool-calling patterns, it creates high-quality, domain-specific datasets that teach models to think, plan, and act effectively, call tools correctly, and conform to strict schema structures.

What sets DeepFabric apart from other dataset generation tools is its ability to ensure high diversity yet domain-anchored relevance through unique topic graph generation algorithms. This guides sample creation to cover all necessary subtopics while avoiding redundancy, which is where other tools often fall short, resulting in model overfit.

<img src="/assets/df-demo.gif" width="100%" height="100%"/>

Constrained decoding and response validation, along with real tool executions within isolated webassembly environments, ensure that generated samples strictly adhere to structured schema, variable constraints, and execution correctness, ensuring datasets have exact syntax and structure for use in model training pipelines. Tool definations can be either directly imported from MCP (Model Context Protocol) server schemas and automatically mocked, real life interfaces along with a standard set of common tools (`list_files()`, `'read_file()` etc)

Once your dataset is generated, it can be automatically uploaded to Hugging Face and directly imported into popular training frameworks like TRL, Unsloth, and Axolotl. 

Post-training, DeepFabric's built-in evaluation engine assesses model performance, whereby models prove their capabilities on unseen tasks derived from training splits—covering evaluation-only questions, answers, and tool traces.

## Quickstart

DeepFabric can be used in several ways, as a library, CLI tool, or via YAML configuration. Here's a quick example using the CLI:

```bash
pip install deepfabric
```

```bash
export OPENAI_API_KEY="your-api-key"

deepfabric generate \
  --topic-prompt "Python programming fundamentals" \
  --generation-system-prompt "You are a Python expert" \
  --mode graph \
  --depth 3 \
  --degree 3 \
  --num-samples 9 \
  --batch-size 3 \
  --provider openai \
  --model gpt-4o \
  --output-save-as dataset.jsonl
```

This generates a topic graph and creates 27 unique nodes, then generates 27 training samples saved to `dataset.jsonl`, giving you 100% topic coverage.

## Configuration

DeepFabric also uses YAML configuration with three main sections and optional shared LLM defaults

> [!NOTE]  
> The following uses mocked tool execution, so will require a runing Spin service, which we provide in a docker image:
```bash
docker run -d -p 3000:3000 ghcr.io/nolabs-ai/deepfabric/tools-sdk:latest`
```

Save the following as `config.yaml`:

```yaml
# Optional: Shared LLM defaults (inherited by topics and generation)
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7

# TOPICS: Generate the topic tree/graph
topics:
  prompt: "Building production-ready REST APIs with Python"
  mode: tree                    # tree | graph
  depth: 3
  degree: 3
  save_as: "topics.jsonl"
  # Optional: Override shared LLM settings
  llm:
    model: "gpt-4o-mini"        # Use cheaper model for topics

# GENERATION: Create training samples from topics
generation:
  system_prompt: |
    You are an expert Python backend developer specializing in REST API design.
    Create practical, production-ready code examples with clear explanations.
    Include error handling, type hints, and follow PEP 8 conventions.
    Use the following tools to read, write, and list files in the virtual filesystem:
    - read_file
    - write_file
    - list_files

  # Additional instructions for sample generation
  instructions: |
    Focus on real-world scenarios developers encounter daily when building REST APIs with Python.
    Include both happy path and edge case handling.
    Provide context on when and why to use specific patterns or libraries.
    Ensure code is modular, testable, and maintainable.

  # Agent mode is implicit when tools are configured
  conversation:
    type: cot      # basic | cot
    reasoning_style: agent      # freetext | agent (for cot)

  # Tool configuration (enables agent mode automatically)
  tools:
    spin_endpoint: "http://localhost:3000"  # Spin service for tool execution
    components:                 # Map component name to tool names
      builtin:                  # Routes to /vfs/execute
        - read_file
        - write_file
        - list_files
    max_per_query: 3            # Maximum tools per query
    max_agent_steps: 5          # Max ReAct reasoning iterations

  # Optional: Seed initial files into the spin before generation, used for tool calling
    scenario_seed:
      files:
        "Dockerfile": |
          FROM python:3.13
          WORKDIR /usr/local/app

          # Install the application dependencies
          COPY requirements.txt ./
          RUN pip install --no-cache-dir -r requirements.txt

          # Copy in the source code
          COPY src ./src
          EXPOSE 8080

          # Setup an app user so the container doesn't run as the root user
          RUN useradd app
          USER app

          CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
        "main.py": |
          def greet(name):
              return f"Hello, {name}!"

          if __name__ == "__main__":
              print(greet("World"))
        "config.json": |
          {
            "version": "1.0.0",
            "debug": true,
            "max_retries": 3
          }

  # Generation control and retry settings
  max_retries: 3                # Retries for failed generations
  sample_retries: 2             # Retries for validation failures
  max_tokens: 2000              # Max tokens per generation

  # Optional: Override shared LLM settings
  llm:
    temperature: 0.3            # Lower temp for consistent code

# OUTPUT: Final dataset configuration
output:
  # System prompt that goes INTO the training data
  # This is what the trained model will see as its system message
  system_prompt: |
    You are a helpful Python programming assistant specialized in REST API
    development. You provide clear, production-ready code with explanations.
    Always consider security, error handling, and best practices.

  include_system_message: true  # Whether to include system message in output
  num_samples: 4                 # Total training samples to generate
  batch_size: 3                 # Parallel generation batch size
  save_as: "api-dataset.jsonl"

 Optional: Upload to Hugging Face
 huggingface:
   repository: "your-username/api-dataset-training-name"
   tags: ["python", "programming"]
```

Run generation by sourcing the `config.yaml`:

```bash
deepfabric generate config.yaml
```

## Generate, Train, Evaluate

DeepFabric returns standard HuggingFace datasets, making it easy to integrate with any training framework.

### Colab Notebooks:

A quick way of seeing DeepFabric in action is via our notebooks in the [notebooks/](./notebooks/) folder or on Google Colab:

**Qwen4b Blender MCP**:

[![Qwen4b Blender MCP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EG1V40v5xkJKLf6Ra6W4378vYqlZNVWqb)

### 1. Generate Dataset

```bash
deepfabric generate config.yaml --output-save-as dataset.jsonl
```

Or upload to HuggingFace Hub:

```bash
deepfabric upload-hf dataset.jsonl --repo your-username/my-dataset
```

### 2. Load and Split for Training

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load from Hub
dataset = load_dataset("alwaysfurther/deepfabric-generic-tools", split="train")

# Split into train/eval
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds = splits["test"]

# Format using your tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def format_example(example):
    messages = [{k: v for k, v in msg.items() if v is not None}
                for msg in example["messages"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

formatted_train = train_ds.map(format_example)
```

### 3. Train with TRL or Unsloth

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    args=SFTConfig(output_dir="./output", num_train_epochs=3),
)
trainer.train()
```

### 4. Evaluate Your Model

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    inference_config=InferenceConfig(
        model_path="./output/checkpoint-final",  # Local path or HF Hub ID
        backend="transformers",
    ),
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_ds)  # Pass HF Dataset directly

print(f"Tool Selection Accuracy: {results.metrics.tool_selection_accuracy:.2%}")
print(f"Parameter Accuracy: {results.metrics.parameter_accuracy:.2%}")
print(f"Overall Score: {results.metrics.overall_score:.2%}")
```

## Evaluation

DeepFabric provides a comprehensive evaluation system to measure how well your fine-tuned models perform on tool-calling tasks.

### Basic Evaluation

```python
from datasets import load_dataset
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

# Load your evaluation dataset
dataset = load_dataset("your-username/your-dataset", split="test")

# Configure the evaluator
config = EvaluatorConfig(
    inference_config=InferenceConfig(
        model_path="./output/checkpoint-final",  # Local path or HF Hub ID
        backend="transformers",                   # "transformers" or "ollama"
        temperature=0.1,                          # Low temp for deterministic outputs
        max_tokens=2048,
    ),
    max_samples=100,           # Limit samples for quick testing (None for all)
    save_predictions=True,     # Save individual predictions
    output_path="eval_results.json",
)

# Run evaluation
evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=dataset)

# Print summary
evaluator.print_summary(results.metrics)

# Cleanup GPU memory
evaluator.cleanup()
```

### Evaluation with LoRA Adapters

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    inference_config=InferenceConfig(
        model_path="Qwen/Qwen2.5-7B-Instruct",    # Base model
        adapter_path="./output/lora-adapter",     # LoRA adapter path
        backend="transformers",
        load_in_4bit=True,     # 4-bit quantization
        max_seq_length=2048,
    ),
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_dataset)
```

### Understanding Evaluation Metrics

The evaluator computes several metrics for tool-calling tasks:

```python
results = evaluator.evaluate(dataset=eval_dataset)
metrics = results.metrics

# Core metrics
print(f"Samples Evaluated: {metrics.samples_evaluated}")
print(f"Samples Processed: {metrics.samples_processed}")
print(f"Processing Errors: {metrics.processing_errors}")

# Tool-calling metrics
print(f"Tool Selection Accuracy: {metrics.tool_selection_accuracy:.2%}")
print(f"Parameter Accuracy: {metrics.parameter_accuracy:.2%}")
print(f"Execution Success Rate: {metrics.execution_success_rate:.2%}")
print(f"Response Quality: {metrics.response_quality:.2%}")
print(f"Overall Score: {metrics.overall_score:.2%}")
```

| Metric | Description |
|--------|-------------|
| `tool_selection_accuracy` | How often the model selects the correct tool |
| `parameter_accuracy` | How often tool parameters match expected values |
| `execution_success_rate` | Rate of valid, executable tool calls |
| `response_quality` | Quality score for non-tool responses |
| `overall_score` | Weighted combination of all metrics |

### Accessing Individual Predictions

```python
results = evaluator.evaluate(dataset=eval_dataset)

# Iterate through individual sample evaluations
for pred in results.predictions:
    print(f"Sample {pred.sample_id}:")
    print(f"  Query: {pred.query}")
    print(f"  Expected Tool: {pred.expected_tool}")
    print(f"  Predicted Tool: {pred.predicted_tool}")
    print(f"  Tool Correct: {pred.tool_selection_correct}")
    print(f"  Params Correct: {pred.parameters_correct}")
    if pred.error:
        print(f"  Error: {pred.error}")
```

### Evaluation from JSONL File

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    dataset_path="eval_dataset.jsonl",  # Load from file instead
    inference_config=InferenceConfig(
        model_path="./my-model",
        backend="transformers",
    ),
    output_path="results.json",
)

evaluator = Evaluator(config)
results = evaluator.evaluate()  # No dataset argument needed
```

### Using Ollama Backend

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    inference_config=InferenceConfig(
        model_path="llama3.2:latest",  # Ollama model name
        backend="ollama",
        temperature=0.1,
    ),
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_dataset)
```

## Providers

| Provider | Local/Cloud | Best For |
|----------|-------------|----------|
| OpenAI | Cloud | High quality, complex tasks |
| Anthropic | Cloud | Nuanced reasoning |
| Google Gemini | Cloud | Cost-effective at scale |
| Ollama | Local | Privacy, unlimited generation |
| OpenRouter | Cloud | Flexible model choice |

## Tool Tracing with Spin

DeepFabric supports **real tool execution** during dataset generation using the [Spin Framework](https://www.fermyon.com/spin). Instead of simulating tool outputs, tools actually execute in isolated WebAssembly sandboxes, producing authentic training data.

### Why Real Execution Matters

Traditional synthetic data generators simulate tool outputs, which creates unrealistic training data:

```
# Simulated (problematic)
Agent: read_file("config.json")
Result: {"setting": "value"}  # LLM hallucinated this content
```

With Spin integration, tools execute against real state:

```
# Real execution (accurate)
Agent: read_file("config.json")
Result: FileNotFound  # Actual filesystem state
Agent: write_file("config.json", "{...}")
Result: Written 42 bytes  # Real operation
```

### ReAct-Style Execution

DeepFabric uses a ReAct (Reason-Act-Observe) loop for tool calling. The agent observes real results before deciding the next action:

```
Step 1: Agent thinks "I should check if config exists"
        -> Calls read_file("config.json")
        -> Observes: FileNotFound

Step 2: Agent thinks "Config doesn't exist, I'll create it"
        -> Calls write_file("config.json", content)
        -> Observes: Success
```

This produces training data where decisions are based on actual observations, not hallucinated assumptions.

### Configuration

Enable tool tracing in your YAML config:

```yaml
generation:
  conversation:
    type: cot
    reasoning_style: agent

  tools:
    spin_endpoint: "http://localhost:3000"  # Spin service URL
    available:                              # Filter to specific tools
      - read_file
      - write_file
      - list_files
    max_agent_steps: 5                      # Max ReAct iterations

    # Optional: Seed initial state for scenarios
    scenario_seed:
      files:
        "config.json": '{"debug": true}'
```

### Built-in VFS Tools

DeepFabric includes a virtual filesystem (VFS) component with these tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read content from a file |
| `write_file` | Write content to a file |
| `list_files` | List all files in the session |
| `delete_file` | Delete a file |

Each session gets an isolated filesystem - changes don't persist between samples.

### Running Spin Locally

```bash
cd tools-sdk
spin build
spin up
```

The Spin service runs at `http://localhost:3000` by default.

### Adding Custom Tools

You can extend DeepFabric with custom tools written in Python, JavaScript, Go, or Rust. See [tool-traces.md](./tool-traces.md) for detailed documentation on:

- Creating custom Spin components
- Tool definition schemas
- Multi-language examples
- Containerization and deployment

## Resources

- [Documentation](https://nolabs-ai.github.io/deepfabric/)
- [Examples](./examples/README.md)
- [Tool Tracing Guide](./tool-traces.md)
- [Discord](https://discord.gg/pPcjYzGvbS)
- [Issues](https://github.com/nolabs-ai/deepfabric/issues)

## Development

```bash
git clone https://github.com/nolabs-ai/deepfabric
cd deepfabric
uv sync --all-extras
make test
```

## Analytics

We collect anonymous usage metrics to improve DeepFabric. No personal data, prompts, or API keys are collected.

```bash
# Disable analytics
export ANONYMIZED_TELEMETRY=False
```
