import warnings

from typing import Literal

import yaml

from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_SAMPLE_RETRIES,
    ENGINE_DEFAULT_BATCH_SIZE,
    ENGINE_DEFAULT_NUM_EXAMPLES,
    ENGINE_DEFAULT_TEMPERATURE,
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .exceptions import ConfigurationError
from .metrics import trace
from .utils import parse_num_samples


def _normalize_reasoning_style(value: str | None) -> str | None:
    """Normalize reasoning_style with deprecation warnings for old values.

    Args:
        value: The reasoning_style value to normalize

    Returns:
        Normalized value ('freetext', 'agent', or None)
    """
    if value is None:
        return None
    if value == "structured":
        warnings.warn(
            "reasoning_style='structured' is deprecated. Use 'agent' instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        return "agent"
    if value == "hybrid":
        warnings.warn(
            "reasoning_style='hybrid' is deprecated and was non-functional. Use 'agent' instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        return "agent"
    return value


# =============================================================================
# NEW CONFIG STRUCTURE
# =============================================================================


class LLMConfig(BaseModel):
    """Shared LLM configuration that can be inherited by topics and generation."""

    provider: str | None = Field(
        default=None,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model: str | None = Field(
        default=None,
        description="The name of the model to be used",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )


class TopicsConfig(BaseModel):
    """Configuration for topic generation (tree or graph mode)."""

    prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start topic generation"
    )
    mode: Literal["tree", "graph"] = Field(
        default="graph", description="Topic generation mode: tree or graph"
    )
    system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    depth: int = Field(
        default=TOPIC_TREE_DEFAULT_DEPTH,
        ge=1,
        le=10,
        description="Depth of the tree/graph",
    )
    degree: int = Field(
        default=TOPIC_TREE_DEFAULT_DEGREE,
        ge=1,
        le=50,
        description="Number of subtopics per node (branching factor)",
    )
    max_concurrent: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum concurrent LLM calls during graph expansion (helps avoid rate limits)",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        description="Maximum tokens for topic generation LLM calls",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated topics")
    prompt_style: Literal["default", "isolated", "anchored"] = Field(
        default="default",
        description="For graph mode: 'default' enables cross-connections with generic prompts, 'isolated' disables connections with generic prompts, 'anchored' disables connections and uses domain-aware prompts with examples for focused topic generation",
    )

    # Optional LLM overrides (inherits from top-level llm if not specified)
    llm: LLMConfig | None = Field(
        default=None, description="Optional LLM configuration overrides for topics"
    )


class ConversationConfig(BaseModel):
    """Configuration for conversation structure in generation."""

    type: Literal["basic", "cot"] = Field(
        default="basic",
        description="Base conversation type: basic (simple chat), cot (with reasoning)",
    )
    reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = Field(
        default=None,
        description="Reasoning style for cot: freetext or agent. Note: 'structured' and 'hybrid' are deprecated.",
    )

    @field_validator("reasoning_style", mode="before")
    @classmethod
    def normalize_reasoning_style(cls, v: str | None) -> str | None:
        """Normalize deprecated reasoning_style values."""
        return _normalize_reasoning_style(v)

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate that configuration combinations are consistent."""
        if self.reasoning_style is not None and self.type != "cot":
            raise ValueError(
                f"reasoning_style can only be set when type='cot', got type='{self.type}'"
            )

        if self.type == "cot" and self.reasoning_style is None:
            raise ValueError(
                "reasoning_style must be specified when type='cot'. "
                "Choose from: 'freetext' or 'agent'"
            )

        return self


class ToolsConfig(BaseModel):
    """Configuration for tool/function calling in generation.

    Tools are organized by component - each component routes to a different
    Spin endpoint (e.g., /vfs/execute, /github/execute, /slack/execute).

    Example:
        tools:
          spin_endpoint: "http://localhost:3000"
          components:
            builtin: [read_file, write_file]     # Routes to /vfs/execute
            github: [gh_get_file_contents]       # Routes to /github/execute
            slack: [send_message]                # Routes to /slack/execute
          tools_endpoint: "http://localhost:3000/mock/list-tools"  # For non-builtin tools
    """

    spin_endpoint: str | None = Field(
        default=None,
        description="Spin service URL for real tool execution (e.g., 'http://localhost:3000')",
    )
    components: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Map of component name to tool names. 'builtin' uses built-in tools "
            "(read_file, write_file, list_files, delete_file) and routes to /vfs/execute. "
            "Other components (github, slack, etc.) load tools from tools_endpoint "
            "and route to /{component}/execute."
        ),
    )
    tools_endpoint: str | None = Field(
        default=None,
        description=(
            "HTTP endpoint to load tool definitions from in MCP format "
            "(e.g., 'http://localhost:3000/mock/list-tools'). "
            "Required for non-builtin components."
        ),
    )
    custom: list[dict] = Field(
        default_factory=list,
        description="Custom tool definitions as dictionaries (for inline tool definitions)",
    )
    max_per_query: int = Field(
        default=3, ge=1, le=10, description="Maximum number of tools per query/turn"
    )
    strict: bool = Field(
        default=True,
        description="If True, discard samples exceeding max_per_query. If False, truncate.",
    )
    scenario_seed: dict | None = Field(
        default=None,
        description="Initial state to seed into Spin VFS before generation starts",
    )
    max_agent_steps: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum ReAct reasoning steps before forcing conclusion",
    )

    tool_execute_path: str | None = Field(
        default=None,
        description=(
            "Custom path for tool execution (e.g., '/mock/execute'). "
            "If not set, uses component-based routing (/{component}/execute)."
        ),
    )


class GenerationConfig(BaseModel):
    """Configuration for sample/conversation generation."""

    system_prompt: str = Field(
        ..., min_length=1, description="System prompt for content generation"
    )
    instructions: str = Field(default="", description="Additional instructions for data generation")
    conversation: ConversationConfig = Field(
        default_factory=ConversationConfig,
        description="Conversation structure configuration",
    )
    tools: ToolsConfig | None = Field(
        default=None, description="Tool configuration (required for agent modes)"
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        le=10,
        description="Maximum retries for failed generations",
    )
    sample_retries: int = Field(
        default=DEFAULT_SAMPLE_RETRIES,
        ge=0,
        le=5,
        description="Retries for individual sample validation failures",
    )
    max_tokens: int = Field(default=2000, ge=1, description="Maximum tokens to generate per call")
    rate_limit: dict[str, int | float | str | bool] | None = Field(
        default=None,
        description="Rate limiting and retry configuration",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated samples")

    # Custom structured output schema (bypasses conversation format)
    output_schema: dict | None = Field(
        default=None,
        description=(
            "JSON Schema for custom structured output. When set, bypasses the conversation "
            "format and generates directly into this schema via constrained decoding."
        ),
    )

    # Optional LLM overrides
    llm: LLMConfig | None = Field(
        default=None, description="Optional LLM configuration overrides for generation"
    )


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint-based resume capability.

    Checkpoints allow pausing and resuming long-running dataset generation
    without losing progress. When enabled, samples are periodically saved
    to disk and can be resumed if generation is interrupted.
    """

    interval: int = Field(
        ...,
        ge=1,
        description="Save checkpoint every N samples",
    )
    path: str | None = Field(
        default=None,
        description="Directory to store checkpoint files. If not specified, uses ~/.deepfabric/checkpoints/{config_hash}/",
    )
    retry_failed: bool = Field(
        default=False,
        description="When resuming, retry previously failed samples",
    )


class OutputConfig(BaseModel):
    """Configuration for final dataset output."""

    system_prompt: str | None = Field(
        None,
        description="System prompt that goes INTO the training data (falls back to generation.system_prompt)",
    )
    include_system_message: bool = Field(
        default=True,
        description="Whether to include system message in output format",
    )
    num_samples: int | str = Field(
        default=ENGINE_DEFAULT_NUM_EXAMPLES,
        description="Number of samples: integer, 'auto' (100% of topics), or percentage like '50%'",
    )
    batch_size: int = Field(
        default=ENGINE_DEFAULT_BATCH_SIZE,
        ge=1,
        description="Number of samples to process at a time",
    )
    save_as: str = Field(..., min_length=1, description="Where to save the final dataset")

    format: Literal["messages", "custom"] = Field(
        default="messages",
        description=(
            "'messages' (default): OpenAI chat format. "
            "'custom': emit records matching generation.output_schema directly."
        ),
    )

    # Optional checkpoint configuration (nested inside output)
    checkpoint: CheckpointConfig | None = Field(
        None, description="Checkpoint configuration for resumable generation"
    )

    @field_validator("num_samples", mode="before")
    @classmethod
    def validate_num_samples(cls, v: int | str) -> int | str:
        """Validate num_samples: integer, 'auto', or percentage like '50%'."""
        result = parse_num_samples(v)
        if result is None:
            raise ValueError("num_samples cannot be None")
        return result


class HuggingFaceConfig(BaseModel):
    """Configuration for Hugging Face Hub integration."""

    repository: str = Field(..., min_length=1, description="HuggingFace repository name")
    tags: list[str] = Field(default_factory=list, description="Tags for the dataset")


class KaggleConfig(BaseModel):
    """Configuration for Kaggle integration."""

    handle: str = Field(
        ..., min_length=1, description="Kaggle dataset handle (username/dataset-name)"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for the dataset")
    description: str | None = Field(None, description="Description for the dataset")
    version_notes: str | None = Field(None, description="Version notes for dataset update")


class DeepFabricCloudConfig(BaseModel):
    """Configuration for DeepFabric Cloud integration."""

    graph: str | None = Field(
        default=None,
        description="DeepFabric Cloud graph handle (e.g., username/graph-name)",
    )
    dataset: str | None = Field(
        default=None,
        description="DeepFabric Cloud dataset handle (e.g., username/dataset-name)",
    )
    description: str | None = Field(
        default=None,
        description="Description for uploaded resources",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for uploaded resources",
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    conversation_type: Literal["basic", "cot"] = Field(
        ...,
        description="Conversation type (must match dataset generation)",
    )
    reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = Field(
        default=None,
        description="Reasoning style for cot type",
    )

    @field_validator("reasoning_style", mode="before")
    @classmethod
    def normalize_reasoning_style(cls, v: str | None) -> str | None:
        """Normalize deprecated reasoning_style values."""
        return _normalize_reasoning_style(v)

    metrics: list[str] = Field(
        default_factory=lambda: [
            "tool_selection_accuracy",
            "parameter_accuracy",
            "execution_success_rate",
            "response_quality",
        ],
        description="Metrics to compute during evaluation",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Pass/fail thresholds for metrics",
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "tool_selection": 0.40,
            "parameter_accuracy": 0.30,
            "execution_success": 0.20,
            "response_quality": 0.10,
        },
        description="Metric weights for overall score calculation",
    )
    output_dir: str = Field(
        default="./eval_results",
        description="Output directory for evaluation results",
    )
    output_formats: list[Literal["json", "html", "csv"]] = Field(
        default_factory=lambda: ["json", "html", "csv"],
        description="Output formats to generate",
    )
    include_failures: bool = Field(
        default=True,
        description="Include failed examples in output",
    )
    generate_charts: bool = Field(
        default=True,
        description="Generate visualization charts",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for model inference",
    )
    max_samples: int | None = Field(
        default=None,
        description="Maximum number of samples to evaluate (None for all)",
    )

    @model_validator(mode="after")
    def validate_evaluation_config(self) -> "EvaluationConfig":
        """Validate evaluation configuration consistency."""
        if self.reasoning_style is not None and self.conversation_type != "cot":
            raise ValueError(
                f"reasoning_style can only be set when conversation_type='cot', "
                f"got conversation_type='{self.conversation_type}'"
            )

        if self.conversation_type == "cot" and self.reasoning_style is None:
            raise ValueError(
                "reasoning_style must be specified when conversation_type='cot'. "
                "Choose from: 'freetext' or 'agent'"
            )

        return self


class DeepFabricConfig(BaseModel):
    """Main configuration for DeepFabric tasks using the new structure."""

    # Optional shared LLM defaults
    llm: LLMConfig | None = Field(
        None, description="Shared LLM defaults inherited by topics and generation"
    )

    # Core sections
    topics: TopicsConfig = Field(..., description="Topic generation configuration")
    generation: GenerationConfig = Field(..., description="Sample generation configuration")
    output: OutputConfig = Field(..., description="Output dataset configuration")

    # Optional integrations
    evaluation: EvaluationConfig | None = Field(None, description="Evaluation configuration")
    huggingface: HuggingFaceConfig | None = Field(None, description="Hugging Face configuration")
    kaggle: KaggleConfig | None = Field(None, description="Kaggle configuration")
    deepfabric_cloud: DeepFabricCloudConfig | None = Field(
        None, description="DeepFabric Cloud configuration"
    )

    @classmethod
    def _detect_old_format(cls, config_dict: dict) -> bool:
        """Detect if config uses old format."""
        old_keys = ["topic_tree", "topic_graph", "data_engine", "dataset_system_prompt"]
        return any(key in config_dict for key in old_keys)

    @classmethod
    def _get_migration_message(cls) -> str:
        """Return migration message for old config format."""
        return """
Configuration format has changed. Please update your config to the new structure:

OLD FORMAT                          NEW FORMAT
-----------                         ----------
dataset_system_prompt          ->   output.system_prompt
topic_tree/topic_graph         ->   topics (with mode: tree|graph)
  topic_prompt                 ->     prompt
  topic_system_prompt          ->     system_prompt
data_engine                    ->   generation
  generation_system_prompt     ->     system_prompt
  conversation_type            ->     conversation.type
  reasoning_style              ->     conversation.reasoning_style
  agent_mode                   ->     conversation.agent_mode
  available_tools              ->     tools.available
  custom_tools                 ->     tools.custom
  max_tools_per_query          ->     tools.max_per_query
  max_tools_strict             ->     tools.strict
  spin_endpoint                ->     tools.spin_endpoint
dataset.creation.num_steps     ->   output.num_samples
dataset.creation.batch_size    ->   output.batch_size
dataset.creation.sys_msg       ->   output.include_system_message
dataset.save_as                ->   output.save_as

See documentation for full examples.
"""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeepFabricConfig":
        """Load configuration from a YAML file."""
        try:
            with open(yaml_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise ConfigurationError(f"not found: {yaml_path}") from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"invalid YAML: {str(e)}") from e
        except Exception as e:
            raise ConfigurationError(f"read error: {str(e)}") from e

        if not isinstance(config_dict, dict):
            raise ConfigurationError("must be dictionary")

        # Detect and reject old format
        if cls._detect_old_format(config_dict):
            raise ConfigurationError(cls._get_migration_message())

        try:
            config = cls(**config_dict)
            trace(
                "config_loaded",
                {
                    "method": "yaml",
                    "topics_mode": config.topics.mode,
                    "has_huggingface": config.huggingface is not None,
                    "has_kaggle": config.kaggle is not None,
                },
            )
        except Exception as e:
            raise ConfigurationError(f"invalid structure: {str(e)}") from e
        else:
            return config

    def _resolve_llm_config(self, section_llm: LLMConfig | None) -> LLMConfig:
        """Resolve LLM config with inheritance from top-level.

        Priority order (highest to lowest):
        1. Section-specific llm config (e.g., generation.llm)
        2. Top-level shared llm config
        3. Built-in defaults (DEFAULT_PROVIDER, DEFAULT_MODEL, etc.)
        """
        # Get values from section-specific config (if any)
        section_provider = section_llm.provider if section_llm else None
        section_model = section_llm.model if section_llm else None
        section_temperature = section_llm.temperature if section_llm else None
        section_base_url = section_llm.base_url if section_llm else None

        # Get values from top-level shared config (if any)
        shared_provider = self.llm.provider if self.llm else None
        shared_model = self.llm.model if self.llm else None
        shared_temperature = self.llm.temperature if self.llm else None
        shared_base_url = self.llm.base_url if self.llm else None

        # Resolve with priority: section > shared > defaults
        return LLMConfig(
            provider=section_provider or shared_provider or DEFAULT_PROVIDER,
            model=section_model or shared_model or DEFAULT_MODEL,
            temperature=(
                section_temperature
                if section_temperature is not None
                else (
                    shared_temperature
                    if shared_temperature is not None
                    else TOPIC_TREE_DEFAULT_TEMPERATURE
                )
            ),
            base_url=section_base_url or shared_base_url,
        )

    def get_topics_params(self, **overrides) -> dict:
        """Get parameters for Tree/Graph instantiation."""
        llm = self._resolve_llm_config(self.topics.llm)

        params = {
            "topic_prompt": self.topics.prompt,
            "topic_system_prompt": self.topics.system_prompt,
            "provider": llm.provider,
            "model_name": llm.model,
            "temperature": llm.temperature,
            "base_url": llm.base_url,
            "depth": self.topics.depth,
            "degree": self.topics.degree,
            "max_concurrent": self.topics.max_concurrent,
            "max_tokens": self.topics.max_tokens,
            "prompt_style": self.topics.prompt_style,
        }

        # Handle overrides
        override_provider = overrides.pop("provider", None)
        override_model = overrides.pop("model", None)

        if override_provider:
            params["provider"] = override_provider
        if override_model:
            params["model_name"] = override_model

        params.update(overrides)
        return params

    def get_generation_params(self, **overrides) -> dict:
        """Get parameters for DataSetGenerator instantiation."""
        llm = self._resolve_llm_config(self.generation.llm)

        params = {
            "generation_system_prompt": self.generation.system_prompt,
            "instructions": self.generation.instructions,
            "provider": llm.provider,
            "model_name": llm.model,
            "temperature": llm.temperature,
            "base_url": llm.base_url,
            "max_retries": self.generation.max_retries,
            "sample_retries": self.generation.sample_retries,
            "max_tokens": self.generation.max_tokens,
            "rate_limit": self.generation.rate_limit,
            # Conversation config
            "conversation_type": self.generation.conversation.type,
            "reasoning_style": self.generation.conversation.reasoning_style,
            # Output config
            "sys_msg": self.output.include_system_message,
            "dataset_system_prompt": self.output.system_prompt or self.generation.system_prompt,
            "output_save_as": self.output.save_as,
            # Checkpoint config (nested inside output)
            # Note: checkpoint_path can be None, meaning "auto-resolve" at runtime
            "checkpoint_interval": self.output.checkpoint.interval
            if self.output.checkpoint
            else None,
            "checkpoint_path": self.output.checkpoint.path if self.output.checkpoint else None,
            "checkpoint_retry_failed": self.output.checkpoint.retry_failed
            if self.output.checkpoint
            else False,
        }

        # Tool config
        if self.generation.tools:
            params["tool_components"] = self.generation.tools.components
            params["tools_endpoint"] = self.generation.tools.tools_endpoint
            params["tool_execute_path"] = self.generation.tools.tool_execute_path
            params["custom_tools"] = self.generation.tools.custom
            params["max_tools_per_query"] = self.generation.tools.max_per_query
            params["max_tools_strict"] = self.generation.tools.strict
            params["spin_endpoint"] = self.generation.tools.spin_endpoint
            params["scenario_seed"] = self.generation.tools.scenario_seed
            params["max_agent_steps"] = self.generation.tools.max_agent_steps

        if self.generation.output_schema:
            params["output_schema"] = self.generation.output_schema
            params["output_format"] = self.output.format

        # Handle overrides
        override_provider = overrides.pop("provider", None)
        override_model = overrides.pop("model", None)

        if override_provider:
            params["provider"] = override_provider
        if override_model:
            params["model_name"] = override_model

        params.update(overrides)
        return params

    def get_output_config(self) -> dict:
        """Get output configuration."""
        return {
            "system_prompt": self.output.system_prompt,
            "include_system_message": self.output.include_system_message,
            "num_samples": self.output.num_samples,
            "batch_size": self.output.batch_size,
            "save_as": self.output.save_as,
        }

    def get_checkpoint_config(self) -> dict:
        """Get checkpoint configuration."""
        if self.output.checkpoint is None:
            return {
                "interval": None,
                "path": None,  # None means "auto-resolve" at runtime
                "retry_failed": False,
            }
        return self.output.checkpoint.model_dump()

    def get_huggingface_config(self) -> dict:
        """Get Hugging Face configuration."""
        return self.huggingface.model_dump() if self.huggingface else {}

    def get_kaggle_config(self) -> dict:
        """Get Kaggle configuration."""
        return self.kaggle.model_dump() if self.kaggle else {}

    def get_deepfabric_cloud_config(self) -> dict:
        """Get DeepFabric Cloud configuration."""
        return self.deepfabric_cloud.model_dump() if self.deepfabric_cloud else {}

    def get_configured_providers(self) -> set[str]:
        """Get the set of LLM providers configured in this config."""
        providers = set()

        # Get topics provider
        topics_llm = self._resolve_llm_config(self.topics.llm)
        providers.add(topics_llm.provider)

        # Get generation provider
        gen_llm = self._resolve_llm_config(self.generation.llm)
        providers.add(gen_llm.provider)

        return providers


# =============================================================================
# LEGACY CONFIG CLASSES (for reference during migration - can be removed later)
# =============================================================================


class TopicTreeConfig(BaseModel):
    """DEPRECATED: Configuration for topic tree generation. Use TopicsConfig instead."""

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic tree"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    provider: str = Field(
        default=DEFAULT_PROVIDER,
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=TOPIC_TREE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    degree: int = Field(
        default=TOPIC_TREE_DEFAULT_DEGREE,
        ge=1,
        le=50,
        description="Number of subtopics per node",
    )
    depth: int = Field(
        default=TOPIC_TREE_DEFAULT_DEPTH,
        ge=1,
        le=10,
        description="Depth of the tree",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated topic tree")


class TopicGraphConfig(BaseModel):
    """DEPRECATED: Configuration for topic graph generation. Use TopicsConfig instead."""

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic graph"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    provider: str = Field(
        default=DEFAULT_PROVIDER,
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    degree: int = Field(default=3, ge=1, le=10, description="The branching factor of the graph")
    depth: int = Field(default=2, ge=1, le=5, description="The depth of the graph")
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated topic graph")


class DataEngineConfig(BaseModel):
    """DEPRECATED: Configuration for data engine generation. Use GenerationConfig instead."""

    instructions: str = Field(default="", description="Additional instructions for data generation")
    generation_system_prompt: str = Field(
        ..., min_length=1, description="System prompt for content generation"
    )
    provider: str = Field(
        default=DEFAULT_PROVIDER,
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=ENGINE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        le=10,
        description="Maximum number of retries for failed generations",
    )
    sample_retries: int = Field(
        default=DEFAULT_SAMPLE_RETRIES,
        ge=0,
        le=5,
        description="Number of retries for individual sample validation failures",
    )
    max_tokens: int = Field(
        default=2000, ge=1, description="Maximum tokens to generate in a single call to the llm"
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated data")
    rate_limit: dict[str, int | float | str | bool] | None = Field(
        default=None,
        description="Rate limiting and retry configuration",
    )
    conversation_type: Literal["basic", "cot"] = Field(
        default="basic",
        description="Base conversation type",
    )
    reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = Field(
        default=None,
        description="Reasoning style for cot type",
    )

    @field_validator("reasoning_style", mode="before")
    @classmethod
    def normalize_reasoning_style(cls, v: str | None) -> str | None:
        return _normalize_reasoning_style(v)

    available_tools: list[str] = Field(
        default_factory=list,
        description="List of tool names available",
    )
    custom_tools: list[dict] = Field(default_factory=list, description="Custom tool definitions")
    max_tools_per_query: int = Field(default=3, ge=1, le=10, description="Maximum tools per query")
    max_tools_strict: bool = Field(
        default=True,
        description="Strict mode for tool limits",
    )

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.reasoning_style is not None and self.conversation_type != "cot":
            raise ValueError(
                f"reasoning_style can only be set when conversation_type='cot', "
                f"got conversation_type='{self.conversation_type}'"
            )

        if self.conversation_type == "cot" and self.reasoning_style is None:
            raise ValueError(
                "reasoning_style must be specified when conversation_type='cot'. "
                "Choose from: 'freetext' or 'agent'"
            )

        return self


class DatasetCreationConfig(BaseModel):
    """DEPRECATED: Configuration for dataset creation. Use OutputConfig instead."""

    num_steps: int = Field(
        default=ENGINE_DEFAULT_NUM_EXAMPLES,
        ge=1,
        description="Number of training examples to generate",
    )
    batch_size: int = Field(
        default=ENGINE_DEFAULT_BATCH_SIZE,
        ge=1,
        description="Number of examples to process at a time",
    )
    sys_msg: bool | None = Field(
        default=None,
        description="Include system messages in output format",
    )
    provider: str | None = Field(
        default=None,
        description="Optional provider override",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override",
    )


class DatasetConfig(BaseModel):
    """DEPRECATED: Configuration for dataset assembly. Use OutputConfig instead."""

    creation: DatasetCreationConfig = Field(
        default_factory=DatasetCreationConfig,
        description="Dataset creation parameters",
    )
    save_as: str = Field(..., min_length=1, description="Where to save the final dataset")
