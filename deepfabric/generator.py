import asyncio
import json
import logging
import math
import os
import random

from collections import Counter, deque
from collections.abc import AsyncGenerator
from contextlib import suppress
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal

from datasets import Dataset as HFDataset
from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from .builders import ConversationBuilderFactory
from .bulk import (
    completion_schema,
    estimate_record_tokens,
    make_collection_model,
    make_completion_model,
    record_validator,
    seed_value_schema,
)
from .checkpoint import iter_jsonl, read_jsonl, validate_checkpoint, version_error
from .config import _normalize_reasoning_style
from .constants import (
    API_ERROR_INDICATORS,
    CHECKPOINT_VERSION,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SAMPLE_RETRIES,
    ENGINE_DEFAULT_BATCH_SIZE,
    ENGINE_DEFAULT_NUM_EXAMPLES,
    ENGINE_DEFAULT_TEMPERATURE,
    ERROR_CATEGORIES,
    ERROR_DATASET_FILENAME,
    INTERRUPTED_DATASET_FILENAME,
)
from .error_codes import classify_error
from .exceptions import DataSetGeneratorError
from .llm import LLMClient
from .llm.client import make_dynamic_model
from .metrics import trace
from .progress import ProgressReporter
from .prompts import (
    AGENT_COT_TOOLS_PROMPT,
    BULK_COMPLETION_PROMPT,
    BULK_FULL_SAMPLE_PROMPT,
    BULK_SEED_SAMPLE_PROMPT,
    CONVERSATION_GENERATION_PROMPT,
    CUSTOM_SCHEMA_PROMPT,
    FREETEXT_COT_PROMPT,
    STRUCTURED_COT_PROMPT,
    AgentPromptBuilder,
)
from .sampling import STEP_TOPIC_ID, resolve_target, topic_plan
from .schemas import ChatMessage, Conversation, ToolRegistry, get_conversation_schema
from .stream_simulator import simulate_stream
from .tools import BUILTIN_TOOL_REGISTRY
from .tools.loader import load_tools_from_dict, load_tools_from_endpoint
from .topic_model import TopicModel, TopicPath
from .topic_sampling import TopicSamplingConfig, warn_duplicate_prompt
from .utils import build_checkpoint_file_paths, ensure_not_running_loop, is_validation_error

# Handle circular import for type hints
if TYPE_CHECKING:
    from .topic_model import TopicModel

logger = logging.getLogger(__name__)


class DataSetGeneratorConfig(TopicSamplingConfig):
    """Configuration for the data engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: str = Field(default="", description="Additional instructions for data generation")
    generation_system_prompt: str = Field(
        ..., min_length=1, description="System prompt for content generation"
    )
    dataset_system_prompt: str | None = Field(
        None,
        description="System prompt that goes into the final dataset (falls back to generation_system_prompt if not provided)",
    )
    provider: str = Field(
        ...,
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model_name: str = Field(..., min_length=1, description="Name of the model to use")
    prompt_template: str | None = Field(default=None, description="Custom prompt template")
    example_data: HFDataset | None = Field(
        default=None, description="Example dataset for few-shot learning"
    )
    temperature: float = Field(
        default=ENGINE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=1,
        le=10,
        description="Maximum number of retries for failed requests (deprecated, use rate_limit config)",
    )
    max_tokens: int = Field(
        default=2000,
        ge=1,
        description="Maximum tokens to generate in a single call to the llm",
    )
    default_batch_size: int = Field(
        default=ENGINE_DEFAULT_BATCH_SIZE,
        ge=1,
        le=100,
        description="Default batch size for generation",
    )
    default_num_examples: int = Field(
        default=ENGINE_DEFAULT_NUM_EXAMPLES,
        ge=0,
        le=10,
        description="Default number of examples to include",
    )
    request_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=5,
        le=300,
        description="Request timeout in seconds",
    )
    sample_retries: int = Field(
        default=DEFAULT_SAMPLE_RETRIES,
        ge=0,
        le=5,
        description="Number of retries for individual sample validation failures",
    )
    sys_msg: bool = Field(default=True, description="Whether to include system message in dataset")
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )

    # Rate limiting configuration
    rate_limit: dict[str, int | float | str | bool] | None = Field(
        default=None,
        description="Rate limiting and retry configuration (uses provider defaults if not specified)",
    )

    # Custom structured output schema
    output_schema: dict | None = Field(
        default=None,
        description=(
            "JSON Schema for custom structured output. When set, bypasses the conversation "
            "format and generates directly into this schema via constrained decoding. "
            "Records in the JSONL output will match the schema instead of the OpenAI messages format."
        ),
    )
    output_format: Literal["messages", "custom"] = Field(
        default="messages",
        description=(
            "'messages' (default): OpenAI chat format. "
            "'custom': emit records matching output_schema directly."
        ),
    )

    # Modular conversation configuration
    conversation_type: Literal["basic", "cot"] = Field(
        default="basic",
        description="Base conversation type: basic (simple chat), cot (with reasoning traces)",
    )

    reasoning_style: Literal["freetext", "agent", "structured", "hybrid"] | None = Field(
        default=None,
        description="Reasoning style for cot type: freetext (natural language) or agent (structured step-by-step for tool-calling). Note: 'structured' and 'hybrid' are deprecated.",
    )

    @field_validator("reasoning_style", mode="before")
    @classmethod
    def normalize_reasoning_style(cls, v: str | None) -> str | None:
        """Normalize deprecated reasoning_style values."""
        return _normalize_reasoning_style(v)

    # Tool configuration (used when tools are configured for agent mode)
    tool_components: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Map of component name to tool names. 'builtin' uses built-in tools "
            "and routes to /vfs/execute. Other components load from tools_endpoint "
            "and route to /{component}/execute."
        ),
    )
    custom_tools: list[dict] = Field(
        default_factory=list, description="Custom tool definitions as dictionaries"
    )
    max_tools_per_query: int = Field(
        default=3, ge=1, le=10, description="Maximum number of tools per query/turn"
    )
    max_tools_strict: bool = Field(
        default=True,
        description="If True, discard samples exceeding max_tools_per_query. If False, keep sample but truncate executions to limit.",
    )

    # Spin integration for real tool execution
    spin_endpoint: str | None = Field(
        default=None,
        description="Spin service URL for real tool execution (e.g., 'http://localhost:3000')",
    )
    scenario_seed: dict | None = Field(
        default=None,
        description="Initial state to seed into Spin VFS before generation (e.g., {'files': {'main.py': '...'}})",
    )
    max_agent_steps: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum ReAct reasoning steps per sample before forcing conclusion",
    )

    # MCP/Mock tool integration - load tools from HTTP endpoint instead of code
    tools_endpoint: str | None = Field(
        default=None,
        description="HTTP endpoint to load tool definitions from (e.g., 'http://localhost:3000/mock/list-tools'). Tools are loaded in MCP format.",
    )
    tool_execute_path: str | None = Field(
        default=None,
        description="Path for tool execution when using tools_endpoint (e.g., '/mock/execute'). Combined with spin_endpoint.",
    )

    tool_inclusion_strategy: Literal["all", "used_only"] = Field(
        default="used_only",
        description="Which tools to include in each sample: 'all' includes full catalog, 'used_only' includes only tools actually called (recommended for training)",
    )

    # Checkpoint configuration
    checkpoint_interval: int | None = Field(
        default=None,
        ge=1,
        description="Save checkpoint every N samples. None disables checkpointing.",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Directory to store checkpoint files. None uses fallback '.checkpoints'",
    )
    checkpoint_retry_failed: bool = Field(
        default=False,
        description="When resuming, retry previously failed samples",
    )
    output_save_as: str | None = Field(
        default=None,
        description="Output file path (used to derive checkpoint file names)",
    )
    topics_file: str | None = Field(
        default=None,
        description="Topics file path (stored in checkpoint metadata for auto-resume)",
    )

    @model_validator(mode="after")
    def validate_sample_mode(self):
        self.validate_topic_sampling(
            self.output_schema, has_tools=bool(self.tool_components or self.custom_tools)
        )
        warn_duplicate_prompt(
            self.dataset_system_prompt or self.generation_system_prompt, self.instructions
        )
        return self


class DataSetGenerator:
    def __init__(self, **kwargs):
        """Initialize DataSetGenerator with parameters."""
        try:
            self.config = DataSetGeneratorConfig.model_validate(kwargs)
        except Exception as e:  # noqa: TRY003
            raise DataSetGeneratorError(f"Invalid generator configuration: {str(e)}") from e

        # Initialize from config
        self.provider = self.config.provider
        self.model_name = self.config.model_name
        self._samples: list[dict] = []
        self.failed_samples = []
        self.failure_analysis = {category: [] for category in ERROR_CATEGORIES}

        # Initialize LLM client with rate limiting configuration
        llm_kwargs: dict[str, Any] = {"rate_limit_config": self.config.rate_limit}
        if self.config.base_url:
            llm_kwargs["base_url"] = self.config.base_url

        self.llm_client = LLMClient(
            provider=self.provider,
            model_name=self.model_name,
            **llm_kwargs,
        )
        trace(
            "generator_created",
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "conversation_type": self.config.conversation_type,
            },
        )

        # Store dataset system prompt for dataset inclusion (with fallback)
        self.dataset_system_prompt = (
            self.config.dataset_system_prompt or self.config.generation_system_prompt
        )
        # Store generation prompt for content generation
        self.generation_prompt = self.config.generation_system_prompt

        # Initialize tool registry when tools are configured (enables agent mode)
        self.tool_registry = None
        if self.config.tool_components or self.config.custom_tools:
            self._initialize_tool_registry()

        # Progress reporter for streaming feedback (set by external callers)
        self.progress_reporter: ProgressReporter | None = None

        # Checkpoint state
        self._checkpoint_samples_since_save = 0
        self._completed: set[tuple[str, int]] = set()  # Attempted (UUID, sample ordinal) slots
        self._sample_slots: list[tuple[str, int]] = []
        self._generation_plan: dict | None = None
        self._topic_seeds: dict[str, dict[str, Any]] = {}
        self._collection_attempts: dict[str, int] = {}
        self._checkpoint_loaded = False  # Explicit load pending consumption by one run
        self._checkpoint_metadata_path: Path | None = None
        self._checkpoint_samples_path: Path | None = None
        self._checkpoint_failures_path: Path | None = None
        # Memory optimization: track flushed counts for checkpoint mode
        self._flushed_samples_count = 0
        self._flushed_failures_count = 0

        # Graceful stop flag - set by signal handler to stop at next checkpoint
        self.stop_requested = False

    def _initialize_tool_registry(self):
        """Initialize tool registry from component configuration.

        Tools are loaded based on the tool_components mapping:
        - 'builtin': Uses BUILTIN_TOOL_REGISTRY (read_file, write_file, etc.)
        - Other components: Loads from tools_endpoint and sets component field

        Each tool's component field determines routing (/{component}/execute).
        """
        try:
            all_tools = []
            endpoint_registry = None

            # Load tools from endpoint if needed for non-builtin components
            non_builtin_components = {
                k: v for k, v in self.config.tool_components.items() if k != "builtin"
            }
            if non_builtin_components:
                if not self.config.tools_endpoint:
                    raise DataSetGeneratorError(
                        f"Non-builtin components {list(non_builtin_components.keys())} require "
                        "'tools_endpoint' to load tool definitions."
                    )
                endpoint_registry = load_tools_from_endpoint(self.config.tools_endpoint)
                logger.info(
                    "Loaded %d tools from endpoint: %s",
                    len(endpoint_registry.tools),
                    self.config.tools_endpoint,
                )

            # Process each component
            for component_name, tool_names in self.config.tool_components.items():
                if component_name == "builtin":
                    # Filter from builtin registry
                    for tool in BUILTIN_TOOL_REGISTRY.tools:
                        if tool.name in tool_names:
                            all_tools.append(tool)
                elif endpoint_registry:
                    # Filter from endpoint registry and set component
                    for tool in endpoint_registry.tools:
                        if tool.name in tool_names:
                            # Create copy with component set
                            tool_copy = tool.model_copy(update={"component": component_name})
                            all_tools.append(tool_copy)

            # Add custom tools if provided
            if self.config.custom_tools:
                custom_registry = load_tools_from_dict(self.config.custom_tools)
                all_tools.extend(custom_registry.tools)

            self.tool_registry = ToolRegistry(tools=all_tools)
            logger.info("Initialized tool registry with %d tools", len(all_tools))

        except Exception as e:  # noqa: BLE001
            raise DataSetGeneratorError(f"Failed to initialize tool registry: {str(e)}") from e

    def _get_checkpoint_paths(self) -> tuple[Path, Path, Path]:
        """Get checkpoint file paths based on output_save_as.

        Returns:
            Tuple of (metadata_path, samples_path, failures_path)
        """
        if not self.config.output_save_as:
            raise DataSetGeneratorError(
                "Cannot create checkpoint paths: output_save_as not configured"
            )

        paths = build_checkpoint_file_paths(self.config.output_save_as, self.config.checkpoint_path)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        return paths

    def _initialize_checkpoint_paths(self) -> None:
        """Initialize checkpoint file paths if checkpointing is enabled."""
        if self.config.checkpoint_interval is not None:
            paths = self._get_checkpoint_paths()
            self._checkpoint_metadata_path = paths[0]
            self._checkpoint_samples_path = paths[1]
            self._checkpoint_failures_path = paths[2]
            logger.info(
                "Checkpointing enabled: saving every %d samples to %s",
                self.config.checkpoint_interval,
                self._checkpoint_samples_path,
            )

    def _save_checkpoint(
        self,
        new_samples: list[dict],
        new_failures: list[dict],
        completed_items: list[tuple[str, int]],
        flush_memory: bool = True,
        *,
        sample_slots: list[tuple[str, int]] | None = None,
    ) -> None:
        """Save checkpoint data incrementally.

        Args:
            new_samples: New successful samples to append
            new_failures: New failed samples to append
            completed_items: List of (uuid, sample_ordinal) slots that settled
            flush_memory: If True, clear flushed samples from memory (memory optimization)
        """
        if self._checkpoint_samples_path is None:
            return

        # A handled write failure must leave the previous checkpoint usable. In
        # particular, recovery may retry these same buffers after a metadata error.
        outcome_paths = [
            path
            for path, rows in (
                (self._checkpoint_samples_path, new_samples),
                (self._checkpoint_failures_path, new_failures),
            )
            if path is not None and rows
        ]
        offsets = {path: path.stat().st_size if path.exists() else 0 for path in outcome_paths}
        slot_count = len(self._sample_slots)
        added_completed = set(completed_items).difference(self._completed)
        saved_samples = list(self._samples) if flush_memory else []
        saved_failures = list(self.failed_samples) if flush_memory else []
        sample_count = self._flushed_samples_count
        failure_count = self._flushed_failures_count
        try:
            for path, rows in (
                (self._checkpoint_samples_path, new_samples),
                (self._checkpoint_failures_path, new_failures),
            ):
                if path is not None and rows:
                    with path.open("a", encoding="utf-8") as stream:
                        for row in rows:
                            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._sample_slots.extend(sample_slots or [])
            self._completed.update(completed_items)
            if flush_memory:
                self._flushed_samples_count += len(new_samples)
                self._flushed_failures_count += len(new_failures)
                self._samples.clear()
                self.failed_samples.clear()
            self._save_checkpoint_metadata()
        except (Exception, KeyboardInterrupt):
            for path, offset in offsets.items():
                if path.exists():
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
            del self._sample_slots[slot_count:]
            self._completed.difference_update(added_completed)
            self._flushed_samples_count = sample_count
            self._flushed_failures_count = failure_count
            if flush_memory:
                self._samples[:] = saved_samples
                self.failed_samples[:] = saved_failures
            raise

        logger.debug(
            "Checkpoint saved: %d samples, %d failures, %d completed tuples (flushed=%s)",
            len(new_samples),
            len(new_failures),
            len(self._completed),
            flush_memory,
        )

    def _save_checkpoint_metadata(self) -> None:
        """Save checkpoint metadata file."""
        if self._checkpoint_metadata_path is None:
            return

        # Total counts include both flushed (on disk) and in-memory samples
        total_samples = self._flushed_samples_count + len(self._samples)
        total_failures = self._flushed_failures_count + len(self.failed_samples)

        # Persist explicit per-slot identities alongside the exact plan
        completed_list = [[uuid, ordinal] for uuid, ordinal in self._completed]

        metadata = {
            "version": CHECKPOINT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider,
            "model_name": self.model_name,
            "conversation_type": self.config.conversation_type,
            "reasoning_style": self.config.reasoning_style,
            "total_samples": total_samples,
            "total_failures": total_failures,
            "completed": completed_list,
            "plan": self._generation_plan,
            "sample_slots": [list(slot) for slot in self._sample_slots],
            "checkpoint_interval": self.config.checkpoint_interval,
            "topics_file": self.config.topics_file,
            "topic_sample_mode": self.config.topic_sample_mode,
            "shared_scope": self.config.shared_scope,
            "seed_field": self.config.seed_field,
            "collection_attempts": self._collection_attempts,
        }

        # Sequential history comes from sample_slots and rows, not a seed map.
        # Retain historical settled seeds when resuming under a different mode.
        if self._topic_seeds or (
            self.config.topic_sample_mode == "bulk" and self.config.shared_scope == "seed"
        ):
            metadata["seeds"] = self._topic_seeds
        # Publish complete JSON atomically; a partial write must not destroy the
        # last valid metadata before _save_checkpoint can roll back appended rows.
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._checkpoint_metadata_path.parent,
                prefix=f".{self._checkpoint_metadata_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                json.dump(metadata, stream, indent=2)
            os.replace(temporary_path, self._checkpoint_metadata_path)
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    def _validate_checkpoint_compatibility(self, metadata: dict) -> None:
        """Validate that current config is compatible with checkpoint.

        Logs warnings for config mismatches but allows resumption.

        Args:
            metadata: Checkpoint metadata dictionary
        """
        mismatches: list[str] = []

        # Check provider
        checkpoint_provider = metadata.get("provider")
        if checkpoint_provider and checkpoint_provider != self.provider:
            mismatches.append(
                f"provider: checkpoint={checkpoint_provider}, current={self.provider}"
            )

        # Check model
        checkpoint_model = metadata.get("model_name")
        if checkpoint_model and checkpoint_model != self.model_name:
            mismatches.append(
                f"model_name: checkpoint={checkpoint_model}, current={self.model_name}"
            )

        # Check conversation type
        checkpoint_conv_type = metadata.get("conversation_type")
        if checkpoint_conv_type and checkpoint_conv_type != self.config.conversation_type:
            mismatches.append(
                f"conversation_type: checkpoint={checkpoint_conv_type}, "
                f"current={self.config.conversation_type}"
            )

        # Check reasoning style
        checkpoint_reasoning = metadata.get("reasoning_style")
        if checkpoint_reasoning and checkpoint_reasoning != self.config.reasoning_style:
            mismatches.append(
                f"reasoning_style: checkpoint={checkpoint_reasoning}, "
                f"current={self.config.reasoning_style}"
            )

        for name, default in (
            ("topic_sample_mode", "independent"),
            ("shared_scope", "seed"),
            ("seed_field", None),
        ):
            previous = metadata.get(name, default)
            current = getattr(self.config, name)
            if previous != current:
                mismatches.append(f"{name}: checkpoint={previous}, current={current}")

        if mismatches:
            logger.warning(
                "Config mismatch with checkpoint. Resuming may produce inconsistent results. "
                "Differences: %s",
                "; ".join(mismatches),
            )

    def _validate_checkpoint_integrity(self, metadata: dict) -> tuple[bool, str | None]:
        try:
            validate_checkpoint(
                metadata, self._checkpoint_samples_path, self._checkpoint_failures_path
            )
        except (DataSetGeneratorError, KeyError, TypeError, ValueError, OSError) as exc:
            return False, str(exc)
        return True, None

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists without loading it.

        Returns:
            True if checkpoint metadata file exists, False otherwise
        """
        if self.config.checkpoint_interval is None:
            return False

        self._initialize_checkpoint_paths()
        return (
            self._checkpoint_metadata_path is not None and self._checkpoint_metadata_path.exists()
        )

    def load_checkpoint(self, retry_failed: bool = False) -> bool:
        """Load v5 per-slot state; incompatible checkpoints never start a fresh run."""
        if self.config.checkpoint_interval is None:
            return False
        self._initialize_checkpoint_paths()
        if not self._checkpoint_metadata_path.exists():
            return False
        with self._checkpoint_metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("version") != CHECKPOINT_VERSION:
            raise DataSetGeneratorError(
                version_error(metadata.get("version"), self._checkpoint_metadata_path)
            )
        valid, error = self._validate_checkpoint_integrity(metadata)
        if not valid:
            raise DataSetGeneratorError(error)
        self._validate_checkpoint_compatibility(metadata)
        self._samples.clear()
        self.failed_samples.clear()
        self.stop_requested = False
        self._generation_plan = metadata["plan"]
        self._checkpoint_loaded = True
        self._completed = {tuple(item) for item in metadata["completed"]}
        self._sample_slots = [tuple(item) for item in metadata["sample_slots"]]
        self._flushed_samples_count = metadata["total_samples"]
        failures = read_jsonl(self._checkpoint_failures_path)
        self._flushed_failures_count = len(failures)
        self._topic_seeds = metadata.get("seeds", {})
        self._collection_attempts = metadata.get("collection_attempts", {})
        if any(
            metadata.get(name, default) != getattr(self.config, name)
            for name, default in (
                ("topic_sample_mode", "independent"),
                ("shared_scope", "seed"),
                ("seed_field", None),
            )
        ):
            retained_slots = self._completed
            if retry_failed:
                retained_slots = retained_slots - {
                    (failure["topic_id"], failure["sample_ordinal"]) for failure in failures
                }
            self._topic_seeds = {
                topic_id: {
                    ordinal: seed
                    for ordinal, seed in seeds.items()
                    if (topic_id, int(ordinal)) in retained_slots
                }
                for topic_id, seeds in self._topic_seeds.items()
            }
            self._collection_attempts.clear()
            logger.warning(
                "Discarded unsettled seed and collection-attempt state after mode change"
            )
        if retry_failed:
            self.retry_checkpoint_failures()
        logger.info(
            "Loaded checkpoint: %d samples, %d failures, %d settled slots",
            self._flushed_samples_count,
            self._flushed_failures_count,
            len(self._completed),
        )
        return True

    def retry_checkpoint_failures(self) -> None:
        """Reopen failed slots after a loaded checkpoint's resume is committed.

        CLI startup first loads read-only and collects all choices. This separate
        step prevents an aborted prompt from truncating checkpoint failures.
        """
        if not self._checkpoint_loaded:
            raise DataSetGeneratorError("Retry requires an actually loaded checkpoint")
        failures = read_jsonl(self._checkpoint_failures_path)
        if not failures:
            return
        self._completed.difference_update((f["topic_id"], f["sample_ordinal"]) for f in failures)
        self._checkpoint_failures_path.write_text("", encoding="utf-8")
        self._flushed_failures_count = 0
        # A deliberate retry starts a fresh collection budget, but keeps pinned seeds.
        for failure in failures:
            self._collection_attempts.pop(failure["topic_id"], None)
        self._save_checkpoint_metadata()

    def clear_checkpoint(self) -> None:
        """Remove checkpoint files."""
        if self._checkpoint_metadata_path and self._checkpoint_metadata_path.exists():
            os.remove(self._checkpoint_metadata_path)
        if self._checkpoint_samples_path and self._checkpoint_samples_path.exists():
            os.remove(self._checkpoint_samples_path)
        if self._checkpoint_failures_path and self._checkpoint_failures_path.exists():
            os.remove(self._checkpoint_failures_path)
        self._completed.clear()
        self._sample_slots.clear()
        self._generation_plan = None
        self._topic_seeds.clear()
        self._collection_attempts.clear()
        self._checkpoint_loaded = False
        self._flushed_samples_count = 0
        self._flushed_failures_count = 0
        logger.info("Checkpoint files cleared")

    def _load_all_samples_from_checkpoint(self) -> list[dict]:
        """Load all samples from checkpoint file.

        Used at end of generation to build final dataset when memory
        optimization has flushed samples to disk.

        Returns:
            List of all sample dictionaries from checkpoint file
        """
        all_samples: list[dict] = []
        if self._checkpoint_samples_path and self._checkpoint_samples_path.exists():
            with open(self._checkpoint_samples_path, encoding="utf-8") as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if stripped:
                        all_samples.append(json.loads(stripped))
        return all_samples

    def get_all_failures(self) -> list[dict]:
        """Get all failures including those flushed to checkpoint.

        This combines in-memory failures with any that were flushed to the
        checkpoint failures file during memory optimization.

        Returns:
            List of all failure dictionaries
        """
        all_failures: list[dict] = []

        # First load from checkpoint file if it exists
        if self._checkpoint_failures_path and self._checkpoint_failures_path.exists():
            with open(self._checkpoint_failures_path, encoding="utf-8") as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if stripped:
                        all_failures.append(json.loads(stripped))

        # Then add any in-memory failures (if not yet flushed)
        all_failures.extend(self.failed_samples)

        return all_failures

    def _is_completed(self, uuid: str, sample_ordinal: int) -> bool:
        return (uuid, sample_ordinal) in self._completed

    def _prepare_topic_paths(
        self,
        num_steps: int,
        batch_size: int,
        topic_model: "TopicModel | None" = None,
    ) -> tuple[list[TopicPath] | None, int]:
        """Legacy path-list helper; the active topic and step runners do not call it."""
        topic_paths: list[TopicPath] | None = None
        if topic_model is not None:
            topic_paths = topic_model.get_all_paths_with_ids()
            total_paths = len(topic_paths)
            required_samples = num_steps * batch_size
            logger.info(
                "Topic preparation: total_paths=%d, required_samples=%d, num_steps=%d, batch_size=%d",
                total_paths,
                required_samples,
                num_steps,
                batch_size,
            )

            if required_samples > total_paths:
                # Cycle through topics to generate more samples than paths
                # Each topic will be used multiple times for even coverage
                # IMPORTANT: Create unique topic_ids using global index position.
                # This handles two cases:
                # 1. Cycling: same path appears multiple times across cycles
                # 2. Graph duplicates: multiple paths can share the same node topic_id
                # Using index ensures every position in the list has a unique ID.
                multiplier = math.ceil(required_samples / total_paths)
                cycled_paths: list[TopicPath] = []
                for _cycle in range(multiplier):
                    for _path_idx, tp in enumerate(topic_paths):
                        if len(cycled_paths) >= required_samples:
                            break
                        # Use global index for uniqueness: handles both cycling and graph duplicates
                        global_idx = len(cycled_paths)
                        unique_id = f"{tp.topic_id}_idx_{global_idx}"
                        cycled_paths.append(TopicPath(path=tp.path, topic_id=unique_id))
                    if len(cycled_paths) >= required_samples:
                        break
                topic_paths = cycled_paths
                logger.info(
                    "Topics cycled: %d original paths × %d cycles = %d total (trimmed to %d)",
                    total_paths,
                    multiplier,
                    total_paths * multiplier,
                    len(topic_paths),
                )
            elif required_samples < total_paths:
                # Sample subset (percentage case or explicit count < total)
                # Bandit: not a security function
                topic_paths = random.sample(topic_paths, required_samples)  # nosec
                # Assign unique IDs to handle graphs with duplicate node IDs
                topic_paths = [
                    TopicPath(path=tp.path, topic_id=f"{tp.topic_id}_idx_{idx}")
                    for idx, tp in enumerate(topic_paths)
                ]
            else:
                # required_samples == total_paths - use all paths but with unique IDs
                # Graphs can have duplicate topic_ids (multiple paths to same node)
                topic_paths = [
                    TopicPath(path=tp.path, topic_id=f"{tp.topic_id}_idx_{idx}")
                    for idx, tp in enumerate(topic_paths)
                ]

            logger.info("Topic paths after preparation: %d paths", len(topic_paths))

        return topic_paths, num_steps

    def _prepare_topic_allocations(self, total_samples: int, topic_model: TopicModel) -> dict:
        topics = topic_model.get_unique_topics()
        return topic_plan(
            total_samples,
            [
                {
                    "uuid": topic.uuid,
                    "topic": topic.topic,
                    "path": topic_model.get_path_by_id(topic.uuid) or [topic.topic],
                }
                for topic in topics
            ],
        )

    def _generate_batch_prompts(
        self,
        batch_size: int,
        start_idx: int,
        topic_paths: list[TopicPath],
        data_creation_prompt: str,
        num_example_demonstrations: int,
    ) -> tuple[list[str], list[TopicPath | None]]:
        """Generate prompts for a batch and return the associated TopicPaths used.

        Returns:
            (prompts, used_topic_paths) where used_topic_paths aligns with prompts order.
        """
        prompts: list[str] = []
        used_topic_paths: list[TopicPath | None] = []
        for i in range(batch_size):
            topic_path: TopicPath | None = None
            path: list[str] | None = None
            if topic_paths:
                current_idx = start_idx + i
                if current_idx < len(topic_paths):
                    topic_path = topic_paths[current_idx]
                    path = topic_path.path
                else:
                    break

            sample_prompt = self.build_prompt(
                data_creation_prompt=data_creation_prompt,
                num_example_demonstrations=num_example_demonstrations,
                subtopics_list=path,
            )
            prompts.append(sample_prompt)
            used_topic_paths.append(topic_path)
        return prompts, used_topic_paths

    def _get_minimal_schema(self) -> type:
        """Get the conversation schema for the current config."""
        return get_conversation_schema(self.config.conversation_type)

    def _get_custom_schema_model(self) -> type:
        """Build a dynamic model class from the configured output_schema."""
        return make_dynamic_model(self.config.output_schema)

    def _get_prompt_template(self) -> str:
        """Return the prompt template, using CUSTOM_SCHEMA_PROMPT when output_schema is set."""
        if self.config.output_schema:
            return CUSTOM_SCHEMA_PROMPT
        return self._get_cot_prompt_template()

    def _emit_retry(
        self,
        sample_idx: int,
        attempt: int,
        max_attempts: int,
        error: Exception | str,
    ) -> None:
        """Emit a retry event if a progress reporter is attached.

        Args:
            sample_idx: 0-based sample index (will be converted to 1-based)
            attempt: 0-based attempt number (will be converted to 1-based)
            max_attempts: Total number of attempts allowed
            error: The error that triggered the retry
        """
        if self.progress_reporter:
            self.progress_reporter.emit_retry(
                sample_idx=sample_idx + 1,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                error_summary=str(error)[:100],
            )

    async def _generate_structured_samples_async(
        self,
        prompts: list[str],
        include_sys_msg: bool,
        start_sample_idx: int = 0,
        topic_paths_for_batch: list[TopicPath | None] | None = None,
    ) -> tuple[list, list]:
        """Generate structured samples using builder pattern.

        Args:
            prompts: List of topic prompts to generate samples for
            include_sys_msg: Whether to include system message in output
            start_sample_idx: Starting sample index for progress reporting
            topic_paths_for_batch: TopicPath objects for each sample (includes topic_id)

        Returns:
            Tuple of (successful samples, failed responses)
        """

        samples = []
        failed_responses = []

        # Create config with overridden sys_msg if needed
        config = self.config
        if include_sys_msg != self.config.sys_msg:
            # Create a copy of config with sys_msg overridden
            config = self.config.model_copy(update={"sys_msg": include_sys_msg})

        async def _generate_with_retry(  # noqa: PLR0911
            prompt: str, sample_idx: int, topic_path_info: TopicPath | None
        ) -> tuple[bool, Exception | Conversation]:
            """Generate a single sample with per-sample retry for validation errors.

            Each parallel task gets its own builder instance to avoid Spin session
            conflicts when running samples concurrently (batch_size > 1).
            """
            last_error: Exception | None = None
            max_attempts = self.config.sample_retries + 1

            # Custom schema mode: bypass conversation builder entirely and
            # generate directly into the user-defined schema via constrained decoding.
            if config.output_schema:
                schema_model = self._get_custom_schema_model()
                for attempt in range(max_attempts):
                    try:
                        result = await self.llm_client.generate_async(
                            prompt,
                            schema_model,
                            max_tokens=config.max_tokens,
                        )
                    except Exception as e:  # noqa: BLE001
                        last_error = e
                        if is_validation_error(e) and attempt < self.config.sample_retries:
                            self._emit_retry(sample_idx, attempt, max_attempts, e)
                            continue
                        return False, last_error
                    else:
                        return True, result
                return False, last_error or Exception("Custom schema generation failed")

            # Normal conversation builder mode
            # Create a fresh builder for this sample to avoid session conflicts
            # when running in parallel batches
            builder = ConversationBuilderFactory.create(
                config=config,
                llm=self.llm_client,
                tool_registry=self.tool_registry,
                progress_reporter=self.progress_reporter,
            )

            error_feedback: str | None = None
            logger.debug(
                "Sample %d: max_attempts=%d (sample_retries=%d)",
                sample_idx + 1,
                max_attempts,
                self.config.sample_retries,
            )

            # Extract path for progress reporting
            path_info = topic_path_info.path if topic_path_info else None

            for attempt in range(max_attempts):
                # Notify progress reporter about which sample we're working on
                if self.progress_reporter:
                    retry_suffix = f" (retry {attempt})" if attempt > 0 else ""
                    self.progress_reporter.emit_step_start(
                        f"Generating sample {sample_idx + 1}{retry_suffix}",
                        sample_idx=sample_idx + 1,
                        topic_path=path_info,
                    )

                try:
                    # Builder handles all generation complexity
                    # Pass error feedback from previous attempt if this is a retry
                    conversation = await builder.generate(prompt, error_feedback)
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    is_validation = is_validation_error(e)
                    can_retry = attempt < self.config.sample_retries
                    logger.debug(
                        "Sample %d error: is_validation=%s, can_retry=%s, attempt=%d/%d, error=%s",
                        sample_idx + 1,
                        is_validation,
                        can_retry,
                        attempt + 1,
                        self.config.sample_retries + 1,
                        str(e)[:200],
                    )
                    # Only retry validation errors, not API/network errors
                    if is_validation and can_retry:
                        # Extract error message for feedback to the model
                        error_feedback = str(e)
                        self._emit_retry(sample_idx, attempt, max_attempts, e)
                        continue
                    # Non-retryable error or exhausted retries
                    return False, last_error or Exception("Sample generation failed")

                else:
                    # Validate tool execution count for agent mode (when tools configured)
                    if self.tool_registry is not None:
                        if (
                            not conversation.tool_context
                            or not conversation.tool_context.executions
                        ):
                            last_error = ValueError(
                                "Agent mode requires at least one tool execution"
                            )
                            if attempt < self.config.sample_retries:
                                self._emit_retry(sample_idx, attempt, max_attempts, last_error)
                                continue
                            return False, last_error or Exception("Sample generation failed")

                        num_executions = len(conversation.tool_context.executions)
                        if num_executions > self.config.max_tools_per_query:
                            if self.config.max_tools_strict:
                                # Strict mode: discard entire sample
                                last_error = ValueError(
                                    f"Sample has {num_executions} tool executions, "
                                    f"exceeds limit of {self.config.max_tools_per_query}"
                                )
                                if attempt < self.config.sample_retries:
                                    self._emit_retry(sample_idx, attempt, max_attempts, last_error)
                                    continue
                                return False, last_error or Exception("Sample generation failed")
                            # Non-strict mode: truncate to limit and keep sample
                            conversation.tool_context.executions = (
                                conversation.tool_context.executions[
                                    : self.config.max_tools_per_query
                                ]
                            )

                    # Add topic_id to conversation metadata for traceability
                    if topic_path_info and hasattr(conversation, "metadata"):
                        if conversation.metadata is None:
                            conversation.metadata = {}
                        conversation.metadata["topic_id"] = topic_path_info.topic_id

                    return True, conversation

            return False, last_error or Exception("Sample generation failed")

        # Generate all samples concurrently with sample indices
        tasks = []
        for idx, prompt in enumerate(prompts):
            topic_path_info = None
            if topic_paths_for_batch and idx < len(topic_paths_for_batch):
                topic_path_info = topic_paths_for_batch[idx]
            tasks.append(
                asyncio.create_task(
                    _generate_with_retry(prompt, start_sample_idx + idx, topic_path_info)
                )
            )
        results = await asyncio.gather(*tasks)

        for idx, (success, payload) in enumerate(results):
            if success:
                samples.append(payload)
            else:
                error = payload
                error_msg = f"Generation failed: {error}"
                # Build failure record with raw content and topic_id if available
                failure_record: dict[str, str | None] = {"error": error_msg}
                if isinstance(error, Exception):
                    context = getattr(error, "context", None)
                    if isinstance(context, dict) and "raw_content" in context:
                        failure_record["raw_content"] = context["raw_content"]
                # Include topic_id and path for checkpoint retry functionality
                if topic_paths_for_batch and idx < len(topic_paths_for_batch):
                    tp = topic_paths_for_batch[idx]
                    if tp:
                        failure_record["topic_id"] = tp.topic_id
                        failure_record["path"] = " -> ".join(tp.path)
                failed_responses.append(failure_record)
                failure_type = self.analyze_failure(
                    str(error), error=error if isinstance(error, Exception) else None
                )
                self.failure_analysis[failure_type].append(error_msg)

                # Classify and emit error to progress reporter
                classified = classify_error(
                    error if isinstance(error, Exception) else str(error),
                    provider=self.provider,
                    context={"error_type": failure_type},
                )
                if self.progress_reporter:
                    self.progress_reporter.emit_error(
                        classified,
                        sample_idx=start_sample_idx + idx,
                    )

        return samples, failed_responses

    def analyze_failure(self, response_content: str, error: Exception | None = None) -> str:
        """Analyze the failure reason for a sample."""
        if error:
            error_str = str(error)
            if "schema" in error_str.lower():
                return "invalid_schema"
            if any(api_err in error_str.lower() for api_err in API_ERROR_INDICATORS):
                return "api_errors"
            return "other_errors"

        if not response_content or response_content.isspace():
            return "empty_responses"

        # Check if response seems to be attempting JSON but failing
        if any(char in response_content for char in "{}[]"):
            return "json_parsing_errors"
        return "malformed_responses"

    def summarize_failures(self) -> dict:
        """Generate a summary of all failures including those flushed to checkpoint."""
        # Include both in-memory and flushed failures for accurate total
        total_failures = self._flushed_failures_count + len(self.failed_samples)
        summary = {
            "total_failures": total_failures,
            "failure_types": {k: len(v) for k, v in self.failure_analysis.items()},
            "failure_examples": {},
        }

        # Add example failures for each category
        for _category, failures in self.failure_analysis.items():
            if failures:
                # Get up to 3 examples for each category
                examples = failures[:3]
                summary["failure_examples"].append(
                    (
                        str(ex)[:200] + "..."
                        if len(str(ex)) > 200  # noqa: PLR2004
                        else str(ex)
                    )
                    for ex in examples
                )
        return summary

    def create_data(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
        *,
        num_samples: int | None = None,
    ):
        ensure_not_running_loop("DataSetGenerator.create_data")
        return asyncio.run(
            self.create_data_async(
                num_steps=num_steps,
                num_example_demonstrations=num_example_demonstrations,
                batch_size=batch_size,
                topic_model=topic_model,
                model_name=model_name,
                sys_msg=sys_msg,
                num_samples=num_samples,
            )
        )

    def create_data_with_events(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
        *,
        num_samples: int | None = None,
    ):
        ensure_not_running_loop("DataSetGenerator.create_data_with_events")

        async def _async_generator() -> AsyncGenerator[dict | HFDataset, None]:
            async for event in self.create_data_with_events_async(
                num_steps=num_steps,
                num_example_demonstrations=num_example_demonstrations,
                batch_size=batch_size,
                topic_model=topic_model,
                model_name=model_name,
                sys_msg=sys_msg,
                num_samples=num_samples,
            ):
                yield event

        agen = _async_generator()

        def _sync_generator():
            loop = asyncio.new_event_loop()
            try:
                while True:
                    try:
                        event = loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
                    else:
                        yield event
            finally:
                loop.run_until_complete(agen.aclose())
                loop.close()

        return _sync_generator()

    async def create_data_async(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
        *,
        num_samples: int | None = None,
    ) -> HFDataset:
        """Return a dataset. Explicit num_samples overrides legacy steps, without rounding."""
        result = None
        events = self.create_data_with_events_async(
            num_steps,
            num_example_demonstrations,
            batch_size,
            topic_model,
            model_name,
            sys_msg,
            num_samples=num_samples,
        )
        try:
            async for event in events:
                if isinstance(event, HFDataset):
                    result = event
        finally:
            await events.aclose()
        if result is None:
            raise DataSetGeneratorError("Dataset generation failed")
        trace(
            "dataset_created",
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "conversation_type": self.config.conversation_type,
                "samples_count": len(result),
                "failed_samples": self._flushed_failures_count + len(self.failed_samples),
                "success": len(result) > 0,
            },
        )
        return result

    async def create_data_with_events_async(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
        *,
        num_samples: int | None = None,
    ) -> AsyncGenerator[dict | HFDataset, None]:
        """Yield native allocation/step events and a dataset; cycle output is not synthesized."""
        if self.config.topic_sample_mode != "independent":
            if topic_model is None:
                raise DataSetGeneratorError("Coordinated topic sample modes require a topic model")
            if not (self._supports_bulk_full_records() or self._supports_bulk_custom_seeds()):
                raise DataSetGeneratorError(
                    f"topic_sample_mode={self.config.topic_sample_mode!r} is not implemented yet"
                )
        target = resolve_target(num_samples, num_steps, batch_size)
        if model_name:
            self.model_name = model_name.strip()
        if not self.model_name:
            raise DataSetGeneratorError("model_name must not be empty")
        if topic_model is not None:
            plan = self._prepare_topic_allocations(target, topic_model)
        else:
            plan = {"generation_mode": "step_based", "target": target}
        self._prepare_generation_run(plan)
        if num_samples is not None and topic_model is None and target % batch_size:
            logger.info(
                "Final step will generate %d samples (batch_size=%d) to match num_samples=%d.",
                target % batch_size,
                batch_size,
                target,
            )
        params = {
            "plan": plan,
            "batch_size": batch_size,
            "data_creation_prompt": self._get_prompt_template(),
            "num_example_demonstrations": num_example_demonstrations,
            "include_sys_msg": sys_msg if sys_msg is not None else self.config.sys_msg,
            "root_topic_prompt": getattr(topic_model, "topic_prompt", None),
            "topic_model_type": type(topic_model).__name__.lower()
            if topic_model is not None
            else None,
        }
        runner = (
            self._run_topic_based_generation_async
            if topic_model is not None
            else self._run_step_based_generation_async
        )
        events = runner(**params)
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()

    def _prepare_generation_run(self, plan: dict) -> None:
        """Consume explicit resume state once; ordinary invocations start independent runs."""
        resuming = self._checkpoint_loaded
        loaded_path = self._checkpoint_metadata_path if resuming else None
        if resuming and self._generation_plan != plan:
            raise DataSetGeneratorError(
                "Checkpoint target/topic plan differs from the requested generation plan"
            )
        if resuming and self.config.checkpoint_interval is None:
            raise DataSetGeneratorError(
                "Checkpointing must remain enabled to resume a loaded checkpoint"
            )
        if self.config.checkpoint_interval is not None:
            if resuming and self._get_checkpoint_paths()[0] != loaded_path:
                raise DataSetGeneratorError(
                    "Checkpoint location differs from the loaded checkpoint"
                )
            self._initialize_checkpoint_paths()
            if self._checkpoint_metadata_path.exists():
                with self._checkpoint_metadata_path.open(encoding="utf-8") as stream:
                    existing = json.load(stream)
                if existing.get("version") != CHECKPOINT_VERSION:
                    raise DataSetGeneratorError(
                        version_error(existing.get("version"), self._checkpoint_metadata_path)
                    )
                if not resuming:
                    raise DataSetGeneratorError(
                        "Existing checkpoint: call load_checkpoint() to resume, "
                        "or use a separate checkpoint location for a new run"
                    )
            elif resuming:
                raise DataSetGeneratorError("Loaded checkpoint metadata is missing")
            elif any(
                path.exists() and path.stat().st_size
                for path in (self._checkpoint_samples_path, self._checkpoint_failures_path)
            ):
                raise DataSetGeneratorError(
                    "Checkpoint outcome files exist without metadata; use a separate location"
                )
        if not resuming:
            self._samples.clear()
            self.failed_samples.clear()
            self._completed.clear()
            self._sample_slots.clear()
            self._topic_seeds.clear()
            self._collection_attempts.clear()
            self._flushed_samples_count = 0
            self._flushed_failures_count = 0
            self.stop_requested = False
            if self.config.checkpoint_interval is None:
                self._checkpoint_metadata_path = None
                self._checkpoint_samples_path = None
                self._checkpoint_failures_path = None
        self.failure_analysis = {category: [] for category in ERROR_CATEGORIES}
        self._generation_plan = plan
        # Resume is explicit for each invocation; saving a checkpoint does not select resume.
        self._checkpoint_loaded = False
        self._warn_bulk_token_budget(plan)

    async def _generate_sample_outcome(
        self,
        prompt: str,
        include_sys_msg: bool,
        sample_idx: int,
        topic_path: TopicPath | None,
        identity: tuple[str, int],
    ) -> tuple[dict | None, dict | None]:
        """Generate owned results; only the coordinator mutates persisted sample buffers."""
        error = None
        for attempt in range(self.config.max_retries):
            try:
                samples, failures = await self._generate_structured_samples_async(
                    [prompt], include_sys_msg, sample_idx, [topic_path]
                )
                if samples:
                    return samples[0].model_dump(exclude_none=True), None
                error = failures[0] if failures else {"error": "Generation returned no sample"}
                break
            except DataSetGeneratorError as exc:
                error = {"error": str(exc)}
                break
            except Exception as exc:  # noqa: BLE001
                error = {"error": str(exc)}
                if attempt + 1 < self.config.max_retries:
                    continue
        failure = dict(error or {"error": "Generation failed"})
        failure.update(topic_id=identity[0], sample_ordinal=identity[1])
        if topic_path is not None:
            failure["path"] = " -> ".join(topic_path.path)
        return None, failure

    async def _checkpoint_topic_state(
        self,
        queue: asyncio.Queue,
        topic_id: str,
        *,
        seeds: dict[str, Any] | None = None,
        attempts: int | None = None,
    ) -> None:
        """Wait for coordinator ownership and persistence before continuing a shared call.

        Seeds map string ordinals to pinned JSON values. Attempts is the cumulative
        collection count, including unsuccessful and partial requests. Neither settles slots.
        """
        acknowledged = asyncio.get_running_loop().create_future()
        await queue.put(("topic_state", (topic_id, seeds, attempts, acknowledged)))
        await acknowledged

    async def _generate_topic_samples_async(
        self,
        allocation: dict,
        offset: int,
        queue: asyncio.Queue,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
    ) -> None:
        """Dispatch a quota without changing topic scheduling or slot accounting."""
        if self._supports_bulk_full_records() or self._supports_bulk_custom_seeds():
            await self._generate_bulk_topic_records_async(
                allocation, queue, num_example_demonstrations, include_sys_msg, offset=offset
            )
            return
        if self.config.topic_sample_mode != "independent":
            await queue.put(
                (
                    "error",
                    DataSetGeneratorError(
                        f"topic_sample_mode={self.config.topic_sample_mode!r} is not implemented yet"
                    ),
                )
            )
            await queue.put(("done", None))
            return
        await self._generate_independent_topic_samples_async(
            allocation,
            offset,
            queue,
            data_creation_prompt,
            num_example_demonstrations,
            include_sys_msg,
        )

    def _supports_bulk_full_records(self) -> bool:
        """Internal execution capability; configuration still gates full scope."""
        return (
            self.config.topic_sample_mode == "bulk"
            and self.config.shared_scope == "full"
            and not (self.config.tool_components or self.config.custom_tools)
        )

    def _supports_bulk_custom_seeds(self) -> bool:
        return (
            self.config.topic_sample_mode == "bulk"
            and self.config.shared_scope == "seed"
            and self.config.output_schema is not None
            and self.config.seed_field is not None
        )

    def _warn_bulk_token_budget(self, plan: dict) -> None:
        """Warn once per resolved plan using a coarse, count-based output estimate."""
        if not self._supports_bulk_full_records() or plan["generation_mode"] != "topic_allocation":
            return
        completed_counts = Counter(topic_id for topic_id, _ in self._completed)
        quota = max(
            (
                allocation["quota"] - completed_counts[allocation["uuid"]]
                for allocation in plan["allocations"]
            ),
            default=0,
        )
        estimated = quota * estimate_record_tokens(self.config.output_schema)
        if estimated > self.config.max_tokens:
            logger.warning(
                "bulk with shared_scope: full may exceed max_tokens=%d: up to %d outstanding "
                "records per topic (coarse estimate %d output tokens). Raise max_tokens, "
                "lower the quota, or use shared_scope: seed. The token budget is unchanged.",
                self.config.max_tokens,
                quota,
                estimated,
            )

    def _prepare_bulk_record(
        self, item: Any, validator, topic_id: str, include_sys_msg: bool
    ) -> dict | None:
        """Validate a whole record and assemble conversation-only shipping fields."""
        if not isinstance(item, dict) or not validator.is_valid(item):
            return None
        if self.config.output_schema is not None:
            return item
        try:
            conversation = Conversation.model_validate(item)
        except ValidationError:
            return None
        messages = conversation.messages
        if (
            len(messages) < 2  # noqa: PLR2004 - a user request and assistant reply
            or messages[0].role != "user"
            or messages[-1].role != "assistant"
            or any(message.role not in ("user", "assistant") for message in messages)
        ):
            return None
        if self.config.conversation_type == "cot" and (
            conversation.reasoning is None
            or conversation.reasoning.style != (self.config.reasoning_style or "freetext")
        ):
            return None
        if include_sys_msg:
            conversation.messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=self.dataset_system_prompt,
                ),
            )
        conversation.metadata = {**(conversation.metadata or {}), "topic_id": topic_id}
        return conversation.model_dump(exclude_none=True)

    def _bulk_record_instructions(self) -> str:
        if self.config.output_schema is not None:
            return "Each item is one complete record matching the supplied item schema."
        instructions = (
            "Each item is a complete conversation: start with a user request and end with "
            "the assistant's reply. Use only user and assistant messages. Do not generate "
            "system messages; these are added in code."
        )
        if self.config.conversation_type == "cot":
            style = self.config.reasoning_style or "freetext"
            content_format = (
                "a plain string" if style == "freetext" else "a list of reasoning steps"
            )
            instructions += (
                f" Include a reasoning trace in every item with style {style} "
                f"and content as {content_format}."
            )
        return instructions

    def _saved_topic_outcomes(self, topic_id: str) -> list[tuple[int, dict]]:
        """Read successful siblings using checkpoint identities, never row content."""
        if self._checkpoint_samples_path is None:
            return []
        rows = islice(iter_jsonl(self._checkpoint_samples_path), self._flushed_samples_count)
        return [
            (slot[1], row)
            for slot, row in zip(self._sample_slots, rows, strict=False)
            if slot[0] == topic_id
        ]

    def _saved_topic_records(self, topic_id: str) -> list[dict]:
        return [row for _, row in self._saved_topic_outcomes(topic_id)]

    async def _generate_bulk_topic_records_async(
        self,
        allocation: dict,
        queue: asyncio.Queue,
        num_example_demonstrations: int,
        include_sys_msg: bool,
        *,
        offset: int = 0,
    ) -> None:
        """Collect full records or pinned seeds using one bounded topic budget."""
        try:
            await self._collect_bulk_topic_records(
                allocation, queue, num_example_demonstrations, include_sys_msg
            )
            if self.config.shared_scope == "seed" and not self.stop_requested:
                await self._complete_bulk_topic_seeds(
                    allocation, offset, queue, num_example_demonstrations
                )
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
            await queue.put(("error", exc))
        await queue.put(("done", None))

    async def _collect_bulk_topic_records(
        self,
        allocation: dict,
        queue: asyncio.Queue,
        num_example_demonstrations: int,
        include_sys_msg: bool,
    ) -> None:
        topic_id = allocation["uuid"]
        seed_scope = self.config.shared_scope == "seed"
        saved_seeds = self._topic_seeds.get(topic_id, {}) if seed_scope else {}
        outstanding = deque(
            ordinal
            for ordinal in range(allocation["quota"])
            if (topic_id, ordinal) not in self._completed
            and (not seed_scope or str(ordinal) not in saved_seeds)
        )
        if not outstanding or self.stop_requested:
            return
        schema = (
            seed_value_schema(self.config.output_schema, self.config.seed_field)
            if seed_scope
            else self.config.output_schema
            if self.config.output_schema is not None
            else Conversation.model_json_schema()
        )
        validator = record_validator(schema)
        collection_model = make_collection_model(schema)
        if seed_scope:
            context = {
                ordinal: row[self.config.seed_field]
                for ordinal, row in self._saved_topic_outcomes(topic_id)
                if self.config.seed_field in row
            }
            context.update({int(ordinal): value for ordinal, value in saved_seeds.items()})
            accepted_siblings = [value for _, value in sorted(context.items())]
        else:
            accepted_siblings = self._saved_topic_records(topic_id)
        attempts = self._collection_attempts.get(topic_id, 0)
        budget = self.config.sample_retries + 1
        last_error = "Bulk collection attempt budget exhausted"
        while outstanding and attempts < budget and not self.stop_requested:
            requested = len(outstanding)
            attempts += 1
            # Reserve an attempt durably before the request, including interrupted calls.
            await self._checkpoint_topic_state(queue, topic_id, attempts=attempts)
            if self.stop_requested:
                return
            common = {
                "requested": requested,
                "topic_path": " -> ".join(allocation["path"]),
                "instructions": self.build_custom_instructions_text(),
                "examples": self.build_examples_text(num_example_demonstrations),
                "accepted_siblings": json.dumps(accepted_siblings, ensure_ascii=False),
            }
            if seed_scope:
                prompt = BULK_SEED_SAMPLE_PROMPT.format(
                    **common,
                    seed_field=json.dumps(self.config.seed_field),
                    tools=self.build_tools_text(),
                )
            else:
                prompt = BULK_FULL_SAMPLE_PROMPT.format(
                    **common,
                    record_instructions=self._bulk_record_instructions(),
                    system_prompt=self.dataset_system_prompt,
                )
            event = {
                "topic_id": topic_id,
                "requested": requested,
                "accepted": 0,
                "missing": requested,
                "attempt": attempts,
                "budget": budget,
            }
            await queue.put(("event", {"event": "seed_request", **event}))
            retryable = True
            accepted = 0
            pending_seeds = {}
            response_received = False
            unit = "seeds" if seed_scope else "samples"
            step_label = (
                f"Generating {requested} {unit} for topic {topic_id} (attempt {attempts}/{budget})"
            )
            step_context = {
                "topic_path": allocation["path"],
                "conversation_type": self.config.conversation_type,
            }
            if self.progress_reporter:
                self.progress_reporter.emit_step_start(step_label, **event, **step_context)
            try:
                response = await self.llm_client.generate_async(
                    prompt,
                    collection_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                response_received = True
                simulate_stream(
                    self.progress_reporter,
                    response.model_dump_json(indent=2),
                    source="bulk_collection",
                    topic_id=topic_id,
                    attempt=attempts,
                )
                for item in response.items:
                    if not outstanding:
                        break
                    if seed_scope:
                        if not validator.is_valid(item):
                            continue
                        record = item
                    else:
                        record = self._prepare_bulk_record(
                            item, validator, topic_id, include_sys_msg
                        )
                        if record is None:
                            continue
                    ordinal = outstanding.popleft()
                    accepted_siblings.append(record)
                    accepted += 1
                    if seed_scope:
                        pending_seeds[str(ordinal)] = record
                    else:
                        await queue.put(("outcome", ((topic_id, ordinal), record, None)))
                last_error = "Bulk collection returned fewer valid records than requested"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                retryable = is_validation_error(exc)
            finally:
                if self.progress_reporter:
                    self.progress_reporter.emit_step_complete(
                        step_label,
                        **{
                            **event,
                            **step_context,
                            "accepted": accepted,
                            "missing": len(outstanding),
                            "success": accepted > 0,
                            "response_received": response_received,
                        },
                    )
            # Seeds and accepted rows must be durable before retries or completions.
            await self._checkpoint_topic_state(
                queue, topic_id, attempts=attempts, seeds=pending_seeds if seed_scope else None
            )
            await queue.put(
                (
                    "event",
                    {
                        "event": "seed_response",
                        **event,
                        "accepted": accepted,
                        "missing": len(outstanding),
                    },
                )
            )
            if outstanding:
                will_retry = retryable and attempts < budget and not self.stop_requested
                logger.warning(
                    "Bulk response for topic %s: requested %d, accepted %d, missing %d; "
                    "attempt %d/%d. %s",
                    topic_id,
                    requested,
                    accepted,
                    len(outstanding),
                    attempts,
                    budget,
                    "Retrying outstanding slots." if will_retry else "No retries remain.",
                )
                if not retryable:
                    break
        if outstanding and not self.stop_requested:
            logger.warning(
                "Bulk generation for topic %s stopped after %d/%d attempts: "
                "accepted %d/%d %s; %d slots marked failed.",
                topic_id,
                attempts,
                budget,
                len(accepted_siblings),
                allocation["quota"],
                "seeds" if seed_scope else "samples",
                len(outstanding),
            )
            for ordinal in outstanding:
                failure = {
                    "topic_id": topic_id,
                    "sample_ordinal": ordinal,
                    "path": " -> ".join(allocation["path"]),
                    "error": last_error,
                }
                await queue.put(("outcome", ((topic_id, ordinal), None, failure)))

    async def _complete_bulk_topic_seeds(
        self,
        allocation: dict,
        offset: int,
        queue: asyncio.Queue,
        num_example_demonstrations: int,
    ) -> None:
        topic_id = allocation["uuid"]
        seeds = self._topic_seeds.get(topic_id, {})
        schema = completion_schema(self.config.output_schema, self.config.seed_field)
        model = make_completion_model(schema)
        remainder_validator = record_validator(schema)
        full_validator = record_validator(self.config.output_schema)
        for ordinal in range(allocation["quota"]):
            if self.stop_requested:
                break
            identity = (topic_id, ordinal)
            if identity in self._completed or str(ordinal) not in seeds:
                continue
            sample, failure = await self._complete_custom_seed_outcome(
                allocation,
                ordinal,
                offset + ordinal,
                seeds[str(ordinal)],
                model,
                remainder_validator,
                full_validator,
                num_example_demonstrations,
            )
            if sample is None and failure is None:
                break
            await queue.put(("outcome", (identity, sample, failure)))
            await asyncio.sleep(0)

    async def _complete_custom_seed_outcome(
        self,
        allocation: dict,
        ordinal: int,
        sample_idx: int,
        seed: Any,
        model,
        remainder_validator,
        full_validator,
        num_example_demonstrations: int,
    ) -> tuple[dict | None, dict | None]:
        """Complete one pinned seed with a fresh, independent slot retry budget."""
        seed_field = self.config.seed_field
        budget = self.config.sample_retries + 1
        last_error = "Seed completion failed"
        pinned = {seed_field: seed}
        if not model.model_json_schema().get("properties") and full_validator.is_valid(pinned):
            return pinned, None
        for attempt in range(budget):
            if self.stop_requested:
                return None, None
            prompt = BULK_COMPLETION_PROMPT.format(
                system_prompt=self.dataset_system_prompt,
                seed_field=json.dumps(seed_field),
                seed=json.dumps(seed, ensure_ascii=False),
                topic_path=(
                    f"<topic_path>\n{' -> '.join(allocation['path'])}\n</topic_path>"
                    if self.config.completion_includes_topic_path
                    else ""
                ),
                instructions=self.build_custom_instructions_text(),
                examples=self.build_examples_text(num_example_demonstrations),
                tools=self.build_tools_text(),
                retry_feedback=f"RETRY: {last_error}" if attempt else "",
            )
            label = f"Completing sample {sample_idx + 1} (attempt {attempt + 1}/{budget})"
            metadata = {
                "sample_idx": sample_idx + 1,
                "topic_id": allocation["uuid"],
                "sample_ordinal": ordinal,
                "topic_path": allocation["path"],
            }
            success = False
            if self.progress_reporter:
                self.progress_reporter.emit_step_start(label, **metadata)
            try:
                result = await self.llm_client.generate_async(
                    prompt,
                    model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                remainder = result.model_dump()
                if seed_field in remainder or not remainder_validator.is_valid(remainder):
                    raise ValueError(
                        "Completion validation error: invalid remainder or repeated seed field"
                    )
                sample = {**remainder, **pinned}
                if not full_validator.is_valid(sample):
                    raise ValueError(
                        "Completion validation error: assembled record does not match output_schema"
                    )
                simulate_stream(
                    self.progress_reporter,
                    json.dumps(sample, ensure_ascii=False),
                    source="bulk_completion",
                    sample_idx=sample_idx + 1,
                )
                success = True
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if self.stop_requested:
                    return None, None
                if is_validation_error(exc) and attempt + 1 < budget:
                    self._emit_retry(sample_idx, attempt, budget, exc)
                    continue
                break
            else:
                return sample, None
            finally:
                if self.progress_reporter:
                    self.progress_reporter.emit_step_complete(label, **metadata, success=success)
        return None, {
            "topic_id": allocation["uuid"],
            "sample_ordinal": ordinal,
            "path": " -> ".join(allocation["path"]),
            "error": last_error,
        }

    async def _generate_independent_topic_samples_async(
        self,
        allocation: dict,
        offset: int,
        queue: asyncio.Queue,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
    ) -> None:
        """Fulfill an independent quota with one owned outcome per slot."""
        try:
            for ordinal in range(allocation["quota"]):
                identity = (allocation["uuid"], ordinal)
                if identity in self._completed:
                    continue
                if self.stop_requested:
                    break
                topic_path = (
                    TopicPath(path=allocation["path"], topic_id=allocation["uuid"])
                    if allocation["path"] is not None
                    else None
                )
                prompt = self.build_prompt(
                    data_creation_prompt=data_creation_prompt,
                    num_example_demonstrations=num_example_demonstrations,
                    subtopics_list=allocation["path"],
                )
                sample, failure = await self._generate_sample_outcome(
                    prompt, include_sys_msg, offset + ordinal, topic_path, identity
                )
                await queue.put(("outcome", (identity, sample, failure)))
                # Let the result coordinator checkpoint and observe stop requests between slots.
                await asyncio.sleep(0)
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
            await queue.put(("error", exc))
        await queue.put(("done", None))

    async def _run_topic_based_generation_async(self, *, plan: dict, **kwargs):
        """Generate topic quotas in fixed concurrent groups, one request per topic at a time."""
        events = self._run_allocated_generation_async(
            plan=plan, allocations=[a for a in plan["allocations"] if a["quota"]], **kwargs
        )
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()

    async def _run_step_based_generation_async(self, *, plan: dict, batch_size: int, **kwargs):
        """Generate sequential sample batches without a topic model, including a short final step."""
        # Each slot is a one-sample task; identity uses the global no-topic ordinal.
        allocations = range(plan["target"])
        events = self._run_allocated_generation_async(
            plan=plan, allocations=allocations, batch_size=batch_size, **kwargs
        )
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()

    async def _run_allocated_generation_async(
        self,
        *,
        plan: dict,
        allocations,
        batch_size: int,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
        root_topic_prompt: str | None = None,
        topic_model_type: str | None = None,
    ):
        """Coordinate owned slot outcomes, persistence, and lifecycle for both strategies."""
        topic_based = plan["generation_mode"] == "topic_allocation"
        historical_samples = self._flushed_samples_count + len(self._samples)
        historical_failures = self._flushed_failures_count + len(self.failed_samples)
        skipped = len(self._completed)
        pending_samples, pending_failures, pending_slots, pending_completed = [], [], [], []
        tasks = []
        queue = None
        recovery_handled = False
        new_samples = new_failures = 0
        metadata_dirty = False

        def record_outcome(payload):
            nonlocal new_samples, new_failures
            identity, sample, failure = payload
            self._completed.add(identity)
            pending_completed.append(identity)
            if sample is not None:
                self._samples.append(sample)
                pending_samples.append(sample)
                pending_slots.append(identity)
                new_samples += 1
            else:
                self.failed_samples.append(failure)
                pending_failures.append(failure)
                new_failures += 1

        def record_topic_state(payload):
            nonlocal metadata_dirty
            topic_id, seeds, attempts, _ = payload
            if seeds is not None:
                self._topic_seeds.setdefault(topic_id, {}).update(seeds)
            if attempts is not None:
                self._collection_attempts[topic_id] = attempts
            metadata_dirty = True

        async def stop_workers(*, collect_outcomes=False):
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()
            if collect_outcomes and queue is not None:
                while not queue.empty():
                    kind, payload = queue.get_nowait()
                    if kind == "outcome":
                        record_outcome(payload)
                    elif kind == "topic_state":
                        record_topic_state(payload)

        def flush():
            nonlocal metadata_dirty
            if self.config.checkpoint_interval is None or not (pending_completed or metadata_dirty):
                return False
            self._save_checkpoint(
                pending_samples, pending_failures, pending_completed, sample_slots=pending_slots
            )
            pending_samples.clear()
            pending_failures.clear()
            pending_slots.clear()
            pending_completed.clear()
            metadata_dirty = False
            return True

        def counts():
            return {
                "total_samples": historical_samples + new_samples,
                "failed_samples": historical_failures + new_failures,
                "expected_samples": plan["target"],
                "skipped_slots": skipped,
                "new_samples": new_samples,
                "new_failures": new_failures,
                "unaccounted": plan["target"]
                - historical_samples
                - historical_failures
                - new_samples
                - new_failures,
            }

        yield {
            "event": "generation_start",
            "model_name": self.model_name,
            **{k: v for k, v in plan.items() if k != "allocations"},
            "total_samples": plan["target"],
            "batch_size": batch_size,
            "concurrency": batch_size,
            **(
                {"num_steps": (plan["target"] + batch_size - 1) // batch_size}
                if not topic_based
                else {}
            ),
            "resumed_samples": historical_samples,
            "resumed_failures": historical_failures,
            "outstanding_slots": plan["target"] - skipped,
            "resumed_from_checkpoint": bool(skipped),
            "previously_processed": skipped,
            "checkpoint_enabled": self.config.checkpoint_interval is not None,
            "checkpoint_interval": self.config.checkpoint_interval,
            "root_topic_prompt": root_topic_prompt,
            "topic_model_type": topic_model_type,
        }
        try:
            offset = 0
            # The task count is bounded by batch_size, never by the total sample target.
            for group_start in range(0, len(allocations), batch_size):
                if self.stop_requested:
                    break
                group = allocations[group_start : group_start + batch_size]
                queue = asyncio.Queue(maxsize=batch_size)
                tasks = []
                step_samples, step_failures = new_samples, new_failures
                step = group_start // batch_size + 1
                if topic_based:
                    yield {
                        "event": "topic_group_start",
                        "group": step,
                        "total_groups": (len(allocations) + batch_size - 1) // batch_size,
                    }
                else:
                    yield {
                        "event": "step_start",
                        "step": step,
                        "total_steps": (plan["target"] + batch_size - 1) // batch_size,
                        "batch_size": len(group),
                    }
                for group_item in group:
                    allocation = group_item if topic_based else {"ordinal": group_item, "quota": 1}
                    if topic_based:
                        tasks.append(
                            asyncio.create_task(
                                self._generate_topic_samples_async(
                                    allocation,
                                    offset,
                                    queue,
                                    data_creation_prompt,
                                    num_example_demonstrations,
                                    include_sys_msg,
                                )
                            )
                        )
                    else:

                        async def generate_slot(a=allocation, result_queue=queue):
                            identity = (STEP_TOPIC_ID, a["ordinal"])
                            try:
                                if identity not in self._completed and not self.stop_requested:
                                    prompt = self.build_prompt(
                                        data_creation_prompt, num_example_demonstrations
                                    )
                                    sample, failure = await self._generate_sample_outcome(
                                        prompt, include_sys_msg, a["ordinal"], None, identity
                                    )
                                    await result_queue.put(("outcome", (identity, sample, failure)))
                            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
                                await result_queue.put(("error", exc))
                            await result_queue.put(("done", None))

                        tasks.append(asyncio.create_task(generate_slot()))
                    offset += allocation["quota"]
                done = 0
                while done < len(tasks):
                    kind, payload = await queue.get()
                    if kind == "done":
                        done += 1
                        continue
                    if kind == "error":
                        raise payload
                    if kind == "event":
                        yield payload
                        continue
                    if kind == "topic_state":
                        record_topic_state(payload)
                        # Flush pending rows too: metadata must never reference unwritten rows.
                        saved = flush()
                        acknowledged = payload[3]
                        if not acknowledged.done():
                            acknowledged.set_result(None)
                        if saved:
                            yield {
                                "event": "checkpoint_saved",
                                "total_samples": self._flushed_samples_count,
                                "total_failures": self._flushed_failures_count,
                            }
                        continue
                    identity, sample, failure = payload
                    record_outcome(payload)
                    yield {
                        "event": "batch_complete",
                        "topic_id": identity[0],
                        "sample_ordinal": identity[1],
                        "samples_generated": int(sample is not None),
                        "samples_failed": int(failure is not None),
                    }
                    if (
                        self.config.checkpoint_interval is not None
                        and len(pending_completed) >= self.config.checkpoint_interval
                        and flush()
                    ):
                        yield {
                            "event": "checkpoint_saved",
                            "total_samples": self._flushed_samples_count,
                            "total_failures": self._flushed_failures_count,
                        }
                await asyncio.gather(*tasks)
                tasks = []
                if not topic_based:
                    yield {
                        "event": "step_complete",
                        "step": step,
                        "samples_generated": new_samples - step_samples,
                        "progress_reported": True,
                        "successful_in_step": new_samples - step_samples,
                        "failed_in_step": new_failures - step_failures,
                        "batch_size": len(group),
                    }
            if flush():
                yield {
                    "event": "checkpoint_saved",
                    "final": True,
                    "total_samples": self._flushed_samples_count,
                    "total_failures": self._flushed_failures_count,
                }
            if self.stop_requested:
                yield {
                    "event": "generation_stopped",
                    **counts(),
                    "total_failures": historical_failures + new_failures,
                }
            else:
                yield {"event": "generation_complete", **counts()}
        except (Exception, KeyboardInterrupt) as exc:
            await stop_workers(collect_outcomes=True)
            # Cleanup/export failures must not replace the generation error's original cause.
            recovery_handled = True
            try:
                flush()
            except Exception:  # noqa: BLE001
                logger.exception("Unable to save checkpoint while recovering partial generation")
            interrupted = isinstance(exc, KeyboardInterrupt)
            fallback = INTERRUPTED_DATASET_FILENAME if interrupted else ERROR_DATASET_FILENAME
            try:
                self._save_samples_to_file(fallback, include_checkpoint=True)
            except Exception:  # noqa: BLE001
                logger.exception("Unable to save partial dataset to %s", fallback)
            self.print_failure_summary()
            yield {
                "event": "generation_interrupted" if interrupted else "generation_error",
                "message": "Generation interrupted by user." if interrupted else str(exc),
                "error": str(exc),
                **counts(),
            }
            if not interrupted:
                raise DataSetGeneratorError("failed") from exc
        finally:
            await stop_workers()
            if not recovery_handled:
                flush()
        samples = list(self._samples)
        if self._flushed_samples_count:
            saved = islice(iter_jsonl(self._checkpoint_samples_path), self._flushed_samples_count)
            samples = list(saved) + samples
        yield HFDataset.from_list(samples)

    async def _process_batch_with_retries_async(
        self,
        prompts: list[str],
        include_sys_msg: bool,
        start_sample_idx: int = 0,
        topic_paths_for_batch: list[TopicPath | None] | None = None,
    ) -> tuple[bool, int]:
        """Process a batch with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                samples, failed_responses = await self._generate_structured_samples_async(
                    prompts, include_sys_msg, start_sample_idx, topic_paths_for_batch
                )

                # Verify all prompts are accounted for (no silent drops)
                accounted = len(samples) + len(failed_responses)
                if accounted != len(prompts):
                    logger.warning(
                        "Sample accounting mismatch: %d prompts, %d samples, %d failures (missing %d)",
                        len(prompts),
                        len(samples),
                        len(failed_responses),
                        len(prompts) - accounted,
                    )

                # Update failed samples
                self.failed_samples.extend(failed_responses)

                if samples:
                    # Convert Pydantic models to dicts and add to samples list
                    sample_dicts = [s.model_dump(exclude_none=True) for s in samples]
                    self._samples.extend(sample_dicts)
                    return True, len(samples)  # Success - exit retry loop

            except DataSetGeneratorError as e:
                # Authentication and API errors are now wrapped in DataSetGeneratorError
                error_str = str(e).lower()
                if any(
                    keyword in error_str
                    for keyword in [
                        "api_key",
                        "api key",
                        "authentication",
                        "unauthorized",
                    ]
                ):
                    error_msg = f"Authentication failed for provider '{self.provider}'. Please set the required API key environment variable."
                    self.failure_analysis["authentication_error"].append(error_msg)
                else:
                    error_msg = f"API error for provider '{self.provider}': {str(e)[:100]}..."
                    self.failure_analysis["api_errors"].append(error_msg)

                # Build failure records for each topic path in the batch
                if topic_paths_for_batch:
                    for tp in topic_paths_for_batch:
                        failure_record: dict[str, str | None] = {"error": error_msg}
                        if tp:
                            failure_record["topic_id"] = tp.topic_id
                            failure_record["path"] = " -> ".join(tp.path)
                        self.failed_samples.append(failure_record)
                else:
                    self.failed_samples.append({"error": error_msg})
                logger.exception("API error: %s", error_msg)
                return False, 0  # Don't retry authentication/API errors
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    error_msg = str(e)
                    # Build failure records for each topic path in the batch
                    if topic_paths_for_batch:
                        for tp in topic_paths_for_batch:
                            failure_record_exc: dict[str, str | None] = {"error": error_msg}
                            if tp:
                                failure_record_exc["topic_id"] = tp.topic_id
                                failure_record_exc["path"] = " -> ".join(tp.path)
                            self.failed_samples.append(failure_record_exc)
                    else:
                        self.failed_samples.append({"error": error_msg})
                    failure_type = self.analyze_failure(error_msg, error=e)
                    self.failure_analysis[failure_type].append(error_msg)
                    return False, 0
            else:
                # If no exception and no samples, return False, 0
                return False, 0

        return False, 0

    def print_failure_summary(self):
        """Print a detailed summary of all failures."""
        summary = self.summarize_failures()

        print("\n=== Failure Analysis Summary ===")
        print(f"Total Failed Samples: {summary['total_failures']}")
        print("\nFailure Types Breakdown:")
        for failure_type, count in summary["failure_types"].items():
            if count > 0:
                print(f"\n{failure_type.replace('_', ' ').title()}: {count}")
                if failure_type in summary["failure_examples"]:
                    print("Example failures:")
                    for i, example in enumerate(
                        summary["failure_examples"].get(failure_type, []), 1
                    ):
                        print(f"  {i}. {example}")
        print("\n=============================")

    def build_prompt(
        self,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        subtopics_list: list[str] | None = None,
    ) -> str:
        prompt = data_creation_prompt.replace("{{{{system_prompt}}}}", self.generation_prompt)
        prompt = prompt.replace("{{{{instructions}}}}", self.build_custom_instructions_text())
        prompt = prompt.replace(
            "{{{{examples}}}}", self.build_examples_text(num_example_demonstrations)
        )
        return prompt.replace("{{{{subtopics}}}}", self.build_subtopics_text(subtopics_list))

    def build_system_prompt(self):
        """Return the original system prompt for dataset inclusion."""
        return self.dataset_system_prompt

    def build_custom_instructions_text(self) -> str:
        if self.config.instructions is None or self.config.instructions == "":
            return ""
        return f"\nHere are additional instructions:\n<instructions>\n{self.config.instructions}\n</instructions>\n"

    def build_examples_text(self, num_example_demonstrations: int):
        if self.config.example_data is None or num_example_demonstrations == 0:
            return ""
        # Bandit: not a security function
        # HF Dataset supports len() and indexing, convert to list for sampling
        example_list = list(self.config.example_data)
        examples = random.sample(example_list, min(num_example_demonstrations, len(example_list)))  # nosec
        examples_text = "Here are output examples:\n\n"
        examples_text += "\n".join(f"Example {i + 1}: \n\n{ex}\n" for i, ex in enumerate(examples))
        return f"\nHere are output examples:\n<examples>\n{examples_text}\n</examples>\n"

    def build_tools_text(self) -> str:
        """Build formatted tools text for XLAM prompts."""
        if not self.tool_registry:
            return "No tools available"

        tools_text = []
        for tool in self.tool_registry.tools:
            params_text = []
            for param in tool.parameters:
                req = " (required)" if param.required else " (optional)"
                params_text.append(f"  - {param.name} ({param.type}){req}: {param.description}")

            tool_text = f"• {tool.name}: {tool.description}\n  Parameters:\n" + "\n".join(
                params_text
            )
            tools_text.append(tool_text)

        return "\n\n".join(tools_text)

    def build_subtopics_text(self, subtopic_list: list[str] | None):
        if subtopic_list is None:
            return ""
        return f"\nLastly, the topic of the training data should be related to the following subtopics: {' -> '.join(subtopic_list)}"

    def _get_cot_prompt_template(self) -> str:  # noqa: PLR0911
        """Get the appropriate prompt template based on modular configuration."""
        # Handle basic conversations
        if self.config.conversation_type == "basic":
            return CONVERSATION_GENERATION_PROMPT

        # Handle chain of thought conversations
        if self.config.conversation_type == "cot":
            # Agent mode with tools - use agent prompts (implicit when tools configured)
            if self.tool_registry:
                # Use agent prompt for single-turn tool calling
                return (
                    AgentPromptBuilder.build_tool_context_prompt(
                        self.tool_registry,
                        max_tools_per_query=self.config.max_tools_per_query,
                    )
                    or AGENT_COT_TOOLS_PROMPT
                )

            # Non-agent CoT - select based on reasoning style
            if self.config.reasoning_style == "freetext":
                return FREETEXT_COT_PROMPT
            if self.config.reasoning_style == "agent":
                return STRUCTURED_COT_PROMPT

        # Fallback to basic conversation prompt
        return CONVERSATION_GENERATION_PROMPT

    def _save_samples_to_file(self, save_path: str, *, include_checkpoint: bool = False):
        """Save in-memory samples, optionally including checkpoint-backed partial results."""

        with open(save_path, "w", encoding="utf-8") as f:
            if include_checkpoint and self._flushed_samples_count:
                for sample in islice(
                    iter_jsonl(self._checkpoint_samples_path), self._flushed_samples_count
                ):
                    f.write(json.dumps(sample, separators=(",", ":")) + "\n")
            for sample in self._samples:
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")

    def save_dataset(self, save_path: str):
        """Save the dataset to a JSONL file."""
        self._save_samples_to_file(save_path)
