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
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal

from datasets import Dataset as HFDataset
from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from .builders import ConversationBuilderFactory, SingleShotBuilder
from .builders_agent import SingleTurnAgentBuilder
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
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
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
    SEQUENTIAL_PROMPT_WARNING_CHARS,
)
from .error_codes import classify_error
from .exceptions import DataSetGeneratorError
from .llm import LLMClient
from .llm.checkpoint_accounting import CheckpointAccounting, validate_accounting
from .llm.client import make_dynamic_model
from .llm.token_usage import TokenUsageTracker, active_token_tracker, track_token_usage
from .metrics import trace
from .progress import ProgressReporter
from .prompt_templates import (
    compatibility_template,
    default_template,
    placeholders,
    render_template,
    template_manifest,
)
from .prompts import (
    AGENT_COT_TOOLS_PROMPT,
    BULK_COMPLETION_PROMPT,
    BULK_CONVERSATION_COMPLETION_PROMPT,
    BULK_CONVERSATION_SEED_PROMPT,
    BULK_FULL_SAMPLE_PROMPT,
    BULK_SEED_SAMPLE_PROMPT,
    CONVERSATION_GENERATION_PROMPT,
    CUSTOM_SCHEMA_PROMPT,
    FREETEXT_COT_PROMPT,
    SEQUENTIAL_SAMPLE_PROMPT,
    STRUCTURED_COT_PROMPT,
    AgentPromptBuilder,
)
from .run_report import generation_scope, record_event
from .sampling import STEP_TOPIC_ID, resolve_target, topic_plan
from .schemas import ChatMessage, Conversation, ToolRegistry, get_conversation_schema
from .stream_simulator import simulate_stream
from .tools import BUILTIN_TOOL_REGISTRY
from .tools.loader import load_tools_from_dict, load_tools_from_endpoint
from .topic_model import TopicModel, TopicPath
from .topic_sampling import TopicSamplingConfig, warn_duplicate_prompt
from .utils import ensure_not_running_loop, get_checkpoint_dir, is_validation_error

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
        description="Base generation allowance; bulk collections scale by requested item count",
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
        self._checkpoint_lock = RLock()
        self._accounting = None
        self._loaded_accounting = None
        self._legacy_accounting = False
        self._accounting_persistence_failed = False
        self._completed: set[tuple[str, int]] = set()  # Attempted (UUID, sample ordinal) slots
        self._sample_slots: list[tuple[str, int]] = []
        self._generation_plan: dict | None = None
        self._topic_seeds: dict[str, dict[str, Any]] = {}
        self._collection_attempts: dict[str, int] = {}
        self._large_sequential_prompt_topics: set[str] = set()
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

        # Create checkpoint directory if needed
        # Use XDG-compliant fallback if checkpoint_path not resolved by CLI
        checkpoint_dir = Path(self.config.checkpoint_path or get_checkpoint_dir(config_path=None))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Derive checkpoint filenames from output filename
        output_stem = Path(self.config.output_save_as).stem
        metadata_path = checkpoint_dir / f"{output_stem}{CHECKPOINT_METADATA_SUFFIX}"
        samples_path = checkpoint_dir / f"{output_stem}{CHECKPOINT_SAMPLES_SUFFIX}"
        failures_path = checkpoint_dir / f"{output_stem}{CHECKPOINT_FAILURES_SUFFIX}"

        return metadata_path, samples_path, failures_path

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

    def _save_checkpoint(self, *args, **kwargs) -> None:
        with self._checkpoint_lock:
            self._save_checkpoint_locked(*args, **kwargs)

    def _save_checkpoint_locked(
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
        with self._checkpoint_lock:
            self._save_checkpoint_metadata_locked()

    def _save_checkpoint_metadata_locked(self) -> None:
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
            "prompt_templates": template_manifest(
                self.config.prompt_templates, self.config.prompt_template
            ),
        }

        if self._accounting is not None:
            metadata["token_accounting"] = self._accounting.snapshot()
        elif self._loaded_accounting is not None:
            metadata["token_accounting"] = self._loaded_accounting

        # Sequential history comes from sample_slots and rows, not a seed map.
        # Retain historical settled seeds when resuming under a different mode.
        if self._topic_seeds or (
            self.config.topic_sample_mode == "bulk" and self.config.shared_scope == "seed"
        ):
            metadata["seeds"] = self._topic_seeds
        self._write_checkpoint_metadata(metadata)

    def _write_checkpoint_metadata(self, metadata: dict) -> None:
        """Publish complete metadata while the shared checkpoint lock is held."""
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

    def _persist_accounting(self, status=None) -> None:
        """Change only accounting in the last durable sample metadata."""
        with self._checkpoint_lock:
            self._accounting.refresh(status)
            if self._checkpoint_metadata_path is None:
                return
            try:
                with self._checkpoint_metadata_path.open(encoding="utf-8") as stream:
                    metadata = json.load(stream)
                metadata["token_accounting"] = self._accounting.snapshot()
                self._write_checkpoint_metadata(metadata)
            except Exception:  # noqa: BLE001 - accounting must not mask provider/sample failures
                self._accounting_persistence_failed = True
                self._accounting.state["invocations"][self._accounting.invocation_id][
                    "persistence_failed"
                ] = True
                logger.exception(
                    "Could not persist observed token usage; sample checkpoint retained"
                )

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

        previous_templates = metadata.get("prompt_templates")
        if previous_templates is not None and previous_templates != template_manifest(
            self.config.prompt_templates, self.config.prompt_template
        ):
            mismatches.append(
                "prompt_templates: resolved content or built-in template hashes changed"
            )
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
        if "token_accounting" in metadata:
            validate_accounting(metadata["token_accounting"])
        self._samples.clear()
        self.failed_samples.clear()
        self.stop_requested = False
        self._accounting = None
        self._loaded_accounting = metadata.get("token_accounting")
        self._legacy_accounting = self._loaded_accounting is None
        self._generation_plan = metadata["plan"]
        self._checkpoint_loaded = True
        self._completed = {tuple(item) for item in metadata["completed"]}
        self._sample_slots = [tuple(item) for item in metadata["sample_slots"]]
        self._flushed_samples_count = metadata["total_samples"]
        failures = read_jsonl(self._checkpoint_failures_path)
        self._flushed_failures_count = len(failures)
        self._topic_seeds = metadata.get("seeds", {})
        self._collection_attempts = metadata.get("collection_attempts", {})
        self._checkpoint_modes_changed = any(
            metadata.get(name, default) != getattr(self.config, name)
            for name, default in (
                ("topic_sample_mode", "independent"),
                ("shared_scope", "seed"),
                ("seed_field", None),
            )
        )
        if self._checkpoint_modes_changed:
            self._topic_seeds = {
                topic_id: {
                    ordinal: seed
                    for ordinal, seed in seeds.items()
                    if (topic_id, int(ordinal)) in self._completed
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
        for failure in failures:
            self._collection_attempts.pop(failure["topic_id"], None)
        if self._checkpoint_modes_changed:
            # Preserve load_checkpoint(True)'s previous mode-change semantics:
            # failed slots lose their incompatible pinned seeds once reopened.
            self._topic_seeds = {
                topic_id: {
                    ordinal: seed
                    for ordinal, seed in seeds.items()
                    if (topic_id, int(ordinal)) in self._completed
                }
                for topic_id, seeds in self._topic_seeds.items()
            }
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
        self._loaded_accounting = None
        self._legacy_accounting = False
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

    def _apply_prompt_override(
        self, step: str, default: str, path: list[str] | None, num_examples: int, **values
    ) -> str:
        source = getattr(self.config.prompt_templates, step)
        if step == "sequential_later" and source is None:
            source = self.config.prompt_templates.full_sample
        text = (
            source.content
            if source is not None
            else self.config.prompt_template
            if step in ("full_sample", "sequential_later")
            else None
        )
        if text is None:
            return default
        example_context = (
            self._build_example_context(num_examples) if "examples" not in values else {}
        )
        context = {
            **example_context,
            "raw_examples": example_context.get("raw_examples"),
            "raw_instructions": self.config.instructions,
            "raw_seed_field": self.config.seed_field,
            "raw_retry_feedback": None,
            "topic_path_json": path,
            "raw_tools": (
                [tool.model_dump(mode="json") for tool in self.tool_registry.tools]
                if "raw_tools" in placeholders(text)
                and self.tool_registry
                and self.tool_registry.tools
                else None
            ),
            "generation_system_prompt": self.generation_prompt,
            "dataset_system_prompt": self.dataset_system_prompt,
            "instructions": self.build_custom_instructions_text(),
            "subtopics": self.build_subtopics_text(path)
            if step != "completion" or self.config.completion_includes_topic_path
            else "",
            "topic_path": " -> ".join(path or []),
            "tools": self.build_tools_text()
            if self.tool_registry and self.tool_registry.tools
            else "",
            "tools_section": self.build_tools_section(),
            "num_items": 1,
            "seed_field": json.dumps(self.config.seed_field) if self.config.seed_field else "",
            "history": "[]",
            "sequential_history": "",
            "accepted_siblings_section": "",
            "scope": self.config.shared_scope,
            "request_instruction": render_template(
                default_template("request_bulk_full_single"), {"num_items": 1}
            ),
            "output_instructions": default_template("output_completion_custom")
            if step == "completion" and self.config.output_schema is not None
            else self._single_output_instructions(),
            "record_instructions": self._bulk_record_instructions(),
            "additional_instructions": (
                f"Additional Instructions: {self.config.instructions}"
                if self.config.instructions
                else ""
            ),
            "retry_feedback": "",
            "seed_section": "",
        }
        context.update(values)
        # A custom template explicitly opts into tool context, including agent framing.
        context["tools_section"] = self.build_tools_section()
        if step == "completion" and self.config.output_schema is not None and "seed" in values:
            context["seed_section"] = f"<pinned_seed>\n{values['seed']}\n</pinned_seed>"
        return render_template(text, context)

    def _format_prompt_step(self, step, template, allocation, num_examples, **values):
        return self._apply_prompt_override(
            step, template.format(**values), allocation["path"], num_examples, **values
        )

    def _single_output_instructions(self) -> str:
        if self.config.output_schema is not None:
            return default_template("output_single_custom")
        if self.tool_registry:
            return default_template("output_single_tools")
        return self._bulk_record_instructions()

    def _build_bulk_agent_question_prompt(
        self,
        path: list[str] | None,
        num_examples: int,
        *,
        num_items: int | str = 1,
        request_instruction: str,
        accepted_siblings_section: str = "",
        history: list | None = None,
    ) -> str:
        """Adapt independent's existing tool-question prompt for seed collection."""
        values = {
            "generation_system_prompt": self.generation_prompt,
            "instructions": self.build_custom_instructions_text(),
            **self._build_example_context(num_examples),
            "subtopics": self.build_subtopics_text(path),
            "tools_section": self.build_tools_section(),
            "accepted_siblings_section": accepted_siblings_section,
            "history": json.dumps(history or [], ensure_ascii=False),
            "num_items": num_items,
            "request_instruction": request_instruction,
            "output_instructions": 'Generate only the user questions as an object with an "items" list of strings:',
        }
        # Preserve independent's existing basic/CoT scenario templates and wrapper.
        # Adapt only the collection count, output shape and accepted-sibling context.
        prompt = self._get_prompt_template()
        for name in ("generation_system_prompt", "instructions", "examples", "subtopics"):
            prompt = prompt.replace("{{{{" + name + "}}}}", values[name])
        if accepted_siblings_section:
            prompt += f"\n{accepted_siblings_section}"
        prompt = SingleTurnAgentBuilder.build_user_question_prompt(
            prompt, num_items=num_items, collection=True, collection_request=request_instruction
        )
        return self._apply_prompt_override("seed", prompt, path, num_examples, **values)

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
        record_event(
            "retry",
            category="sample_validation",
            sample_idx=sample_idx + 1,
            attempt=attempt + 1,
            max_attempts=max_attempts,
            error_type=type(error).__name__,
        )
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
                            temperature=config.temperature,
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
            if not (
                self.config.topic_sample_mode == "sequential"
                or self._supports_bulk_full_records()
                or self._supports_bulk_seeds()
            ):
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
        tracker = active_token_tracker() or TokenUsageTracker()
        self._prepare_generation_run(plan)
        self._accounting = CheckpointAccounting(
            tracker, self._loaded_accounting, legacy=self._legacy_accounting
        )
        self._accounting_persistence_failed = False
        if self.config.checkpoint_interval is not None:
            tracker.accounting = self._accounting
            self._save_checkpoint_metadata()
        tracker.subscribe(self._persist_accounting)
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
        completed = False
        try:
            while True:
                # A synchronous event consumer may resume each iteration in a new
                # asyncio task/context. Never hold a ContextVar token across yield.
                with track_token_usage(tracker):
                    try:
                        event = await anext(events)
                    except StopAsyncIteration:
                        completed = not self.stop_requested and not self.get_all_failures()
                        break
                yield event
        finally:
            try:
                with track_token_usage(tracker):
                    await events.aclose()
            finally:
                tracker.unsubscribe(self._persist_accounting)
                self._persist_accounting("completed" if completed else "unfinished")
                tracker._checkpoint_usage_baseline = tracker.snapshot(
                    include_accounting=False, include_records=False
                )
                if self._large_sequential_prompt_topics:
                    logger.warning(self._sequential_prompt_warning_summary())

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
            self._loaded_accounting = None
            self._legacy_accounting = False
            self._accounting = None
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
        self._large_sequential_prompt_topics.clear()
        self._generation_plan = plan
        record_event(
            "generation_plan",
            plan=plan,
            resumed=resuming,
            resumed_samples=self._flushed_samples_count,
            resumed_failures=self._flushed_failures_count,
            completed_slots=[list(slot) for slot in sorted(self._completed)],
            resumed_output_slots=[list(slot) for slot in self._sample_slots],
        )
        # Resume is explicit for each invocation; saving a checkpoint does not select resume.
        self._checkpoint_loaded = False
        self._warn_bulk_token_budget(plan)

    @generation_scope
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
            if attempt:
                record_event("retry", category="sample_generation", attempt=attempt + 1)
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

    @generation_scope
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
        if self.config.topic_sample_mode == "sequential":
            await self._generate_sequential_topic_samples_async(
                allocation, offset, queue, num_example_demonstrations, include_sys_msg
            )
            return
        if self._supports_bulk_full_records() or self._supports_bulk_seeds():
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
        """Full-record bulk execution supports non-tool generation."""
        return (
            self.config.topic_sample_mode == "bulk"
            and self.config.shared_scope == "full"
            and not (self.config.tool_components or self.config.custom_tools)
        )

    def _supports_bulk_seeds(self) -> bool:
        return self.config.topic_sample_mode == "bulk" and self.config.shared_scope == "seed"

    def _bulk_max_tokens(self, requested: int) -> int:
        """Resolve a pooled ceiling for the items in this collection request."""
        override = self.config.bulk.max_tokens_per_call
        return override if override is not None else self.config.max_tokens * requested

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
        resolved = self._bulk_max_tokens(quota)
        if estimated > resolved:
            logger.warning(
                "bulk with shared_scope: full may exceed resolved max_tokens_per_call=%d: "
                "up to %d outstanding "
                "records per topic (coarse estimate %d output tokens). Raise max_tokens, "
                "raise bulk.max_tokens_per_call if explicitly set, lower the quota, "
                "or use shared_scope: seed.",
                resolved,
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
            return default_template("output_bulk_custom")
        instructions = default_template("output_bulk_conversation")
        if self.config.conversation_type == "cot":
            style = self.config.reasoning_style or "freetext"
            content_format = (
                "a plain string" if style == "freetext" else "a list of reasoning steps"
            )
            instructions += render_template(
                default_template("output_bulk_reasoning"),
                {"style": style, "content_format": content_format},
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

    def _seed_from_saved_row(self, row: dict) -> tuple[bool, Any]:
        if self.config.output_schema is not None:
            field = self.config.seed_field
            return (True, row[field]) if field in row else (False, None)
        for message in row.get("messages", []):
            if message.get("role") == "system":
                continue
            content = message.get("content")
            return message.get("role") == "user" and isinstance(content, str), content
        return False, None

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
                    allocation, offset, queue, num_example_demonstrations, include_sys_msg
                )
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
            await queue.put(("error", exc))
        await queue.put(("done", None))

    def _build_bulk_collection_prompt(
        self,
        allocation: dict,
        num_items: int | str,
        accepted_siblings: list,
        num_example_demonstrations: int,
    ) -> str:
        """Compose a bulk collection request without checkpoint or provider operations."""
        seed_scope = self.config.shared_scope == "seed"
        kind = (
            "full"
            if not seed_scope
            else "seed"
            if self.config.output_schema is not None
            else "conversation"
        )
        request = render_template(
            default_template(f"request_bulk_{kind}_{'single' if num_items == 1 else 'plural'}"),
            {"num_items": num_items, "seed_field": json.dumps(self.config.seed_field)},
        )
        siblings = (
            render_template(
                default_template("bulk_accepted_siblings"),
                {"history": json.dumps(accepted_siblings, ensure_ascii=False)},
            )
            if accepted_siblings
            else ""
        )
        if seed_scope and self.config.output_schema is None and self.tool_registry:
            return self._build_bulk_agent_question_prompt(
                allocation["path"],
                num_example_demonstrations,
                num_items=num_items,
                request_instruction=request,
                accepted_siblings_section=siblings,
                history=accepted_siblings,
            )
        if (
            seed_scope
            and self.config.output_schema is None
            and self.config.conversation_type == "cot"
        ):
            example_context = self._build_example_context(num_example_demonstrations)
            examples = example_context["examples"]
            scenario = (
                STRUCTURED_COT_PROMPT
                if self.config.reasoning_style == "agent"
                else FREETEXT_COT_PROMPT
            )
            # Adapt collection wording without changing independent's template or supplied data.
            scenario = (
                scenario.replace(
                    "Generate a reasoning problem that requires",
                    "Generate reasoning problems that require",
                )
                .replace(
                    "Generate a training conversation that demonstrates systematic problem-solving.",
                    "Generate reasoning problems that demonstrate systematic problem-solving.",
                )
                .replace(
                    "Create realistic dialogues where complex problems are solved through methodical reasoning.",
                    "Create realistic reasoning problems that require methodical reasoning to solve.",
                )
            )
            for name, value in {
                "instructions": self.build_custom_instructions_text(),
                "examples": examples,
                "subtopics": self.build_subtopics_text(allocation["path"]),
            }.items():
                scenario = scenario.replace("{{{{" + name + "}}}}", value)
            return self._format_prompt_step(
                "seed",
                compatibility_template("bulk_cot_seed", "format"),
                allocation,
                num_example_demonstrations,
                generation_system_prompt=self.generation_prompt,
                instructions=self.build_custom_instructions_text(),
                **example_context,
                subtopics=self.build_subtopics_text(allocation["path"]),
                record_instructions=scenario,
                additional_instructions=(
                    f"Additional Instructions: {self.config.instructions}"
                    if self.config.instructions
                    else ""
                ),
                accepted_siblings_section=siblings,
                request_instruction=request,
                num_items=num_items,
                history=json.dumps(accepted_siblings, ensure_ascii=False),
                tools_section="",
                output_instructions=default_template("output_seed_conversation"),
            )
        if seed_scope and self.config.output_schema is None and self.config.instructions:
            # Match the other sample modes for now, including repeated instructions.
            # Prompt repetition will be addressed separately.
            request += f"\n\nAdditional Instructions: {self.config.instructions}"
        common = {
            "request_instruction": request,
            "generation_system_prompt": self.generation_prompt,
            "subtopics": self.build_subtopics_text(allocation["path"]),
            "instructions": self.build_custom_instructions_text(),
            **self._build_example_context(num_example_demonstrations),
            "accepted_siblings_section": siblings,
            "tools_section": self.build_tools_section(),
        }
        if not seed_scope and self.config.output_schema is None and not self.tool_registry:
            # Keep independent framing without conflicting single-sample requirements.
            template = self._get_prompt_template().replace(
                "a realistic single q&a", "realistic q&a"
            )
            template = template.replace("Generate one training sample as question and answer.", "")
            topic_prompt = render_template(template, common)
            if siblings:
                topic_prompt += f"\n{siblings}"
            builder = SingleShotBuilder(self.llm_client, self.config)
            prompt = builder._build_prompt(topic_prompt, num_items=num_items)
            return self._apply_prompt_override(
                "full_sample",
                prompt,
                allocation["path"],
                num_example_demonstrations,
                **common,
                num_items=num_items,
                history=json.dumps(accepted_siblings, ensure_ascii=False),
                record_instructions=self._bulk_record_instructions(),
                output_instructions=self._bulk_record_instructions(),
            )
        template = (
            (
                BULK_SEED_SAMPLE_PROMPT
                if self.config.output_schema is not None
                else BULK_CONVERSATION_SEED_PROMPT
            )
            if seed_scope
            else BULK_FULL_SAMPLE_PROMPT
        )
        return self._format_prompt_step(
            "seed" if seed_scope else "full_sample",
            template,
            allocation,
            num_example_demonstrations,
            **common,
            record_instructions=self._bulk_record_instructions(),
            output_instructions=default_template(
                "output_seed_custom"
                if self.config.output_schema is not None
                else "output_seed_conversation"
            )
            if seed_scope
            else self._bulk_record_instructions(),
            num_items=num_items,
            history=json.dumps(accepted_siblings, ensure_ascii=False),
        )

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
            (
                seed_value_schema(self.config.output_schema, self.config.seed_field)
                if self.config.output_schema is not None
                else {"type": "string", "minLength": 1}
            )
            if seed_scope
            else self.config.output_schema
            if self.config.output_schema is not None
            else Conversation.model_json_schema()
        )
        validator = record_validator(schema)
        collection_model = make_collection_model(schema)
        if seed_scope:
            context = {}
            for ordinal, row in self._saved_topic_outcomes(topic_id):
                found, value = self._seed_from_saved_row(row)
                if found:
                    context[ordinal] = value
            context.update({int(ordinal): value for ordinal, value in saved_seeds.items()})
            accepted_siblings = [value for _, value in sorted(context.items())]
        else:
            accepted_siblings = self._saved_topic_records(topic_id)
        attempts = self._collection_attempts.get(topic_id, 0)
        budget = self.config.sample_retries + 1
        last_error = "Bulk collection attempt budget exhausted"
        while outstanding and attempts < budget and not self.stop_requested:
            requested = len(outstanding)
            max_tokens = self._bulk_max_tokens(requested)
            budget_source = (
                "explicit" if self.config.bulk.max_tokens_per_call is not None else "calculated"
            )
            attempts += 1
            # Reserve an attempt durably before the request, including interrupted calls.
            await self._checkpoint_topic_state(queue, topic_id, attempts=attempts)
            if self.stop_requested:
                return
            prompt = self._build_bulk_collection_prompt(
                allocation, requested, accepted_siblings, num_example_demonstrations
            )
            event = {
                "topic_id": topic_id,
                "requested": requested,
                "accepted": 0,
                "missing": requested,
                "attempt": attempts,
                "budget": budget,
                "max_tokens_per_call": max_tokens,
                "token_budget_source": budget_source,
                "base_max_tokens": self.config.max_tokens,
                "completion_max_tokens": self.config.max_tokens if seed_scope else None,
            }
            if attempts > 1:
                record_event(
                    "retry", category="bulk_collection", attempt=attempts, requested=requested
                )
            await queue.put(("event", {"event": "seed_request", **event}))
            retryable = True
            accepted = 0
            pending_seeds = {}
            response_received = False
            unit = "seeds" if seed_scope else "samples"
            step_label = (
                f"Generating {requested} {unit} for topic {topic_id} (attempt {attempts}/{budget}; "
                f"token cap {max_tokens}, {budget_source}"
                + (
                    f" as {self.config.max_tokens} x {requested}"
                    if budget_source == "calculated"
                    else ""
                )
                + (
                    f"; completions separately capped at {self.config.max_tokens} per call)"
                    if seed_scope
                    else ")"
                )
            )
            logger.info("%s", step_label)
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
                    max_tokens=max_tokens,
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
        include_sys_msg: bool,
    ) -> None:
        topic_id = allocation["uuid"]
        seeds = self._topic_seeds.get(topic_id, {})
        if self.config.output_schema is not None:
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
            if self.config.output_schema is not None:
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
            else:
                sample, failure = await self._complete_conversation_seed_outcome(
                    allocation,
                    ordinal,
                    offset + ordinal,
                    seeds[str(ordinal)],
                    num_example_demonstrations,
                    include_sys_msg,
                )
            if sample is None and failure is None:
                break
            await queue.put(("outcome", (identity, sample, failure)))
            await asyncio.sleep(0)

    def _conversation_completion_prompt(
        self, allocation: dict, seed: str, num_example_demonstrations: int
    ) -> str:
        agent = bool(self.config.tool_components or self.config.custom_tools)
        template = BULK_CONVERSATION_COMPLETION_PROMPT
        if agent:
            output = default_template("output_completion_agent")
        elif self.config.conversation_type == "cot":
            style = self.config.reasoning_style or "freetext"
            template = compatibility_template("bulk_cot_completion", "format")
            output = default_template(
                "freetext_cot_completion_format"
                if style == "freetext"
                else "structured_cot_completion_format"
            ).strip()
            output += (
                "\n\nIMPORTANT: Generate and return only the 'assistant' message (the response). "
                "Do NOT reproduce a user turn or a message array. "
                "Do NOT include any 'system' role messages - those are added separately."
            )
        else:
            output = default_template("output_completion_basic")
        if agent or self.config.conversation_type != "cot":
            output += default_template("output_completion_pinned")
        return self._format_prompt_step(
            "completion",
            template,
            allocation,
            num_example_demonstrations,
            seed=json.dumps(seed, ensure_ascii=False),
            dataset_system_prompt=self.dataset_system_prompt,
            seed_section=(
                ""
                if agent
                else "Here is the user request you should answer:\n"
                f"<pinned_user_request>\n{json.dumps(seed, ensure_ascii=False)}\n</pinned_user_request>"
            ),
            output_instructions=output,
            subtopics=self.build_subtopics_text(allocation["path"])
            if self.config.completion_includes_topic_path
            else "",
            instructions=self.build_custom_instructions_text(),
            **self._build_example_context(num_example_demonstrations),
            tools_section="" if agent else self.build_tools_section(),
            context_guidance="",
        )

    def _validate_pinned_conversation(
        self, conversation: Conversation, seed: str, include_sys_msg: bool
    ) -> None:
        user_index = int(include_sys_msg)
        if (
            len(conversation.messages) <= user_index
            or conversation.messages[user_index].role != "user"
            or conversation.messages[user_index].content != seed
        ):
            raise ValueError(
                "Conversation completion validation error: pinned user request changed"
            )
        self._validate_conversation_tools(conversation)

    def _validate_conversation_tools(self, conversation: Conversation) -> None:
        """Apply the existing execution-count policy to coordinated conversations."""
        if self.config.tool_components or self.config.custom_tools:
            if not conversation.tool_context or not conversation.tool_context.executions:
                raise ValueError(
                    "Agent completion validation error: at least one tool execution is required"
                )
            if len(conversation.tool_context.executions) > self.config.max_tools_per_query:
                if self.config.max_tools_strict:
                    raise ValueError(
                        "Agent completion validation error: tool execution limit exceeded"
                    )
                conversation.tool_context.executions = conversation.tool_context.executions[
                    : self.config.max_tools_per_query
                ]

    @generation_scope
    async def _complete_conversation_seed_outcome(
        self,
        allocation: dict,
        ordinal: int,
        sample_idx: int,
        seed: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
    ) -> tuple[dict | None, dict | None]:
        prompt = self._conversation_completion_prompt(allocation, seed, num_example_demonstrations)
        config = self.config.model_copy(
            update={
                "sys_msg": include_sys_msg,
                "dataset_system_prompt": self.dataset_system_prompt,
            }
        )
        budget = self.config.sample_retries + 1
        error_feedback = None
        for attempt in range(budget):
            if self.stop_requested:
                return None, None
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
                builder = ConversationBuilderFactory.create(
                    config=config,
                    llm=self.llm_client,
                    tool_registry=self.tool_registry,
                    progress_reporter=self.progress_reporter,
                )
                conversation = await builder.generate(
                    prompt,
                    error_feedback,
                    pinned_user_message=ChatMessage(role="user", content=seed),
                )
                self._validate_pinned_conversation(conversation, seed, include_sys_msg)
                conversation.metadata = {
                    **(conversation.metadata or {}),
                    "topic_id": allocation["uuid"],
                }
                sample = conversation.model_dump(exclude_none=True)
                success = True
            except Exception as exc:  # noqa: BLE001
                error_feedback = str(exc)
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
            "error": error_feedback,
        }

    @generation_scope
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
            prompt = self._format_prompt_step(
                "completion",
                BULK_COMPLETION_PROMPT,
                allocation,
                num_example_demonstrations,
                dataset_system_prompt=self.dataset_system_prompt,
                seed_field=json.dumps(seed_field),
                seed=json.dumps(seed, ensure_ascii=False),
                subtopics=self.build_subtopics_text(allocation["path"])
                if self.config.completion_includes_topic_path
                else "",
                context_guidance=self._completion_context_guidance(),
                instructions=self.build_custom_instructions_text(),
                **self._build_example_context(num_example_demonstrations),
                tools_section=self.build_tools_section(),
                retry_feedback=f"RETRY: {last_error}" if attempt else "",
                raw_retry_feedback=last_error if attempt else None,
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

    def _sequential_history(self, successes: dict[int, dict], ordinal: int) -> list:
        """Select earlier successful siblings by identity, regardless of JSONL order."""
        earlier = sorted(index for index in successes if index < ordinal)
        if self.config.shared_scope == "full":
            return [successes[index] for index in reversed(earlier)]
        seeds = []
        for index in earlier:
            found, seed = self._seed_from_saved_row(successes[index])
            if found:
                seeds.append(seed)
        return seeds

    @generation_scope
    async def _generate_sequential_sample_outcome(
        self,
        prompt: str,
        allocation: dict,
        ordinal: int,
        sample_idx: int,
        include_sys_msg: bool,
    ) -> tuple[dict | None, dict | None]:
        """Generate one whole sample with one validation budget and fresh builders."""
        config = self.config.model_copy(
            update={
                "sys_msg": include_sys_msg,
                "dataset_system_prompt": self.dataset_system_prompt,
            }
        )
        custom = config.output_schema is not None
        model = make_completion_model(config.output_schema) if custom else None
        validator = record_validator(config.output_schema) if custom else None
        budget = config.sample_retries + 1
        error = None
        for attempt in range(budget):
            if self.stop_requested:
                return None, None
            label = f"Generating sample {sample_idx + 1} (attempt {attempt + 1}/{budget})"
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
                if custom:
                    retry_prompt = prompt + (f"\nRETRY: {error}" if attempt else "")
                    result = await self.llm_client.generate_async(
                        retry_prompt,
                        model,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    )
                    sample = result.model_dump(exclude_none=True)
                    if not validator.is_valid(sample):
                        raise ValueError(
                            "Sequential sample validation error: output_schema mismatch"
                        )
                else:
                    builder = ConversationBuilderFactory.create(
                        config=config,
                        llm=self.llm_client,
                        tool_registry=self.tool_registry,
                        progress_reporter=self.progress_reporter,
                    )
                    if isinstance(builder, SingleShotBuilder):
                        conversation = await builder.generate(
                            prompt,
                            error,
                            prompt_step="sequential_later" if ordinal > 0 else "full_sample",
                        )
                    else:
                        conversation = await builder.generate(prompt, error)
                    messages = conversation.messages[int(include_sys_msg) :]
                    if (
                        len(messages) < 2  # noqa: PLR2004 - request and reply
                        or messages[0].role != "user"
                        or messages[-1].role != "assistant"
                    ):
                        raise ValueError(
                            "Sequential conversation validation error: invalid message order"
                        )
                    if not self.tool_registry and any(
                        message.role not in ("user", "assistant") for message in messages
                    ):
                        raise ValueError(
                            "Sequential conversation validation error: unexpected message role"
                        )
                    if config.conversation_type == "cot" and (
                        conversation.reasoning is None
                        or conversation.reasoning.style != (config.reasoning_style or "freetext")
                    ):
                        raise ValueError(
                            "Sequential conversation validation error: reasoning style mismatch"
                        )
                    self._validate_conversation_tools(conversation)
                    conversation.metadata = {
                        **(conversation.metadata or {}),
                        "topic_id": allocation["uuid"],
                    }
                    sample = conversation.model_dump(exclude_none=True)
                success = True
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
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
            "error": error,
        }

    def _sequential_prompt_warning_summary(self) -> str:
        count = len(self._large_sequential_prompt_topics)
        suppressed = max(0, count - 1)
        return (
            f"Large sequential prompts detected for {count} {'topic' if count == 1 else 'topics'} "
            f"({suppressed} {'warning' if suppressed == 1 else 'warnings'} suppressed)."
        )

    def _check_sequential_prompt_size(
        self, prompt: str, allocation: dict, ordinal: int
    ) -> dict | None:
        """Warn once per run and report each newly affected topic to live consumers."""
        topic_id = allocation["uuid"]
        if (
            len(prompt) < SEQUENTIAL_PROMPT_WARNING_CHARS
            or topic_id in self._large_sequential_prompt_topics
        ):
            return None
        self._large_sequential_prompt_topics.add(topic_id)
        affected = len(self._large_sequential_prompt_topics)
        message = (
            f"Sequential prompt for topic {topic_id} "
            f"(sample {ordinal + 1} of {allocation['quota']}) has reached {len(prompt)} characters. "
            "Large prompts increase input cost and may exceed the model's context window. "
            "All history is retained."
        )
        if affected == 1:
            logger.warning(message)
        return {
            "event": "sequential_prompt_warning",
            "topic_id": topic_id,
            "sample_ordinal": ordinal,
            "topic_quota": allocation["quota"],
            "characters": len(prompt),
            "affected_topics": affected,
            "warnings_suppressed": affected - 1,
            "message": message if affected == 1 else None,
            "summary": self._sequential_prompt_warning_summary(),
        }

    def _build_sequential_prompt(
        self,
        allocation: dict,
        history: list,
        num_example_demonstrations: int,
        *,
        later: bool | None = None,
    ) -> str:
        """Use the same context formatting for previews and real sequential calls."""
        sequential_history = ""
        if history:
            if self.config.shared_scope == "full":
                description = default_template("history_header_full")
            elif self.config.output_schema is not None:
                description = render_template(
                    default_template("history_header_seed"),
                    {"seed_field": json.dumps(self.config.seed_field)},
                )
            else:
                description = default_template("history_header_conversation")
            sequential_history = render_template(
                default_template("sequential_history"),
                {
                    "description": description,
                    "scope": self.config.shared_scope,
                    "history": json.dumps(history, ensure_ascii=False),
                },
            )
        step = "sequential_later" if (bool(history) if later is None else later) else "full_sample"
        if self.config.output_schema is None:
            return self.build_prompt(
                self._get_prompt_template(),
                num_example_demonstrations,
                allocation["path"],
                prompt_step=step,
                sequential_history=sequential_history,
                history=history,
            )
        return self._format_prompt_step(
            step,
            SEQUENTIAL_SAMPLE_PROMPT,
            allocation,
            num_example_demonstrations,
            generation_system_prompt=self.generation_prompt,
            subtopics=self.build_subtopics_text(allocation["path"]),
            instructions=self.build_custom_instructions_text(),
            **self._build_example_context(num_example_demonstrations),
            tools_section=self.build_tools_section(),
            sequential_history=sequential_history,
            history=json.dumps(history, ensure_ascii=False),
            output_instructions=self._single_output_instructions(),
        )

    async def _generate_sequential_topic_samples_async(
        self,
        allocation: dict,
        offset: int,
        queue: asyncio.Queue,
        num_example_demonstrations: int,
        include_sys_msg: bool,
    ) -> None:
        """Keep successful siblings locally; persisted rows rebuild them on resume."""
        try:
            successes = dict(self._saved_topic_outcomes(allocation["uuid"]))
            for ordinal in range(allocation["quota"]):
                identity = (allocation["uuid"], ordinal)
                if identity in self._completed:
                    continue
                if self.stop_requested:
                    break
                prompt = self._build_sequential_prompt(
                    allocation,
                    self._sequential_history(successes, ordinal),
                    num_example_demonstrations,
                    later=ordinal > 0,
                )
                warning = self._check_sequential_prompt_size(prompt, allocation, ordinal)
                if warning is not None:
                    await queue.put(("event", warning))
                sample, failure = await self._generate_sequential_sample_outcome(
                    prompt, allocation, ordinal, offset + ordinal, include_sys_msg
                )
                if sample is None and failure is None:
                    break
                if sample is not None:
                    successes[ordinal] = sample
                await queue.put(("outcome", (identity, sample, failure)))
                await asyncio.sleep(0)
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
            await queue.put(("error", exc))
        await queue.put(("done", None))

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
            record_event(
                "sample_outcome",
                topic_id=identity[0],
                sample_ordinal=identity[1],
                status="accepted" if sample is not None else "failed",
                output_row=historical_samples + new_samples + 1 if sample is not None else None,
                error_type=failure.get("error_type") if failure else None,
                error=failure.get("error") if failure else None,
            )
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
                        record_event("generation_event", details=payload)
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
            record_event("generation_result", stopped=self.stop_requested, **counts())
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
        *,
        prompt_step: str = "full_sample",
        sequential_history: str = "",
        history: list | None = None,
    ) -> str:
        prompt = data_creation_prompt.replace(
            "{{{{generation_system_prompt}}}}", self.generation_prompt
        )
        prompt = prompt.replace("{{{{instructions}}}}", self.build_custom_instructions_text())
        example_context = self._build_example_context(num_example_demonstrations)
        prompt = prompt.replace("{{{{examples}}}}", example_context["examples"])
        prompt = prompt.replace("{{{{subtopics}}}}", self.build_subtopics_text(subtopics_list))
        if sequential_history:
            prompt += f"\n{sequential_history}"
        return self._apply_prompt_override(
            prompt_step,
            prompt,
            subtopics_list,
            num_example_demonstrations,
            **example_context,
            sequential_history=sequential_history,
            history=json.dumps(history or [], ensure_ascii=False),
        )

    def build_system_prompt(self):
        """Return the original system prompt for dataset inclusion."""
        return self.dataset_system_prompt

    def build_custom_instructions_text(self) -> str:
        if self.config.instructions is None or self.config.instructions == "":
            return ""
        return f"\nHere are additional instructions:\n<instructions>\n{self.config.instructions}\n</instructions>\n"

    def _build_example_context(self, num_example_demonstrations: int) -> dict:
        """Select once and expose the same examples with and without framing."""
        if self.config.example_data is None or num_example_demonstrations == 0:
            return {"examples": "", "raw_examples": None}
        # Bandit: not a security function
        # HF Dataset supports len() and indexing, convert to list for sampling
        example_list = list(self.config.example_data)
        examples = random.sample(example_list, min(num_example_demonstrations, len(example_list)))  # nosec
        examples_text = "Here are output examples:\n\n"
        examples_text += "\n".join(f"Example {i + 1}: \n\n{ex}\n" for i, ex in enumerate(examples))
        return {
            "examples": f"\nHere are output examples:\n<examples>\n{examples_text}\n</examples>\n",
            "raw_examples": examples,
        }

    def build_examples_text(self, num_example_demonstrations: int) -> str:
        """Return the existing framed example text."""
        return self._build_example_context(num_example_demonstrations)["examples"]

    def _completion_context_guidance(self) -> str:
        """Clarify the scope of unchanged user instructions when ancestry is withheld."""
        if self.config.completion_includes_topic_path:
            return default_template("completion_context_with_path")
        return default_template("completion_context_without_path")

    def build_tools_section(self) -> str:
        """Include a tools block only when actual tool definitions are available."""
        if not self.tool_registry or not self.tool_registry.tools:
            return ""
        return f"<tools>\n{self.build_tools_text()}\n</tools>"

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
            if self.tool_registry:
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
