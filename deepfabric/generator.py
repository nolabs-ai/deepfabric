import asyncio
import json
import logging
import math
import os
import random

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datasets import Dataset as HFDataset
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .builders import ConversationBuilderFactory
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
)
from .error_codes import classify_error
from .exceptions import DataSetGeneratorError
from .llm import LLMClient
from .metrics import trace
from .progress import ProgressReporter
from .prompts import (
    AGENT_COT_TOOLS_PROMPT,
    CONVERSATION_GENERATION_PROMPT,
    FREETEXT_COT_PROMPT,
    STRUCTURED_COT_PROMPT,
    AgentPromptBuilder,
)
from .schemas import Conversation, ToolRegistry, get_conversation_schema
from .tools import BUILTIN_TOOL_REGISTRY
from .tools.loader import load_tools_from_dict, load_tools_from_endpoint
from .topic_model import Topic, TopicModel, TopicPath
from .utils import ensure_not_running_loop, get_checkpoint_dir, is_validation_error

# Handle circular import for type hints
if TYPE_CHECKING:
    from .topic_model import TopicModel

logger = logging.getLogger(__name__)


class DataSetGeneratorConfig(BaseModel):
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
    gemini_safety_settings: list[dict] | None = Field(
        default=None,
        description="Safety settings for Gemini models",
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
        if self.config.gemini_safety_settings:
            llm_kwargs["gemini_safety_settings"] = self.config.gemini_safety_settings

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
        self._completed: set[tuple[str, int]] = set()  # Track (uuid, cycle) tuples
        self._checkpoint_metadata_path: Path | None = None
        self._checkpoint_samples_path: Path | None = None
        self._checkpoint_failures_path: Path | None = None
        # Memory optimization: track flushed counts for checkpoint mode
        self._flushed_samples_count = 0
        self._flushed_failures_count = 0
        # Generation state for cycle-based iteration
        self._unique_topics: int = 0  # Number of unique topics
        self._cycles_needed: int = 0  # Total cycles for requested samples

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

    def _save_checkpoint(
        self,
        new_samples: list[dict],
        new_failures: list[dict],
        completed_items: list[tuple[str, int]],
        flush_memory: bool = True,
    ) -> None:
        """Save checkpoint data incrementally.

        Args:
            new_samples: New successful samples to append
            new_failures: New failed samples to append
            completed_items: List of (uuid, cycle) tuples that were completed
            flush_memory: If True, clear flushed samples from memory (memory optimization)
        """
        if self._checkpoint_samples_path is None:
            return

        # Append new samples to checkpoint file
        if new_samples:
            with open(self._checkpoint_samples_path, "a", encoding="utf-8") as f:
                for sample in new_samples:
                    f.write(json.dumps(sample, separators=(",", ":")) + "\n")

        # Append new failures to failures file
        if new_failures and self._checkpoint_failures_path:
            with open(self._checkpoint_failures_path, "a", encoding="utf-8") as f:
                for failure in new_failures:
                    f.write(json.dumps(failure, separators=(",", ":")) + "\n")

        # Track completed (uuid, cycle) tuples
        for item in completed_items:
            self._completed.add(item)

        # Memory optimization: track flushed counts and clear in-memory lists
        # Must happen BEFORE saving metadata so counts are accurate
        if flush_memory:
            self._flushed_samples_count += len(new_samples)
            self._flushed_failures_count += len(new_failures)
            # Clear the in-memory lists since data is now on disk
            self._samples.clear()
            self.failed_samples.clear()

        # Update metadata (after flush counts are updated)
        self._save_checkpoint_metadata()

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

        # Convert completed set of (uuid, cycle) tuples to list of [uuid, cycle] arrays
        completed_list = [[uuid, cycle] for uuid, cycle in self._completed]

        metadata = {
            "version": CHECKPOINT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider,
            "model_name": self.model_name,
            "conversation_type": self.config.conversation_type,
            "reasoning_style": self.config.reasoning_style,
            "total_samples": total_samples,
            "total_failures": total_failures,
            "completed": completed_list,  # List of [uuid, cycle] arrays
            "unique_topics": self._unique_topics,
            "cycles_needed": self._cycles_needed,
            "checkpoint_interval": self.config.checkpoint_interval,
            "topics_file": self.config.topics_file,
        }

        with open(self._checkpoint_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

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

        if mismatches:
            logger.warning(
                "Config mismatch with checkpoint. Resuming may produce inconsistent results. "
                "Differences: %s",
                "; ".join(mismatches),
            )

    def _validate_checkpoint_integrity(self, metadata: dict) -> tuple[bool, str | None]:
        """Validate checkpoint file integrity.

        Checks that:
        1. Metadata version is supported
        2. Required metadata fields are present
        3. Sample count in metadata matches actual file line count
        4. Sample file contains valid JSON on each line

        Args:
            metadata: Checkpoint metadata dictionary

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        error_msg: str | None = None

        # Check version
        version = metadata.get("version")
        if version is None:
            error_msg = "Missing 'version' field in checkpoint metadata"
        elif version < CHECKPOINT_VERSION:
            # Old checkpoint format - require fresh start
            error_msg = (
                f"Checkpoint format v{version} is incompatible with current version v{CHECKPOINT_VERSION}. "
                "Please delete the checkpoint and restart: rm -rf .checkpoints/"
            )
        elif version > CHECKPOINT_VERSION:
            error_msg = (
                f"Checkpoint version {version} is newer than supported version {CHECKPOINT_VERSION}"
            )

        # Check required fields for v4 format
        if error_msg is None:
            required_fields = [
                "created_at",
                "total_samples",
                "completed",
                "unique_topics",
                "cycles_needed",
            ]
            for field in required_fields:
                if field not in metadata:
                    error_msg = f"Missing required field in checkpoint metadata: {field}"
                    break

        # Validate sample count matches file
        if error_msg is None:
            expected_samples = metadata.get("total_samples", 0)
            if self._checkpoint_samples_path and self._checkpoint_samples_path.exists():
                actual_count = 0
                try:
                    with open(self._checkpoint_samples_path, encoding="utf-8") as f:
                        for line_num, raw_line in enumerate(f, 1):
                            stripped = raw_line.strip()
                            if stripped:
                                try:
                                    json.loads(stripped)
                                    actual_count += 1
                                except json.JSONDecodeError as e:
                                    error_msg = f"Invalid JSON on line {line_num} of checkpoint samples: {e}"
                                    break
                except OSError as e:
                    error_msg = f"Failed to read checkpoint samples file: {e}"

                if error_msg is None and actual_count != expected_samples:
                    error_msg = (
                        f"Sample count mismatch: metadata says {expected_samples}, "
                        f"file has {actual_count} samples"
                    )
            elif expected_samples > 0:
                error_msg = f"Checkpoint metadata expects {expected_samples} samples but samples file missing"

        return (error_msg is None, error_msg)

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
        """Load checkpoint data if it exists.

        Args:
            retry_failed: If True, remove failed IDs from processed set to retry them

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists
        """
        if self.config.checkpoint_interval is None:
            return False

        self._initialize_checkpoint_paths()

        if self._checkpoint_metadata_path is None or not self._checkpoint_metadata_path.exists():
            return False

        try:
            # Load metadata
            with open(self._checkpoint_metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            # Validate checkpoint integrity
            is_valid, error_msg = self._validate_checkpoint_integrity(metadata)
            if not is_valid:
                logger.error("Checkpoint integrity check failed: %s", error_msg)
                return False

            # Validate config compatibility
            self._validate_checkpoint_compatibility(metadata)

            # Restore completed (uuid, cycle) tuples
            completed_list = metadata.get("completed", [])
            self._completed = {(item[0], item[1]) for item in completed_list}
            self._unique_topics = metadata.get("unique_topics", 0)
            self._cycles_needed = metadata.get("cycles_needed", 0)

            # Count existing samples (don't load into memory - they're already on disk)
            # Memory optimization: track as flushed counts instead of loading into RAM
            if self._checkpoint_samples_path and self._checkpoint_samples_path.exists():
                sample_count = 0
                with open(self._checkpoint_samples_path, encoding="utf-8") as f:
                    for raw_line in f:
                        if raw_line.strip():
                            sample_count += 1
                self._flushed_samples_count = sample_count

            # Load failure IDs for retry logic (these are small)
            failed_ids: set[str] = set()
            if self._checkpoint_failures_path and self._checkpoint_failures_path.exists():
                failure_count = 0
                failed_tuples: set[tuple[str, int]] = set()
                with open(self._checkpoint_failures_path, encoding="utf-8") as f:
                    for raw_line in f:
                        stripped = raw_line.strip()
                        if stripped:
                            failure = json.loads(stripped)
                            failure_count += 1
                            # Track failed (topic_id, cycle) for targeted retry
                            if "topic_id" in failure:
                                failed_ids.add(failure["topic_id"])
                                if "cycle" in failure:
                                    failed_tuples.add((failure["topic_id"], failure["cycle"]))
                self._flushed_failures_count = failure_count

            # If retry_failed is True, remove failed entries from completed set
            # so they will be retried during generation
            if retry_failed and failed_ids:
                if failed_tuples:
                    # Targeted retry: only remove the specific (uuid, cycle) that failed
                    tuples_to_remove = self._completed & failed_tuples
                else:
                    # Legacy fallback: no cycle info, remove all cycles for failed UUIDs
                    tuples_to_remove = {
                        (uuid, cycle) for uuid, cycle in self._completed if uuid in failed_ids
                    }
                self._completed -= tuples_to_remove
                # Clear failures file since we're retrying
                if self._checkpoint_failures_path and self._checkpoint_failures_path.exists():
                    os.remove(self._checkpoint_failures_path)
                self._flushed_failures_count = 0
                logger.info(
                    "Retry mode: %d failed UUIDs (%d tuples) will be retried",
                    len(failed_ids),
                    len(tuples_to_remove),
                )

            logger.info(
                "Loaded checkpoint: %d samples, %d failures, %d completed (uuid, cycle) tuples",
                self._flushed_samples_count,
                self._flushed_failures_count,
                len(self._completed),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load checkpoint: %s", e)
            return False
        else:
            return True

    def clear_checkpoint(self) -> None:
        """Remove checkpoint files."""
        if self._checkpoint_metadata_path and self._checkpoint_metadata_path.exists():
            os.remove(self._checkpoint_metadata_path)
        if self._checkpoint_samples_path and self._checkpoint_samples_path.exists():
            os.remove(self._checkpoint_samples_path)
        if self._checkpoint_failures_path and self._checkpoint_failures_path.exists():
            os.remove(self._checkpoint_failures_path)
        self._completed.clear()
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

    def _is_completed(self, uuid: str, cycle: int) -> bool:
        """Check if a (uuid, cycle) combination has been completed.

        Args:
            uuid: Topic UUID
            cycle: Cycle number (0-indexed)

        Returns:
            True if this (uuid, cycle) was already completed
        """
        return (uuid, cycle) in self._completed

    def _is_uuid_completed_any_cycle(self, uuid: str) -> bool:
        """Check if a UUID has been completed in any cycle.

        Used during transition period when step-based generation
        still needs to check for completed topics.

        Args:
            uuid: Topic UUID to check

        Returns:
            True if this UUID has been completed in at least one cycle
        """
        return any(u == uuid for u, _cycle in self._completed)

    def _validate_create_data_params(
        self,
        num_steps: int,
        batch_size: int,
        topic_model: "TopicModel | None" = None,
    ) -> None:
        """Validate parameters for data creation."""
        if num_steps is None or num_steps <= 0:
            raise DataSetGeneratorError("num_steps must be a positive integer")

        if batch_size <= 0:
            raise DataSetGeneratorError("batch_size must be a positive integer")

        if topic_model and len(topic_model.get_all_paths()) == 0:
            raise DataSetGeneratorError(
                "Topic model has no paths. Ensure the topic tree was built successfully."
            )

    def _prepare_topic_paths(
        self,
        num_steps: int,
        batch_size: int,
        topic_model: "TopicModel | None" = None,
    ) -> tuple[list[TopicPath] | None, int]:
        """Prepare and validate topic paths for data generation."""
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

    def _prepare_unique_topics(
        self,
        total_samples: int,
        topic_model: "TopicModel",
    ) -> tuple[list[Topic], int]:
        """Prepare unique topics and calculate cycles needed for generation.

        This method supports the new cycle-based generation model where we iterate
        over unique topics (by UUID) multiple times (cycles) to generate the
        requested number of samples.

        Args:
            total_samples: Total number of samples to generate.
            topic_model: The topic model (Tree or Graph) to extract topics from.

        Returns:
            Tuple of (unique_topics, cycles_needed):
            - unique_topics: List of Topic namedtuples with (uuid, topic)
            - cycles_needed: Number of times to iterate through all topics
        """
        unique_topics = topic_model.get_unique_topics()
        unique_count = len(unique_topics)

        if unique_count == 0:
            raise DataSetGeneratorError(
                "Topic model has no unique topics. Ensure the topic model was built successfully."
            )

        # Calculate cycles needed to cover the requested samples
        # e.g., 5000 samples from 1875 topics = ceil(5000/1875) = 3 cycles
        cycles_needed = math.ceil(total_samples / unique_count)

        # Calculate how many samples we'll generate in the final (partial) cycle
        final_cycle_size = total_samples - (cycles_needed - 1) * unique_count

        logger.info(
            "Topic preparation: unique_topics=%d, requested_samples=%d, cycles_needed=%d, "
            "final_cycle_size=%d",
            unique_count,
            total_samples,
            cycles_needed,
            final_cycle_size,
        )

        return unique_topics, cycles_needed

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

        async def _generate_with_retry(
            prompt: str, sample_idx: int, topic_path_info: TopicPath | None
        ) -> tuple[bool, Exception | Conversation]:
            """Generate a single sample with per-sample retry for validation errors.

            Each parallel task gets its own builder instance to avoid Spin session
            conflicts when running samples concurrently (batch_size > 1).
            """
            # Create a fresh builder for this sample to avoid session conflicts
            # when running in parallel batches
            builder = ConversationBuilderFactory.create(
                config=config,
                llm=self.llm_client,
                tool_registry=self.tool_registry,
                progress_reporter=self.progress_reporter,
            )

            last_error: Exception | None = None
            error_feedback: str | None = None
            max_attempts = self.config.sample_retries + 1
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
    ) -> HFDataset:
        if num_steps is None:
            num_steps = 1

        self._validate_create_data_params(num_steps, batch_size, topic_model)

        if model_name:
            self.model_name = model_name.strip()

        if not self.model_name:
            raise DataSetGeneratorError("")

        include_sys_msg = sys_msg if sys_msg is not None else self.config.sys_msg

        # Calculate total samples requested
        total_samples = num_steps * batch_size
        data_creation_prompt = self._get_cot_prompt_template()

        # Ensure checkpoint_interval is at least as large as concurrency/batch_size
        # so checkpoints align with batch boundaries
        if (
            self.config.checkpoint_interval is not None
            and self.config.checkpoint_interval < batch_size
        ):
            logger.warning(
                "checkpoint_interval (%d) is less than concurrency/batch_size (%d), "
                "adjusting to %d to align with batch boundaries",
                self.config.checkpoint_interval,
                batch_size,
                batch_size,
            )
            self.config.checkpoint_interval = batch_size

        final_result: HFDataset | dict | None = None

        # Use cycle-based generation when a topic model is provided
        if topic_model is not None:
            unique_topics, cycles_needed = self._prepare_unique_topics(total_samples, topic_model)

            # batch_size becomes concurrency in the new model
            concurrency = batch_size

            async for event in self._run_cycle_based_generation_async(
                unique_topics=unique_topics,
                cycles_needed=cycles_needed,
                total_samples=total_samples,
                concurrency=concurrency,
                data_creation_prompt=data_creation_prompt,
                num_example_demonstrations=num_example_demonstrations,
                include_sys_msg=include_sys_msg,
                topic_model=topic_model,
            ):
                final_result = event
        else:
            # Fall back to step-based generation when no topic model
            topic_paths, num_steps = self._prepare_topic_paths(num_steps, batch_size, topic_model)

            async for event in self._run_generation_loop_async(
                num_steps=num_steps,
                batch_size=batch_size,
                total_samples=total_samples,
                topic_paths=topic_paths or [],
                data_creation_prompt=data_creation_prompt,
                num_example_demonstrations=num_example_demonstrations,
                include_sys_msg=include_sys_msg,
            ):
                final_result = event

        if isinstance(final_result, HFDataset):
            trace(
                "dataset_created",
                {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "conversation_type": self.config.conversation_type,
                    "samples_count": len(final_result),
                    "failed_samples": len(self.failed_samples),
                    "success": len(final_result) > 0,
                },
            )
            return final_result

        msg = "Dataset generation failed"
        raise DataSetGeneratorError(msg)

    async def create_data_with_events_async(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
    ) -> AsyncGenerator[dict | HFDataset, None]:
        if num_steps is None:
            num_steps = 1

        self._validate_create_data_params(num_steps, batch_size, topic_model)

        if model_name:
            self.model_name = model_name.strip()

        if not self.model_name:
            raise DataSetGeneratorError("")

        include_sys_msg = sys_msg if sys_msg is not None else self.config.sys_msg

        # Calculate total samples requested
        total_samples = num_steps * batch_size
        data_creation_prompt = self._get_cot_prompt_template()

        # Ensure checkpoint_interval is at least as large as concurrency/batch_size
        # so checkpoints align with batch boundaries
        if (
            self.config.checkpoint_interval is not None
            and self.config.checkpoint_interval < batch_size
        ):
            logger.warning(
                "checkpoint_interval (%d) is less than concurrency/batch_size (%d), "
                "adjusting to %d to align with batch boundaries",
                self.config.checkpoint_interval,
                batch_size,
                batch_size,
            )
            self.config.checkpoint_interval = batch_size

        root_topic_prompt = None
        topic_model_type = None
        if topic_model is not None:
            root_topic_prompt = getattr(topic_model, "topic_prompt", None)
            topic_model_type = type(topic_model).__name__.lower()

        # Use cycle-based generation when a topic model is provided
        if topic_model is not None:
            unique_topics, cycles_needed = self._prepare_unique_topics(total_samples, topic_model)

            # batch_size becomes concurrency in the new model
            concurrency = batch_size

            async for event in self._run_cycle_based_generation_async(
                unique_topics=unique_topics,
                cycles_needed=cycles_needed,
                total_samples=total_samples,
                concurrency=concurrency,
                data_creation_prompt=data_creation_prompt,
                num_example_demonstrations=num_example_demonstrations,
                include_sys_msg=include_sys_msg,
                root_topic_prompt=root_topic_prompt,
                topic_model_type=topic_model_type,
                topic_model=topic_model,
            ):
                yield event
        else:
            # Fall back to step-based generation when no topic model
            topic_paths, num_steps = self._prepare_topic_paths(num_steps, batch_size, topic_model)

            async for event in self._run_generation_loop_async(
                num_steps=num_steps,
                batch_size=batch_size,
                total_samples=total_samples,
                topic_paths=topic_paths or [],
                data_creation_prompt=data_creation_prompt,
                num_example_demonstrations=num_example_demonstrations,
                include_sys_msg=include_sys_msg,
                root_topic_prompt=root_topic_prompt,
                topic_model_type=topic_model_type,
            ):
                yield event

    async def _run_generation_loop_async(  # noqa: PLR0912, PLR0915
        self,
        num_steps: int,
        batch_size: int,
        total_samples: int,
        topic_paths: list[TopicPath],
        data_creation_prompt: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
        root_topic_prompt: str | None = None,
        topic_model_type: str | None = None,
    ) -> AsyncGenerator[dict | HFDataset, None]:
        """Run the main generation loop yielding progress events."""
        # Verify topic paths cover all expected samples (only when a topic model is used)
        expected_prompts = num_steps * batch_size
        actual_paths = len(topic_paths) if topic_paths else 0
        if topic_paths and actual_paths < expected_prompts:
            logger.warning(
                "Topic paths (%d) < expected samples (%d). Steps beyond path %d will produce 0 samples.",
                actual_paths,
                expected_prompts,
                actual_paths // batch_size,
            )

        # Initialize checkpoint paths if checkpointing is enabled
        if self.config.checkpoint_interval is not None:
            self._initialize_checkpoint_paths()

        # Track samples added in this run for checkpointing
        samples_since_checkpoint = 0
        samples_in_current_batch: list[dict] = []
        failures_in_current_batch: list[dict] = []
        topic_paths_in_current_batch: list[TopicPath | None] = []
        topics_exhausted_count = 0  # Samples not generated due to topic exhaustion

        try:
            yield {
                "event": "generation_start",
                "model_name": self.model_name,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "total_samples": total_samples,
                "root_topic_prompt": root_topic_prompt,
                "topic_model_type": topic_model_type,
                "resumed_from_checkpoint": len(self._completed) > 0,
                "previously_processed": len(self._completed),
                "resumed_samples": self._flushed_samples_count,
                "resumed_failures": self._flushed_failures_count,
                "checkpoint_enabled": self.config.checkpoint_interval is not None,
                "checkpoint_interval": self.config.checkpoint_interval,
            }

            for step in range(num_steps):
                yield {
                    "event": "step_start",
                    "step": step + 1,
                    "total_steps": num_steps,
                }

                start_idx = step * batch_size
                prompts, used_topic_paths = self._generate_batch_prompts(
                    batch_size,
                    start_idx,
                    topic_paths,
                    data_creation_prompt,
                    num_example_demonstrations,
                )

                # Handle topic exhaustion - when we've run out of topics
                if not prompts and topic_paths:
                    # Topics exhausted - remaining steps will produce nothing
                    exhausted_in_step = batch_size  # All samples in this batch had no topics
                    topics_exhausted_count += exhausted_in_step
                    logger.warning(
                        "Step %d: Topics exhausted at index %d (only %d topics available for %d expected)",
                        step + 1,
                        start_idx,
                        len(topic_paths),
                        total_samples,
                    )
                    yield {
                        "event": "step_complete",
                        "step": step + 1,
                        "samples_generated": 0,
                        "success": True,
                        "failed_in_step": 0,
                        "failure_reasons": [],
                        "topics_exhausted": exhausted_in_step,
                    }
                    continue

                # Filter out already-completed topics when resuming
                if self._completed:
                    filtered_prompts = []
                    filtered_topic_paths: list[TopicPath | None] = []
                    for prompt, tp in zip(prompts, used_topic_paths, strict=False):
                        # Check if this topic_id has been completed in any cycle
                        if tp is None or not self._is_uuid_completed_any_cycle(tp.topic_id):
                            filtered_prompts.append(prompt)
                            filtered_topic_paths.append(tp)

                    if not filtered_prompts:
                        # All topics in this batch were already processed
                        yield {
                            "event": "step_complete",
                            "step": step + 1,
                            "samples_generated": 0,
                            "success": True,
                            "failed_in_step": 0,
                            "failure_reasons": [],
                            "skipped": len(prompts),
                        }
                        continue

                    prompts = filtered_prompts
                    used_topic_paths = filtered_topic_paths

                failed_before = len(self.failed_samples)
                samples_before = len(self._samples)

                success, samples_generated = await self._process_batch_with_retries_async(
                    prompts, include_sys_msg, start_idx, used_topic_paths
                )

                # Track new samples and failures for checkpointing
                new_samples = self._samples[samples_before:]
                new_failures = self.failed_samples[failed_before:]
                samples_in_current_batch.extend(new_samples)
                failures_in_current_batch.extend(new_failures)
                topic_paths_in_current_batch.extend(used_topic_paths)
                samples_since_checkpoint += samples_generated

                # Save checkpoint if we've reached the interval
                if (
                    self.config.checkpoint_interval is not None
                    and samples_since_checkpoint >= self.config.checkpoint_interval
                ):
                    # Convert topic_paths to (uuid, cycle) tuples (cycle=0 for step-based)
                    completed_items = [
                        (tp.topic_id, 0) for tp in topic_paths_in_current_batch if tp is not None
                    ]
                    self._save_checkpoint(
                        samples_in_current_batch,
                        failures_in_current_batch,
                        completed_items,
                    )
                    samples_in_current_batch = []
                    failures_in_current_batch = []
                    topic_paths_in_current_batch = []
                    samples_since_checkpoint = 0
                    yield {
                        "event": "checkpoint_saved",
                        "total_samples": self._flushed_samples_count,
                        "total_failures": self._flushed_failures_count,
                    }

                    # Check for graceful stop request after checkpoint save
                    if self.stop_requested:
                        yield {
                            "event": "generation_stopped",
                            "message": "Stopped at checkpoint as requested",
                            "total_samples": self._flushed_samples_count,
                            "total_failures": self._flushed_failures_count,
                        }
                        return  # Exit generator cleanly

                # Use new_failures captured BEFORE checkpoint clear, not recalculated after
                failed_in_batch = len(new_failures)
                failure_reasons: list[str] = []
                if failed_in_batch > 0 and new_failures:
                    for f in new_failures[:3]:
                        if isinstance(f, dict):
                            failure_reasons.append(f.get("error", str(f)))
                        else:
                            failure_reasons.append(str(f))

                yield {
                    "event": "step_complete",
                    "step": step + 1,
                    "samples_generated": samples_generated,
                    "success": success,
                    "failed_in_step": failed_in_batch,
                    "failure_reasons": failure_reasons,
                }

                if not success:
                    yield {
                        "event": "step_failed",
                        "step": step + 1,
                        "message": f"Failed to process batch {step + 1} after all retries",
                    }

            # Save final checkpoint with any remaining samples
            if self.config.checkpoint_interval is not None and (
                samples_in_current_batch or failures_in_current_batch
            ):
                completed_items = [
                    (tp.topic_id, 0) for tp in topic_paths_in_current_batch if tp is not None
                ]
                self._save_checkpoint(
                    samples_in_current_batch,
                    failures_in_current_batch,
                    completed_items,
                )
                yield {
                    "event": "checkpoint_saved",
                    "total_samples": self._flushed_samples_count,
                    "total_failures": self._flushed_failures_count,
                    "final": True,
                }

            # Calculate total counts including flushed data
            actual_samples = self._flushed_samples_count + len(self._samples)
            actual_failures = self._flushed_failures_count + len(self.failed_samples)
            total_accounted = actual_samples + actual_failures + topics_exhausted_count
            true_unaccounted = total_samples - total_accounted

            # Log accounting summary for debugging
            logger.info(
                "Generation complete: expected=%d, generated=%d, failed=%d, topics_exhausted=%d, "
                "accounted=%d, unaccounted=%d",
                total_samples,  # This is the parameter passed in (expected count)
                actual_samples,
                actual_failures,
                topics_exhausted_count,
                total_accounted,
                true_unaccounted,
            )

            yield {
                "event": "generation_complete",
                "total_samples": actual_samples,
                "failed_samples": actual_failures,
                "expected_samples": total_samples,
                "topics_exhausted": topics_exhausted_count,
                "unaccounted": true_unaccounted,
            }

        except KeyboardInterrupt:
            # Save checkpoint on interrupt
            if self.config.checkpoint_interval is not None and (
                samples_in_current_batch or failures_in_current_batch
            ):
                completed_items = [
                    (tp.topic_id, 0) for tp in topic_paths_in_current_batch if tp is not None
                ]
                self._save_checkpoint(
                    samples_in_current_batch,
                    failures_in_current_batch,
                    completed_items,
                )
            yield {
                "event": "generation_interrupted",
                "message": "Generation interrupted by user.",
            }
            self.print_failure_summary()
            self._save_samples_to_file(INTERRUPTED_DATASET_FILENAME)

        except Exception as e:  # noqa: BLE001
            # Save checkpoint on error
            if self.config.checkpoint_interval is not None and (
                samples_in_current_batch or failures_in_current_batch
            ):
                completed_items = [
                    (tp.topic_id, 0) for tp in topic_paths_in_current_batch if tp is not None
                ]
                self._save_checkpoint(
                    samples_in_current_batch,
                    failures_in_current_batch,
                    completed_items,
                )
            yield {"event": "generation_error", "error": str(e)}
            self.print_failure_summary()
            self._save_samples_to_file(ERROR_DATASET_FILENAME)
            raise DataSetGeneratorError("failed") from e

        # Build final dataset: if samples were flushed to disk, load them from checkpoint
        if self._flushed_samples_count > 0:
            all_samples = self._load_all_samples_from_checkpoint()
            yield HFDataset.from_list(all_samples) if all_samples else HFDataset.from_list([])
        else:
            yield (HFDataset.from_list(self._samples) if self._samples else HFDataset.from_list([]))

    async def _run_cycle_based_generation_async(  # noqa: PLR0912, PLR0915
        self,
        unique_topics: list[Topic],
        cycles_needed: int,
        total_samples: int,
        concurrency: int,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
        root_topic_prompt: str | None = None,
        topic_model_type: str | None = None,
        topic_model: "TopicModel | None" = None,
    ) -> AsyncGenerator[dict | HFDataset, None]:
        """Run cycle-based generation loop yielding progress events.

        This is the new generation model that iterates over unique topics (by UUID)
        for multiple cycles, rather than the old step-based batching approach.

        Args:
            unique_topics: List of Topic namedtuples with (uuid, topic).
            cycles_needed: Number of cycles to iterate through topics.
            total_samples: Total number of samples to generate.
            concurrency: Maximum parallel LLM calls (semaphore limit).
            data_creation_prompt: The prompt template for data creation.
            num_example_demonstrations: Number of example demonstrations to include.
            include_sys_msg: Whether to include system message in output.
            root_topic_prompt: Original topic prompt for display.
            topic_model_type: Type of topic model (tree, graph) for display.
            topic_model: Topic model for path lookup (recovers full path context).

        Yields:
            Progress event dicts and final HFDataset.
        """
        unique_topic_count = len(unique_topics)
        final_cycle_size = total_samples - (cycles_needed - 1) * unique_topic_count

        # Initialize checkpoint paths if checkpointing is enabled
        if self.config.checkpoint_interval is not None:
            self._initialize_checkpoint_paths()

        # Track samples for checkpointing
        samples_since_checkpoint = 0
        pending_samples: list[dict] = []
        pending_failures: list[dict] = []
        pending_completed: list[tuple[str, int]] = []
        samples_generated = 0

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        try:
            yield {
                "event": "generation_start",
                "model_name": self.model_name,
                "unique_topics": unique_topic_count,
                "cycles_needed": cycles_needed,
                "final_cycle_size": final_cycle_size,
                "concurrency": concurrency,
                "total_samples": total_samples,
                "root_topic_prompt": root_topic_prompt,
                "topic_model_type": topic_model_type,
                "resumed_from_checkpoint": len(self._completed) > 0,
                "previously_completed": len(self._completed),
                "resumed_samples": self._flushed_samples_count,
                "resumed_failures": self._flushed_failures_count,
                "checkpoint_enabled": self.config.checkpoint_interval is not None,
                "checkpoint_interval": self.config.checkpoint_interval,
            }

            for cycle in range(cycles_needed):
                # Determine how many topics to process in this cycle
                if cycle == cycles_needed - 1:
                    # Final cycle may be partial
                    topics_in_cycle = min(final_cycle_size, unique_topic_count)
                else:
                    topics_in_cycle = unique_topic_count

                yield {
                    "event": "cycle_start",
                    "cycle": cycle + 1,
                    "total_cycles": cycles_needed,
                    "topics_in_cycle": topics_in_cycle,
                }

                # Collect topics to process in this cycle
                topics_to_process: list[Topic] = []
                for topic_idx, topic in enumerate(unique_topics):
                    if cycle == cycles_needed - 1 and topic_idx >= final_cycle_size:
                        break  # Partial final cycle - stop early

                    if not self._is_completed(topic.uuid, cycle):
                        topics_to_process.append(topic)

                if not topics_to_process:
                    # All topics in this cycle already completed (resume scenario)
                    yield {
                        "event": "cycle_complete",
                        "cycle": cycle + 1,
                        "samples_in_cycle": 0,
                        "skipped": topics_in_cycle,
                    }
                    continue

                # Process topics with concurrency control
                cycle_samples = 0
                cycle_failures = 0

                async def process_topic(
                    topic: Topic, cycle_num: int, sample_idx: int
                ) -> tuple[tuple[str, int], bool, int]:
                    """Process a single topic with semaphore-controlled concurrency.

                    Note: Samples/failures are NOT extracted here because concurrent
                    tasks share self._samples. Extracting per-task would cause
                    duplicates since each task's slice overlaps with others.
                    Instead, samples are captured in bulk after asyncio.gather.

                    Returns:
                        Tuple of (completed_item, success, count)
                    """
                    async with semaphore:
                        # Recover full path context (root -> ... -> leaf) for prompt quality
                        full_path = topic_model.get_path_by_id(topic.uuid) if topic_model else None
                        subtopics = full_path if full_path else [topic.topic]

                        # Build prompt for this topic
                        sample_prompt = self.build_prompt(
                            data_creation_prompt=data_creation_prompt,
                            num_example_demonstrations=num_example_demonstrations,
                            subtopics_list=subtopics,
                        )

                        # Use existing batch processing for a single sample
                        topic_path = TopicPath(path=subtopics, topic_id=topic.uuid)
                        success, count = await self._process_batch_with_retries_async(
                            prompts=[sample_prompt],
                            include_sys_msg=include_sys_msg,
                            start_sample_idx=sample_idx,
                            topic_paths_for_batch=[topic_path],
                        )

                        completed_item = (topic.uuid, cycle_num)
                        return completed_item, success, count

                # Process topics in batches for checkpoint saving
                for batch_start in range(0, len(topics_to_process), concurrency):
                    batch_end = min(batch_start + concurrency, len(topics_to_process))
                    batch_topics = topics_to_process[batch_start:batch_end]

                    # Snapshot list lengths before tasks start so we can
                    # capture new items in one safe slice after all complete.
                    samples_before_gather = len(self._samples)
                    failures_before_gather = len(self.failed_samples)

                    # Create concurrent tasks
                    # Pass current samples_generated as starting index for each task
                    tasks = [
                        asyncio.create_task(process_topic(topic, cycle, samples_generated + i))
                        for i, topic in enumerate(batch_topics)
                    ]

                    # Process results as each task finishes (not waiting for
                    # the whole batch) so the progress bar advances per-sample.
                    batch_samples = 0
                    batch_failures = 0
                    for future in asyncio.as_completed(tasks):
                        completed_item, success, count = await future
                        pending_completed.append(completed_item)

                        if success:
                            cycle_samples += count
                            samples_generated += count
                            batch_samples += count
                        else:
                            cycle_failures += 1
                            batch_failures += 1

                        samples_since_checkpoint += 1

                        # Emit per-sample progress so progress bars advance immediately
                        yield {
                            "event": "batch_complete",
                            "samples_generated": count if success else 0,
                            "samples_failed": 1 if not success else 0,
                        }

                    # After all tasks complete, capture samples/failures in
                    # one safe slice (no concurrent tasks are running now).
                    batch_new_samples = list(self._samples[samples_before_gather:])
                    batch_new_failures = list(self.failed_samples[failures_before_gather:])
                    # Tag each failure with the cycle number so retry logic can
                    # target only the specific (uuid, cycle) that failed
                    for failure in batch_new_failures:
                        failure["cycle"] = cycle
                    pending_samples.extend(batch_new_samples)
                    pending_failures.extend(batch_new_failures)

                    # Save checkpoint if we've reached the interval
                    if (
                        self.config.checkpoint_interval is not None
                        and samples_since_checkpoint >= self.config.checkpoint_interval
                    ):
                        self._save_checkpoint(
                            pending_samples,
                            pending_failures,
                            pending_completed,
                        )
                        pending_samples = []
                        pending_failures = []
                        pending_completed = []
                        samples_since_checkpoint = 0
                        yield {
                            "event": "checkpoint_saved",
                            "total_samples": self._flushed_samples_count,
                            "total_failures": self._flushed_failures_count,
                        }

                        # Check for graceful stop request
                        if self.stop_requested:
                            yield {
                                "event": "generation_stopped",
                                "message": "Stopped at checkpoint as requested",
                                "total_samples": self._flushed_samples_count,
                                "total_failures": self._flushed_failures_count,
                            }
                            return

                yield {
                    "event": "cycle_complete",
                    "cycle": cycle + 1,
                    "samples_in_cycle": cycle_samples,
                    "failures_in_cycle": cycle_failures,
                }

            # Save final checkpoint with any remaining samples
            if self.config.checkpoint_interval is not None and (
                pending_samples or pending_failures
            ):
                self._save_checkpoint(
                    pending_samples,
                    pending_failures,
                    pending_completed,
                )
                yield {
                    "event": "checkpoint_saved",
                    "total_samples": self._flushed_samples_count,
                    "total_failures": self._flushed_failures_count,
                    "final": True,
                }

            # Calculate total counts including flushed data
            actual_samples = self._flushed_samples_count + len(self._samples)
            actual_failures = self._flushed_failures_count + len(self.failed_samples)
            unaccounted = total_samples - actual_samples - actual_failures

            logger.info(
                "Generation complete: expected=%d, generated=%d, failed=%d, "
                "accounted=%d, unaccounted=%d",
                total_samples,
                actual_samples,
                actual_failures,
                actual_samples + actual_failures,
                unaccounted,
            )

            yield {
                "event": "generation_complete",
                "total_samples": actual_samples,
                "failed_samples": actual_failures,
                "expected_samples": total_samples,
                "unaccounted": unaccounted,
                "cycles_completed": cycles_needed,
                "unique_topics": unique_topic_count,
            }

        except KeyboardInterrupt:
            # Save checkpoint on interrupt
            if self.config.checkpoint_interval is not None and (
                pending_samples or pending_failures
            ):
                self._save_checkpoint(
                    pending_samples,
                    pending_failures,
                    pending_completed,
                )
            yield {
                "event": "generation_interrupted",
                "message": "Generation interrupted by user.",
            }
            self.print_failure_summary()
            self._save_samples_to_file(INTERRUPTED_DATASET_FILENAME)

        except Exception as e:  # noqa: BLE001
            # Save checkpoint on error
            if self.config.checkpoint_interval is not None and (
                pending_samples or pending_failures
            ):
                self._save_checkpoint(
                    pending_samples,
                    pending_failures,
                    pending_completed,
                )
            yield {"event": "generation_error", "error": str(e)}
            self.print_failure_summary()
            self._save_samples_to_file(ERROR_DATASET_FILENAME)
            raise DataSetGeneratorError("failed") from e

        # Build final dataset: if samples were flushed to disk, load them from checkpoint
        if self._flushed_samples_count > 0:
            all_samples = self._load_all_samples_from_checkpoint()
            yield HFDataset.from_list(all_samples) if all_samples else HFDataset.from_list([])
        else:
            yield (HFDataset.from_list(self._samples) if self._samples else HFDataset.from_list([]))

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

    def _save_samples_to_file(self, save_path: str):
        """Save the current samples to a JSONL file."""

        with open(save_path, "w", encoding="utf-8") as f:
            for sample in self._samples:
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")

    def save_dataset(self, save_path: str):
        """Save the dataset to a JSONL file."""
        self._save_samples_to_file(save_path)
