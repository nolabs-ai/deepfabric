import asyncio
import contextlib
import json
import math
import os
import traceback

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset as HFDataset
from rich.layout import Layout
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column

from .config import DeepFabricConfig
from .config_manager import DEFAULT_MODEL
from .exceptions import ConfigurationError
from .generator import DataSetGenerator
from .progress import ProgressReporter
from .tui import STREAM_PANEL_WIDTH, get_dataset_tui, get_tui
from .utils import ensure_not_running_loop


# Lazy/defensive access to TUI settings to avoid early import issues
def _get_tui_settings():
    try:
        from .tui import get_tui_settings as _gts  # noqa: PLC0415

        return _gts()
    except Exception:

        class _S:
            mode = "rich"

        return _S()


def _get_preview_lines() -> int:
    try:
        from .tui import get_preview_lines as _gpl  # noqa: PLC0415

        return _gpl()
    except Exception:
        return 16


if TYPE_CHECKING:
    from .topic_model import TopicModel

# Constants for debug output
DEBUG_MAX_FAILURES_TO_SHOW = 10


def resolve_num_samples(num_samples: int | str, topic_count: int) -> int:
    """Resolve num_samples to an integer based on topic count.

    Args:
        num_samples: Integer, "auto", or percentage string like "50%"
        topic_count: Number of available unique topics

    Returns:
        Resolved integer sample count

    Raises:
        ConfigurationError: If topic_count is 0 and dynamic sampling is requested
    """
    if isinstance(num_samples, int):
        return num_samples

    if topic_count == 0:
        raise ConfigurationError(
            "Cannot use 'auto' or percentage num_samples with empty topic model. "
            "Ensure topic generation produced topics."
        )

    if num_samples == "auto":
        return topic_count

    if isinstance(num_samples, str) and num_samples.endswith("%"):
        percentage = float(num_samples[:-1]) / 100.0
        return max(1, int(topic_count * percentage))

    # Fallback - try to parse as int (shouldn't reach here if validated properly)
    return int(num_samples)


async def handle_dataset_events_async(
    generator: AsyncIterator[dict | HFDataset], engine=None, debug: bool = False
) -> HFDataset | None:
    """Handle dataset generation with TUI progress and streaming feedback."""
    tui = get_dataset_tui()
    footer_prog = None
    task = None
    live = None
    simple_progress = None  # Progress bar for simple/headless mode
    simple_progress_task = None
    simple_checkpoint_task = None  # Checkpoint progress task for simple mode
    checkpoint_interval = 0
    headless_completed = 0  # Counter for non-TTY headless mode
    headless_total = 0
    headless_batch_size = 0  # Print threshold for headless mode
    headless_last_printed = 0  # Last count at which we printed

    final_result: HFDataset | None = None
    try:
        async for event in generator:
            if isinstance(event, dict) and "event" in event:
                if event["event"] == "generation_start":
                    settings = _get_tui_settings()
                    # Handle both cycle-based and step-based event formats
                    # Cycle-based: unique_topics, cycles_needed, concurrency
                    # Step-based: num_steps, batch_size
                    is_cycle_based = "cycles_needed" in event
                    if is_cycle_based:
                        display_steps = event.get("cycles_needed", 1)
                        display_batch_size = event.get("concurrency", 1)
                    else:
                        display_steps = event.get("num_steps", 1)
                        display_batch_size = event.get("batch_size", 1)
                    # Build header and params panels for layout
                    header_panel, params_panel = tui.build_generation_panels(
                        event["model_name"],
                        display_steps,
                        display_batch_size,
                        total_samples=event["total_samples"],
                        is_cycle_based=is_cycle_based,
                        unique_topics=event.get("unique_topics", 0),
                        final_cycle_size=event.get("final_cycle_size", 0),
                        checkpoint_interval=event.get("checkpoint_interval", 0),
                    )
                    # Capture context for the run
                    tui.root_topic_prompt = event.get("root_topic_prompt")
                    tui.topic_model_type = event.get("topic_model_type")

                    if settings.mode == "rich":
                        # Initialize status tracking
                        tui.init_status(
                            total_steps=display_steps,
                            total_samples=event["total_samples"],
                            checkpoint_enabled=event.get("checkpoint_enabled", False),
                            is_cycle_based=is_cycle_based,
                        )

                        # Build layout with footer card
                        layout = Layout(name="root")
                        layout.split(Layout(name="main"), Layout(name="footer", size=3))
                        left = Layout(name="left", ratio=3)
                        right = Layout(name="right", ratio=2)
                        right.minimum_size = STREAM_PANEL_WIDTH
                        # Right column: status on top, streaming preview fills remaining space
                        right.split(
                            Layout(name="status", size=8),
                            Layout(name="preview"),
                        )
                        left.split(
                            Layout(name="header", size=4),
                            Layout(name="params", size=6),
                            Layout(name="context", size=5),
                            Layout(name="events"),
                        )
                        left["header"].update(header_panel)
                        left["params"].update(params_panel)
                        left["context"].update(tui._context_panel())
                        left["events"].update(tui.tui.build_events_panel([], title="Events"))
                        right["status"].update(tui._status_panel())
                        right["preview"].update(
                            tui.tui.build_stream_panel("Waiting for LLM output...")
                        )
                        layout["main"].split_row(left, right)

                        prog_total = event["total_samples"]
                        resumed_samples = event.get("resumed_samples", 0)

                        # Footer run status
                        footer_prog = tui.tui.create_footer(layout, title="Run Status")
                        task = footer_prog.add_task(
                            "Generating dataset samples",
                            total=prog_total,
                            completed=min(resumed_samples, prog_total),
                        )

                        # Use alternate screen to avoid scroll trails; leave a clean terminal
                        live = Live(
                            layout,
                            console=tui.console,
                            refresh_per_second=15,
                            screen=True,
                        )
                        tui.live_display = live  # Give TUI reference to update it
                        tui.live_layout = layout  # Allow TUI to update panes
                        live.start()
                        if resumed_samples >= prog_total:
                            tui.log_event(
                                f"Checkpoint already complete: {resumed_samples} samples "
                                f"(target: {prog_total})"
                            )
                    else:
                        prog_total = event["total_samples"]
                        resumed_samples = event.get("resumed_samples", 0)

                        # Simple/headless mode: runtime summary then progress bar
                        tui.console.print("\n[bold cyan]Dataset Generation[/bold cyan]")
                        model_line = f"Model: {event['model_name']}"
                        if event.get("topic_model_type"):
                            topic_type = event["topic_model_type"]
                            if is_cycle_based and event.get("unique_topics"):
                                model_line += (
                                    f" ({topic_type}, {event['unique_topics']} unique topics)"
                                )
                            else:
                                model_line += f" ({topic_type})"
                        tui.info(model_line)

                        if is_cycle_based:
                            output_line = (
                                f"Output: num_samples={prog_total}, "
                                f"concurrency={display_batch_size}"
                            )
                        else:
                            output_line = (
                                f"Output: num_samples={prog_total}, batch_size={display_batch_size}"
                            )
                        tui.info(output_line)

                        if is_cycle_based:
                            cycles = event.get("cycles_needed", 1)
                            unique = event.get("unique_topics", 0)
                            tui.info(
                                f"  → Cycles needed: {cycles} "
                                f"({prog_total} samples ÷ {unique} unique topics)"
                            )
                            final_cycle = event.get("final_cycle_size", 0)
                            if final_cycle and unique and final_cycle < unique:
                                tui.info(f"  → Final cycle: {final_cycle} topics (partial)")

                        tui.console.print()
                        cp_interval = event.get("checkpoint_interval")
                        if cp_interval and cp_interval > 0:
                            total_cp = math.ceil(prog_total / cp_interval)
                            tui.info(
                                f"Checkpoint: every {cp_interval} samples "
                                f"({total_cp} total checkpoints)"
                            )
                            tui.console.print()

                        if resumed_samples >= prog_total:
                            # Checkpoint already has enough samples
                            tui.success(
                                f"Checkpoint already complete: {resumed_samples} samples "
                                f"(target: {prog_total})"
                            )
                        elif tui.console.is_terminal:
                            simple_progress = Progress(
                                SpinnerColumn(),
                                TextColumn("[progress.description]{task.description}"),
                                BarColumn(),
                                MofNCompleteColumn(table_column=Column(justify="right")),
                                TimeElapsedColumn(),
                                console=tui.console,
                            )
                            simple_progress_task = simple_progress.add_task(
                                "Generating",
                                total=prog_total,
                                completed=resumed_samples,
                            )
                            simple_progress.start()
                            tui.simple_progress = simple_progress
                            # Add checkpoint progress task if interval is set
                            checkpoint_interval = event.get("checkpoint_interval") or 0
                            if checkpoint_interval > 0:
                                simple_checkpoint_task = simple_progress.add_task(
                                    "Next checkpoint",
                                    total=checkpoint_interval,
                                    completed=0,
                                )
                        else:
                            # Headless (non-TTY): track progress with counters
                            headless_total = prog_total
                            headless_completed = resumed_samples
                            headless_batch_size = display_batch_size
                            headless_last_printed = resumed_samples
                            checkpoint_interval = event.get("checkpoint_interval") or 0
                elif event["event"] == "step_complete":
                    samples_generated = event.get("samples_generated", 0)
                    if footer_prog and task is not None:
                        if samples_generated > 0:
                            with contextlib.suppress(Exception):
                                footer_prog.update(task, advance=samples_generated)
                            tui.log_event(f"✓ Generated +{samples_generated} samples")
                            # Update status totals
                            tui.status_step_complete(
                                samples_generated, int(event.get("failed_in_step", 0))
                            )
                    elif simple_progress is not None and simple_progress_task is not None:
                        with contextlib.suppress(Exception):
                            simple_progress.update(simple_progress_task, advance=samples_generated)
                        if simple_checkpoint_task is not None and samples_generated > 0:
                            with contextlib.suppress(Exception):
                                simple_progress.update(
                                    simple_checkpoint_task, advance=samples_generated
                                )
                        tui.clear_step_retries()
                    elif headless_total > 0 and samples_generated > 0:
                        headless_completed += samples_generated
                        if (
                            headless_completed - headless_last_printed >= headless_batch_size
                            or headless_completed >= headless_total
                        ):
                            tui.info(f"Generated {headless_completed}/{headless_total} samples")
                            headless_last_printed = headless_completed
                elif event["event"] == "step_start":
                    # Keep status panel in sync
                    step = int(event.get("step", 0))
                    total = int(event.get("total_steps", 0))
                    tui.status_step_start(step, total)

                elif event["event"] == "cycle_start":
                    # Cycle-based generation: keep status panel in sync
                    cycle = int(event.get("cycle", 0))
                    total_cycles = int(event.get("total_cycles", 0))
                    tui.status_step_start(cycle, total_cycles)

                elif event["event"] == "batch_complete":
                    # Per-batch progress: advance bars after each concurrency batch
                    batch_generated = event.get("samples_generated", 0)
                    batch_failed = event.get("samples_failed", 0)
                    if footer_prog and task is not None:
                        if batch_generated > 0:
                            with contextlib.suppress(Exception):
                                footer_prog.update(task, advance=batch_generated)
                            tui.status_step_complete(batch_generated, batch_failed)
                    elif simple_progress is not None and simple_progress_task is not None:
                        with contextlib.suppress(Exception):
                            simple_progress.update(simple_progress_task, advance=batch_generated)
                        if simple_checkpoint_task is not None and batch_generated > 0:
                            with contextlib.suppress(Exception):
                                simple_progress.update(
                                    simple_checkpoint_task, advance=batch_generated
                                )
                    elif headless_total > 0 and batch_generated > 0:
                        headless_completed += batch_generated
                        if (
                            headless_completed - headless_last_printed >= headless_batch_size
                            or headless_completed >= headless_total
                        ):
                            tui.info(f"Generated {headless_completed}/{headless_total} samples")
                            headless_last_printed = headless_completed

                elif event["event"] == "cycle_complete":
                    # Cycle-based generation: log cycle summary (progress already advanced by batch_complete)
                    samples_in_cycle = event.get("samples_in_cycle", 0)
                    failures_in_cycle = event.get("failures_in_cycle", 0)
                    if footer_prog and task is not None:
                        tui.log_event(
                            f"✓ Cycle {event.get('cycle')}: "
                            f"+{samples_in_cycle} samples"
                            + (f" (-{failures_in_cycle} failed)" if failures_in_cycle else "")
                        )
                    elif headless_total > 0:
                        msg = f"Cycle {event.get('cycle')}: +{samples_in_cycle} samples"
                        if failures_in_cycle:
                            msg += f" (-{failures_in_cycle} failed)"
                        tui.info(msg)

                elif event["event"] == "checkpoint_saved":
                    # Display checkpoint save notification
                    total_samples = event.get("total_samples", 0)
                    is_final = event.get("final", False)

                    if footer_prog and task is not None:
                        # Rich mode: log to events panel and update status
                        if is_final:
                            tui.log_event(f"💾 Final checkpoint: {total_samples} samples")
                        else:
                            tui.log_event(f"💾 Checkpoint: {total_samples} samples")
                        tui.status_checkpoint_saved(total_samples)
                    elif simple_progress is not None:
                        # Simple mode: reset checkpoint progress bar instead of stacking print lines
                        if simple_checkpoint_task is not None and not is_final:
                            with contextlib.suppress(Exception):
                                simple_progress.reset(
                                    simple_checkpoint_task, total=checkpoint_interval
                                )
                    elif headless_total > 0:
                        if is_final:
                            tui.info(f"Checkpoint (final): {total_samples} samples saved")
                        else:
                            tui.info(f"Checkpoint: {total_samples} samples saved")

                elif event["event"] == "generation_stopped":
                    # Graceful stop at checkpoint
                    if live:
                        live.stop()
                    if simple_progress is not None:
                        simple_progress.stop()
                        tui.simple_progress = None
                    tui.console.print()
                    tui.success(
                        f"Gracefully stopped: {event['total_samples']} samples saved to checkpoint"
                    )
                    if event.get("total_failures", 0) > 0:
                        tui.info(f"({event['total_failures']} failures recorded)")
                    tui.info("Resume with: --resume flag")

                elif event["event"] == "generation_complete":
                    if live:
                        live.stop()
                    if simple_progress is not None:
                        simple_progress.stop()
                        tui.simple_progress = None
                    tui.console.print()  # Add blank line after live display
                    tui.success(f"Successfully generated {event['total_samples']} samples")

                    # Show accounting summary
                    expected = event.get("expected_samples", 0)
                    topics_exhausted = event.get("topics_exhausted", 0)
                    unaccounted = event.get("unaccounted", 0)
                    tui.log_event(
                        f"Done • expected={expected} generated={event['total_samples']} "
                        f"failed={event['failed_samples']} topics_exhausted={topics_exhausted} "
                        f"unaccounted={unaccounted}"
                    )
                    if event["failed_samples"] > 0:
                        tui.warning(f"Failed to generate {event['failed_samples']} samples")
                    if topics_exhausted > 0:
                        tui.warning(
                            f"Topics exhausted: {topics_exhausted} samples could not be generated "
                            f"(not enough unique topics for requested sample count)"
                        )
                    if unaccounted > 0:
                        tui.error(
                            f"WARNING: {unaccounted} samples unaccounted for "
                            f"(neither generated nor recorded as failures)"
                        )

                        # Show detailed failure information in debug mode
                        if debug and engine and hasattr(engine, "failed_samples"):
                            get_tui().error("\n🔍 Debug: Dataset generation failures:")
                            for idx, failure in enumerate(
                                engine.failed_samples[:DEBUG_MAX_FAILURES_TO_SHOW], 1
                            ):
                                get_tui().error(f"  [{idx}] {failure}")
                            if len(engine.failed_samples) > DEBUG_MAX_FAILURES_TO_SHOW:
                                remaining = len(engine.failed_samples) - DEBUG_MAX_FAILURES_TO_SHOW
                                get_tui().error(f"  ... and {remaining} more failures")

            elif isinstance(event, HFDataset):
                final_result = event
            else:
                # Handle unexpected non-dict, non-Dataset events
                get_tui().warning(f"Unexpected event type: {type(event)}")
    except Exception as e:
        if live:
            live.stop()
        if simple_progress is not None:
            simple_progress.stop()
            tui.simple_progress = None
        if debug:
            get_tui().error(f"🔍 Debug: Full traceback:\n{traceback.format_exc()}")
        get_tui().error(f"Dataset generation failed: {str(e)}")
        raise

    return final_result


def handle_dataset_events(generator, engine=None, debug: bool = False) -> HFDataset | None:
    """Synchronous wrapper for async dataset event handling."""
    ensure_not_running_loop("handle_dataset_events")
    return asyncio.run(handle_dataset_events_async(generator, engine=engine, debug=debug))


def create_dataset(
    engine: DataSetGenerator,
    topic_model: "TopicModel",
    config: DeepFabricConfig,
    num_samples: int | str | None = None,
    batch_size: int | None = None,
    include_system_message: bool | None = None,
    provider: str | None = None,  # noqa: ARG001
    model: str | None = None,
    generation_overrides: dict | None = None,
    debug: bool = False,
) -> HFDataset:
    """
    Create dataset using the data engine and topic model.

    Args:
        engine: DataSetGenerator instance
        topic_model: TopicModel (Tree or Graph) to use for generation
        config: DeepFabricConfig object
        num_samples: Override for number of samples (int, "auto", or percentage like "50%")
        batch_size: Override for batch size
        include_system_message: Override for including system message
        provider: Override for LLM provider
        model: Override for model name
        generation_overrides: Additional generation parameter overrides

    Returns:
        Generated HuggingFace Dataset object

    Raises:
        ConfigurationError: If dataset generation fails
    """
    ensure_not_running_loop("create_dataset")
    return asyncio.run(
        create_dataset_async(
            engine=engine,
            topic_model=topic_model,
            config=config,
            num_samples=num_samples,
            batch_size=batch_size,
            include_system_message=include_system_message,
            provider=provider,
            model=model,
            generation_overrides=generation_overrides,
            debug=debug,
        )
    )


async def create_dataset_async(
    engine: DataSetGenerator,
    topic_model: "TopicModel",
    config: DeepFabricConfig,
    num_samples: int | str | None = None,
    batch_size: int | None = None,
    include_system_message: bool | None = None,
    provider: str | None = None,  # noqa: ARG001
    model: str | None = None,
    generation_overrides: dict | None = None,
    debug: bool = False,
) -> HFDataset:
    output_config = config.get_output_config()

    raw_num_samples = num_samples if num_samples is not None else output_config["num_samples"]
    final_batch_size = batch_size or output_config["batch_size"]

    # Use the same unique topics that drive cycle-based generation.
    topic_count = len(topic_model.get_unique_topics())
    final_num_samples = resolve_num_samples(raw_num_samples, topic_count)

    # Log resolution for dynamic values
    tui = get_dataset_tui()
    if isinstance(raw_num_samples, str):
        tui.info(f"Resolved num_samples: {raw_num_samples} → {final_num_samples} samples")

    generation_params = config.get_generation_params(**(generation_overrides or {}))
    final_model = model or generation_params.get("model_name", DEFAULT_MODEL)

    # Still compute num_steps for backward compat with the generator's step-based path
    final_num_steps = math.ceil(final_num_samples / final_batch_size)

    # Create progress reporter and attach TUI as observer for streaming feedback
    progress_reporter = ProgressReporter()
    progress_reporter.attach(tui)

    # Attach progress reporter to engine and its LLM retry handler
    engine.progress_reporter = progress_reporter
    if hasattr(engine, "llm_client"):
        engine.llm_client.retry_handler.progress_reporter = progress_reporter

    try:
        generator = engine.create_data_with_events_async(
            num_steps=final_num_steps,
            batch_size=final_batch_size,
            topic_model=topic_model,
            model_name=final_model,
            sys_msg=include_system_message,
            num_example_demonstrations=output_config.get("num_example_demonstrations") or 3,
        )
        dataset = await handle_dataset_events_async(generator, engine=engine, debug=debug)
    except Exception as e:  # noqa: BLE001
        raise ConfigurationError(f"Error creating dataset: {str(e)}") from e

    if dataset is None:
        raise ConfigurationError("Dataset generation failed - no dataset returned")

    return dataset


def _upload_to_service(
    service_name: str,
    dataset_path: str,
    config: dict,
    credential_check_func,
    uploader_import_func,
    uploader_args_func,
    push_args_func,
    tui,
) -> None:
    """Generic function to upload dataset to any configured service."""
    try:
        tui.info(f"Uploading dataset to {service_name}...")

        # Check credentials
        credentials = credential_check_func()
        if not credentials:
            return

        # Import uploader class
        uploader_class = uploader_import_func()

        # Create uploader instance
        uploader_args = uploader_args_func(credentials)
        uploader = (
            uploader_class(*uploader_args)
            if isinstance(uploader_args, tuple)
            else uploader_class(**uploader_args)
        )

        # Prepare push arguments
        push_args = push_args_func(config, dataset_path)

        # Upload dataset
        result = uploader.push_to_hub(**push_args)

        if result["status"] == "success":
            tui.success(result["message"])
        else:
            tui.warning(f"{service_name} upload failed: {result['message']}")

    except Exception as e:
        tui.warning(f"Error uploading to {service_name}: {str(e)}")


def _upload_to_huggingface(dataset_path: str, hf_config: dict, tui) -> None:
    """Upload dataset to HuggingFace Hub if configured."""

    def check_credentials():
        token = os.getenv("HF_TOKEN")
        if not token:
            tui.warning("HF_TOKEN not set. Skipping HuggingFace upload.")
            return None
        return token

    def import_uploader():
        from .hf_hub import HFUploader  # noqa: PLC0415

        return HFUploader

    def get_uploader_args(credentials):
        return (credentials,)  # HFUploader takes token as single argument

    def get_push_args(config, dataset_path):
        return {
            "hf_dataset_repo": config["repository"],
            "jsonl_file_path": dataset_path,
            "tags": config.get("tags", []),
        }

    _upload_to_service(
        "HuggingFace Hub",
        dataset_path,
        hf_config,
        check_credentials,
        import_uploader,
        get_uploader_args,
        get_push_args,
        tui,
    )


def _upload_to_kaggle(dataset_path: str, kaggle_config: dict, tui) -> None:
    """Upload dataset to Kaggle if configured."""

    def check_credentials():
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if not username or not key:
            tui.warning("KAGGLE_USERNAME or KAGGLE_KEY not set. Skipping Kaggle upload.")
            return None
        return (username, key)

    def import_uploader():
        from .kaggle_hub import KaggleUploader  # noqa: PLC0415

        return KaggleUploader

    def get_uploader_args(credentials):
        return credentials  # KaggleUploader takes username, key as tuple

    def get_push_args(config, dataset_path):
        return {
            "dataset_handle": config["handle"],
            "jsonl_file_path": dataset_path,
            "tags": config.get("tags", []),
            "version_notes": config.get("version_notes"),
            "description": config.get("description"),
        }

    _upload_to_service(
        "Kaggle",
        dataset_path,
        kaggle_config,
        check_credentials,
        import_uploader,
        get_uploader_args,
        get_push_args,
        tui,
    )


def _strip_nulls(obj: Any) -> Any:
    """Recursively strip null values from nested dicts and lists.

    HuggingFace Dataset's Arrow schema injects null for missing fields across rows.
    This function removes those nulls for clean JSON output.
    """
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(item) for item in obj]
    return obj


def _save_jsonl_without_nulls(dataset: HFDataset, save_path: str) -> None:
    """Save HF Dataset to JSONL, stripping null values injected by Arrow schema."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        for row in dataset:
            cleaned = _strip_nulls(dict(row))
            f.write(json.dumps(cleaned, separators=(",", ":")) + "\n")


def _save_failed_samples(
    save_path: str,
    failed_samples: list,
    tui,
    use_path_directly: bool = False,
) -> None:
    """Save failed samples to a file.

    Args:
        save_path: Path for failures file. If use_path_directly is False, this is treated as the
                   main dataset path and a timestamped filename is generated alongside it.
        failed_samples: List of failed samples - can be dicts with 'error' and 'raw_content' keys,
                       or plain strings/other types for legacy compatibility
        tui: TUI instance for output
        use_path_directly: If True, use save_path as-is. If False, generate timestamped filename.
    """
    if use_path_directly:
        # Use the provided path directly
        failures_path = save_path
    else:
        # Generate timestamped filename: my-dataset.jsonl -> my-dataset_failures_20231130_143022.jsonl
        base_path = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        failures_path = f"{base_path}_failures_{timestamp}.jsonl"

    try:
        with open(failures_path, "w") as f:
            for idx, failure in enumerate(failed_samples):
                # Structure each failure as a JSON object with metadata
                failure_record = {
                    "index": idx,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
                if isinstance(failure, dict):
                    # New format: dict with 'error' and optionally 'raw_content'
                    failure_record["error"] = failure.get("error", str(failure))
                    if "raw_content" in failure:
                        failure_record["raw_content"] = failure["raw_content"]
                else:
                    # Legacy format: plain string or other type
                    failure_record["error"] = str(failure)
                f.write(json.dumps(failure_record) + "\n")
        tui.warning(f"Failed samples saved to: {failures_path} ({len(failed_samples)} failures)")
    except Exception as e:
        tui.error(f"Could not save failed samples: {str(e)}")


def save_dataset(
    dataset: HFDataset,
    save_path: str,
    config: DeepFabricConfig | None = None,
    engine: DataSetGenerator | None = None,
    failures_path: str | None = None,
) -> None:
    """
    Save dataset to file.

    Args:
        dataset: HuggingFace Dataset object to save
        save_path: Path where to save the dataset
        config: Optional configuration for upload settings
        engine: Optional DataSetGenerator to save failed samples from
        failures_path: Optional explicit path for failures file (overrides auto-generated path)

    Raises:
        ConfigurationError: If saving fails
    """
    tui = get_tui()
    try:
        # Save the dataset as JSONL, stripping null values injected by HF Dataset
        # HuggingFace Dataset's Arrow schema adds null for missing fields across rows,
        # but we want clean output without null values for optional fields
        _save_jsonl_without_nulls(dataset, save_path)
        tui.success(f"Dataset saved to: {save_path}")

        # Save failed samples if engine has any (including flushed to checkpoint)
        if engine:
            all_failures = engine.get_all_failures()
            if all_failures:
                # Use explicit failures_path if provided, otherwise auto-generate from save_path
                actual_failures_path = failures_path or save_path
                _save_failed_samples(
                    actual_failures_path, all_failures, tui, use_path_directly=bool(failures_path)
                )

        # Handle automatic uploads if configured
        if config:
            # HuggingFace upload
            if config.huggingface:
                _upload_to_huggingface(save_path, config.get_huggingface_config(), tui)

            # Kaggle upload
            if config.kaggle:
                _upload_to_kaggle(save_path, config.get_kaggle_config(), tui)

    except Exception as e:
        raise ConfigurationError(f"Error saving dataset: {str(e)}") from e
