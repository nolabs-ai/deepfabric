import contextlib
import json
import math
import os
import re

from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

from rich.align import Align
from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text

from .progress import StreamObserver

if TYPE_CHECKING:
    from .error_codes import ClassifiedError


# Constants
STREAM_BUFFER_DISPLAY_THRESHOLD = 1000  # Show ellipsis if accumulated text exceeds this
STREAM_TEXT_MAX_LENGTH = 8000  # Max characters to display in streaming text
STREAM_PANEL_WIDTH = 64  # Minimum width for the right-hand streaming pane
EVENT_LOG_MAX_LINES = 8
STREAM_RENDER_THROTTLE_S = 0.06
STREAM_FIXED_LINES = 16  # Fixed visible lines for streaming preview (used by all previews)
MIN_PREVIEW_LINES = 4  # Minimum preview lines to enforce
# Vertical space occupied by other UI elements when calculating dynamic preview height.
# Accounts for: footer (3) + status panel (8) + panel borders and margins (~9)
PREVIEW_VERTICAL_OFFSET = 20
# Offset for Graph/Tree TUIs which have simpler layouts: footer (3) + status (8) + borders (2)
TOPIC_PREVIEW_OFFSET = 13
# Truncation limits for event log display
EVENT_TOPIC_MAX_LENGTH = 20  # Max chars for topic names in events
EVENT_ERROR_MAX_LENGTH = 80  # Max chars for error summaries in events
ERROR_MESSAGE_MAX_LENGTH = 200  # Max chars for detailed error messages in simple mode


@dataclass
class TUISettings:
    mode: str = "rich"  # 'rich' or 'simple'
    syntax: bool = True  # enable syntax highlighting in preview
    show_failures: bool = False  # show failure details in real-time


_tui_settings = TUISettings()


def configure_tui(mode: str, show_failures: bool = False) -> None:
    mode = (mode or "rich").lower().strip()
    if mode not in {"rich", "simple"}:
        mode = "rich"
    _tui_settings.mode = mode
    _tui_settings.syntax = mode == "rich"
    _tui_settings.show_failures = show_failures


def get_tui_settings() -> TUISettings:
    return _tui_settings


def get_preview_lines() -> int:
    """Get preview height in lines, overridable by DF_TUI_PREVIEW_LINES env var."""
    try:
        v = int(os.getenv("DF_TUI_PREVIEW_LINES", str(STREAM_FIXED_LINES)))
    except Exception:  # noqa: BLE001
        return STREAM_FIXED_LINES
    else:
        return v if v > MIN_PREVIEW_LINES else STREAM_FIXED_LINES


class DeepFabricTUI:
    """Main TUI controller for DeepFabric operations."""

    def __init__(self, console: Console | None = None):
        """Initialize the TUI with rich console."""
        self.console = console or Console()

    def create_header(self, title: str, subtitle: str = "") -> Panel:
        """Create a styled header panel."""
        content = Text(title, style="bold cyan")
        if subtitle:
            content.append(f"\n{subtitle}", style="dim")

        return Panel(
            content,
            border_style="bright_blue",
            padding=(0, 1),
        )

    def build_stream_panel(
        self, content: RenderableType | str, title: str = "Streaming Preview"
    ) -> Panel:
        """Create a compact right-hand panel showing recent streaming output.

        Accepts any Rich renderable (Text, Syntax, Group, etc.) or a plain string.
        """
        renderable: RenderableType
        renderable = Text(content, style="dim") if isinstance(content, str) else content
        # Wrap in Align to ensure content is top-aligned when panel expands
        aligned = Align(renderable, vertical="top")
        return Panel(aligned, title=title, border_style="dim", padding=(0, 1), expand=True)

    def build_events_panel(self, events: list[str], title: str = "Events") -> Panel:
        """Create a compact events panel for the left column."""
        if not events:
            text = Text("Waiting...", style="dim")
        else:
            # Keep events short; show newest at bottom
            # Colorize based on prefix:
            #   X = red (error)
            #   checkmark = green (success)
            #   T+ = green (tool success)
            #   T- = red (tool failure)
            #   T = cyan (tool execution, fallback)
            text = Text()
            for i, event in enumerate(events[-EVENT_LOG_MAX_LINES:]):
                if i > 0:
                    text.append("\n")
                if event.startswith("X "):
                    text.append("X ", style="bold red")
                    text.append(event[2:])
                elif event.startswith("T+ "):
                    # Tool execution success - green
                    text.append("TOOL: ", style="bold green")
                    text.append(event[3:])
                elif event.startswith("T- "):
                    # Tool execution failure - red
                    text.append("TOOL ", style="bold red")
                    text.append(event[3:])
                elif event.startswith("✓ ") or event.startswith("✔ "):
                    text.append(event[0] + " ", style="bold green")
                    text.append(event[2:])
                elif event.startswith("T "):
                    # Fallback for generic tool events - cyan
                    text.append("T ", style="bold cyan")
                    text.append(event[2:])
                else:
                    text.append(event)
        return Panel(text, title=title, border_style="dim", padding=(0, 1), expand=True)

    def create_footer(self, layout: Layout, title: str = "Run Status") -> Progress:
        """Attach a footer progress panel to the provided root layout.

        Expects the root layout to have children: 'main' and 'footer'.
        Returns a Progress instance so callers can create tasks and update.
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]{task.description}",
                table_column=Column(ratio=1, overflow="ellipsis"),
            ),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        layout["footer"].update(Panel(progress, title=title, border_style="dim", padding=(0, 1)))
        return progress

    def create_stats_table(self, stats: dict[str, Any]) -> Table:
        """Create a statistics table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")

        for key, value in stats.items():
            table.add_row(f"{key}:", str(value))

        return table

    def build_context_panel(
        self,
        *,
        root_topic: str | None,
        topic_model_type: str | None,
        path: list[str] | None,
    ) -> Panel:
        """Create a small context panel for current topic path info."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")

        label_current = "Current Leaf" if topic_model_type == "tree" else "Current Node"

        if root_topic:
            table.add_row("Root Topic:", root_topic)

        if path:
            parent = path[-2] if len(path) > 1 else "-"
            leaf = path[-1]
            table.add_row(f"{label_current}:", leaf)
            table.add_row("Parent:", str(parent))
            table.add_row("Path:", " → ".join(path))
        else:
            table.add_row("Status:", "Waiting for first sample...")

        return Panel(table, title="Context", border_style="dim", padding=(0, 1))

    def success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f"✓ {message}", style="green")

    def warning(self, message: str) -> None:
        """Display a warning message."""
        self.console.print(f"⚠ {message}", style="yellow")

    def error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"✗ {message}", style="red")

    def info(self, message: str) -> None:
        """Display an info message."""
        self.console.print(f"• {message}", style="blue")


class TopicGenerationTUI(StreamObserver):
    """Abstract base for Tree and Graph building TUIs.

    Provides shared initialization, layout setup, streaming, retry handling,
    and event management. Subclasses customize via template methods.

    Subclasses must implement:
    - _get_title() -> str
    - _get_subtitle(model_name) -> str
    - _get_footer_description() -> str
    - _topic_model_type() -> str  ('tree' or 'graph')
    - _status_panel() -> Panel
    """

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.progress = None
        self.overall_task = None
        self.failed_attempts = 0
        self.stream_buffer = deque(maxlen=2000)
        self.live_display: Live | None = None
        self.live_layout: Layout | None = None
        self.events_log = deque(maxlen=EVENT_LOG_MAX_LINES)
        self.current_topic_path: list[str] | None = None
        self.root_topic: str | None = None
        self.max_depth = 0
        self.current_depth = 0
        self._is_simple = get_tui_settings().mode == "simple"
        self.simple_progress: Progress | None = None
        self.simple_task = None

    # ---- Template methods for subclass customization ----

    @abstractmethod
    def _get_title(self) -> str:
        """Return header title for this TUI."""
        ...

    @abstractmethod
    def _get_subtitle(self, model_name: str) -> str:
        """Return header subtitle for this TUI."""
        ...

    @abstractmethod
    def _get_footer_description(self) -> str:
        """Return footer progress description."""
        ...

    @abstractmethod
    def _topic_model_type(self) -> str:
        """Return 'tree' or 'graph' for context panel."""
        ...

    @abstractmethod
    def _status_panel(self) -> Panel:
        """Create status panel - must be implemented by subclass."""
        ...

    @abstractmethod
    def _get_simple_total(self, depth: int, degree: int) -> int:
        """Return the total for the simple mode progress bar."""
        ...

    # ---- Lifecycle ----

    def start_building(self, model_name: str, depth: int, degree: int, root_topic: str) -> None:
        """Start the building process. Handles both simple and rich modes."""
        self.max_depth = depth
        self.root_topic = root_topic

        header_panel = self.tui.create_header(
            self._get_title(),
            self._get_subtitle(model_name),
        )

        # Simple/headless mode: print config summary, optional progress bar, no Live
        if self._is_simple:
            self.console.print("\n[bold cyan]Topic Generation[/bold cyan]")
            self.tui.info(f"Model: {model_name}")
            self.tui.info(f"Topic configuration: depth={depth}, degree={degree}")
            self.console.print()
            total = self._get_simple_total(depth, degree)
            if self.console.is_terminal:
                self.simple_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(table_column=Column(justify="right")),
                    TimeElapsedColumn(),
                    console=self.console,
                )
                self.simple_task = self.simple_progress.add_task(
                    self._get_footer_description(), total=total
                )
                self.simple_progress.start()
            return

        # Rich mode: build full two-pane layout with footer
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]{task.description}",
                table_column=Column(ratio=1, overflow="ellipsis"),
            ),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
            console=self.console,
        )

        layout = Layout(name="root")
        layout.split(Layout(name="main"), Layout(name="footer", size=3))
        left = Layout(name="left", ratio=3)
        right = Layout(name="right", ratio=2)
        right.minimum_size = STREAM_PANEL_WIDTH

        stats = {"Model": model_name, "Depth": f"{depth}", "Degree": f"{degree}"}
        stats_table = self.tui.create_stats_table(stats)
        params_panel = Panel(stats_table, title="Generation Parameters", border_style="dim")

        left.split(
            Layout(name="header", size=4),
            Layout(name="params", size=5),
            Layout(name="context", size=5),
            Layout(name="events"),
        )
        left["header"].update(header_panel)
        left["params"].update(params_panel)
        left["context"].update(self._context_panel())
        left["events"].update(self.tui.build_events_panel(list(self.events_log)))

        right.split(
            Layout(name="status", size=8),
            Layout(name="preview"),
        )
        layout["main"].split_row(left, right)
        right["status"].update(self._status_panel())
        right["preview"].update(self.tui.build_stream_panel("Waiting for generation..."))

        self.live_layout = layout
        footer_desc = self._get_footer_description()
        self.footer_progress = self.tui.create_footer(layout, title="Run Status")
        self.footer_task = self.footer_progress.add_task(footer_desc, total=depth)

        self.live_display = Live(layout, console=self.console, refresh_per_second=15, screen=True)
        self.live_display.start()
        self.overall_task = self.progress.add_task(f"{footer_desc} (depth 1/{depth})")

    def stop_live(self) -> None:
        """Stop the Live display if it's running."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

    def advance_simple_progress(self, advance: int = 1, description: str = "") -> None:
        """Advance the simple mode progress bar."""
        if self.simple_progress is not None and self.simple_task is not None:
            with contextlib.suppress(Exception):
                if description:
                    self.simple_progress.update(
                        self.simple_task, advance=advance, description=description
                    )
                else:
                    self.simple_progress.update(self.simple_task, advance=advance)

    def stop_simple_progress(self) -> None:
        """Stop the simple mode progress bar."""
        if self.simple_progress is not None:
            self.simple_progress.stop()
            self.simple_progress = None

    def _simple_print(self, message: str) -> None:
        """Print a message in simple mode, routing through the progress bar if active."""
        if self.simple_progress is not None:
            self.simple_progress.console.print(message)
        else:
            self.console.print(message)

    # ---- Panel refresh helpers ----

    def _refresh_left(self) -> None:
        """Update events panel in left column."""
        if self.live_layout is not None:
            try:
                self.live_layout["main"]["left"]["events"].update(
                    self.tui.build_events_panel(list(self.events_log))
                )
            except Exception:
                return

    def update_status_panel(self) -> None:
        """Update the status panel in the right column."""
        if self.live_layout is None:
            return
        try:
            self.live_layout["main"]["right"]["status"].update(self._status_panel())
        except Exception:
            return

    def _context_panel(self) -> Panel:
        return self.tui.build_context_panel(
            root_topic=self.root_topic,
            topic_model_type=self._topic_model_type(),
            path=self.current_topic_path,
        )

    def _refresh_context(self) -> None:
        if self.live_layout is not None:
            try:
                self.live_layout["main"]["left"]["context"].update(self._context_panel())
            except Exception:
                return

    # ---- StreamObserver event handlers ----

    def on_error(self, error: "ClassifiedError", metadata: dict[str, Any]) -> None:  # noqa: ARG002
        """Handle error events - log to events panel."""
        error_event = error.to_event()
        self.events_log.append(f"X {error_event}")
        self._refresh_left()

        # In simple mode with --show-failures, print detailed error immediately
        if self._is_simple and get_tui_settings().show_failures:
            self.tui.console.print(f"[red]✗ FAILURE:[/red] {error_event}")
            if error.message and error.message != error_event:
                msg = (
                    error.message[:ERROR_MESSAGE_MAX_LENGTH] + "..."
                    if len(error.message) > ERROR_MESSAGE_MAX_LENGTH
                    else error.message
                )
                self.tui.console.print(f"  [dim]{msg}[/dim]")

    def on_step_start(self, step_name: str, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        """Handle step start - topic building doesn't need specific handling."""
        pass

    def on_step_complete(self, step_name: str, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        """Handle step complete - topic building doesn't need specific handling."""
        pass

    def on_stream_chunk(self, _source: str, chunk: str, _metadata: dict[str, Any]) -> None:
        """Handle streaming text from topic generation."""
        self.stream_buffer.append(chunk)
        if self.live_display and self.live_layout is not None:
            accumulated_text = "".join(self.stream_buffer)
            # Trim to last N chars for performance
            if len(accumulated_text) > STREAM_TEXT_MAX_LENGTH:
                accumulated_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]
            # Clean CRs, normalize spaces but preserve newlines
            display_text = accumulated_text.replace("\r", "")
            display_text = re.sub(r"[^\S\n]+", " ", display_text)

            # Compute dynamic preview lines based on terminal height
            # Use TOPIC_PREVIEW_OFFSET for tree/graph TUIs (simpler layout)
            terminal_height = self.console.size.height
            target_lines = max(MIN_PREVIEW_LINES, terminal_height - TOPIC_PREVIEW_OFFSET)
            lines = display_text.splitlines()

            # Handle low-newline content (like JSON) to fill panel properly
            if len(lines) >= int(target_lines / 2):
                # Plenty of newlines: take the last N lines
                visible_lines = lines[-target_lines:]
            else:
                # Low-newline content: take a character tail and then split
                approx_right_cols = max(40, int(self.console.size.width * 0.42))
                char_tail = max(800, approx_right_cols * max(8, target_lines - 2))
                tail = display_text[-char_tail:]
                visible_lines = tail.splitlines()[-target_lines:]

            visible = "\n".join(visible_lines)

            # Update right-hand panel (nested preview)
            try:
                container = self.live_layout["main"]["right"]["preview"]
            except Exception:
                container = self.live_layout["main"]["right"]
            container.update(self.tui.build_stream_panel(visible))

    def on_retry(
        self,
        sample_idx: int,
        attempt: int,
        max_attempts: int,
        error_summary: str,
        metadata: dict[str, Any],
    ) -> None:
        """Handle retry events by logging a concise message."""
        _ = metadata  # Unused for now
        try:
            self.events_log.append(
                f"↻ Retry sample {sample_idx} attempt {attempt}/{max_attempts}: {error_summary}"
            )
            self._refresh_left()
        except Exception:
            # Swallow errors to avoid breaking progress reporting
            return

    def on_llm_retry(
        self,
        provider: str,
        attempt: int,
        wait: float,
        error_summary: str,
        metadata: dict[str, Any],
    ) -> None:
        """Handle LLM API retry events (rate limits, transient errors)."""
        _ = metadata
        try:
            short_msg = f"↻ {provider} retry (attempt {attempt}), backoff {wait:.1f}s"
            self.events_log.append(short_msg)
            self._refresh_left()

            if self._is_simple:
                self._simple_print(f"    [yellow]{short_msg}: {error_summary}[/yellow]")
        except Exception:
            return


class TreeBuildingTUI(TopicGenerationTUI):
    """TUI for tree building operations."""

    def __init__(self, tui: DeepFabricTUI):
        super().__init__(tui)
        self.generated_paths = 0

    def _get_title(self) -> str:
        return "DeepFabric Tree Generation"

    def _get_subtitle(self, model_name: str) -> str:
        return f"Building hierarchical topic structure with {model_name}"

    def _get_footer_description(self) -> str:
        return "Building topic tree"

    def _topic_model_type(self) -> str:
        return "tree"

    def _get_simple_total(self, depth: int, degree: int) -> int:
        if degree <= 1:
            return depth
        return (degree**depth - 1) // (degree - 1)

    def start_depth_level(self, depth: int) -> None:
        """Update progress for new depth level."""
        self.current_depth = depth
        if self.progress and self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                description=f"Building topic tree (depth {depth}/{self.max_depth})",
            )
        self.events_log.append(f"→ Depth {depth}/{self.max_depth} started")
        self._refresh_left()
        self.update_status_panel()

    def start_subtree_generation(self, node_path: list[str], _num_subtopics: int) -> None:
        """Log subtree generation without updating progress to avoid flicker."""
        self.current_topic_path = node_path
        self._refresh_context()

    def complete_subtree_generation(self, success: bool, generated_count: int) -> None:
        """Track completion without updating progress bar."""
        if success:
            self.generated_paths += generated_count
        else:
            self.failed_attempts += 1
        # Log succinct outcome
        status = "ok" if success else "fail"
        self.events_log.append(f"✓ Subtree {status} (+{generated_count} paths)")
        self._refresh_left()
        self.update_status_panel()
        # Advance footer on completed depth
        with contextlib.suppress(Exception):
            self.footer_progress.update(self.footer_task, advance=1)

    def add_failure(self) -> None:
        """Record a generation failure."""
        self.failed_attempts += 1
        self.advance_simple_progress()
        self.events_log.append("✗ Generation failed")
        self._refresh_left()
        self.update_status_panel()

    def finish_building(self, total_paths: int, failed_generations: int) -> None:
        """Finish the tree building process."""
        if self.live_display:
            self.live_display.stop()
        self.stop_simple_progress()

        # Final summary
        self.console.print()
        if failed_generations > 0:
            self.tui.warning(f"Created {total_paths} unique topics ({failed_generations} failed)")
        else:
            self.tui.success(f"Created {total_paths} unique topics")
        self.events_log.append("✓ Tree building completed")
        self.update_status_panel()

    # ---- Status panel for Tree ----
    def _status_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Depth:", f"{self.current_depth}/{self.max_depth}")
        table.add_row("Nodes:", str(self.generated_paths))
        if self.failed_attempts:
            table.add_row("Failed:", str(self.failed_attempts))
        return Panel(table, title="Status", border_style="dim", padding=(0, 1))


class GraphBuildingTUI(TopicGenerationTUI):
    """TUI for graph building operations."""

    def __init__(self, tui: DeepFabricTUI):
        super().__init__(tui)
        self.nodes_count = 1  # Start with root
        self.edges_count = 0

    def _get_title(self) -> str:
        return "DeepFabric Graph Generation"

    def _get_subtitle(self, model_name: str) -> str:
        return f"Building interconnected topic structure with {model_name}"

    def _get_footer_description(self) -> str:
        return "Building topic graph"

    def _topic_model_type(self) -> str:
        return "graph"

    def _get_simple_total(self, depth: int, degree: int) -> int:  # noqa: ARG002
        return depth

    def start_depth_level(self, depth: int, leaf_count: int) -> None:
        """Update for new depth level."""
        if self._is_simple:
            desc = f"Depth {depth}/{self.max_depth} ({leaf_count} nodes)"
            if self.simple_progress is not None:
                self.advance_simple_progress(advance=0, description=desc)
            else:
                self.console.print(f"  Depth {depth}: expanding {leaf_count} nodes...")
        elif self.progress and self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                description=f"  Building graph - depth {depth} ({leaf_count} nodes to expand)",
            )
        self.events_log.append(f"→ Depth {depth} start ({leaf_count} nodes)")
        self._refresh_left()
        self.current_depth = depth
        self.update_status_panel()

    def complete_node_expansion(
        self, node_topic: str, subtopics_added: int, connections_added: int
    ) -> None:
        """Track node expansion."""
        _ = node_topic  # Mark as intentionally unused
        self.nodes_count += subtopics_added
        self.edges_count += subtopics_added + connections_added

    def complete_depth_level(self, depth: int) -> None:
        """Complete a depth level."""
        if self._is_simple:
            if self.simple_progress is not None:
                self.advance_simple_progress()
            else:
                self.console.print(
                    f"  Depth {depth} complete (nodes: {self.nodes_count}, edges: {self.edges_count})"
                )
        elif self.progress and self.overall_task is not None:
            self.progress.advance(self.overall_task, 1)
        self.events_log.append(f"✓ Depth {depth} complete")
        self._refresh_left()
        self.update_status_panel()
        # Advance footer on depth complete
        with contextlib.suppress(Exception):
            self.footer_progress.update(self.footer_task, advance=1)

    def add_failure(self, node_topic: str) -> None:
        """Record a generation failure."""
        self.failed_attempts += 1
        if self._is_simple:
            if len(node_topic) > EVENT_ERROR_MAX_LENGTH:
                topic_display = node_topic[:EVENT_ERROR_MAX_LENGTH] + "..."
            else:
                topic_display = node_topic
            self._simple_print(f"    [red]✗ Node expansion failed: {topic_display}[/red]")
        self.events_log.append("✗ Node expansion failed")
        self._refresh_left()

    def on_node_retry(
        self,
        node_topic: str,
        attempt: int,
        max_attempts: int,
        error_summary: str,
        metadata: dict[str, Any],
    ) -> None:
        """Handle node expansion retry events from the progress reporter.

        Args:
            node_topic: Topic of the node being expanded
            attempt: Current attempt number (1-based)
            max_attempts: Total number of attempts allowed
            error_summary: Brief description of the error
            metadata: Additional context
        """
        _ = metadata  # Unused for now
        try:
            # Truncate node topic for display
            if len(node_topic) > EVENT_TOPIC_MAX_LENGTH:
                topic_display = node_topic[:EVENT_TOPIC_MAX_LENGTH] + "..."
            else:
                topic_display = node_topic
            # Truncate error summary
            if len(error_summary) > EVENT_ERROR_MAX_LENGTH:
                error_display = error_summary[:EVENT_ERROR_MAX_LENGTH] + "..."
            else:
                error_display = error_summary

            # In simple mode, print through progress-aware helper
            if self._is_simple:
                self._simple_print(
                    f"    [yellow]↻ Retry {attempt}/{max_attempts} '{topic_display}': {error_display}[/yellow]"
                )

            self.events_log.append(
                f"↻ Retry {attempt}/{max_attempts} '{topic_display}': {error_display}"
            )
            self._refresh_left()
        except Exception:
            # Best-effort, swallow errors to avoid breaking progress reporting
            return

    def finish_building(self, failed_generations: int) -> None:
        """Finish the graph building process."""
        if self.live_display:
            self.live_display.stop()
        self.stop_simple_progress()

        self.console.print()
        if self._is_simple:
            # One-liner summary for simple/headless mode
            if failed_generations > 0:
                self.tui.warning(
                    f"Created {self.nodes_count} unique topics ({failed_generations} failed)"
                )
            else:
                self.tui.success(f"Created {self.nodes_count} unique topics")
        else:
            # Rich mode: show detailed stats panel
            stats_table = self.tui.create_stats_table(
                {
                    "Total Nodes": self.nodes_count,
                    "Total Edges": self.edges_count,
                    "Failed Attempts": self.failed_attempts,
                }
            )
            self.console.print(Panel(stats_table, title="Final Statistics", border_style="dim"))
            if failed_generations > 0:
                self.tui.warning(f"Graph building complete with {failed_generations} failures")
            else:
                self.tui.success("Graph building completed successfully")
        self.events_log.append("✓ Graph building completed")
        self.update_status_panel()

    # ---- Status panel for Graph ----
    def _status_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Depth:", f"{self.current_depth}/{self.max_depth}")
        table.add_row("Nodes:", str(self.nodes_count))
        table.add_row("Edges:", str(self.edges_count))
        if self.failed_attempts:
            table.add_row("Failed:", str(self.failed_attempts))
        return Panel(table, title="Status", border_style="dim", padding=(0, 1))


class DatasetGenerationTUI(StreamObserver):
    """Enhanced TUI for dataset generation with rich integration and streaming display."""

    live_display: Live | None
    live_layout: Layout | None

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.stream_buffer = deque(maxlen=2000)  # Last ~2000 chars of streaming text
        self.current_step = ""
        self.current_sample_type = ""  # Track the type of sample being generated
        self.live_display = None  # Will be set by dataset_manager
        self.live_layout = None  # Provided by dataset_manager
        self.progress = None
        self.stream_text = Text("Waiting for generation...", style="dim")  # Streaming content
        self.events_log = deque(maxlen=EVENT_LOG_MAX_LINES)
        # Context tracking
        self.root_topic_prompt: str | None = None
        self.topic_model_type: str | None = None  # 'tree' or 'graph'
        self.current_topic_path: list[str] | None = None
        self._last_render_t = 0.0
        self._last_visible_key = ""
        # Status tracking
        self.status_total_steps = 0
        self.status_current_step = 0
        self.status_total_samples = 0
        self.status_samples_done = 0
        self.status_failed_total = 0
        self.status_step_started_at = 0.0
        self.status_last_step_duration = 0.0
        # Checkpoint tracking for status panel
        self.checkpoint_enabled = False  # Set to True when checkpointing is configured
        self.checkpoint_count = 0
        self.last_checkpoint_samples = 0
        self._resumed_from_checkpoint = False  # Set by set_checkpoint_resume_status()
        self._stop_requested = False  # Set when graceful stop requested via Ctrl+C
        self._is_topic_based = False  # Set by init_status; controls topic-group versus step labels
        self._is_simple = get_tui_settings().mode != "rich"
        self.simple_progress: Progress | None = None  # Set by dataset_manager for simple mode
        # Retry tracking for simple mode
        self.step_retries: list[dict] = []  # Retries in current step

    def create_rich_progress(self) -> Progress:
        """Create a rich progress bar for dataset generation (without TimeRemainingColumn)."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]{task.description}",
                table_column=Column(ratio=1, overflow="ellipsis"),
            ),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        return self.progress

    def build_generation_panels(
        self,
        model_name: str,
        num_steps: int,
        batch_size: int,
        total_samples: int | None = None,
        is_topic_based: bool = False,
        unique_topics: int = 0,
        selected_topics: int = 0,
        base_quota: int = 0,
        remainder: int = 0,
        checkpoint_interval: int = 0,
    ) -> tuple[Panel, Panel]:
        """Show exact targets and either topic allocation or sequential sample batches."""
        header = self.tui.create_header(
            "DeepFabric Dataset Generation",
            f"Creating synthetic traces with {model_name}",
        )

        display_total = total_samples if total_samples is not None else num_steps * batch_size

        lines = [f"[cyan]Model:[/] {model_name}"]

        if is_topic_based:
            lines.append(
                f"[cyan]Number samples:[/] {display_total}, [cyan]Concurrency:[/] {batch_size}"
            )
            lines.append(
                f"[cyan]One pass:[/] {selected_topics}/{unique_topics} topics; "
                f"base quota {base_quota}, {remainder} topics receive one extra"
            )
            log_msg = f"Start • topic concurrency={batch_size} total={display_total}"
        else:
            lines.append(
                f"[cyan]Number samples:[/] {display_total}, [cyan]Batch size:[/] {batch_size}"
            )
            log_msg = f"Start • steps={num_steps} batch={batch_size} total={display_total}"

        if checkpoint_interval and checkpoint_interval > 0:
            total_cp = math.ceil(display_total / checkpoint_interval)
            lines.append(
                f"[cyan]Checkpoint:[/] every {checkpoint_interval} samples "
                f"({total_cp} total checkpoints)"
            )

        params_panel = Panel(
            Text.from_markup("\n".join(lines)),
            title="Generation Parameters",
            border_style="dim",
        )

        self.events_log.append(log_msg)
        return header, params_panel

    def on_stream_chunk(self, _source: str, chunk: str, _metadata: dict[str, Any]) -> None:
        """Handle incoming streaming text chunks from LLM.

        Args:
            source: Source identifier (e.g., "user_question", "tool_sim_weather")
            chunk: Text chunk from LLM
            metadata: Additional context
        """
        # Append chunk to buffer (deque auto-trims to maxlen)
        self.stream_buffer.append(chunk)

        # Update the live display if it's running
        if self.live_display and self.live_layout is not None:
            self.update_stream_panel()

    def on_step_start(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Update current step display.

        Args:
            step_name: Human-readable step name
            metadata: Additional context (sample_idx, conversation_type, etc.)
        """
        # Update current step
        self.current_step = step_name

        # Extract and update sample type from metadata if available
        if "conversation_type" in metadata:
            conv_type = metadata["conversation_type"]
            # Map conversation types to friendly names
            type_map = {
                "basic": "Basic Q&A",
                "cot": "Chain of Thought",
            }
            self.current_sample_type = type_map.get(conv_type, conv_type)

        # Update current topic path if provided
        topic_path = metadata.get("topic_path") if isinstance(metadata, dict) else None
        if topic_path:
            # Ensure list[str]
            try:
                self.current_topic_path = list(topic_path)
            except Exception:  # noqa: BLE001
                self.current_topic_path = None
            self.update_context_panel()

        # Don't print anything - the progress bar already shows progress
        # Just silently update internal state

    def on_step_complete(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Handle step completion.

        Args:
            step_name: Human-readable step name
            metadata: Additional context
        """
        # Could add completion markers or timing info here if desired
        pass

    def on_tool_execution(self, tool_name: str, success: bool, metadata: dict[str, Any]) -> None:
        """Handle tool execution events from Spin.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            metadata: Additional context (arguments, error_type, etc.)
        """
        # Format arguments preview
        args = metadata.get("arguments", {})
        args_preview = self._format_args_preview(args, max_len=20)

        # Use prefix patterns that build_events_panel recognizes for coloring
        # T+ = green (success), T- = red (failure)
        if success:
            self.log_event(f"T+ {tool_name}({args_preview})")
        else:
            error_type = metadata.get("error_type", "error")
            self.log_event(f"T- {tool_name}({args_preview}) -> {error_type}")

    def _format_args_preview(self, args: dict[str, Any], max_len: int = 20) -> str:
        """Format tool arguments as truncated JSON preview.

        Args:
            args: Tool arguments dictionary
            max_len: Maximum length before truncation

        Returns:
            Truncated JSON string with ellipsis if needed
        """
        if not args:
            return ""
        try:
            json_str = json.dumps(args, separators=(",", ":"))
            if len(json_str) <= max_len:
                return json_str
            return json_str[:max_len] + "..."
        except Exception:  # noqa: BLE001
            return "..."

    def get_stream_display(self) -> str:
        """Build the streaming text display from buffer.

        Returns:
            Formatted string of recent LLM output
        """
        if not self.stream_buffer:
            return "[dim italic]Waiting for generation...[/dim italic]"

        # Get recent text from buffer
        recent_text = "".join(self.stream_buffer)

        # Truncate if too long and add ellipsis
        max_display_length = 300
        if len(recent_text) > max_display_length:
            recent_text = "..." + recent_text[-max_display_length:]

        return f"[dim]{recent_text}[/dim]"

    def clear_stream_buffer(self) -> None:
        """Clear the streaming text buffer (e.g., between samples)."""
        self.stream_buffer.clear()

    # Deprecated printer retained for backward compatibility
    def show_generation_header(
        self,
        model_name: str,
        num_steps: int,
        batch_size: int,
        total_samples: int | None = None,
        is_topic_based: bool = False,
    ) -> None:
        header, params_panel = self.build_generation_panels(
            model_name, num_steps, batch_size, total_samples, is_topic_based
        )
        self.console.print(header)
        self.console.print(params_panel)
        self.console.print()

    def _context_panel(self) -> Panel:
        return self.tui.build_context_panel(
            root_topic=self.root_topic_prompt,
            topic_model_type=self.topic_model_type,
            path=self.current_topic_path,
        )

    def update_context_panel(self) -> None:
        if self.live_layout is None:
            return
        try:
            self.live_layout["main"]["left"]["context"].update(self._context_panel())
        except Exception:
            return

    # --- Status Panel helpers ---
    def init_status(
        self,
        total_steps: int,
        total_samples: int,
        checkpoint_enabled: bool = False,
        is_topic_based: bool = False,
    ) -> None:
        self.status_total_steps = total_steps
        self.status_total_samples = total_samples
        self.status_current_step = 0
        self._is_topic_based = is_topic_based
        # Preserve samples_done and failed_total if resuming from checkpoint
        if not getattr(self, "_resumed_from_checkpoint", False):
            self.status_samples_done = 0
            self.status_failed_total = 0
            self.checkpoint_count = 0
            self.last_checkpoint_samples = 0
        self.status_step_started_at = 0.0
        self.status_last_step_duration = 0.0
        self.checkpoint_enabled = checkpoint_enabled

    def status_step_start(self, step: int, total_steps: int | None = None) -> None:
        self.status_current_step = step
        if total_steps is not None:
            self.status_total_steps = total_steps
        self.status_step_started_at = monotonic()
        self.update_status_panel()

    def status_step_complete(self, samples_generated: int, failed_in_step: int = 0) -> None:
        # Calculate step duration before updating counters
        if self.status_step_started_at:
            self.status_last_step_duration = max(0.0, monotonic() - self.status_step_started_at)
            self.status_step_started_at = 0.0  # Reset for next step
        self.status_samples_done += max(0, int(samples_generated))
        self.status_failed_total += max(0, int(failed_in_step))
        self.update_status_panel()

    def set_checkpoint_resume_status(
        self, samples_done: int, failed_total: int, checkpoint_count: int = 0
    ) -> None:
        """Initialize status counters from checkpoint data when resuming.

        Args:
            samples_done: Number of samples already generated in checkpoint
            failed_total: Number of failures already recorded in checkpoint
            checkpoint_count: Number of checkpoints already saved (optional)
        """
        self._resumed_from_checkpoint = True
        self.status_samples_done = max(0, int(samples_done))
        self.status_failed_total = max(0, int(failed_total))
        if checkpoint_count > 0:
            self.checkpoint_count = checkpoint_count
            self.last_checkpoint_samples = samples_done
        self.update_status_panel()

    def status_checkpoint_saved(self, total_samples: int) -> None:
        """Update checkpoint tracking when a checkpoint is saved."""
        self.checkpoint_count += 1
        self.last_checkpoint_samples = total_samples
        self.update_status_panel()

    def status_stop_requested(self) -> None:
        """Mark that a graceful stop has been requested."""
        self._stop_requested = True
        self.update_status_panel()

    def _status_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        label = "Topic group:" if self._is_topic_based else "Step:"
        last_label = "Last group:" if self._is_topic_based else "Last Step:"
        table.add_row(label, f"{self.status_current_step}/{self.status_total_steps}")
        if self.status_last_step_duration > 0:
            table.add_row(last_label, f"{self.status_last_step_duration:0.1f}s")
        table.add_row("Generated:", f"{self.status_samples_done}/{self.status_total_samples}")
        if self.status_failed_total:
            table.add_row("Failed:", str(self.status_failed_total))
        if self.checkpoint_enabled:
            if self.checkpoint_count > 0:
                table.add_row(
                    "Checkpoints:",
                    f"{self.checkpoint_count} ({self.last_checkpoint_samples} samples)",
                )
            else:
                table.add_row("Checkpoints:", "0 (enabled)")
        if self._stop_requested:
            if self.checkpoint_enabled:
                table.add_row("[yellow]Stopping:[/yellow]", "[yellow]at next checkpoint[/yellow]")
            else:
                table.add_row(
                    "[yellow]Stopping:[/yellow]", "[yellow]saving partial results[/yellow]"
                )
        return Panel(table, title="Status", border_style="dim", padding=(0, 1))

    def update_status_panel(self) -> None:
        if self.live_layout is None:
            return
        try:
            self.live_layout["main"]["right"]["status"].update(self._status_panel())
        except Exception:
            return

    def success(self, message: str) -> None:
        """Display a success message."""
        self.tui.success(message)

    def warning(self, message: str) -> None:
        """Display a warning message."""
        self.tui.warning(message)

    def error(self, message: str) -> None:
        """Display an error message."""
        self.tui.error(message)

    def info(self, message: str) -> None:
        """Display an info message."""
        self.tui.info(message)

    # --- Compact two-pane helpers ---
    def update_stream_panel(self) -> None:
        """Refresh the right-hand streaming panel with current buffer text."""
        if self.live_layout is None:
            return

        # Throttle: avoid re-rendering too frequently
        now = monotonic()
        if now - getattr(self, "_last_render_t", 0.0) < STREAM_RENDER_THROTTLE_S:  # noqa: PLR2004
            return

        # Build multi-line text; show the last N lines based on terminal height
        accumulated_text = "".join(self.stream_buffer)
        if len(accumulated_text) > STREAM_TEXT_MAX_LENGTH:
            accumulated_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]

        normalized = accumulated_text.replace("\r", "")
        normalized = re.sub(r"[^\S\n]+", " ", normalized)

        # Calculate target lines based on terminal height
        terminal_height = self.console.size.height
        target_lines = max(MIN_PREVIEW_LINES, terminal_height - PREVIEW_VERTICAL_OFFSET)
        lines = normalized.splitlines()
        if len(lines) >= int(target_lines / 2):
            # Plenty of newlines: take the last N lines
            visible_lines = lines[-target_lines:]
        else:
            # Low-newline content: take a character tail and then split
            approx_right_cols = max(40, int(self.console.size.width * 0.42))
            char_tail = max(800, approx_right_cols * max(8, target_lines - 2))
            tail = normalized[-char_tail:]
            visible_lines = tail.splitlines()[-target_lines:]

        # Flexible-height layout handles stability; render just the last N lines
        visible = "\n".join(visible_lines)

        # Skip update if content suffix hasn't changed
        key = visible[-200:]
        if key == getattr(self, "_last_visible_key", ""):
            return
        self._last_visible_key = key

        # Build simple dim text renderable (syntax highlighting removed)
        renderable = Text(visible, style="dim")

        title = (
            f"Streaming Preview • {self.current_sample_type}"
            if self.current_sample_type
            else "Streaming Preview"
        )
        # Support both old layout (right only) and new split layout (right.preview)
        try:
            container = self.live_layout["main"]["right"]["preview"]
        except Exception:
            container = self.live_layout["main"]["right"]
        container.update(self.tui.build_stream_panel(renderable, title=title))
        self._last_render_t = now

    def log_event(self, message: str) -> None:
        """Append an event to the left-side event log and refresh."""
        self.events_log.append(message)
        if self.live_layout is not None:
            self.live_layout["main"]["left"]["events"].update(
                self.tui.build_events_panel(list(self.events_log))
            )

    def on_error(self, error: "ClassifiedError", metadata: dict[str, Any]) -> None:
        """Handle error events from the progress reporter.

        Displays concise error information in the Events panel using
        standardized DeepFabric error codes.

        Args:
            error: ClassifiedError with error code and details
            metadata: Additional context (sample_idx, etc.)
        """
        # Format concise error message for Events panel
        error_event = error.to_event()

        # Add sample context if available
        sample_idx = metadata.get("sample_idx")
        if sample_idx is not None:
            error_event = f"[{sample_idx}] {error_event}"

        # Log to events panel with error indicator
        self.log_event(f"X {error_event}")

        # In simple mode with --show-failures, print detailed error immediately
        if self._is_simple and get_tui_settings().show_failures:
            self.tui.console.print(f"[red]✗ FAILURE:[/red] {error_event}")
            if error.message and error.message != error_event:
                # Show truncated full message if different from event
                msg = (
                    error.message[:ERROR_MESSAGE_MAX_LENGTH] + "..."
                    if len(error.message) > ERROR_MESSAGE_MAX_LENGTH
                    else error.message
                )
                self.tui.console.print(f"  [dim]{msg}[/dim]")

    def on_retry(
        self,
        sample_idx: int,
        attempt: int,
        max_attempts: int,
        error_summary: str,
        metadata: dict[str, Any],
    ) -> None:
        """Handle retry events from the progress reporter.

        In rich mode, we don't log individual retries to avoid cluttering the
        events panel - the streaming preview shows activity and final errors
        are logged via on_error.

        In simple mode, tracks retries for display at step completion.

        Args:
            sample_idx: 1-based sample index
            attempt: Current attempt number (1-based)
            max_attempts: Total number of attempts allowed
            error_summary: Brief description of the validation error
            metadata: Additional context
        """
        _ = metadata  # Unused for now

        if self._is_simple:
            # Simple mode: track for summary at step completion
            self.step_retries.append(
                {
                    "sample_idx": sample_idx,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error_summary": error_summary,
                }
            )

    def clear_step_retries(self) -> None:
        """Clear retry tracking for the current step."""
        self.step_retries.clear()

    def get_step_retry_summary(self) -> str | None:
        """Get a summary of retries in the current step.

        Returns:
            Summary string or None if no retries occurred
        """
        if not self.step_retries:
            return None

        # Count unique samples that had retries
        samples_with_retries = {r["sample_idx"] for r in self.step_retries}
        total_retries = len(self.step_retries)

        if len(samples_with_retries) == 1:
            return f"{total_retries} retry for sample {list(samples_with_retries)[0]}"
        return f"{total_retries} retries across {len(samples_with_retries)} samples"

    def on_llm_retry(
        self,
        provider: str,
        attempt: int,
        wait: float,
        error_summary: str,
        metadata: dict[str, Any],
    ) -> None:
        """Handle LLM API retry events (rate limits, transient errors)."""
        _ = metadata
        try:
            short_msg = f"↻ {provider} retry (attempt {attempt}), backoff {wait:.1f}s"
            self.events_log.append(short_msg)
            if self.live_layout is not None:
                self.live_layout["main"]["left"]["events"].update(
                    self.tui.build_events_panel(list(self.events_log))
                )

            if self._is_simple:
                if self.simple_progress is not None:
                    self.simple_progress.console.print(
                        f"    [yellow]{short_msg}: {error_summary}[/yellow]"
                    )
                else:
                    self.console.print(f"    [yellow]{short_msg}: {error_summary}[/yellow]")
        except Exception:
            return


# Global TUI instances
_tui_instance = None
_dataset_tui_instance = None


def get_tui() -> DeepFabricTUI:
    """Get the global TUI instance."""
    global _tui_instance  # noqa: PLW0603
    if _tui_instance is None:
        _tui_instance = DeepFabricTUI()
    return _tui_instance


def get_tree_tui() -> TreeBuildingTUI:
    """Get a tree building TUI instance."""
    return TreeBuildingTUI(get_tui())


def get_graph_tui() -> GraphBuildingTUI:
    """Get a graph building TUI instance."""
    return GraphBuildingTUI(get_tui())


def get_dataset_tui() -> DatasetGenerationTUI:
    """Get the global dataset generation TUI instance (singleton)."""
    global _dataset_tui_instance  # noqa: PLW0603
    if _dataset_tui_instance is None:
        _dataset_tui_instance = DatasetGenerationTUI(get_tui())
    return _dataset_tui_instance
