import asyncio
import json
import traceback

from pathlib import Path
from typing import TYPE_CHECKING

from .config import DeepFabricConfig
from .constants import (
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .exceptions import ConfigurationError
from .factory import create_topic_generator
from .graph import Graph
from .output_safety import save_output
from .topic_model import TopicModel
from .tree import Tree
from .tui import get_graph_tui, get_tree_tui, get_tui
from .utils import read_topic_tree_from_jsonl

if TYPE_CHECKING:
    from .topic_model import TopicModel


def _ensure_not_running_loop(func_name: str) -> None:
    """Raise a helpful error if invoked from an active asyncio loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    if loop.is_running():
        msg = (
            f"{func_name} cannot be called from within an active event loop. "
            f"Use `{func_name}_async` instead."
        )
        raise RuntimeError(msg)


async def _process_graph_events(graph: Graph, debug: bool = False) -> dict | None:
    from .progress import ProgressReporter  # noqa: PLC0415

    tui = get_graph_tui()

    # Create and attach progress reporter for streaming
    progress_reporter = ProgressReporter()
    progress_reporter.attach(tui)
    graph.progress_reporter = progress_reporter
    if hasattr(graph, "llm_client"):
        graph.llm_client.retry_handler.progress_reporter = progress_reporter

    tui_started = False

    final_event = None
    try:
        async for event in graph.build_async():
            if event["event"] == "depth_start":
                if not tui_started:
                    tui.start_building(
                        graph.model_name, graph.depth, graph.degree, graph.topic_prompt
                    )
                    tui_started = True
                depth = int(event["depth"]) if isinstance(event["depth"], str | int) else 0
                leaf_count = (
                    int(event.get("leaf_count", 0))
                    if isinstance(event.get("leaf_count", 0), str | int)
                    else 0
                )
                tui.start_depth_level(depth, leaf_count)
            elif event["event"] == "node_expanded":
                subtopics_added = (
                    int(event["subtopics_added"])
                    if isinstance(event["subtopics_added"], str | int)
                    else 0
                )
                connections_added = (
                    int(event.get("connections_added", 0))
                    if isinstance(event.get("connections_added", 0), str | int)
                    else 0
                )
                tui.complete_node_expansion(event["node_topic"], subtopics_added, connections_added)
            elif event["event"] == "depth_complete":
                depth = int(event["depth"]) if isinstance(event["depth"], str | int) else 0
                tui.complete_depth_level(depth)
            elif event["event"] == "build_complete":
                failed_generations = (
                    int(event.get("failed_generations", 0))
                    if isinstance(event.get("failed_generations", 0), str | int)
                    else 0
                )
                tui.finish_building(failed_generations)
                final_event = event

                if failed_generations > 0 and hasattr(graph, "failed_generations"):
                    truncated = sum(
                        1
                        for f in graph.failed_generations
                        if "EOF while parsing" in f.get("last_error", "")
                    )
                    if truncated:
                        get_tui().warning(
                            f"Hint: {truncated} of {failed_generations} failures appear to be "
                            f"truncated responses. Consider increasing max_tokens "
                            f"(currently {graph.max_tokens})."
                        )

                if debug and failed_generations > 0 and hasattr(graph, "failed_generations"):
                    get_tui().error("\nDebug: Graph generation failures:")
                    for idx, failure in enumerate(graph.failed_generations, 1):
                        node_id = failure.get("node_id", "unknown")
                        attempts = failure.get("attempts", 0)
                        last_error = failure.get("last_error", "Unknown error")
                        get_tui().error(f"  [{idx}] Node ID: {node_id}, Attempts: {attempts}")
                        get_tui().error(f"      Error: {last_error}")
    except Exception as e:
        # Stop TUI before printing error to ensure visibility
        tui.stop_live()
        if debug:
            get_tui().error(f"Debug: Full traceback:\n{traceback.format_exc()}")
        get_tui().error(f"Graph build failed: {str(e)}")
        raise
    else:
        return final_event


async def _process_tree_events(tree: Tree, debug: bool = False) -> dict | None:
    from .progress import ProgressReporter  # noqa: PLC0415

    tui = get_tree_tui()

    # Create and attach progress reporter for streaming
    progress_reporter = ProgressReporter()
    progress_reporter.attach(tui)
    tree.progress_reporter = progress_reporter
    if hasattr(tree, "llm_client"):
        tree.llm_client.retry_handler.progress_reporter = progress_reporter

    final_event = None
    try:
        async for event in tree.build_async():
            if event["event"] == "build_start":
                depth = int(event["depth"]) if isinstance(event["depth"], str | int) else 0
                degree = int(event["degree"]) if isinstance(event["degree"], str | int) else 0
                tui.start_building(event["model_name"], depth, degree, tree.topic_prompt)
            elif event["event"] == "subtopics_generated":
                if not event["success"]:
                    tui.add_failure()
                    if debug and "error" in event:
                        get_tui().error(f"Debug: Tree generation failure - {event['error']}")
                else:
                    tui.advance_simple_progress()
            elif event["event"] == "build_complete":
                total_paths = (
                    int(event["total_paths"]) if isinstance(event["total_paths"], str | int) else 0
                )
                failed_generations = (
                    int(event["failed_generations"])
                    if isinstance(event["failed_generations"], str | int)
                    else 0
                )
                tui.finish_building(total_paths, failed_generations)
                final_event = event

                if failed_generations > 0 and hasattr(tree, "failed_generations"):
                    truncated = sum(
                        1
                        for f in tree.failed_generations
                        if "EOF while parsing" in f.get("error", "")
                    )
                    if truncated:
                        get_tui().warning(
                            f"Hint: {truncated} of {failed_generations} failures appear to be "
                            f"truncated responses. Consider increasing max_tokens "
                            f"(currently {tree.max_tokens})."
                        )

                if debug and failed_generations > 0 and hasattr(tree, "failed_generations"):
                    get_tui().error("\nDebug: Tree generation failures:")
                    for idx, failure in enumerate(tree.failed_generations, 1):
                        get_tui().error(
                            f"  [{idx}] Path: {' -> '.join(failure.get('node_path', []))}"
                        )
                        get_tui().error(f"      Error: {failure.get('error', 'Unknown error')}")
    except Exception as e:
        # Stop TUI before printing error to ensure visibility
        tui.stop_live()
        if debug:
            get_tui().error(f"Debug: Full traceback:\n{traceback.format_exc()}")
        get_tui().error(f"Tree build failed: {str(e)}")
        raise
    else:
        return final_event


def handle_graph_events(graph: Graph, debug: bool = False) -> dict | None:
    """
    Build graph with TUI progress.

    Args:
        graph: Graph object to build
        debug: Enable debug output

    Returns:
        Final build event dictionary or None

    Raises:
        Exception: If graph build fails
    """
    _ensure_not_running_loop("handle_graph_events")
    return asyncio.run(_process_graph_events(graph, debug=debug))


async def handle_graph_events_async(graph: Graph, debug: bool = False) -> dict | None:
    return await _process_graph_events(graph, debug=debug)


def handle_tree_events(tree: Tree, debug: bool = False) -> dict | None:
    """
    Build tree with TUI progress.

    Args:
        tree: Tree object to build
        debug: Enable debug output

    Returns:
        Final build event dictionary or None

    Raises:
        Exception: If tree build fails
    """
    _ensure_not_running_loop("handle_tree_events")
    return asyncio.run(_process_tree_events(tree, debug=debug))


async def handle_tree_events_async(tree: Tree, debug: bool = False) -> dict | None:
    return await _process_tree_events(tree, debug=debug)


def load_or_build_topic_model(
    config: DeepFabricConfig,
    topics_load: str | None = None,
    topics_overrides: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    debug: bool = False,
) -> TopicModel:
    """
    Load topic model from file or build new one.

    Args:
        config: DeepFabricConfig object
        topics_load: Path to existing topics file (JSONL for tree, JSON for graph)
        topics_overrides: Override parameters for topic generation
        provider: LLM provider
        model: Model name
        base_url: Base URL for LLM API
        debug: Enable debug output

    Returns:
        TopicModel (Tree or Graph)

    Raises:
        ConfigurationError: If loading or building fails
    """
    tui = get_tui()

    if topics_load:
        # Config mode takes precedence; file extension is only used to warn on mismatch
        is_graph = config.topics.mode == "graph"

        # Warn if file extension doesn't match the configured mode
        if not is_graph and topics_load.endswith(".json"):
            tui.warning(
                f"File '{topics_load}' has .json extension (typically a graph) "
                f"but mode is '{config.topics.mode}'. "
                "If this is a graph set mode: graph in config."
            )
        elif is_graph and topics_load.endswith(".jsonl"):
            tui.warning(
                f"File '{topics_load}' has .jsonl extension (typically a tree) "
                "but mode is 'graph'. "
                "If this is a tree set mode: tree in config."
            )

        if is_graph:
            tui.info(f"Reading topic graph from JSON file: {topics_load}")
            topics_params = config.get_topics_params(**(topics_overrides or {}))
            return Graph.from_json(topics_load, topics_params)
        tui.info(f"Reading topic tree from JSONL file: {topics_load}")
        dict_list = read_topic_tree_from_jsonl(topics_load)

        final_provider = provider or "ollama"
        final_model = model or "mistral:latest"

        topic_model = Tree(
            topic_prompt="default",
            provider=final_provider,
            model_name=final_model,
            topic_system_prompt="",
            degree=TOPIC_TREE_DEFAULT_DEGREE,
            depth=TOPIC_TREE_DEFAULT_DEPTH,
            temperature=TOPIC_TREE_DEFAULT_TEMPERATURE,
            base_url=base_url,
        )
        topic_model.from_dict_list(dict_list)
        return topic_model

    # Build new topic model
    topic_model = create_topic_generator(config, topics_overrides=topics_overrides)

    # Build with appropriate event handler
    if isinstance(topic_model, Graph):
        handle_graph_events(topic_model, debug=debug)
    elif isinstance(topic_model, Tree):
        handle_tree_events(topic_model, debug=debug)

    return topic_model


def save_topic_model(
    topic_model: TopicModel,
    config: DeepFabricConfig,
    topics_save_as: str | None = None,
    *,
    replace: bool = False,
) -> str:
    """Safely save generated topics and sidecars, returning the actual topic path."""
    tui = get_tui()
    is_tree = isinstance(topic_model, Tree)
    default = "topic_tree.jsonl" if is_tree else "topic_graph.json"
    destination = topics_save_as or config.topics.save_as or default

    def write_topics(path):
        with open(path, "w", encoding="utf-8") as stream:
            if is_tree:
                for topic_path in topic_model.tree_paths:
                    stream.write(json.dumps({"path": topic_path}) + "\n")
            else:
                stream.write(topic_model.to_json())

    try:
        actual = save_output(destination, write_topics, replace=replace)
        tui.success(f"Topic {'tree' if is_tree else 'graph'} saved to {actual}")
        failures = getattr(topic_model, "failed_generations", None)
        if failures:
            sidecar = Path(actual).with_name(Path(actual).stem + "_failed.jsonl")

            def write_failures(path):
                with open(path, "w", encoding="utf-8") as stream:
                    for failure in failures:
                        stream.write(json.dumps({"failed_generation": failure}) + "\n")

            failed_path = save_output(sidecar, write_failures)
            tui.warning(f"Failed generations saved to: {failed_path}")
    except OSError as error:
        raise ConfigurationError(f"Error saving topics: {error}") from error
    else:
        return actual
