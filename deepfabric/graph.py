import asyncio
import hashlib
import json
import textwrap
import uuid

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    DEFAULT_MAX_TOKENS,
    MAX_RETRY_ATTEMPTS,
    TOPIC_GRAPH_DEFAULT_MODEL,
    TOPIC_GRAPH_DEFAULT_TEMPERATURE,
    TOPIC_GRAPH_SUMMARY,
)
from .llm import LLMClient
from .llm.rate_limit_detector import RateLimitDetector
from .metrics import trace
from .prompts import (
    GRAPH_EXPANSION_PROMPT,
    GRAPH_EXPANSION_PROMPT_NO_CONNECTIONS,
    GraphPromptBuilder,
)
from .schemas import GraphSubtopics
from .stream_simulator import simulate_stream
from .topic_model import Topic, TopicModel, TopicPath

if TYPE_CHECKING:  # only for type hints to avoid runtime cycles
    from .progress import ProgressReporter

RETRY_BASE_DELAY = 0.5  # seconds
ERROR_MESSAGE_MAX_LENGTH = 40  # Max chars for error messages in TUI


class GraphConfig(BaseModel):
    """Configuration for constructing a topic graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic graph"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    provider: str = Field(
        default="ollama",
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model_name: str = Field(
        default=TOPIC_GRAPH_DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=TOPIC_GRAPH_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    degree: int = Field(default=3, ge=1, le=10, description="The branching factor of the graph")
    depth: int = Field(default=2, ge=1, le=5, description="The depth of the graph")
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
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    gemini_safety_settings: list[dict] | None = Field(
        default=None,
        description="Safety settings for Gemini models",
    )
    prompt_style: Literal["default", "isolated", "anchored"] = Field(
        default="default",
        description="Prompt style: 'default' (cross-connections, generic), 'isolated' (no connections, generic), 'anchored' (no connections, domain-aware)",
    )


class GraphMetadata(BaseModel):
    """Metadata for the entire graph for provenance tracking."""

    provider: str = Field(..., description="LLM provider used (e.g., openai, ollama)")
    model: str = Field(..., description="Model name used (e.g., gpt-4o)")
    temperature: float = Field(..., description="Temperature setting used for generation")
    created_at: str = Field(..., description="ISO 8601 timestamp when graph was created")


class NodeModel(BaseModel):
    """Pydantic model for a node in the graph."""

    id: int
    topic: str
    children: list[int] = Field(default_factory=list)
    parents: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphModel(BaseModel):
    """Pydantic model for the entire topic graph."""

    nodes: dict[int, NodeModel]
    root_id: int
    metadata: GraphMetadata | None = Field(
        default=None, description="Graph-level metadata for provenance tracking"
    )


class Node:
    """Represents a node in the Graph for runtime manipulation."""

    def __init__(self, topic: str, node_id: int, metadata: dict[str, Any] | None = None):
        self.topic: str = topic
        self.id: int = node_id
        self.children: list[Node] = []
        self.parents: list[Node] = []
        self.metadata: dict[str, Any] = metadata.copy() if metadata is not None else {}

        # Auto-generate uuid if not present (stable node identification)
        if "uuid" not in self.metadata:
            self.metadata["uuid"] = str(uuid.uuid4())

        # Auto-generate topic_hash if not present (duplicate detection via SHA256)
        if "topic_hash" not in self.metadata:
            self.metadata["topic_hash"] = hashlib.sha256(topic.encode("utf-8")).hexdigest()

    def to_pydantic(self) -> NodeModel:
        """Converts the runtime Node to its Pydantic model representation."""
        return NodeModel(
            id=self.id,
            topic=self.topic,
            children=[child.id for child in self.children],
            parents=[parent.id for parent in self.parents],
            metadata=self.metadata,
        )


class Graph(TopicModel):
    """Represents the topic graph and manages its structure."""

    def __init__(self, **kwargs):
        try:
            self.config = GraphConfig.model_validate(kwargs)
        except Exception as e:
            raise ValueError(f"Invalid graph configuration: {str(e)}") from e  # noqa: TRY003

        # Initialize from config
        self.topic_prompt = self.config.topic_prompt
        self.model_system_prompt = self.config.topic_system_prompt
        self.provider = self.config.provider
        self.model_name = self.config.model_name
        self.temperature = self.config.temperature
        self.degree = self.config.degree
        self.depth = self.config.depth
        self.max_concurrent = self.config.max_concurrent
        self.max_tokens = self.config.max_tokens
        self.prompt_style = self.config.prompt_style

        # Initialize LLM client
        llm_kwargs = {}
        if self.config.base_url:
            llm_kwargs["base_url"] = self.config.base_url
        if self.config.gemini_safety_settings:
            llm_kwargs["gemini_safety_settings"] = self.config.gemini_safety_settings

        self.llm_client = LLMClient(
            provider=self.provider,
            model_name=self.model_name,
            **llm_kwargs,
        )

        # Progress reporter for streaming feedback (set by topic_manager)
        self.progress_reporter: ProgressReporter | None = None

        # Store creation timestamp for provenance tracking
        self.created_at: datetime = datetime.now(timezone.utc)

        trace(
            "graph_created",
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "degree": self.degree,
                "depth": self.depth,
            },
        )

        self.root: Node = Node(self.config.topic_prompt, 0)
        self.nodes: dict[int, Node] = {0: self.root}
        self._next_node_id: int = 1
        self.failed_generations: list[dict[str, Any]] = []

    def _wrap_text(self, text: str, width: int = 30) -> str:
        """Wrap text to a specified width."""
        return "\n".join(textwrap.wrap(text, width=width))

    def add_node(self, topic: str, metadata: dict[str, Any] | None = None) -> Node:
        """Adds a new node to the graph."""
        node = Node(topic, self._next_node_id, metadata)
        self.nodes[node.id] = node
        self._next_node_id += 1
        return node

    def add_edge(self, parent_id: int, child_id: int) -> None:
        """Adds a directed edge from a parent to a child node, avoiding duplicates."""
        parent_node = self.nodes.get(parent_id)
        child_node = self.nodes.get(child_id)
        if parent_node and child_node:
            if child_node not in parent_node.children:
                parent_node.children.append(child_node)
            if parent_node not in child_node.parents:
                child_node.parents.append(parent_node)

    def find_node_by_uuid(self, uuid: str) -> Node | None:
        """Find a node by its UUID.

        Args:
            uuid: The UUID string to search for.

        Returns:
            The Node if found, None otherwise.
        """
        for node in self.nodes.values():
            if node.metadata.get("uuid") == uuid:
                return node
        return None

    def remove_node(self, node_id: int) -> None:
        """Remove a single node from the graph, cleaning up bidirectional references.

        Does not remove children — use remove_subtree() for cascading removal.

        Args:
            node_id: The ID of the node to remove.

        Raises:
            ValueError: If node_id is the root node or does not exist.
        """
        if node_id == self.root.id:
            raise ValueError("Cannot remove the root node")  # noqa: TRY003
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph")  # noqa: TRY003

        for parent in node.parents:
            if node in parent.children:
                parent.children.remove(node)

        for child in node.children:
            if node in child.parents:
                child.parents.remove(node)

        del self.nodes[node_id]

    def remove_subtree(self, node_id: int) -> list[int]:
        """Remove a node and all its descendants from the graph.

        Args:
            node_id: The ID of the node to remove (along with all descendants).

        Returns:
            List of removed node IDs.

        Raises:
            ValueError: If node_id is the root node or does not exist.
        """
        if node_id == self.root.id:
            raise ValueError("Cannot remove the root node")  # noqa: TRY003
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph")  # noqa: TRY003

        # BFS to collect all descendant node IDs
        to_remove: list[int] = []
        queue = [node]
        visited: set[int] = set()
        while queue:
            current = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)
            to_remove.append(current.id)
            for child in current.children:
                if child.id not in visited:
                    queue.append(child)

        # Remove in reverse order (leaves first)
        for nid in reversed(to_remove):
            self.remove_node(nid)

        return to_remove

    def prune_at_level(self, max_depth: int) -> list[int]:
        """Remove all nodes below the given depth level.

        Nodes at exactly max_depth become leaf nodes. Root is depth 0.

        Args:
            max_depth: Maximum depth to keep (inclusive).
                       0 = keep only root, 1 = root and its children, etc.

        Returns:
            List of removed node IDs.

        Raises:
            ValueError: If max_depth is negative.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")  # noqa: TRY003

        # BFS from root to compute node depths
        node_depths: dict[int, int] = {}
        queue: list[tuple[Node, int]] = [(self.root, 0)]
        visited: set[int] = set()
        while queue:
            current, depth = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)
            node_depths[current.id] = depth
            for child in current.children:
                if child.id not in visited:
                    queue.append((child, depth + 1))

        to_remove_set = {nid for nid, d in node_depths.items() if d > max_depth}

        # Sever children links from boundary nodes
        for nid, d in node_depths.items():
            if d == max_depth:
                self.nodes[nid].children = [
                    c for c in self.nodes[nid].children if c.id not in to_remove_set
                ]

        # Remove deeper nodes
        for nid in to_remove_set:
            node = self.nodes[nid]
            for parent in node.parents:
                if node in parent.children:
                    parent.children.remove(node)
            for child in node.children:
                if node in child.parents:
                    child.parents.remove(node)
            del self.nodes[nid]

        return list(to_remove_set)

    def to_pydantic(self) -> GraphModel:
        """Converts the runtime graph to its Pydantic model representation."""
        return GraphModel(
            nodes={node_id: node.to_pydantic() for node_id, node in self.nodes.items()},
            root_id=self.root.id,
            metadata=GraphMetadata(
                provider=self.provider,
                model=self.model_name,
                temperature=self.temperature,
                created_at=self.created_at.isoformat(),
            ),
        )

    def to_json(self) -> str:
        """Returns a JSON representation of the graph."""
        pydantic_model = self.to_pydantic()
        return pydantic_model.model_dump_json(indent=2)

    def save(self, save_path: str) -> None:
        """Save the topic graph to a file."""
        from pathlib import Path  # noqa: PLC0415

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(self.to_json())

        # Save failed generations if any
        if self.failed_generations:
            failed_path = save_path.replace(".json", "_failed.jsonl")
            with open(failed_path, "w") as f:
                for failed in self.failed_generations:
                    f.write(json.dumps({"failed_generation": failed}) + "\n")

    @classmethod
    def from_json(cls, json_path: str, params: dict) -> "Graph":
        """Load a topic graph from a JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        graph_model = GraphModel(**data)
        graph = cls(**params)
        graph.nodes = {}

        # Restore original creation timestamp if present in the loaded graph
        if graph_model.metadata and graph_model.metadata.created_at:
            # Handle 'Z' suffix for Python < 3.11 compatibility
            created_at_str = graph_model.metadata.created_at.replace("Z", "+00:00")
            graph.created_at = datetime.fromisoformat(created_at_str)

        # Create nodes
        for node_model in graph_model.nodes.values():
            node = Node(node_model.topic, node_model.id, node_model.metadata)
            graph.nodes[node.id] = node
            if node.id == graph_model.root_id:
                graph.root = node

        # Create edges
        for node_model in graph_model.nodes.values():
            for child_id in node_model.children:
                graph.add_edge(node_model.id, child_id)

        graph._next_node_id = max(graph.nodes.keys()) + 1
        return graph

    @classmethod
    def load(cls, json_path: str) -> "Graph":
        """Load a graph from JSON without initializing LLM client.

        Intended for inspection and manipulation operations that don't
        require LLM generation capabilities. Restores provider, model,
        and temperature from the file metadata so saves preserve them.
        """
        params = {
            "topic_prompt": "loaded",
            "model_name": "placeholder/model",
            "degree": 3,
            "depth": 2,
            "temperature": 0.7,
        }
        graph = cls.from_json(json_path, params)

        # Restore original metadata so save() preserves provenance
        with open(json_path) as f:
            raw = json.load(f)
        file_meta = raw.get("metadata") or {}
        if file_meta.get("provider"):
            graph.provider = file_meta["provider"]
        if file_meta.get("model"):
            graph.model_name = file_meta["model"]
        if file_meta.get("temperature") is not None:
            graph.temperature = file_meta["temperature"]

        return graph

    def visualize(self, save_path: str) -> None:
        """Visualize the graph and save it to a file."""
        try:
            from mermaid import Mermaid  # noqa: PLC0415
        except ImportError as err:
            raise ImportError(
                "Mermaid package is required for graph visualization. "
                "Please install it via 'pip install mermaid'."
            ) from err

        graph_definition = "graph TD\n"
        for node in self.nodes.values():
            graph_definition += f'    {node.id}["{self._wrap_text(node.topic)}"]\n'

        for node in self.nodes.values():
            for child in node.children:
                graph_definition += f"    {node.id} --> {child.id}\n"

        mermaid = Mermaid(graph_definition)
        mermaid.to_svg(f"{save_path}.svg")

    async def build_async(self):
        """Builds the graph by iteratively calling the LLM to get subtopics and connections.

        Yields:
            dict: Progress events with event type and associated data
        """

        def _raise_if_build_failed():
            """Check if build failed completely and raise appropriate error."""
            if len(self.nodes) == 1 and self.failed_generations:
                # Surface the actual first error instead of a generic message
                first_error = self.failed_generations[0]["last_error"]
                raise RuntimeError(first_error)

        try:
            for depth in range(self.depth):
                leaf_nodes = [node for node in self.nodes.values() if not node.children]
                yield {"event": "depth_start", "depth": depth + 1, "leaf_count": len(leaf_nodes)}

                if leaf_nodes:
                    # Use semaphore to limit concurrent LLM calls and avoid rate limits
                    semaphore = asyncio.Semaphore(self.max_concurrent)

                    async def bounded_expand(
                        node: Node, sem: asyncio.Semaphore = semaphore
                    ) -> tuple[int, int]:
                        async with sem:
                            return await self.get_subtopics_and_connections(node, self.degree)

                    tasks = [bounded_expand(node) for node in leaf_nodes]
                    results = await asyncio.gather(*tasks)

                    for node, (subtopics_added, connections_added) in zip(
                        leaf_nodes, results, strict=True
                    ):
                        yield {
                            "event": "node_expanded",
                            "node_topic": node.topic,
                            "subtopics_added": subtopics_added,
                            "connections_added": connections_added,
                        }

                yield {"event": "depth_complete", "depth": depth + 1}

            # Check if build was completely unsuccessful (only root node exists)
            _raise_if_build_failed()

            trace(
                "graph_built",
                {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "nodes_count": len(self.nodes),
                    "failed_generations": len(self.failed_generations),
                    "success": len(self.nodes) > 1,
                },
            )

            yield {
                "event": "build_complete",
                "nodes_count": len(self.nodes),
                "failed_generations": len(self.failed_generations),
            }

        except Exception as e:
            yield {"event": "error", "error": str(e)}
            raise

    def _process_subtopics_response(
        self, response: GraphSubtopics, parent_node: Node
    ) -> tuple[int, int]:
        """Process a GraphSubtopics response, adding nodes and edges to the graph.

        Args:
            response: The structured response containing subtopics and connections.
            parent_node: The parent node to connect new subtopics to.

        Returns:
            A tuple of (subtopics_added, connections_added).
        """
        subtopics_added = 0
        connections_added = 0

        for subtopic_data in response.subtopics:
            new_node = self.add_node(subtopic_data.topic)
            self.add_edge(parent_node.id, new_node.id)
            subtopics_added += 1
            for connection_id in subtopic_data.connections:
                if connection_id in self.nodes:
                    self.add_edge(connection_id, new_node.id)
                    connections_added += 1

        return subtopics_added, connections_added

    def _get_friendly_error_message(self, exception: Exception) -> str:
        """Convert an exception into a user-friendly error message for TUI display.

        Args:
            exception: The exception that occurred during generation.

        Returns:
            A concise, user-friendly error message suitable for display.
        """
        # Check for rate limit errors using the detector
        if RateLimitDetector.is_rate_limit_error(exception, self.provider):
            return self._format_rate_limit_message(exception)

        error_str = str(exception).lower()

        # Check for validation/schema errors (Pydantic issues)
        validation_indicators = ["validation failed", "validation error", "validationerror"]
        if any(ind in error_str for ind in validation_indicators):
            return "Response format issue - retrying"

        # Check for network/connection errors
        network_indicators = ["timeout", "connection", "network", "socket"]
        if any(ind in error_str for ind in network_indicators):
            return "Connection issue - retrying"

        # Check for server errors
        server_indicators = ["503", "502", "500", "504", "server error", "service unavailable"]
        if any(ind in error_str for ind in server_indicators):
            return "Server error - retrying"

        # Fallback: truncate the original error for display
        return self._truncate_error_message(str(exception))

    def _format_rate_limit_message(self, exception: Exception) -> str:
        """Format a rate limit error into a user-friendly message."""
        quota_info = RateLimitDetector.extract_quota_info(exception, self.provider)
        if quota_info.daily_quota_exhausted:
            return "Daily quota exhausted - waiting"
        if quota_info.quota_type:
            return f"Rate limit ({quota_info.quota_type}) - backing off"
        return "Rate limit reached - backing off"

    def _truncate_error_message(self, message: str) -> str:
        """Truncate an error message to fit within the TUI display limit."""
        if len(message) > ERROR_MESSAGE_MAX_LENGTH:
            return message[: ERROR_MESSAGE_MAX_LENGTH - 3] + "..."
        return message

    async def _generate_subtopics_with_retry(
        self, prompt: str, parent_node: Node
    ) -> GraphSubtopics | None:
        """Generate subtopics with retry logic and exponential backoff.

        Args:
            prompt: The prompt to send to the LLM.
            parent_node: The parent node (used for error tracking and retry events).

        Returns:
            The GraphSubtopics response, or None if all retries failed.

        Raises:
            RuntimeError: If authentication fails (API key errors are not retried).
        """
        last_error: str | None = None

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = await self.llm_client.generate_async(
                    prompt=prompt,
                    schema=GraphSubtopics,
                    max_retries=1,  # Don't retry inside - we handle it here
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
                simulate_stream(
                    self.progress_reporter,
                    response.model_dump_json(indent=2),
                    source="graph_generation",
                )

            except Exception as e:
                last_error = str(e)
                error_str = str(e).lower()

                # Check if it's an API key related error - don't retry these
                if any(
                    keyword in error_str
                    for keyword in ["api_key", "api key", "authentication", "unauthorized"]
                ):
                    error_msg = (
                        f"Authentication failed for provider '{self.provider}'. "
                        "Please set the required API key environment variable."
                    )
                    raise RuntimeError(error_msg) from e

                # Log retry attempt if not the last one
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    if self.progress_reporter:
                        # Use friendly error message for TUI display
                        friendly_error = self._get_friendly_error_message(e)
                        self.progress_reporter.emit_node_retry(
                            node_topic=parent_node.topic,
                            attempt=attempt + 1,
                            max_attempts=MAX_RETRY_ATTEMPTS,
                            error_summary=friendly_error,
                        )
                    # Brief delay before retry with exponential backoff
                    delay = (2**attempt) * RETRY_BASE_DELAY
                    await asyncio.sleep(delay)

            else:
                # Success - return the response
                return response

        # All retries exhausted - record failure
        self.failed_generations.append(
            {"node_id": parent_node.id, "attempts": MAX_RETRY_ATTEMPTS, "last_error": last_error}
        )
        return None

    def _get_path_to_node(self, node: Node) -> list[str]:
        """Get the topic path from root to the given node."""
        path = []
        current = node
        while current is not None:
            path.append(current.topic)
            # First parent is the primary parent from tree expansion;
            # cross-connections are added later and appear after index 0
            current = current.parents[0] if current.parents else None
        return list(reversed(path))

    async def get_subtopics_and_connections(
        self, parent_node: Node, num_subtopics: int
    ) -> tuple[int, int]:
        """Generate subtopics and connections for a given node.

        Args:
            parent_node: The node to generate subtopics for.
            num_subtopics: The number of subtopics to generate.

        Returns:
            A tuple of (subtopics_added, connections_added).
        """
        # Choose prompt based on prompt_style setting
        if self.prompt_style == "anchored":
            # Domain-aware prompts with examples for focused generation
            topic_path = self._get_path_to_node(parent_node)
            domain = GraphPromptBuilder.detect_domain(self.model_system_prompt, topic_path)
            graph_prompt = GraphPromptBuilder.build_anchored_prompt(
                topic_path=topic_path,
                num_subtopics=num_subtopics,
                system_prompt=self.model_system_prompt,
                domain=domain,
            )
        elif self.prompt_style == "isolated":
            # No connections, generic prompt
            graph_prompt = GRAPH_EXPANSION_PROMPT_NO_CONNECTIONS.replace(
                "{{current_topic}}", parent_node.topic
            )
            graph_prompt = graph_prompt.replace("{{num_subtopics}}", str(num_subtopics))
        else:
            # default: cross-connections enabled, generic prompt
            graph_summary = (
                self.to_json()
                if len(self.nodes) <= TOPIC_GRAPH_SUMMARY
                else "Graph too large to display"
            )
            graph_prompt = GRAPH_EXPANSION_PROMPT.replace(
                "{{current_graph_summary}}", graph_summary
            )
            graph_prompt = graph_prompt.replace("{{current_topic}}", parent_node.topic)
            graph_prompt = graph_prompt.replace("{{num_subtopics}}", str(num_subtopics))

        response = await self._generate_subtopics_with_retry(graph_prompt, parent_node)
        if response is None:
            return 0, 0

        return self._process_subtopics_response(
            response, parent_node
        )  # No subtopics or connections added on failure

    def get_all_paths(self) -> list[list[str]]:
        """Returns all paths from the root to leaf nodes."""
        paths = []
        visited: set[int] = set()
        self._dfs_paths(self.root, [self.root.topic], paths, visited)
        return paths

    def get_all_paths_with_ids(self) -> list[TopicPath]:
        """Returns all paths from root to leaf nodes with their leaf node UUIDs.

        Returns:
            List of TopicPath namedtuples containing (path, topic_id).
            The topic_id is the UUID of the leaf node for each path.
        """
        result: list[TopicPath] = []
        visited: set[int] = set()
        self._dfs_paths_with_ids(self.root, [self.root.topic], result, visited)
        return result

    def _dfs_paths_with_ids(
        self,
        node: Node,
        current_path: list[str],
        result: list[TopicPath],
        visited: set[int],
    ) -> None:
        """Helper function for DFS traversal to find all paths with leaf node UUIDs.

        Args:
            node: Current node being visited
            current_path: Path from root to current node
            result: Accumulated list of TopicPath namedtuples
            visited: Set of node IDs already visited in current path to prevent cycles
        """
        if node.id in visited:
            return

        visited.add(node.id)

        if not node.children:
            # Leaf node - add path with this node's UUID
            topic_id = node.metadata.get("uuid", str(node.id))
            result.append(TopicPath(path=current_path, topic_id=topic_id))

        for child in node.children:
            self._dfs_paths_with_ids(child, current_path + [child.topic], result, visited)

        visited.remove(node.id)

    def get_unique_topics(self) -> list[Topic]:
        """Returns deduplicated topics by UUID.

        Iterates through all nodes in the graph and returns unique topics.
        Each node has a UUID in its metadata, ensuring uniqueness.

        Returns:
            List of Topic namedtuples containing (uuid, topic).
            Each UUID appears exactly once.
        """
        seen_uuids: set[str] = set()
        result: list[Topic] = []

        for node in self.nodes.values():
            # Skip root node — it holds the generation seed prompt, not a topic
            if node.id == self.root.id:
                continue
            node_uuid = node.metadata.get("uuid")
            if node_uuid and node_uuid not in seen_uuids:
                seen_uuids.add(node_uuid)
                result.append(Topic(uuid=node_uuid, topic=node.topic))

        return result

    def _dfs_paths(
        self, node: Node, current_path: list[str], paths: list[list[str]], visited: set[int]
    ) -> None:
        """Helper function for DFS traversal to find all paths.

        Args:
            node: Current node being visited
            current_path: Path from root to current node
            paths: Accumulated list of complete paths
            visited: Set of node IDs already visited in current path to prevent cycles
        """
        # Prevent cycles by tracking visited nodes in the current path
        if node.id in visited:
            return

        visited.add(node.id)

        if not node.children:
            paths.append(current_path)

        for child in node.children:
            self._dfs_paths(child, current_path + [child.topic], paths, visited)

        # Remove node from visited when backtracking to allow it in other paths
        visited.remove(node.id)

    def _has_cycle_util(self, node: Node, visited: set[int], recursion_stack: set[int]) -> bool:
        """Utility function for cycle detection."""
        visited.add(node.id)
        recursion_stack.add(node.id)

        for child in node.children:
            if child.id not in visited:
                if self._has_cycle_util(child, visited, recursion_stack):
                    return True
            elif child.id in recursion_stack:
                return True

        recursion_stack.remove(node.id)
        return False

    def has_cycle(self) -> bool:
        """Checks if the graph contains a cycle."""
        visited: set[int] = set()
        recursion_stack: set[int] = set()
        for node_id in self.nodes:
            if node_id not in visited and self._has_cycle_util(
                self.nodes[node_id], visited, recursion_stack
            ):
                return True
        return False
