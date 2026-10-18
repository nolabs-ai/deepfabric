from abc import ABC, abstractmethod
from typing import NamedTuple


class TopicPath(NamedTuple):
    """A topic path with its associated unique identifier."""

    path: list[str]
    topic_id: str


class Topic(NamedTuple):
    """A unique topic with its UUID and content.

    Used for generation where we iterate over unique topics (by UUID)
    rather than paths. This deduplicated view is essential for graphs
    where multiple paths can lead to the same topic node.
    """

    uuid: str
    topic: str  # The topic text/content


class TopicModel(ABC):
    """Abstract base class for topic models like Tree and Graph."""

    @abstractmethod
    async def build_async(self) -> None:
        """Asynchronously build the topic model."""
        raise NotImplementedError

    def build(self) -> None:  # pragma: no cover - legacy compatibility
        """Deprecated synchronous entry point kept for legacy compatibility."""
        msg = "TopicModel.build() is no longer supported. Use build_async() instead."
        raise RuntimeError(msg)

    @abstractmethod
    def get_all_paths(self) -> list[list[str]]:
        """Returns all the paths in the topic model."""
        raise NotImplementedError

    @abstractmethod
    def get_all_paths_with_ids(self) -> list[TopicPath]:
        """Returns all paths with their unique identifiers.

        Returns:
            List of TopicPath namedtuples containing (path, topic_id).
            The topic_id is a stable identifier for the leaf node of each path.
        """
        raise NotImplementedError

    @abstractmethod
    def get_unique_topics(self) -> list[Topic]:
        """Returns deduplicated topics by UUID.

        For generation, we iterate over unique topics rather than paths.
        This is important for graphs where multiple paths can lead to the
        same topic node. Each UUID receives one allocation quota, independent of route count.

        Returns:
            List of Topic namedtuples containing (uuid, topic).
            Each UUID appears exactly once.
        """
        raise NotImplementedError

    def get_path_by_id(self, topic_id: str) -> list[str] | None:
        """Look up a path by its topic_id.

        Searches enumerated leaf paths by default. Subclasses (e.g. graph)
        may override this method to support lookups for non-leaf topics.

        Args:
            topic_id: The unique identifier for a topic path.

        Returns:
            The path list if found, None otherwise.
        """
        for topic_path in self.get_all_paths_with_ids():
            if topic_path.topic_id == topic_id:
                return topic_path.path
        return None
