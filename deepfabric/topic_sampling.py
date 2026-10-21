"""Configuration shared by YAML and direct Python topic-sampling callers."""

import warnings

from typing import Literal

from pydantic import BaseModel, Field


class TopicSamplingConfig(BaseModel):
    """Options for execution within a topic's allocated quota."""

    topic_sample_mode: Literal["independent", "sequential", "bulk"] = "independent"
    shared_scope: Literal["seed", "full"] = "seed"
    seed_field: str | None = Field(default=None, min_length=1)
    completion_includes_topic_path: bool = False

    def validate_topic_sampling(self, output_schema: dict | None, *, has_tools: bool) -> None:
        """Validate options without changing existing independent-generation defaults."""
        if self.seed_field is not None:
            if output_schema is None:
                raise ValueError("seed_field requires generation.output_schema")
            if self.seed_field not in output_schema.get("properties", {}):
                raise ValueError("seed_field must name a top-level output_schema property")
            if self.seed_field not in output_schema.get("required", []):
                raise ValueError("seed_field must be required in output_schema")
        if self.topic_sample_mode == "independent" and "shared_scope" in self.model_fields_set:
            raise ValueError("shared_scope cannot be explicitly set with independent mode")
        if (
            self.topic_sample_mode != "independent"
            and output_schema is not None
            and self.shared_scope == "seed"
            and self.seed_field is None
        ):
            raise ValueError("shared_scope: seed with output_schema requires seed_field")
        if self.topic_sample_mode == "bulk" and self.shared_scope == "full":
            if has_tools:
                raise ValueError("The agent builder cannot produce a collection in one call")
            raise ValueError("bulk with shared_scope: full requires the truncation fix")
        if "completion_includes_topic_path" in self.model_fields_set and not (
            self.topic_sample_mode == "bulk" and self.shared_scope == "seed"
        ):
            raise ValueError("completion_includes_topic_path requires bulk with shared_scope: seed")


def warn_duplicate_prompt(dataset_system_prompt: str, instructions: str) -> None:
    """Warn about configured text that would reach a completion twice."""
    if instructions and dataset_system_prompt == instructions:
        warnings.warn(
            "dataset_system_prompt equals generation.instructions; completion prompts "
            "will contain the same text twice",
            UserWarning,
            stacklevel=3,
        )
