"""Configuration shared by YAML and direct Python topic-sampling callers."""

import warnings

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .prompt_templates import GenerationPromptTemplates, validate_template


class BulkConfig(BaseModel):
    """Output-token ceiling for one bulk collection request."""

    max_tokens_per_call: int | None = Field(
        default=None,
        ge=1,
        description="Bulk collection ceiling; defaults to max_tokens times requested items",
    )


class TopicSamplingConfig(BaseModel):
    """Options for execution within a topic's allocated quota."""

    topic_sample_mode: Literal["independent", "sequential", "bulk"] = "independent"
    shared_scope: Literal["seed", "full"] = "seed"
    bulk: BulkConfig = Field(default_factory=BulkConfig)
    seed_field: str | None = Field(default=None, min_length=1)
    completion_includes_topic_path: bool = False
    prompt_templates: GenerationPromptTemplates = Field(default_factory=GenerationPromptTemplates)
    prompt_template: str | None = Field(
        default=None, description="Shorthand for the full_sample template"
    )

    @model_validator(mode="after")
    def validate_prompt_overrides(self):
        if self.prompt_template is not None:
            if self.prompt_templates.full_sample is not None:
                raise ValueError(
                    "Use either prompt_template or prompt_templates.full_sample, not both"
                )
            validate_template(self.prompt_template, "full_sample")
        return self

    def validate_topic_sampling(self, output_schema: dict | None, *, has_tools: bool) -> None:
        """Validate options using configured tool components and custom tools."""
        if self.bulk.max_tokens_per_call is not None and self.topic_sample_mode != "bulk":
            raise ValueError("bulk.max_tokens_per_call requires topic_sample_mode: bulk")
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
        if self.topic_sample_mode == "bulk" and self.shared_scope == "full" and has_tools:
            raise ValueError("The agent builder cannot produce a collection in one call")
        if "completion_includes_topic_path" in self.model_fields_set and not (
            self.topic_sample_mode == "bulk" and self.shared_scope == "seed"
        ):
            raise ValueError("completion_includes_topic_path requires bulk with shared_scope: seed")
        unused = self._unused_prompt_templates(has_tools=has_tools, output_schema=output_schema)
        if unused:
            # A warning, not an error: one template set may be shared across mode experiments.
            # Default stacklevel keeps one warning when both config layers validate.
            warnings.warn(
                f"Prompt templates not used by this configuration and ignored: "
                f"{', '.join(unused)} (topic_sample_mode={self.topic_sample_mode}, "
                f"shared_scope={self.shared_scope}, tools={'yes' if has_tools else 'no'})",
                UserWarning,
                stacklevel=1,
            )

    def _unused_prompt_templates(
        self, *, has_tools: bool, output_schema: dict | None = None
    ) -> list[str]:
        """Name overrides the selected mode never renders, so they cannot silently do nothing."""
        bulk_seed = self.topic_sample_mode == "bulk" and self.shared_scope == "seed"
        # A custom output schema bypasses the tool agent in every mode.
        agent = has_tools and output_schema is None
        used = {
            # Bulk/seed collects with "seed" and completes with "completion" instead.
            "full_sample": not bulk_seed,
            "sequential_later": self.topic_sample_mode == "sequential",
            "seed": bulk_seed,
            # Tool-agent completions use the agent_step/agent_response stages.
            "completion": bulk_seed
            and not agent
            and not (
                output_schema is not None and seed_only_schema(output_schema, self.seed_field)
            ),
            "agent_step": agent,
            "agent_response": agent,
        }
        unused = [
            f"prompt_templates.{step}"
            for step, is_used in used.items()
            if getattr(self.prompt_templates, step) is not None and not is_used
        ]
        if self.prompt_template is not None and not used["full_sample"]:
            unused.append("prompt_template")
        return unused


def seed_only_schema(output_schema: dict, seed_field: str | None) -> bool:
    """Whether the seed is the whole record, so seed completion is skipped."""
    return set(output_schema.get("properties", {})) <= {seed_field}


def warn_duplicate_prompt(dataset_system_prompt: str, instructions: str) -> None:
    """Warn about configured text that would reach a completion twice."""
    if instructions and dataset_system_prompt == instructions:
        warnings.warn(
            "dataset_system_prompt equals generation.instructions; completion prompts "
            "will contain the same text twice",
            UserWarning,
            stacklevel=3,
        )
