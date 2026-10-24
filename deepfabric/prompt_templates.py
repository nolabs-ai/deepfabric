"""Validated, literal prompt templates and packaged default assets."""

import hashlib
import re

from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

COMMON_FIELDS = frozenset(
    {
        "system_prompt",
        "generation_system_prompt",
        "dataset_system_prompt",
        "instructions",
        "examples",
        "subtopics",
        "topic_path",
        "tools",
        "tools_section",
        "requested",
    }
)
COLLECTION_FIELDS = frozenset(
    {
        "request_instruction",
        "output_instructions",
        "record_instructions",
        "history",
        "history_section",
        "accepted_siblings_section",
        "seed_field",
        "scope",
    }
)
COMPLETION_FIELDS = frozenset(
    {
        "seed",
        "seed_field",
        "seed_section",
        "context_guidance",
        "output_instructions",
        "retry_feedback",
    }
)
PLACEHOLDER = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")
LEGACY_PLACEHOLDER = re.compile(r"\{\{\{\{\s*([A-Za-z_]\w*)\s*\}\}\}\}")
Step = Literal["full_sample", "sequential_later", "seed", "completion"]
TEMPLATE_STEPS = ("full_sample", "sequential_later", "seed", "completion")


def normalize_legacy(text: str) -> str:
    return LEGACY_PLACEHOLDER.sub(lambda match: "{{ " + match[1] + " }}", text)


def placeholders(text: str) -> set[str]:
    if "{{" in PLACEHOLDER.sub("", text):
        raise ValueError("Unclosed prompt placeholder; use {{ name }}")
    names = set()
    for match in PLACEHOLDER.finditer(text):
        name = match[1].strip()
        if name.startswith("."):  # Literal Docker Go-template fields, e.g. {{.ID}}.
            continue
        if not name.isidentifier():
            raise ValueError(
                f"Invalid prompt placeholder {match[0]!r}; use a bare name such as {{{{ topic_path }}}}"
            )
        names.add(name)
    return names


TOPIC_FIELDS = frozenset(
    {
        "system_prompt",
        "topic_prompt",
        "current_topic",
        "topic_path",
        "num_subtopics",
        "requested",
        "examples",
        "current_graph_summary",
        "prompt_style",
        "mode",
        "default_prompt",
    }
)


def validate_template(text: str, step: str) -> None:
    if not text.strip():
        raise ValueError(f"{step} prompt template must not be empty")
    allowed = (
        TOPIC_FIELDS
        if step in ("initial", "expansion")
        else COMMON_FIELDS | (COMPLETION_FIELDS if step == "completion" else COLLECTION_FIELDS)
    )
    unknown = placeholders(normalize_legacy(text)) - allowed
    if unknown:
        raise ValueError(
            f"Unknown or unavailable {step} prompt placeholders: {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(allowed))}"
        )


def render_template(text: str, values: dict, *, strict: bool = True) -> str:
    """Replace known slots once, never reinterpreting inserted data as a template."""

    def replace(match):
        name = match[1].strip()
        if name in values:
            return str(values[name])
        if name.startswith(".") or not strict:
            return match[0]
        raise ValueError(f"Missing prompt placeholder value: {name}")

    return PLACEHOLDER.sub(replace, normalize_legacy(text))


@cache
def default_template(name: str) -> str:
    return files("deepfabric").joinpath("templates", name + ".txt").read_text(encoding="utf-8")


def compatibility_template(name: str, style: Literal["legacy", "format", "raw"]) -> str:
    """Keep existing Python prompt constants compatible with their public callers."""
    text = default_template(name)
    if style == "raw":
        return text
    return PLACEHOLDER.sub(
        lambda match: "{{{{" + match[1].strip() + "}}}}"
        if style == "legacy"
        else "{" + match[1].strip() + "}",
        text,
    )


class PromptTemplateSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    file: str | None = None
    _content: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def load(self, info: ValidationInfo):
        if (self.text is None) == (self.file is None):
            raise ValueError("A prompt template requires exactly one of text or file")
        if self.file is not None:
            path = Path(self.file).expanduser()
            if not path.is_absolute():
                path = Path((info.context or {}).get("config_dir", Path.cwd())) / path
            try:
                self._content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise ValueError(f"Cannot read UTF-8 prompt template {path}: {error}") from error
            self.file = str(path.resolve())
        else:
            self._content = self.text
        if not self._content.strip():
            raise ValueError("Prompt template must not be empty")
        return self

    @property
    def content(self) -> str:
        return self._content


class GenerationPromptTemplates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_sample: PromptTemplateSource | None = None
    sequential_later: PromptTemplateSource | None = None
    seed: PromptTemplateSource | None = None
    completion: PromptTemplateSource | None = None

    @field_validator("full_sample", "sequential_later", "seed", "completion", mode="before")
    @classmethod
    def inline_source(cls, value):
        return {"text": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_sources(self):
        for name in TEMPLATE_STEPS:
            source = getattr(self, name)
            if source is not None:
                validate_template(source.content, name)
        return self


class TopicPromptTemplates(BaseModel):
    """Independent overrides for expanding the root and deeper topic nodes."""

    model_config = ConfigDict(extra="forbid")
    initial: PromptTemplateSource | None = None
    expansion: PromptTemplateSource | None = None

    @field_validator("initial", "expansion", mode="before")
    @classmethod
    def inline_source(cls, value):
        return {"text": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_sources(self):
        for name in ("initial", "expansion"):
            source = getattr(self, name)
            if source is not None:
                validate_template(source.content, name)
        return self

    def render(self, *, initial: bool, default_prompt: str, **values) -> str:
        source = self.initial if initial else self.expansion
        if source is None:
            return default_prompt
        return render_template(source.content, {**values, "default_prompt": default_prompt})


def template_manifest(
    templates: GenerationPromptTemplates | TopicPromptTemplates, legacy: str | None = None
) -> dict:
    overrides = {}
    for step in templates.__class__.model_fields:
        source = getattr(templates, step)
        content = (
            source.content if source is not None else legacy if step == "full_sample" else None
        )
        if content is not None:
            overrides[step] = {
                "text": content,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "file": source.file if source is not None else None,
            }
    assets = files("deepfabric").joinpath("templates")
    builtins = {
        asset.name: hashlib.sha256(default_template(asset.name[:-4]).encode()).hexdigest()
        for asset in sorted(assets.iterdir(), key=lambda asset: asset.name)
        if asset.name.endswith(".txt")
    }
    return {"overrides": overrides, "builtins": builtins}
