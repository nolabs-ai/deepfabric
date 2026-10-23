"""Preview real prompt composition, stopping at the shared request boundary."""

from collections.abc import Awaitable
from contextlib import suppress
from typing import Any

from .builders import ConversationBuilderFactory
from .builders_agent import SingleTurnAgentBuilder
from .bulk import (
    completion_schema,
    make_collection_model,
    make_completion_model,
    record_validator,
    seed_value_schema,
)
from .config import DeepFabricConfig
from .generator import DataSetGenerator
from .graph import Graph
from .llm.prompt_trace import PromptPreviewCaptured, get_prompt_trace
from .prompts import SEQUENTIAL_SAMPLE_PROMPT
from .schemas import AgentStep, ChatMessage, ToolExecution
from .tree import Tree

TOPIC_PATH = "{{{topic path}}}"
USER_REQUEST = "{{{user request}}}"


async def _capture(request: Awaitable[Any]) -> None:
    """Stop after composing and recording a prompt, without fabricating an LLM result."""
    with suppress(PromptPreviewCaptured):
        await request


def _offline_generation_params(params: dict) -> dict:
    """Keep local tool definitions, substituting placeholders for remote definitions."""
    params = params.copy()
    components = params.get("tool_components", {})
    custom = list(params.get("custom_tools", []))
    for component, names in components.items():
        if component != "builtin":
            custom.extend(
                {
                    "name": name,
                    "component": component,
                    "description": "{{{remote tool description and parameters}}}",
                    "parameters": [],
                    "returns": "{{{tool result}}}",
                }
                for name in names
            )
    params.update(
        tool_components={k: v for k, v in components.items() if k == "builtin"},
        custom_tools=custom,
        spin_endpoint=None,
        checkpoint_interval=None,
        checkpoint_path=None,
    )
    return params


async def _preview_builder(engine: DataSetGenerator, prompt: str, *, pinned: bool = False) -> None:
    builder = ConversationBuilderFactory.create(
        config=engine.config, llm=engine.llm_client, tool_registry=engine.tool_registry
    )
    if not isinstance(builder, SingleTurnAgentBuilder):
        await _capture(
            builder.generate(
                prompt,
                pinned_user_message=ChatMessage(role="user", content=USER_REQUEST)
                if pinned
                else None,
            )
        )
        return

    # Preview each dependent agent stage separately. Never enter the tool execution loop.
    if not pinned:
        await _capture(builder._generate_user_question(prompt))
    builder._completion_context = (
        prompt if pinned or engine.config.topic_sample_mode == "sequential" else None
    )
    user = ChatMessage(role="user", content=USER_REQUEST)
    result = ToolExecution(
        function_name="{{{tool name}}}",
        arguments="{}",
        reasoning="{{{reasoning}}}",
        result="{{{tool result}}}",
    )
    step = AgentStep(thought="{{{previous reasoning}}}", is_final=False, tool_calls=[])
    await _capture(builder._generate_next_step_impl(user, [step], [result]))
    await _capture(builder._generate_agent_conclusion(user, [step], [result]))


async def preview_generation(
    config: DeepFabricConfig,
    *,
    topics_overrides: dict,
    generation_overrides: dict,
    topic_only: bool = False,
    topics_load: bool = False,
) -> None:
    """Show representative first/later cases using configured production builders.

    Verbose previews show later cases too; placeholders stand for unavailable
    model output. No topic/dataset/checkpoint files or provider clients are created.
    """
    session = get_prompt_trace()
    if session is None or not session.dry_run:
        raise RuntimeError("Prompt preview requires an active dry-run session")
    print("Dry run: no model requests or tool executions. Braced values are placeholders.")
    if not topics_load:
        params = config.get_topics_params(**topics_overrides)
        if config.topics.mode == "tree":
            topic_model = Tree(**params)
            await _capture(
                topic_model.get_subtopics(
                    topic_model.system_prompt, [topic_model.topic_prompt], topic_model.degree
                )
            )
            if session.verbose and topic_model.depth > 1:
                await _capture(
                    topic_model.get_subtopics(
                        topic_model.system_prompt,
                        [topic_model.topic_prompt, TOPIC_PATH],
                        topic_model.degree,
                    )
                )
        else:
            topic_model = Graph(**params)
            await _capture(
                topic_model.get_subtopics_and_connections(topic_model.root, topic_model.degree)
            )
            if session.verbose and topic_model.depth > 1:
                node = topic_model.add_node(TOPIC_PATH)
                topic_model.add_edge(topic_model.root.id, node.id)
                await _capture(topic_model.get_subtopics_and_connections(node, topic_model.degree))
    if topic_only:
        return

    engine = DataSetGenerator(
        **_offline_generation_params(config.get_generation_params(**generation_overrides))
    )
    allocation = {"uuid": "preview", "path": [TOPIC_PATH], "quota": 2}
    if engine.config.topic_sample_mode == "bulk":
        item_schema = (
            seed_value_schema(engine.config.output_schema, engine.config.seed_field)
            if engine.config.output_schema is not None
            else {"type": "string", "minLength": 1}
        )
        for later in [False, True] if session.verbose else [False]:
            prompt = engine._build_bulk_collection_prompt(
                allocation,
                1 if later else allocation["quota"],
                ["{{{previous accepted seed}}}"] if later else [],
                0,
            )
            await _capture(
                engine.llm_client.generate_async(prompt, make_collection_model(item_schema))
            )
        if engine.config.shared_scope == "full":
            return
        if engine.config.output_schema is not None:
            schema = completion_schema(engine.config.output_schema, engine.config.seed_field)
            if not schema.get("properties"):
                return  # A validated seed alone already forms the entire record.
            await _capture(
                engine._complete_custom_seed_outcome(
                    allocation,
                    0,
                    0,
                    "{{{seed}}}",
                    make_completion_model(schema),
                    record_validator(schema),
                    record_validator(engine.config.output_schema),
                    0,
                )
            )
        else:
            prompt = engine._conversation_completion_prompt(allocation, USER_REQUEST, 0)
            await _preview_builder(engine, prompt, pinned=True)
        return

    for later in [False, True] if session.verbose else [False]:
        if engine.config.topic_sample_mode == "sequential":
            prompt = SEQUENTIAL_SAMPLE_PROMPT.format(
                system_prompt=engine.dataset_system_prompt,
                topic_path=TOPIC_PATH,
                instructions=engine.build_custom_instructions_text(),
                examples=engine.build_examples_text(0),
                tools=engine.build_tools_text(),
                scope=engine.config.shared_scope,
                history="{{{previous successful samples}}}" if later else "[]",
                output_instructions=(
                    "Generate a complete tool-using conversation with a user request, tool executions, and a final assistant reply."
                    if engine.tool_registry and engine.config.output_schema is None
                    else engine._bulk_record_instructions()
                ),
            )
        else:
            prompt = engine.build_prompt(engine._get_prompt_template(), 0, [TOPIC_PATH])
        if engine.config.output_schema is not None:
            await _capture(
                engine.llm_client.generate_async(prompt, engine._get_custom_schema_model())
            )
        else:
            await _preview_builder(engine, prompt)
