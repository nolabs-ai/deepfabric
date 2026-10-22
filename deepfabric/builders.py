import logging

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from .progress import ProgressReporter
from .schemas import ChatMessage, Conversation, ReasoningTrace
from .stream_simulator import simulate_stream

if TYPE_CHECKING:
    from .generator import DataSetGeneratorConfig
    from .llm import LLMClient
    from .schemas import ToolRegistry

logger = logging.getLogger(__name__)


class ConversationBuilder(ABC):
    """Abstract base class for conversation builders.

    Each builder implements a specific strategy for generating conversations.
    Builders receive typed configuration and dependencies via constructor.

    Attributes:
        llm: LLM client for generation
        config: Typed configuration for the generator
        tool_registry: Optional tool registry for tool-calling conversations
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "DataSetGeneratorConfig",
        tool_registry: "ToolRegistry | None" = None,
        progress_reporter: ProgressReporter | None = None,
    ):
        """Initialize the conversation builder.

        Args:
            llm: LLM client for making generation requests
            config: Generator configuration (must be Pydantic model)
            tool_registry: Optional tool registry for tool-calling
            progress_reporter: Optional progress reporter for streaming feedback
        """
        self.llm = llm
        self.config = config
        self.tool_registry = tool_registry
        self.progress_reporter = progress_reporter

    @abstractmethod
    async def generate(
        self,
        topic_prompt: str,
        error_feedback: str | None = None,
        *,
        pinned_user_message: ChatMessage | None = None,
    ) -> Conversation:
        """Generate a complete conversation.

        Args:
            topic_prompt: The topic/scenario prompt to generate conversation about
            error_feedback: Optional error message from a previous failed attempt,
                           used to help the model correct its output on retry

        Returns:
            Complete Conversation object (Pydantic model)

        Raises:
            ValueError: If generation fails validation
        """
        pass


class BuilderType(BaseModel):
    """Type discriminator for builder selection.

    This model ensures type-safe builder selection based on configuration.
    """

    name: str = Field(description="Builder type name")
    requires_tools: bool = Field(default=False, description="Whether this builder requires tools")

    class Config:
        frozen = True


# Builder type constants
SINGLE_SHOT_BUILDER = BuilderType(name="single_shot", requires_tools=False)
SINGLE_TURN_AGENT_BUILDER = BuilderType(name="single_turn_agent", requires_tools=True)


def determine_builder_type(config: "DataSetGeneratorConfig") -> BuilderType:
    """Determine the appropriate builder type from configuration.

    Agent mode is implicit when tools are configured (tool_components or custom_tools).
    Single-turn agent mode is used for tool-calling conversations.

    Args:
        config: Generator configuration (Pydantic model)

    Returns:
        BuilderType indicating which builder to use

    Raises:
        ValueError: If configuration is invalid or unsupported
    """
    # Agent mode is implicit when tools are configured
    has_tools = config.tool_components or config.custom_tools
    if has_tools:
        return SINGLE_TURN_AGENT_BUILDER

    # Non-agent conversations use single-shot generation
    if config.conversation_type in ("basic", "cot"):
        return SINGLE_SHOT_BUILDER

    msg = f"Cannot determine builder type for conversation_type={config.conversation_type}"
    raise ValueError(msg)


class AssistantCompletion(BaseModel):
    """Assistant-only output: the user message is pinned and assembled in code."""

    model_config = {"extra": "forbid"}
    content: str = Field(min_length=1)


class ReasonedAssistantCompletion(AssistantCompletion):
    reasoning: ReasoningTrace


def pinned_user_copy(message: ChatMessage) -> ChatMessage:
    """Validate and copy a pinned request without trimming or regenerating its text."""
    if message.role != "user" or not isinstance(message.content, str) or not message.content:
        raise ValueError("Pinned user validation error: expected a nonempty user request")
    if message.tool_calls or message.tool_call_id:
        raise ValueError("Pinned user validation error: user requests cannot contain tool calls")
    return message.model_copy(deep=True)


class SingleShotBuilder(ConversationBuilder):
    """Builder for simple conversations using single-shot JSON generation.

    This builder generates the entire conversation in one LLM call using
    structured output with JSON schema validation. Suitable for:
    - Basic Q&A conversations
    - Chain-of-thought reasoning without tools
    - Any conversation that can be generated in one pass
    """

    async def generate(
        self,
        topic_prompt: str,
        error_feedback: str | None = None,
        *,
        pinned_user_message: ChatMessage | None = None,
    ) -> Conversation:
        """Generate conversation using single LLM call with JSON schema.

        Args:
            topic_prompt: Topic or scenario to generate conversation about
            error_feedback: Optional error message from a previous failed attempt

        Returns:
            Complete Conversation object

        Raises:
            ValueError: If LLM fails to generate valid conversation
        """
        if pinned_user_message is not None:
            return await self._complete_pinned_user(
                topic_prompt, pinned_user_message, error_feedback
            )
        # Build the generation prompt
        generation_prompt = self._build_prompt(topic_prompt, error_feedback)

        # Always use non-streaming for reliable structured output
        conversation = await self._generate_non_streaming(generation_prompt)

        # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
        simulate_stream(
            self.progress_reporter,
            conversation.model_dump_json(indent=2),
            source="conversation_gen",
        )

        # Ensure type checker knows this is a Conversation
        conversation = cast(Conversation, conversation)

        # Validate that generated conversation starts with user message
        # (system messages are added by builder, not generated by LLM)
        if conversation.messages and conversation.messages[0].role != "user":
            msg = (
                f"Generated conversation must start with 'user' message, got '{conversation.messages[0].role}'. "
                "System messages are added automatically by the builder."
            )
            raise ValueError(msg)

        # Insert system message if configured
        if self.config.sys_msg:
            conversation.messages.insert(
                0,
                ChatMessage(role="system", content=self.config.dataset_system_prompt or ""),
            )

        return conversation

    async def _complete_pinned_user(
        self, prompt: str, user_message: ChatMessage, error_feedback: str | None
    ) -> Conversation:
        user_message = pinned_user_copy(user_message)
        schema = (
            ReasonedAssistantCompletion
            if self.config.conversation_type == "cot"
            else AssistantCompletion
        )
        if error_feedback:
            prompt += f"\nRETRY: {error_feedback}"
        result = await self.llm.generate_async(
            prompt=prompt,
            schema=schema,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        reasoning = result.reasoning if isinstance(result, ReasonedAssistantCompletion) else None
        if self.config.conversation_type == "cot" and (
            reasoning is None or reasoning.style != (self.config.reasoning_style or "freetext")
        ):
            raise ValueError("Assistant completion validation error: reasoning style mismatch")
        messages = [user_message, ChatMessage(role="assistant", content=result.content)]
        if self.config.sys_msg:
            messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content=self.config.dataset_system_prompt
                    or self.config.generation_system_prompt,
                ),
            )
        conversation = Conversation(messages=messages, reasoning=reasoning)
        simulate_stream(
            self.progress_reporter,
            conversation.model_dump_json(indent=2),
            source="conversation_gen",
        )
        return conversation

    async def _generate_non_streaming(self, prompt: str) -> Conversation:
        """Generate conversation using non-streaming LLM call.

        Args:
            prompt: The complete generation prompt

        Returns:
            Generated Conversation object
        """
        return await self.llm.generate_async(
            prompt=prompt,
            schema=Conversation,
            max_retries=self.config.max_retries,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

    def _build_prompt(self, topic_prompt: str, error_feedback: str | None = None) -> str:
        """Build the generation prompt for single-shot generation.

        Args:
            topic_prompt: The topic to generate about
            error_feedback: Optional error message from a previous failed attempt

        Returns:
            Complete prompt string for the LLM
        """
        # Use the generation system prompt as the base
        sequential = self.config.topic_sample_mode == "sequential"
        prompt_parts = (
            [topic_prompt]
            if sequential
            else [self.config.generation_system_prompt, f"\nTopic/Scenario: {topic_prompt}"]
        )

        # Add error feedback if this is a retry
        if error_feedback:
            prompt_parts.append(
                f"\n\nRETRY: {error_feedback}. Use real values, not null/empty/placeholders."
            )

        # Add any additional instructions
        if self.config.instructions and not sequential:
            prompt_parts.append(f"\nAdditional Instructions: {self.config.instructions}")

        # Add reasoning-specific guidance based on style
        if self.config.conversation_type == "cot":
            if self.config.reasoning_style == "freetext":
                prompt_parts.append(
                    "\nREASONING FORMAT: Generate natural, conversational reasoning content (string format). "
                    "Show your actual thinking process - explore ideas, consider alternatives, work through the problem. "
                    "Think like a human would: 'Hmm, let me think about this...', 'Wait, that doesn't work...', "
                    "'Actually, if I approach it this way...'. "
                    "DO NOT use numbered steps or structured outlines. "
                    "Use the 'content' field in reasoning as a plain string (not a list)."
                )
            elif self.config.reasoning_style == "agent":
                prompt_parts.append(
                    "\nREASONING FORMAT: Generate structured reasoning steps as a list of ReasoningStep objects. "
                    "Each step should have clear thought and action fields."
                )

        # Add explicit structure requirement
        prompt_parts.append(
            "\nIMPORTANT: Generate the conversation messages array starting with a 'user' message "
            "(the user's question or request), followed by an 'assistant' message (the response). "
            "Do NOT include any 'system' role messages - those are added separately."
        )

        return "\n".join(prompt_parts)


class ConversationBuilderFactory:
    """Factory for creating conversation builders.

    Provides type-safe builder instantiation based on configuration.
    """

    @staticmethod
    def create(
        config: "DataSetGeneratorConfig",
        llm: "LLMClient",
        tool_registry: "ToolRegistry | None" = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> ConversationBuilder:
        """Create the appropriate conversation builder.

        Args:
            config: Generator configuration (Pydantic model)
            llm: LLM client for generation
            tool_registry: Optional tool registry (required for agent builders)
            progress_reporter: Optional progress reporter for streaming feedback

        Returns:
            Appropriate ConversationBuilder instance

        Raises:
            ValueError: If configuration is invalid or builder requirements not met
        """
        builder_type = determine_builder_type(config)

        # Validate tool registry requirement
        if builder_type.requires_tools and tool_registry is None:
            msg = (
                f"Builder type '{builder_type.name}' requires tool_registry but it was not provided"
            )
            raise ValueError(msg)

        # Instantiate appropriate builder
        if builder_type == SINGLE_SHOT_BUILDER:
            return SingleShotBuilder(llm, config, progress_reporter=progress_reporter)
        if builder_type == SINGLE_TURN_AGENT_BUILDER:
            from .builders_agent import SingleTurnAgentBuilder  # noqa: PLC0415

            return SingleTurnAgentBuilder(
                llm, config, cast("ToolRegistry", tool_registry), progress_reporter
            )
        msg = f"Unknown builder type: {builder_type.name}"
        raise ValueError(msg)
