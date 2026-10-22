import json
import logging
import uuid

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .builders import ConversationBuilder, pinned_user_copy
from .constants import DEFAULT_SAMPLE_RETRIES
from .exceptions import DataSetGeneratorError
from .progress import ProgressReporter
from .schemas import (
    AgentContext,
    AgentStep,
    ChatMessage,
    Conversation,
    PendingToolCall,
    ReasoningStep,
    ReasoningTrace,
    ToolCall,
    ToolContext,
    ToolDefinition,
    ToolExecution,
    generate_tool_call_id,
)
from .spin import SpinClient, SpinSession
from .stream_simulator import simulate_stream
from .utils import is_validation_error

if TYPE_CHECKING:
    from .generator import DataSetGeneratorConfig
    from .llm import LLMClient
    from .schemas import ToolRegistry

logger = logging.getLogger(__name__)


def _convert_steps_to_reasoning(
    steps: list["AgentStep"],
    final_action_text: str = "Ready to respond",
) -> list[ReasoningStep]:
    """Convert AgentStep objects to ReasoningStep objects.

    Args:
        steps: List of AgentStep objects to convert
        final_action_text: Text to use for the action when step is final

    Returns:
        List of ReasoningStep objects
    """
    result = []
    for i, step in enumerate(steps, 1):
        action = None
        if step.tool_calls:
            actions = [f"{tc.function_name}({tc.arguments})" for tc in step.tool_calls]
            action = "; ".join(actions)
        elif step.is_final:
            action = final_action_text
        result.append(
            ReasoningStep(
                step_number=i,
                thought=step.thought,
                action=action,
            )
        )
    return result


class UserQuestion(BaseModel):
    """User's question or request."""

    content: str = Field(
        description="The user's question or request text - just the question itself, nothing else",
        min_length=10,
    )


class AgentResponse(BaseModel):
    """Agent's response to user."""

    content: str = Field(
        description="The agent's response text - clear and concise",
        min_length=10,
    )


class ToolOutput(BaseModel):
    """Simulated tool execution output."""

    result: str = Field(description="The tool's output/result", min_length=1)


class SingleTurnAgentBuilder(ConversationBuilder):
    """Builder for single-turn agent conversations with tool calling.

    Generates conversations using a multi-step process:
    1. Generate user question
    2. Generate agent reasoning + tool calls
    3. Execute tools via Spin (or simulate if no Spin endpoint)
    4. Generate agent's final response

    This produces realistic tool-calling training data.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "DataSetGeneratorConfig",
        tool_registry: "ToolRegistry",
        progress_reporter: ProgressReporter | None = None,
    ):
        """Initialize with required tool registry.

        Args:
            llm: LLM client for generation
            config: Generator configuration
            tool_registry: Tool registry (required for agent builders)
            progress_reporter: Optional progress reporter for streaming feedback
        """
        super().__init__(llm, config, tool_registry, progress_reporter)
        # Store as non-optional for type checker
        self.tool_registry: ToolRegistry = tool_registry

        # Spin integration for real tool execution
        self._spin_client: SpinClient | None = None
        self._spin_session: SpinSession | None = None

        # Track seen tool signatures to skip duplicates
        self._seen_tool_signatures: set[str] = set()
        self._completion_context: str | None = None

        # Initialize Spin client if endpoint is configured
        spin_endpoint = getattr(config, "spin_endpoint", None)
        tool_execute_path = getattr(config, "tool_execute_path", None)
        if spin_endpoint:
            self._spin_client = SpinClient(
                endpoint=spin_endpoint,
                tool_execute_path=tool_execute_path,
            )
            if tool_execute_path:
                logger.info(
                    "Spin execution enabled: %s (execute path: %s)",
                    spin_endpoint,
                    tool_execute_path,
                )
            else:
                logger.info("Spin execution enabled: %s", spin_endpoint)

    async def generate(
        self,
        topic_prompt: str,
        error_feedback: str | None = None,
        *,
        pinned_user_message: ChatMessage | None = None,
    ) -> Conversation:
        """Generate single-turn agent conversation with tools using ReAct loop.

        Uses a think-act-observe loop where each step's tool calls are based on
        observations from previous steps. This prevents the agent from making
        decisions (like writes) before observing results (like reads).

        Args:
            topic_prompt: Topic or scenario to generate conversation about
            error_feedback: Optional error message from a previous failed attempt

        Returns:
            Complete Conversation with tool calling

        Raises:
            ValueError: If generation fails at any step
        """
        self._completion_context = (
            topic_prompt
            if pinned_user_message is not None or self.config.topic_sample_mode == "sequential"
            else None
        )
        try:
            # Initialize Spin session if configured
            await self._ensure_spin_session()

            # Step 1: Generate user question
            user_message = (
                pinned_user_copy(pinned_user_message)
                if pinned_user_message is not None
                else await self._generate_user_question(topic_prompt)
            )

            # Step 2: ReAct loop - think, act, observe
            all_steps: list[AgentStep] = []
            all_tool_results: list[ToolExecution] = []
            max_steps = getattr(self.config, "max_agent_steps", 5)

            # Reset duplicate tracking for this conversation
            self._seen_tool_signatures.clear()

            for step_num in range(max_steps):
                if self.progress_reporter:
                    self.progress_reporter.emit_step_start(f"ReAct step {step_num + 1}/{max_steps}")

                # Generate next step based on observations so far
                step = await self._generate_next_step(
                    user_message,
                    all_steps,
                    all_tool_results,
                    error_feedback if step_num == 0 else None,
                )
                all_steps.append(step)

                # Check if agent is done
                if step.is_final or not step.tool_calls:
                    if self.progress_reporter:
                        self.progress_reporter.emit_step_complete(
                            f"Agent decided to conclude after {step_num + 1} steps"
                        )
                    break

                # Execute THIS step's tools
                step_results = await self._execute_step_tools(step.tool_calls)
                all_tool_results.extend(step_results)

                if self.progress_reporter:
                    self.progress_reporter.emit_step_complete(
                        f"Executed {len(step.tool_calls)} tool(s) in step {step_num + 1}"
                    )

            # Step 3: Generate agent's final response based on all observations
            agent_response = await self._generate_agent_conclusion(
                user_message, all_steps, all_tool_results
            )

            # Assemble into Conversation
            return self._build_conversation(
                user_message, all_steps, all_tool_results, agent_response, topic_prompt
            )
        finally:
            # Always clean up the session and request-scoped completion framing.
            try:
                await self._cleanup_spin_session()
            finally:
                self._completion_context = None

    async def _ensure_spin_session(self) -> None:
        """Initialize Spin session if configured."""
        if self._spin_client is None:
            return

        # Create new session
        session_id = str(uuid.uuid4())
        self._spin_session = SpinSession(self._spin_client, session_id)

        # Seed initial state if configured
        scenario_seed = getattr(self.config, "scenario_seed", None)
        if scenario_seed and isinstance(scenario_seed, dict):
            files = scenario_seed.get("files", {})
            if files:
                success = await self._spin_session.seed_files(files)
                if success:
                    logger.debug("Seeded %d files for session %s", len(files), session_id)
                else:
                    logger.warning("Failed to seed some files for session %s", session_id)

    async def _cleanup_spin_session(self) -> None:
        """Clean up Spin session after generation."""
        if self._spin_session is not None:
            await self._spin_session.cleanup()
            self._spin_session = None

    async def _generate_user_question(self, topic_prompt: str) -> ChatMessage:
        """Generate the user's question for this scenario.

        Args:
            topic_prompt: The scenario topic

        Returns:
            User message (typed ChatMessage)
        """
        prompt = f"""Generate a short, natural user question for this scenario:
{topic_prompt}

Requirements:
- Just the user's question - no reasoning, no explanations, no examples
- Should require using tools to answer
- 1-2 sentences maximum
- Natural, conversational tone

Example format: "Can you tell me the weather in Paris tomorrow and suggest what to wear?"

Generate only the user's question:"""

        # Always use non-streaming for reliable structured output
        response = await self.llm.generate_async(
            prompt=prompt,
            schema=UserQuestion,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
        simulate_stream(
            self.progress_reporter,
            response.model_dump_json(),
            source="user_question",
        )

        return ChatMessage(role="user", content=response.content)

    async def _generate_next_step(
        self,
        user_message: ChatMessage,
        previous_steps: list[AgentStep],
        previous_results: list[ToolExecution],
        error_feedback: str | None = None,
    ) -> AgentStep:
        """Generate the next ReAct step based on observations so far.

        This is the core of the ReAct loop - the agent decides its next action
        based on what it has already observed from previous tool executions.

        Args:
            user_message: The original user question
            previous_steps: Steps taken so far (thoughts + tool calls)
            previous_results: Results from executed tools
            error_feedback: Optional error from previous generation attempt

        Returns:
            AgentStep with thought, optional tool calls, and is_final flag
        """
        # Coordinated generation has one outer slot budget; do not multiply it here.
        max_retries = (
            0
            if self._completion_context is not None
            else getattr(self.config, "sample_retries", DEFAULT_SAMPLE_RETRIES)
        )
        last_error: Exception | None = None
        current_feedback = error_feedback

        for attempt in range(max_retries + 1):
            try:
                return await self._generate_next_step_impl(
                    user_message, previous_steps, previous_results, current_feedback
                )
            except Exception as e:
                last_error = e
                if is_validation_error(e) and attempt < max_retries:
                    current_feedback = str(e)
                    if self.progress_reporter:
                        self.progress_reporter.emit_step_start(
                            f"Retrying step generation (attempt {attempt + 2}/{max_retries + 1})"
                        )
                    continue
                raise

        raise last_error  # type: ignore[misc]

    async def _generate_next_step_impl(
        self,
        user_message: ChatMessage,
        previous_steps: list[AgentStep],
        previous_results: list[ToolExecution],
        error_feedback: str | None = None,
    ) -> AgentStep:
        """Implementation of next step generation."""
        tools_info = self._format_tools_for_prompt()
        history = self._format_step_history(previous_steps, previous_results)

        prompt_parts = [
            "## User Request",
            user_message.content or "",
            "",
            "## Available Tools",
            tools_info,
            "",
            "## Previous Actions & Results",
            history if history else "None yet - this is your first action.",
            "",
            "## Instructions",
            "Based on what you've observed so far, decide your next action:",
            "- If you need more information, specify tool_calls for THIS step only",
            "- If you have enough information to answer, set is_final=true and leave tool_calls empty",
            "- IMPORTANT: Do NOT call write/modify operations until you've confirmed current state via read operations",
            "- Tool arguments must use concrete values (no placeholders like '<user_input>' or null)",
            "",
            "What is your next step?",
        ]

        if self._completion_context is not None:
            prompt_parts.insert(0, f"## Completion Context\n{self._completion_context}\n")
        if error_feedback:
            prompt_parts.insert(
                -1,
                f"\n## Previous Attempt Failed\nError: {error_feedback}\nPlease fix this issue.\n",
            )

        prompt = "\n".join(prompt_parts)

        # Always use non-streaming for reliable structured output
        response = await self.llm.generate_async(
            prompt=prompt,
            schema=AgentStep,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
        simulate_stream(
            self.progress_reporter,
            response.model_dump_json(),
            source="agent_step",
        )

        return response

    def _format_step_history(self, steps: list[AgentStep], results: list[ToolExecution]) -> str:
        """Format previous steps and results for the prompt.

        Args:
            steps: Previous AgentSteps taken
            results: Tool execution results (with actual outputs)

        Returns:
            Formatted string showing the progression of actions and observations
        """
        if not steps:
            return ""

        history_parts = []
        result_idx = 0

        for step_num, step in enumerate(steps, 1):
            history_parts.append(f"### Step {step_num}")
            history_parts.append(f"Thought: {step.thought}")

            if step.tool_calls:
                history_parts.append("Tool calls:")
                for tool_call in step.tool_calls:
                    history_parts.append(f"  - {tool_call.function_name}({tool_call.arguments})")
                    # Match with result if available
                    if result_idx < len(results):
                        result = results[result_idx]
                        history_parts.append(f"    Result: {result.result}")
                        result_idx += 1

            history_parts.append("")

        return "\n".join(history_parts)

    def _get_tool_signature(self, pending_call: PendingToolCall) -> str:
        """Generate a signature for deduplication.

        Args:
            pending_call: The pending tool call

        Returns:
            Signature string combining tool name and arguments
        """
        try:
            # Normalize arguments by parsing and re-dumping JSON to handle
            # differences in whitespace and key order.
            args = json.loads(pending_call.arguments)
            normalized_args = json.dumps(args, sort_keys=True)
            return f"{pending_call.function_name}:{normalized_args}"  # noqa: TRY300
        except json.JSONDecodeError:
            # Fallback for any case where arguments are not valid JSON,
            # though this should be caught by Pydantic validation.
            return f"{pending_call.function_name}:{pending_call.arguments}"

    async def _execute_step_tools(self, tool_calls: list[PendingToolCall]) -> list[ToolExecution]:
        """Execute tool calls for a single ReAct step.

        Skips duplicate tool calls (same tool + same arguments) that were
        already executed in a previous step of this conversation.

        Args:
            tool_calls: Pending tool calls from the current step (without results)

        Returns:
            List of ToolExecutions with results populated from Spin
        """
        completed_executions = []

        for pending_call in tool_calls:
            # Check for duplicate
            signature = self._get_tool_signature(pending_call)
            if signature in self._seen_tool_signatures:
                logger.debug("Skipping duplicate tool call: %s", pending_call.function_name)
                continue

            # Mark as seen
            self._seen_tool_signatures.add(signature)

            # Get tool definition from registry
            tool_def = self.tool_registry.get_tool(pending_call.function_name)
            if not tool_def:
                # Return error as result instead of raising
                completed_executions.append(
                    pending_call.to_tool_execution(
                        f"Error: Tool '{pending_call.function_name}' not found in registry"
                    )
                )
                continue

            # Execute tool via Spin
            result = await self._generate_tool_output(tool_def, pending_call)
            completed_executions.append(pending_call.to_tool_execution(result.result))

        return completed_executions

    async def _generate_tool_output(
        self,
        tool_def: ToolDefinition,
        pending_call: PendingToolCall,
        error_feedback: str | None = None,  # noqa: ARG002 - kept for interface compatibility
    ) -> ToolOutput:
        """Execute tool via Spin and return real output.

        Args:
            tool_def: Tool definition from registry
            pending_call: Pending tool call with arguments (no result yet)
            error_feedback: Unused - kept for interface compatibility with retry logic

        Returns:
            ToolOutput with real execution result

        Raises:
            DataSetGeneratorError: If Spin is not configured or execution fails
        """
        # Require Spin for tool execution
        if self._spin_session is None:
            raise DataSetGeneratorError(
                "Spin endpoint not configured. Tool execution requires a Spin service. "
                "Set 'spin_endpoint' in your tools configuration."
            )

        # Parse arguments from JSON string
        try:
            args: dict[str, Any] = json.loads(pending_call.arguments)
        except json.JSONDecodeError as e:
            return ToolOutput(result=f"Error: Invalid JSON arguments: {e}")

        # Execute via Spin
        result = await self._spin_session.execute_tool(
            tool_name=tool_def.name,
            arguments=args,
            component=tool_def.component,
        )

        if result.success:
            if self.progress_reporter:
                self.progress_reporter.emit_tool_execution(
                    tool_def.name, success=True, arguments=args
                )
            return ToolOutput(result=result.result)

        # Return error as tool output (this is valid training data for error handling)
        error_msg = result.result
        if result.error_type:
            error_msg = f"Error ({result.error_type}): {result.result}"
        if self.progress_reporter:
            self.progress_reporter.emit_tool_execution(
                tool_def.name,
                success=False,
                arguments=args,
                error_type=result.error_type or "error",
            )
        return ToolOutput(result=error_msg)

    async def _generate_agent_conclusion(
        self,
        user_message: ChatMessage,
        steps: list[AgentStep],  # noqa: ARG002 - kept for potential future use
        tool_results: list[ToolExecution],
        context: str = "",
    ) -> ChatMessage:
        """Generate agent's final response interpreting tool results.

        Args:
            user_message: Original user question
            steps: All ReAct steps taken (with thoughts)
            tool_results: All tool execution results
            context: Previous conversation context (for multi-turn)

        Returns:
            Agent's final response message
        """
        # Format tool results summary
        results_text = (
            "\n".join([f"Tool: {r.function_name}\nResult: {r.result}" for r in tool_results])
            if tool_results
            else "No tools were executed."
        )

        # Format available tools for context
        tools_info = self._format_tools_for_prompt()

        # Build context section if provided
        context_section = ""
        if context:
            context_section = f"Previous conversation:\n{context}\n\n"

        framing = (
            self._completion_context
            if self._completion_context is not None
            else self.config.dataset_system_prompt
        )
        prompt = f"""{framing}

Available tools:
{tools_info}

{context_section}User request: {user_message.content}

You executed these tools:
{results_text}

Based on these results, provide a clear, helpful response to the user.
Remember: You have access to the tools listed above and have used them in this conversation."""

        # Always use non-streaming for reliable structured output
        response = await self.llm.generate_async(
            prompt=prompt,
            schema=AgentResponse,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
        simulate_stream(
            self.progress_reporter,
            response.model_dump_json(),
            source="agent_response",
        )

        return ChatMessage(role="assistant", content=response.content)

    def _build_conversation(
        self,
        user_message: ChatMessage,
        steps: list[AgentStep],
        tool_results: list[ToolExecution],
        agent_response: ChatMessage,
        _topic_prompt: str = "",
    ) -> Conversation:
        """Assemble all components into a Conversation.

        Preserves ReAct step-by-step structure: each step's tool calls become
        a separate assistant message followed by tool responses. This ensures
        training data shows the agent making decisions AFTER observing results.

        Args:
            user_message: User's question
            steps: All ReAct steps (thoughts + tool calls)
            tool_results: All tool execution results
            agent_response: Agent's final response
            _topic_prompt: Topic used to generate this conversation (unused, for interface)

        Returns:
            Complete Conversation object
        """
        messages = []

        # Add user message
        messages.append(user_message)

        # Process each ReAct step separately to preserve step-by-step structure
        # This is critical: agent should see results from step N before deciding step N+1
        result_idx = 0

        for step in steps:
            # Skip steps with no tool calls (e.g., final "is_final=true" step)
            if not step.tool_calls:
                continue

            # Build tool_calls for THIS step only
            step_tool_calls: list[ToolCall] = []
            step_tool_call_ids: list[str] = []

            for _pending_call in step.tool_calls:
                tool_call_id = generate_tool_call_id()
                step_tool_call_ids.append(tool_call_id)
                # Get the matching result
                if result_idx < len(tool_results):
                    result = tool_results[result_idx]
                    step_tool_calls.append(result.to_tool_call(tool_call_id))
                    result_idx += 1

            # Assistant message with tool_calls for this step
            if step_tool_calls:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=step_tool_calls,
                    )
                )

                # Tool response messages for this step
                # We need to re-iterate to get matching results
                result_base_idx = result_idx - len(step_tool_calls)
                for idx, _tc in enumerate(step_tool_calls):
                    res_idx = result_base_idx + idx
                    if res_idx < len(tool_results):
                        messages.append(
                            ChatMessage(
                                role="tool",
                                content=tool_results[res_idx].result,
                                tool_call_id=step_tool_call_ids[idx],
                            )
                        )

        # Add final assistant response with the answer
        messages.append(agent_response)

        # Build tool context (executions only - tools are in top-level 'tools' field)
        tool_context = ToolContext(
            executions=tool_results,
        )

        # Build reasoning trace from AgentSteps
        reasoning_steps = _convert_steps_to_reasoning(steps, "Ready to respond to user")

        reasoning_trace = ReasoningTrace(
            style=self.config.reasoning_style or "agent",  # type: ignore
            content=(
                "\n".join(step.thought for step in reasoning_steps)
                if self._completion_context is not None
                and self.config.reasoning_style == "freetext"
                else reasoning_steps
            ),
        )

        # Build agent context
        agent_context = AgentContext(mode="single_turn")

        # Build metadata
        metadata = {
            "conversation_type": "cot",
            "react_steps": len(steps),
        }

        # Insert system message if configured
        self._insert_system_message_if_configured(messages)

        # Convert tools to OpenAI format, filtering based on inclusion strategy
        if self.config.tool_inclusion_strategy == "used_only" and tool_results:
            used_names = {te.function_name for te in tool_results}
            tools_openai = [
                tool.to_openai() for tool in self.tool_registry.tools if tool.name in used_names
            ]
        else:
            tools_openai = [tool.to_openai() for tool in self.tool_registry.tools]

        return Conversation(
            messages=messages,
            reasoning=reasoning_trace,
            tool_context=tool_context,
            tools=tools_openai,
            agent_context=agent_context,
            question=user_message.content or "",
            final_answer=agent_response.content or "",
            metadata=metadata,
        )

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for inclusion in prompts.

        Provides detailed tool information including parameter descriptions
        and whether parameters are required, helping the LLM generate
        correct tool calls.

        Returns:
            Formatted string describing available tools
        """
        tool_descriptions = []
        for tool in self.tool_registry.tools:
            # Build parameter details
            if tool.parameters:
                param_lines = []
                for p in tool.parameters:
                    req_marker = "(required)" if p.required else "(optional)"
                    param_lines.append(f"    - {p.name}: {p.type} {req_marker} - {p.description}")
                params_section = "\n".join(param_lines)
                tool_descriptions.append(
                    f"### {tool.name}\n"
                    f"{tool.description}\n"
                    f"Parameters:\n{params_section}\n"
                    f"Returns: {tool.returns}"
                )
            else:
                # No parameters - make this explicit
                tool_descriptions.append(
                    f"### {tool.name}\n"
                    f"{tool.description}\n"
                    f"Parameters: None (use empty object {{}})\n"
                    f"Returns: {tool.returns}"
                )

        return "\n\n".join(tool_descriptions)

    def _insert_system_message_if_configured(self, messages: list[ChatMessage]) -> None:
        """Insert system message at the beginning of messages if configured.

        Args:
            messages: List of messages to potentially prepend system message to
        """
        if self.config.sys_msg:
            messages.insert(
                0,
                ChatMessage(role="system", content=self.config.dataset_system_prompt or ""),
            )
