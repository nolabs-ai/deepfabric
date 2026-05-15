class TreePromptBuilder:
    """Build dynamic prompts for topic tree expansion with domain-specific examples."""

    # Domain-specific expansion examples
    EXAMPLES = {
        "general": [
            {
                "path": ["Technology", "Artificial Intelligence"],
                "subtopics": [
                    "machine learning",
                    "neural networks",
                    "computer vision",
                    "natural language processing",
                    "robotics",
                ],
            },
            {
                "path": ["Entertainment", "Movies", "Actors"],
                "subtopics": [
                    "Tom Hanks",
                    "Meryl Streep",
                    "Leonardo DiCaprio",
                    "Jennifer Lawrence",
                    "Denzel Washington",
                ],
            },
        ],
        "conversational": [
            {
                "path": ["Small Talk Topics"],
                "subtopics": [
                    "weather",
                    "weekend plans",
                    "hobbies",
                    "family",
                    "books",
                    "food",
                    "music",
                ],
            },
            {
                "path": ["Small Talk Topics", "Family"],
                "subtopics": [
                    "parents",
                    "grandparents",
                    "siblings",
                    "family traditions",
                    "family vacations",
                ],
            },
            {
                "path": ["Small Talk Topics", "Hobbies", "Cooking"],
                "subtopics": [
                    "recipes",
                    "asian food",
                    "favourite dishes",
                    "cookbooks",
                    "kitchen gadgets",
                    "vegan cooking",
                ],
            },
        ],
        "technical": [
            {
                "path": ["Programming"],
                "subtopics": [
                    "algorithms",
                    "data structures",
                    "debugging",
                    "testing",
                    "version control",
                ],
            },
            {
                "path": ["Programming", "Python"],
                "subtopics": ["pandas", "flask", "pytest", "asyncio", "django"],
            },
        ],
        "educational": [
            {
                "path": ["Mathematics"],
                "subtopics": ["algebra", "geometry", "calculus", "statistics", "probability"],
            },
            {
                "path": ["Mathematics", "Algebra"],
                "subtopics": [
                    "linear equations",
                    "quadratic functions",
                    "polynomials",
                    "matrices",
                    "systems",
                ],
            },
        ],
    }

    @classmethod
    def build_expansion_prompt(
        cls,
        topic_path: list[str],
        num_subtopics: int,
        system_prompt: str = "",
        domain: str = "general",
    ) -> str:
        """Build a topic expansion prompt with relevant examples."""

        path_str = " -> ".join(f'"{topic}"' for topic in topic_path)
        examples = cls._format_examples(cls.EXAMPLES.get(domain, cls.EXAMPLES["general"]))

        return f"""Generate {num_subtopics} subtopics for training data organization.

Task: Create diverse but related subtopics that expand on the given topic path.

Examples:
{examples}

Context: {system_prompt}

Topic path: {path_str}
Generate {num_subtopics} subtopics as a Python list. Return only the list, nothing else."""

    @classmethod
    def _format_examples(cls, examples: list) -> str:
        """Format examples for inclusion in prompt."""
        formatted = []
        for ex in examples[:3]:  # Limit to 3 examples
            path_str = " -> ".join(f'"{topic}"' for topic in ex["path"])
            subtopics_str = str(ex["subtopics"])
            formatted.append(f"Path: {path_str}\nSubtopics: {subtopics_str}")
        return "\n\n".join(formatted)


# Structured Agent Tool-Calling Prompt Builder
class AgentPromptBuilder:
    """Build structured prompts for agent tool-calling training."""

    @staticmethod
    def build_tool_context_prompt(tool_registry, max_tools_per_query: int = 3) -> str:
        """Build a minimal context prompt that relies on structured generation.

        Returns a template with {{{{instructions}}}} and {{{{subtopics}}}} placeholders
        that will be filled in by build_prompt() with actual topic paths from the tree.
        """
        tool_signatures = []
        for tool in tool_registry.tools:
            tool_signatures.append(f"- {tool.to_signature()}")

        return f"""Generate a realistic agent training example with tool usage reasoning.

Available tools:
{chr(10).join(tool_signatures)}

You may use 1 to {max_tools_per_query} tools to complete the task.

Focus on WHY each tool is selected and HOW parameters are constructed.

ARGUMENT REQUIREMENTS:
- All argument values must be concrete and realistic (e.g., owner="acme-corp", repo="web-app", issue_number=42)
- Never use template placeholders like {{{{owner}}}} or {{{{repo}}}}
- Never use null values - omit optional parameters entirely if not needed
- String fields must contain actual content, not empty strings

{{{{{{{{instructions}}}}}}}}
{{{{{{{{subtopics}}}}}}}}

Generate a complete agent reasoning example using structured output with tool_executions list."""


# Simplified prompts that delegate to structured generation
AGENT_COT_TOOLS_PROMPT = """Generate an agent tool-calling training example using the available tool definitions.

You may use multiple tools (up to the specified limit) to complete the task.

Focus on the reasoning process: WHY each tool is selected, HOW parameters are constructed, and WHAT results are expected.

Create realistic scenarios that teach proper tool reasoning patterns and multi-tool orchestration.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

AGENT_COT_HYBRID_PROMPT = """Generate agent tool-calling examples with rich CoT reasoning traces and tool execution.

You may use multiple tools (up to the specified limit) to complete the task.

Combine natural language reasoning with structured step-by-step traces that include:
- Chain of thought analysis
- Structured reasoning steps with thoughts and actions
- Clear tool selection and parameter reasoning
- Multiple tool executions with results

Focus on teaching both the reasoning process AND multi-tool usage patterns.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

CUSTOM_SCHEMA_PROMPT = """{{{{system_prompt}}}}
{{{{instructions}}}}
{{{{subtopics}}}}

Generate a single JSON object for the topic above. Fill in all required fields accurately and completely."""

CONVERSATION_GENERATION_PROMPT = """Generate a training conversation for a language model with this system prompt:

<system_prompt>
{{{{system_prompt}}}}
</system_prompt>

Create a realistic single q&a that demonstrates the system's capabilities. The conversation should:
- Start with a user question/request
- Have the assistant respond helpfully according to the system prompt
- Be natural and educational

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}

Generate one training sample as question and answer."""

GRAPH_EXPANSION_PROMPT = """
You are an expert in knowledge graph generation. Your task is to expand a topic into a set of subtopics. For each subtopic, you should also identify if it connects to any other existing topics in the graph.

Here is the current state of the graph:
{{current_graph_summary}}

You are expanding the topic: "{{current_topic}}"

Generate a list of {{num_subtopics}} subtopics. For each subtopic, provide:
1. A "topic" string - the name of the new subtopic
2. A "connections" list of IDs of existing topics it should connect to for creating cross-links (use empty list if no connections)
"""

GRAPH_EXPANSION_PROMPT_NO_CONNECTIONS = """
You are an expert in topic generation. Your task is to expand a topic into a set of focused subtopics.

You are expanding the topic: "{{current_topic}}"

Generate a list of {{num_subtopics}} subtopics. For each subtopic, provide:
1. A "topic" string - the name of the new subtopic
2. A "connections" list - ALWAYS use an empty list []

IMPORTANT: Do NOT create cross-connections between topics. Each subtopic should be independent and directly related only to its parent topic. Always return connections as an empty list [].
"""


class GraphPromptBuilder:
    """Build domain-aware prompts for graph topic expansion with anchoring examples."""

    MAX_PROMPT_EXAMPLES = 3

    SECURITY_KEYWORDS = frozenset(
        {
            "security",
            "attack",
            "credential",
            "exfiltration",
            "injection",
            "malicious",
            "adversarial",
            "threat",
        }
    )

    # Domain-specific expansion examples - formatted to match GraphSubtopics schema
    EXAMPLES = {
        "security": [
            {
                "path": ["Security Threats", "Credential Access"],
                "subtopics": [
                    {"topic": "reading .env files", "connections": []},
                    {"topic": "extracting API keys", "connections": []},
                    {"topic": "accessing SSH keys", "connections": []},
                    {"topic": "dumping AWS credentials", "connections": []},
                    {"topic": "stealing database passwords", "connections": []},
                ],
            },
            {
                "path": ["Security Threats", "Data Exfiltration"],
                "subtopics": [
                    {"topic": "sending to webhooks", "connections": []},
                    {"topic": "encoding in base64", "connections": []},
                    {"topic": "uploading to external URLs", "connections": []},
                    {"topic": "email forwarding", "connections": []},
                    {"topic": "DNS tunneling", "connections": []},
                ],
            },
        ],
        "technical": [
            {
                "path": ["Programming", "Python"],
                "subtopics": [
                    {"topic": "pandas", "connections": []},
                    {"topic": "flask", "connections": []},
                    {"topic": "pytest", "connections": []},
                    {"topic": "asyncio", "connections": []},
                    {"topic": "django", "connections": []},
                ],
            },
            {
                "path": ["Infrastructure", "Kubernetes"],
                "subtopics": [
                    {"topic": "pods", "connections": []},
                    {"topic": "deployments", "connections": []},
                    {"topic": "services", "connections": []},
                    {"topic": "ingress", "connections": []},
                    {"topic": "helm charts", "connections": []},
                ],
            },
        ],
    }

    @classmethod
    def build_anchored_prompt(
        cls,
        topic_path: list[str],
        num_subtopics: int,
        system_prompt: str = "",
        domain: str = "technical",
    ) -> str:
        """Build a domain-anchored prompt for graph expansion.

        Returns a prompt that produces focused, on-topic subtopics by providing
        domain-specific examples and the full topic path context.
        """
        path_str = " -> ".join(f'"{topic}"' for topic in topic_path)
        examples = cls._format_examples(cls.EXAMPLES.get(domain, cls.EXAMPLES["technical"]))

        return f"""Generate {num_subtopics} subtopics for training data organization.

Task: Create diverse but related subtopics that expand on the given topic path.

Examples:
{examples}

Context: {system_prompt}

Topic path: {path_str}

Generate {num_subtopics} subtopics. For each subtopic, provide:
1. A "topic" string - a specific, concrete subtopic directly related to the parent
2. A "connections" list - ALWAYS use an empty list []

Return focused subtopics that stay on-topic with the path above."""

    @classmethod
    def _format_examples(cls, examples: list) -> str:
        """Format examples for inclusion in prompt."""
        formatted = []
        for ex in examples[: cls.MAX_PROMPT_EXAMPLES]:
            path_str = " -> ".join(f'"{topic}"' for topic in ex["path"])
            subtopics_str = str(ex["subtopics"])
            formatted.append(f"Path: {path_str}\nSubtopics: {subtopics_str}")
        return "\n\n".join(formatted)

    @classmethod
    def detect_domain(cls, system_prompt: str, topic_path: list[str]) -> str:
        """Detect the appropriate domain for prompt examples based on context.

        Returns 'security' or 'technical' based on keywords in the system prompt
        and topic path. Defaults to 'technical' if no security keywords found.
        """
        combined_text = f"{system_prompt} {' '.join(topic_path)}".lower()

        if any(word in combined_text for word in cls.SECURITY_KEYWORDS):
            return "security"
        return "technical"


# Chain of Thought prompts for reasoning-based dataset generation
FREETEXT_COT_PROMPT = """Generate a reasoning problem that requires analytical thinking to solve.

Create problems involving mathematics, logic, science, or analytical reasoning that can be solved through clear thinking steps.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

STRUCTURED_COT_PROMPT = """Generate a training conversation that demonstrates systematic problem-solving.

Create realistic dialogues where complex problems are solved through methodical reasoning.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

HYBRID_COT_PROMPT = """Generate problems that require analytical and systematic thinking.

Create challenging reasoning problems suitable for training systematic problem-solving skills.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""
