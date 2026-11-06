from string import Template

from .prompt_templates import compatibility_template, default_template

SEQUENTIAL_SAMPLE_PROMPT = compatibility_template("sequential_sample", "format")


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

        return Template(default_template("tree_expansion")).substitute(
            num_subtopics=num_subtopics,
            examples=examples,
            system_prompt=system_prompt,
            path_str=path_str,
        )

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

        return Template(default_template("agent_tool_context")).substitute(
            tool_signatures="\n".join(tool_signatures), max_tools_per_query=max_tools_per_query
        )


# Simplified prompts that delegate to structured generation
AGENT_COT_TOOLS_PROMPT = compatibility_template("agent_cot_tools", "legacy")

AGENT_COT_HYBRID_PROMPT = compatibility_template("agent_cot_hybrid", "legacy")

CUSTOM_SCHEMA_PROMPT = compatibility_template("custom_schema", "legacy")

CONVERSATION_GENERATION_PROMPT = compatibility_template("conversation_generation", "legacy")

GRAPH_EXPANSION_PROMPT = compatibility_template("graph_expansion", "raw")

GRAPH_EXPANSION_PROMPT_NO_CONNECTIONS = compatibility_template(
    "graph_expansion_prompt_no_connections", "raw"
)


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

        return Template(default_template("graph_anchored")).substitute(
            num_subtopics=num_subtopics,
            examples=examples,
            system_prompt=system_prompt,
            path_str=path_str,
        )

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
FREETEXT_COT_PROMPT = compatibility_template("freetext_cot", "legacy")

STRUCTURED_COT_PROMPT = compatibility_template("structured_cot", "legacy")

HYBRID_COT_PROMPT = compatibility_template("hybrid_cot", "legacy")


BULK_FULL_SAMPLE_PROMPT = compatibility_template("bulk_full_sample", "format")


BULK_SEED_SAMPLE_PROMPT = compatibility_template("bulk_seed_sample", "format")

BULK_COMPLETION_PROMPT = compatibility_template("bulk_completion", "format")


BULK_CONVERSATION_SEED_PROMPT = compatibility_template("bulk_conversation_seed", "format")

BULK_CONVERSATION_COMPLETION_PROMPT = compatibility_template(
    "bulk_conversation_completion", "format"
)
