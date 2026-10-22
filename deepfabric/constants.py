# Default values
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_DEGREE = 3
DEFAULT_DEPTH = 2
DEFAULT_MAX_RETRIES = 3
DEFAULT_BATCH_SIZE = 5
DEFAULT_NUM_EXAMPLES = 3
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_MAX_TOKENS = 1000

# Engine defaults
ENGINE_DEFAULT_TEMPERATURE = 0.2
ENGINE_DEFAULT_BATCH_SIZE = 5
ENGINE_DEFAULT_NUM_EXAMPLES = 3

# Topic tree defaults
TOPIC_TREE_DEFAULT_DEGREE = 10
TOPIC_TREE_DEFAULT_DEPTH = 3
TOPIC_TREE_DEFAULT_TEMPERATURE = 0.2
TOPIC_TREE_DEFAULT_MODEL = "gpt-4o"

# Topic graph defaults
TOPIC_GRAPH_DEFAULT_DEGREE = 10
TOPIC_GRAPH_DEFAULT_DEPTH = 3
TOPIC_GRAPH_SUMMARY = 20
TOPIC_GRAPH_DEFAULT_MODEL = "gpt-4o"
TOPIC_GRAPH_DEFAULT_TEMPERATURE = 0.7

# File extensions and patterns
JSONL_EXTENSION = ".jsonl"
YAML_EXTENSIONS = (".yaml", ".yml")

# Message roles
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
VALID_ROLES = [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT]

# Placeholders
SYSTEM_PROMPT_VAR = "{{{{system_prompt}}}}"
INSTRUCTIONS_VAR = "{{{{instructions}}}}"
EXAMPLES_VAR = "{{{{examples}}}}"
SUBTOPICS_VAR = "{{{{subtopics}}}}"

# Retry and backoff settings
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2  # seconds
EXPONENTIAL_BACKOFF_MULTIPLIER = 2
DEFAULT_SAMPLE_RETRIES = 2  # per-sample retries for validation errors

# String length limits
MAX_ERROR_PREVIEW_LENGTH = 200
TRUNCATION_SUFFIX = "..."

# Validation patterns
JSON_BLOCK_PATTERN = r"(?s)\{.*\}"
JSON_ARRAY_PATTERN = r"\[.*\]"
JSON_CODE_BLOCK_PATTERN = r"```json\s*|\s*```"

# Default tags
DEFAULT_HF_TAGS = ["deepfabric", "synthetic"]
DEFAULT_KAGGLE_TAGS = ["deepfabric", "synthetic"]

# Error categories
ERROR_CATEGORIES = {
    "json_parsing_errors": "JSON Parsing Errors",
    "invalid_schema": "Invalid Schema",
    "api_errors": "API Errors",
    "authentication_error": "Authentication Errors",
    "empty_responses": "Empty Responses",
    "malformed_responses": "Malformed Responses",
    "other_errors": "Other Errors",
}

# API error indicators
API_ERROR_INDICATORS = ["timeout", "rate limit", "connection"]

# Special characters that need cleaning in JSON responses
JSON_SPECIAL_CHARS = "{}"

# Progress display settings
PROGRESS_BAR_DESC = "Progress"

# File save patterns
INTERRUPTED_DATASET_FILENAME = "interrupted_dataset.jsonl"
ERROR_DATASET_FILENAME = "error_dataset.jsonl"
PARTIAL_TREE_FILENAME = "partial_tree.jsonl"
FAILED_TREE_SUFFIX = "_failed.jsonl"

# Checkpoint file patterns
CHECKPOINT_METADATA_SUFFIX = ".checkpoint.json"
CHECKPOINT_SAMPLES_SUFFIX = ".checkpoint.jsonl"
CHECKPOINT_FAILURES_SUFFIX = ".checkpoint.failures.jsonl"
CHECKPOINT_VERSION = 5  # v5: exact allocation plan and per-sample ordinal tracking
# Pragmatic early warning for growing prompt cost: roughly 12k-17k tokens for
# typical English, with substantial variation by language/content. Agreed as an
# initial heuristic, not benchmark-derived or a model context limit; never trims.
SEQUENTIAL_PROMPT_WARNING_CHARS = 50_000

# Stream simulation defaults
STREAM_SIM_CHUNK_SIZE = 8  # characters per chunk
STREAM_SIM_CHUNK_DELAY_MS = 10.0  # milliseconds between chunks
