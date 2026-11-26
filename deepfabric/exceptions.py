class DeepFabricError(Exception):
    """Base exception class for DeepFabric."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ConfigurationError(DeepFabricError):
    """Raised when there is an error in configuration."""

    pass


class ValidationError(DeepFabricError):
    """Raised when data validation fails."""

    pass


class ToolValidationError(ValidationError):
    """Raised when a conversation violates tool-execution acceptance rules."""


class ModelError(DeepFabricError):
    """Raised when there is an error with LLM model operations."""

    pass


class TreeError(DeepFabricError):
    """Raised when there is an error in topic tree operations."""

    pass


class DataSetGeneratorError(DeepFabricError):
    """Raised when there is an error in data engine operations."""

    pass


class DatasetError(DeepFabricError):
    """Raised when there is an error in dataset operations."""

    pass


class HubUploadError(DeepFabricError):
    """Raised when there is an error uploading to Hugging Face Hub."""

    pass


class JSONParsingError(ValidationError):
    """Raised when JSON parsing fails."""

    pass


class APIError(ModelError):
    """Raised when API calls fail."""

    pass


class RetryExhaustedError(ModelError):
    """Raised when maximum retries are exceeded."""

    pass


class LoaderError(DeepFabricError):
    """Raised when dataset loading fails.

    Common causes:
    - File not found
    - Invalid file format (malformed JSON/JSONL)
    - Cloud authentication failure
    - Network errors
    - Empty dataset
    """

    pass
