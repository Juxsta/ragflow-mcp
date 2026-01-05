"""Configuration module for RAGFlow MCP Server.

Loads settings from environment variables with validation.
Fails fast with clear error if required RAGFLOW_API_KEY is missing.
"""
import os
from typing import Literal

from pydantic import BaseModel, field_validator


class Settings(BaseModel):
    """Application settings loaded from environment variables.

    Attributes:
        ragflow_api_key: Required API key for RAGFlow authentication.
        ragflow_url: Base URL for RAGFlow API (default: http://localhost:9380/api/v1).
        log_level: Logging level (default: INFO).
    """

    ragflow_api_key: str
    ragflow_url: str = "http://localhost:9380/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def __init__(self, **data):
        """Initialize settings from environment variables."""
        # Load from environment if not provided
        if "ragflow_api_key" not in data:
            api_key = os.environ.get("RAGFLOW_API_KEY")
            if not api_key:
                raise ValueError(
                    "RAGFLOW_API_KEY environment variable is required but not set. "
                    "Please set it with your RAGFlow API key."
                )
            data["ragflow_api_key"] = api_key

        if "ragflow_url" not in data:
            data["ragflow_url"] = os.environ.get(
                "RAGFLOW_URL", "http://localhost:9380/api/v1"
            )

        if "log_level" not in data:
            data["log_level"] = os.environ.get("LOG_LEVEL", "INFO").upper()

        super().__init__(**data)

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        if isinstance(v, str):
            v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            return "INFO"
        return v

    model_config = {"frozen": True}


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings instance loaded from environment variables.

    Raises:
        ValueError: If RAGFLOW_API_KEY is not set.
    """
    return Settings()
