"""Tests for configuration module."""
import os
import pytest
from unittest.mock import patch


class TestConfiguration:
    """Test suite for configuration loading."""

    def test_config_loads_from_environment_variables(self):
        """Test 1: Configuration loads from environment variables correctly."""
        env_vars = {
            "RAGFLOW_API_KEY": "test-api-key-12345",
            "RAGFLOW_URL": "http://test-server:9380/api/v1",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            from src.config import Settings

            settings = Settings()

            assert settings.ragflow_api_key == "test-api-key-12345"
            assert settings.ragflow_url == "http://test-server:9380/api/v1"
            assert settings.log_level == "DEBUG"

    def test_config_fails_when_api_key_missing(self):
        """Test 2: Configuration fails fast when RAGFLOW_API_KEY is missing."""
        env_vars = {
            "RAGFLOW_URL": "http://test-server:9380/api/v1",
            "LOG_LEVEL": "INFO",
        }
        # Remove RAGFLOW_API_KEY if it exists
        env_without_key = {k: v for k, v in os.environ.items() if k != "RAGFLOW_API_KEY"}
        env_without_key.update(env_vars)

        with patch.dict(os.environ, env_without_key, clear=True):
            from src.config import Settings

            with pytest.raises(ValueError) as exc_info:
                Settings()

            assert "RAGFLOW_API_KEY" in str(exc_info.value)

    def test_config_uses_default_values(self):
        """Test that configuration uses default values when optional vars not set."""
        env_vars = {
            "RAGFLOW_API_KEY": "test-api-key-12345",
        }
        # Remove optional vars if they exist
        env_with_only_key = {k: v for k, v in os.environ.items()
                            if k not in ("RAGFLOW_URL", "LOG_LEVEL")}
        env_with_only_key.update(env_vars)

        with patch.dict(os.environ, env_with_only_key, clear=True):
            from src.config import Settings

            settings = Settings()

            assert settings.ragflow_api_key == "test-api-key-12345"
            assert settings.ragflow_url == "http://localhost:9380/api/v1"
            assert settings.log_level == "INFO"
