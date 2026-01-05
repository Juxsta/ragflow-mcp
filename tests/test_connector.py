"""Tests for RAGFlowConnector class."""
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock


class TestRAGFlowConnector:
    """Test suite for RAGFlowConnector."""

    @pytest.mark.asyncio
    async def test_connector_initializes_with_correct_auth_headers(self):
        """Test 5: RAGFlowConnector initializes with correct auth headers."""
        from src.connector import RAGFlowConnector

        api_key = "test-api-key-12345"
        base_url = "http://localhost:9380/api/v1"

        async with RAGFlowConnector(api_key=api_key, base_url=base_url) as connector:
            # Verify auth header is set correctly
            assert connector.client.headers["Authorization"] == f"Bearer {api_key}"
            assert connector.base_url == base_url

    @pytest.mark.asyncio
    async def test_connector_handles_connection_errors_gracefully(self):
        """Test 6: RAGFlowConnector handles connection errors gracefully."""
        from src.connector import RAGFlowConnector, RAGFlowConnectionError

        api_key = "test-api-key-12345"
        # Use non-routable address to trigger connection error
        base_url = "http://10.255.255.1:9999/api/v1"

        async with RAGFlowConnector(api_key=api_key, base_url=base_url, timeout=1.0) as connector:
            with pytest.raises(RAGFlowConnectionError) as exc_info:
                await connector.get("/datasets")

            # Error should contain useful information
            assert "connection" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_connector_parses_error_responses(self):
        """Test that connector parses RAGFlow error responses correctly."""
        from src.connector import RAGFlowConnector, RAGFlowAPIError

        api_key = "test-api-key-12345"
        base_url = "http://localhost:9380/api/v1"

        # Mock the httpx client response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "code": 1001,
            "message": "Invalid dataset ID"
        }
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with RAGFlowConnector(api_key=api_key, base_url=base_url) as connector:
                with pytest.raises(RAGFlowAPIError) as exc_info:
                    await connector.get("/datasets/invalid-id")

                error = exc_info.value
                assert error.code == 1001
                assert "Invalid dataset ID" in str(error)

    @pytest.mark.asyncio
    async def test_connector_async_context_manager(self):
        """Test that connector properly manages client lifecycle."""
        from src.connector import RAGFlowConnector

        api_key = "test-api-key-12345"
        base_url = "http://localhost:9380/api/v1"

        connector = RAGFlowConnector(api_key=api_key, base_url=base_url)

        # Client should not exist before entering context
        assert connector.client is None

        async with connector:
            # Client should exist inside context
            assert connector.client is not None

        # Client should be closed after exiting context
        assert connector.client.is_closed
