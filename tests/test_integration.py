"""Integration tests for RAGFlow MCP Server.

These tests verify end-to-end workflows spanning multiple tools
and critical error handling scenarios.
"""
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


class TestIntegrationWorkflows:
    """Integration tests for multi-stage workflows."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for integration testing."""
        from src.connector import RAGFlowConnector
        connector = MagicMock(spec=RAGFlowConnector)
        connector.cache = MagicMock()
        connector.invalidate_cache = MagicMock()
        return connector

    @pytest.mark.asyncio
    async def test_full_workflow_dataset_creation_to_retrieval(self, mock_connector):
        """Integration test: Full workflow from dataset creation to retrieval.

        Tests the complete flow:
        1. Create a dataset
        2. Upload a document to the dataset
        3. Parse the document
        4. Perform retrieval on the dataset
        """
        from src.tools.datasets import ragflow_create_dataset
        from src.tools.documents import ragflow_upload_document, ragflow_parse_document_sync
        from src.tools.retrieval import ragflow_retrieval

        # Step 1: Create dataset
        mock_connector.create_dataset = AsyncMock(return_value={
            "id": "dataset-integration-001",
            "name": "Integration Test Dataset",
            "description": "Dataset for integration testing",
        })

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            dataset = await ragflow_create_dataset(
                name="Integration Test Dataset",
                description="Dataset for integration testing",
            )
        assert dataset["id"] == "dataset-integration-001"

        # Step 2: Upload document to the dataset
        mock_connector.upload_document = AsyncMock(return_value={
            "id": "doc-integration-001",
            "name": "test_doc.txt",
            "status": "pending",
        })

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            document = await ragflow_upload_document(
                dataset_id=dataset["id"],
                base64_content="VGVzdCBkb2N1bWVudCBjb250ZW50",  # "Test document content"
                filename="test_doc.txt",
            )
        assert document["id"] == "doc-integration-001"

        # Step 3: Parse the document (sync - wait for completion)
        mock_connector.parse_document = AsyncMock(return_value={
            "task_id": "task-parse-001",
            "status": "processing",
            "document_id": document["id"],
            "dataset_id": dataset["id"],
        })
        mock_connector.list_documents = AsyncMock(return_value={
            "documents": [{"id": document["id"], "run": "DONE", "progress": 1.0}]
        })

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            parse_result = await ragflow_parse_document_sync(
                dataset_id=dataset["id"],
                document_id=document["id"],
                poll_interval=0.01,
            )
        assert parse_result["status"] == "completed"

        # Step 4: Perform retrieval on the dataset
        mock_connector.retrieval = AsyncMock(return_value={
            "chunks": [
                {
                    "content": "Test document content",
                    "document_name": "test_doc.txt",
                    "dataset_name": "Integration Test Dataset",
                    "similarity": 0.95,
                }
            ],
            "total": 1,
        })

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            results = await ragflow_retrieval(
                query="test content",
                dataset_ids=[dataset["id"]],
            )
        assert len(results["chunks"]) == 1
        assert results["chunks"][0]["content"] == "Test document content"

    @pytest.mark.asyncio
    async def test_document_upload_parse_and_chunk_access(self, mock_connector):
        """Integration test: Document upload, parse, and chunk access workflow."""
        from src.tools.documents import ragflow_upload_document, ragflow_parse_document
        from src.tools.chunks import ragflow_list_chunks

        # Step 1: Upload document
        mock_connector.upload_document = AsyncMock(return_value={
            "id": "doc-chunk-test-001",
            "name": "chunk_test.md",
            "status": "pending",
        })

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            document = await ragflow_upload_document(
                dataset_id="dataset-001",
                base64_content="IyBIZWFkaW5nCgpQYXJhZ3JhcGggdGV4dC4=",  # "# Heading\n\nParagraph text."
                filename="chunk_test.md",
            )
        assert document["id"] == "doc-chunk-test-001"

        # Step 2: Parse document (async - returns status immediately)
        mock_connector.parse_document = AsyncMock(return_value={
            "status": "processing",
            "document_id": document["id"],
            "dataset_id": "dataset-001",
        })

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            parse_result = await ragflow_parse_document(
                dataset_id="dataset-001",
                document_id=document["id"],
            )
        assert parse_result["status"] == "processing"

        # Step 3: List chunks (after parsing completes)
        mock_connector.list_chunks = AsyncMock(return_value={
            "chunks": [
                {"id": "chunk-001", "content": "# Heading", "keywords": ["heading"]},
                {"id": "chunk-002", "content": "Paragraph text.", "keywords": ["paragraph"]},
            ],
            "total": 2,
            "page": 1,
            "page_size": 10,
        })

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            chunks = await ragflow_list_chunks(document_id=document["id"])
        assert len(chunks["chunks"]) == 2
        assert chunks["total"] == 2

    @pytest.mark.asyncio
    async def test_chat_assistant_creation_with_dataset_linking(self, mock_connector):
        """Integration test: Chat assistant creation with dataset linking."""
        from src.tools.datasets import ragflow_create_dataset, ragflow_list_datasets
        from src.tools.chat import ragflow_create_chat, ragflow_create_session, ragflow_chat

        # Step 1: Create datasets for the chat assistant
        mock_connector.create_dataset = AsyncMock(side_effect=[
            {"id": "ds-faq", "name": "FAQ Dataset"},
            {"id": "ds-docs", "name": "Documentation Dataset"},
        ])

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            faq_dataset = await ragflow_create_dataset(name="FAQ Dataset")
            docs_dataset = await ragflow_create_dataset(name="Documentation Dataset")

        # Step 2: Create chat assistant linked to datasets
        mock_connector.create_chat = AsyncMock(return_value={
            "id": "chat-support-001",
            "name": "Support Assistant",
            "dataset_ids": [faq_dataset["id"], docs_dataset["id"]],
            "llm_config": {"model": "gpt-4", "temperature": 0.7},
        })

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            chat = await ragflow_create_chat(
                name="Support Assistant",
                dataset_ids=[faq_dataset["id"], docs_dataset["id"]],
                llm_config={"model": "gpt-4", "temperature": 0.7},
            )
        assert chat["id"] == "chat-support-001"
        assert len(chat["dataset_ids"]) == 2

        # Step 3: Create session and send message
        mock_connector.create_session = AsyncMock(return_value={
            "id": "session-support-001",
            "chat_id": chat["id"],
        })
        mock_connector.send_message = AsyncMock(return_value={
            "session_id": "session-support-001",
            "message": "How do I reset my password?",
            "response": "To reset your password, go to Settings > Security > Reset Password.",
            "sources": [
                {"document_name": "faq.md", "chunk_id": "chunk-faq-001", "similarity": 0.92},
            ],
        })

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            session = await ragflow_create_session(chat_id=chat["id"])
            response = await ragflow_chat(
                session_id=session["id"],
                message="How do I reset my password?",
            )
        assert response["response"] is not None
        assert len(response["sources"]) == 1


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_network_timeout_simulation(self):
        """Error handling: Network timeout simulation."""
        from src.connector import RAGFlowConnector, RAGFlowConnectionError

        api_key = "test-api-key"
        # Use a non-routable IP to trigger timeout
        base_url = "http://10.255.255.1:9999/api/v1"

        async with RAGFlowConnector(api_key=api_key, base_url=base_url, timeout=0.5) as connector:
            with pytest.raises(RAGFlowConnectionError) as exc_info:
                await connector.get("/datasets")

            error_message = str(exc_info.value).lower()
            assert "timeout" in error_message or "connection" in error_message

    @pytest.mark.asyncio
    async def test_invalid_api_key_response(self):
        """Error handling: Invalid API key response."""
        from src.connector import RAGFlowConnector, RAGFlowAPIError

        api_key = "invalid-api-key"
        base_url = "http://localhost:9380/api/v1"

        # Mock the httpx response for invalid API key
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "code": 401,
            "message": "Invalid API key or unauthorized access"
        }
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            async with RAGFlowConnector(api_key=api_key, base_url=base_url) as connector:
                with pytest.raises(RAGFlowAPIError) as exc_info:
                    await connector.get("/datasets")

                error = exc_info.value
                assert error.status_code == 401 or error.code == 401

    @pytest.mark.asyncio
    async def test_missing_required_parameters_validation(self):
        """Error handling: Missing required parameters validation.

        Tests that the connector properly validates upload document parameters.
        The validation occurs in the connector.upload_document method.
        """
        from src.connector import RAGFlowConnector
        from src.tools.datasets import ragflow_delete_dataset

        # Test 1: Delete dataset without confirm=True
        mock_connector = MagicMock()
        mock_connector.delete_dataset = AsyncMock()
        mock_connector.invalidate_cache = MagicMock()

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_delete_dataset(id="dataset-123", confirm=False)

        assert "error" in result
        mock_connector.delete_dataset.assert_not_called()

        # Test 2: Connector validates upload document parameters
        # Create a real connector to test its validation logic
        api_key = "test-api-key"
        base_url = "http://localhost:9380/api/v1"

        async with RAGFlowConnector(api_key=api_key, base_url=base_url) as connector:
            # Both file_path and base64_content are None - should raise error
            with pytest.raises(ValueError) as exc_info:
                await connector.upload_document(
                    dataset_id="dataset-123",
                    file_path=None,
                    base64_content=None,
                )
            assert "file_path" in str(exc_info.value).lower() or "base64" in str(exc_info.value).lower()

            # Both file_path and base64_content provided - should raise error
            with pytest.raises(ValueError) as exc_info:
                await connector.upload_document(
                    dataset_id="dataset-123",
                    file_path="/some/path.txt",
                    base64_content="c29tZSBjb250ZW50",  # "some content"
                )
            assert "not both" in str(exc_info.value).lower() or "either" in str(exc_info.value).lower()

            # base64_content without filename - should raise error
            with pytest.raises(ValueError) as exc_info:
                await connector.upload_document(
                    dataset_id="dataset-123",
                    base64_content="c29tZSBjb250ZW50",  # "some content"
                    filename=None,
                )
            assert "filename" in str(exc_info.value).lower()


class TestEdgeCases:
    """Tests for edge case scenarios."""

    @pytest.mark.asyncio
    async def test_empty_dataset_retrieval_behavior(self):
        """Edge case: Empty dataset retrieval behavior."""
        from src.tools.retrieval import ragflow_retrieval

        mock_connector = MagicMock()
        mock_connector.retrieval = AsyncMock(return_value={
            "chunks": [],
            "total": 0,
        })

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            result = await ragflow_retrieval(
                query="any query",
                dataset_ids=["empty-dataset-001"],
            )

        # Should return empty results without error
        assert result["chunks"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_large_result_pagination(self):
        """Edge case: Large result pagination."""
        from src.tools.datasets import ragflow_list_datasets

        mock_connector = MagicMock()

        # Simulate a large dataset list with pagination
        mock_connector.list_datasets = AsyncMock(return_value={
            "datasets": [
                {"id": f"dataset-{i}", "name": f"Dataset {i}"} for i in range(50)
            ],
            "total": 500,
            "page": 1,
            "page_size": 50,
        })
        mock_connector.cache = MagicMock()
        mock_connector.cache.get.return_value = None

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_list_datasets(page=1, page_size=50)

        # Verify pagination metadata
        assert len(result["datasets"]) == 50
        assert result["total"] == 500
        assert result["page"] == 1
        assert result["page_size"] == 50

        # Verify pagination parameters were passed
        mock_connector.list_datasets.assert_called_once()
        call_kwargs = mock_connector.list_datasets.call_args[1]
        assert call_kwargs.get("page") == 1
        assert call_kwargs.get("page_size") == 50


class TestConfigurationValidation:
    """Tests for configuration and environment validation."""

    def test_environment_variable_validation(self):
        """Configuration: Environment variable validation."""
        from src.config import Settings

        # Test with valid configuration
        env_vars = {
            "RAGFLOW_API_KEY": "valid-api-key",
            "RAGFLOW_URL": "http://custom-server:9380/api/v1",
            "LOG_LEVEL": "DEBUG",
        }

        env_without_existing = {k: v for k, v in os.environ.items()
                               if k not in ("RAGFLOW_API_KEY", "RAGFLOW_URL", "LOG_LEVEL")}
        env_without_existing.update(env_vars)

        with patch.dict(os.environ, env_without_existing, clear=True):
            settings = Settings()

            assert settings.ragflow_api_key == "valid-api-key"
            assert settings.ragflow_url == "http://custom-server:9380/api/v1"
            assert settings.log_level == "DEBUG"

    def test_missing_api_key_raises_error(self):
        """Configuration: Missing API key raises clear error."""
        from src.config import Settings

        # Remove RAGFLOW_API_KEY from environment
        env_without_key = {k: v for k, v in os.environ.items() if k != "RAGFLOW_API_KEY"}

        with patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Settings()

            # Error message should mention the missing variable
            assert "RAGFLOW_API_KEY" in str(exc_info.value)


class TestServerLifecycle:
    """Tests for server lifecycle management."""

    @pytest.mark.asyncio
    async def test_clean_startup_and_shutdown(self):
        """Server lifecycle: Clean startup and shutdown."""
        from src.connector import RAGFlowConnector

        api_key = "test-api-key"
        base_url = "http://localhost:9380/api/v1"

        # Create connector instance
        connector = RAGFlowConnector(api_key=api_key, base_url=base_url)

        # Before entering context, client should be None
        assert connector.client is None

        # Enter context - client should be initialized
        async with connector:
            assert connector.client is not None
            assert not connector.client.is_closed
            assert connector.client.headers["Authorization"] == f"Bearer {api_key}"

        # After exiting context, client should be closed
        assert connector.client.is_closed

    @pytest.mark.asyncio
    async def test_connector_raises_when_not_initialized(self):
        """Server lifecycle: Connector raises error when not initialized."""
        from src.connector import RAGFlowConnector

        api_key = "test-api-key"
        base_url = "http://localhost:9380/api/v1"

        connector = RAGFlowConnector(api_key=api_key, base_url=base_url)

        # Should raise RuntimeError when trying to use without context
        with pytest.raises(RuntimeError) as exc_info:
            await connector.get("/datasets")

        assert "not initialized" in str(exc_info.value).lower()
