"""Tests for dataset CRUD tool functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestDatasetTools:
    """Test suite for dataset CRUD tools."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for testing."""
        from src.connector import RAGFlowConnector
        connector = MagicMock(spec=RAGFlowConnector)
        connector.create_dataset = AsyncMock()
        connector.list_datasets = AsyncMock()
        connector.update_dataset = AsyncMock()
        connector.delete_dataset = AsyncMock()
        connector.cache = MagicMock()
        connector.invalidate_cache = MagicMock()
        return connector

    @pytest.mark.asyncio
    async def test_create_dataset_with_valid_parameters_succeeds(self, mock_connector):
        """Test 1: Create dataset with valid parameters succeeds."""
        from src.tools.datasets import ragflow_create_dataset

        # Mock create dataset response
        mock_connector.create_dataset.return_value = {
            "id": "dataset-123abc",
            "name": "My Knowledge Base",
            "description": "A test knowledge base",
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "chunk_method": "naive",
            "parser_config": {"chunk_size": 1024},
            "created_at": "2026-01-04T00:00:00Z",
        }

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_create_dataset(
                name="My Knowledge Base",
                description="A test knowledge base",
                embedding_model="BAAI/bge-large-en-v1.5",
                chunk_method="naive",
                parser_config={"chunk_size": 1024},
            )

        # Verify dataset was created with ID
        assert "id" in result
        assert result["id"] == "dataset-123abc"
        assert result["name"] == "My Knowledge Base"
        assert result["description"] == "A test knowledge base"
        mock_connector.create_dataset.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_datasets_returns_paginated_results(self, mock_connector):
        """Test 2: List datasets returns paginated results."""
        from src.tools.datasets import ragflow_list_datasets

        # Mock list datasets response with pagination
        mock_connector.list_datasets.return_value = {
            "datasets": [
                {"id": "dataset-1", "name": "Dataset One", "description": "First dataset"},
                {"id": "dataset-2", "name": "Dataset Two", "description": "Second dataset"},
            ],
            "total": 5,
            "page": 1,
            "page_size": 2,
        }

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_list_datasets(page=1, page_size=2)

        # Verify paginated results are returned
        assert "datasets" in result
        assert len(result["datasets"]) == 2
        assert result["total"] == 5
        assert result["page"] == 1
        assert result["page_size"] == 2

        # Verify pagination parameters were passed
        mock_connector.list_datasets.assert_called_once()
        call_kwargs = mock_connector.list_datasets.call_args[1]
        assert call_kwargs.get("page") == 1
        assert call_kwargs.get("page_size") == 2

    @pytest.mark.asyncio
    async def test_list_datasets_with_name_filter_works(self, mock_connector):
        """Test 3: List datasets with name filter works."""
        from src.tools.datasets import ragflow_list_datasets

        # Mock filtered list response
        mock_connector.list_datasets.return_value = {
            "datasets": [
                {"id": "dataset-1", "name": "Python Docs", "description": "Python documentation"},
            ],
            "total": 1,
            "page": 1,
            "page_size": 10,
        }

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_list_datasets(name="Python")

        # Verify filtered results
        assert len(result["datasets"]) == 1
        assert "Python" in result["datasets"][0]["name"]

        # Verify name filter was passed
        mock_connector.list_datasets.assert_called_once()
        call_kwargs = mock_connector.list_datasets.call_args[1]
        assert call_kwargs.get("name") == "Python"

    @pytest.mark.asyncio
    async def test_update_dataset_modifies_fields_correctly(self, mock_connector):
        """Test 4: Update dataset modifies fields correctly."""
        from src.tools.datasets import ragflow_update_dataset

        # Mock update response
        mock_connector.update_dataset.return_value = {
            "id": "dataset-123",
            "name": "Updated Name",
            "description": "Updated description",
            "chunk_method": "qa",
            "parser_config": {"chunk_size": 2048},
        }

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_update_dataset(
                id="dataset-123",
                name="Updated Name",
                description="Updated description",
                chunk_method="qa",
                parser_config={"chunk_size": 2048},
            )

        # Verify fields were updated
        assert result["id"] == "dataset-123"
        assert result["name"] == "Updated Name"
        assert result["description"] == "Updated description"
        assert result["chunk_method"] == "qa"

        # Verify cache was invalidated
        mock_connector.invalidate_cache.assert_called()

    @pytest.mark.asyncio
    async def test_delete_dataset_requires_confirm_true(self, mock_connector):
        """Test 5: Delete dataset requires confirm=True."""
        from src.tools.datasets import ragflow_delete_dataset

        # Mock successful delete response
        mock_connector.delete_dataset.return_value = {
            "success": True,
            "message": "Dataset deleted successfully",
        }

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_delete_dataset(id="dataset-123", confirm=True)

        # Verify deletion succeeded
        assert result["success"] is True

        # Verify cache was invalidated
        mock_connector.invalidate_cache.assert_called()

    @pytest.mark.asyncio
    async def test_delete_dataset_fails_when_confirm_false(self, mock_connector):
        """Test 6: Delete dataset fails when confirm=False."""
        from src.tools.datasets import ragflow_delete_dataset

        with patch("src.tools.datasets.get_connector", return_value=mock_connector):
            result = await ragflow_delete_dataset(id="dataset-123", confirm=False)

        # Verify deletion was rejected
        assert "error" in result
        assert "confirm" in result["error"].lower()

        # Verify delete was NOT called on connector
        mock_connector.delete_dataset.assert_not_called()
