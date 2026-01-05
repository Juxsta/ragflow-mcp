"""Tests for retrieval tool functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestRetrievalTool:
    """Test suite for ragflow_retrieval tool."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for testing."""
        from src.connector import RAGFlowConnector
        connector = MagicMock(spec=RAGFlowConnector)
        connector.retrieval = AsyncMock()
        connector.cache = MagicMock()
        return connector

    @pytest.mark.asyncio
    async def test_basic_retrieval_returns_chunks_with_content(self, mock_connector):
        """Test 1: Basic retrieval query returns chunks with content."""
        from src.tools.retrieval import ragflow_retrieval

        # Mock retrieval response with chunks
        mock_connector.retrieval.return_value = {
            "chunks": [
                {
                    "content": "This is the first chunk content about Python programming.",
                    "document_name": "python_guide.pdf",
                    "dataset_name": "Programming Docs",
                    "similarity": 0.95,
                    "highlight": "Python <em>programming</em>",
                },
                {
                    "content": "Another chunk about Python basics.",
                    "document_name": "python_basics.md",
                    "dataset_name": "Programming Docs",
                    "similarity": 0.88,
                    "highlight": "Python <em>basics</em>",
                },
            ],
            "total": 2,
        }

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            result = await ragflow_retrieval(query="Python programming")

        # Verify chunks are returned with content
        assert "chunks" in result
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["content"] == "This is the first chunk content about Python programming."
        assert result["chunks"][1]["content"] == "Another chunk about Python basics."

    @pytest.mark.asyncio
    async def test_retrieval_respects_similarity_threshold(self, mock_connector):
        """Test 2: Retrieval respects similarity_threshold parameter."""
        from src.tools.retrieval import ragflow_retrieval

        mock_connector.retrieval.return_value = {"chunks": [], "total": 0}

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            await ragflow_retrieval(query="test query", similarity_threshold=0.85)

        # Verify similarity_threshold was passed to connector
        mock_connector.retrieval.assert_called_once()
        call_kwargs = mock_connector.retrieval.call_args[1]
        assert call_kwargs.get("similarity_threshold") == 0.85

    @pytest.mark.asyncio
    async def test_retrieval_respects_top_k_parameter(self, mock_connector):
        """Test 3: Retrieval respects top_k parameter."""
        from src.tools.retrieval import ragflow_retrieval

        mock_connector.retrieval.return_value = {"chunks": [], "total": 0}

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            await ragflow_retrieval(query="test query", top_k=5)

        # Verify top_k was passed to connector
        mock_connector.retrieval.assert_called_once()
        call_kwargs = mock_connector.retrieval.call_args[1]
        assert call_kwargs.get("top_k") == 5

    @pytest.mark.asyncio
    async def test_retrieval_with_dataset_ids_filter(self, mock_connector):
        """Test 4: Retrieval with dataset_ids filter works correctly."""
        from src.tools.retrieval import ragflow_retrieval

        mock_connector.retrieval.return_value = {
            "chunks": [
                {
                    "content": "Filtered content from specific dataset.",
                    "document_name": "doc.pdf",
                    "dataset_name": "Specific Dataset",
                    "similarity": 0.92,
                },
            ],
            "total": 1,
        }

        dataset_ids = ["dataset-123", "dataset-456"]

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            result = await ragflow_retrieval(query="test query", dataset_ids=dataset_ids)

        # Verify dataset_ids was passed to connector
        mock_connector.retrieval.assert_called_once()
        call_kwargs = mock_connector.retrieval.call_args[1]
        assert call_kwargs.get("dataset_ids") == dataset_ids
        assert len(result["chunks"]) == 1

    @pytest.mark.asyncio
    async def test_retrieval_with_document_ids_filter(self, mock_connector):
        """Test 5: Retrieval with document_ids filter works correctly."""
        from src.tools.retrieval import ragflow_retrieval

        mock_connector.retrieval.return_value = {
            "chunks": [
                {
                    "content": "Content from specific document.",
                    "document_name": "specific_doc.pdf",
                    "dataset_name": "My Dataset",
                    "similarity": 0.90,
                },
            ],
            "total": 1,
        }

        document_ids = ["doc-abc", "doc-def"]

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            result = await ragflow_retrieval(query="test query", document_ids=document_ids)

        # Verify document_ids was passed to connector
        mock_connector.retrieval.assert_called_once()
        call_kwargs = mock_connector.retrieval.call_args[1]
        assert call_kwargs.get("document_ids") == document_ids
        assert len(result["chunks"]) == 1

    @pytest.mark.asyncio
    async def test_retrieval_handles_empty_results_gracefully(self, mock_connector):
        """Test 6: Retrieval handles empty results gracefully."""
        from src.tools.retrieval import ragflow_retrieval

        # Mock empty response
        mock_connector.retrieval.return_value = {"chunks": [], "total": 0}

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            result = await ragflow_retrieval(query="nonexistent topic xyz123")

        # Should return empty list without error
        assert "chunks" in result
        assert result["chunks"] == []
        assert result["total"] == 0
