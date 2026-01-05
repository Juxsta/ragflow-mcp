"""Tests for chunk management tool functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestChunkTools:
    """Test suite for chunk management tools."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for testing."""
        from src.connector import RAGFlowConnector
        connector = MagicMock(spec=RAGFlowConnector)
        connector.add_chunk = AsyncMock()
        connector.list_chunks = AsyncMock()
        connector.update_chunk = AsyncMock()
        connector.delete_chunk = AsyncMock()
        connector.delete_chunks_batch = AsyncMock()
        connector.cache = MagicMock()
        connector.invalidate_cache = MagicMock()
        return connector

    @pytest.mark.asyncio
    async def test_add_chunk_with_content_and_keywords_succeeds(self, mock_connector):
        """Test 1: Add chunk with content and keywords succeeds."""
        from src.tools.chunks import ragflow_add_chunk

        # Mock add chunk response
        mock_connector.add_chunk.return_value = {
            "id": "chunk-abc123",
            "content": "This is the chunk content for testing.",
            "keywords": ["testing", "chunk", "content"],
            "document_id": "doc-456",
            "created_at": "2026-01-04T00:00:00Z",
        }

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            result = await ragflow_add_chunk(
                document_id="doc-456",
                content="This is the chunk content for testing.",
                keywords=["testing", "chunk", "content"],
            )

        # Verify chunk was created with ID
        assert "id" in result
        assert result["id"] == "chunk-abc123"
        assert result["content"] == "This is the chunk content for testing."
        assert result["keywords"] == ["testing", "chunk", "content"]
        mock_connector.add_chunk.assert_called_once()

        # Verify parameters were passed correctly
        call_kwargs = mock_connector.add_chunk.call_args[1]
        assert call_kwargs.get("document_id") == "doc-456"
        assert call_kwargs.get("content") == "This is the chunk content for testing."
        assert call_kwargs.get("keywords") == ["testing", "chunk", "content"]

    @pytest.mark.asyncio
    async def test_list_chunks_returns_paginated_results(self, mock_connector):
        """Test 2: List chunks returns paginated results."""
        from src.tools.chunks import ragflow_list_chunks

        # Mock list chunks response with pagination
        mock_connector.list_chunks.return_value = {
            "chunks": [
                {"id": "chunk-1", "content": "First chunk content", "keywords": ["first"]},
                {"id": "chunk-2", "content": "Second chunk content", "keywords": ["second"]},
                {"id": "chunk-3", "content": "Third chunk content", "keywords": ["third"]},
            ],
            "total": 10,
            "page": 1,
            "page_size": 3,
        }

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            result = await ragflow_list_chunks(
                document_id="doc-789",
                page=1,
                page_size=3,
            )

        # Verify paginated results are returned
        assert "chunks" in result
        assert len(result["chunks"]) == 3
        assert result["total"] == 10
        assert result["page"] == 1
        assert result["page_size"] == 3

        # Verify list was called with correct parameters
        mock_connector.list_chunks.assert_called_once()
        call_kwargs = mock_connector.list_chunks.call_args[1]
        assert call_kwargs.get("document_id") == "doc-789"
        assert call_kwargs.get("page") == 1
        assert call_kwargs.get("page_size") == 3

    @pytest.mark.asyncio
    async def test_update_chunk_modifies_content_correctly(self, mock_connector):
        """Test 3: Update chunk modifies content correctly."""
        from src.tools.chunks import ragflow_update_chunk

        # Mock update chunk response with new content
        mock_connector.update_chunk.return_value = {
            "id": "chunk-def456",
            "content": "Updated chunk content with new information.",
            "keywords": ["original", "keywords"],
            "updated_at": "2026-01-04T12:00:00Z",
        }

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            result = await ragflow_update_chunk(
                chunk_id="chunk-def456",
                content="Updated chunk content with new information.",
            )

        # Verify content was updated
        assert result["id"] == "chunk-def456"
        assert result["content"] == "Updated chunk content with new information."

        # Verify update was called with correct parameters
        mock_connector.update_chunk.assert_called_once()
        call_kwargs = mock_connector.update_chunk.call_args[1]
        assert call_kwargs.get("chunk_id") == "chunk-def456"
        assert call_kwargs.get("content") == "Updated chunk content with new information."

    @pytest.mark.asyncio
    async def test_update_chunk_modifies_keywords_correctly(self, mock_connector):
        """Test 4: Update chunk modifies keywords correctly."""
        from src.tools.chunks import ragflow_update_chunk

        # Mock update chunk response with new keywords
        mock_connector.update_chunk.return_value = {
            "id": "chunk-ghi789",
            "content": "Original content stays the same.",
            "keywords": ["updated", "new", "keywords"],
            "updated_at": "2026-01-04T14:00:00Z",
        }

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            result = await ragflow_update_chunk(
                chunk_id="chunk-ghi789",
                keywords=["updated", "new", "keywords"],
            )

        # Verify keywords were updated
        assert result["id"] == "chunk-ghi789"
        assert result["keywords"] == ["updated", "new", "keywords"]

        # Verify update was called with correct parameters
        mock_connector.update_chunk.assert_called_once()
        call_kwargs = mock_connector.update_chunk.call_args[1]
        assert call_kwargs.get("chunk_id") == "chunk-ghi789"
        assert call_kwargs.get("keywords") == ["updated", "new", "keywords"]

    @pytest.mark.asyncio
    async def test_delete_single_chunk_requires_confirm_true(self, mock_connector):
        """Test 5: Delete single chunk requires confirm=True."""
        from src.tools.chunks import ragflow_delete_chunk

        # Mock successful delete response
        mock_connector.delete_chunk.return_value = {
            "success": True,
            "message": "Chunk deleted successfully",
        }

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            # Test with confirm=True - should succeed
            result = await ragflow_delete_chunk(
                dataset_id="ds-123",
                document_id="doc-456",
                chunk_id="chunk-123",
                confirm=True,
            )

        # Verify deletion succeeded
        assert result["success"] is True

        # Verify cache was invalidated
        mock_connector.invalidate_cache.assert_called()

        # Test with confirm=False - should fail
        mock_connector.invalidate_cache.reset_mock()
        mock_connector.delete_chunk.reset_mock()

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            result_fail = await ragflow_delete_chunk(
                dataset_id="ds-123",
                document_id="doc-456",
                chunk_id="chunk-123",
                confirm=False,
            )

        # Verify deletion was rejected
        assert "error" in result_fail
        assert result_fail.get("success") is False
        mock_connector.delete_chunk.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_batch_of_chunks_works_with_chunk_ids_list(self, mock_connector):
        """Test 6: Delete batch of chunks works with chunk_ids list."""
        from src.tools.chunks import ragflow_delete_chunk

        # Mock successful batch delete response
        mock_connector.delete_chunks_batch.return_value = {
            "success": True,
            "deleted_count": 3,
            "message": "3 chunks deleted successfully",
        }

        with patch("src.tools.chunks.get_connector", return_value=mock_connector):
            result = await ragflow_delete_chunk(
                dataset_id="ds-123",
                document_id="doc-456",
                chunk_ids=["chunk-1", "chunk-2", "chunk-3"],
                confirm=True,
            )

        # Verify batch deletion succeeded
        assert result["success"] is True
        assert result["deleted_count"] == 3

        # Verify batch delete was called with correct parameters
        mock_connector.delete_chunks_batch.assert_called_once()
        call_kwargs = mock_connector.delete_chunks_batch.call_args[1]
        assert call_kwargs.get("dataset_id") == "ds-123"
        assert call_kwargs.get("document_id") == "doc-456"
        assert call_kwargs.get("chunk_ids") == ["chunk-1", "chunk-2", "chunk-3"]

        # Verify cache was invalidated
        mock_connector.invalidate_cache.assert_called()
