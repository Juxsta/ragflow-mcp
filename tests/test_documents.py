"""Tests for document management tool functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import base64


class TestDocumentTools:
    """Test suite for document management tools."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for testing."""
        from src.connector import RAGFlowConnector
        connector = MagicMock(spec=RAGFlowConnector)
        connector.upload_document = AsyncMock()
        connector.list_documents = AsyncMock()
        connector.parse_document = AsyncMock()
        connector.get_parse_status = AsyncMock()
        connector.download_document = AsyncMock()
        connector.delete_document = AsyncMock()
        connector.stop_parsing = AsyncMock()
        connector.cache = MagicMock()
        connector.invalidate_cache = MagicMock()
        return connector

    @pytest.mark.asyncio
    async def test_upload_document_from_file_path_succeeds(self, mock_connector, tmp_path):
        """Test 1: Upload document from local file path succeeds."""
        from src.tools.documents import ragflow_upload_document

        # Create a test file
        test_file = tmp_path / "test_document.txt"
        test_file.write_text("This is test content for the document.")

        # Mock upload document response
        mock_connector.upload_document.return_value = {
            "id": "doc-123abc",
            "name": "test_document.txt",
            "size": 39,
            "status": "pending",
            "created_at": "2026-01-04T00:00:00Z",
        }

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result = await ragflow_upload_document(
                dataset_id="dataset-456",
                file_path=str(test_file),
            )

        # Verify document was uploaded with ID
        assert "id" in result
        assert result["id"] == "doc-123abc"
        assert result["name"] == "test_document.txt"
        assert result["status"] == "pending"
        mock_connector.upload_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_document_from_base64_succeeds(self, mock_connector):
        """Test 2: Upload document from base64 content succeeds."""
        from src.tools.documents import ragflow_upload_document

        # Create base64 encoded content
        test_content = "This is test content encoded in base64."
        base64_content = base64.b64encode(test_content.encode()).decode()

        # Mock upload document response
        mock_connector.upload_document.return_value = {
            "id": "doc-789xyz",
            "name": "my_document.txt",
            "size": len(test_content),
            "status": "pending",
            "created_at": "2026-01-04T00:00:00Z",
        }

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result = await ragflow_upload_document(
                dataset_id="dataset-456",
                base64_content=base64_content,
                filename="my_document.txt",
            )

        # Verify document was uploaded with ID
        assert "id" in result
        assert result["id"] == "doc-789xyz"
        assert result["name"] == "my_document.txt"
        mock_connector.upload_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_returns_filtered_results(self, mock_connector):
        """Test 3: List documents returns filtered results."""
        from src.tools.documents import ragflow_list_documents

        # Mock list documents response with filters
        mock_connector.list_documents.return_value = {
            "documents": [
                {"id": "doc-1", "name": "report.pdf", "status": "parsed", "file_type": "pdf"},
                {"id": "doc-2", "name": "notes.pdf", "status": "parsed", "file_type": "pdf"},
            ],
            "total": 2,
            "page": 1,
            "page_size": 10,
        }

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result = await ragflow_list_documents(
                dataset_id="dataset-123",
                status="parsed",
                file_type="pdf",
            )

        # Verify filtered results are returned
        assert "documents" in result
        assert len(result["documents"]) == 2
        assert all(doc["status"] == "parsed" for doc in result["documents"])
        assert all(doc["file_type"] == "pdf" for doc in result["documents"])

        # Verify filter parameters were passed
        mock_connector.list_documents.assert_called_once()
        call_kwargs = mock_connector.list_documents.call_args[1]
        assert call_kwargs.get("status") == "parsed"
        assert call_kwargs.get("file_type") == "pdf"

    @pytest.mark.asyncio
    async def test_parse_document_async_returns_task_id(self, mock_connector):
        """Test 4: Parse document (async) returns status."""
        from src.tools.documents import ragflow_parse_document

        # Mock parse document response
        mock_connector.parse_document.return_value = {
            "status": "processing",
            "document_id": "doc-123",
            "dataset_id": "dataset-abc",
        }

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result = await ragflow_parse_document(
                dataset_id="dataset-abc",
                document_id="doc-123",
                chunk_method="naive",
            )

        # Verify status is returned immediately
        assert result["status"] == "processing"
        assert result["document_id"] == "doc-123"

        # Verify parse was called with correct parameters
        mock_connector.parse_document.assert_called_once()
        call_kwargs = mock_connector.parse_document.call_args[1]
        assert call_kwargs.get("dataset_id") == "dataset-abc"
        assert call_kwargs.get("document_id") == "doc-123"
        assert call_kwargs.get("chunk_method") == "naive"

    @pytest.mark.asyncio
    async def test_parse_document_sync_waits_for_completion(self, mock_connector):
        """Test 5: Parse document (sync) waits for completion."""
        from src.tools.documents import ragflow_parse_document_sync

        # Mock parse response
        mock_connector.parse_document.return_value = {
            "status": "processing",
            "document_id": "doc-456",
            "dataset_id": "dataset-xyz",
        }

        # Simulate document status progression: RUNNING -> DONE
        mock_connector.list_documents.side_effect = [
            {"documents": [{"id": "doc-456", "run": "RUNNING", "progress": 0.5}]},
            {"documents": [{"id": "doc-456", "run": "DONE", "progress": 1.0}]},
        ]

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result = await ragflow_parse_document_sync(
                dataset_id="dataset-xyz",
                document_id="doc-456",
                poll_interval=0.01,  # Short interval for testing
            )

        # Verify completion status is returned
        assert result["status"] == "completed"
        assert result["progress"] == 1.0
        assert result["document_id"] == "doc-456"
        assert result["dataset_id"] == "dataset-xyz"

        # Verify polling occurred
        assert mock_connector.list_documents.call_count >= 1

    @pytest.mark.asyncio
    async def test_download_document_returns_content(self, mock_connector):
        """Test 6: Download document returns content."""
        from src.tools.documents import ragflow_download_document

        # Mock download response with base64 encoded content (for binary)
        test_content = b"Binary content of the document"
        encoded_content = base64.b64encode(test_content).decode()

        mock_connector.download_document.return_value = {
            "id": "doc-123",
            "name": "document.pdf",
            "content": encoded_content,
            "content_type": "application/pdf",
            "is_base64": True,
        }

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result = await ragflow_download_document(document_id="doc-123")

        # Verify content is returned
        assert "content" in result
        assert result["content"] == encoded_content
        assert result["is_base64"] is True
        assert result["name"] == "document.pdf"

    @pytest.mark.asyncio
    async def test_delete_document_requires_confirm_true(self, mock_connector):
        """Test 7: Delete document requires confirm=True."""
        from src.tools.documents import ragflow_delete_document

        # Mock successful delete response
        mock_connector.delete_document.return_value = {
            "success": True,
            "message": "Document deleted successfully",
        }

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            # Test with confirm=True - should succeed
            result = await ragflow_delete_document(
                dataset_id="ds-123",
                document_id="doc-123",
                confirm=True,
            )

        # Verify deletion succeeded
        assert result["success"] is True

        # Verify cache was invalidated
        mock_connector.invalidate_cache.assert_called()

        # Test with confirm=False - should fail
        mock_connector.invalidate_cache.reset_mock()
        mock_connector.delete_document.reset_mock()

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result_fail = await ragflow_delete_document(
                dataset_id="ds-123",
                document_id="doc-123",
                confirm=False,
            )

        # Verify deletion was rejected
        assert "error" in result_fail
        assert result_fail.get("success") is False
        mock_connector.delete_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_parsing_cancels_active_job(self, mock_connector):
        """Test 8: Stop parsing cancels active job."""
        from src.tools.documents import ragflow_stop_parsing

        # Mock stop parsing response
        mock_connector.stop_parsing.return_value = {
            "task_id": "task-xyz789",
            "status": "cancelled",
            "message": "Parsing job cancelled successfully",
        }

        with patch("src.tools.documents.get_connector", return_value=mock_connector):
            result = await ragflow_stop_parsing(task_id="task-xyz789")

        # Verify cancellation succeeded
        assert result["status"] == "cancelled"
        assert result["task_id"] == "task-xyz789"

        # Verify stop was called with correct task_id
        mock_connector.stop_parsing.assert_called_once()
        call_kwargs = mock_connector.stop_parsing.call_args[1]
        assert call_kwargs.get("task_id") == "task-xyz789"
