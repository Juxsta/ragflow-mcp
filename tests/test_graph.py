"""Tests for GraphRAG and RAPTOR tool functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestGraphRAGAndRAPTORTools:
    """Test suite for GraphRAG and RAPTOR tools."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for testing."""
        from src.connector import RAGFlowConnector
        connector = MagicMock(spec=RAGFlowConnector)
        connector.build_graph = AsyncMock()
        connector.get_graph_status = AsyncMock()
        connector.get_graph = AsyncMock()
        connector.delete_graph = AsyncMock()
        connector.build_raptor = AsyncMock()
        connector.get_raptor_status = AsyncMock()
        connector.cache = MagicMock()
        connector.invalidate_cache = MagicMock()
        return connector

    @pytest.mark.asyncio
    async def test_build_graph_triggers_construction_and_returns_task_id(self, mock_connector):
        """Test 1: Build graph triggers construction and returns task_id."""
        from src.tools.graph import ragflow_build_graph

        # Mock build graph response
        mock_connector.build_graph.return_value = {
            "task_id": "task-graph-001",
            "dataset_id": "dataset-abc123",
            "status": "processing",
            "message": "Knowledge graph construction started",
        }

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            result = await ragflow_build_graph(dataset_id="dataset-abc123")

        # Verify graph construction was triggered and task_id returned
        assert "task_id" in result
        assert result["task_id"] == "task-graph-001"
        assert result["status"] == "processing"
        mock_connector.build_graph.assert_called_once()

        # Verify dataset_id was passed correctly
        call_kwargs = mock_connector.build_graph.call_args[1]
        assert call_kwargs.get("dataset_id") == "dataset-abc123"

    @pytest.mark.asyncio
    async def test_graph_status_returns_construction_progress(self, mock_connector):
        """Test 2: Graph status returns construction progress."""
        from src.tools.graph import ragflow_graph_status

        # Mock graph status response - in progress
        mock_connector.get_graph_status.return_value = {
            "dataset_id": "dataset-abc123",
            "status": "processing",
            "progress": 45,
            "message": "Building entity relationships...",
        }

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            result = await ragflow_graph_status(dataset_id="dataset-abc123")

        # Verify status and progress are returned
        assert "dataset_id" in result
        assert result["dataset_id"] == "dataset-abc123"
        assert result["status"] == "processing"
        assert result["progress"] == 45
        mock_connector.get_graph_status.assert_called_once()

        # Verify dataset_id was passed correctly
        call_kwargs = mock_connector.get_graph_status.call_args[1]
        assert call_kwargs.get("dataset_id") == "dataset-abc123"

    @pytest.mark.asyncio
    async def test_get_graph_returns_entities_and_relationships(self, mock_connector):
        """Test 3: Get graph returns entities and relationships."""
        from src.tools.graph import ragflow_get_graph

        # Mock get graph response with entities and relationships
        mock_connector.get_graph.return_value = {
            "dataset_id": "dataset-abc123",
            "entities": [
                {"id": "entity-1", "name": "Python", "type": "Technology", "properties": {}},
                {"id": "entity-2", "name": "Machine Learning", "type": "Concept", "properties": {}},
                {"id": "entity-3", "name": "TensorFlow", "type": "Framework", "properties": {}},
            ],
            "relationships": [
                {"id": "rel-1", "source": "entity-1", "target": "entity-2", "type": "used_for"},
                {"id": "rel-2", "source": "entity-3", "target": "entity-2", "type": "implements"},
            ],
            "statistics": {
                "entity_count": 3,
                "relationship_count": 2,
            },
        }

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            result = await ragflow_get_graph(dataset_id="dataset-abc123")

        # Verify entities and relationships are returned
        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 3
        assert len(result["relationships"]) == 2

        # Verify entity structure
        entity = result["entities"][0]
        assert "id" in entity
        assert "name" in entity
        assert "type" in entity

        # Verify relationship structure
        relationship = result["relationships"][0]
        assert "id" in relationship
        assert "source" in relationship
        assert "target" in relationship
        assert "type" in relationship

        mock_connector.get_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_graph_requires_confirm_true(self, mock_connector):
        """Test 4: Delete graph requires confirm=True."""
        from src.tools.graph import ragflow_delete_graph

        # Mock successful delete response
        mock_connector.delete_graph.return_value = {
            "success": True,
            "message": "Knowledge graph deleted successfully",
        }

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            # Test with confirm=True - should succeed
            result = await ragflow_delete_graph(dataset_id="dataset-abc123", confirm=True)

        # Verify deletion succeeded
        assert result["success"] is True
        mock_connector.delete_graph.assert_called_once()

        # Verify cache was invalidated
        mock_connector.invalidate_cache.assert_called()

        # Test with confirm=False - should fail
        mock_connector.invalidate_cache.reset_mock()
        mock_connector.delete_graph.reset_mock()

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            result_fail = await ragflow_delete_graph(dataset_id="dataset-abc123", confirm=False)

        # Verify deletion was rejected
        assert "error" in result_fail
        assert result_fail.get("success") is False
        mock_connector.delete_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_raptor_triggers_construction_and_returns_task_id(self, mock_connector):
        """Test 5: Build RAPTOR triggers construction and returns task_id."""
        from src.tools.graph import ragflow_build_raptor

        # Mock build RAPTOR response
        mock_connector.build_raptor.return_value = {
            "task_id": "task-raptor-001",
            "dataset_id": "dataset-xyz789",
            "status": "processing",
            "message": "RAPTOR tree construction started",
        }

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            result = await ragflow_build_raptor(dataset_id="dataset-xyz789")

        # Verify RAPTOR construction was triggered and task_id returned
        assert "task_id" in result
        assert result["task_id"] == "task-raptor-001"
        assert result["status"] == "processing"
        mock_connector.build_raptor.assert_called_once()

        # Verify dataset_id was passed correctly
        call_kwargs = mock_connector.build_raptor.call_args[1]
        assert call_kwargs.get("dataset_id") == "dataset-xyz789"

    @pytest.mark.asyncio
    async def test_raptor_status_returns_construction_progress(self, mock_connector):
        """Test 6: RAPTOR status returns construction progress."""
        from src.tools.graph import ragflow_raptor_status

        # Mock RAPTOR status response - completed
        mock_connector.get_raptor_status.return_value = {
            "dataset_id": "dataset-xyz789",
            "status": "completed",
            "progress": 100,
            "message": "RAPTOR tree construction completed",
            "tree_depth": 3,
            "node_count": 156,
        }

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            result = await ragflow_raptor_status(dataset_id="dataset-xyz789")

        # Verify status and progress are returned
        assert "dataset_id" in result
        assert result["dataset_id"] == "dataset-xyz789"
        assert result["status"] == "completed"
        assert result["progress"] == 100
        mock_connector.get_raptor_status.assert_called_once()

        # Verify dataset_id was passed correctly
        call_kwargs = mock_connector.get_raptor_status.call_args[1]
        assert call_kwargs.get("dataset_id") == "dataset-xyz789"

    @pytest.mark.asyncio
    async def test_graph_operations_handle_dataset_without_graph_gracefully(self, mock_connector):
        """Test 7: Graph operations handle dataset without graph gracefully."""
        from src.tools.graph import ragflow_get_graph

        # Mock get graph response for dataset without a graph
        mock_connector.get_graph.return_value = {
            "dataset_id": "dataset-no-graph",
            "entities": [],
            "relationships": [],
            "statistics": {
                "entity_count": 0,
                "relationship_count": 0,
            },
            "message": "No knowledge graph exists for this dataset",
        }

        with patch("src.tools.graph.get_connector", return_value=mock_connector):
            result = await ragflow_get_graph(dataset_id="dataset-no-graph")

        # Verify empty but valid response
        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 0
        assert len(result["relationships"]) == 0

        # Verify no error was raised - graceful handling
        assert "error" not in result or result.get("error") is None
        mock_connector.get_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_long_running_operations_report_progress_correctly(self, mock_connector):
        """Test 8: Long-running operations report progress correctly."""
        from src.tools.graph import ragflow_graph_status

        # Simulate progress updates at different stages
        progress_stages = [
            {"dataset_id": "dataset-abc123", "status": "processing", "progress": 10, "message": "Extracting entities..."},
            {"dataset_id": "dataset-abc123", "status": "processing", "progress": 50, "message": "Building relationships..."},
            {"dataset_id": "dataset-abc123", "status": "processing", "progress": 90, "message": "Finalizing graph..."},
            {"dataset_id": "dataset-abc123", "status": "completed", "progress": 100, "message": "Graph construction complete"},
        ]

        for expected_stage in progress_stages:
            mock_connector.get_graph_status.return_value = expected_stage

            with patch("src.tools.graph.get_connector", return_value=mock_connector):
                result = await ragflow_graph_status(dataset_id="dataset-abc123")

            # Verify progress reporting
            assert "progress" in result
            assert result["progress"] == expected_stage["progress"]
            assert result["status"] == expected_stage["status"]
            assert "message" in result

        # Verify final status is completed
        assert result["status"] == "completed"
        assert result["progress"] == 100
