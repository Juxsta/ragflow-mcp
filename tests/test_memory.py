"""Tests for memory management tools in RAGFlow MCP Server.

This module contains tests for the memory management functionality including:
- Creating memories
- Listing memories with filters
- Getting memory configuration
- Updating memory settings
- Deleting memories
- Saving messages to memory
- Listing messages in memory
"""
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.tools.memory import (
    ragflow_create_memory,
    ragflow_delete_memory,
    ragflow_get_memory_config,
    ragflow_list_memory_messages,
    ragflow_list_memories,
    ragflow_save_message,
    ragflow_update_memory,
)


@pytest.fixture
def mock_connector():
    """Create a mock RAGFlowConnector."""
    connector = Mock()
    connector.post = AsyncMock()
    connector.get = AsyncMock()
    connector.put = AsyncMock()
    connector.delete = AsyncMock()
    return connector


@pytest.fixture
def patch_get_connector(mock_connector):
    """Patch the get_connector function to return mock connector."""
    with patch("src.tools.memory.get_connector", return_value=mock_connector):
        yield mock_connector


class TestCreateMemory:
    """Tests for ragflow_create_memory function."""

    @pytest.mark.asyncio
    async def test_create_memory_basic(self, patch_get_connector):
        """Test creating a memory with basic parameters."""
        # Setup mock response
        patch_get_connector.post.return_value = {
            "code": 0,
            "data": {
                "id": "test-memory-id",
                "name": "test_memory",
                "memory_type": ["raw", "semantic"],
                "embd_id": "BAAI/bge-large-zh-v1.5@BAAI",
                "llm_id": "glm-4-flash@ZHIPU-AI",
                "create_time": "2025-01-06T10:00:00Z",
            },
        }

        # Execute
        result = await ragflow_create_memory(
            name="test_memory",
            memory_type=["raw", "semantic"],
            embd_id="BAAI/bge-large-zh-v1.5@BAAI",
            llm_id="glm-4-flash@ZHIPU-AI",
        )

        # Verify
        patch_get_connector.post.assert_called_once_with(
            "/memories",
            json={
                "name": "test_memory",
                "memory_type": ["raw", "semantic"],
                "embd_id": "BAAI/bge-large-zh-v1.5@BAAI",
                "llm_id": "glm-4-flash@ZHIPU-AI",
            },
        )
        assert result["id"] == "test-memory-id"
        assert result["name"] == "test_memory"

    @pytest.mark.asyncio
    async def test_create_memory_with_description(self, patch_get_connector):
        """Test creating a memory with description."""
        patch_get_connector.post.return_value = {
            "code": 0,
            "data": {
                "id": "test-memory-id",
                "name": "test_memory",
                "description": "Test memory for conversations",
                "memory_type": ["semantic"],
                "embd_id": "model@factory",
                "llm_id": "llm@factory",
            },
        }

        result = await ragflow_create_memory(
            name="test_memory",
            memory_type=["semantic"],
            embd_id="model@factory",
            llm_id="llm@factory",
            description="Test memory for conversations",
        )

        # Verify description is included
        call_args = patch_get_connector.post.call_args
        assert call_args[1]["json"]["description"] == "Test memory for conversations"
        assert result["description"] == "Test memory for conversations"


class TestListMemories:
    """Tests for ragflow_list_memories function."""

    @pytest.mark.asyncio
    async def test_list_memories_basic(self, patch_get_connector):
        """Test listing memories without filters."""
        patch_get_connector.get.return_value = {
            "code": 0,
            "data": {
                "memory_list": [
                    {
                        "id": "mem-1",
                        "name": "memory_one",
                        "memory_type": ["raw", "semantic"],
                    },
                    {
                        "id": "mem-2",
                        "name": "memory_two",
                        "memory_type": ["episodic"],
                    },
                ],
                "total_count": 2,
            },
        }

        result = await ragflow_list_memories()

        patch_get_connector.get.assert_called_once_with("/memories", params=None)
        assert len(result["memory_list"]) == 2
        assert result["total_count"] == 2

    @pytest.mark.asyncio
    async def test_list_memories_with_filters(self, patch_get_connector):
        """Test listing memories with filters."""
        patch_get_connector.get.return_value = {
            "code": 0,
            "data": {
                "memory_list": [{"id": "mem-1", "name": "semantic_mem"}],
                "total_count": 1,
            },
        }

        result = await ragflow_list_memories(
            memory_type="semantic",
            keywords="test",
            page=1,
            page_size=10,
        )

        call_args = patch_get_connector.get.call_args
        assert call_args[1]["params"]["memory_type"] == "semantic"
        assert call_args[1]["params"]["keywords"] == "test"
        assert call_args[1]["params"]["page"] == 1
        assert call_args[1]["params"]["page_size"] == 10


class TestGetMemoryConfig:
    """Tests for ragflow_get_memory_config function."""

    @pytest.mark.asyncio
    async def test_get_memory_config(self, patch_get_connector):
        """Test getting memory configuration."""
        expected_config = {
            "id": "mem-1",
            "name": "test_memory",
            "memory_type": ["raw", "semantic"],
            "embd_id": "BAAI/bge-large-zh-v1.5@BAAI",
            "llm_id": "glm-4-flash@ZHIPU-AI",
            "memory_size": 5242880,
            "forgetting_policy": "FIFO",
            "temperature": 0.7,
            "permission": "me",
        }
        patch_get_connector.get.return_value = {
            "code": 0,
            "data": expected_config,
        }

        result = await ragflow_get_memory_config(memory_id="mem-1")

        patch_get_connector.get.assert_called_once_with("/memories/mem-1/config")
        assert result["id"] == "mem-1"
        assert result["forgetting_policy"] == "FIFO"


class TestUpdateMemory:
    """Tests for ragflow_update_memory function."""

    @pytest.mark.asyncio
    async def test_update_memory_partial(self, patch_get_connector):
        """Test updating memory with partial fields."""
        patch_get_connector.put.return_value = {
            "code": 0,
            "data": {
                "id": "mem-1",
                "name": "updated_name",
                "description": "Updated description",
            },
        }

        result = await ragflow_update_memory(
            memory_id="mem-1",
            name="updated_name",
            description="Updated description",
        )

        call_args = patch_get_connector.put.call_args
        assert call_args[0][0] == "/memories/mem-1"
        assert call_args[1]["json"]["name"] == "updated_name"
        assert call_args[1]["json"]["description"] == "Updated description"
        assert result["name"] == "updated_name"

    @pytest.mark.asyncio
    async def test_update_memory_all_fields(self, patch_get_connector):
        """Test updating memory with all fields."""
        patch_get_connector.put.return_value = {
            "code": 0,
            "data": {"id": "mem-1"},
        }

        await ragflow_update_memory(
            memory_id="mem-1",
            name="new_name",
            description="new_desc",
            avatar="base64avatar",
            permission="team",
            llm_id="new@llm",
            memory_size=10000000,
            forgetting_policy="FIFO",
            temperature=0.5,
            system_prompt="System prompt",
            user_prompt="User prompt",
        )

        call_args = patch_get_connector.put.call_args
        payload = call_args[1]["json"]
        assert payload["name"] == "new_name"
        assert payload["permission"] == "team"
        assert payload["memory_size"] == 10000000
        assert payload["temperature"] == 0.5


class TestDeleteMemory:
    """Tests for ragflow_delete_memory function."""

    @pytest.mark.asyncio
    async def test_delete_memory(self, patch_get_connector):
        """Test deleting a memory."""
        patch_get_connector.delete.return_value = {
            "code": 0,
            "data": None,
            "message": True,
        }

        result = await ragflow_delete_memory(memory_id="mem-1")

        patch_get_connector.delete.assert_called_once_with("/memories/mem-1")
        assert result["code"] == 0
        assert result["message"] is True


class TestSaveMessage:
    """Tests for ragflow_save_message function."""

    @pytest.mark.asyncio
    async def test_save_message_basic(self, patch_get_connector):
        """Test saving a message to memory."""
        patch_get_connector.post.return_value = {
            "code": 0,
            "data": None,
            "message": "All add to task.",
        }

        result = await ragflow_save_message(
            memory_id=["mem-1"],
            agent_id="agent-1",
            session_id="session-1",
            user_input="Hello",
            agent_response="Hi there!",
        )

        call_args = patch_get_connector.post.call_args
        assert call_args[0][0] == "/messages"
        payload = call_args[1]["json"]
        assert payload["memory_id"] == ["mem-1"]
        assert payload["agent_id"] == "agent-1"
        assert payload["user_input"] == "Hello"
        assert payload["agent_response"] == "Hi there!"
        assert result["message"] == "All add to task."

    @pytest.mark.asyncio
    async def test_save_message_with_user_id(self, patch_get_connector):
        """Test saving a message with user ID."""
        patch_get_connector.post.return_value = {
            "code": 0,
            "data": None,
            "message": "All add to task.",
        }

        await ragflow_save_message(
            memory_id=["mem-1", "mem-2"],
            agent_id="agent-1",
            session_id="session-1",
            user_input="Test",
            agent_response="Response",
            user_id="user-123",
        )

        call_args = patch_get_connector.post.call_args
        payload = call_args[1]["json"]
        assert payload["user_id"] == "user-123"
        assert payload["memory_id"] == ["mem-1", "mem-2"]


class TestListMemoryMessages:
    """Tests for ragflow_list_memory_messages function."""

    @pytest.mark.asyncio
    async def test_list_memory_messages_basic(self, patch_get_connector):
        """Test listing messages from memory."""
        patch_get_connector.get.return_value = {
            "code": 0,
            "data": {
                "messages": [
                    {
                        "id": "msg-1",
                        "user_input": "Hello",
                        "agent_response": "Hi!",
                    },
                    {
                        "id": "msg-2",
                        "user_input": "How are you?",
                        "agent_response": "I'm doing well!",
                    },
                ],
                "total": 2,
            },
        }

        result = await ragflow_list_memory_messages(memory_id="mem-1")

        patch_get_connector.get.assert_called_once_with("/memories/mem-1", params=None)
        assert len(result["messages"]) == 2
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_list_memory_messages_with_filters(self, patch_get_connector):
        """Test listing messages with filters."""
        patch_get_connector.get.return_value = {
            "code": 0,
            "data": {"messages": [], "total": 0},
        }

        await ragflow_list_memory_messages(
            memory_id="mem-1",
            agent_id="agent-1",
            session_id="session-123",
            page=2,
            page_size=20,
        )

        call_args = patch_get_connector.get.call_args
        params = call_args[1]["params"]
        assert params["agent_id"] == "agent-1"
        assert params["keywords"] == "session-123"  # API uses "keywords" for session filter
        assert params["page"] == 2
        assert params["page_size"] == 20
