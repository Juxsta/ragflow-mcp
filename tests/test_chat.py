"""Tests for chat and session management tool functionality."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestChatTools:
    """Test suite for chat assistant and session management tools."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector for testing."""
        from src.connector import RAGFlowConnector
        connector = MagicMock(spec=RAGFlowConnector)
        connector.create_chat = AsyncMock()
        connector.list_chats = AsyncMock()
        connector.update_chat = AsyncMock()
        connector.delete_chat = AsyncMock()
        connector.create_session = AsyncMock()
        connector.list_sessions = AsyncMock()
        connector.send_message = AsyncMock()
        connector.cache = MagicMock()
        connector.invalidate_cache = MagicMock()
        return connector

    @pytest.mark.asyncio
    async def test_create_chat_assistant_with_valid_config_succeeds(self, mock_connector):
        """Test 1: Create chat assistant with valid config succeeds."""
        from src.tools.chat import ragflow_create_chat

        # Mock create chat response
        mock_connector.create_chat.return_value = {
            "id": "chat-abc123",
            "name": "Customer Support Bot",
            "dataset_ids": ["dataset-001", "dataset-002"],
            "llm_config": {"model": "gpt-4", "temperature": 0.7},
            "prompt_config": {"system_prompt": "You are a helpful assistant."},
            "created_at": "2026-01-04T00:00:00Z",
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result = await ragflow_create_chat(
                name="Customer Support Bot",
                dataset_ids=["dataset-001", "dataset-002"],
                llm_config={"model": "gpt-4", "temperature": 0.7},
                prompt_config={"system_prompt": "You are a helpful assistant."},
            )

        # Verify chat assistant was created with ID
        assert "id" in result
        assert result["id"] == "chat-abc123"
        assert result["name"] == "Customer Support Bot"
        assert result["dataset_ids"] == ["dataset-001", "dataset-002"]
        mock_connector.create_chat.assert_called_once()

        # Verify parameters were passed correctly
        call_kwargs = mock_connector.create_chat.call_args[1]
        assert call_kwargs.get("name") == "Customer Support Bot"
        assert call_kwargs.get("dataset_ids") == ["dataset-001", "dataset-002"]

    @pytest.mark.asyncio
    async def test_list_chat_assistants_returns_results(self, mock_connector):
        """Test 2: List chat assistants returns results."""
        from src.tools.chat import ragflow_list_chats

        # Mock list chats response
        mock_connector.list_chats.return_value = {
            "chats": [
                {"id": "chat-1", "name": "Support Bot", "dataset_ids": ["ds-1"]},
                {"id": "chat-2", "name": "Sales Bot", "dataset_ids": ["ds-2"]},
                {"id": "chat-3", "name": "FAQ Bot", "dataset_ids": ["ds-1", "ds-3"]},
            ],
            "total": 3,
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result = await ragflow_list_chats()

        # Verify list returns chat assistants
        assert "chats" in result
        assert len(result["chats"]) == 3
        assert result["total"] == 3
        mock_connector.list_chats.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_chat_assistant_modifies_config_correctly(self, mock_connector):
        """Test 3: Update chat assistant modifies config correctly."""
        from src.tools.chat import ragflow_update_chat

        # Mock update chat response
        mock_connector.update_chat.return_value = {
            "id": "chat-def456",
            "name": "Updated Support Bot",
            "dataset_ids": ["dataset-001", "dataset-003"],
            "llm_config": {"model": "gpt-4-turbo", "temperature": 0.5},
            "updated_at": "2026-01-04T12:00:00Z",
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result = await ragflow_update_chat(
                chat_id="chat-def456",
                name="Updated Support Bot",
                dataset_ids=["dataset-001", "dataset-003"],
                llm_config={"model": "gpt-4-turbo", "temperature": 0.5},
            )

        # Verify chat was updated
        assert result["id"] == "chat-def456"
        assert result["name"] == "Updated Support Bot"
        assert result["dataset_ids"] == ["dataset-001", "dataset-003"]

        # Verify update was called with correct parameters
        mock_connector.update_chat.assert_called_once()
        call_kwargs = mock_connector.update_chat.call_args[1]
        assert call_kwargs.get("chat_id") == "chat-def456"
        assert call_kwargs.get("name") == "Updated Support Bot"

    @pytest.mark.asyncio
    async def test_delete_chat_assistant_requires_confirm_true(self, mock_connector):
        """Test 4: Delete chat assistant requires confirm=True."""
        from src.tools.chat import ragflow_delete_chat

        # Mock successful delete response
        mock_connector.delete_chat.return_value = {
            "success": True,
            "message": "Chat assistant deleted successfully",
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            # Test with confirm=True - should succeed
            result = await ragflow_delete_chat(chat_id="chat-123", confirm=True)

        # Verify deletion succeeded
        assert result["success"] is True
        mock_connector.delete_chat.assert_called_once()

        # Verify cache was invalidated
        mock_connector.invalidate_cache.assert_called()

        # Test with confirm=False - should fail
        mock_connector.invalidate_cache.reset_mock()
        mock_connector.delete_chat.reset_mock()

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result_fail = await ragflow_delete_chat(chat_id="chat-123", confirm=False)

        # Verify deletion was rejected
        assert "error" in result_fail
        assert result_fail.get("success") is False
        mock_connector.delete_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_session_for_chat_assistant_succeeds(self, mock_connector):
        """Test 5: Create session for chat assistant succeeds."""
        from src.tools.chat import ragflow_create_session

        # Mock create session response
        mock_connector.create_session.return_value = {
            "id": "session-xyz789",
            "chat_id": "chat-abc123",
            "created_at": "2026-01-04T10:00:00Z",
            "messages": [],
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result = await ragflow_create_session(chat_id="chat-abc123")

        # Verify session was created with ID
        assert "id" in result
        assert result["id"] == "session-xyz789"
        assert result["chat_id"] == "chat-abc123"
        mock_connector.create_session.assert_called_once()

        # Verify parameter was passed correctly
        call_kwargs = mock_connector.create_session.call_args[1]
        assert call_kwargs.get("chat_id") == "chat-abc123"

    @pytest.mark.asyncio
    async def test_list_sessions_for_chat_assistant_returns_results(self, mock_connector):
        """Test 6: List sessions for chat assistant returns results."""
        from src.tools.chat import ragflow_list_sessions

        # Mock list sessions response
        mock_connector.list_sessions.return_value = {
            "sessions": [
                {"id": "session-1", "chat_id": "chat-abc", "created_at": "2026-01-04T08:00:00Z"},
                {"id": "session-2", "chat_id": "chat-abc", "created_at": "2026-01-04T09:00:00Z"},
                {"id": "session-3", "chat_id": "chat-abc", "created_at": "2026-01-04T10:00:00Z"},
            ],
            "total": 3,
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result = await ragflow_list_sessions(chat_id="chat-abc")

        # Verify sessions are returned
        assert "sessions" in result
        assert len(result["sessions"]) == 3
        assert result["total"] == 3
        mock_connector.list_sessions.assert_called_once()

        # Verify parameter was passed correctly
        call_kwargs = mock_connector.list_sessions.call_args[1]
        assert call_kwargs.get("chat_id") == "chat-abc"

    @pytest.mark.asyncio
    async def test_send_message_to_session_returns_response(self, mock_connector):
        """Test 7: Send message to session returns response."""
        from src.tools.chat import ragflow_chat

        # Mock chat response
        mock_connector.send_message.return_value = {
            "session_id": "session-xyz789",
            "message": "What are the store hours?",
            "response": "Our store is open from 9 AM to 9 PM, Monday through Saturday.",
            "created_at": "2026-01-04T10:05:00Z",
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result = await ragflow_chat(
                session_id="session-xyz789",
                message="What are the store hours?",
            )

        # Verify response was returned
        assert "response" in result
        assert result["response"] == "Our store is open from 9 AM to 9 PM, Monday through Saturday."
        assert result["session_id"] == "session-xyz789"
        mock_connector.send_message.assert_called_once()

        # Verify parameters were passed correctly
        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs.get("session_id") == "session-xyz789"
        assert call_kwargs.get("message") == "What are the store hours?"

    @pytest.mark.asyncio
    async def test_chat_response_includes_source_citations(self, mock_connector):
        """Test 8: Chat response includes source citations."""
        from src.tools.chat import ragflow_chat

        # Mock chat response with source citations
        mock_connector.send_message.return_value = {
            "session_id": "session-xyz789",
            "message": "What is the return policy?",
            "response": "Our return policy allows returns within 30 days of purchase with a receipt.",
            "sources": [
                {
                    "document_name": "return_policy.pdf",
                    "dataset_name": "Customer Policies",
                    "chunk_id": "chunk-001",
                    "content": "Returns are accepted within 30 days of purchase...",
                    "similarity": 0.95,
                },
                {
                    "document_name": "faq.md",
                    "dataset_name": "FAQ",
                    "chunk_id": "chunk-002",
                    "content": "Q: What is the return policy? A: 30 days with receipt...",
                    "similarity": 0.88,
                },
            ],
            "created_at": "2026-01-04T10:10:00Z",
        }

        with patch("src.tools.chat.get_connector", return_value=mock_connector):
            result = await ragflow_chat(
                session_id="session-xyz789",
                message="What is the return policy?",
            )

        # Verify response includes source citations
        assert "response" in result
        assert "sources" in result
        assert len(result["sources"]) == 2

        # Verify source citation structure
        first_source = result["sources"][0]
        assert "document_name" in first_source
        assert "dataset_name" in first_source
        assert "chunk_id" in first_source
        assert "content" in first_source
        assert "similarity" in first_source
        assert first_source["similarity"] == 0.95
