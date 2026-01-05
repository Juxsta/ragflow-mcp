"""Chat assistant and session management tools for RAGFlow MCP Server.

Provides operations for RAGFlow chat assistants and sessions including:
- Create chat assistant
- List chat assistants
- Update chat assistant
- Delete chat assistant with confirmation
- Create session
- List sessions
- Send message (chat)
"""
from typing import Any

from mcp.server.fastmcp import FastMCP


def get_connector():
    """Get the global connector instance.

    This function is imported from server module to avoid circular imports.
    It will be patched during testing.
    """
    from src.server import get_connector as _get_connector
    return _get_connector()


async def ragflow_create_chat(
    name: str,
    dataset_ids: list[str] | None = None,
    llm_config: dict[str, Any] | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new chat assistant in RAGFlow.

    Creates a chat assistant that can engage in conversations using
    knowledge from the specified datasets.

    Args:
        name: Name of the chat assistant. Required.
        dataset_ids: List of dataset IDs to associate with the assistant.
            The assistant will search these datasets for relevant information.
        llm_config: LLM configuration options.
            Can include model, temperature, max_tokens, etc.
        prompt_config: Prompt configuration options.
            Can include system_prompt for customizing assistant behavior.

    Returns:
        Dictionary containing the created chat assistant with:
            - id: Unique identifier for the chat assistant
            - name: Chat assistant name
            - dataset_ids: Associated dataset IDs
            - llm_config: LLM configuration
            - prompt_config: Prompt configuration
            - created_at: Creation timestamp
    """
    connector = get_connector()

    result = await connector.create_chat(
        name=name,
        dataset_ids=dataset_ids,
        llm_config=llm_config,
        prompt_config=prompt_config,
    )

    return result


async def ragflow_list_chats(
    name: str | None = None,
) -> dict[str, Any]:
    """List chat assistants with optional name filter.

    Retrieves all chat assistants. Results can be filtered by name
    for easier navigation.

    Args:
        name: Optional filter to search chat assistants by name.
            Matches assistants containing this string.

    Returns:
        Dictionary containing:
            - chats: List of chat assistant objects with id, name, dataset_ids
            - total: Total number of chat assistants
    """
    connector = get_connector()

    result = await connector.list_chats(name=name)

    return result


async def ragflow_update_chat(
    chat_id: str,
    name: str | None = None,
    dataset_ids: list[str] | None = None,
    llm_config: dict[str, Any] | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update an existing chat assistant.

    Modifies the configuration of an existing chat assistant. Only the
    fields that are provided will be updated; others remain unchanged.

    Args:
        chat_id: ID of the chat assistant to update. Required.
        name: New name for the chat assistant.
        dataset_ids: New list of dataset IDs to associate.
        llm_config: New LLM configuration options.
        prompt_config: New prompt configuration options.

    Returns:
        Dictionary containing the updated chat assistant with all current fields.
    """
    connector = get_connector()

    result = await connector.update_chat(
        chat_id=chat_id,
        name=name,
        dataset_ids=dataset_ids,
        llm_config=llm_config,
        prompt_config=prompt_config,
    )

    # Invalidate cache after update
    connector.invalidate_cache()

    return result


async def ragflow_delete_chat(
    chat_id: str,
    confirm: bool,
) -> dict[str, Any]:
    """Delete a chat assistant.

    Permanently removes a chat assistant and all its sessions.
    This action cannot be undone.

    IMPORTANT: The confirm parameter must be set to True to prevent
    accidental deletions. If confirm is False or not provided,
    the deletion will be rejected with an error.

    Args:
        chat_id: ID of the chat assistant to delete. Required.
        confirm: Must be True to confirm the deletion. Required.
            Set to True to proceed with deletion.
            Any other value will reject the deletion.

    Returns:
        On success: Dictionary with success=True and confirmation message.
        On rejection: Dictionary with error message explaining the issue.
    """
    # Safety check: require explicit confirmation
    if confirm is not True:
        return {
            "error": "Deletion rejected: confirm parameter must be True to delete a chat assistant. "
                     "This is a safety measure to prevent accidental deletions.",
            "success": False,
        }

    connector = get_connector()

    result = await connector.delete_chat(chat_id=chat_id)

    # Invalidate cache after deletion
    connector.invalidate_cache()

    return result


async def ragflow_create_session(
    chat_id: str,
) -> dict[str, Any]:
    """Create a new session for a chat assistant.

    Creates a conversation session that maintains message history
    and context for a chat assistant.

    Args:
        chat_id: ID of the chat assistant to create a session for. Required.

    Returns:
        Dictionary containing the created session with:
            - id: Unique identifier for the session
            - chat_id: Parent chat assistant ID
            - created_at: Creation timestamp
            - messages: Empty list of messages
    """
    connector = get_connector()

    result = await connector.create_session(chat_id=chat_id)

    return result


async def ragflow_list_sessions(
    chat_id: str,
) -> dict[str, Any]:
    """List sessions for a chat assistant.

    Retrieves all conversation sessions for a specific chat assistant.

    Args:
        chat_id: ID of the chat assistant to list sessions for. Required.

    Returns:
        Dictionary containing:
            - sessions: List of session objects with id, chat_id, created_at
            - total: Total number of sessions
    """
    connector = get_connector()

    result = await connector.list_sessions(chat_id=chat_id)

    return result


async def ragflow_chat(
    session_id: str,
    message: str,
) -> dict[str, Any]:
    """Send a message to a session and receive a response.

    Sends a user message to an active session and receives an AI-generated
    response based on the chat assistant's configuration and linked datasets.

    Args:
        session_id: ID of the session to send the message to. Required.
        message: The user's message. Required.

    Returns:
        Dictionary containing:
            - session_id: The session ID
            - message: The sent message
            - response: The assistant's response text
            - sources: List of source citations with document names,
                       chunk content, and similarity scores (if available)
            - created_at: Message timestamp
    """
    connector = get_connector()

    result = await connector.send_message(
        session_id=session_id,
        message=message,
    )

    return result


def register_chat_tools(mcp: FastMCP) -> None:
    """Register chat assistant and session management tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """

    @mcp.tool()
    async def ragflow_create_chat_tool(
        name: str,
        dataset_ids: list[str] | None = None,
        llm_config: dict[str, Any] | None = None,
        prompt_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new RAGFlow chat assistant.

        Creates a conversational assistant linked to specified datasets.

        Args:
            name: Chat assistant name. Required.
            dataset_ids: Dataset IDs to link for knowledge retrieval.
            llm_config: LLM settings (model, temperature, etc.).
            prompt_config: Prompt settings (system_prompt, etc.).

        Returns:
            Created chat assistant with id, name, and configuration.
        """
        return await ragflow_create_chat(
            name=name,
            dataset_ids=dataset_ids,
            llm_config=llm_config,
            prompt_config=prompt_config,
        )

    @mcp.tool()
    async def ragflow_list_chats_tool(
        name: str | None = None,
    ) -> dict[str, Any]:
        """List RAGFlow chat assistants.

        Retrieves available chat assistants with optional name filter.

        Args:
            name: Filter chat assistants containing this name.

        Returns:
            Dictionary with 'chats' list and 'total' count.
        """
        return await ragflow_list_chats(name=name)

    @mcp.tool()
    async def ragflow_update_chat_tool(
        chat_id: str,
        name: str | None = None,
        dataset_ids: list[str] | None = None,
        llm_config: dict[str, Any] | None = None,
        prompt_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an existing RAGFlow chat assistant.

        Modifies chat assistant configuration. Only provided fields are updated.

        Args:
            chat_id: Chat assistant ID to update. Required.
            name: New chat assistant name.
            dataset_ids: New list of dataset IDs.
            llm_config: New LLM configuration.
            prompt_config: New prompt configuration.

        Returns:
            Updated chat assistant with all current fields.
        """
        return await ragflow_update_chat(
            chat_id=chat_id,
            name=name,
            dataset_ids=dataset_ids,
            llm_config=llm_config,
            prompt_config=prompt_config,
        )

    @mcp.tool()
    async def ragflow_delete_chat_tool(
        chat_id: str,
        confirm: bool,
    ) -> dict[str, Any]:
        """Delete a RAGFlow chat assistant permanently.

        CAUTION: This permanently removes the chat assistant and all sessions.
        The confirm parameter MUST be True to proceed.

        Args:
            chat_id: Chat assistant ID to delete. Required.
            confirm: Must be True to confirm deletion. Required.

        Returns:
            Success status or error if confirm is not True.
        """
        return await ragflow_delete_chat(
            chat_id=chat_id,
            confirm=confirm,
        )

    @mcp.tool()
    async def ragflow_create_session_tool(
        chat_id: str,
    ) -> dict[str, Any]:
        """Create a new session for a RAGFlow chat assistant.

        Creates a conversation session for message history.

        Args:
            chat_id: Chat assistant ID. Required.

        Returns:
            Created session with id, chat_id, and timestamps.
        """
        return await ragflow_create_session(chat_id=chat_id)

    @mcp.tool()
    async def ragflow_list_sessions_tool(
        chat_id: str,
    ) -> dict[str, Any]:
        """List sessions for a RAGFlow chat assistant.

        Retrieves all sessions for a specific chat assistant.

        Args:
            chat_id: Chat assistant ID. Required.

        Returns:
            Dictionary with 'sessions' list and 'total' count.
        """
        return await ragflow_list_sessions(chat_id=chat_id)

    @mcp.tool()
    async def ragflow_chat_tool(
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Send a message to a RAGFlow chat session.

        Sends a message and receives an AI response with source citations.

        Args:
            session_id: Session ID. Required.
            message: User message to send. Required.

        Returns:
            Response with answer text and source citations.
        """
        return await ragflow_chat(
            session_id=session_id,
            message=message,
        )
