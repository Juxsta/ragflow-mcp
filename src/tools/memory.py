"""Memory management tools for RAGFlow MCP Server.

Provides CRUD operations for RAGFlow memory system including:
- Create memory (with type, embedding, and LLM configuration)
- List memories with filters
- Get memory configuration
- Update memory settings
- Delete memory
- Save messages to memory
- List messages in a memory
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


async def ragflow_create_memory(
    name: str,
    memory_type: list[str],
    embd_id: str,
    llm_id: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a new RAGFlow memory.

    Creates a new memory with specified configuration for storing
    conversation history and extracting knowledge.

    Args:
        name: Unique name for the memory (max 128 characters, BMP only).
        memory_type: List of memory types to extract.
            Options: "raw" (dialogue content), "semantic" (knowledge/facts),
            "episodic" (time-stamped events), "procedural" (skills/habits).
        embd_id: Embedding model ID in format "model_name@model_factory".
            Example: "BAAI/bge-large-zh-v1.5@BAAI"
        llm_id: Chat model ID in format "model_name@model_factory".
            Example: "glm-4-flash@ZHIPU-AI"
        description: Optional description of the memory.

    Returns:
        Dictionary containing the created memory with:
            - id: Unique identifier for the memory
            - name: Memory name
            - memory_type: List of memory types
            - embd_id: Embedding model ID
            - llm_id: LLM model ID
            - description: Memory description
            - create_time: Creation timestamp

    Raises:
        ValueError: If validation fails (e.g., invalid memory_type).
        RAGFlowAPIError: If API returns an error.
    """
    connector = get_connector()

    payload: dict[str, Any] = {
        "name": name,
        "memory_type": memory_type,
        "embd_id": embd_id,
        "llm_id": llm_id,
    }

    if description is not None:
        payload["description"] = description

    result = await connector.post("/memories", json=payload)

    return result.get("data", {})


async def ragflow_list_memories(
    tenant_id: str | None = None,
    memory_type: str | list[str] | None = None,
    storage_type: str | None = None,
    keywords: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """List RAGFlow memories with optional filters.

    Retrieves memories with support for filtering and pagination.

    Args:
        tenant_id: Filter by owner ID (supports multiple IDs).
        memory_type: Filter by memory type(s).
            Options: "raw", "semantic", "episodic", "procedural".
            A memory matches if its type is included in the provided value(s).
        storage_type: Filter by storage format. Options: "table" (default).
        keywords: Fuzzy search by memory name.
        page: Page number for pagination (1-based). Default: 1.
        page_size: Number of items per page. Default: 50.

    Returns:
        Dictionary containing:
            - memory_list: List of memory objects with id, name, type, etc.
            - total_count: Total number of memories matching the filter
    """
    connector = get_connector()

    params: dict[str, Any] = {}
    if tenant_id is not None:
        params["tenant_id"] = tenant_id
    if memory_type is not None:
        params["memory_type"] = memory_type
    if storage_type is not None:
        params["storage_type"] = storage_type
    if keywords is not None:
        params["keywords"] = keywords
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size

    result = await connector.get(
        "/memories",
        params=params if params else None,
    )

    return result.get("data", {})


async def ragflow_get_memory_config(
    memory_id: str,
) -> dict[str, Any]:
    """Get the configuration of a RAGFlow memory.

    Retrieves detailed configuration settings for a specific memory.

    Args:
        memory_id: ID of the memory to get configuration for. Required.

    Returns:
        Dictionary containing the memory configuration with:
            - id: Memory ID
            - name: Memory name
            - description: Memory description
            - memory_type: List of memory types
            - embd_id: Embedding model ID
            - llm_id: LLM model ID
            - memory_size: Size limit in bytes
            - forgetting_policy: Policy for evicting old data (e.g., "FIFO")
            - temperature: Temperature setting for generation
            - permission: Permission level ("me" or "team")
            - avatar: Base64-encoded avatar
            - system_prompt: System-level instructions
            - user_prompt: User-level instructions
            - create_time: Creation timestamp
    """
    connector = get_connector()

    result = await connector.get(f"/memories/{memory_id}/config")

    return result.get("data", {})


async def ragflow_update_memory(
    memory_id: str,
    name: str | None = None,
    description: str | None = None,
    avatar: str | None = None,
    permission: str | None = None,
    llm_id: str | None = None,
    memory_size: int | None = None,
    forgetting_policy: str | None = None,
    temperature: float | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    """Update a RAGFlow memory configuration.

    Modifies an existing memory's settings. Only the fields that are
    provided will be updated; others remain unchanged.

    Args:
        memory_id: ID of the memory to update. Required.
        name: New name for the memory (max 128 characters, BMP only).
        description: New description for the memory.
        avatar: New base64-encoded avatar (max 65535 characters).
        permission: Permission level. Options: "me" (default), "team".
        llm_id: New LLM model ID in format "model_name@model_factory".
        memory_size: New size limit in bytes (max 10MB, default ~5MB).
        forgetting_policy: Policy for evicting old data. Default: "FIFO".
        temperature: Temperature for generation (range [0, 1]).
        system_prompt: New system-level instructions.
        user_prompt: New user-level instructions.

    Returns:
        Dictionary containing the updated memory configuration.
    """
    connector = get_connector()

    payload: dict[str, Any] = {}

    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    if avatar is not None:
        payload["avatar"] = avatar
    if permission is not None:
        payload["permission"] = permission
    if llm_id is not None:
        payload["llm_id"] = llm_id
    if memory_size is not None:
        payload["memory_size"] = memory_size
    if forgetting_policy is not None:
        payload["forgetting_policy"] = forgetting_policy
    if temperature is not None:
        payload["temperature"] = temperature
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if user_prompt is not None:
        payload["user_prompt"] = user_prompt

    result = await connector.put(f"/memories/{memory_id}", json=payload)

    return result.get("data", {})


async def ragflow_delete_memory(
    memory_id: str,
) -> dict[str, Any]:
    """Delete a RAGFlow memory.

    Permanently removes a memory and all its stored messages.
    This action cannot be undone.

    Args:
        memory_id: ID of the memory to delete. Required.

    Returns:
        Dictionary containing:
            - code: Response code (0 on success)
            - data: Always null on success
            - message: Success confirmation
    """
    connector = get_connector()

    result = await connector.delete(f"/memories/{memory_id}")

    return result


async def ragflow_save_message(
    memory_id: list[str],
    agent_id: str,
    session_id: str,
    user_input: str,
    agent_response: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Save a conversation message to RAGFlow memory.

    Stores a user input and agent response pair in one or more memories.
    The messages are processed according to each memory's configuration
    and memory_type settings.

    Args:
        memory_id: List of memory IDs to store the message in. Required.
        agent_id: ID of the agent that generated the response. Required.
        session_id: ID of the conversation session. Required.
        user_input: The text input provided by the user. Required.
        agent_response: The text response generated by the AI agent. Required.
        user_id: Optional ID of the user participating in the conversation.

    Returns:
        Dictionary containing:
            - code: Response code (0 on success)
            - data: Always null on success
            - message: Success message (e.g., "All add to task.")
    """
    connector = get_connector()

    payload: dict[str, Any] = {
        "memory_id": memory_id,
        "agent_id": agent_id,
        "session_id": session_id,
        "user_input": user_input,
        "agent_response": agent_response,
    }

    if user_id is not None:
        payload["user_id"] = user_id

    result = await connector.post("/messages", json=payload)

    return result


async def ragflow_list_memory_messages(
    memory_id: str,
    agent_id: str | list[str] | None = None,
    session_id: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """List messages stored in a RAGFlow memory.

    Retrieves messages from a memory with optional filtering.

    Args:
        memory_id: ID of the memory to list messages from. Required.
        agent_id: Filter by agent ID (supports multiple IDs).
        session_id: Filter by session ID (supports fuzzy search).
        page: Page number for pagination (1-based). Default: 1.
        page_size: Number of items per page. Default: 50.

    Returns:
        Dictionary containing:
            - messages: List of message objects with content, timestamps, etc.
            - total: Total number of messages
            - page: Current page number
            - page_size: Items per page
    """
    connector = get_connector()

    params: dict[str, Any] = {}
    if agent_id is not None:
        params["agent_id"] = agent_id
    if session_id is not None:
        params["keywords"] = session_id  # API uses "keywords" for session_id filter
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size

    result = await connector.get(
        f"/memories/{memory_id}",
        params=params if params else None,
    )

    return result.get("data", {})


def register_memory_tools(mcp: FastMCP) -> None:
    """Register memory management tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """

    @mcp.tool()
    async def ragflow_create_memory_tool(
        name: str,
        memory_type: list[str],
        embd_id: str,
        llm_id: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a new RAGFlow memory.

        Creates a memory for storing conversation history and extracting knowledge.

        Args:
            name: Unique name for the memory (max 128 chars). Required.
            memory_type: List of types: "raw", "semantic", "episodic", "procedural". Required.
            embd_id: Embedding model ID (e.g., "BAAI/bge-large-zh-v1.5@BAAI"). Required.
            llm_id: LLM model ID (e.g., "glm-4-flash@ZHIPU-AI"). Required.
            description: Optional description of the memory.

        Returns:
            Created memory with id, name, memory_type, embd_id, llm_id.
        """
        return await ragflow_create_memory(
            name=name,
            memory_type=memory_type,
            embd_id=embd_id,
            llm_id=llm_id,
            description=description,
        )

    @mcp.tool()
    async def ragflow_list_memories_tool(
        tenant_id: str | None = None,
        memory_type: str | list[str] | None = None,
        storage_type: str | None = None,
        keywords: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """List RAGFlow memories with filters.

        Retrieves memories with optional filtering and pagination.

        Args:
            tenant_id: Filter by owner ID.
            memory_type: Filter by type(s): "raw", "semantic", "episodic", "procedural".
            storage_type: Filter by storage format (e.g., "table").
            keywords: Fuzzy search by memory name.
            page: Page number (1-based). Default: 1.
            page_size: Items per page. Default: 50.

        Returns:
            Dictionary with 'memory_list' and 'total_count'.
        """
        return await ragflow_list_memories(
            tenant_id=tenant_id,
            memory_type=memory_type,
            storage_type=storage_type,
            keywords=keywords,
            page=page,
            page_size=page_size,
        )

    @mcp.tool()
    async def ragflow_get_memory_config_tool(
        memory_id: str,
    ) -> dict[str, Any]:
        """Get RAGFlow memory configuration.

        Retrieves detailed configuration for a specific memory.

        Args:
            memory_id: Memory ID to get configuration for. Required.

        Returns:
            Memory configuration with all settings.
        """
        return await ragflow_get_memory_config(memory_id=memory_id)

    @mcp.tool()
    async def ragflow_update_memory_tool(
        memory_id: str,
        name: str | None = None,
        description: str | None = None,
        avatar: str | None = None,
        permission: str | None = None,
        llm_id: str | None = None,
        memory_size: int | None = None,
        forgetting_policy: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Update RAGFlow memory configuration.

        Modifies memory settings. Only provided fields are updated.

        Args:
            memory_id: Memory ID to update. Required.
            name: New name (max 128 chars).
            description: New description.
            avatar: New base64 avatar.
            permission: Permission level ("me" or "team").
            llm_id: New LLM model ID.
            memory_size: New size limit in bytes.
            forgetting_policy: Eviction policy (e.g., "FIFO").
            temperature: Temperature (0-1).
            system_prompt: New system prompt.
            user_prompt: New user prompt.

        Returns:
            Updated memory configuration.
        """
        return await ragflow_update_memory(
            memory_id=memory_id,
            name=name,
            description=description,
            avatar=avatar,
            permission=permission,
            llm_id=llm_id,
            memory_size=memory_size,
            forgetting_policy=forgetting_policy,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    @mcp.tool()
    async def ragflow_delete_memory_tool(
        memory_id: str,
    ) -> dict[str, Any]:
        """Delete a RAGFlow memory.

        Permanently removes a memory and all its messages.

        Args:
            memory_id: Memory ID to delete. Required.

        Returns:
            Success confirmation with code and message.
        """
        return await ragflow_delete_memory(memory_id=memory_id)

    @mcp.tool()
    async def ragflow_save_message_tool(
        memory_id: list[str],
        agent_id: str,
        session_id: str,
        user_input: str,
        agent_response: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Save a conversation message to RAGFlow memory.

        Stores user input and agent response in memories.

        Args:
            memory_id: List of memory IDs to store in. Required.
            agent_id: Agent ID that generated the response. Required.
            session_id: Conversation session ID. Required.
            user_input: User's text input. Required.
            agent_response: Agent's text response. Required.
            user_id: Optional user ID.

        Returns:
            Success confirmation with message like "All add to task."
        """
        return await ragflow_save_message(
            memory_id=memory_id,
            agent_id=agent_id,
            session_id=session_id,
            user_input=user_input,
            agent_response=agent_response,
            user_id=user_id,
        )

    @mcp.tool()
    async def ragflow_list_memory_messages_tool(
        memory_id: str,
        agent_id: str | list[str] | None = None,
        session_id: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """List messages in a RAGFlow memory.

        Retrieves stored messages with optional filtering.

        Args:
            memory_id: Memory ID to list messages from. Required.
            agent_id: Filter by agent ID (supports multiple).
            session_id: Filter by session ID (fuzzy search).
            page: Page number (1-based). Default: 1.
            page_size: Items per page. Default: 50.

        Returns:
            Dictionary with messages list and pagination info.
        """
        return await ragflow_list_memory_messages(
            memory_id=memory_id,
            agent_id=agent_id,
            session_id=session_id,
            page=page,
            page_size=page_size,
        )
