"""Chunk management tools for RAGFlow MCP Server.

Provides CRUD operations for RAGFlow document chunks including:
- Add chunk to document
- List chunks with pagination
- Update chunk content and keywords
- Delete chunk (single and batch) with confirmation
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


async def ragflow_add_chunk(
    document_id: str,
    content: str,
    keywords: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a new chunk to a RAGFlow document.

    Creates a new text chunk within a document. Chunks are the basic
    units of content that can be retrieved during semantic search.

    Args:
        document_id: ID of the document to add the chunk to. Required.
        content: Text content of the chunk. Required.
            This is the actual text that will be indexed and searchable.
        keywords: Optional list of keywords for the chunk.
            These can be used to improve search relevance.
        metadata: Optional metadata dictionary for the chunk.
            Can include additional context like source, page number, etc.

    Returns:
        Dictionary containing the created chunk with:
            - id: Unique identifier for the chunk
            - content: The chunk content
            - keywords: List of keywords
            - document_id: Parent document ID
            - created_at: Creation timestamp
    """
    connector = get_connector()

    result = await connector.add_chunk(
        document_id=document_id,
        content=content,
        keywords=keywords,
        metadata=metadata,
    )

    return result


async def ragflow_list_chunks(
    document_id: str,
    dataset_id: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """List chunks in a document with pagination.

    Retrieves all chunks within a document. Results are paginated
    for efficient handling of documents with many chunks.

    Args:
        document_id: ID of the document to list chunks from. Required.
        dataset_id: ID of the dataset containing the document. Required for proper API access.
        page: Page number for pagination (1-based). Default is 1.
        page_size: Number of items per page. Default is 10.

    Returns:
        Dictionary containing:
            - chunks: List of chunk objects with id, content, keywords
            - total: Total number of chunks in the document
            - page: Current page number
            - page_size: Items per page
    """
    connector = get_connector()

    result = await connector.list_chunks(
        document_id=document_id,
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
    )

    return result


async def ragflow_update_chunk(
    chunk_id: str,
    content: str | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Update a chunk's content and/or keywords.

    Modifies an existing chunk. Only the fields that are provided
    will be updated; others remain unchanged.

    Note: Updating chunk content may trigger re-indexing, which
    can affect search results.

    Args:
        chunk_id: ID of the chunk to update. Required.
        content: New text content for the chunk.
            If not provided, content remains unchanged.
        keywords: New list of keywords for the chunk.
            If not provided, keywords remain unchanged.

    Returns:
        Dictionary containing the updated chunk with:
            - id: Chunk ID
            - content: Updated content (or original if not changed)
            - keywords: Updated keywords (or original if not changed)
            - updated_at: Update timestamp
    """
    connector = get_connector()

    result = await connector.update_chunk(
        chunk_id=chunk_id,
        content=content,
        keywords=keywords,
    )

    return result


async def ragflow_delete_chunk(
    dataset_id: str,
    document_id: str,
    chunk_id: str | None = None,
    chunk_ids: list[str] | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Delete one or more chunks.

    Permanently removes chunks from a document. Supports both single
    chunk deletion (via chunk_id) and batch deletion (via chunk_ids).
    At least one of chunk_id or chunk_ids must be provided.

    IMPORTANT: The confirm parameter must be set to True to prevent
    accidental deletions. If confirm is False, the deletion will be
    rejected with an error.

    Args:
        dataset_id: ID of the dataset. Required.
        document_id: ID of the document. Required.
        chunk_id: ID of a single chunk to delete.
            Use this for deleting individual chunks.
        chunk_ids: List of chunk IDs to delete in batch.
            Use this for deleting multiple chunks at once.
        confirm: Must be True to confirm the deletion. Required.
            This is a safety measure to prevent accidental deletions.

    Returns:
        On success (single): Dictionary with success=True and confirmation message.
        On success (batch): Dictionary with success=True, deleted_count, and message.
        On rejection: Dictionary with error message explaining the issue.

    Raises:
        ValueError: If neither chunk_id nor chunk_ids is provided.
    """
    # Safety check: require explicit confirmation
    if confirm is not True:
        return {
            "error": "Deletion rejected: confirm parameter must be True to delete chunks. "
                     "This is a safety measure to prevent accidental deletions.",
            "success": False,
        }

    # Validate input - at least one must be provided
    if chunk_id is None and chunk_ids is None:
        return {
            "error": "Either chunk_id or chunk_ids must be provided.",
            "success": False,
        }

    connector = get_connector()

    # Handle batch delete if chunk_ids is provided
    if chunk_ids is not None:
        result = await connector.delete_chunks_batch(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_ids=chunk_ids,
        )
    else:
        # Single chunk delete
        result = await connector.delete_chunk(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=chunk_id,
        )

    # Invalidate cache after deletion
    connector.invalidate_cache()

    return result


def register_chunk_tools(mcp: FastMCP) -> None:
    """Register chunk management tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """

    @mcp.tool()
    async def ragflow_add_chunk_tool(
        document_id: str,
        content: str,
        keywords: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a new chunk to a RAGFlow document.

        Creates a new searchable text chunk within a document.

        Args:
            document_id: Document ID to add the chunk to. Required.
            content: Text content of the chunk. Required.
            keywords: Optional list of keywords for search.
            metadata: Optional metadata dictionary.

        Returns:
            Created chunk with id, content, keywords, document_id.
        """
        return await ragflow_add_chunk(
            document_id=document_id,
            content=content,
            keywords=keywords,
            metadata=metadata,
        )

    @mcp.tool()
    async def ragflow_list_chunks_tool(
        document_id: str,
        dataset_id: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """List chunks in a RAGFlow document.

        Retrieves chunks with pagination support.

        Args:
            document_id: Document ID to list chunks from. Required.
            dataset_id: Dataset ID containing the document. Required for API access.
            page: Page number (1-based). Default: 1.
            page_size: Items per page. Default: 10.

        Returns:
            Dictionary with 'chunks' list, 'total' count, page info.
        """
        return await ragflow_list_chunks(
            document_id=document_id,
            dataset_id=dataset_id,
            page=page,
            page_size=page_size,
        )

    @mcp.tool()
    async def ragflow_update_chunk_tool(
        chunk_id: str,
        content: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a chunk's content and/or keywords.

        Modifies an existing chunk. Only provided fields are updated.

        Args:
            chunk_id: Chunk ID to update. Required.
            content: New text content for the chunk.
            keywords: New list of keywords.

        Returns:
            Updated chunk with id, content, keywords, updated_at.
        """
        return await ragflow_update_chunk(
            chunk_id=chunk_id,
            content=content,
            keywords=keywords,
        )

    @mcp.tool()
    async def ragflow_delete_chunk_tool(
        dataset_id: str,
        document_id: str,
        confirm: bool,
        chunk_id: str | None = None,
        chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Delete one or more chunks permanently.

        CAUTION: This permanently removes chunks from the document.
        The confirm parameter MUST be True to proceed.
        Provide either chunk_id (single) or chunk_ids (batch).

        Args:
            dataset_id: Dataset ID. Required.
            document_id: Document ID. Required.
            confirm: Must be True to confirm deletion. Required.
            chunk_id: Single chunk ID to delete.
            chunk_ids: List of chunk IDs for batch delete.

        Returns:
            Success status or error if confirm is not True.
        """
        return await ragflow_delete_chunk(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_id=chunk_id,
            chunk_ids=chunk_ids,
            confirm=confirm,
        )
