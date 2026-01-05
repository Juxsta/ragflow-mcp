"""Dataset management tools for RAGFlow MCP Server.

Provides CRUD operations for RAGFlow datasets including:
- Create dataset
- List datasets with pagination and filtering
- Update dataset
- Delete dataset with confirmation
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


async def ragflow_create_dataset(
    name: str,
    description: str | None = None,
    embedding_model: str | None = None,
    chunk_method: str | None = None,
    parser_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new dataset in RAGFlow.

    Creates a new knowledge base dataset for storing and organizing documents.
    The dataset will be configured with the specified embedding model and
    chunking method for document processing.

    Args:
        name: Name of the dataset. Required.
        description: Optional description of the dataset's purpose.
        embedding_model: Embedding model to use for vectorization.
            Examples: "BAAI/bge-large-en-v1.5", "text-embedding-ada-002".
            If not specified, uses server default.
        chunk_method: Method for chunking documents.
            Options: "naive" (simple splitting), "qa" (Q&A pairs),
            "manual" (preserve existing structure).
        parser_config: Additional parser configuration options.
            Can include chunk_size, overlap, etc.

    Returns:
        Dictionary containing the created dataset with:
            - id: Unique identifier for the dataset
            - name: Dataset name
            - description: Dataset description
            - embedding_model: The embedding model used
            - chunk_method: The chunking method
            - created_at: Creation timestamp
    """
    connector = get_connector()

    result = await connector.create_dataset(
        name=name,
        description=description,
        embedding_model=embedding_model,
        chunk_method=chunk_method,
        parser_config=parser_config,
    )

    return result


async def ragflow_list_datasets(
    page: int | None = None,
    page_size: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """List datasets with optional pagination and filtering.

    Retrieves a list of available datasets. Results can be paginated
    and filtered by name for easier navigation.

    Args:
        page: Page number for pagination (1-based). Default is 1.
        page_size: Number of items per page. Default is 10.
        name: Optional filter to search datasets by name.
            Matches datasets containing this string.

    Returns:
        Dictionary containing:
            - datasets: List of dataset objects with id, name, description
            - total: Total number of datasets matching the filter
            - page: Current page number
            - page_size: Items per page
    """
    connector = get_connector()

    result = await connector.list_datasets(
        page=page,
        page_size=page_size,
        name=name,
    )

    return result


async def ragflow_update_dataset(
    id: str,
    name: str | None = None,
    description: str | None = None,
    chunk_method: str | None = None,
    parser_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update an existing dataset.

    Modifies the configuration of an existing dataset. Only the fields
    that are provided will be updated; others remain unchanged.

    Note: Changing chunk_method or parser_config does not automatically
    reprocess existing documents. You may need to reparse documents
    for changes to take effect.

    Args:
        id: ID of the dataset to update. Required.
        name: New name for the dataset.
        description: New description for the dataset.
        chunk_method: New chunking method.
            Options: "naive", "qa", "manual".
        parser_config: New parser configuration options.

    Returns:
        Dictionary containing the updated dataset with all current fields.
    """
    connector = get_connector()

    result = await connector.update_dataset(
        dataset_id=id,
        name=name,
        description=description,
        chunk_method=chunk_method,
        parser_config=parser_config,
    )

    # Invalidate cache after update
    connector.invalidate_cache()

    # Return the updated dataset with the ID included
    if "id" not in result:
        result["id"] = id
    if name is not None and "name" not in result:
        result["name"] = name
    if description is not None and "description" not in result:
        result["description"] = description
    if chunk_method is not None and "chunk_method" not in result:
        result["chunk_method"] = chunk_method
    if parser_config is not None and "parser_config" not in result:
        result["parser_config"] = parser_config

    return result


async def ragflow_delete_dataset(
    id: str,
    confirm: bool,
) -> dict[str, Any]:
    """Delete a dataset.

    Permanently removes a dataset and all its documents and chunks.
    This action cannot be undone.

    IMPORTANT: The confirm parameter must be set to True to prevent
    accidental deletions. If confirm is False or not provided,
    the deletion will be rejected with an error.

    Args:
        id: ID of the dataset to delete. Required.
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
            "error": "Deletion rejected: confirm parameter must be True to delete a dataset. "
                     "This is a safety measure to prevent accidental deletions.",
            "success": False,
        }

    connector = get_connector()

    result = await connector.delete_dataset(dataset_id=id)

    # Invalidate cache after deletion
    connector.invalidate_cache()

    return result


def register_dataset_tools(mcp: FastMCP) -> None:
    """Register dataset management tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """

    @mcp.tool()
    async def ragflow_create_dataset_tool(
        name: str,
        description: str | None = None,
        embedding_model: str | None = None,
        chunk_method: str | None = None,
        parser_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new RAGFlow dataset for document storage.

        Creates a knowledge base dataset with specified configuration
        for embedding and chunking documents.

        Args:
            name: Dataset name. Required.
            description: Optional description.
            embedding_model: Model for vectorization (e.g., "BAAI/bge-large-en-v1.5").
            chunk_method: Chunking strategy ("naive", "qa", "manual").
            parser_config: Additional parser options like chunk_size.

        Returns:
            Created dataset with id, name, and configuration.
        """
        return await ragflow_create_dataset(
            name=name,
            description=description,
            embedding_model=embedding_model,
            chunk_method=chunk_method,
            parser_config=parser_config,
        )

    @mcp.tool()
    async def ragflow_list_datasets_tool(
        page: int | None = None,
        page_size: int | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """List RAGFlow datasets with pagination.

        Retrieves available datasets with optional filtering by name.

        Args:
            page: Page number (1-based). Default: 1.
            page_size: Items per page. Default: 10.
            name: Filter datasets containing this name.

        Returns:
            Dictionary with 'datasets' list, 'total' count, page info.
        """
        return await ragflow_list_datasets(
            page=page,
            page_size=page_size,
            name=name,
        )

    @mcp.tool()
    async def ragflow_update_dataset_tool(
        id: str,
        name: str | None = None,
        description: str | None = None,
        chunk_method: str | None = None,
        parser_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an existing RAGFlow dataset.

        Modifies dataset configuration. Only provided fields are updated.

        Args:
            id: Dataset ID to update. Required.
            name: New dataset name.
            description: New description.
            chunk_method: New chunking method.
            parser_config: New parser configuration.

        Returns:
            Updated dataset with all current fields.
        """
        return await ragflow_update_dataset(
            id=id,
            name=name,
            description=description,
            chunk_method=chunk_method,
            parser_config=parser_config,
        )

    @mcp.tool()
    async def ragflow_delete_dataset_tool(
        id: str,
        confirm: bool,
    ) -> dict[str, Any]:
        """Delete a RAGFlow dataset permanently.

        CAUTION: This permanently removes the dataset and all its documents.
        The confirm parameter MUST be True to proceed.

        Args:
            id: Dataset ID to delete. Required.
            confirm: Must be True to confirm deletion. Required.

        Returns:
            Success status or error if confirm is not True.
        """
        return await ragflow_delete_dataset(
            id=id,
            confirm=confirm,
        )
