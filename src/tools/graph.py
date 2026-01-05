"""GraphRAG and RAPTOR tools for RAGFlow MCP Server.

Provides operations for knowledge graph and RAPTOR tree management including:
- Build knowledge graph
- Check graph construction status
- Retrieve graph entities and relationships
- Delete knowledge graph with confirmation
- Build RAPTOR tree
- Check RAPTOR construction status
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


async def ragflow_build_graph(
    dataset_id: str,
) -> dict[str, Any]:
    """Trigger knowledge graph construction for a dataset.

    Starts the asynchronous process of building a knowledge graph from
    the documents in the specified dataset. The graph construction extracts
    entities and relationships from the content.

    Args:
        dataset_id: ID of the dataset to build the graph for. Required.

    Returns:
        Dictionary containing:
            - task_id: ID for tracking construction progress
            - dataset_id: The dataset being processed
            - status: Current status (e.g., "processing")
            - message: Status message
    """
    connector = get_connector()

    result = await connector.build_graph(dataset_id=dataset_id)

    return result


async def ragflow_graph_status(
    dataset_id: str,
) -> dict[str, Any]:
    """Check the status of a graph construction task.

    Polls the status of an ongoing or completed graph construction task.
    Use this to monitor progress of long-running graph building operations.

    Args:
        dataset_id: ID of the dataset to check graph status for. Required.

    Returns:
        Dictionary containing:
            - dataset_id: The dataset ID
            - status: Current status (e.g., "processing", "completed", "failed")
            - progress: Progress percentage (0-100)
            - message: Status message
    """
    connector = get_connector()

    result = await connector.get_graph_status(dataset_id=dataset_id)

    return result


async def ragflow_get_graph(
    dataset_id: str,
) -> dict[str, Any]:
    """Retrieve the knowledge graph for a dataset.

    Returns the constructed knowledge graph including all entities and
    their relationships. The graph must have been previously built
    using ragflow_build_graph.

    Args:
        dataset_id: ID of the dataset. Required.

    Returns:
        Dictionary containing:
            - dataset_id: The dataset ID
            - entities: List of entities with id, name, type, properties
            - relationships: List of relationships with id, source, target, type
            - statistics: Graph statistics (entity_count, relationship_count)

        If no graph exists, entities and relationships will be empty lists.
    """
    connector = get_connector()

    result = await connector.get_graph(dataset_id=dataset_id)

    return result


async def ragflow_delete_graph(
    dataset_id: str,
    confirm: bool,
) -> dict[str, Any]:
    """Delete the knowledge graph for a dataset.

    Permanently removes the knowledge graph associated with the dataset.
    This action cannot be undone.

    IMPORTANT: The confirm parameter must be set to True to prevent
    accidental deletions. If confirm is False or not provided,
    the deletion will be rejected with an error.

    Args:
        dataset_id: ID of the dataset. Required.
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
            "error": "Deletion rejected: confirm parameter must be True to delete a knowledge graph. "
                     "This is a safety measure to prevent accidental deletions.",
            "success": False,
        }

    connector = get_connector()

    result = await connector.delete_graph(dataset_id=dataset_id)

    # Invalidate cache after deletion
    connector.invalidate_cache()

    return result


async def ragflow_build_raptor(
    dataset_id: str,
) -> dict[str, Any]:
    """Trigger RAPTOR tree construction for a dataset.

    Starts the asynchronous process of building a RAPTOR (Recursive
    Abstractive Processing for Tree-Organized Retrieval) tree from
    the documents in the specified dataset.

    RAPTOR creates hierarchical summaries of document content for
    improved multi-level retrieval.

    Args:
        dataset_id: ID of the dataset to build the RAPTOR tree for. Required.

    Returns:
        Dictionary containing:
            - task_id: ID for tracking construction progress
            - dataset_id: The dataset being processed
            - status: Current status (e.g., "processing")
            - message: Status message
    """
    connector = get_connector()

    result = await connector.build_raptor(dataset_id=dataset_id)

    return result


async def ragflow_raptor_status(
    dataset_id: str,
) -> dict[str, Any]:
    """Check the status of a RAPTOR construction task.

    Polls the status of an ongoing or completed RAPTOR tree construction task.
    Use this to monitor progress of long-running RAPTOR building operations.

    Args:
        dataset_id: ID of the dataset to check RAPTOR status for. Required.

    Returns:
        Dictionary containing:
            - dataset_id: The dataset ID
            - status: Current status (e.g., "processing", "completed", "failed")
            - progress: Progress percentage (0-100)
            - message: Status message
    """
    connector = get_connector()

    result = await connector.get_raptor_status(dataset_id=dataset_id)

    return result


def register_graph_tools(mcp: FastMCP) -> None:
    """Register GraphRAG and RAPTOR tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """

    @mcp.tool()
    async def ragflow_build_graph_tool(
        dataset_id: str,
    ) -> dict[str, Any]:
        """Build a knowledge graph for a RAGFlow dataset.

        Triggers asynchronous graph construction. Returns a task_id
        for progress tracking.

        Args:
            dataset_id: Dataset ID to build graph for. Required.

        Returns:
            Task information with task_id and status.
        """
        return await ragflow_build_graph(dataset_id=dataset_id)

    @mcp.tool()
    async def ragflow_graph_status_tool(
        dataset_id: str,
    ) -> dict[str, Any]:
        """Check status of a graph construction task.

        Polls construction progress for long-running operations.

        Args:
            dataset_id: Dataset ID to check graph status for. Required.

        Returns:
            Status with progress percentage and current state.
        """
        return await ragflow_graph_status(dataset_id=dataset_id)

    @mcp.tool()
    async def ragflow_get_graph_tool(
        dataset_id: str,
    ) -> dict[str, Any]:
        """Retrieve knowledge graph for a RAGFlow dataset.

        Returns entities and relationships from the constructed graph.

        Args:
            dataset_id: Dataset ID. Required.

        Returns:
            Graph with entities, relationships, and statistics.
        """
        return await ragflow_get_graph(dataset_id=dataset_id)

    @mcp.tool()
    async def ragflow_delete_graph_tool(
        dataset_id: str,
        confirm: bool,
    ) -> dict[str, Any]:
        """Delete a knowledge graph permanently.

        CAUTION: This permanently removes the graph. The confirm
        parameter MUST be True to proceed.

        Args:
            dataset_id: Dataset ID. Required.
            confirm: Must be True to confirm deletion. Required.

        Returns:
            Success status or error if confirm is not True.
        """
        return await ragflow_delete_graph(
            dataset_id=dataset_id,
            confirm=confirm,
        )

    @mcp.tool()
    async def ragflow_build_raptor_tool(
        dataset_id: str,
    ) -> dict[str, Any]:
        """Build a RAPTOR tree for a RAGFlow dataset.

        RAPTOR creates hierarchical summaries for improved retrieval.
        Triggers asynchronous construction. Returns a task_id for
        progress tracking.

        Args:
            dataset_id: Dataset ID to build RAPTOR for. Required.

        Returns:
            Task information with task_id and status.
        """
        return await ragflow_build_raptor(dataset_id=dataset_id)

    @mcp.tool()
    async def ragflow_raptor_status_tool(
        dataset_id: str,
    ) -> dict[str, Any]:
        """Check status of a RAPTOR construction task.

        Polls construction progress for long-running operations.

        Args:
            dataset_id: Dataset ID to check RAPTOR status for. Required.

        Returns:
            Status with progress percentage and current state.
        """
        return await ragflow_raptor_status(dataset_id=dataset_id)
