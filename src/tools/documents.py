"""Document management tools for RAGFlow MCP Server.

Provides document operations for RAGFlow datasets including:
- Upload document (from file path or base64 content)
- List documents with filters
- Parse document (async and sync)
- Download document
- Delete document with confirmation
- Stop parsing job
"""
import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP


def get_connector():
    """Get the global connector instance.

    This function is imported from server module to avoid circular imports.
    It will be patched during testing.
    """
    from src.server import get_connector as _get_connector
    return _get_connector()


async def ragflow_upload_document(
    dataset_id: str,
    file_path: str | None = None,
    base64_content: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Upload a document to a RAGFlow dataset.

    Upload a document either from a local file path or from base64-encoded
    content. Exactly one input method must be provided.

    Args:
        dataset_id: ID of the target dataset. Required.
        file_path: Local file path to upload. Use this for files accessible
            on the local filesystem. Mutually exclusive with base64_content.
        base64_content: Base64-encoded file content. Use this when the file
            content is already in memory. Mutually exclusive with file_path.
        filename: Filename when using base64_content. Required when using
            base64_content, ignored when using file_path.

    Returns:
        Dictionary containing the uploaded document with:
            - id: Unique identifier for the document
            - name: Document filename
            - size: File size in bytes
            - status: Processing status (typically "pending")
            - created_at: Upload timestamp

    Raises:
        ValueError: If input validation fails (e.g., both or neither input
            methods provided, missing filename with base64_content).
    """
    connector = get_connector()

    result = await connector.upload_document(
        dataset_id=dataset_id,
        file_path=file_path,
        base64_content=base64_content,
        filename=filename,
    )

    return result


async def ragflow_list_documents(
    dataset_id: str,
    name: str | None = None,
    status: str | None = None,
    file_type: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """List documents in a dataset with optional filters.

    Retrieves documents from a dataset with support for filtering and
    pagination. Results are cached for performance.

    Args:
        dataset_id: ID of the dataset to list documents from. Required.
        name: Filter documents by name (partial match).
        status: Filter by document status.
            Options: "pending", "parsing", "parsed", "failed".
        file_type: Filter by file type (e.g., "pdf", "txt", "docx").
        created_after: Filter documents created after this date (ISO format).
        created_before: Filter documents created before this date (ISO format).
        page: Page number for pagination (1-based). Default: 1.
        page_size: Number of items per page. Default: 10.

    Returns:
        Dictionary containing:
            - documents: List of document objects with id, name, status, etc.
            - total: Total number of documents matching the filter
            - page: Current page number
            - page_size: Items per page
    """
    connector = get_connector()

    result = await connector.list_documents(
        dataset_id=dataset_id,
        name=name,
        status=status,
        file_type=file_type,
        created_after=created_after,
        created_before=created_before,
        page=page,
        page_size=page_size,
    )

    return result


async def ragflow_parse_document(
    dataset_id: str,
    document_id: str,
    chunk_method: str | None = None,
) -> dict[str, Any]:
    """Trigger parsing of a document (async).

    Initiates document parsing and returns immediately.
    Use ragflow_parse_document_sync for blocking behavior.

    Args:
        dataset_id: ID of the dataset containing the document. Required.
        document_id: ID of the document to parse. Required.
        chunk_method: Optional override for the chunking method.
            Options: "naive" (simple splitting), "qa" (Q&A pairs),
            "manual" (preserve existing structure).
            If not specified, uses the dataset's default method.

    Returns:
        Dictionary containing:
            - status: Current status (typically "processing")
            - document_id: The document being parsed
            - dataset_id: The dataset containing the document

    Note:
        This is an asynchronous operation. The document is not immediately
        available for retrieval. Check document status to monitor progress.
    """
    connector = get_connector()

    result = await connector.parse_document(
        dataset_id=dataset_id,
        document_id=document_id,
        chunk_method=chunk_method,
    )

    return result


async def ragflow_parse_document_sync(
    dataset_id: str,
    document_id: str,
    chunk_method: str | None = None,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Trigger parsing of a document and wait for completion.

    Initiates document parsing and polls for completion by checking
    the document's run status. This is a blocking operation that waits
    until parsing is complete, fails, or times out.

    Args:
        dataset_id: ID of the dataset containing the document. Required.
        document_id: ID of the document to parse. Required.
        chunk_method: Optional override for the chunking method.
            Options: "naive", "qa", "manual".
            If not specified, uses the dataset's default method.
        poll_interval: Seconds between status checks. Default: 2.0.
        timeout: Maximum seconds to wait for completion. Default: 600.0.

    Returns:
        Dictionary containing:
            - status: Final status ("completed" or "failed")
            - progress: Final progress (1.0 if completed)
            - document_id: The document that was parsed
            - dataset_id: The dataset containing the document

    Raises:
        TimeoutError: If parsing does not complete within the timeout period.
    """
    connector = get_connector()

    # Start parsing
    await connector.parse_document(
        dataset_id=dataset_id,
        document_id=document_id,
        chunk_method=chunk_method,
    )

    # Poll for completion by checking document status
    elapsed = 0.0
    while elapsed < timeout:
        docs = await connector.list_documents(dataset_id=dataset_id)
        doc = next((d for d in docs.get("documents", []) if d["id"] == document_id), None)

        if doc:
            run_status = doc.get("run", "")
            progress = doc.get("progress", 0)

            if run_status == "DONE" or progress >= 1.0:
                return {
                    "status": "completed",
                    "progress": progress,
                    "document_id": document_id,
                    "dataset_id": dataset_id,
                }
            elif run_status == "FAIL":
                return {
                    "status": "failed",
                    "progress": progress,
                    "document_id": document_id,
                    "dataset_id": dataset_id,
                    "error": doc.get("progress_msg", "Parsing failed"),
                }

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(
        f"Parsing did not complete within {timeout} seconds. "
        f"Document ID: {document_id}, Dataset ID: {dataset_id}"
    )


async def ragflow_download_document(
    document_id: str,
) -> dict[str, Any]:
    """Download a document's content.

    Retrieves the content of a document. Binary files are returned
    as base64-encoded strings.

    Args:
        document_id: ID of the document to download. Required.

    Returns:
        Dictionary containing:
            - id: Document ID
            - name: Document filename
            - content: Document content (base64 encoded for binary files)
            - content_type: MIME type of the content
            - is_base64: Whether the content is base64 encoded
    """
    connector = get_connector()

    result = await connector.download_document(document_id=document_id)

    return result


async def ragflow_delete_document(
    dataset_id: str,
    document_id: str,
    confirm: bool,
) -> dict[str, Any]:
    """Delete a document.

    Permanently removes a document and all its parsed chunks.
    This action cannot be undone.

    IMPORTANT: The confirm parameter must be set to True to prevent
    accidental deletions. If confirm is False or not provided,
    the deletion will be rejected with an error.

    Args:
        dataset_id: ID of the dataset containing the document. Required.
        document_id: ID of the document to delete. Required.
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
            "error": "Deletion rejected: confirm parameter must be True to delete a document. "
                     "This is a safety measure to prevent accidental deletions.",
            "success": False,
        }

    connector = get_connector()

    result = await connector.delete_document(
        dataset_id=dataset_id,
        document_id=document_id,
    )

    # Invalidate cache after deletion
    connector.invalidate_cache()

    return result


async def ragflow_stop_parsing(
    task_id: str,
) -> dict[str, Any]:
    """Stop an active parsing job.

    Cancels a document parsing operation that is currently in progress.
    This is useful for long-running parsing jobs that need to be aborted.

    Args:
        task_id: ID of the parsing task to stop. Required.
            This is the task_id returned by ragflow_parse_document.

    Returns:
        Dictionary containing:
            - task_id: The stopped task ID
            - status: New status (typically "cancelled")
            - message: Confirmation message
    """
    connector = get_connector()

    result = await connector.stop_parsing(task_id=task_id)

    return result


def register_document_tools(mcp: FastMCP) -> None:
    """Register document management tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """

    @mcp.tool()
    async def ragflow_upload_document_tool(
        dataset_id: str,
        file_path: str | None = None,
        base64_content: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Upload a document to a RAGFlow dataset.

        Uploads a document either from a local file or base64 content.
        Provide exactly one: file_path OR (base64_content + filename).

        Args:
            dataset_id: Target dataset ID. Required.
            file_path: Local file path to upload.
            base64_content: Base64-encoded file content.
            filename: Filename when using base64_content.

        Returns:
            Uploaded document with id, name, size, status.
        """
        return await ragflow_upload_document(
            dataset_id=dataset_id,
            file_path=file_path,
            base64_content=base64_content,
            filename=filename,
        )

    @mcp.tool()
    async def ragflow_list_documents_tool(
        dataset_id: str,
        name: str | None = None,
        status: str | None = None,
        file_type: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """List documents in a RAGFlow dataset.

        Retrieves documents with optional filtering and pagination.

        Args:
            dataset_id: Dataset ID to list documents from. Required.
            name: Filter by document name (partial match).
            status: Filter by status ("pending", "parsing", "parsed", "failed").
            file_type: Filter by file type (e.g., "pdf", "txt").
            created_after: Filter by creation date (ISO format).
            created_before: Filter by creation date (ISO format).
            page: Page number (1-based). Default: 1.
            page_size: Items per page. Default: 10.

        Returns:
            Dictionary with 'documents' list, 'total' count, page info.
        """
        return await ragflow_list_documents(
            dataset_id=dataset_id,
            name=name,
            status=status,
            file_type=file_type,
            created_after=created_after,
            created_before=created_before,
            page=page,
            page_size=page_size,
        )

    @mcp.tool()
    async def ragflow_parse_document_tool(
        dataset_id: str,
        document_id: str,
        chunk_method: str | None = None,
    ) -> dict[str, Any]:
        """Parse a document asynchronously.

        Triggers document parsing and returns immediately.
        Use ragflow_parse_document_sync_tool to wait for completion.

        Args:
            dataset_id: Dataset ID containing the document. Required.
            document_id: Document ID to parse. Required.
            chunk_method: Chunking method override ("naive", "qa", "manual").

        Returns:
            Dictionary with status, document_id, dataset_id.
        """
        return await ragflow_parse_document(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_method=chunk_method,
        )

    @mcp.tool()
    async def ragflow_parse_document_sync_tool(
        dataset_id: str,
        document_id: str,
        chunk_method: str | None = None,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Parse a document and wait for completion.

        Triggers parsing and polls until complete or timeout.
        Use this when you need to wait for parsing to finish.

        Args:
            dataset_id: Dataset ID containing the document. Required.
            document_id: Document ID to parse. Required.
            chunk_method: Chunking method override ("naive", "qa", "manual").
            poll_interval: Seconds between status checks. Default: 2.0.
            timeout: Max seconds to wait. Default: 600.0.

        Returns:
            Dictionary with status, progress, document_id, dataset_id.
        """
        return await ragflow_parse_document_sync(
            dataset_id=dataset_id,
            document_id=document_id,
            chunk_method=chunk_method,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    @mcp.tool()
    async def ragflow_download_document_tool(
        document_id: str,
    ) -> dict[str, Any]:
        """Download a document's content.

        Retrieves document content. Binary files are base64 encoded.

        Args:
            document_id: Document ID to download. Required.

        Returns:
            Dictionary with id, name, content, content_type, is_base64.
        """
        return await ragflow_download_document(document_id=document_id)

    @mcp.tool()
    async def ragflow_delete_document_tool(
        dataset_id: str,
        document_id: str,
        confirm: bool,
    ) -> dict[str, Any]:
        """Delete a document permanently.

        CAUTION: This permanently removes the document and all its chunks.
        The confirm parameter MUST be True to proceed.

        Args:
            dataset_id: Dataset ID containing the document. Required.
            document_id: Document ID to delete. Required.
            confirm: Must be True to confirm deletion. Required.

        Returns:
            Success status or error if confirm is not True.
        """
        return await ragflow_delete_document(
            dataset_id=dataset_id,
            document_id=document_id,
            confirm=confirm,
        )

    @mcp.tool()
    async def ragflow_stop_parsing_tool(
        task_id: str,
    ) -> dict[str, Any]:
        """Stop an active parsing job.

        Cancels a document parsing operation in progress.

        Args:
            task_id: Parsing task ID to stop. Required.

        Returns:
            Dictionary with task_id, status ("cancelled"), message.
        """
        return await ragflow_stop_parsing(task_id=task_id)
