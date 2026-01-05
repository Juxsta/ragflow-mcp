"""RAGFlowConnector - HTTP client for RAGFlow API communication.

Provides async HTTP client with:
- Bearer token authentication
- Request/response logging (DEBUG level)
- Consistent error handling with RAGFlow error codes
- Async context manager for proper lifecycle management
"""
import base64
import logging
from pathlib import Path
from typing import Any

import httpx

from src.cache import LRUCache


logger = logging.getLogger(__name__)


class RAGFlowConnectionError(Exception):
    """Exception raised when connection to RAGFlow fails."""

    pass


class RAGFlowAPIError(Exception):
    """Exception raised when RAGFlow API returns an error.

    Attributes:
        code: RAGFlow error code if available.
        message: Error message from RAGFlow.
        status_code: HTTP status code.
    """

    def __init__(
        self,
        message: str,
        code: int | None = None,
        status_code: int | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.code is not None:
            parts.append(f"(code: {self.code})")
        if self.status_code is not None:
            parts.append(f"[HTTP {self.status_code}]")
        return " ".join(parts)


class RAGFlowConnector:
    """Async HTTP client for RAGFlow API.

    Features:
    - Bearer token authentication
    - Request/response logging at DEBUG level
    - Consistent error response parsing
    - Async context manager for proper client lifecycle
    - Built-in metadata caching

    Usage:
        async with RAGFlowConnector(api_key="...", base_url="...") as connector:
            result = await connector.get("/datasets")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:9380/api/v1",
        timeout: float = 300.0,
        cache_ttl: float = 300.0,
    ):
        """Initialize the connector.

        Args:
            api_key: RAGFlow API key for authentication.
            base_url: Base URL for RAGFlow API.
            timeout: Request timeout in seconds (default 300s to allow MCP caller to manage).
            cache_ttl: TTL for cached metadata in seconds.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client: httpx.AsyncClient | None = None
        self.cache = LRUCache(max_size=1000, ttl=cache_ttl)

    async def __aenter__(self) -> "RAGFlowConnector":
        """Enter async context and create HTTP client."""
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and close HTTP client."""
        if self.client:
            await self.client.aclose()

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith("/"):
            return f"{self.base_url}{path}"
        return f"{self.base_url}/{path}"

    async def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle response and parse errors.

        Args:
            response: The HTTP response to handle.

        Returns:
            Parsed JSON response data.

        Raises:
            RAGFlowAPIError: If the response indicates an error.
        """
        logger.debug(
            "Response: status=%d, url=%s",
            response.status_code,
            response.url,
        )

        try:
            data = response.json()
        except Exception:
            data = {}

        logger.debug("Response body: %s", data)

        # Check for RAGFlow API errors (even on 200 status)
        if isinstance(data, dict):
            # RAGFlow uses code=0 for success, non-zero for errors
            code = data.get("code")
            if code is not None and code != 0:
                message = data.get("message", "Unknown RAGFlow error")
                raise RAGFlowAPIError(
                    message=message,
                    code=code,
                    status_code=response.status_code,
                )

        # Handle HTTP errors
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            message = data.get("message", response.text) if data else response.text
            code = data.get("code") if data else None
            raise RAGFlowAPIError(
                message=message,
                code=code,
                status_code=response.status_code,
            )

        return data

    async def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make GET request to RAGFlow API.

        Args:
            path: API path (e.g., "/datasets").
            params: Optional query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        if not self.client:
            raise RuntimeError("Connector not initialized. Use 'async with' context.")

        url = self._build_url(path)
        logger.debug("GET %s params=%s", url, params)

        try:
            response = await self.client.get(url, params=params)
            return await self._handle_response(response)
        except httpx.ConnectError as e:
            raise RAGFlowConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise RAGFlowConnectionError(f"Request timeout: {e}")

    async def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        data: Any = None,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make POST request to RAGFlow API.

        Args:
            path: API path.
            json: JSON body to send.
            data: Form data to send.
            files: Files to upload.

        Returns:
            Parsed JSON response.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        if not self.client:
            raise RuntimeError("Connector not initialized. Use 'async with' context.")

        url = self._build_url(path)
        logger.debug("POST %s json=%s", url, json)

        try:
            response = await self.client.post(url, json=json, data=data, files=files)
            return await self._handle_response(response)
        except httpx.ConnectError as e:
            raise RAGFlowConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise RAGFlowConnectionError(f"Request timeout: {e}")

    async def post_multipart(
        self,
        path: str,
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, Any]:
        """Make multipart POST request to RAGFlow API for file uploads.

        Args:
            path: API path.
            files: Files to upload in format {field_name: (filename, content, content_type)}.

        Returns:
            Parsed JSON response.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        if not self.client:
            raise RuntimeError("Connector not initialized. Use 'async with' context.")

        url = self._build_url(path)
        logger.debug("POST multipart %s files=%s", url, list(files.keys()))

        try:
            # Create a temporary client without Content-Type header for multipart
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient(
                headers=headers, timeout=httpx.Timeout(self.timeout)
            ) as temp_client:
                response = await temp_client.post(url, files=files)
                return await self._handle_response(response)
        except httpx.ConnectError as e:
            raise RAGFlowConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise RAGFlowConnectionError(f"Request timeout: {e}")

    async def put(
        self,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make PUT request to RAGFlow API.

        Args:
            path: API path.
            json: JSON body to send.

        Returns:
            Parsed JSON response.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        if not self.client:
            raise RuntimeError("Connector not initialized. Use 'async with' context.")

        url = self._build_url(path)
        logger.debug("PUT %s json=%s", url, json)

        try:
            response = await self.client.put(url, json=json)
            return await self._handle_response(response)
        except httpx.ConnectError as e:
            raise RAGFlowConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise RAGFlowConnectionError(f"Request timeout: {e}")

    async def delete(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make DELETE request to RAGFlow API.

        Args:
            path: API path.
            params: Optional query parameters.
            json: Optional JSON body.

        Returns:
            Parsed JSON response.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        if not self.client:
            raise RuntimeError("Connector not initialized. Use 'async with' context.")

        url = self._build_url(path)
        logger.debug("DELETE %s params=%s", url, params)

        try:
            # Use request() instead of delete() to support json body
            response = await self.client.request("DELETE", url, params=params, json=json)
            return await self._handle_response(response)
        except httpx.ConnectError as e:
            raise RAGFlowConnectionError(f"Connection failed: {e}")
        except httpx.TimeoutException as e:
            raise RAGFlowConnectionError(f"Request timeout: {e}")

    def invalidate_cache(self, key: str | None = None) -> None:
        """Invalidate cached data.

        Args:
            key: Specific key to invalidate. If None, clears entire cache.
        """
        if key is None:
            self.cache.clear()
        else:
            self.cache.delete(key)

    async def retrieval(
        self,
        query: str,
        similarity_threshold: float | None = None,
        top_k: int | None = None,
        keyword_weight: float | None = None,
        dataset_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform semantic retrieval against RAGFlow knowledge base.

        Args:
            query: The search query string.
            similarity_threshold: Minimum similarity score for results (0-1).
            top_k: Maximum number of chunks to return.
            keyword_weight: Weight for keyword matching (0-1).
            dataset_ids: Optional list of dataset IDs to search within.
            document_ids: Optional list of document IDs to further filter.

        Returns:
            Dictionary containing:
                - chunks: List of matching chunks with metadata
                - total: Total number of matching chunks

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        # Build request payload
        payload: dict[str, Any] = {"question": query}

        if similarity_threshold is not None:
            payload["similarity_threshold"] = similarity_threshold
        if top_k is not None:
            payload["top_k"] = top_k
        if keyword_weight is not None:
            payload["keyword_weight"] = keyword_weight
        if dataset_ids is not None:
            payload["dataset_ids"] = dataset_ids
        if document_ids is not None:
            payload["document_ids"] = document_ids

        logger.debug("Retrieval request: query=%s, payload=%s", query, payload)

        # Make POST request to retrieval endpoint
        response = await self.post("/retrieval", json=payload)

        # Extract data from response
        data = response.get("data", {})

        # Format chunks with metadata
        chunks = []
        raw_chunks = data.get("chunks", [])

        for chunk in raw_chunks:
            formatted_chunk = {
                "content": chunk.get("content", ""),
                "document_name": chunk.get("document_name", ""),
                "dataset_name": chunk.get("dataset_name", ""),
                "similarity": chunk.get("similarity", 0.0),
            }

            # Include highlight if available
            if "highlight" in chunk:
                formatted_chunk["highlight"] = chunk["highlight"]

            # Include additional metadata if present
            if "document_id" in chunk:
                formatted_chunk["document_id"] = chunk["document_id"]
            if "dataset_id" in chunk:
                formatted_chunk["dataset_id"] = chunk["dataset_id"]
            if "chunk_id" in chunk:
                formatted_chunk["chunk_id"] = chunk["chunk_id"]

            chunks.append(formatted_chunk)

        return {
            "chunks": chunks,
            "total": data.get("total", len(chunks)),
        }

    # Dataset CRUD Methods

    async def create_dataset(
        self,
        name: str,
        description: str | None = None,
        embedding_model: str | None = None,
        chunk_method: str | None = None,
        parser_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new dataset in RAGFlow.

        Args:
            name: Name of the dataset (required).
            description: Optional description of the dataset.
            embedding_model: Embedding model to use (e.g., "BAAI/bge-large-en-v1.5").
            chunk_method: Chunking method (e.g., "naive", "qa", "manual").
            parser_config: Parser configuration options.

        Returns:
            Dictionary containing created dataset with ID.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {"name": name}

        if description is not None:
            payload["description"] = description
        if embedding_model is not None:
            payload["embedding_model"] = embedding_model
        if chunk_method is not None:
            payload["chunk_method"] = chunk_method
        if parser_config is not None:
            payload["parser_config"] = parser_config

        logger.debug("Creating dataset: %s", payload)

        response = await self.post("/datasets", json=payload)

        # Extract dataset data from response
        data = response.get("data", {})

        return data

    async def list_datasets(
        self,
        page: int | None = None,
        page_size: int | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """List datasets with optional pagination and filtering.

        Args:
            page: Page number (1-based).
            page_size: Number of items per page.
            name: Optional name filter.

        Returns:
            Dictionary containing:
                - datasets: List of dataset objects
                - total: Total number of datasets
                - page: Current page number
                - page_size: Items per page

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        # Check cache first
        cache_key = f"datasets:page={page}:size={page_size}:name={name}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for datasets list")
            return cached

        params: dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size
        if name is not None:
            params["name"] = name

        logger.debug("Listing datasets: %s", params)

        response = await self.get("/datasets", params=params if params else None)

        # Extract data from response
        data = response.get("data", [])

        # Handle both list and dict response formats
        if isinstance(data, list):
            datasets = data
            total = len(data)
        else:
            datasets = data.get("datasets", [])
            total = data.get("total", len(datasets))

        result = {
            "datasets": datasets,
            "total": total,
            "page": page or 1,
            "page_size": page_size or 10,
        }

        # Cache the result
        self.cache.set(cache_key, result)

        return result

    async def update_dataset(
        self,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None,
        chunk_method: str | None = None,
        parser_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an existing dataset.

        Args:
            dataset_id: ID of the dataset to update (required).
            name: New name for the dataset.
            description: New description.
            chunk_method: New chunking method.
            parser_config: New parser configuration.

        Returns:
            Dictionary containing updated dataset.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {}

        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if chunk_method is not None:
            payload["chunk_method"] = chunk_method
        if parser_config is not None:
            payload["parser_config"] = parser_config

        logger.debug("Updating dataset %s: %s", dataset_id, payload)

        response = await self.put(f"/datasets/{dataset_id}", json=payload)

        # Extract data from response
        data = response.get("data", {})

        return data

    async def delete_dataset(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Delete a dataset.

        Args:
            dataset_id: ID of the dataset to delete (required).

        Returns:
            Dictionary containing success status.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Deleting dataset: %s", dataset_id)

        # RAGFlow uses bulk DELETE endpoint with ids in body
        response = await self.delete("/datasets", json={"ids": [dataset_id]})

        return {
            "success": True,
            "message": response.get("message", "Dataset deleted successfully"),
        }

    # Document Management Methods

    async def upload_document(
        self,
        dataset_id: str,
        file_path: str | None = None,
        base64_content: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Upload a document to a dataset.

        Args:
            dataset_id: ID of the target dataset (required).
            file_path: Local file path to upload. Mutually exclusive with base64_content.
            base64_content: Base64-encoded file content. Mutually exclusive with file_path.
            filename: Filename when using base64_content (required with base64_content).

        Returns:
            Dictionary containing uploaded document with ID.

        Raises:
            ValueError: If input validation fails.
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        # Validate input - exactly one method must be provided
        if file_path and base64_content:
            raise ValueError(
                "Provide either file_path OR base64_content, not both"
            )
        if not file_path and not base64_content:
            raise ValueError(
                "Either file_path or base64_content must be provided"
            )
        if base64_content and not filename:
            raise ValueError(
                "filename is required when using base64_content"
            )

        # Prepare file content
        if file_path:
            path = Path(file_path)
            if not path.exists():
                raise ValueError(f"File not found: {file_path}")
            content = path.read_bytes()
            upload_filename = path.name
        else:
            # Decode base64 content
            content = base64.b64decode(base64_content)
            upload_filename = filename

        # Determine content type
        content_type = self._guess_content_type(upload_filename)

        logger.debug(
            "Uploading document to dataset %s: filename=%s, size=%d",
            dataset_id,
            upload_filename,
            len(content),
        )

        # Upload using multipart form
        files = {"file": (upload_filename, content, content_type)}
        response = await self.post_multipart(
            f"/datasets/{dataset_id}/documents",
            files=files,
        )

        # Extract document data from response
        data = response.get("data", {})

        # If data is a list, get the first item (common API pattern)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        return data

    def _guess_content_type(self, filename: str) -> str:
        """Guess content type based on file extension.

        Args:
            filename: Name of the file.

        Returns:
            MIME content type string.
        """
        ext = Path(filename).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".html": "text/html",
            ".htm": "text/html",
            ".csv": "text/csv",
            ".json": "application/json",
            ".xml": "application/xml",
        }
        return content_types.get(ext, "application/octet-stream")

    async def list_documents(
        self,
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

        Args:
            dataset_id: ID of the dataset (required).
            name: Filter by document name.
            status: Filter by status (e.g., "pending", "parsing", "parsed", "failed").
            file_type: Filter by file type (e.g., "pdf", "txt", "docx").
            created_after: Filter documents created after this date (ISO format).
            created_before: Filter documents created before this date (ISO format).
            page: Page number (1-based).
            page_size: Number of items per page.

        Returns:
            Dictionary containing:
                - documents: List of document objects
                - total: Total number of documents
                - page: Current page number
                - page_size: Items per page

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        # Check cache first
        cache_key = (
            f"documents:{dataset_id}:name={name}:status={status}:"
            f"type={file_type}:after={created_after}:before={created_before}:"
            f"page={page}:size={page_size}"
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for documents list")
            return cached

        params: dict[str, Any] = {}
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        if file_type is not None:
            params["file_type"] = file_type
        if created_after is not None:
            params["created_after"] = created_after
        if created_before is not None:
            params["created_before"] = created_before
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size

        logger.debug("Listing documents in dataset %s: %s", dataset_id, params)

        response = await self.get(
            f"/datasets/{dataset_id}/documents",
            params=params if params else None,
        )

        # Extract data from response
        data = response.get("data", {})

        # Handle both API response formats: "docs" (actual) or "documents" (expected)
        if isinstance(data, dict):
            documents = data.get("docs", data.get("documents", []))
            total = data.get("total", len(documents) if isinstance(documents, list) else 0)
        elif isinstance(data, list):
            documents = data
            total = len(data)
        else:
            documents = []
            total = 0

        result = {
            "documents": documents,
            "total": total,
            "page": page or 1,
            "page_size": page_size or 10,
        }

        # Cache the result
        self.cache.set(cache_key, result)

        return result

    async def parse_document(
        self,
        dataset_id: str,
        document_id: str,
        chunk_method: str | None = None,
    ) -> dict[str, Any]:
        """Trigger parsing of a document (async).

        Args:
            dataset_id: ID of the dataset containing the document (required).
            document_id: ID of the document to parse (required).
            chunk_method: Optional chunk method override.

        Returns:
            Dictionary containing:
                - status: Current status (e.g., "processing")
                - document_id: The document being parsed

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {"document_ids": [document_id]}

        logger.debug("Parsing document %s in dataset %s", document_id, dataset_id)

        response = await self.post(
            f"/datasets/{dataset_id}/chunks",
            json=payload,
        )

        # Return success with document info
        return {
            "document_id": document_id,
            "dataset_id": dataset_id,
            "status": "processing",
        }

    async def get_parse_status(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        """Get the status of a parsing task.

        Args:
            task_id: ID of the parsing task.

        Returns:
            Dictionary containing:
                - task_id: The task ID
                - status: Current status (e.g., "processing", "completed", "failed")
                - progress: Progress percentage (0-100)

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Getting parse status for task %s", task_id)

        response = await self.get(f"/tasks/{task_id}")

        # Extract data from response
        data = response.get("data", {})

        # Ensure task_id is included in response
        if "task_id" not in data:
            data["task_id"] = task_id

        return data

    async def download_document(
        self,
        document_id: str,
    ) -> dict[str, Any]:
        """Download a document's content.

        Args:
            document_id: ID of the document to download (required).

        Returns:
            Dictionary containing:
                - id: Document ID
                - name: Document filename
                - content: Document content (base64 encoded for binary files)
                - content_type: MIME type of the content
                - is_base64: Whether content is base64 encoded

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Downloading document %s", document_id)

        response = await self.get(f"/documents/{document_id}")

        # Extract data from response
        data = response.get("data", {})

        # Ensure document_id is included
        if "id" not in data:
            data["id"] = document_id

        return data

    async def delete_document(
        self,
        dataset_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        """Delete a document.

        Args:
            dataset_id: ID of the dataset containing the document (required).
            document_id: ID of the document to delete (required).

        Returns:
            Dictionary containing success status.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Deleting document: %s from dataset %s", document_id, dataset_id)

        # RAGFlow uses bulk DELETE endpoint with ids in body
        response = await self.delete(
            f"/datasets/{dataset_id}/documents",
            json={"ids": [document_id]}
        )

        return {
            "success": True,
            "message": response.get("message", "Document deleted successfully"),
        }

    async def stop_parsing(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        """Stop an active parsing job.

        Args:
            task_id: ID of the parsing task to stop (required).

        Returns:
            Dictionary containing:
                - task_id: The stopped task ID
                - status: New status (e.g., "cancelled")
                - message: Confirmation message

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Stopping parsing task: %s", task_id)

        response = await self.post(f"/tasks/{task_id}/stop")

        # Extract data from response
        data = response.get("data", {})

        # Ensure task_id is included
        if "task_id" not in data:
            data["task_id"] = task_id

        # Add default status if not present
        if "status" not in data:
            data["status"] = "cancelled"

        return data

    # Chunk Management Methods

    async def add_chunk(
        self,
        document_id: str,
        content: str,
        keywords: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a chunk to a document.

        Args:
            document_id: ID of the document to add the chunk to (required).
            content: Text content of the chunk (required).
            keywords: Optional list of keywords for the chunk.
            metadata: Optional metadata dictionary for the chunk.

        Returns:
            Dictionary containing the created chunk with:
                - id: Unique identifier for the chunk
                - content: The chunk content
                - keywords: List of keywords
                - document_id: Parent document ID
                - created_at: Creation timestamp

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {"content": content}

        if keywords is not None:
            payload["keywords"] = keywords
        if metadata is not None:
            payload["metadata"] = metadata

        logger.debug("Adding chunk to document %s: content_length=%d", document_id, len(content))

        response = await self.post(f"/documents/{document_id}/chunks", json=payload)

        # Extract chunk data from response
        data = response.get("data", {})

        # Ensure document_id is included
        if "document_id" not in data:
            data["document_id"] = document_id

        return data

    async def list_chunks(
        self,
        document_id: str,
        dataset_id: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        """List chunks in a document with pagination.

        Args:
            document_id: ID of the document (required).
            dataset_id: ID of the dataset containing the document (optional, for proper API path).
            page: Page number (1-based).
            page_size: Number of items per page.

        Returns:
            Dictionary containing:
                - chunks: List of chunk objects with id, content, keywords
                - total: Total number of chunks
                - page: Current page number
                - page_size: Items per page

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        params: dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size

        logger.debug("Listing chunks in document %s: %s", document_id, params)

        # Use full path with dataset_id if provided, otherwise try shorter path
        if dataset_id:
            endpoint = f"/datasets/{dataset_id}/documents/{document_id}/chunks"
        else:
            endpoint = f"/documents/{document_id}/chunks"

        response = await self.get(
            endpoint,
            params=params if params else None,
        )

        # Extract data from response
        data = response.get("data", {})

        result = {
            "chunks": data.get("chunks", data) if isinstance(data, dict) else data,
            "total": data.get("total", len(data) if isinstance(data, list) else 0),
            "page": page or 1,
            "page_size": page_size or 10,
        }

        # Handle case where data is a list directly
        if isinstance(data, list):
            result["chunks"] = data
            result["total"] = len(data)

        return result

    async def update_chunk(
        self,
        chunk_id: str,
        content: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a chunk's content and/or keywords.

        Args:
            chunk_id: ID of the chunk to update (required).
            content: New content for the chunk.
            keywords: New keywords for the chunk.

        Returns:
            Dictionary containing the updated chunk with:
                - id: Chunk ID
                - content: Updated content
                - keywords: Updated keywords
                - updated_at: Update timestamp

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {}

        if content is not None:
            payload["content"] = content
        if keywords is not None:
            payload["keywords"] = keywords

        logger.debug("Updating chunk %s: %s", chunk_id, list(payload.keys()))

        response = await self.put(f"/chunks/{chunk_id}", json=payload)

        # Extract data from response
        data = response.get("data", {})

        # Ensure chunk_id is included
        if "id" not in data:
            data["id"] = chunk_id

        return data

    async def delete_chunk(
        self,
        dataset_id: str,
        document_id: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        """Delete a single chunk.

        Args:
            dataset_id: ID of the dataset (required).
            document_id: ID of the document (required).
            chunk_id: ID of the chunk to delete (required).

        Returns:
            Dictionary containing success status.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Deleting chunk: %s from document %s", chunk_id, document_id)

        # RAGFlow uses bulk DELETE endpoint with ids in body
        response = await self.delete(
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            json={"ids": [chunk_id]}
        )

        return {
            "success": True,
            "message": response.get("message", "Chunk deleted successfully"),
        }

    async def delete_chunks_batch(
        self,
        dataset_id: str,
        document_id: str,
        chunk_ids: list[str],
    ) -> dict[str, Any]:
        """Delete multiple chunks in batch.

        Args:
            dataset_id: ID of the dataset (required).
            document_id: ID of the document (required).
            chunk_ids: List of chunk IDs to delete (required).

        Returns:
            Dictionary containing:
                - success: Whether deletion succeeded
                - deleted_count: Number of chunks deleted
                - message: Confirmation message

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Deleting chunks batch: %s", chunk_ids)

        response = await self.delete(
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
            json={"ids": chunk_ids}
        )

        return {
            "success": True,
            "deleted_count": len(chunk_ids),
            "message": response.get("message", f"{len(chunk_ids)} chunks deleted successfully"),
        }

    # Chat Assistant and Session Management Methods

    async def create_chat(
        self,
        name: str,
        dataset_ids: list[str] | None = None,
        llm_config: dict[str, Any] | None = None,
        prompt_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new chat assistant.

        Args:
            name: Name of the chat assistant (required).
            dataset_ids: List of dataset IDs to associate with the assistant.
            llm_config: LLM configuration (model, temperature, etc.).
            prompt_config: Prompt configuration (system_prompt, etc.).

        Returns:
            Dictionary containing the created chat assistant with:
                - id: Unique identifier for the chat assistant
                - name: Chat assistant name
                - dataset_ids: Associated dataset IDs
                - llm_config: LLM configuration
                - prompt_config: Prompt configuration
                - created_at: Creation timestamp

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {"name": name}

        if dataset_ids is not None:
            payload["dataset_ids"] = dataset_ids
        if llm_config is not None:
            payload["llm_config"] = llm_config
        if prompt_config is not None:
            payload["prompt_config"] = prompt_config

        logger.debug("Creating chat assistant: %s", payload)

        response = await self.post("/chats", json=payload)

        # Extract data from response
        data = response.get("data", {})

        return data

    async def list_chats(
        self,
        name: str | None = None,
    ) -> dict[str, Any]:
        """List chat assistants with optional name filter.

        Args:
            name: Optional filter to search chat assistants by name.

        Returns:
            Dictionary containing:
                - chats: List of chat assistant objects
                - total: Total number of chat assistants

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        params: dict[str, Any] = {}
        if name is not None:
            params["name"] = name

        logger.debug("Listing chat assistants: %s", params)

        response = await self.get("/chats", params=params if params else None)

        # Extract data from response
        data = response.get("data", [])

        # Handle both list and dict response formats
        if isinstance(data, list):
            chats = data
            total = len(data)
        else:
            chats = data.get("chats", [])
            total = data.get("total", len(chats))

        return {
            "chats": chats,
            "total": total,
        }

    async def update_chat(
        self,
        chat_id: str,
        name: str | None = None,
        dataset_ids: list[str] | None = None,
        llm_config: dict[str, Any] | None = None,
        prompt_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update an existing chat assistant.

        Args:
            chat_id: ID of the chat assistant to update (required).
            name: New name for the chat assistant.
            dataset_ids: New list of dataset IDs.
            llm_config: New LLM configuration.
            prompt_config: New prompt configuration.

        Returns:
            Dictionary containing the updated chat assistant.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {}

        if name is not None:
            payload["name"] = name
        if dataset_ids is not None:
            payload["dataset_ids"] = dataset_ids
        if llm_config is not None:
            payload["llm_config"] = llm_config
        if prompt_config is not None:
            payload["prompt_config"] = prompt_config

        logger.debug("Updating chat assistant %s: %s", chat_id, payload)

        response = await self.put(f"/chats/{chat_id}", json=payload)

        # Extract data from response
        data = response.get("data", {})

        # Ensure chat_id is included
        if "id" not in data:
            data["id"] = chat_id

        return data

    async def delete_chat(
        self,
        chat_id: str,
    ) -> dict[str, Any]:
        """Delete a chat assistant.

        Args:
            chat_id: ID of the chat assistant to delete (required).

        Returns:
            Dictionary containing success status.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Deleting chat assistant: %s", chat_id)

        # RAGFlow uses bulk DELETE endpoint with ids in body
        response = await self.delete("/chats", json={"ids": [chat_id]})

        return {
            "success": True,
            "message": response.get("message", "Chat assistant deleted successfully"),
        }

    async def create_session(
        self,
        chat_id: str,
    ) -> dict[str, Any]:
        """Create a new session for a chat assistant.

        Args:
            chat_id: ID of the chat assistant (required).

        Returns:
            Dictionary containing the created session with:
                - id: Unique identifier for the session
                - chat_id: Parent chat assistant ID
                - created_at: Creation timestamp
                - messages: Empty list of messages

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Creating session for chat assistant %s", chat_id)

        response = await self.post(f"/chats/{chat_id}/sessions", json={})

        # Extract data from response
        data = response.get("data", {})

        # Ensure chat_id is included
        if "chat_id" not in data:
            data["chat_id"] = chat_id

        return data

    async def list_sessions(
        self,
        chat_id: str,
    ) -> dict[str, Any]:
        """List sessions for a chat assistant.

        Args:
            chat_id: ID of the chat assistant (required).

        Returns:
            Dictionary containing:
                - sessions: List of session objects
                - total: Total number of sessions

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Listing sessions for chat assistant %s", chat_id)

        response = await self.get(f"/chats/{chat_id}/sessions")

        # Extract data from response
        data = response.get("data", [])

        # Handle both list and dict response formats
        if isinstance(data, list):
            sessions = data
            total = len(data)
        else:
            sessions = data.get("sessions", [])
            total = data.get("total", len(sessions))

        return {
            "sessions": sessions,
            "total": total,
        }

    async def send_message(
        self,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Send a message to a session and get a response.

        Args:
            session_id: ID of the session (required).
            message: The message to send (required).

        Returns:
            Dictionary containing:
                - session_id: The session ID
                - message: The sent message
                - response: The assistant's response
                - sources: List of source citations (if available)
                - created_at: Message timestamp

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        payload: dict[str, Any] = {"message": message}

        logger.debug("Sending message to session %s: %s", session_id, message[:50])

        response = await self.post(f"/sessions/{session_id}/messages", json=payload)

        # Extract data from response
        data = response.get("data", {})

        # Ensure session_id and message are included
        if "session_id" not in data:
            data["session_id"] = session_id
        if "message" not in data:
            data["message"] = message

        return data

    # GraphRAG and RAPTOR Methods

    async def build_graph(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Trigger knowledge graph construction for a dataset.

        Args:
            dataset_id: ID of the dataset to build the graph for (required).

        Returns:
            Dictionary containing:
                - dataset_id: The dataset being processed
                - status: Current status (e.g., "processing")
                - message: Status message

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Building knowledge graph for dataset %s", dataset_id)

        response = await self.post(f"/datasets/{dataset_id}/run_graphrag")

        # Extract data from response
        data = response.get("data", {})

        # Ensure dataset_id is included
        if "dataset_id" not in data:
            data["dataset_id"] = dataset_id

        return data

    async def get_graph_status(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Get the status of a graph construction task.

        Args:
            dataset_id: ID of the dataset to check graph status for.

        Returns:
            Dictionary containing:
                - dataset_id: The dataset ID
                - status: Current status (e.g., "processing", "completed", "failed")
                - progress: Progress percentage (0-100)
                - message: Status message

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Getting graph status for dataset %s", dataset_id)

        response = await self.get(f"/datasets/{dataset_id}/trace_graphrag")

        # Extract data from response
        data = response.get("data", {})

        # Ensure dataset_id is included
        if "dataset_id" not in data:
            data["dataset_id"] = dataset_id

        return data

    async def get_graph(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Retrieve the knowledge graph for a dataset.

        Args:
            dataset_id: ID of the dataset (required).

        Returns:
            Dictionary containing:
                - dataset_id: The dataset ID
                - entities: List of entities with id, name, type, properties
                - relationships: List of relationships with id, source, target, type
                - statistics: Graph statistics (entity_count, relationship_count)

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Getting knowledge graph for dataset %s", dataset_id)

        response = await self.get(f"/datasets/{dataset_id}/knowledge_graph")

        # Extract data from response
        data = response.get("data", {})

        # Ensure dataset_id is included
        if "dataset_id" not in data:
            data["dataset_id"] = dataset_id

        # Ensure entities and relationships exist (even if empty)
        if "entities" not in data:
            data["entities"] = []
        if "relationships" not in data:
            data["relationships"] = []

        return data

    async def delete_graph(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Delete the knowledge graph for a dataset.

        Args:
            dataset_id: ID of the dataset (required).

        Returns:
            Dictionary containing success status.

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Deleting knowledge graph for dataset %s", dataset_id)

        response = await self.delete(f"/datasets/{dataset_id}/knowledge_graph")

        return {
            "success": True,
            "message": response.get("message", "Knowledge graph deleted successfully"),
        }

    async def build_raptor(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Trigger RAPTOR tree construction for a dataset.

        Args:
            dataset_id: ID of the dataset to build the RAPTOR tree for (required).

        Returns:
            Dictionary containing:
                - dataset_id: The dataset being processed
                - status: Current status (e.g., "processing")
                - message: Status message

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Building RAPTOR tree for dataset %s", dataset_id)

        response = await self.post(f"/datasets/{dataset_id}/run_raptor")

        # Extract data from response
        data = response.get("data", {})

        # Ensure dataset_id is included
        if "dataset_id" not in data:
            data["dataset_id"] = dataset_id

        return data

    async def get_raptor_status(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Get the status of a RAPTOR construction task.

        Args:
            dataset_id: ID of the dataset to check RAPTOR status for.

        Returns:
            Dictionary containing:
                - dataset_id: The dataset ID
                - status: Current status (e.g., "processing", "completed", "failed")
                - progress: Progress percentage (0-100)
                - message: Status message

        Raises:
            RAGFlowConnectionError: If connection fails.
            RAGFlowAPIError: If API returns an error.
        """
        logger.debug("Getting RAPTOR status for dataset %s", dataset_id)

        response = await self.get(f"/datasets/{dataset_id}/trace_raptor")

        # Extract data from response
        data = response.get("data", {})

        # Ensure dataset_id is included
        if "dataset_id" not in data:
            data["dataset_id"] = dataset_id

        return data
