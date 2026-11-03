"""Retrieval tool for RAGFlow MCP Server.

Provides semantic retrieval capabilities using the RAGFlow API.
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


async def ragflow_retrieval(
    query: str,
    similarity_threshold: float | None = None,
    top_k: int | None = None,
    keyword_weight: float | None = None,
    dataset_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
    page: int | None = None,
    page_size: int | None = None,
    vector_similarity_weight: float | None = None,
    rerank_id: str | None = None,
    keyword: bool | None = None,
    force_refresh: bool | None = None,
    highlight: bool | None = None,
    knn_top_k: int | None = None,
    knn_num_candidates: int | None = None,
    use_kg: bool | None = None,
    ) -> dict[str, Any]:
    """Perform semantic retrieval against RAGFlow knowledge base.

    Search for relevant chunks across datasets using natural language queries.
    Returns chunks with content, metadata, and similarity scores.

    Args:
        query: The search query string. Required.
        similarity_threshold: Minimum similarity score for results (0-1).
            Higher values return more relevant but fewer results.
        top_k: Maximum number of chunks to return. Default is server-configured.
        keyword_weight: Weight for keyword matching vs semantic similarity (0-1).
            Higher values give more weight to exact keyword matches.
        dataset_ids: Optional list of dataset IDs to limit search scope.
            If not provided, searches all accessible datasets.
        document_ids: Optional list of document IDs to further filter results.
            Can only be used with documents from the specified datasets.
        page: 1-based page of results (RAGFlow v1). Do not combine with
            top_k: top_k caps the candidate set, so paging then slices
            inside it and consecutive pages overlap.
        page_size: Number of results per page (RAGFlow v1).
        vector_similarity_weight: Weight of the vector score in the hybrid score
            (0-1, RAGFlow v1). The keyword weight is its complement.
        rerank_id: ID of a rerank model used to reorder the candidates
            (RAGFlow v1).
        keyword: Run keyword matching only (RAGFlow v1).
        force_refresh: Bypass cached retrieval results (RAGFlow v1).
        highlight: Ask the server to mark the matched terms (RAGFlow v1).
        knn_top_k: Number of candidates the vector stage keeps (RAGFlow v1).
        knn_num_candidates: Candidate pool size for the vector stage
            (RAGFlow v1).
        use_kg: Include knowledge-graph expansion (RAGFlow v1).

    Returns:
        Dictionary containing:
            - chunks: List of matching chunks, each with:
                - content: The chunk text content
                - document_name: Name of the source document
                - dataset_name: Name of the source dataset
                - similarity: Similarity score (0-1)
                - highlight: Keyword-highlighted text (when available)
            - total: Total number of matching chunks
    """
    connector = get_connector()

    result = await connector.retrieval(
        query=query,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        keyword_weight=keyword_weight,
        dataset_ids=dataset_ids,
        document_ids=document_ids,
        page=page,
        page_size=page_size,
        vector_similarity_weight=vector_similarity_weight,
        rerank_id=rerank_id,
        keyword=keyword,
        force_refresh=force_refresh,
        highlight=highlight,
        knn_top_k=knn_top_k,
        knn_num_candidates=knn_num_candidates,
        use_kg=use_kg,
    )

    return result


def register_retrieval_tools(mcp: FastMCP) -> None:
    """Register retrieval tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance to register tools with.
    """
    @mcp.tool()
    async def ragflow_retrieval_tool(
        query: str,
        similarity_threshold: float | None = None,
        top_k: int | None = None,
        keyword_weight: float | None = None,
        dataset_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        page: int | None = None,
        page_size: int | None = None,
        vector_similarity_weight: float | None = None,
        rerank_id: str | None = None,
        keyword: bool | None = None,
        force_refresh: bool | None = None,
        highlight: bool | None = None,
        knn_top_k: int | None = None,
        knn_num_candidates: int | None = None,
        use_kg: bool | None = None,
    ) -> dict[str, Any]:
        """Search RAGFlow knowledge base using semantic retrieval.

        Performs semantic search across configured datasets to find relevant
        chunks based on natural language queries. Returns chunks with content,
        metadata, similarity scores, and keyword highlighting.

        Args:
            query: The search query string. Required.
            similarity_threshold: Minimum similarity score (0-1). Higher = more relevant.
            top_k: Maximum number of chunks to return.
            keyword_weight: Weight for keyword matching (0-1). Higher = more keyword focus.
            dataset_ids: List of dataset IDs to search within. Omit to search all.
            document_ids: List of document IDs to filter results.
            page: 1-based page of results (RAGFlow v1). Do not combine with
                top_k: top_k caps the candidate set, so paging then slices
                inside it and consecutive pages overlap.
            page_size: Number of results per page (RAGFlow v1).
            vector_similarity_weight: Weight of the vector score in the hybrid
                score (0-1, RAGFlow v1); the keyword weight is its complement.
            rerank_id: ID of a rerank model used to reorder candidates
                (RAGFlow v1).
            keyword: Run keyword matching only (RAGFlow v1).
            force_refresh: Bypass cached retrieval results (RAGFlow v1).
            highlight: Mark the matched terms in the returned chunks
                (RAGFlow v1).
            knn_top_k: Number of candidates the vector stage keeps (RAGFlow v1).
            knn_num_candidates: Candidate pool size for the vector stage
                (RAGFlow v1).
            use_kg: Include knowledge-graph expansion (RAGFlow v1).

        Returns:
            Dictionary with 'chunks' list and 'total' count.
            Each chunk contains: content, document_name, dataset_name, similarity, highlight.
        """
        return await ragflow_retrieval(
            query=query,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            keyword_weight=keyword_weight,
            dataset_ids=dataset_ids,
            document_ids=document_ids,
            page=page,
            page_size=page_size,
            vector_similarity_weight=vector_similarity_weight,
            rerank_id=rerank_id,
            keyword=keyword,
            force_refresh=force_refresh,
            highlight=highlight,
            knn_top_k=knn_top_k,
            knn_num_candidates=knn_num_candidates,
            use_kg=use_kg,
        )
