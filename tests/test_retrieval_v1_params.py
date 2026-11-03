"""Tests for the RAGFlow v1 retrieval parameters and the response mapping."""
import base64

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestRetrievalPayload:
    """The connector must send exactly the parameters the caller supplied."""

    @pytest.fixture
    def connector(self):
        from src.connector import RAGFlowConnector

        connector = RAGFlowConnector(
            api_key="test-api-key", base_url="http://localhost:9380/api/v1"
        )
        connector.post = AsyncMock(return_value={"data": {"chunks": [], "total": 0}})
        return connector

    @pytest.mark.asyncio
    async def test_only_supplied_parameters_are_sent(self, connector):
        """Unset parameters stay out of the request body."""
        await connector.retrieval(query="hello", top_k=5)

        payload = connector.post.await_args.kwargs["json"]
        assert payload == {"question": "hello", "top_k": 5}

    @pytest.mark.asyncio
    async def test_keyword_weight_maps_to_vector_similarity_weight(self, connector):
        """RAGFlow v1 dropped `keyword_weight` and derives it from the vector weight."""
        await connector.retrieval(query="hello", keyword_weight=0.7)

        payload = connector.post.await_args.kwargs["json"]
        assert payload["keyword_weight"] == 0.7
        assert payload["vector_similarity_weight"] == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_explicit_vector_weight_wins(self, connector):
        """An explicit vector weight is not overwritten by the legacy mapping."""
        await connector.retrieval(
            query="hello", keyword_weight=0.7, vector_similarity_weight=0.9
        )

        payload = connector.post.await_args.kwargs["json"]
        assert payload["vector_similarity_weight"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_v1_parameters_are_forwarded(self, connector):
        """The RAGFlow v1 search parameters reach the request body."""
        await connector.retrieval(
            query="hello",
            dataset_ids=["ds-1"],
            document_ids=["doc-1"],
            page=2,
            page_size=25,
            rerank_id="rerank-model",
            keyword=True,
            force_refresh=True,
            highlight=True,
            knn_top_k=200,
            knn_num_candidates=1000,
            use_kg=True,
        )

        payload = connector.post.await_args.kwargs["json"]
        assert payload["dataset_ids"] == ["ds-1"]
        assert payload["document_ids"] == ["doc-1"]
        assert payload["page"] == 2
        assert payload["page_size"] == 25
        assert payload["rerank_id"] == "rerank-model"
        assert payload["keyword"] is True
        assert payload["force_refresh"] is True
        assert payload["highlight"] is True
        assert payload["knn_top_k"] == 200
        assert payload["knn_num_candidates"] == 1000
        assert payload["use_kg"] is True

    @pytest.mark.asyncio
    async def test_score_split_is_returned(self, connector):
        """The hybrid score split and the chunk id survive the response mapping."""
        connector.post = AsyncMock(
            return_value={
                "data": {
                    "chunks": [
                        {
                            "id": "chunk-1",
                            "content": "text",
                            "document_name": "doc.md",
                            "dataset_name": "kb",
                            "similarity": 0.6,
                            "vector_similarity": 0.81,
                            "term_similarity": 0.52,
                            "positions": [[1, 10, 20]],
                        }
                    ],
                    "total": 1,
                }
            }
        )

        result = await connector.retrieval(query="hello")

        chunk = result["chunks"][0]
        assert chunk["id"] == "chunk-1"
        assert chunk["vector_similarity"] == pytest.approx(0.81)
        assert chunk["term_similarity"] == pytest.approx(0.52)
        assert chunk["positions"] == [[1, 10, 20]]


class TestRetrievalToolParameters:
    """The tool forwards the new parameters to the connector."""

    @pytest.fixture
    def mock_connector(self):
        from src.connector import RAGFlowConnector

        connector = AsyncMock(spec=RAGFlowConnector)
        connector.retrieval = AsyncMock(return_value={"chunks": [], "total": 0})
        return connector

    @pytest.mark.asyncio
    async def test_tool_forwards_v1_parameters(self, mock_connector):
        from unittest.mock import patch

        from src.tools.retrieval import ragflow_retrieval

        with patch("src.tools.retrieval.get_connector", return_value=mock_connector):
            await ragflow_retrieval(
                query="hello",
                page=3,
                page_size=10,
                vector_similarity_weight=0.8,
                rerank_id="rm",
                keyword=False,
                force_refresh=True,
                highlight=True,
                knn_top_k=100,
                knn_num_candidates=500,
                use_kg=False,
            )

        kwargs = mock_connector.retrieval.await_args.kwargs
        assert kwargs["page"] == 3
        assert kwargs["page_size"] == 10
        assert kwargs["vector_similarity_weight"] == pytest.approx(0.8)
        assert kwargs["rerank_id"] == "rm"
        assert kwargs["keyword"] is False
        assert kwargs["force_refresh"] is True
        assert kwargs["highlight"] is True
        assert kwargs["knn_top_k"] == 100
        assert kwargs["knn_num_candidates"] == 500
        assert kwargs["use_kg"] is False


class TestDocumentDownload:
    """RAGFlow v1 serves the file behind a separate route."""

    @pytest.fixture
    def connector(self):
        from src.connector import RAGFlowConnector

        connector = RAGFlowConnector(
            api_key="test-api-key", base_url="http://localhost:9380/api/v1"
        )
        connector.get = AsyncMock(
            return_value={
                "data": {
                    "id": "doc-1",
                    "name": "note.md",
                    "suffix": "md",
                    "kb_id": "ds-1",
                }
            }
        )
        return connector

    def _file_response(self, body: bytes, content_type: str):
        response = MagicMock()
        response.content = body
        response.headers = {"content-type": content_type}
        response.raise_for_status = MagicMock()
        return response

    @pytest.mark.asyncio
    async def test_v1_metadata_response_falls_back_to_the_file_route(self, connector):
        """v1 returns metadata only, so the file route is used for the content."""
        connector.client = MagicMock()
        connector.client.get = AsyncMock(
            return_value=self._file_response(b"# hello\n", "application/octet-stream")
        )

        result = await connector.download_document(document_id="doc-1")

        assert result["content"] == "# hello\n"
        assert result["is_base64"] is False
        url = connector.client.get.await_args.args[0]
        assert url.endswith("/datasets/ds-1/documents/doc-1")

    @pytest.mark.asyncio
    async def test_inline_content_is_kept(self, connector):
        """Servers that return the content inline keep working."""
        connector.get = AsyncMock(
            return_value={
                "data": {
                    "id": "doc-1",
                    "name": "note.md",
                    "content": "# inline\n",
                    "content_type": "text/markdown",
                    "is_base64": False,
                }
            }
        )
        connector.client = MagicMock()
        connector.client.get = AsyncMock()

        result = await connector.download_document(document_id="doc-1")

        assert result["content"] == "# inline\n"
        connector.client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_binary_content_is_base64_encoded(self, connector):
        """Binary payloads are base64 encoded."""
        connector.get = AsyncMock(
            return_value={
                "data": {
                    "id": "doc-1",
                    "name": "report.pdf",
                    "suffix": "pdf",
                    "kb_id": "ds-1",
                }
            }
        )
        connector.client = MagicMock()
        connector.client.get = AsyncMock(
            return_value=self._file_response(b"\x00\x01\x02", "application/pdf")
        )

        result = await connector.download_document(document_id="doc-1")

        assert result["is_base64"] is True
        assert base64.b64decode(result["content"]) == b"\x00\x01\x02"
