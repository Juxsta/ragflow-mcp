"""Tests for the RAGFlow v1 request contracts.

RAGFlow v1 (Go backend) tightened or moved several endpoints. The connector
must use the v1 contract when the server is v1 and keep the older contract
working, so these tests pin both sides of every adaptation.
"""
import pytest
from unittest.mock import AsyncMock

from src.connector import RAGFlowAPIError, RAGFlowConnector


@pytest.fixture
def connector():
    conn = RAGFlowConnector(api_key="test-key", base_url="http://localhost:9380/api/v1")
    conn.get = AsyncMock(return_value={"data": {}})
    conn.post = AsyncMock(return_value={"data": {}})
    conn.put = AsyncMock(return_value={"data": {}})
    conn.patch = AsyncMock(return_value={"data": {}})
    conn.delete = AsyncMock(return_value={"data": {}})
    return conn


class TestCreateDataset:
    """POST /datasets"""

    @pytest.mark.asyncio
    async def test_accepted_fields_are_sent(self, connector):
        await connector.create_dataset(
            name="kb", embedding_model="bge-m3", parser_config={"chunk_token_num": 128}
        )

        payload = connector.post.await_args.kwargs["json"]
        assert payload == {
            "name": "kb",
            "embedding_model": "bge-m3",
            "parser_config": {"chunk_token_num": 128},
        }

    @pytest.mark.asyncio
    async def test_unsupported_fields_are_retried_without_them(self, connector):
        """v1 rejects `description` and `chunk_method`; older servers accept them."""
        connector.post = AsyncMock(
            side_effect=[
                RAGFlowAPIError(
                    message="Extra inputs are not permitted", code=102, status_code=400
                ),
                {"data": {"id": "ds-1"}},
            ]
        )

        result = await connector.create_dataset(
            name="kb", description="text", chunk_method="naive"
        )

        assert connector.post.await_count == 2
        assert connector.post.await_args_list[0].kwargs["json"]["description"] == "text"
        assert connector.post.await_args_list[1].kwargs["json"] == {"name": "kb"}
        assert result == {"id": "ds-1"}

    @pytest.mark.asyncio
    async def test_other_errors_are_not_retried(self, connector):
        connector.post = AsyncMock(
            side_effect=RAGFlowAPIError(message="name is required", code=102, status_code=400)
        )

        with pytest.raises(RAGFlowAPIError):
            await connector.create_dataset(name="kb", description="text")

        assert connector.post.await_count == 1

    @pytest.mark.asyncio
    async def test_list_response_is_normalized_to_a_dict(self, connector):
        """Older servers answer with a list, v1 with the dataset object."""
        connector.post = AsyncMock(return_value={"data": [{"id": "ds-1"}]})

        result = await connector.create_dataset(name="kb")

        assert result == {"id": "ds-1"}


class TestUpdateDataset:
    """PUT /datasets/{id}"""

    @pytest.mark.asyncio
    async def test_chunk_method_is_retried_without_it(self, connector):
        connector.put = AsyncMock(
            side_effect=[
                RAGFlowAPIError(
                    message="Extra inputs are not permitted", code=102, status_code=400
                ),
                {"data": {"id": "ds-1"}},
            ]
        )

        await connector.update_dataset(dataset_id="ds-1", chunk_method="naive")

        assert connector.put.await_count == 2
        assert connector.put.await_args_list[1].kwargs["json"] == {}


class TestUpdateChunk:
    """PATCH /datasets/{ds}/documents/{doc}/chunks/{chunk}"""

    @pytest.mark.asyncio
    async def test_v1_route_and_field_names(self, connector):
        await connector.update_chunk(
            chunk_id="ch-1",
            content="new text",
            keywords=["a", "b"],
            dataset_id="ds-1",
            document_id="doc-1",
            questions=["q?"],
            available=False,
        )

        connector.patch.assert_awaited_once()
        args, kwargs = connector.patch.await_args
        assert args[0] == "/datasets/ds-1/documents/doc-1/chunks/ch-1"
        assert kwargs["json"] == {
            "content": "new text",
            "important_keywords": ["a", "b"],
            "questions": ["q?"],
            "available": False,
        }
        connector.put.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_null_payload_is_reported_as_success(self, connector):
        """v1 answers a chunk update with `"data": null` and a message."""
        connector.patch = AsyncMock(
            return_value={"code": 0, "data": None, "message": "chunk updated successfully"}
        )

        result = await connector.update_chunk(
            chunk_id="ch-1", content="text", dataset_id="ds-1", document_id="doc-1"
        )

        assert result["id"] == "ch-1"
        assert result["success"] is True
        assert result["message"] == "chunk updated successfully"

    @pytest.mark.asyncio
    async def test_legacy_route_without_dataset_and_document(self, connector):
        await connector.update_chunk(chunk_id="ch-1", content="new text", keywords=["a"])

        connector.put.assert_awaited_once()
        args, kwargs = connector.put.await_args
        assert args[0] == "/chunks/ch-1"
        assert kwargs["json"] == {"content": "new text", "keywords": ["a"]}


class TestDeleteChunk:
    """DELETE /datasets/{ds}/documents/{doc}/chunks"""

    @pytest.mark.asyncio
    async def test_chunk_ids_field_is_used(self, connector):
        await connector.delete_chunk(dataset_id="ds-1", document_id="doc-1", chunk_id="ch-1")

        args, kwargs = connector.delete.await_args
        assert args[0] == "/datasets/ds-1/documents/doc-1/chunks"
        assert kwargs["json"] == {"chunk_ids": ["ch-1"]}

    @pytest.mark.asyncio
    async def test_legacy_ids_field_is_used_when_rejected(self, connector):
        connector.delete = AsyncMock(
            side_effect=[
                RAGFlowAPIError(
                    message="either chunk_ids or delete_all must be provided",
                    code=500,
                    status_code=500,
                ),
                {"data": {}},
            ]
        )

        await connector.delete_chunk(dataset_id="ds-1", document_id="doc-1", chunk_id="ch-1")

        assert connector.delete.await_count == 2
        assert connector.delete.await_args_list[1].kwargs["json"] == {"ids": ["ch-1"]}


class TestStopParsing:
    """POST /datasets/{ds}/documents/stop"""

    @pytest.mark.asyncio
    async def test_v1_route_is_used(self, connector):
        await connector.stop_parsing(
            task_id="task-1", dataset_id="ds-1", document_ids=["doc-1", "doc-2"]
        )

        args, kwargs = connector.post.await_args
        assert args[0] == "/datasets/ds-1/documents/stop"
        assert kwargs["json"] == {"document_ids": ["doc-1", "doc-2"]}

    @pytest.mark.asyncio
    async def test_legacy_task_route_without_dataset(self, connector):
        await connector.stop_parsing(task_id="task-1")

        args, _ = connector.post.await_args
        assert args[0] == "/tasks/task-1/stop"


class TestAddChunkResponse:
    """POST /datasets/{ds}/documents/{doc}/chunks"""

    @pytest.mark.asyncio
    async def test_v1_chunk_object_is_unwrapped(self, connector):
        connector.post = AsyncMock(return_value={"data": {"chunk": {"id": "ch-1"}}})

        result = await connector.add_chunk(
            dataset_id="ds-1", document_id="doc-1", content="text"
        )

        assert result["id"] == "ch-1"
        assert result["document_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_legacy_chunk_list_is_unwrapped(self, connector):
        connector.post = AsyncMock(return_value={"data": {"chunks": [{"id": "ch-1"}]}})

        result = await connector.add_chunk(
            dataset_id="ds-1", document_id="doc-1", content="text"
        )

        assert result["id"] == "ch-1"
