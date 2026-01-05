# RAGFlow MCP Server

A comprehensive Model Context Protocol (MCP) server for [RAGFlow](https://github.com/infiniflow/ragflow) that provides full API access for semantic retrieval and knowledge base management.

## Features

- **Semantic Retrieval**: Search across datasets using natural language queries
- **Dataset Management**: Create, list, update, and delete datasets
- **Document Management**: Upload, parse, list, download, and delete documents
- **Chunk Management**: Add, list, update, and delete document chunks
- **Chat Assistants**: Create and manage chat assistants with RAG capabilities
- **Session Management**: Create and manage chat sessions
- **GraphRAG & RAPTOR**: Build and query knowledge graphs (when supported by your RAGFlow instance)

## Installation

### Prerequisites

- Python 3.10+
- RAGFlow server running and accessible (v0.16.0+ for core features)
- RAGFlow API key

> **Note:** GraphRAG and RAPTOR build APIs require RAGFlow v0.21.0 or later.

### Install from source

```bash
git clone https://github.com/Juxsta/ragflow-mcp.git
cd ragflow-mcp
pip install -e .
```

### Configure Claude Code

Add to your Claude Code MCP settings:

```bash
claude mcp add ragflow -e RAGFLOW_API_KEY=your-api-key -e RAGFLOW_URL=http://localhost:9380/api/v1 -- python -m src.server
```

Or manually add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "ragflow": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/ragflow-mcp",
      "env": {
        "RAGFLOW_API_KEY": "your-api-key",
        "RAGFLOW_URL": "http://localhost:9380/api/v1"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RAGFLOW_API_KEY` | Yes | - | Your RAGFlow API key |
| `RAGFLOW_URL` | No | `http://localhost:9380/api/v1` | RAGFlow API base URL |
| `RAGFLOW_TIMEOUT` | No | `300` | Request timeout in seconds |
| `RAGFLOW_LOG_LEVEL` | No | `INFO` | Logging level |

## Available Tools

### Retrieval
- `ragflow_retrieval_tool` - Semantic search across datasets

### Dataset Management
- `ragflow_list_datasets_tool` - List all datasets
- `ragflow_create_dataset_tool` - Create a new dataset
- `ragflow_update_dataset_tool` - Update dataset configuration
- `ragflow_delete_dataset_tool` - Delete a dataset (requires confirmation)

### Document Management
- `ragflow_list_documents_tool` - List documents in a dataset
- `ragflow_upload_document_tool` - Upload a document (file path or base64)
- `ragflow_parse_document_tool` - Trigger async document parsing
- `ragflow_parse_document_sync_tool` - Parse and wait for completion
- `ragflow_download_document_tool` - Download document content
- `ragflow_delete_document_tool` - Delete a document (requires confirmation)
- `ragflow_stop_parsing_tool` - Cancel an active parsing job

### Chunk Management
- `ragflow_list_chunks_tool` - List chunks in a document
- `ragflow_add_chunk_tool` - Add a chunk to a document
- `ragflow_update_chunk_tool` - Update chunk content/keywords
- `ragflow_delete_chunk_tool` - Delete chunks (requires confirmation)

### Chat & Sessions
- `ragflow_list_chats_tool` - List chat assistants
- `ragflow_create_chat_tool` - Create a chat assistant
- `ragflow_update_chat_tool` - Update chat configuration
- `ragflow_delete_chat_tool` - Delete a chat assistant (requires confirmation)
- `ragflow_list_sessions_tool` - List sessions for a chat
- `ragflow_create_session_tool` - Create a new session
- `ragflow_chat_tool` - Send a message and get a response

### GraphRAG & RAPTOR
- `ragflow_build_graph_tool` - Build knowledge graph for a dataset
- `ragflow_graph_status_tool` - Check graph construction status
- `ragflow_get_graph_tool` - Retrieve the knowledge graph
- `ragflow_delete_graph_tool` - Delete a knowledge graph (requires confirmation)
- `ragflow_build_raptor_tool` - Build RAPTOR tree for a dataset
- `ragflow_raptor_status_tool` - Check RAPTOR construction status

## Usage Examples

### Semantic Search
```
Query: "What is the main character's motivation?"
Dataset: your-dataset-id
```

### Upload and Parse a Document
```
1. Upload: ragflow_upload_document_tool(dataset_id, file_path="/path/to/doc.pdf")
2. Parse: ragflow_parse_document_sync_tool(document_id)
3. Search: ragflow_retrieval_tool(query="your question", dataset_ids=[dataset_id])
```

## Development

### Run Tests
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Project Structure
```
ragflow-mcp/
├── src/
│   ├── __init__.py
│   ├── server.py          # FastMCP server setup
│   ├── connector.py       # RAGFlow API client
│   ├── config.py          # Configuration management
│   ├── cache.py           # LRU cache implementation
│   └── tools/
│       ├── retrieval.py   # Semantic search
│       ├── datasets.py    # Dataset CRUD
│       ├── documents.py   # Document management
│       ├── chunks.py      # Chunk management
│       ├── chat.py        # Chat & sessions
│       └── graph.py       # GraphRAG & RAPTOR
├── tests/
│   └── ...
├── pyproject.toml
└── README.md
```

## Safety Features

All delete operations require explicit `confirm=True` parameter to prevent accidental data loss.

## License

MIT License

## Acknowledgments

- [RAGFlow](https://github.com/infiniflow/ragflow) - The RAG engine this MCP server integrates with
- [FastMCP](https://github.com/jlowin/fastmcp) - The MCP framework used for building this server
