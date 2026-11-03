"""FastMCP server for RAGFlow.

Provides MCP tools for semantic retrieval and knowledge management
through the RAGFlow API.
"""
import argparse
import asyncio
import atexit
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from src.config import Settings, get_settings
from src.connector import RAGFlowConnector
from src.tools.retrieval import register_retrieval_tools
from src.tools.datasets import register_dataset_tools
from src.tools.documents import register_document_tools
from src.tools.chunks import register_chunk_tools
from src.tools.chat import register_chat_tools
from src.tools.graph import register_graph_tools


logger = logging.getLogger(__name__)

# Global connector instance
_connector: RAGFlowConnector | None = None
_initialized: bool = False


def _initialize_connector() -> None:
    """Initialize the global connector instance (lazy initialization)."""
    global _connector, _initialized

    if _initialized:
        return

    settings = get_settings()

    # Configure logging
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting RAGFlow MCP Server")
    logger.info("Connecting to RAGFlow at %s", settings.ragflow_url)

    # Initialize connector (sync init, async connect on first use)
    _connector = RAGFlowConnector(
        api_key=settings.ragflow_api_key,
        base_url=settings.ragflow_url,
    )
    # Initialize the httpx client synchronously
    import httpx
    _connector.client = httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {settings.ragflow_api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(_connector.timeout),
    )

    _initialized = True
    logger.info("RAGFlow connector initialized")


def get_connector() -> RAGFlowConnector:
    """Get the global connector instance.

    Returns:
        The initialized RAGFlowConnector.

    Raises:
        RuntimeError: If connector is not initialized.
    """
    if not _initialized:
        _initialize_connector()
    if _connector is None:
        raise RuntimeError("Connector not initialized")
    return _connector


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle.

    Initializes and closes the RAGFlowConnector during server lifecycle.
    """
    # Initialize if not already done
    _initialize_connector()

    try:
        yield
    finally:
        global _connector, _initialized
        if _connector is not None and _connector.client is not None:
            logger = logging.getLogger(__name__)
            logger.info("Shutting down RAGFlow MCP Server")
            await _connector.client.aclose()
            _connector = None
            _initialized = False


# Create FastMCP server
mcp = FastMCP(
    "ragflow-mcp",
    instructions="""RAGFlow MCP Server - Semantic retrieval and knowledge management.

Available capabilities:
- Semantic search across datasets using ragflow_retrieval_tool
- Dataset management (create, list, update, delete)
- Document management (upload, parse, download, delete)
- Chunk management (add, list, update, delete)
- Chat assistant and session management
- GraphRAG and RAPTOR construction

Use the status resource to check server health.""",
    lifespan=lifespan,
)

# Register all tools
register_retrieval_tools(mcp)
register_dataset_tools(mcp)
register_document_tools(mcp)
register_chunk_tools(mcp)
register_chat_tools(mcp)
register_graph_tools(mcp)


def apply_tool_allowlist(server: FastMCP) -> list[str]:
    """Drop tools that are not listed in `RAGFLOW_MCP_TOOL_ALLOWLIST`.

    Every registered tool contributes its name, description and JSON schema to
    the client's tool catalogue, which costs context on every request even
    when the capability is never used. Deployments that only need a subset can
    set `RAGFLOW_MCP_TOOL_ALLOWLIST` to a comma-separated list of tool names;
    every other tool is then removed at startup.

    When the variable is unset or empty, all tools stay registered.

    Args:
        server: The FastMCP server whose tools should be filtered.

    Returns:
        The names of the tools that were removed.
    """
    raw = os.environ.get("RAGFLOW_MCP_TOOL_ALLOWLIST", "").strip()
    if not raw:
        return []

    allowed = {name.strip() for name in raw.split(",") if name.strip()}
    manager = getattr(server, "_tool_manager", None)
    if manager is None or not hasattr(manager, "list_tools"):
        logger.warning(
            "RAGFLOW_MCP_TOOL_ALLOWLIST is set but the tool inventory is "
            "unavailable; keeping all tools"
        )
        return []

    registered = {tool.name for tool in manager.list_tools()}
    removed = sorted(registered - allowed)
    for name in removed:
        server.remove_tool(name)

    unknown = sorted(allowed - registered)
    if unknown:
        logger.warning(
            "RAGFLOW_MCP_TOOL_ALLOWLIST names that are not registered: %s", unknown
        )

    logger.info(
        "Tool allowlist active: kept %d of %d tools (removed %s)",
        len(allowed & registered),
        len(registered),
        ", ".join(removed) if removed else "none",
    )
    return removed


apply_tool_allowlist(mcp)


@mcp.resource("http://ragflow/status")
async def get_status() -> str:
    """Get server health status.

    Returns:
        JSON string with server status information.
    """
    connector = get_connector()
    return f"""{{
    "status": "healthy",
    "connected": {str(connector.client is not None).lower()},
    "base_url": "{connector.base_url}"
}}"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAGFlow MCP Server - Semantic retrieval and knowledge management"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the server."""
    args = parse_args()

    try:
        # Validate configuration early
        get_settings()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()
