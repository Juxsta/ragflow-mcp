"""Tests for the optional tool allowlist."""
import pytest
from mcp.server.fastmcp import FastMCP

from src.server import apply_tool_allowlist


@pytest.fixture
def server_with_tools():
    """A FastMCP server with three tools registered."""
    server = FastMCP("test-server")

    @server.tool()
    async def tool_alpha() -> str:
        """Alpha tool."""
        return "alpha"

    @server.tool()
    async def tool_beta() -> str:
        """Beta tool."""
        return "beta"

    @server.tool()
    async def tool_gamma() -> str:
        """Gamma tool."""
        return "gamma"

    return server


def tool_names(server: FastMCP) -> set[str]:
    return {tool.name for tool in server._tool_manager.list_tools()}


def test_no_allowlist_keeps_all_tools(server_with_tools, monkeypatch):
    """Without the environment variable every tool stays registered."""
    monkeypatch.delenv("RAGFLOW_MCP_TOOL_ALLOWLIST", raising=False)

    removed = apply_tool_allowlist(server_with_tools)

    assert removed == []
    assert tool_names(server_with_tools) == {"tool_alpha", "tool_beta", "tool_gamma"}


def test_allowlist_removes_unlisted_tools(server_with_tools, monkeypatch):
    """Only the listed tools survive."""
    monkeypatch.setenv("RAGFLOW_MCP_TOOL_ALLOWLIST", "tool_alpha, tool_beta")

    removed = apply_tool_allowlist(server_with_tools)

    assert removed == ["tool_gamma"]
    assert tool_names(server_with_tools) == {"tool_alpha", "tool_beta"}


def test_allowlist_tolerates_whitespace(server_with_tools, monkeypatch):
    """Blank entries and surrounding whitespace are ignored."""
    monkeypatch.setenv("RAGFLOW_MCP_TOOL_ALLOWLIST", " tool_alpha , , tool_gamma ")

    removed = apply_tool_allowlist(server_with_tools)

    assert removed == ["tool_beta"]
    assert tool_names(server_with_tools) == {"tool_alpha", "tool_gamma"}


def test_allowlist_with_unknown_name_keeps_known_tools(server_with_tools, monkeypatch):
    """An unknown name is reported but does not remove the known ones."""
    monkeypatch.setenv("RAGFLOW_MCP_TOOL_ALLOWLIST", "tool_alpha,tool_missing")

    removed = apply_tool_allowlist(server_with_tools)

    assert removed == ["tool_beta", "tool_gamma"]
    assert tool_names(server_with_tools) == {"tool_alpha"}


def test_empty_allowlist_is_a_no_op(server_with_tools, monkeypatch):
    """An empty or whitespace-only value keeps every tool."""
    monkeypatch.setenv("RAGFLOW_MCP_TOOL_ALLOWLIST", "   ")

    removed = apply_tool_allowlist(server_with_tools)

    assert removed == []
    assert tool_names(server_with_tools) == {"tool_alpha", "tool_beta", "tool_gamma"}
