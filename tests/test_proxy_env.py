"""Tests for the proxy environment normalization."""
import os

import httpx

from src.connector import _normalize_no_proxy_env


class TestNormalizeNoProxyEnv:
    """A bracketed IPv6 entry must not break client construction."""

    def test_bracketed_ipv6_is_unbracketed(self, monkeypatch):
        """`[::1]` becomes `::1`, which httpx parses without error."""
        monkeypatch.setenv("no_proxy", "localhost,127.0.0.1,[::1]")
        monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,[::1]")

        _normalize_no_proxy_env()

        assert os.environ["no_proxy"] == "localhost,127.0.0.1,::1"
        assert os.environ["NO_PROXY"] == "localhost,127.0.0.1,::1"

    def test_plain_entries_are_untouched(self, monkeypatch):
        """A regular no_proxy list is left exactly as it was."""
        value = "localhost,127.0.0.1,example.com"
        monkeypatch.setenv("no_proxy", value)
        monkeypatch.setenv("NO_PROXY", value)

        _normalize_no_proxy_env()

        assert os.environ["no_proxy"] == value

    def test_missing_variable_is_fine(self, monkeypatch):
        """No proxy configuration at all is not an error."""
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)

        _normalize_no_proxy_env()

    def test_client_construction_succeeds(self, monkeypatch):
        """The client can be built with a bracketed IPv6 entry in the env."""
        monkeypatch.setenv("no_proxy", "localhost,[::1]")
        monkeypatch.setenv("NO_PROXY", "localhost,[::1]")

        _normalize_no_proxy_env()

        # Without the normalization this raises InvalidURL: Invalid port: ':1]'
        client = httpx.AsyncClient()
        assert client is not None
