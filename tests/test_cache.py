"""Tests for LRU cache with TTL."""
import time
import pytest
from src.cache import LRUCache


class TestLRUCache:
    """Test suite for LRU cache with TTL."""

    def test_cache_stores_and_retrieves_items(self):
        """Test 3: LRU cache stores and retrieves items correctly."""
        cache = LRUCache(max_size=100, ttl=300)

        # Store items
        cache.set("dataset_1", {"id": "1", "name": "Test Dataset"})
        cache.set("document_1", {"id": "doc1", "name": "Test Document"})

        # Retrieve items
        dataset = cache.get("dataset_1")
        document = cache.get("document_1")

        assert dataset == {"id": "1", "name": "Test Dataset"}
        assert document == {"id": "doc1", "name": "Test Document"}

        # Non-existent key returns None
        assert cache.get("nonexistent") is None

    def test_cache_respects_ttl_expiration(self):
        """Test 4: LRU cache respects TTL expiration."""
        # Use very short TTL for testing
        cache = LRUCache(max_size=100, ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")

        # Should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired now
        assert cache.get("key1") is None

    def test_cache_clear_method(self):
        """Test that cache clear method removes all items."""
        cache = LRUCache(max_size=100, ttl=300)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_lru_eviction(self):
        """Test that cache evicts least recently used items when full."""
        cache = LRUCache(max_size=3, ttl=300)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new item, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Was accessed recently
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
