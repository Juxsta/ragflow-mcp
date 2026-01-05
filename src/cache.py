"""LRU cache with TTL for dataset and document metadata.

Provides a thread-safe LRU cache implementation with configurable TTL
for caching dataset and document metadata to reduce API calls.
"""
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A cache entry with value and expiration time."""

    value: Any
    expires_at: float


class LRUCache:
    """LRU (Least Recently Used) cache with TTL (Time To Live).

    Thread-safe implementation that supports:
    - Configurable maximum size with LRU eviction
    - Configurable TTL for automatic expiration
    - Thread-safe operations

    Attributes:
        max_size: Maximum number of items in the cache.
        ttl: Time to live in seconds for cache entries.
    """

    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of items to store (default: 1000).
            ttl: Time to live in seconds for entries (default: 300s = 5 minutes).
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        """Get an item from the cache.

        Args:
            key: The cache key to retrieve.

        Returns:
            The cached value if found and not expired, None otherwise.
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check if expired
            if time.time() > entry.expires_at:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.value

    def set(self, key: str, value: Any) -> None:
        """Set an item in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        with self._lock:
            expires_at = time.time() + self.ttl
            entry = CacheEntry(value=value, expires_at=expires_at)

            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Evict oldest if at capacity
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete an item from the cache.

        Args:
            key: The cache key to delete.

        Returns:
            True if the key was found and deleted, False otherwise.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache (without updating LRU order)."""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if time.time() > entry.expires_at:
                del self._cache[key]
                return False
            return True
