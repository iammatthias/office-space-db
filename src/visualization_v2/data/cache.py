"""Data caching utilities."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import heapq
import structlog
from collections import OrderedDict
import hashlib

from models.sensor import SensorData, SensorType

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: List[SensorData]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    
    def __post_init__(self):
        """Calculate cache entry size."""
        # Rough estimate: each SensorData is ~100 bytes
        self.size_bytes = len(self.data) * 100


class DataCache:
    """LRU cache for sensor data with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize the cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def _create_cache_key(
        self,
        sensor_type: SensorType,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> str:
        """Create a cache key from parameters."""
        key_str = f"{sensor_type.value}:{start_time.isoformat()}:{end_time.isoformat()}:{limit}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(
        self,
        sensor_type: SensorType,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> Optional[List[SensorData]]:
        """Get data from cache."""
        cache_key = self._create_cache_key(sensor_type, start_time, end_time, limit)
        self._stats["total_requests"] += 1
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            # Check if entry has expired
            if self._is_expired(entry):
                del self._cache[cache_key]
                self._stats["misses"] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            self._stats["hits"] += 1
            return entry.data.copy()
        
        self._stats["misses"] += 1
        return None
    
    async def put(
        self,
        sensor_type: SensorType,
        data: List[SensorData],
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> None:
        """Store data in cache."""
        cache_key = self._create_cache_key(sensor_type, start_time, end_time, limit)
        
        now = datetime.now()
        entry = CacheEntry(
            data=data.copy(),
            created_at=now,
            last_accessed=now,
            access_count=1
        )
        
        # Remove existing entry if present
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        # Add new entry
        self._cache[cache_key] = entry
        
        # Evict if necessary
        while len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats["evictions"] += 1
        
        logger.debug(
            "Cache entry added",
            sensor_type=sensor_type.value,
            data_points=len(data),
            cache_size=len(self._cache)
        )
    
    async def invalidate_sensor(self, sensor_type: SensorType) -> None:
        """Invalidate all entries for a sensor."""
        to_remove = []
        for key, entry in self._cache.items():
            if key.startswith(f"{sensor_type.value}:"):
                to_remove.append(key)
        
        for key in to_remove:
            del self._cache[key]
        
        logger.info(
            "Cache invalidated for sensor",
            sensor_type=sensor_type.value,
            removed_entries=len(to_remove)
        )
    
    async def invalidate_time_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """Invalidate entries overlapping with time range."""
        # For simplicity, we'll clear all entries if time range invalidation is requested
        # A more sophisticated implementation would parse cache keys and check overlaps
        cleared_count = len(self._cache)
        self._cache.clear()
        
        logger.info(
            "Cache invalidated for time range",
            start_time=start_time,
            end_time=end_time,
            removed_entries=cleared_count
        )
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        cleared_count = len(self._cache)
        self._cache.clear()
        
        logger.info("Cache cleared", removed_entries=cleared_count)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_data_points = sum(len(entry.data) for entry in self._cache.values())
        total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
        
        hit_rate = 0.0
        if self._stats["total_requests"] > 0:
            hit_rate = self._stats["hits"] / self._stats["total_requests"] * 100
        
        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "total_data_points": total_data_points,
            "total_size_bytes": total_size_bytes,
            "hit_rate_percent": round(hit_rate, 1),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "total_requests": self._stats["total_requests"]
        }
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        to_remove = []
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                to_remove.append(key)
        
        for key in to_remove:
            del self._cache[key]
        
        if to_remove:
            logger.debug("Expired cache entries removed", count=len(to_remove))
        
        return len(to_remove)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        age = datetime.now() - entry.created_at
        return age.total_seconds() > self.ttl_seconds 