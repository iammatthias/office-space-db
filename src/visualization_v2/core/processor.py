"""Data processing utilities."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import structlog

from models.sensor import SensorData, SensorType
from models.config import ProcessingConfig
from data import SensorDataRepository, DataCache

logger = structlog.get_logger()


class DataProcessor:
    """Handles data retrieval and processing with caching."""
    
    def __init__(
        self,
        repository: SensorDataRepository,
        cache: DataCache,
        config: ProcessingConfig
    ):
        """Initialize the processor."""
        self.repository = repository
        self.cache = cache
        self.config = config
    
    async def get_sensor_data(
        self,
        sensor_type: SensorType,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[SensorData]:
        """Get sensor data with caching."""
        # Try cache first
        cached_data = await self.cache.get(
            sensor_type, start_time, end_time, limit
        )
        
        if cached_data is not None:
            logger.debug(
                "Cache hit for sensor data",
                sensor_type=sensor_type.value,
                data_points=len(cached_data)
            )
            return cached_data
        
        # Fetch from repository
        data = await self.repository.get_data_range(
            sensor_type, start_time, end_time, limit
        )
        
        # Store in cache
        await self.cache.put(sensor_type, data, start_time, end_time, limit)
        
        logger.debug(
            "Fetched sensor data from repository",
            sensor_type=sensor_type.value,
            data_points=len(data),
            start_time=start_time,
            end_time=end_time
        )
        
        return data
    
    async def get_data_statistics(
        self,
        sensor_type: SensorType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict:
        """Get statistics about available data."""
        count = await self.repository.get_data_count(
            sensor_type, start_time, end_time
        )
        
        earliest = await self.repository.get_earliest_timestamp()
        latest = await self.repository.get_latest_timestamp()
        
        return {
            "sensor_type": sensor_type.value,
            "count": count,
            "earliest_timestamp": earliest.isoformat() if earliest else None,
            "latest_timestamp": latest.isoformat() if latest else None,
            "date_range_days": (
                (latest - earliest).days if earliest and latest else 0
            )
        }
    
    async def invalidate_cache(
        self,
        sensor_type: Optional[SensorType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """Invalidate cache entries."""
        if sensor_type:
            await self.cache.invalidate_sensor(sensor_type)
        elif start_time or end_time:
            await self.cache.invalidate_time_range(start_time, end_time)
        else:
            await self.cache.clear()
        
        logger.info(
            "Cache invalidated",
            sensor_type=sensor_type.value if sensor_type else None,
            start_time=start_time,
            end_time=end_time
        ) 