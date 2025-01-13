"""
Data service for caching and managing environmental sensor data.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple
import logging
from zoneinfo import ZoneInfo
from supabase.client import Client

from .utils import EnvironmentalData, convert_to_pst

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching and caching environmental sensor data."""
    
    def __init__(self, db_client: Client, update_interval_minutes: int = 5):
        """Initialize the data service."""
        self.db_client = db_client
        self.update_interval = timedelta(minutes=update_interval_minutes)
        self.column_map = {
            'temperature': 'temp',
            'humidity': 'hum',
            'pressure': 'pressure',
            'light': 'lux',
            'uv': 'uv',
            'gas': 'gas',
        }
        # Main data store: sensor -> list of (datetime, value) tuples
        self._data: Dict[str, List[Tuple[datetime, float]]] = {
            sensor: [] for sensor in self.column_map.keys()
        }
        self._last_update = datetime.min.replace(tzinfo=timezone.utc)
        self._update_lock = asyncio.Lock()
        self._last_timestamp: Optional[datetime] = None
        
    async def initialize(self):
        """Initialize the data service by fetching all historical data."""
        logger.info("Initializing data service...")
        await self._fetch_all_historical_data()
        logger.info("Data service initialized")
        
        # Start background update task
        asyncio.create_task(self._periodic_update())
        
    async def _fetch_all_historical_data(self):
        """Fetch all historical data from the database."""
        async with self._update_lock:
            columns = ['time'] + list(self.column_map.values())
            page_size = 1000  # Supabase's maximum limit
            last_timestamp = None
            total_points = 0
            
            while True:
                query = self.db_client.table('environmental_data').select(','.join(columns))
                if last_timestamp:
                    query = query.gt('time', last_timestamp)
                query = query.order('time', desc=False).limit(page_size)
                
                data = query.execute()
                rows = data.data
                if not rows:
                    break
                
                # Process rows
                for row in rows:
                    try:
                        timestamp = row['time'].replace('Z', '+00:00')
                        dt = datetime.fromisoformat(timestamp)
                        
                        # Update last timestamp for pagination
                        last_timestamp = dt
                        
                        # Update last timestamp for data tracking
                        if self._last_timestamp is None or dt > self._last_timestamp:
                            self._last_timestamp = dt
                        
                        # Process each sensor's value
                        for sensor, db_column in self.column_map.items():
                            value = row[db_column]
                            if value is not None:
                                try:
                                    float_value = float(value)
                                    self._data[sensor].append((dt, float_value))
                                    total_points += 1
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid value for {db_column} at {dt}: {value}")
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Error parsing row timestamp: {e}")
                        continue
                
                if len(rows) < page_size:
                    break
            
            # Sort all data by timestamp
            for sensor in self._data:
                self._data[sensor].sort(key=lambda x: x[0])
            
            self._last_update = datetime.now(timezone.utc)
            logger.info(f"Fetched historical data: {total_points} total points across {len(self.column_map)} sensors")
    
    async def _fetch_new_data(self):
        """Fetch new data since last update."""
        if not self._last_timestamp:
            await self._fetch_all_historical_data()
            return
            
        async with self._update_lock:
            columns = ['time'] + list(self.column_map.values())
            query = self.db_client.table('environmental_data').select(','.join(columns))
            query = query.gt('time', self._last_timestamp.isoformat())
            query = query.order('time', desc=False)
            
            data = query.execute()
            rows = data.data
            
            # Process new rows
            new_points = 0
            for row in rows:
                try:
                    timestamp = row['time'].replace('Z', '+00:00')
                    dt = datetime.fromisoformat(timestamp)
                    
                    # Update last timestamp
                    if dt > self._last_timestamp:
                        self._last_timestamp = dt
                    
                    # Process each sensor's value
                    for sensor, db_column in self.column_map.items():
                        value = row[db_column]
                        if value is not None:
                            try:
                                float_value = float(value)
                                self._data[sensor].append((dt, float_value))
                                new_points += 1
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid value for {db_column} at {dt}: {value}")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing row timestamp: {e}")
                    continue
            
            self._last_update = datetime.now(timezone.utc)
            if new_points > 0:
                logger.info(f"Fetched {new_points} new data points")
    
    async def _periodic_update(self):
        """Periodically update the data from the database."""
        while True:
            try:
                await asyncio.sleep(self.update_interval.total_seconds())
                await self._fetch_new_data()
            except Exception as e:
                logger.error(f"Error updating data: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    async def get_sensor_data(
        self,
        sensor: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[EnvironmentalData]:
        """
        Get sensor data from the in-memory store.
        Dates should be in UTC.
        Returns data with timestamps in UTC.
        """
        # Validate sensor type
        if sensor not in self.column_map:
            raise ValueError(f"Unknown sensor type: {sensor}")
        
        # Ensure dates are in UTC
        if start_date and not start_date.tzinfo:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date and not end_date.tzinfo:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        # Get data from in-memory store
        async with self._update_lock:
            # Filter data by date range
            filtered_data = self._data[sensor]
            if start_date:
                filtered_data = [(dt, val) for dt, val in filtered_data if dt >= start_date]
            if end_date:
                filtered_data = [(dt, val) for dt, val in filtered_data if dt < end_date]
            if limit:
                filtered_data = filtered_data[:limit]
            
            # Convert to EnvironmentalData objects
            parsed_data = [EnvironmentalData(dt, value) for dt, value in filtered_data]
        
        # Fill minute-by-minute data if date range provided
        if start_date and end_date and parsed_data:
            start_pst = convert_to_pst(start_date)
            end_pst = convert_to_pst(end_date)
            current = start_pst.replace(second=0, microsecond=0)
            
            all_minutes = []
            while current < end_pst:
                all_minutes.append(current)
                current += timedelta(minutes=1)
            
            minute_dict = {
                convert_to_pst(point.time).replace(second=0, microsecond=0): point.value
                for point in parsed_data
            }
            
            valid_data = [
                EnvironmentalData(minute, minute_dict.get(minute, None))
                for minute in all_minutes
            ]
            
            # Interpolate missing values
            valid_data = self._interpolate_missing_values(valid_data)
        else:
            valid_data = parsed_data
        
        logger.info(f"Retrieved {len(valid_data)} data points for {sensor}")
        return valid_data
    
    def _interpolate_missing_values(self, data: List[EnvironmentalData]) -> List[EnvironmentalData]:
        """Replace None values with linear interpolation based on nearest neighbors."""
        i = 0
        n = len(data)
        while i < n:
            if data[i].value is None:
                start_idx = i - 1
                while i < n and data[i].value is None:
                    i += 1
                end_idx = i
                
                if start_idx >= 0 and end_idx < n:
                    start_val = data[start_idx].value
                    end_val = data[end_idx].value
                    gap_len = end_idx - start_idx
                    if start_val is not None and end_val is not None:
                        step = (end_val - start_val) / gap_len
                        for fill_idx in range(start_idx + 1, end_idx):
                            offset = fill_idx - start_idx
                            data[fill_idx].value = start_val + step * offset
                    else:
                        for fill_idx in range(start_idx + 1, end_idx):
                            data[fill_idx].value = start_val if start_val is not None else end_val
                else:
                    if start_idx < 0 and end_idx < n:
                        for fill_idx in range(end_idx):
                            data[fill_idx].value = data[end_idx].value
                    elif start_idx >= 0 and end_idx >= n:
                        for fill_idx in range(start_idx+1, n):
                            data[fill_idx].value = data[start_idx].value
            else:
                i += 1
        return data 