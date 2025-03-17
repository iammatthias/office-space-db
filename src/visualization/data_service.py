"""
Data service for caching and managing environmental sensor data.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple
import logging
from zoneinfo import ZoneInfo
from functools import partial
import threading

from .utils import EnvironmentalData, convert_to_pst
from .sqlite_service import SQLiteService
from .migrate_to_sqlite import sync_since_timestamp
from config.config import (
    SUPABASE_URL,
    SUPABASE_KEY
)
from supabase.client import create_client

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching and caching environmental sensor data."""
    
    def __init__(self, db_path: str = "data/environmental.db", update_interval_minutes: int = 5):
        """Initialize the data service."""
        self.update_interval = timedelta(minutes=update_interval_minutes)
        self.db_path = db_path
        self._sqlite_services = {}  # Thread-local SQLite services
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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
        
    def _get_sqlite_service(self) -> SQLiteService:
        """Get or create a thread-local SQLite service instance."""
        thread_id = threading.get_ident()
        if thread_id not in self._sqlite_services:
            sqlite_service = SQLiteService(self.db_path)
            self._sqlite_services[thread_id] = sqlite_service
        return self._sqlite_services[thread_id]
        
    async def initialize(self):
        """Initialize the data service by fetching all historical data."""
        logger.info("Initializing data service...")
        # Initialize SQLite service in the main thread
        sqlite_service = self._get_sqlite_service()
        await sqlite_service.initialize()
        await self._fetch_all_historical_data()
        logger.info("Data service initialized")
        
        # Start background update tasks
        asyncio.create_task(self._periodic_update())
        asyncio.create_task(self._periodic_supabase_sync())
        
    async def _periodic_supabase_sync(self):
        """Periodically sync with Supabase to get new data."""
        while True:
            try:
                # Wait for the update interval
                await asyncio.sleep(self.update_interval.total_seconds())
                
                # Get last timestamp from SQLite
                sqlite_service = self._get_sqlite_service()
                await sqlite_service.initialize()  # Ensure initialized in this thread
                last_timestamp = await sqlite_service.get_last_timestamp()
                if last_timestamp:
                    # Sync new data from Supabase
                    synced_count = await sync_since_timestamp(
                        self.supabase,
                        sqlite_service,
                        last_timestamp
                    )
                    
                    if synced_count > 0:
                        # If we got new data, refresh our in-memory cache
                        await self._fetch_new_data()
                        
            except Exception as e:
                logger.error(f"Error syncing with Supabase: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    async def _fetch_all_historical_data(self):
        """Fetch all historical data from the database."""
        async with self._update_lock:
            total_points = 0
            sqlite_service = self._get_sqlite_service()
            
            # Fetch data for each sensor
            for sensor in self.column_map.keys():
                data = await sqlite_service.get_sensor_data(sensor)
                self._data[sensor] = [(point.time, point.value) for point in data]
                total_points += len(data)
                
                # Update last timestamp
                if data:
                    last_time = data[-1].time
                    if self._last_timestamp is None or last_time > self._last_timestamp:
                        self._last_timestamp = last_time
            
            self._last_update = datetime.now(timezone.utc)
            logger.info(f"Fetched historical data: {total_points} total points across {len(self.column_map)} sensors")
                
    async def _fetch_new_data(self):
        """Fetch new data since last update."""
        if not self._last_timestamp:
            await self._fetch_all_historical_data()
            return
            
        async with self._update_lock:
            new_points = 0
            sqlite_service = self._get_sqlite_service()
            
            # Fetch new data for each sensor
            for sensor in self.column_map.keys():
                data = await sqlite_service.get_sensor_data(
                    sensor,
                    start_date=self._last_timestamp
                )
                
                if data:
                    # Add new points to in-memory store
                    new_data = [(point.time, point.value) for point in data]
                    self._data[sensor].extend(new_data)
                    new_points += len(data)
                    
                    # Update last timestamp
                    last_time = data[-1].time
                    if last_time > self._last_timestamp:
                        self._last_timestamp = last_time
            
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
        """Interpolate missing values in the data."""
        if not data:
            return data
            
        # Find runs of None values
        runs = []
        start_idx = None
        
        for i, point in enumerate(data):
            if point.value is None:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                runs.append((start_idx, i))
                start_idx = None
                
        if start_idx is not None:
            runs.append((start_idx, len(data)))
            
        # Interpolate each run
        for start, end in runs:
            if start == 0:
                # Fill forward from next valid value
                next_valid = next((i for i in range(end, len(data)) if data[i].value is not None), None)
                if next_valid is not None:
                    for i in range(start, end):
                        data[i] = EnvironmentalData(data[i].time, data[next_valid].value)
            elif end == len(data):
                # Fill backward from last valid value
                for i in range(start, end):
                    data[i] = EnvironmentalData(data[i].time, data[start-1].value)
            else:
                # Interpolate between valid values
                left_val = data[start-1].value
                right_val = data[end].value
                total_gap = end - start + 1
                for i in range(start, end):
                    weight = (i - start + 1) / total_gap
                    interpolated = left_val * (1 - weight) + right_val * weight
                    data[i] = EnvironmentalData(data[i].time, interpolated)
                    
        return data

    async def get_earliest_data_point(self, sensor: str) -> Optional[EnvironmentalData]:
        """Get the earliest data point for a sensor."""
        try:
            if sensor not in self.column_map:
                return None
                
            # Get the earliest point from SQLite
            async with self._update_lock:
                sqlite_service = self._get_sqlite_service()
                await sqlite_service.initialize()
                    
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    partial(
                        self._get_earliest_data_point_sync,
                        sensor,
                        sqlite_service
                    )
                )
                return result
        except Exception as e:
            logger.error(f"Error getting earliest data point for {sensor}: {str(e)}")
            return None
            
    def _get_earliest_data_point_sync(self, sensor: str, sqlite_service: SQLiteService) -> Optional[EnvironmentalData]:
        """Synchronous helper to get earliest data point."""
        try:
            db_column = self.column_map[sensor]
            conn = sqlite_service._get_connection()
            cursor = conn.execute(
                f"SELECT time, {db_column} FROM environmental_data WHERE {db_column} IS NOT NULL ORDER BY time ASC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                time_str, value = row
                return EnvironmentalData(
                    time=datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc),
                    value=float(value)
                )
            return None
        except Exception as e:
            logger.error(f"Error in _get_earliest_data_point_sync for {sensor}: {str(e)}")
            return None 