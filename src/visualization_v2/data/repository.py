"""Sensor data repository for database access."""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog
import threading
import logging
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from models.sensor import SensorData, SensorType
from models.config import DatabaseConfig

logger = structlog.get_logger()

# Timezone constants
PST_TZ = ZoneInfo("America/Los_Angeles")
UTC_TZ = ZoneInfo("UTC")


class SensorDataRepository:
    """Repository for accessing sensor data from SQLite database."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize the repository."""
        self.config = config
        self.db_path = config.path
        self._connection_pool: Dict[int, sqlite3.Connection] = {}
        self._lock = asyncio.Lock()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the database connection and tables."""
        async with self._lock:
            conn = await self._get_connection()
            await self._ensure_tables_exist(conn)
            logger.info("Database initialized", db_path=str(self.db_path))
    
    async def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection for the current thread."""
        thread_id = threading.get_ident()
        
        if thread_id not in self._connection_pool:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.config.connection_timeout,
                isolation_level=None  # autocommit mode
            )
            # Configure connection for performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            
            self._connection_pool[thread_id] = conn
            
        return self._connection_pool[thread_id]
    
    async def _ensure_tables_exist(self, conn: sqlite3.Connection) -> None:
        """Ensure required tables exist."""
        # Create environmental_data table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS environmental_data (
                time TEXT NOT NULL PRIMARY KEY,
                temp REAL,
                hum REAL,
                pressure REAL,
                lux REAL,
                uv REAL,
                gas REAL
            )
        """)
        
        # Create indexes for better performance
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_environmental_data_time 
            ON environmental_data(time)
        """)
        
        conn.commit()
    
    async def get_data_range(
        self,
        sensor_type: SensorType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[SensorData]:
        """Get sensor data for a time range.
        
        Args:
            sensor_type: The type of sensor data to retrieve
            start_time: Start time in UTC (database stores UTC)
            end_time: End time in UTC (database stores UTC)
            limit: Maximum number of records to return
        """
        conn = await self._get_connection()
        
        # Map sensor type to database column
        column_map = {
            SensorType.TEMPERATURE: "temp",
            SensorType.HUMIDITY: "hum",
            SensorType.PRESSURE: "pressure", 
            SensorType.LIGHT: "lux",
            SensorType.UV: "uv",
            SensorType.GAS: "gas",
        }
        
        column = column_map[sensor_type]
        query = f"SELECT time, {column} FROM environmental_data WHERE {column} IS NOT NULL"
        params = []
        
        if start_time:
            # Ensure start_time is UTC for database query
            if start_time.tzinfo is None:
                logger.warning("start_time provided without timezone, assuming UTC")
                start_utc = start_time.replace(tzinfo=UTC_TZ)
            else:
                start_utc = start_time.astimezone(UTC_TZ)
            query += " AND time >= ?"
            params.append(start_utc.isoformat())
            
        if end_time:
            # Ensure end_time is UTC for database query  
            if end_time.tzinfo is None:
                logger.warning("end_time provided without timezone, assuming UTC")
                end_utc = end_time.replace(tzinfo=UTC_TZ)
            else:
                end_utc = end_time.astimezone(UTC_TZ)
            query += " AND time < ?"
            params.append(end_utc.isoformat())
            
        query += " ORDER BY time ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        logger.debug(
            "Executing database query",
            sensor_type=sensor_type.value,
            query=query,
            start_time=start_time.isoformat() if start_time else None,
            end_time=end_time.isoformat() if end_time else None,
            limit=limit
        )
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        result = []
        for time_str, value in rows:
            if value is not None:
                # Parse timestamp and ensure it's UTC
                timestamp = datetime.fromisoformat(time_str)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC_TZ)
                else:
                    timestamp = timestamp.astimezone(UTC_TZ)
                    
                result.append(SensorData(
                    timestamp=timestamp,
                    value=float(value),
                    sensor_type=sensor_type
                ))
        
        logger.debug(
            "Retrieved sensor data",
            sensor_type=sensor_type.value,
            count=len(result),
            start_time=start_time.isoformat() if start_time else None,
            end_time=end_time.isoformat() if end_time else None,
            first_timestamp=result[0].timestamp.isoformat() if result else None,
            last_timestamp=result[-1].timestamp.isoformat() if result else None
        )
        
        return result
    
    async def get_latest_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the most recent data point."""
        conn = await self._get_connection()
        cursor = conn.execute(
            "SELECT time FROM environmental_data ORDER BY time DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        if row:
            timestamp = datetime.fromisoformat(row[0])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC_TZ)
            else:
                timestamp = timestamp.astimezone(UTC_TZ)
            return timestamp
        return None
    
    async def get_earliest_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the earliest data point."""
        conn = await self._get_connection()
        cursor = conn.execute(
            "SELECT time FROM environmental_data ORDER BY time ASC LIMIT 1"
        )
        row = cursor.fetchone()
        
        if row:
            timestamp = datetime.fromisoformat(row[0])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC_TZ)
            else:
                timestamp = timestamp.astimezone(UTC_TZ)
            return timestamp
        return None
    
    async def get_data_count(
        self,
        sensor_type: Optional[SensorType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """Get count of data points matching criteria."""
        conn = await self._get_connection()
        
        if sensor_type:
            column_map = {
                SensorType.TEMPERATURE: "temp",
                SensorType.HUMIDITY: "hum", 
                SensorType.PRESSURE: "pressure",
                SensorType.LIGHT: "lux",
                SensorType.UV: "uv",
                SensorType.GAS: "gas",
            }
            column = column_map[sensor_type]
            query = f"SELECT COUNT(*) FROM environmental_data WHERE {column} IS NOT NULL"
        else:
            query = "SELECT COUNT(*) FROM environmental_data"
            
        params = []
        
        if start_time:
            query += " AND time >= ?"
            params.append(start_time.isoformat())
            
        if end_time:
            query += " AND time < ?"
            params.append(end_time.isoformat())
        
        cursor = conn.execute(query, params)
        return cursor.fetchone()[0]
    
    async def close(self) -> None:
        """Close all database connections."""
        async with self._lock:
            for conn in self._connection_pool.values():
                try:
                    conn.close()
                except Exception as e:
                    logger.warning("Error closing connection", error=str(e))
            self._connection_pool.clear()
            logger.info("Database connections closed") 