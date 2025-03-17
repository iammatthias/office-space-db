"""SQLite service for storing and retrieving environmental sensor data."""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict
import asyncio
from functools import partial
import threading

from .utils import EnvironmentalData

logger = logging.getLogger(__name__)

class SQLiteService:
    """Service for managing SQLite database operations."""
    
    def __init__(self, db_path: str = "data/environmental.db"):
        """Initialize the SQLite service."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connections = {}  # Thread-local connections
        self.column_map = {
            'temperature': 'temp',
            'humidity': 'hum',
            'pressure': 'pressure',
            'light': 'lux',
            'uv': 'uv',
            'gas': 'gas',
        }
        self._lock = asyncio.Lock()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        thread_id = threading.get_ident()
        if thread_id not in self._connections:
            conn = sqlite3.connect(self.db_path, isolation_level=None)  # autocommit mode
            conn.execute("PRAGMA journal_mode=WAL")  # Use WAL mode for better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes with reasonable safety
            self._connections[thread_id] = conn
            self._create_tables(conn)
        return self._connections[thread_id]
        
    async def initialize(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            # Run in thread pool since sqlite3 operations are blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._initialize_sync
            )
            logger.info(f"Connected to SQLite database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise
            
    def _initialize_sync(self):
        """Synchronous initialization of database."""
        # This will create a connection for the current thread if it doesn't exist
        self._get_connection()
            
    def _create_tables(self, conn: sqlite3.Connection):
        """Create the necessary tables if they don't exist."""
        columns = [f"{col} REAL" for col in self.column_map.values()]
        columns = ["time TEXT NOT NULL"] + columns
        
        with conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS environmental_data (
                    {', '.join(columns)},
                    PRIMARY KEY (time)
                )
                """
            )
            
    async def close(self):
        """Close all database connections."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._close_sync)
            
    def _close_sync(self):
        """Synchronous database close."""
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()
            
    async def insert_data(self, data: Dict):
        """Insert a single row of data."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._insert_data_sync,
                data
            )
            
    def _insert_data_sync(self, data: Dict):
        """Synchronous data insertion."""
        conn = self._get_connection()
        columns = ['time'] + list(self.column_map.values())
        placeholders = ','.join(['?' for _ in columns])
        values = [data.get(col) for col in columns]
        
        with conn:
            conn.execute(
                f"INSERT INTO environmental_data ({','.join(columns)}) VALUES ({placeholders})",
                values
            )
            
    async def insert_many(self, data_rows: List[Dict]):
        """Insert multiple rows of data."""
        if not data_rows:
            return
            
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._insert_many_sync,
                data_rows
            )
            
    def _insert_many_sync(self, data_rows: List[Dict]):
        """Synchronous batch insertion."""
        conn = self._get_connection()
        columns = ['time'] + list(self.column_map.values())
        placeholders = ','.join(['?' for _ in columns])
        values = [[row.get(col) for col in columns] for row in data_rows]
        
        with conn:
            conn.executemany(
                f"INSERT INTO environmental_data ({','.join(columns)}) VALUES ({placeholders})",
                values
            )
            
    async def get_sensor_data(
        self,
        sensor: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[EnvironmentalData]:
        """Get sensor data from the database."""
        if sensor not in self.column_map:
            raise ValueError(f"Unknown sensor type: {sensor}")
            
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                partial(
                    self._get_sensor_data_sync,
                    sensor,
                    start_date,
                    end_date,
                    limit
                )
            )
            
    def _get_sensor_data_sync(
        self,
        sensor: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[EnvironmentalData]:
        """Synchronous helper to get sensor data."""
        conn = self._get_connection()
        db_column = self.column_map[sensor]
        
        query = f"SELECT time, {db_column} FROM environmental_data WHERE {db_column} IS NOT NULL"
        params = []
        
        if start_date:
            query += " AND time >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND time < ?"
            params.append(end_date.isoformat())
            
        query += " ORDER BY time ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
            
        with conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
        return [
            EnvironmentalData(
                time=datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc),
                value=float(value)
            )
            for time_str, value in rows
        ]
        
    async def get_first_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the first data point."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._get_first_timestamp_sync
            )
            
    def _get_first_timestamp_sync(self) -> Optional[datetime]:
        """Synchronous first timestamp retrieval."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("SELECT time FROM environmental_data ORDER BY time ASC LIMIT 1")
            row = cursor.fetchone()
            
        if row:
            return datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc)
        return None
        
    async def get_last_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the last data point."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._get_last_timestamp_sync
            )
            
    def _get_last_timestamp_sync(self) -> Optional[datetime]:
        """Synchronous last timestamp retrieval."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("SELECT time FROM environmental_data ORDER BY time DESC LIMIT 1")
            row = cursor.fetchone()
            
        if row:
            return datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc)
        return None 