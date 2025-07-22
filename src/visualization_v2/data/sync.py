"""Data synchronization utilities."""

import os
import sqlite3
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog
from supabase import create_client, Client
from pathlib import Path

from models.config import DatabaseConfig

logger = structlog.get_logger()


class DataSynchronizer:
    """Handles data synchronization from external sources."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize the synchronizer."""
        self.config = config
        self.supabase_client: Optional[Client] = None
        self._initialize_supabase()
        
    def _initialize_supabase(self):
        """Initialize Supabase client from environment variables."""
        try:
            # Try to get credentials from environment
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_KEY')
            
            if not supabase_url or not supabase_key:
                # Try loading from config directory
                config_dir = Path(__file__).parent.parent.parent.parent / "config"
                env_file = config_dir / ".env"
                
                if env_file.exists():
                    from dotenv import load_dotenv
                    load_dotenv(env_file)
                    supabase_url = os.getenv('SUPABASE_URL')
                    supabase_key = os.getenv('SUPABASE_KEY')
            
            if supabase_url and supabase_key:
                self.supabase_client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully")
            else:
                logger.warning("Supabase credentials not found - sync disabled")
                
        except Exception as e:
            logger.error("Failed to initialize Supabase client", error=str(e))
            self.supabase_client = None
        
    async def sync_from_supabase(
        self, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None  # Remove default limit to get all data
    ) -> Dict[str, Any]:
        """Sync data from Supabase to local database."""
        if not self.supabase_client:
            return {
                "status": "Error: Supabase client not initialized",
                "synced_records": 0,
                "last_sync": datetime.now().isoformat()
            }
        
        try:
            # Get local database connection
            conn = sqlite3.connect(self.config.path)
            
            # Ensure table exists
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
            
            # Get latest timestamp from local database
            if not since:
                cursor = conn.execute("SELECT MAX(time) FROM environmental_data")
                last_time = cursor.fetchone()[0]
                if last_time:
                    since = datetime.fromisoformat(last_time.replace('+00:00', ''))
                    logger.info(f"Syncing from last local timestamp: {since}")
            
            # Query Supabase for new data
            query = self.supabase_client.table("environmental_data").select("*")
            
            if since:
                query = query.gte("time", since.isoformat())
                
            query = query.order("time")
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if not response.data:
                logger.info("No new data found in Supabase")
                return {
                    "status": "Success - no new data",
                    "synced_records": 0,
                    "last_sync": datetime.now().isoformat()
                }
            
            # Insert data into local database
            synced_count = 0
            skipped_count = 0
            
            for record in response.data:
                try:
                    # Insert or replace record
                    conn.execute("""
                        INSERT OR REPLACE INTO environmental_data 
                        (time, temp, hum, pressure, lux, uv, gas)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.get('time'),
                        record.get('temp'),
                        record.get('hum'), 
                        record.get('pressure'),
                        record.get('lux'),
                        record.get('uv'),
                        record.get('gas')
                    ))
                    synced_count += 1
                    
                except Exception as e:
                    logger.warning(f"Skipped record due to error: {e}")
                    skipped_count += 1
            
            conn.commit()
            conn.close()
            
            logger.info(
                "Supabase sync completed successfully",
                synced_records=synced_count,
                skipped_records=skipped_count,
                total_fetched=len(response.data)
            )
            
            return {
                "status": "Success",
                "synced_records": synced_count,
                "skipped_records": skipped_count,
                "last_sync": datetime.now().isoformat(),
                "total_fetched": len(response.data)
            }
            
        except Exception as e:
            logger.error("Supabase sync failed", error=str(e))
            return {
                "status": f"Error: {str(e)}",
                "synced_records": 0,
                "last_sync": datetime.now().isoformat()
            }
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status."""
        if not self.supabase_client:
            return {
                "mode": "disabled",
                "reason": "Supabase client not initialized",
                "last_sync": None,
                "next_sync": None,
                "sync_enabled": False
            }
        
        # Check local database for last sync info
        try:
            conn = sqlite3.connect(self.config.path)
            cursor = conn.execute("SELECT COUNT(*), MAX(time), MIN(time) FROM environmental_data")
            count, max_time, min_time = cursor.fetchone()
            conn.close()
            
            return {
                "mode": "enabled",
                "sync_enabled": True,
                "local_records": count or 0,
                "earliest_record": min_time,
                "latest_record": max_time,
                "supabase_connected": True
            }
            
        except Exception as e:
            logger.error("Failed to get sync status", error=str(e))
            return {
                "mode": "error",
                "sync_enabled": False,
                "error": str(e)
            }
    
    async def full_sync(self, batch_size: int = 10000) -> Dict[str, Any]:
        """Perform a full synchronization from Supabase."""
        logger.info("Starting full sync from Supabase")
        
        if not self.supabase_client:
            return {
                "status": "Error: Supabase client not initialized",
                "synced_records": 0
            }
        
        try:
            # Clear existing data for full sync
            conn = sqlite3.connect(self.config.path)
            
            # Ensure table exists
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
            
            conn.execute("DELETE FROM environmental_data")
            conn.commit()
            logger.info("Cleared existing local data for full sync")
            
            # Get all data from Supabase using pagination
            logger.info("Fetching all data from Supabase with pagination...")
            
            total_synced = 0
            skipped_count = 0
            page = 0
            page_size = 1000  # Supabase default limit
            
            while True:
                start_range = page * page_size
                end_range = start_range + page_size - 1
                
                logger.info(f"Fetching page {page + 1} (records {start_range}-{end_range})")
                
                # Fetch this page of data
                response = (self.supabase_client
                    .table("environmental_data")
                    .select("*")
                    .order("time")
                    .range(start_range, end_range)
                    .execute())
                
                if not response.data or len(response.data) == 0:
                    logger.info("No more data to fetch - sync complete")
                    break
                
                logger.info(f"Retrieved {len(response.data)} records from page {page + 1}")
                
                # Process this batch
                batch_synced = 0
                for record in response.data:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO environmental_data 
                            (time, temp, hum, pressure, lux, uv, gas)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            record.get('time'),
                            record.get('temp'),
                            record.get('hum'), 
                            record.get('pressure'),
                            record.get('lux'),
                            record.get('uv'),
                            record.get('gas')
                        ))
                        batch_synced += 1
                        
                    except Exception as e:
                        logger.warning(f"Skipped record due to error: {e}")
                        skipped_count += 1
                
                # Commit this batch
                conn.commit()
                total_synced += batch_synced
                logger.info(f"Committed page {page + 1} - {batch_synced} records (Total: {total_synced})")
                
                # If we got fewer records than page_size, we're at the end
                if len(response.data) < page_size:
                    logger.info("Reached end of data - final page")
                    break
                
                page += 1
            
            conn.close()
            
            logger.info("Full sync completed", 
                       total_synced=total_synced, 
                       skipped=skipped_count,
                       pages_processed=page + 1)
            
            return {
                "status": "Success - Full sync completed",
                "synced_records": total_synced,
                "skipped_records": skipped_count,
                "pages_processed": page + 1,
                "last_sync": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Full sync failed", error=str(e))
            try:
                if 'conn' in locals():
                    conn.close()
            except:
                pass
            return {
                "status": f"Error: {str(e)}",
                "synced_records": 0,
                "last_sync": datetime.now().isoformat()
            } 