"""Module for migrating data from Supabase to SQLite."""

import logging
from datetime import datetime, timezone
from typing import Optional
from supabase.client import create_client

from .sqlite_service import SQLiteService
from config.config import (
    SUPABASE_URL,
    SUPABASE_KEY
)

logger = logging.getLogger(__name__)

async def sync_since_timestamp(
    supabase,
    sqlite_service: SQLiteService,
    last_timestamp: datetime,
    batch_size: int = 1000
) -> int:
    """Sync data from Supabase since the given timestamp."""
    logger.info(f"Syncing data since {last_timestamp.isoformat()}")
    
    try:
        # Ensure SQLite service is initialized in this thread
        await sqlite_service.initialize()
        
        # Get count of new records
        count_result = (
            supabase.table('environmental_data')
            .select('count', count='exact')
            .gt('time', last_timestamp.isoformat())
            .execute()
        )
        total_new_records = count_result.count if hasattr(count_result, 'count') else 0
        logger.info(f"Found {total_new_records} new records to sync")
        
        if total_new_records == 0:
            return 0
            
        # Fetch and insert new records in batches
        offset = 0
        synced_count = 0
        
        while True:
            # Fetch batch from Supabase
            query = (
                supabase.table('environmental_data')
                .select('*')
                .gt('time', last_timestamp.isoformat())
                .order('time', desc=False)
                .range(offset, offset + batch_size - 1)
            )
            result = query.execute()
            
            if not result.data:
                break
                
            # Insert batch into SQLite
            await sqlite_service.insert_many(result.data)
            
            synced_count += len(result.data)
            offset += batch_size
            
            logger.info(f"Synced {synced_count}/{total_new_records} new records")
            
            if len(result.data) < batch_size:
                break
                
        logger.info(f"Sync complete! Total new records synced: {synced_count}")
        return synced_count
        
    except Exception as e:
        logger.error(f"Error during sync: {e}")
        raise

async def migrate_data(
    supabase_url: str,
    supabase_key: str,
    db_path: str = "data/environmental.db"
) -> None:
    """
    Migrate all data from Supabase to SQLite.
    If SQLite database exists, only sync new data since last record.
    """
    logger.info("Starting data migration...")
    
    try:
        # Initialize SQLite service
        sqlite_service = SQLiteService(db_path)
        await sqlite_service.initialize()
        
        # Get last timestamp from SQLite
        last_timestamp = await sqlite_service.get_last_timestamp()
        
        # Create Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        if last_timestamp:
            # Sync only new data
            await sync_since_timestamp(supabase, sqlite_service, last_timestamp)
        else:
            # First time sync - get all data
            await sync_since_timestamp(
                supabase,
                sqlite_service,
                datetime.min.replace(tzinfo=timezone.utc)
            )
            
        logger.info("Migration complete!")
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        raise 