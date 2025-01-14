"""Script to migrate data from Supabase to SQLite."""

import asyncio
import logging
from datetime import datetime, timezone
from supabase.client import create_client

from .sqlite_service import SQLiteService
from config.config import (
    SUPABASE_URL,
    SUPABASE_KEY
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
    supabase_url: str = SUPABASE_URL,
    supabase_key: str = SUPABASE_KEY,
    db_path: str = "data/environmental.db",
    batch_size: int = 1000
):
    """Migrate data from Supabase to SQLite."""
    logger.info("Starting migration/sync from Supabase to SQLite")
    
    # Initialize services
    supabase = create_client(supabase_url, supabase_key)
    sqlite_service = SQLiteService(db_path)
    await sqlite_service.initialize()
    
    try:
        # Get last timestamp from SQLite
        last_timestamp = await sqlite_service.get_last_timestamp()
        
        if last_timestamp is None:
            # No existing data, do full migration
            logger.info("No existing data found. Starting full migration...")
            
            # Get total count of records
            count_result = supabase.table('environmental_data').select('count', count='exact').execute()
            total_records = count_result.count if hasattr(count_result, 'count') else 0
            logger.info(f"Total records to migrate: {total_records}")
            
            # Migrate data in batches
            offset = 0
            migrated_count = 0
            
            while True:
                # Fetch batch from Supabase
                query = supabase.table('environmental_data').select('*')
                query = query.order('time', desc=False).range(offset, offset + batch_size - 1)
                result = query.execute()
                
                if not result.data:
                    break
                    
                # Insert batch into SQLite
                await sqlite_service.insert_many(result.data)
                
                migrated_count += len(result.data)
                offset += batch_size
                
                logger.info(f"Migrated {migrated_count}/{total_records} records")
                
                if len(result.data) < batch_size:
                    break
                    
            logger.info("Full migration complete!")
            logger.info(f"Total records migrated: {migrated_count}")
        else:
            # Existing data found, sync new records
            await sync_since_timestamp(supabase, sqlite_service, last_timestamp, batch_size)
        
    except Exception as e:
        logger.error(f"Error during migration/sync: {e}")
        raise
    finally:
        await sqlite_service.close()

if __name__ == "__main__":
    asyncio.run(migrate_data()) 