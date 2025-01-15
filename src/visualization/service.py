"""Service for generating and managing environmental data visualizations."""

import asyncio
from datetime import datetime, timezone, timedelta, time
import os
from typing import List, Dict, Optional, Iterator, Tuple, Set
from zoneinfo import ZoneInfo
import json
import logging
import io
from pathlib import Path
from tqdm.asyncio import tqdm
import contextlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import itertools

from .generator import VisualizationGenerator
from .utils import EnvironmentalData, convert_to_pst
from .data_service import DataService
from .migrate_to_sqlite import migrate_data
from .cloudflare_service import CloudflareService
from config.config import (
    SUPABASE_URL,
    SUPABASE_KEY
)

# Configure logging with a cleaner format
logging.basicConfig(
    level=logging.INFO,  # Change to INFO level by default
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('visualization_service.log')  # File handler for debugging
    ]
)
logger = logging.getLogger(__name__)

# Create a separate debug logger for detailed logging
debug_logger = logging.getLogger(f"{__name__}.debug")
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('visualization_service_debug.log')
debug_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s'))
debug_logger.addHandler(debug_handler)

class AsyncLogger:
    """Asynchronous logger that batches messages and logs them in the background."""
    
    def __init__(self, logger):
        self.logger = logger
        self.queue = asyncio.Queue()
        self.running = False
        self.task = None
        self._error_count = 0
        self.MAX_ERRORS = 100
    
    async def start(self):
        """Start the background logging task."""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._process_logs())
    
    async def stop(self):
        """Stop the background logging task."""
        if self.running:
            self.running = False
            if self.task:
                try:
                    # Give remaining logs a chance to process but don't block indefinitely
                    await asyncio.wait_for(self.queue.join(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
    
    async def _process_logs(self):
        """Process logs from the queue in the background."""
        while self.running:
            try:
                # Use get_nowait() to avoid blocking when queue is empty
                try:
                    level, msg = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.1)  # Short sleep when queue is empty
                    continue

                try:
                    if level == logging.DEBUG:
                        debug_logger.debug(msg)
                    else:
                        self.logger.log(level, msg)
                except Exception as e:
                    self._error_count += 1
                    if self._error_count <= self.MAX_ERRORS:
                        print(f"Error in logger: {str(e)}")
                finally:
                    self.queue.task_done()
                    
            except Exception as e:
                self._error_count += 1
                if self._error_count <= self.MAX_ERRORS:
                    print(f"Error in log processor: {str(e)}")
                await asyncio.sleep(0.1)  # Prevent tight loop on repeated errors
                
        # Process remaining logs after running is set to False
        while not self.queue.empty():
            try:
                level, msg = self.queue.get_nowait()
                if level == logging.DEBUG:
                    debug_logger.debug(msg)
                else:
                    self.logger.log(level, msg)
            except Exception:
                pass
            finally:
                self.queue.task_done()
    
    async def log(self, level: int, msg: str):
        """Asynchronously queue a log message."""
        if not self.running:
            # Fall back to synchronous logging if not running
            if level == logging.DEBUG:
                debug_logger.debug(msg)
            else:
                self.logger.log(level, msg)
            return
            
        try:
            # Use put_nowait to avoid blocking
            self.queue.put_nowait((level, msg))
        except asyncio.QueueFull:
            # If queue is full, log synchronously
            if level == logging.DEBUG:
                debug_logger.debug(msg)
            else:
                self.logger.log(level, msg)

    async def error(self, msg: str, exc_info: Optional[Exception] = None):
        """Log an error message with optional exception info."""
        if exc_info:
            import traceback
            tb_str = ''.join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
            msg = f"{msg}\n{tb_str}"
        await self.log(logging.ERROR, msg)

class ProgressManager:
    """Manages progress bars and logging during batch operations."""
    
    def __init__(self, total: int, desc: str, disable_logging: bool = True):
        self.total = total
        self.desc = desc
        self.disable_logging = disable_logging
        self.async_logger = None
    
    @contextlib.asynccontextmanager
    async def progress_bar(self):
        """Async context manager for progress bar that temporarily modifies logging."""
        self.async_logger = AsyncLogger(logger)
        await self.async_logger.start()
        
        if self.disable_logging:
            logging_level = logger.getEffectiveLevel()
            logger.setLevel(logging.WARNING)
        
        try:
            with tqdm(total=self.total, desc=self.desc) as pbar:
                yield pbar
        finally:
            if self.disable_logging:
                logger.setLevel(logging_level)
            # Ensure all logs are processed before closing
            if self.async_logger:
                await self.async_logger.stop()

class VisualizationService:
    """Service for generating and managing environmental data visualizations."""
    
    def __init__(
        self,
        sensors: List[Dict[str, str]],
        config: Dict[str, any]
    ):
        """Initialize the visualization service with a configuration dictionary."""
        self.config = config
        self.sensors = sensors
        self.output_dir = Path(config['output_dir'])
        self.data_service = DataService(config['db_path'])
        self.batch_size = config['batch_size']
        self.async_logger = AsyncLogger(logger)
        
        # Initialize thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config['max_workers'])
        
        # Initialize visualization generator with scale factor
        self.generator = VisualizationGenerator(scale_factor=config['scale_factor'])
        
        # Initialize Cloudflare service
        self.cloudflare = CloudflareService()
        
        # Create output directory if it doesn't exist (for temporary files)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized visualization service:")
        logger.info(f"- Output directory: {config['output_dir']}")
        logger.info(f"- Sensors: {', '.join(s['column'] for s in sensors)}")
        logger.info(f"- Batch size: {config['batch_size']}")
        logger.info(f"- Scale factor: {config['scale_factor']}")

    async def initialize(self):
        """Initialize async components of the service."""
        await self.async_logger.start()
        await self.data_service.initialize()
        logger.info("Initialized async components of visualization service")

    def get_r2_key(self, sensor: str, start_time: datetime, end_time: datetime, interval: str = 'daily') -> str:
        """Generate the R2 storage key for a visualization."""
        # Convert to PST for consistent naming
        start_pst = convert_to_pst(start_time)
        end_pst = convert_to_pst(end_time)
        
        # Calculate days since start
        days_since_start = (end_pst.date() - start_pst.date()).days
        
        # Calculate minutes since midnight based on interval
        if interval == 'daily':
            minutes_since_midnight = 1439
        elif interval == 'hourly':
            minutes_since_midnight = end_pst.hour * 60 + 59
        else:  # minute
            minutes_since_midnight = end_pst.hour * 60 + end_pst.minute
        
        # Format: sensor/interval/YYYY-MM-DD_DDDD_MMMM.png
        return f"{sensor}/{interval}/{end_pst.strftime('%Y-%m-%d')}_{days_since_start:04d}_{minutes_since_midnight:04d}.png"

    def get_kv_key(self, sensor: str, interval: str) -> str:
        """Generate the KV key for storing the latest image path."""
        return f"latest_{sensor}_{interval}"

    async def _save_image(self, image: object, path: Path) -> None:
        """Save a single image to R2."""
        try:
            # Convert image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG', optimize=True, compress_level=6)
            img_bytes = img_bytes.getvalue()
            
            def _save():
                # Upload to R2 only
                r2_url = self.cloudflare.upload_image(img_bytes, str(path))
                return r2_url
            
            r2_url = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                _save
            )
            
            await self.async_logger.log(logging.DEBUG, f"Successfully saved image to R2: {r2_url}")
            return r2_url
        except Exception as e:
            await self.async_logger.log(logging.ERROR, f"Error saving image: {str(e)}")
            raise

    async def _update_kv_records(self, interval: str, sensor_status: Dict):
        """Update KV records for all sensors after a batch completes.
        Stores detailed information about the last processed state.
        """
        try:
            for sensor in self.sensors:
                if sensor['column'] in sensor_status and sensor_status[sensor['column']].get('image_path'):
                    kv_key = self.get_kv_key(sensor['column'], interval)
                    # Store detailed information about the processed state
                    kv_data = {
                        'path': sensor_status[sensor['column']]['image_path'],
                        'last_processed': datetime.now(timezone.utc).isoformat(),
                        'last_success': sensor_status[sensor['column']].get('last_success'),
                        'last_error': sensor_status[sensor['column']].get('last_error'),
                        'status': 'success' if not sensor_status[sensor['column']].get('last_error') else 'error'
                    }
                    # Pass the dictionary directly to update_kv_record - it will handle the JSON encoding
                    self.cloudflare.update_kv_record(kv_key, kv_data)
                    await self.async_logger.log(
                        logging.INFO,
                        f"Updated KV record for {sensor['column']} {interval} with status {kv_data['status']}"
                    )
        except Exception as e:
            await self.async_logger.log(logging.ERROR, f"Error updating KV records: {str(e)}")
            # Don't raise the exception - we want to continue processing even if KV update fails
            # The next run will detect the inconsistency and handle it appropriately

    async def process_sensor_update(
        self,
        sensor: Dict[str, str],
        start_time: datetime,
        end_time: datetime,
        sensor_status: Dict,
        interval: str,
        existing_data: Optional[List] = None
    ):
        """Process a single sensor update with immediate saving."""
        try:
            # Use existing data if provided, otherwise fetch from database
            data = existing_data if existing_data is not None else await self.data_service.get_sensor_data(
                sensor['column'],
                start_date=start_time.astimezone(timezone.utc),
                end_date=end_time.astimezone(timezone.utc)
            )
            
            if data:
                start_pst = convert_to_pst(start_time)
                end_pst = convert_to_pst(end_time)
                
                # Generate visualization
                image = self.generator.generate_visualization(
                    data=data,
                    column=sensor['column'],
                    color_scheme=sensor['color_scheme'],
                    start_time=start_pst,
                    end_time=end_pst,
                    interval=interval
                )
                
                # Generate R2 key and save image
                r2_key = self.get_r2_key(sensor['column'], start_pst, end_pst, interval)
                await self._save_image(image, Path(r2_key))
                
                # Update status
                sensor_status[sensor['column']] = {
                    'last_success': end_pst.isoformat(),
                    'last_error': None,
                    'image_path': r2_key
                }
                
                # Explicitly cleanup
                del image
                if existing_data is None:  # Only delete if we fetched it
                    del data
            else:
                await self.async_logger.log(
                    logging.WARNING,
                    f"No data available for {sensor['column']} between {start_time} and {end_time}"
                )
        except Exception as e:
            await self.async_logger.log(
                logging.ERROR,
                f"Error processing {sensor['column']}: {str(e)}"
            )
            sensor_status[sensor['column']]['last_error'] = str(e)

    async def process_interval_updates(
        self,
        start_date: datetime,
        end_date: datetime,
        sensor_status: Dict,
        interval: str,
        existing_data: Optional[Dict] = None
    ):
        """Process updates for a specific interval with optimized processing."""
        try:
            start_pst = convert_to_pst(start_date)
            end_pst = convert_to_pst(end_date)
            
            await self.async_logger.log(logging.INFO, f"Processing {interval} updates from {start_pst} to {end_pst}")
            
            # Use existing data if provided, otherwise fetch new data
            sensor_data = existing_data or {}
            if not existing_data:
                # For minute visualizations, we need all historical data from the first data point
                if interval == 'minute':
                    first_data = await self.data_service.get_sensor_data(self.sensors[0]['column'], limit=1)
                    if first_data:
                        historical_start = convert_to_pst(first_data[0].time)
                        data_futures = [
                            self.data_service.get_sensor_data(
                                sensor['column'],
                                start_date=historical_start.astimezone(timezone.utc),
                                end_date=end_date.astimezone(timezone.utc)
                            )
                            for sensor in self.sensors
                        ]
                else:
                    data_futures = [
                        self.data_service.get_sensor_data(
                            sensor['column'],
                            start_date=start_date.astimezone(timezone.utc),
                            end_date=end_date.astimezone(timezone.utc)
                        )
                        for sensor in self.sensors
                    ]
                all_sensor_data = await asyncio.gather(*data_futures)
                sensor_data = {
                    sensor['column']: data
                    for sensor, data in zip(self.sensors, all_sensor_data)
                    if data
                }
            
            if not sensor_data:
                await self.async_logger.log(logging.WARNING, f"No data found for {interval} updates")
                return
            
            # Calculate time step based on interval and config
            if interval == 'minute':
                chunk_interval = self.config.get('minute_chunk_interval', 'minute')
                if chunk_interval == 'daily':
                    # For daily chunks, start from midnight and increment by days
                    current = start_pst.replace(hour=0, minute=0, second=0, microsecond=0)
                    time_delta = timedelta(days=1)
                elif chunk_interval == 'hourly':
                    # For hourly chunks, start from the hour and increment by hours
                    current = start_pst.replace(minute=0, second=0, microsecond=0)
                    time_delta = timedelta(hours=1)
                elif chunk_interval == 'weekly':
                    # For weekly chunks, start from Monday midnight and increment by weeks
                    current = start_pst.replace(hour=0, minute=0, second=0, microsecond=0)
                    current = current - timedelta(days=current.weekday())  # Go back to Monday
                    time_delta = timedelta(weeks=1)
                else:  # minute or invalid value
                    current = start_pst
                    time_delta = timedelta(minutes=1)
            else:
                current = start_pst
                time_delta = timedelta(days=1) if interval == 'daily' else \
                           timedelta(hours=1) if interval == 'hourly' else \
                           timedelta(minutes=1)
                           
            total_updates = int((end_pst - current) / time_delta) * len(self.sensors)
            
            progress = ProgressManager(total_updates, f"Generating {interval} visualizations")
            async with progress.progress_bar() as pbar:
                while current < end_pst:
                    next_time = min(current + time_delta, end_pst)
                    
                    # Process each sensor
                    update_futures = []
                    for sensor in self.sensors:
                        if sensor['column'] in sensor_data:
                            # For minute visualizations, incrementally build up data
                            if interval == 'minute':
                                # Filter data up to the current time
                                current_data = [
                                    point for point in sensor_data[sensor['column']]
                                    if convert_to_pst(point.time) <= next_time
                                ]
                                if current_data:
                                    historical_start = convert_to_pst(current_data[0].time)
                                    update_futures.append(
                                        self.process_sensor_update(
                                            sensor,
                                            historical_start,
                                            next_time,
                                            sensor_status,
                                            interval,
                                            current_data  # Pass incrementally built data
                                        )
                                    )
                            else:
                                update_futures.append(
                                    self.process_sensor_update(
                                        sensor,
                                        current,
                                        next_time,
                                        sensor_status,
                                        interval
                                    )
                                )
                        pbar.update(1)
                    
                    # Wait for all sensor updates to complete
                    if update_futures:
                        await asyncio.gather(*update_futures)
                    
                    current = next_time
                
                # Update KV records only once after all processing is complete
                await self._update_kv_records(interval, sensor_status)
                
                await self.async_logger.log(logging.INFO, f"Completed {interval} updates")
                
        except Exception as e:
            await self.async_logger.log(logging.ERROR, f"Error processing {interval} updates: {str(e)}")
            raise

    # Replace existing process methods with optimized versions
    async def process_daily_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict):
        """Process daily updates."""
        await self.process_interval_updates(start_date, end_date, sensor_status, 'daily')

    async def process_hourly_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict):
        """Process hourly updates."""
        await self.process_interval_updates(start_date, end_date, sensor_status, 'hourly')

    async def process_minute_updates(
        self,
        start_date: datetime,
        end_date: datetime,
        sensor_status: Dict,
        existing_data: Optional[Dict] = None
    ):
        """Process minute updates."""
        await self.process_interval_updates(start_date, end_date, sensor_status, 'minute', existing_data)

    async def generate_visualization(
        self,
        sensor: str,
        start_time: datetime,
        end_time: datetime,
        color_scheme: str = 'base'
    ) -> Optional[Path]:
        """Generate a visualization for a specific time range."""
        try:
            # Convert to PST for visualization boundaries
            start_pst = convert_to_pst(start_time)
            end_pst = convert_to_pst(end_time)
            
            # Get data from service
            data = await self.data_service.get_sensor_data(
                sensor,
                start_date=start_time.astimezone(timezone.utc),
                end_date=end_time.astimezone(timezone.utc)
            )
            
            if data:
                image = self.generator.generate_visualization(
                    data=data,
                    column=sensor,
                    color_scheme=color_scheme,
                    start_time=start_pst,
                    end_time=end_pst
                )
                
                # Save and return image path
                image_path = self.get_image_path(sensor, start_pst, end_pst)
                image.save(image_path)
                return image_path
            
            logger.warning(f"No data found for {sensor} between {start_time} and {end_time}")
            return None
            
        except Exception as e:
            logger.error(f"Error generating visualization for {sensor}: {str(e)}")
            raise
            
    async def _find_latest_image(self, sensor: str, interval: str) -> Optional[Tuple[datetime, datetime]]:
        """Find the latest image for a given sensor and interval type.
        Returns a tuple of (start_time, end_time) in PST timezone if found, None otherwise.
        """
        try:
            # Get the KV record for this sensor/interval
            kv_key = self.get_kv_key(sensor, interval)
            kv_value = self.cloudflare.get_kv_record(kv_key)
            
            if not kv_value:
                return None
                
            # Parse the JSON response - handle double encoding
            try:
                outer_data = json.loads(kv_value)
                # Handle KV API response format which includes a 'value' field
                if isinstance(outer_data, dict) and 'value' in outer_data:
                    kv_value = outer_data['value']
                
                kv_data = json.loads(kv_value)
                # Handle both old format (string path) and new format (JSON object)
                if isinstance(kv_data, dict):
                    path = kv_data.get('path')
                    # If we have last_processed time, use that directly
                    if kv_data.get('last_processed'):
                        try:
                            last_processed = datetime.fromisoformat(kv_data['last_processed'])
                            # Convert to PST for consistency
                            return None, convert_to_pst(last_processed)
                        except (ValueError, TypeError):
                            # If timestamp parsing fails, fall back to path parsing
                            pass
                else:
                    path = kv_value
                
                if not path:
                    return None
                    
                # Check if the record indicates an error state
                if isinstance(kv_data, dict) and kv_data.get('status') == 'error':
                    await self.async_logger.log(
                        logging.WARNING,
                        f"Found error state in KV for {sensor} {interval}. Will reprocess."
                    )
                    return None
                    
            except json.JSONDecodeError:
                # If not JSON, assume the value is the path directly (old format)
                path = kv_value
            
            # Parse the filename to get the date and minutes
            # Format: sensor/interval/YYYY-MM-DD_DDDD_MMMM.png
            try:
                parts = path.split('/')
                if len(parts) != 3:
                    return None
                    
                date_parts = parts[2].split('_')
                if len(date_parts) != 3:
                    return None
                    
                base_date = datetime.strptime(date_parts[0], '%Y-%m-%d')
                days_offset = int(date_parts[1])
                minutes_offset = int(date_parts[2].split('.')[0])
                
                # Convert to PST timezone
                base_date = base_date.replace(tzinfo=ZoneInfo('America/Los_Angeles'))
                
                # Calculate end time
                end_time = base_date
                if interval == 'daily':
                    end_time = end_time.replace(hour=23, minute=59)
                elif interval == 'hourly':
                    end_time = end_time.replace(hour=minutes_offset // 60, minute=59)
                else:  # minute
                    end_time = end_time.replace(hour=minutes_offset // 60, minute=minutes_offset % 60)
                
                # Calculate start time based on interval
                if interval == 'daily':
                    start_time = end_time.replace(hour=0, minute=0)
                elif interval == 'hourly':
                    start_time = end_time.replace(minute=0)
                else:  # minute
                    start_time = end_time
                
                # Adjust for days offset
                end_time = end_time - timedelta(days=days_offset)
                start_time = start_time - timedelta(days=days_offset)
                
                return start_time, end_time
                
            except (ValueError, IndexError) as e:
                await self.async_logger.log(
                    logging.ERROR,
                    f"Error parsing image path {path} for {sensor} {interval}: {str(e)}"
                )
                return None
                
        except Exception as e:
            await self.async_logger.log(
                logging.ERROR,
                f"Error finding latest image for {sensor} {interval}: {str(e)}"
            )
            return None

    async def backfill(self):
        """Backfill visualizations from last stored image to current time."""
        await self.async_logger.log(logging.INFO, "Starting backfill process")
        
        try:
            now = datetime.now(timezone.utc)
            now_pst = convert_to_pst(now)
            today_midnight_pst = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
            
            first_data = await self.data_service.get_sensor_data(self.sensors[0]['column'], limit=1)
            if not first_data:
                await self.async_logger.log(logging.WARNING, "No data found")
                return
            
            first_point_time = first_data[0].time
            first_point_pst = convert_to_pst(first_point_time)
            
            await self.async_logger.log(logging.INFO, f"First data point: {first_point_pst.isoformat()}")
            await self.async_logger.log(logging.INFO, f"Current time: {now_pst.isoformat()}")
            
            # Find latest images for each sensor and interval
            latest_times = {}
            for sensor in self.sensors:
                latest_times[sensor['column']] = {
                    interval: await self._find_latest_image(sensor['column'], interval)
                    for interval in ['daily', 'hourly', 'minute']
                }
                
                # Log the latest times found
                await self.async_logger.log(logging.INFO, f"Latest images for {sensor['column']}:")
                for interval, times in latest_times[sensor['column']].items():
                    if times:
                        await self.async_logger.log(logging.INFO, f"  - {interval}: {times[0].isoformat()} to {times[1].isoformat()}")
                    else:
                        await self.async_logger.log(logging.INFO, f"  - {interval}: No existing images")
            
            # Initialize start dates based on latest images or first data point
            start_dates = {
                'daily': first_point_pst.replace(hour=0, minute=0, second=0, microsecond=0),
                'hourly': first_point_pst.replace(minute=0, second=0, microsecond=0),
                'minute': first_point_pst
            }
            
            # Update start dates based on latest images
            for sensor in self.sensors:
                for interval in ['daily', 'hourly', 'minute']:
                    latest = latest_times[sensor['column']][interval]
                    if latest:
                        _, end_time = latest
                        if interval == 'daily':
                            next_start = end_time.replace(hour=0, minute=0) + timedelta(days=1)
                        elif interval == 'hourly':
                            next_start = end_time.replace(minute=0) + timedelta(hours=1)
                        else:  # minute
                            next_start = end_time + timedelta(minutes=1)
                        
                        start_dates[interval] = max(start_dates[interval], next_start)
            
            await self.async_logger.log(logging.INFO, "Calculated start dates:")
            for interval, start_date in start_dates.items():
                await self.async_logger.log(logging.INFO, f"  - {interval}: {start_date.isoformat()}")
            
            sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
            
            # Process daily visualizations
            if start_dates['daily'] < today_midnight_pst:
                await self.async_logger.log(logging.INFO, "=== Starting Daily Processing ===")
                await self.process_daily_updates(start_dates['daily'], today_midnight_pst, sensor_status)
                await self.async_logger.log(logging.INFO, "=== Daily Processing Complete ===")
            
            # Process hourly visualizations
            if start_dates['hourly'] < now_pst:
                await self.async_logger.log(logging.INFO, "=== Starting Hourly Processing ===")
                await self.process_hourly_updates(start_dates['hourly'], now_pst, sensor_status)
                await self.async_logger.log(logging.INFO, "=== Hourly Processing Complete ===")
            
            # Process minute visualizations
            if start_dates['minute'] < now_pst:
                await self.async_logger.log(logging.INFO, "=== Starting Minute Processing ===")
                # Get all data for compound visualizations
                compound_data = {}
                for sensor in self.sensors:
                    data = await self.data_service.get_sensor_data(
                        sensor['column'],
                        start_date=start_dates['minute'].astimezone(timezone.utc),
                        end_date=now_pst.astimezone(timezone.utc)
                    )
                    if data:
                        compound_data[sensor['column']] = data
                        await self.async_logger.log(logging.INFO, f"Retrieved {len(data)} total data points for {sensor['column']}")
                
                await self.process_minute_updates(
                    start_date=start_dates['minute'],
                    end_date=now_pst,
                    sensor_status=sensor_status,
                    existing_data=compound_data
                )
                await self.async_logger.log(logging.INFO, "=== Minute Processing Complete ===")
            
            await self.async_logger.log(logging.INFO, "=== Backfill Complete ===")
            
        except Exception as e:
            await self.async_logger.error("Error during backfill", exc_info=e)
            raise

    async def run(self):
        """Run the visualization service."""
        await self.async_logger.start()  # Start async logger
        await self.async_logger.log(logging.INFO, "Starting visualization service...")
        
        try:
            # Initial backfill
            await self.backfill()
            
            await self.async_logger.log(logging.INFO, "Starting continuous updates")
            
            # Create tasks for each update frequency
            daily_task = asyncio.create_task(self._run_daily_updates())
            hourly_task = asyncio.create_task(self._run_hourly_updates())
            minute_task = asyncio.create_task(self._run_minute_updates())
            
            # Wait for all tasks (they run indefinitely)
            await asyncio.gather(daily_task, hourly_task, minute_task)
            
        finally:
            await self.async_logger.stop()  # Ensure logger is stopped
            
    async def _get_last_processed_time(self, interval: str) -> Optional[datetime]:
        """Get the last processed time for a given interval from KV.
        KV is treated as the source of truth for resumption.
        """
        try:
            # Check each sensor's last processed time and return the earliest
            earliest_time = None
            latest_time = None
            missing_sensors = []

            for sensor in self.sensors:
                kv_key = self.get_kv_key(sensor['column'], interval)
                kv_value = self.cloudflare.get_kv_record(kv_key)
                
                if kv_value:
                    try:
                        # Handle double-encoded JSON from KV
                        outer_data = json.loads(kv_value)
                        # Handle KV API response format which includes a 'value' field
                        if isinstance(outer_data, dict) and 'value' in outer_data:
                            kv_value = outer_data['value']
                        
                        kv_data = json.loads(kv_value)
                        if isinstance(kv_data, dict):
                            # First try to get the last_processed timestamp
                            if kv_data.get('last_processed'):
                                try:
                                    last_processed = datetime.fromisoformat(kv_data['last_processed'])
                                    last_processed_pst = convert_to_pst(last_processed)
                                    if earliest_time is None or last_processed_pst < earliest_time:
                                        earliest_time = last_processed_pst
                                    if latest_time is None or last_processed_pst > latest_time:
                                        latest_time = last_processed_pst
                                    continue  # Skip to next sensor if we successfully got timestamp
                                except (ValueError, TypeError):
                                    pass  # Fall through to path parsing if timestamp invalid
                    except json.JSONDecodeError:
                        pass  # Fall through to path parsing if not JSON
                    
                    # Fall back to parsing the path if we couldn't get timestamp
                    times = await self._find_latest_image(sensor['column'], interval)
                    if times:
                        _, end_time = times  # We only care about the end time
                        if earliest_time is None or end_time < earliest_time:
                            earliest_time = end_time
                        if latest_time is None or end_time > latest_time:
                            latest_time = end_time
                else:
                    missing_sensors.append(sensor['column'])

            if missing_sensors:
                await self.async_logger.log(
                    logging.INFO,
                    f"No KV records found for sensors {', '.join(missing_sensors)} with interval {interval}. Will process from the beginning."
                )
                return None
            
            if earliest_time and latest_time:
                # If there's more than 2 intervals difference between earliest and latest,
                # we should reprocess from the earliest to ensure consistency
                interval_delta = self._get_interval_delta(interval)
                if latest_time - earliest_time > interval_delta * 2:
                    await self.async_logger.log(
                        logging.WARNING,
                        f"Large gap detected in {interval} processing times ({earliest_time} to {latest_time}). Will reprocess from {earliest_time}."
                    )
                return earliest_time
            
            return None
            
        except Exception as e:
            await self.async_logger.log(logging.ERROR, f"Error getting last processed time: {str(e)}")
            return None

    def _get_interval_delta(self, interval: str) -> timedelta:
        """Get the timedelta for a given interval type."""
        if interval == 'daily':
            return timedelta(days=1)
        elif interval == 'hourly':
            return timedelta(hours=1)
        else:  # minute
            return timedelta(minutes=1)

    async def _run_daily_updates(self):
        """Run daily updates at midnight PST."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                current_midnight = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Get last processed time
                last_processed = await self._get_last_processed_time('daily')
                if not last_processed:
                    # If no last processed time, start from beginning of current day
                    start_time = current_midnight
                else:
                    # Ensure we don't process future data
                    start_time = min(last_processed, now_pst)
                    # Round down to midnight
                    start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Calculate next midnight
                next_midnight = current_midnight + timedelta(days=1)
                
                # Only process if we have a full day's worth of data
                process_until = min(next_midnight, now_pst - timedelta(minutes=5))
                
                if start_time < process_until:
                    # Process daily updates
                    sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
                    await self.process_daily_updates(start_time, process_until, sensor_status)
                    # Update KV records
                    await self._update_kv_records('daily', sensor_status)
                
                # Calculate wait time to next processing cycle
                wait_seconds = max(0, (next_midnight - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO, f"Daily updates will run in {int(wait_seconds/3600)} hours")
                await asyncio.sleep(wait_seconds)
                
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in daily update cycle: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def _run_hourly_updates(self):
        """Run hourly updates at the start of each hour."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                current_hour = now_pst.replace(minute=0, second=0, microsecond=0)
                
                # Get last processed time
                last_processed = await self._get_last_processed_time('hourly')
                if not last_processed:
                    # If no last processed time, start from beginning of current hour
                    start_time = current_hour
                else:
                    # Ensure we don't process future data
                    start_time = min(last_processed, now_pst)
                    # Round down to hour
                    start_time = start_time.replace(minute=0, second=0, microsecond=0)
                
                # Calculate next hour
                next_hour = current_hour + timedelta(hours=1)
                
                # Only process if we have enough data (wait 5 minutes after the hour)
                process_until = min(next_hour, now_pst - timedelta(minutes=5))
                
                if start_time < process_until:
                    # Process hourly updates
                    sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
                    await self.process_hourly_updates(start_time, process_until, sensor_status)
                    # Update KV records
                    await self._update_kv_records('hourly', sensor_status)
                
                # Calculate wait time to next processing cycle
                wait_seconds = max(0, (next_hour - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO, f"Hourly updates will run in {int(wait_seconds/60)} minutes")
                await asyncio.sleep(wait_seconds)
                
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in hourly update cycle: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def _run_minute_updates(self):
        """Run minute updates every 5 minutes."""
        UPDATE_INTERVAL = timedelta(minutes=5)  # Update every 5 minutes
        
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                
                # Round current time down to nearest 5 minutes
                minutes = (now_pst.minute // 5) * 5
                current_time = now_pst.replace(minute=minutes, second=0, microsecond=0)
                
                # Get last processed time
                last_processed = await self._get_last_processed_time('minute')
                if not last_processed:
                    # If no last processed time, start from 5 minutes ago
                    start_time = current_time - UPDATE_INTERVAL
                else:
                    # Ensure we don't process future data
                    start_time = min(last_processed, now_pst)
                    # Round down to nearest 5 minutes
                    minutes = (start_time.minute // 5) * 5
                    start_time = start_time.replace(minute=minutes, second=0, microsecond=0)
                
                next_time = current_time + UPDATE_INTERVAL
                
                # Only process if we have enough data (wait 30 seconds after the 5-minute mark)
                process_until = min(next_time, now_pst - timedelta(seconds=30))
                
                if start_time < process_until:
                    # Process minute updates
                    sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
                    
                    # Get all historical data for compound visualizations
                    compound_data = {}
                    first_data = await self.data_service.get_sensor_data(self.sensors[0]['column'], limit=1)
                    
                    if first_data:
                        historical_start = convert_to_pst(first_data[0].time)
                        
                        # Fetch all historical data for each sensor
                        for sensor in self.sensors:
                            data = await self.data_service.get_sensor_data(
                                sensor['column'],
                                start_date=historical_start.astimezone(timezone.utc),
                                end_date=process_until.astimezone(timezone.utc)
                            )
                            if data:
                                compound_data[sensor['column']] = data
                                await self.async_logger.log(
                                    logging.DEBUG, 
                                    f"Retrieved {len(data)} historical points for {sensor['column']} from {historical_start} to {process_until}"
                                )
                    
                        await self.process_minute_updates(
                            start_date=historical_start,
                            end_date=process_until,
                            sensor_status=sensor_status,
                            existing_data=compound_data
                        )
                        
                        # Update KV records
                        await self._update_kv_records('minute', sensor_status)
                    else:
                        await self.async_logger.log(logging.WARNING, "No historical data found for minute updates")
                
                # Calculate wait time to next processing cycle
                next_process_time = current_time + UPDATE_INTERVAL
                wait_seconds = max(0, (next_process_time - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO, f"Minute updates will run in {int(wait_seconds)} seconds")
                await asyncio.sleep(wait_seconds)
                
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in minute update cycle: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

async def start_service(
    db_path: str = "data/environmental.db",
    output_dir: str = "data/visualizations",
    scale_factor: int = 4
):
    """Start the visualization service."""
    logging.info("Starting visualization service with configuration:")
    logging.info(f"Database path: {db_path}")
    logging.info(f"Output directory: {output_dir}")
    
    # Always sync with Supabase to ensure we have the latest data
    logger.info("Syncing data from Supabase...")
    await migrate_data(
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY,
        db_path=db_path
    )
    logger.info("Data sync complete")
    
    sensors = [
        {'column': 'temperature', 'color_scheme': 'redblue'},
        {'column': 'humidity', 'color_scheme': 'cyan'},
        {'column': 'pressure', 'color_scheme': 'green'},
        {'column': 'light', 'color_scheme': 'base'},
        {'column': 'uv', 'color_scheme': 'purple'},
        {'column': 'gas', 'color_scheme': 'green'}
    ]
    
    config = {
        'db_path': db_path,
        'output_dir': output_dir,
        'scale_factor': scale_factor,
        'max_workers': None,  # None means use CPU count
        'batch_size': 30,  # Number of images to process in parallel
        'minute_chunk_interval': 'hourly'  # Interval for minute cumulative view - minute, hourly, daily, weekly
    }
    
    service = VisualizationService(
        sensors=sensors,
        config=config
    )
    
    # Initialize async components
    await service.initialize()
    
    await service.run()

if __name__ == "__main__":
    asyncio.run(start_service())
