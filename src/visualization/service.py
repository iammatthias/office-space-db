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
from config.config import (
    SUPABASE_URL,
    SUPABASE_KEY
)

# Configure logging with a cleaner format
logging.basicConfig(
    level=logging.DEBUG,  # Enable debug logging
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class ProgressManager:
    """Manages progress bars and logging during batch operations."""
    
    def __init__(self, total: int, desc: str, disable_logging: bool = True):
        self.total = total
        self.desc = desc
        self.disable_logging = disable_logging
        
    @contextlib.contextmanager
    def progress_bar(self):
        """Context manager for progress bar that temporarily modifies logging."""
        if self.disable_logging:
            logging_level = logger.getEffectiveLevel()
            logger.setLevel(logging.WARNING)
            
        try:
            with tqdm(total=self.total, desc=self.desc) as pbar:
                yield pbar
        finally:
            if self.disable_logging:
                logger.setLevel(logging_level)

class VisualizationService:
    """Service for generating and managing environmental data visualizations."""
    
    def __init__(
        self,
        sensors: List[Dict[str, str]],
        db_path: str = "data/environmental.db",
        output_dir: str = "data/visualizations",
        scale_factor: int = 4,
        max_workers: int = None,  # None means use CPU count
        batch_size: int = 120  # Number of images to process in parallel
    ):
        """Initialize the visualization service."""
        self.output_dir = Path(output_dir)
        self.sensors = sensors
        self.data_service = DataService(db_path)
        self.batch_size = batch_size
        
        # Initialize thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization generator with scale factor
        self.generator = VisualizationGenerator(scale_factor=scale_factor)
        
        # Cache for created directories to avoid redundant checks
        self._directory_cache: Set[Path] = set()
        self._directory_cache.add(self.output_dir)
        
        # Create sensor directories upfront
        for sensor in sensors:
            for interval in ['daily', 'hourly', 'minute']:
                sensor_dir = self.output_dir / sensor['column'] / interval
                sensor_dir.mkdir(parents=True, exist_ok=True)
                self._directory_cache.add(sensor_dir)
        
        logger.info(f"Initialized visualization service:")
        logger.info(f"- Output directory: {output_dir}")
        logger.info(f"- Sensors: {', '.join(s['column'] for s in sensors)}")
        logger.info(f"- Batch size: {batch_size}")

    def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure directory exists, using cache to avoid redundant checks."""
        if directory not in self._directory_cache:
            directory.mkdir(parents=True, exist_ok=True)
            self._directory_cache.add(directory)
        
    def get_image_path(self, sensor: str, start_time: datetime, end_time: datetime, interval: str = 'daily') -> Path:
        """
        Generate file path for visualization.
        Images are stored by interval (daily/hourly/minute) using consistent format.
        Format: YYYY-MM-DD_DDDD_MMMM.png
        where:
        - YYYY-MM-DD is the date
        - DDDD is days since start (0-padded)
        - MMMM is minutes since midnight (0-padded)
        """
        # Ensure we're working with UTC times before converting to PST
        start_utc = start_time.astimezone(timezone.utc)
        end_utc = end_time.astimezone(timezone.utc)
        
        # Convert to PST for file organization
        start_pst = convert_to_pst(start_utc)
        end_pst = convert_to_pst(end_utc)
        
        # Get directory path
        image_dir = self.output_dir / sensor / interval
        self._ensure_directory_exists(image_dir)
        
        # Calculate days since start
        days_since_start = (end_pst.date() - start_pst.date()).days
        
        # Calculate minutes since midnight based on interval
        if interval == 'daily':
            # For daily, use end of day (1439 minutes)
            minutes_since_midnight = 1439
        elif interval == 'hourly':
            # For hourly, use end of hour (59 minutes)
            minutes_since_midnight = end_pst.hour * 60 + 59
        else:  # minute
            # For minute view, use exact minute
            minutes_since_midnight = end_pst.hour * 60 + end_pst.minute
        
        # Format: YYYY-MM-DD_DDDD_MMMM.png
        filename = f"{end_pst.strftime('%Y-%m-%d')}_{days_since_start:04d}_{minutes_since_midnight:04d}.png"
        
        logger.info(
            f"Generated path for {interval} image: {filename} "
            f"(date={end_pst.date()}, days_since_start={days_since_start}, "
            f"minutes={minutes_since_midnight}, start={start_pst.date()}, end={end_pst.date()})"
        )
        
        return image_dir / filename

    async def _save_image_batch(self, image_saves: List[Tuple[object, Path]]) -> None:
        """Save a batch of images concurrently using thread pool."""
        async def save_single(image, path: Path) -> None:
            def _save():
                try:
                    # Ensure parent directory exists
                    path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with io.BytesIO() as bio:
                        logger.info(f"Starting to save image to {path}")
                        image.save(bio, format='PNG', optimize=True)
                        img_data = bio.getvalue()
                        path.write_bytes(img_data)
                        logger.info(f"Successfully saved image to {path}")
                except Exception as e:
                    logger.error(f"Error saving image to {path}: {str(e)}")
                    raise
            
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    _save
                )
            except Exception as e:
                logger.error(f"Error in thread pool execution for {path}: {str(e)}")
                raise
        
        # Log the batch of files being saved
        logger.info(f"Starting to save batch of {len(image_saves)} images:")
        for _, path in image_saves:
            logger.info(f"  - Will save to: {path}")
        
        try:
            await asyncio.gather(*(save_single(img, path) for img, path in image_saves))
            logger.info("Successfully completed saving all images in batch")
        except Exception as e:
            logger.error(f"Error during batch save operation: {str(e)}")
            raise

    async def process_sensor_batch(
        self,
        tasks: List[Tuple[Dict[str, str], datetime, datetime, Dict, str]],
        pbar: Optional[tqdm] = None
    ):
        """Process a batch of sensor updates concurrently."""
        image_saves = []
        
        # Log the start of batch processing
        logger.info(f"Starting to process batch of {len(tasks)} tasks")
        
        # Gather all data first
        data_futures = [
            self.data_service.get_sensor_data(
                sensor['column'],
                start_date=start.astimezone(timezone.utc),
                end_date=end.astimezone(timezone.utc)
            )
            for sensor, start, end, _, _ in tasks
        ]
        all_data = await asyncio.gather(*data_futures)
        logger.info(f"Retrieved data for {len(all_data)} sensors")
        
        # Process each task with its data
        for (sensor, start, end, sensor_status, interval), data in zip(tasks, all_data):
            if data:
                start_pst = convert_to_pst(start)
                end_pst = convert_to_pst(end)
                
                # Debug log the visualization parameters
                logger.info(
                    f"Generating {interval} visualization for {sensor['column']}: "
                    f"start={start_pst.isoformat()}, end={end_pst.isoformat()}, "
                    f"days_since_start={(end_pst.date() - start_pst.date()).days}, "
                    f"data_points={len(data)}"
                )
                
                try:
                    # Generate visualization
                    image = self.generator.generate_visualization(
                        data=data,
                        column=sensor['column'],
                        color_scheme=sensor['color_scheme'],
                        start_time=start_pst,
                        end_time=end_pst,
                        interval=interval
                    )
                    
                    # Queue image for saving
                    image_path = self.get_image_path(sensor['column'], start_pst, end_pst, interval)
                    image_saves.append((image, image_path))
                    
                    # Debug log the queued save
                    logger.info(f"Successfully generated and queued save for {image_path}")
                    
                    sensor_status[sensor['column']] = {
                        'start_time': start_pst.isoformat(),
                        'end_time': end_pst.isoformat(),
                        'image_path': str(image_path)
                    }
                except Exception as e:
                    logger.error(f"Error generating visualization: {str(e)}")
            else:
                logger.warning(f"No data available for {sensor['column']} between {start} and {end}")
            
            if pbar:
                pbar.update(1)
        
        # Save all images in batch
        if image_saves:
            logger.info(f"Attempting to save batch of {len(image_saves)} images")
            try:
                await self._save_image_batch(image_saves)
                logger.info(f"Successfully saved batch of {len(image_saves)} images")
            except Exception as e:
                logger.error(f"Error saving image batch: {str(e)}")
        else:
            logger.warning("No images to save in this batch")

    async def process_hourly_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict):
        """Process hourly updates for all sensors with batching."""
        current = start_date
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        total_tasks = total_hours * len(self.sensors)
        
        progress = ProgressManager(total_tasks, "Generating hourly visualizations")
        
        with progress.progress_bar() as pbar:
            while current < end_date:
                batch_tasks = []
                
                # Create batch of tasks
                for _ in range(self.batch_size):
                    if current >= end_date:
                        break
                        
                    next_hour = current + timedelta(hours=1)
                    batch_tasks.extend([
                        (sensor, current, next_hour, sensor_status, 'hourly')
                        for sensor in self.sensors
                    ])
                    current = next_hour
                
                if batch_tasks:
                    await self.process_sensor_batch(batch_tasks, pbar)

    async def process_minute_updates(
        self,
        start_date: datetime,
        end_date: datetime,
        sensor_status: Dict,
        existing_data: Optional[Dict] = None
    ):
        """Process minute updates with optimized batching and continuous compound visualization."""
        try:
            start_pst = convert_to_pst(start_date)
            end_pst = convert_to_pst(end_date)
            
            logger.info(f"Processing minute updates from {start_pst} to {end_pst}")
            
            # Use existing data if provided, otherwise fetch new data
            sensor_data = existing_data
            if not sensor_data:
                # Get all data for the time range in parallel
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
                logger.warning(f"No data found for minute updates between {start_pst} and {end_pst}")
                return
            
            # Calculate total minutes for progress bar
            total_minutes = int((end_pst - start_pst).total_seconds() / 60)
            logger.info(f"Processing {total_minutes} minutes of data")
            
            with ProgressManager(total_minutes * len(self.sensors), "Generating minute images").progress_bar() as pbar:
                # Process each minute
                current = start_pst
                while current < end_pst:
                    batch_tasks = []
                    
                    # Calculate batch end time, ensuring we don't exceed end_pst
                    batch_end = min(current + timedelta(minutes=self.batch_size), end_pst)
                    logger.debug(f"Processing minute batch from {current} to {batch_end}")
                    
                    # Generate tasks for each minute in the batch
                    current_minute = current
                    while current_minute < batch_end:
                        minute_end = current_minute + timedelta(minutes=1)
                        minute_end_utc = minute_end.astimezone(timezone.utc)
                        
                        for sensor in self.sensors:
                            if sensor['column'] not in sensor_data:
                                pbar.update(1)
                                continue
                            
                            # For compound visualization, we want all data up to this point
                            # This ensures we maintain continuity across days
                            current_data = sensor_data[sensor['column']]
                            
                            if current_data:
                                # Find the earliest data point
                                earliest_data = min(current_data, key=lambda x: x.time)
                                earliest_time_pst = convert_to_pst(earliest_data.time)
                                
                                # Create task with the full data range for compound visualization
                                batch_tasks.append((
                                    sensor,
                                    earliest_time_pst,  # Always use earliest time as start
                                    minute_end,         # Use current minute as end
                                    sensor_status,
                                    'minute'
                                ))
                                
                                logger.debug(
                                    f"Queuing minute image: sensor={sensor['column']}, "
                                    f"start={earliest_time_pst.isoformat()}, "
                                    f"end={minute_end.isoformat()}, "
                                    f"data_points={len(current_data)}"
                                )
                            
                            pbar.update(1)
                            
                        current_minute = minute_end
                    
                    if batch_tasks:
                        # Process and save the batch
                        await self.process_sensor_batch(batch_tasks)
                        logger.debug(f"Processed and saved batch of {len(batch_tasks)} minute images")
                    
                    # Move current pointer to start of next batch
                    current = batch_end
                    
        except Exception as e:
            logger.error(f"Error processing minute updates: {str(e)}")
            raise

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
            
    async def backfill(self):
        """Backfill visualizations from last stored image to current time."""
        logger.info("Starting backfill process")
        
        now = datetime.now(timezone.utc)
        now_pst = convert_to_pst(now)
        today_midnight_pst = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
        
        first_data = await self.data_service.get_sensor_data(self.sensors[0]['column'], limit=1)
        if not first_data:
            logger.warning("No data found")
            return
        
        first_point_time = first_data[0].time
        first_point_pst = convert_to_pst(first_point_time)
        start_date = first_point_pst.replace(hour=0, minute=0, second=0, microsecond=0)
        
        sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
        
        total_days = (today_midnight_pst - start_date).days
        total_daily_tasks = total_days * len(self.sensors)
        
        logger.info(f"Processing {total_days} days of data for {len(self.sensors)} sensors")
        
        # Get all data upfront for compound visualizations
        compound_data = {}
        for sensor in self.sensors:
            data = await self.data_service.get_sensor_data(
                sensor['column'],
                start_date=first_point_time.astimezone(timezone.utc),
                end_date=now.astimezone(timezone.utc)
            )
            if data:
                compound_data[sensor['column']] = data
                logger.info(f"Retrieved {len(data)} total data points for {sensor['column']}")
        
        with tqdm(total=total_daily_tasks, desc="Generating daily images") as pbar:
            current = start_date
            while current < today_midnight_pst:
                next_day = current + timedelta(days=1)
                logger.info(f"Processing day {current.date()} to {next_day.date()}")
                
                # Create batch tasks for daily processing
                batch_tasks = [
                    (sensor, current, next_day, sensor_status, 'daily')
                    for sensor in self.sensors
                ]
                
                # Process the batch
                await self.process_sensor_batch(batch_tasks, pbar)
                
                # Generate hourly visualizations for this day
                await self.process_hourly_updates(current, next_day, sensor_status)
                
                current = next_day
        
        # Process minute visualizations as a continuous stream
        logger.info("Processing minute visualizations as continuous stream")
        await self.process_minute_updates(
            start_date=first_point_time,  # Start from first data point
            end_date=now,                 # End at current time
            sensor_status=sensor_status,
            existing_data=compound_data    # Use all data for compound visualization
        )
        
        logger.info("Backfill complete")
        
    async def run(self):
        """Run the visualization service."""
        logger.info("Starting visualization service...")
        
        # Initial backfill
        await self.backfill()
        
        logger.info("Starting continuous updates")
        
        sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
        
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                
                today_midnight_pst = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
                tomorrow_midnight_pst = today_midnight_pst + timedelta(days=1)
                
                wait_seconds = (tomorrow_midnight_pst - now_pst).total_seconds()
                logger.info(f"Waiting {int(wait_seconds/3600)} hours until next update cycle")
                await asyncio.sleep(wait_seconds)
                
                logger.info("Starting daily update cycle")
                
                # Process daily visualization with progress tracking
                with tqdm(total=len(self.sensors), desc="Generating daily images") as pbar:
                    await asyncio.gather(
                        *(self.process_sensor_update(
                            sensor,
                            today_midnight_pst,
                            tomorrow_midnight_pst,
                            sensor_status,
                            'daily'
                        ) for sensor in self.sensors)
                    )
                    pbar.update(len(self.sensors))
                
                # Process hourly visualizations
                await self.process_hourly_updates(today_midnight_pst, tomorrow_midnight_pst, sensor_status)
                
                # Process minute visualizations
                await self.process_minute_updates(today_midnight_pst, tomorrow_midnight_pst, sensor_status)
                
            except Exception as e:
                logger.error(f"Update cycle error: {str(e)}")
                await asyncio.sleep(60)

async def start_service(
    db_path: str = "data/environmental.db",
    output_dir: str = "data/visualizations",
    scale_factor: int = 4
):
    """Start the visualization service."""
    logging.info("Starting visualization service with configuration:")
    logging.info(f"Database path: {db_path}")
    logging.info(f"Output directory: {output_dir}")
    
    # Check if database exists and has data
    db_file = Path(db_path)
    if not db_file.exists() or db_file.stat().st_size < 1024:  # Less than 1KB is probably empty
        logger.info("Database not found or empty. Running migration from Supabase...")
        await migrate_data(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY,
            db_path=db_path
        )
    
    sensors = [
        {'column': 'temperature', 'color_scheme': 'redblue'},
        {'column': 'humidity', 'color_scheme': 'cyan'},
        {'column': 'pressure', 'color_scheme': 'green'},
        {'column': 'light', 'color_scheme': 'base'},
        {'column': 'uv', 'color_scheme': 'purple'},
        {'column': 'gas', 'color_scheme': 'green'}
    ]
    
    service = VisualizationService(
        sensors=sensors,
        db_path=db_path,
        output_dir=output_dir,
        scale_factor=scale_factor
    )
    
    # Initialize data service
    await service.data_service.initialize()
    
    await service.run()

if __name__ == "__main__":
    asyncio.run(start_service())
