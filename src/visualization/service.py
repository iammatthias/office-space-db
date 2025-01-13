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
        batch_size: int = 50  # Number of images to process in parallel
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
        
        # Calculate days since start and minutes since midnight
        days_since_start = (end_pst.date() - start_pst.date()).days
        
        if interval == 'daily':
            # For daily, use end of day (1439 minutes)
            minutes_since_midnight = 1439
        elif interval == 'hourly':
            # For hourly, use end of hour (59 minutes)
            minutes_since_midnight = end_pst.hour * 60 + 59
        else:  # minute (compound)
            # For minute, use exact minute
            minutes_since_midnight = end_pst.hour * 60 + end_pst.minute
        
        # Format: YYYY-MM-DD_DDDD_MMMM.png
        filename = f"{end_pst.strftime('%Y-%m-%d')}_{days_since_start:04d}_{minutes_since_midnight:04d}.png"
        
        return image_dir / filename

    async def _save_image_batch(self, image_saves: List[Tuple[object, Path]]) -> None:
        """Save a batch of images concurrently using thread pool."""
        async def save_single(image, path: Path) -> None:
            def _save():
                with io.BytesIO() as bio:
                    image.save(bio, format='PNG', optimize=True)
                    img_data = bio.getvalue()
                    path.write_bytes(img_data)
                    logger.debug(f"Saved image to {path}")
            
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                _save
            )
        
        # Log the batch of files being saved
        logger.debug(f"Saving batch of {len(image_saves)} images:")
        for _, path in image_saves:
            logger.debug(f"  - {path}")
        
        await asyncio.gather(*(save_single(img, path) for img, path in image_saves))

    async def process_sensor_batch(
        self,
        tasks: List[Tuple[Dict[str, str], datetime, datetime, Dict, str]],
        pbar: Optional[tqdm] = None
    ):
        """Process a batch of sensor updates concurrently."""
        image_saves = []
        
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
        
        # Process each task with its data
        for (sensor, start, end, sensor_status, interval), data in zip(tasks, all_data):
            if data:
                start_pst = convert_to_pst(start)
                end_pst = convert_to_pst(end)
                
                # Debug log the visualization parameters
                logger.debug(
                    f"Generating {interval} visualization for {sensor['column']}: "
                    f"start={start_pst.isoformat()}, end={end_pst.isoformat()}, "
                    f"days_since_start={(end_pst.date() - start_pst.date()).days}, "
                    f"data_points={len(data)}"
                )
                
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
                logger.debug(f"Queued save for {image_path}")
                
                sensor_status[sensor['column']] = {
                    'start_time': start_pst.isoformat(),
                    'end_time': end_pst.isoformat(),
                    'image_path': str(image_path)
                }
            
            if pbar:
                pbar.update(1)
        
        # Save all images in batch
        if image_saves:
            await self._save_image_batch(image_saves)

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
            
            # For compound visualization, we need to process every minute up to end_pst
            current = start_pst
            
            # Calculate total minutes including subsequent days
            total_minutes = int((end_pst - start_pst).total_seconds() / 60)
            
            with ProgressManager(total_minutes * len(self.sensors), "Generating minute images").progress_bar() as pbar:
                # Process each minute up to end_pst, including subsequent days
                while current < end_pst:
                    batch_tasks = []
                    batch_end = min(current + timedelta(minutes=self.batch_size), end_pst)
                    
                    # Create tasks for the current batch window
                    minutes_in_batch = int((batch_end - current).total_seconds() / 60)
                    for i in range(minutes_in_batch):
                        minute_start = current + timedelta(minutes=i)
                        minute_end = minute_start + timedelta(minutes=1)
                        
                        for sensor in self.sensors:
                            if sensor['column'] not in sensor_data:
                                pbar.update(1)
                                continue
                            
                            # For compound visualization, use all data from start_date up to current minute
                            current_data = [
                                point for point in sensor_data[sensor['column']]
                                if start_pst <= point.time <= minute_end
                            ]
                            
                            if current_data:
                                # For compound visualization, we need:
                                # - start_time: always the first data point time
                                # - end_time: the current minute we're processing
                                # This ensures proper row calculation in the visualization
                                batch_tasks.append((
                                    sensor,
                                    start_pst,  # Always use initial start time for compound visualization
                                    minute_end,  # Use the current minute as end time
                                    sensor_status,
                                    'minute'
                                ))
                                
                                # Debug logging to track image generation
                                logger.debug(
                                    f"Queuing minute image: start={start_pst.isoformat()}, "
                                    f"end={minute_end.isoformat()}, "
                                    f"days_since_start={(minute_end.date() - start_pst.date()).days}"
                                )
                            
                            pbar.update(1)
                    
                    if batch_tasks:
                        # Process and save the batch
                        await self.process_sensor_batch(batch_tasks)
                        
                        # Debug logging to confirm batch processing
                        logger.debug(f"Processed and saved batch of {len(batch_tasks)} minute images")
                    
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
        
        with tqdm(total=total_daily_tasks, desc="Generating daily images") as pbar:
            current = start_date
            while current < today_midnight_pst:
                next_day = current + timedelta(days=1)
                
                # Create batch tasks for daily processing
                batch_tasks = [
                    (sensor, current, next_day, sensor_status, 'daily')
                    for sensor in self.sensors
                ]
                
                # Process the batch
                await self.process_sensor_batch(batch_tasks, pbar)
                
                # Generate hourly visualizations for this day
                await self.process_hourly_updates(current, next_day, sensor_status)
                
                # Generate minute visualizations for this day
                # Always start from the first data point for compound visualizations
                await self.process_minute_updates(
                    first_point_time,  # Always start from first point
                    next_day,  # End at the current day's end
                    sensor_status,
                    compound_data  # Pass the complete dataset
                )
                
                current = next_day
        
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
        # {'column': 'humidity', 'color_scheme': 'cyan'},
        # {'column': 'pressure', 'color_scheme': 'green'},
        # {'column': 'light', 'color_scheme': 'base'},
        # {'column': 'uv', 'color_scheme': 'purple'},
        # {'column': 'gas', 'color_scheme': 'green'}
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
