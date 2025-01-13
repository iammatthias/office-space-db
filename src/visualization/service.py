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

from .generator import VisualizationGenerator
from .utils import EnvironmentalData, convert_to_pst
from .data_service import DataService
from .migrate_to_sqlite import migrate_data
from config.config import (
    SUPABASE_URL,
    SUPABASE_KEY
)

# Set up simplified logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class VisualizationService:
    """Service for generating and managing environmental data visualizations."""
    
    def __init__(
        self,
        sensors: List[Dict[str, str]],
        db_path: str = "data/environmental.db",
        output_dir: str = "data/visualizations",
        scale_factor: int = 4
    ):
        """Initialize the visualization service."""
        self.output_dir = Path(output_dir)
        self.sensors = sensors
        self.data_service = DataService(db_path)
        
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
        
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Configured sensors: {sensors}")
        
    def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure directory exists, using cache to avoid redundant checks."""
        if directory not in self._directory_cache:
            directory.mkdir(parents=True, exist_ok=True)
            self._directory_cache.add(directory)
        
    def get_image_path(self, sensor: str, start_time: datetime, end_time: datetime, interval: str = 'daily') -> Path:
        """
        Generate file path for visualization.
        Images are stored by interval (daily/hourly/minute) using ISO format.
        For compound visualizations (minute interval), we use the end_time for naming
        since it represents how much data is included.
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
        
        # Generate filename with ISO format based on interval
        if interval == 'hourly':
            filename = f"{start_pst.strftime('%Y-%m-%dT%H')}.png"
        elif interval == 'minute':
            # For compound visualizations, use end_time for the filename
            filename = f"{end_pst.strftime('%Y-%m-%dT%H%M')}.png"
        else:  # daily
            filename = f"{start_pst.date().isoformat()}.png"
        
        return image_dir / filename

    def _save_image_buffered(self, image, path: Path) -> None:
        """Save image with buffering for improved I/O performance."""
        # Use a memory buffer for the image data
        with io.BytesIO() as bio:
            # Save image to memory buffer with optimized settings
            image.save(bio, format='PNG', optimize=True)
            # Get the buffer content
            img_data = bio.getvalue()
            
            # Write the buffer to disk in one operation
            path.write_bytes(img_data)

    async def process_sensor_update(
        self,
        sensor: Dict[str, str],
        start_date: datetime,
        end_date: datetime,
        sensor_status: Dict,
        interval: str = 'daily'
    ):
        """Process update for a single sensor."""
        try:
            # Ensure we're working with UTC times for database query
            start_utc = start_date.astimezone(timezone.utc)
            end_utc = end_date.astimezone(timezone.utc)
            
            data = await self.data_service.get_sensor_data(
                sensor['column'],
                start_date=start_utc,
                end_date=end_utc
            )
            
            if data:
                # Convert back to PST for visualization
                start_pst = convert_to_pst(start_utc)
                end_pst = convert_to_pst(end_utc)
                
                image = self.generator.generate_visualization(
                    data=data,
                    column=sensor['column'],
                    color_scheme=sensor['color_scheme'],
                    start_time=start_pst,
                    end_time=end_pst,
                    interval=interval
                )
                
                # Save image to file using buffered I/O
                image_path = self.get_image_path(sensor['column'], start_pst, end_pst, interval)
                self._save_image_buffered(image, image_path)
                
                sensor_status[sensor['column']] = {
                    'start_time': start_pst.isoformat(),
                    'end_time': end_pst.isoformat(),
                    'image_path': str(image_path)
                }
                
                logger.info(f"Saved {interval} visualization for {sensor['column']} to {image_path}")
        except Exception as e:
            logger.error(f"Error updating {sensor['column']}: {str(e)}")
            raise
            
    async def process_hourly_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict):
        """Process hourly updates for all sensors."""
        current = start_date
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        total_tasks = total_hours * len(self.sensors)
        
        with tqdm(total=total_tasks, desc="Generating hourly images") as pbar:
            while current < end_date:
                next_hour = current + timedelta(hours=1)
                await asyncio.gather(
                    *(self.process_sensor_update(
                        sensor,
                        current,
                        next_hour,
                        sensor_status,
                        'hourly'
                    ) for sensor in self.sensors)
                )
                pbar.update(len(self.sensors))
                current = next_hour
            
    async def process_minute_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict):
        """Process minute updates for all sensors with compounding time ranges."""
        # Convert to PST for consistent midnight boundaries
        start_pst = convert_to_pst(start_date)
        end_pst = convert_to_pst(end_date)
        
        # Ensure we start at the beginning of a day
        current = start_pst.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate total minutes for progress bar
        total_minutes = int((end_pst - current).total_seconds() / 60)
        total_tasks = total_minutes * len(self.sensors)
        
        # First, fetch all data for each sensor for the entire time range
        sensor_data = {}
        for sensor in self.sensors:
            data = await self.data_service.get_sensor_data(
                sensor['column'],
                start_date=start_pst.astimezone(timezone.utc),
                end_date=end_pst.astimezone(timezone.utc)
            )
            if data:
                sensor_data[sensor['column']] = data
                logger.info(f"Retrieved {len(data)} data points for {sensor['column']} between {start_pst} and {end_pst}")
            else:
                logger.info(f"No data found for {sensor['column']} between {start_pst} and {end_pst}")
                continue
        
        with tqdm(total=total_tasks, desc="Generating minute images") as pbar:
            while current < end_pst:
                next_minute = current + timedelta(minutes=1)
                
                # Process each sensor in parallel
                tasks = []
                for sensor in self.sensors:
                    if sensor['column'] not in sensor_data:
                        pbar.update(1)
                        continue
                    
                    # Filter data up to the current minute
                    current_data = [
                        point for point in sensor_data[sensor['column']]
                        if point.time <= next_minute
                    ]
                    
                    if current_data:
                        image = self.generator.generate_visualization(
                            data=current_data,
                            column=sensor['column'],
                            color_scheme=sensor['color_scheme'],
                            start_time=start_pst,
                            end_time=next_minute,
                            interval='minute'
                        )
                        
                        # Save image to file using buffered I/O
                        image_path = self.get_image_path(sensor['column'], start_pst, next_minute, 'minute')
                        self._save_image_buffered(image, image_path)
                        
                        sensor_status[sensor['column']] = {
                            'start_time': start_pst.isoformat(),
                            'end_time': next_minute.isoformat(),
                            'image_path': str(image_path)
                        }
                    
                    pbar.update(1)
                
                current = next_minute
            
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
        
        with tqdm(total=total_daily_tasks, desc="Generating daily images") as pbar:
            current = start_date
            while current < today_midnight_pst:
                next_day = current + timedelta(days=1)
                
                # Generate daily visualization
                await asyncio.gather(
                    *(self.process_sensor_update(
                        sensor,
                        current,
                        next_day,
                        sensor_status,
                        'daily'
                    ) for sensor in self.sensors)
                )
                pbar.update(len(self.sensors))
                
                # Generate hourly visualizations for this day
                await self.process_hourly_updates(current, next_day, sensor_status)
                
                # Generate minute visualizations for this day
                # Start from the first data point for the first day
                minute_start = first_point_time if current == start_date else current
                await self.process_minute_updates(minute_start, next_day, sensor_status)
                
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
