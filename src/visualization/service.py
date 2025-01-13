"""Service for generating and managing environmental data visualizations."""

import asyncio
from datetime import datetime, timezone, timedelta, time
import os
from typing import List, Dict, Optional, Iterator, Tuple
from zoneinfo import ZoneInfo
import json
import logging
import io
from pathlib import Path

from supabase.client import create_client, Client
from .generator import VisualizationGenerator
from .utils import EnvironmentalData, convert_to_pst
from .data_service import DataService
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

class VisualizationService:
    """Service for generating and managing environmental data visualizations."""
    
    def __init__(
        self,
        sensors: List[Dict[str, str]],
        db_client: Client,
        output_dir: str = "data/visualizations",
        scale_factor: int = 4
    ):
        """Initialize the visualization service."""
        self.output_dir = Path(output_dir)
        self.sensors = sensors
        self.data_service = DataService(db_client)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization generator with scale factor
        self.generator = VisualizationGenerator(scale_factor=scale_factor)
        
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Configured sensors: {sensors}")

    def get_image_path(self, sensor: str, start_time: datetime, end_time: datetime) -> Path:
        """
        Generate file path for visualization.
        Images are stored by year/month/day and include the start and end timestamps
        at minute resolution.
        """
        start_pst = convert_to_pst(start_time)
        end_pst = convert_to_pst(end_time)
        
        # Create nested directory structure based on start time
        image_dir = self.output_dir / sensor / str(start_pst.year) / f"{start_pst.month:02d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with start and end timestamps
        filename = f"{start_pst.strftime('%Y%m%d_%H%M')}-{end_pst.strftime('%Y%m%d_%H%M')}.png"
        
        return image_dir / filename

    async def process_sensor_update(
        self,
        sensor: Dict[str, str],
        start_date: datetime,
        end_date: datetime,
        sensor_status: Dict
    ):
        """Process update for a single sensor."""
        try:
            # Convert PST dates to UTC for database query
            start_utc = start_date.astimezone(timezone.utc)
            end_utc = end_date.astimezone(timezone.utc)
            
            data = await self.data_service.get_sensor_data(
                sensor['column'],
                start_date=start_utc,
                end_date=end_utc
            )
            
            if data:
                image = self.generator.generate_visualization(
                    data=data,
                    column=sensor['column'],
                    color_scheme=sensor['color_scheme'],
                    start_time=start_date,  # Keep PST for visualization
                    end_time=end_date
                )
                
                # Save image to file
                image_path = self.get_image_path(sensor['column'], start_date, end_date)
                image.save(image_path)
                
                sensor_status[sensor['column']] = {
                    'start_time': start_date.isoformat(),
                    'end_time': end_date.isoformat(),
                    'image_path': str(image_path)
                }
                
                logger.info(f"Saved visualization for {sensor['column']} to {image_path}")
        except Exception as e:
            logger.error(f"Error updating {sensor['column']}: {str(e)}")
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
        current_time = datetime.now(timezone.utc)
        current_time_pst = convert_to_pst(current_time)
        
        first_data = await self.data_service.get_sensor_data(self.sensors[0]['column'], limit=1)
        if not first_data:
            logger.warning("No data found")
            return
            
        # Ensure we start at midnight PST
        first_point_time = convert_to_pst(first_data[0].time)
        start_date = datetime.combine(
            first_point_time.date(),
            time.min,
            tzinfo=ZoneInfo("America/Los_Angeles")
        )
        
        sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
        
        current = start_date
        while current < current_time_pst:
            next_day = current + timedelta(days=1)
            await asyncio.gather(
                *(self.process_sensor_update(
                    sensor,
                    current,
                    next_day,
                    sensor_status
                ) for sensor in self.sensors)
            )
            current = next_day
            
        logger.info("Backfill complete")

    async def run(self):
        """Run the visualization service."""
        logger.info("Starting visualization service...")
        
        # Initial backfill
        await self.backfill()
        
        logger.info("Starting updates")
        
        sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
        
        while True:
            try:
                # Always work in PST for visualization boundaries
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                
                # Wait until midnight PST
                tomorrow_pst = now_pst.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                wait_seconds = (tomorrow_pst - now_pst).total_seconds()
                
                logger.info(f"Next update in {int(wait_seconds)} seconds")
                await asyncio.sleep(wait_seconds)
                
                # Process the previous period (midnight to midnight PST)
                start_time = tomorrow_pst - timedelta(days=1)  # Previous midnight PST
                end_time = tomorrow_pst  # Next midnight PST
                
                await asyncio.gather(
                    *(self.process_sensor_update(
                        sensor,
                        start_time,
                        end_time,
                        sensor_status
                    ) for sensor in self.sensors)
                )
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)

async def start_service(
    supabase_url: str = SUPABASE_URL,
    supabase_key: str = SUPABASE_KEY,
    output_dir: str = "data/visualizations",
    scale_factor: int = 4
):
    """Start the visualization service."""
    logging.info("Starting visualization service with configuration:")
    logging.info(f"Supabase URL: {supabase_url}")
    logging.info(f"Output directory: {output_dir}")
    
    db_client = create_client(supabase_url, supabase_key)
    
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
        db_client=db_client,
        output_dir=output_dir,
        scale_factor=scale_factor
    )
    
    # Initialize data service
    await service.data_service.initialize()
    
    await service.run()

if __name__ == "__main__":
    asyncio.run(start_service())
