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
        self.db_client = db_client
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization generator with scale factor
        self.generator = VisualizationGenerator(scale_factor=scale_factor)
        
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Configured sensors: {sensors}")

    async def get_sensor_data(
        self, 
        sensor: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List["EnvironmentalData"]:
        """
        Fetch sensor data from database with pagination support.
        Expects UTC timestamps for database queries.
        Returns data with timestamps in UTC.
        """
        column_map = {
            'temperature': 'temp',
            'humidity': 'hum',
            'pressure': 'pressure',
            'light': 'lux',
            'uv': 'uv',
            'gas': 'gas',
        }
        
        db_column = column_map.get(sensor)
        if not db_column:
            raise ValueError(f"Unknown sensor type: {sensor}")
        
        # Ensure dates are in UTC for database query
        if start_date and not start_date.tzinfo:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date and not end_date.tzinfo:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        all_rows = []
        page_size = 1000  # Supabase's maximum limit
        last_timestamp = None
        
        while True:
            query = self.db_client.table('environmental_data').select('time', db_column)
            
            if start_date:
                query = query.gte('time', start_date.isoformat())
            if end_date:
                query = query.lt('time', end_date.isoformat())
            if last_timestamp:
                query = query.gt('time', last_timestamp)
            
            query = query.order('time', desc=False).limit(page_size)
            
            if limit is not None:
                remaining = limit - len(all_rows)
                if remaining <= 0:
                    break
                query = query.limit(min(page_size, remaining))
            
            data = query.execute()
            page_rows = data.data
            if not page_rows:
                break
                
            all_rows.extend(page_rows)
            last_timestamp = page_rows[-1]['time']
            
            if len(page_rows) < page_size:
                break
            if limit is not None and len(all_rows) >= limit:
                break
        
        # Parse rows and ensure all timestamps are in UTC
        parsed_data = [EnvironmentalData(dt, value) for dt, value in self._parse_rows(all_rows, db_column)]
        
        # Only do minute-by-minute fill if a date range is provided
        if start_date and end_date and parsed_data:
            # Convert to PST for minute alignment
            start_pst = convert_to_pst(start_date)
            end_pst = convert_to_pst(end_date)
            current = start_pst.replace(second=0, microsecond=0)
            
            # Generate all minutes in PST
            all_minutes = []
            while current < end_pst:
                all_minutes.append(current)
                current += timedelta(minutes=1)
            
            # Create minute map in PST
            minute_dict = {
                convert_to_pst(point.time).replace(second=0, microsecond=0): point.value
                for point in parsed_data
            }
            
            # Create data points for each minute
            valid_data = [
                EnvironmentalData(minute, minute_dict.get(minute, None))
                for minute in all_minutes
            ]
            
            # Interpolate missing values
            valid_data = self._interpolate_missing_values(valid_data)
        else:
            valid_data = parsed_data
                
        logger.info(f"Retrieved {len(valid_data)} valid data points for {sensor}")
        return valid_data
    
    def _parse_rows(self, rows: List[Dict], db_column: str) -> Iterator[Tuple[datetime, float]]:
        """Parse database rows into timestamp-value pairs."""
        for row in rows:
            try:
                timestamp = row['time'].replace('Z', '+00:00')
                dt = datetime.fromisoformat(timestamp)
                
                value = row[db_column]
                if value is not None:
                    try:
                        float_value = float(value)
                        yield dt, float_value
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {db_column} at {dt}: {value}")
                        continue
            except (ValueError, AttributeError) as e:
                logger.warning(f"Error parsing data for {db_column}: {e}")
                continue

    def _interpolate_missing_values(self, data: List["EnvironmentalData"]) -> List["EnvironmentalData"]:
        """Replace None values with a simple linear interpolation based on the nearest known neighbors."""
        i = 0
        n = len(data)
        while i < n:
            if data[i].value is None:
                start_idx = i - 1
                while i < n and data[i].value is None:
                    i += 1
                end_idx = i
                
                if start_idx >= 0 and end_idx < n:
                    start_val = data[start_idx].value
                    end_val = data[end_idx].value
                    gap_len = end_idx - start_idx
                    if start_val is not None and end_val is not None:
                        step = (end_val - start_val) / gap_len
                        for fill_idx in range(start_idx + 1, end_idx):
                            offset = fill_idx - start_idx
                            data[fill_idx].value = start_val + step * offset
                    else:
                        for fill_idx in range(start_idx + 1, end_idx):
                            data[fill_idx].value = start_val if start_val is not None else end_val
                else:
                    if start_idx < 0 and end_idx < n:
                        for fill_idx in range(end_idx):
                            data[fill_idx].value = data[end_idx].value
                    elif start_idx >= 0 and end_idx >= n:
                        for fill_idx in range(start_idx+1, n):
                            data[fill_idx].value = data[start_idx].value
            else:
                i += 1
        return data

    def get_image_path(self, sensor: str, timestamp: datetime) -> Path:
        """
        Generate file path for visualization.
        Images are stored by year/month/day and include the timestamp
        at minute resolution (1440px wide).
        """
        pst_time = convert_to_pst(timestamp)
        start_time = (pst_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        # Create nested directory structure
        image_dir = self.output_dir / sensor / str(start_time.year) / f"{start_time.month:02d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with start and end timestamps
        filename = f"{start_time.strftime('%Y%m%d_%H%M')}-{end_time.strftime('%Y%m%d_%H%M')}.png"
        
        return image_dir / filename

    async def process_sensor_update(self, sensor: Dict[str, str], start_date: datetime, end_date: datetime, sensor_status: Dict):
        """Process update for a single sensor."""
        try:
            # Convert PST dates to UTC for database query
            start_utc = start_date.astimezone(timezone.utc)
            end_utc = end_date.astimezone(timezone.utc)
            
            data = await self.get_sensor_data(
                sensor['column'],
                start_date=start_utc,
                end_date=end_utc
            )
            
            if data:
                image = self.generator.generate_visualization(
                    data=data,
                    column=sensor['column'],
                    color_scheme=sensor['color_scheme'],
                    start_time=start_date  # Keep PST for visualization
                )
                
                # Save image to file
                image_path = self.get_image_path(sensor['column'], end_date)
                image.save(image_path)
                
                sensor_status[sensor['column']] = {
                    'last_update': end_date.isoformat(),
                    'image_path': str(image_path)
                }
                
                logger.info(f"Saved visualization for {sensor['column']} to {image_path}")
        except Exception as e:
            logger.error(f"Error updating {sensor['column']}: {str(e)}")
            raise

    async def backfill(self):
        """Backfill visualizations from last stored image to current time."""
        logger.info("Starting backfill process")
        current_time = datetime.now(timezone.utc)
        current_time_pst = convert_to_pst(current_time)
        
        first_data = await self.get_sensor_data(self.sensors[0]['column'], limit=1)
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
        {'column': 'humidity', 'color_scheme': 'cyan'},
        {'column': 'pressure', 'color_scheme': 'green'},
        {'column': 'light', 'color_scheme': 'base'},
        {'column': 'uv', 'color_scheme': 'purple'},
        {'column': 'gas', 'color_scheme': 'green'}
    ]
    
    service = VisualizationService(
        sensors=sensors,
        db_client=db_client,
        output_dir=output_dir,
        scale_factor=scale_factor
    )
    
    await service.run()

if __name__ == "__main__":
    asyncio.run(start_service())
