import asyncio
from datetime import datetime, timezone, timedelta, time
import os
from typing import List, Dict, Optional, Iterator, Tuple
import aioboto3
from zoneinfo import ZoneInfo
import json
import logging
import httpx
import io
import boto3
import cloudflare

from supabase.client import create_client, Client
from .generator import VisualizationGenerator
from .utils import EnvironmentalData, convert_to_pst
from .helpers import hourly_range, get_next_hour, format_timestamp
from config.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_ACCESS_KEY_ID,
    CLOUDFLARE_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME,
    CLOUDFLARE_API_TOKEN,
    KV_NAMESPACE_ID
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
        r2_credentials: Dict[str, str],
        kv_namespace: str,
        sensors: List[Dict[str, str]],
        db_client: Client,
        bucket_name: str,
        scale_factor: int = 4
    ):
        """Initialize the visualization service."""
        self.bucket_name = bucket_name
        self.kv_namespace = kv_namespace
        self.sensors = sensors
        self.db_client = db_client
        self.r2_credentials = r2_credentials
        
        # Initialize R2 session
        self.r2_session = aioboto3.Session()
        
        # Initialize Cloudflare client
        self.cf = cloudflare.Cloudflare(api_key=CLOUDFLARE_API_TOKEN)
        
        # Initialize HTTP client for KV operations
        self.http_client = httpx.AsyncClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/storage/kv/namespaces/{kv_namespace}",
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                "Content-Type": "application/json"
            }
        )
        
        # Initialize visualization generator with scale factor
        self.generator = VisualizationGenerator(scale_factor=scale_factor)
        
        logging.info(f"Bucket name: {bucket_name}")
        logging.info(f"KV namespace: {kv_namespace}")
        logging.info(f"Configured sensors: {sensors}")

    async def get_kv(self, key: str) -> Optional[str]:
        """Get value from KV store."""
        try:
            response = await self.http_client.get(f"/values/{key}")
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            logger.error(f"Failed to get KV value for {key}: {str(e)}")
            return None

    async def put_kv(self, key: str, value: str):
        """Put value in KV store."""
        try:
            response = await self.http_client.put(f"/values/{key}", content=value)
            response.raise_for_status()
            logger.info(f"Successfully updated KV value for {key}")
        except httpx.HTTPError as e:
            logger.error(f"Failed to put KV value for {key}: {str(e)}")
            raise

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

    async def get_sensor_data(
        self, 
        sensor: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List["EnvironmentalData"]:
        """
        Fetch sensor data from database with pagination support.
        Now performs a simple linear interpolation for missing values 
        to avoid dark vertical bands in the generated images.
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
        
        parsed_data = [EnvironmentalData(dt, value) for dt, value in self._parse_rows(all_rows, db_column)]
        
        # Only do minute-by-minute fill if a date range is provided
        if start_date and end_date and parsed_data:
            start_pst = convert_to_pst(start_date)
            end_pst = convert_to_pst(end_date)
            current = start_pst.replace(second=0, microsecond=0)
            all_minutes = []
            while current < end_pst:
                all_minutes.append(current)
                current += timedelta(minutes=1)
            
            minute_dict = {
                convert_to_pst(point.time).replace(second=0, microsecond=0): point.value
                for point in parsed_data
            }
            
            # Assign None where data is missing so we can interpolate
            valid_data = [
                EnvironmentalData(minute, minute_dict.get(minute, None))
                for minute in all_minutes
            ]
            
            # Perform a simple linear interpolation for missing values
            valid_data = self._interpolate_missing_values(valid_data)
        else:
            valid_data = parsed_data
                
        logger.info(f"Retrieved {len(valid_data)} valid data points for {sensor}")
        return valid_data
    
    def _interpolate_missing_values(self, data: List["EnvironmentalData"]) -> List["EnvironmentalData"]:
        """
        Replace None values with a simple linear interpolation 
        based on the nearest known neighbors. Fallback is forward/backward fill.
        """
        # First pass: gather indices of consecutive gaps
        i = 0
        n = len(data)
        while i < n:
            # If we have a gap, find where it ends
            if data[i].value is None:
                start_idx = i - 1
                while i < n and data[i].value is None:
                    i += 1
                end_idx = i  # first non-None after the gap
                
                # If both sides known, do linear interpolation
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
                        # fallback forward fill or backward fill
                        for fill_idx in range(start_idx + 1, end_idx):
                            data[fill_idx].value = start_val if start_val is not None else end_val
                else:
                    # fill from left side or right side if possible
                    if start_idx < 0 and end_idx < n:
                        # use the first known value to fill
                        for fill_idx in range(end_idx):
                            data[fill_idx].value = data[end_idx].value
                    elif start_idx >= 0 and end_idx >= n:
                        # use the last known value to fill
                        for fill_idx in range(start_idx+1, n):
                            data[fill_idx].value = data[start_idx].value
            else:
                i += 1
        return data

    async def upload_to_r2(self, image_bytes: bytes, key: str) -> str:
        """Upload image to R2 bucket."""
        logger.info(f"Uploading to R2: {key}")
        try:
            async with self.r2_session.client(
                's3',
                endpoint_url=self.r2_credentials['endpoint'],
                aws_access_key_id=self.r2_credentials['access_key_id'],
                aws_secret_access_key=self.r2_credentials['secret_access_key']
            ) as s3:
                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=image_bytes,
                    ContentType='image/png'
                )
                logger.info(f"Successfully uploaded to R2: {key}")
                return f"https://{self.bucket_name}.r2.dev/{key}"
        except Exception as e:
            logger.error(f"Failed to upload to R2: {str(e)}")
            raise

    def get_image_key(self, sensor: str, timestamp: datetime) -> str:
        """
        Generate R2 key for hourly image.
        Hourly images are stored by year and grow vertically throughout the year.
        Each row represents 24 hours of data at minute resolution.
        """
        pst_time = convert_to_pst(timestamp)
        return f"{sensor}/{pst_time.strftime('%Y')}/incremental.png"

    def get_daily_image_key(self, sensor: str, timestamp: datetime) -> str:
        """
        Generate R2 key for daily image.
        Daily images are stored by year/month/day and represent a single 24-hour period
        at minute resolution (1440px wide).
        """
        pst_time = convert_to_pst(timestamp)
        return f"{sensor}/{pst_time.strftime('%Y/%m/%d')}/daily.png"

    async def generate_and_store(
        self,
        sensor_config: Dict[str, str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        single_row: bool = False
    ):
        """Generate visualization and store in R2."""
        logger.info(f"Starting generate_and_store for {sensor_config['column']}")
        try:
            data = await self.get_sensor_data(
                sensor_config['column'],
                start_date,
                end_date,
                limit=None
            )
            if not data:
                logger.warning(f"No data found for {sensor_config['column']}")
                return
            
            logger.info(f"Generating visualization for {sensor_config['column']} with {len(data)} data points")
            image = self.generator.generate_hourly_visualization(
                data=data,
                column=sensor_config['column'],
                color_scheme=sensor_config['color_scheme'],
                start_date=start_date,
                end_date=end_date
            )
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            timestamp = end_date or datetime.now(timezone.utc)
            key = self.get_image_key(sensor_config['column'], timestamp)
            image_url = await self.upload_to_r2(img_byte_arr, key)
            
            # Update KV for hourly visualization
            await self.update_kv(sensor_config['column'], image_url, 'hourly', timestamp)
            
            # Generate daily if it's the last hour of the day
            if timestamp.hour == 23 or (end_date and end_date.hour == 23):
                day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                daily_data = await self.get_sensor_data(
                    sensor_config['column'],
                    start_date=day_start,
                    end_date=timestamp,
                    limit=None
                )
                
                if daily_data:
                    daily_image = self.generator.generate_daily_visualization(
                        data=daily_data,
                        column=sensor_config['column'],
                        color_scheme=sensor_config['color_scheme'],
                        day_start=day_start
                    )
                    daily_img_byte_arr = io.BytesIO()
                    daily_image.save(daily_img_byte_arr, format='PNG')
                    daily_img_byte_arr = daily_img_byte_arr.getvalue()
                    
                    daily_key = self.get_daily_image_key(sensor_config['column'], timestamp)
                    daily_image_url = await self.upload_to_r2(daily_img_byte_arr, daily_key)
                    
                    # Update KV for daily visualization
                    await self.update_kv(sensor_config['column'], daily_image_url, 'daily', timestamp)
            
            logger.info(f"Successfully completed generate_and_store for {sensor_config['column']}")
        except Exception as e:
            logger.error(f"Error in generate_and_store for {sensor_config['column']}: {str(e)}")
            raise

    async def process_sensor_backfill(self, sensor: str, start_date: datetime, end_date: datetime):
        """Process backfill for a single sensor."""
        sensor_config = next((s for s in self.sensors if s['column'] == sensor), None)
        if not sensor_config:
            raise ValueError(f"Unknown sensor: {sensor}")
        
        data = await self.get_sensor_data(sensor, start_date, end_date)
        
        current = start_date
        while current < end_date:
            day_end = current + timedelta(days=1)
            day_data = [point for point in data if current <= point.time < day_end]
            
            if day_data:
                daily_image = self.generator.generate_daily_visualization(
                    day_data,
                    sensor,
                    sensor_config['color_scheme'],
                    current
                )
                daily_key = self.get_daily_image_key(sensor, current)
                daily_bytes = io.BytesIO()
                daily_image.save(daily_bytes, format='PNG')
                await self.upload_to_r2(daily_bytes.getvalue(), daily_key)
            
            current = day_end
        
        # Generate incremental visualization for the year
        year = start_date.year
        year_data = [point for point in data if point.time.year == year]
        if year_data:
            incremental_image = self.generator.generate_incremental_visualization(
                year_data,
                sensor,
                sensor_config['color_scheme'],
                year
            )
            incremental_key = self.get_image_key(sensor, start_date)
            incremental_bytes = io.BytesIO()
            incremental_image.save(incremental_bytes, format='PNG')
            await self.upload_to_r2(incremental_bytes.getvalue(), incremental_key)

    async def process_sensor_update(self, sensor: Dict[str, str], start_date: datetime, end_date: datetime, sensor_status: Dict):
        """Process update for a single sensor."""
        try:
            data = await self.get_sensor_data(
                sensor['column'],
                start_date=start_date,
                end_date=end_date
            )
            
            if data:
                daily_image = self.generator.generate_daily_visualization(
                    data=data,
                    column=sensor['column'],
                    color_scheme=sensor['color_scheme'],
                    day_start=start_date
                )
                daily_img_byte_arr = io.BytesIO()
                daily_image.save(daily_img_byte_arr, format='PNG')
                daily_img_byte_arr = daily_img_byte_arr.getvalue()
                
                daily_key = self.get_daily_image_key(sensor['column'], end_date)
                daily_url = await self.upload_to_r2(daily_img_byte_arr, daily_key)
                
                await self.update_kv(sensor['column'], daily_url, 'daily', end_date)
                
                year = start_date.year
                year_start = datetime(year, 1, 1, tzinfo=ZoneInfo("America/Los_Angeles"))
                year_end = datetime(year + 1, 1, 1, tzinfo=ZoneInfo("America/Los_Angeles"))
                
                year_data = await self.get_sensor_data(
                    sensor['column'],
                    start_date=year_start.astimezone(timezone.utc),
                    end_date=year_end.astimezone(timezone.utc)
                )
                incremental_image = self.generator.generate_incremental_visualization(
                    data=year_data,
                    column=sensor['column'],
                    color_scheme=sensor['color_scheme'],
                    year=year
                )
                incremental_img_byte_arr = io.BytesIO()
                incremental_image.save(incremental_img_byte_arr, format='PNG')
                incremental_img_byte_arr = incremental_img_byte_arr.getvalue()
                
                incremental_key = self.get_image_key(sensor['column'], end_date)
                incremental_url = await self.upload_to_r2(incremental_img_byte_arr, incremental_key)
                
                await self.update_kv(sensor['column'], incremental_url, 'incremental', end_date)
                
                sensor_status[sensor['column']] = {
                    'last_update': end_date.isoformat(),
                    'daily_url': daily_url,
                    'incremental_url': incremental_url
                }
        except Exception as e:
            logger.error(f"Error updating {sensor['column']}: {str(e)}")
            raise

    async def backfill(self):
        """Backfill visualizations from last stored image to current time."""
        logger.info("Starting backfill process")
        current_time = datetime.now(timezone.utc)
        next_hour = get_next_hour(current_time)
        
        first_data = await self.get_sensor_data(self.sensors[0]['column'], limit=1)
        if not first_data:
            logger.warning("No data found")
            return
            
        first_point_time = convert_to_pst(first_data[0].time)
        start_date = datetime.combine(
            first_point_time.date(),
            time.min,
            tzinfo=ZoneInfo("America/Los_Angeles")
        )
        
        await asyncio.gather(
            *(self.process_sensor_backfill(
                sensor['column'],
                start_date,
                next_hour
            ) for sensor in self.sensors)
        )
        logger.info("Backfill complete")

    async def run(self):
        """Run the visualization service."""
        logger.info("Starting visualization service...")
        
        # Initial backfill
        await self.backfill()
        
        logger.info("Starting hourly updates")
        
        sensor_status = {sensor['column']: {'last_success': None, 'last_error': None} for sensor in self.sensors}
        
        while True:
            try:
                now = datetime.now(timezone.utc)
                next_hour = get_next_hour(now)
                
                if now < next_hour:
                    wait_seconds = (next_hour - now).total_seconds()
                    logger.info(f"Next update in {int(wait_seconds)} seconds")
                    await asyncio.sleep(wait_seconds)
                
                start_time = now.replace(minute=0, second=0, microsecond=0)
                await asyncio.gather(
                    *(self.process_sensor_update(sensor, start_time, next_hour, sensor_status) 
                      for sensor in self.sensors)
                )
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)

    async def update_kv(self, sensor: str, url: str, type: str, timestamp: datetime):
        """
        Update KV store with image URL.
        Type can be 'daily' or 'incremental'.
        """
        if type not in ['daily', 'incremental', 'hourly']:
            raise ValueError(f"Invalid type: {type}")
        
        pst_time = convert_to_pst(timestamp)
        
        key = f"{sensor}_images"
        data_str = await self.get_kv(key)
        data = json.loads(data_str) if data_str else {}
        
        year = str(pst_time.year)
        if year not in data:
            data[year] = {'daily': {}, 'incremental': None, 'hourly': None}
            
        if type == 'daily':
            date_key = pst_time.strftime('%Y-%m-%d')
            data[year]['daily'][date_key] = url
        elif type == 'incremental':
            data[year]['incremental'] = url
        else:
            data[year]['hourly'] = url
            
        await self.put_kv(key, json.dumps(data))

async def start_service(
    supabase_url: str = SUPABASE_URL,
    supabase_key: str = SUPABASE_KEY,
    r2_endpoint: str = f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com",
    bucket_name: str = R2_BUCKET_NAME,
    kv_namespace: str = KV_NAMESPACE_ID,
    scale_factor: int = 4
):
    """Start the visualization service."""
    logging.info("Starting visualization service with configuration:")
    logging.info(f"Supabase URL: {supabase_url}")
    logging.info(f"R2 Endpoint: {r2_endpoint}")
    logging.info(f"Bucket: {bucket_name}")
    logging.info(f"KV Namespace: {kv_namespace}")
    
    db_client = create_client(supabase_url, supabase_key)
    
    r2_credentials = {
        'endpoint': r2_endpoint,
        'access_key_id': CLOUDFLARE_ACCESS_KEY_ID,
        'secret_access_key': CLOUDFLARE_SECRET_ACCESS_KEY
    }
    
    sensors = [
        {'column': 'temperature', 'color_scheme': 'redblue'},
        {'column': 'humidity', 'color_scheme': 'cyan'},
        {'column': 'pressure', 'color_scheme': 'green'},
        {'column': 'light', 'color_scheme': 'base'},
        {'column': 'uv', 'color_scheme': 'purple'},
        {'column': 'gas', 'color_scheme': 'green'}
    ]
    
    service = VisualizationService(
        r2_credentials=r2_credentials,
        kv_namespace=kv_namespace,
        sensors=sensors,
        db_client=db_client,
        bucket_name=bucket_name,
        scale_factor=scale_factor
    )
    
    await service.run()

if __name__ == "__main__":
    asyncio.run(start_service())
