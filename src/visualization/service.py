"""Service for generating and managing environmental data visualizations."""

import asyncio
from datetime import datetime, timezone, timedelta
import os
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo
import json
import logging
import io
from pathlib import Path
from tqdm.asyncio import tqdm
import contextlib
from concurrent.futures import ThreadPoolExecutor
import threading

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
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visualization_service.log')
    ]
)
logger = logging.getLogger(__name__)

# A separate debug logger
debug_logger = logging.getLogger(f"{__name__}.debug")
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('visualization_service_debug.log')
debug_handler.setFormatter(
    logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s')
)
debug_logger.addHandler(debug_handler)

class AsyncLogger:
    """
    Asynchronous logger that batches messages to avoid blocking the main event loop.
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.queue = asyncio.Queue()
        self.running = False
        self.task = None
        self._error_count = 0
        self.MAX_ERRORS = 100
    
    async def start(self):
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._process_logs())
    
    async def stop(self):
        if self.running:
            self.running = False
            if self.task:
                try:
                    await asyncio.wait_for(self.queue.join(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
    
    async def _process_logs(self):
        while self.running:
            try:
                try:
                    level, msg = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    if level == logging.DEBUG:
                        debug_logger.debug(msg)
                    else:
                        self.logger.log(level, msg)
                except Exception as e:
                    self._error_count += 1
                    if self._error_count <= self.MAX_ERRORS:
                        print(f"Error in logger: {e}")
                finally:
                    self.queue.task_done()
                    
            except Exception as e:
                self._error_count += 1
                if self._error_count <= self.MAX_ERRORS:
                    print(f"Error in log processor: {e}")
                await asyncio.sleep(0.1)
                
        # Drain any remaining logs before fully stopping
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
        if not self.running:
            # If logger isn't running, log synchronously
            if level == logging.DEBUG:
                debug_logger.debug(msg)
            else:
                self.logger.log(level, msg)
            return
        try:
            self.queue.put_nowait((level, msg))
        except asyncio.QueueFull:
            # If queue is full, fall back to direct logging
            if level == logging.DEBUG:
                debug_logger.debug(msg)
            else:
                self.logger.log(level, msg)

    async def error(self, msg: str, exc_info: Optional[Exception] = None):
        if exc_info:
            import traceback
            tb_str = ''.join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
            msg = f"{msg}\n{tb_str}"
        await self.log(logging.ERROR, msg)

class ProgressManager:
    """
    Manages a tqdm progress bar and optionally suppresses logging
    while a batch of operations is in progress.
    """
    
    def __init__(self, total: int, desc: str, disable_logging: bool = True):
        self.total = total
        self.desc = desc
        self.disable_logging = disable_logging
        self.async_logger = None
    
    @contextlib.asynccontextmanager
    async def progress_bar(self):
        """
        Creates a progress bar context in async, optionally disabling
        normal logging to avoid clutter.
        """
        self.async_logger = AsyncLogger(logger)
        await self.async_logger.start()
        
        original_level = logger.getEffectiveLevel()
        if self.disable_logging:
            logger.setLevel(logging.WARNING)
        
        try:
            with tqdm(total=self.total, desc=self.desc) as pbar:
                yield pbar
        finally:
            if self.disable_logging:
                logger.setLevel(original_level)
            if self.async_logger:
                await self.async_logger.stop()

class VisualizationService:
    """
    Main service for fetching environmental data, generating visualizations,
    and uploading them to R2 (S3-compatible storage), as well as maintaining
    KV records for "latest" versions.
    """
    
    def __init__(self, sensors: List[Dict[str, str]], config: Dict[str, any]):
        self.sensors = sensors
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.db_path = config['db_path']
        self._data_services = {}  # Thread-local DataService instances
        self.async_logger = AsyncLogger(logger)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=config['max_workers'])
        self.generator = VisualizationGenerator(scale_factor=config['scale_factor'])
        self.cloudflare = CloudflareService()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized visualization service:")
        logger.info(f"- Output directory: {config['output_dir']}")
        logger.info(f"- Sensors: {', '.join(s['column'] for s in sensors)}")
        logger.info(f"- Batch size: {config.get('batch_size')}")
        logger.info(f"- Scale factor: {config.get('scale_factor')}")

    def _get_data_service(self) -> DataService:
        """Get or create a thread-local DataService instance."""
        thread_id = threading.get_ident()
        if thread_id not in self._data_services:
            data_service = DataService(self.db_path)
            self._data_services[thread_id] = data_service
        return self._data_services[thread_id]

    async def initialize(self):
        """Sets up async services like the logger and database connections."""
        await self.async_logger.start()
        # Initialize the main thread's DataService
        data_service = self._get_data_service()
        await data_service.initialize()
        logger.info("Initialized async components of visualization service")

    def get_r2_key(self, sensor: str, start_time: datetime, end_time: datetime, interval: str) -> str:
        """
        Returns the path that will be used to store the generated PNG in R2.

        Format:
        sensor/interval/YYYY-MM-DD_daysOffset_minutesOffset.png

        - daily: uses the start day (minutes=1439 => 23:59)
        - hourly/minute: uses the end day
        """
        start_pst = convert_to_pst(start_time)
        end_pst = convert_to_pst(end_time)
        day_diff = (end_pst.date() - start_pst.date()).days
        
        if interval == 'daily':
            minutes_offset = 1439
            return (
                f"{sensor}/{interval}/"
                f"{start_pst.strftime('%Y-%m-%d')}_{day_diff:04d}_{minutes_offset:04d}.png"
            )
        elif interval == 'hourly':
            minutes_offset = end_pst.hour * 60 + 59
        else:  # minute
            minutes_offset = end_pst.hour * 60 + end_pst.minute
        
        return (
            f"{sensor}/{interval}/"
            f"{end_pst.strftime('%Y-%m-%d')}_{day_diff:04d}_{minutes_offset:04d}.png"
        )

    def get_kv_key(self, sensor: str, interval: str) -> str:
        """Returns a unique KV store key for the 'latest' PNG record for a sensor/interval."""
        return f"latest_{sensor}_{interval}"

    async def _save_image(self, image, path: Path) -> str:
        """
        Saves a Pillow image to an in-memory buffer and uploads it to R2.
        """
        buf = io.BytesIO()
        image.save(buf, format='PNG', optimize=True, compress_level=6)
        data_bytes = buf.getvalue()

        def do_upload():
            return self.cloudflare.upload_image(data_bytes, str(path))
        
        r2_url = await asyncio.get_event_loop().run_in_executor(self.thread_pool, do_upload)
        await self.async_logger.log(logging.DEBUG, f"Saved image to R2: {r2_url}")
        return r2_url

    async def _update_kv_records(self, interval: str, sensor_status: Dict):
        """
        After generating images, updates each sensor's KV record if it is newer
        than the existing record (comparing end_time in the filename).
        """
        for s in self.sensors:
            col = s['column']
            if col not in sensor_status or not sensor_status[col].get('image_path'):
                continue
            
            kv_key = self.get_kv_key(col, interval)
            new_path = sensor_status[col]['image_path']
            new_end_time = await self._extract_end_time_from_path(new_path, interval)

            old_val = self.cloudflare.get_kv_record(kv_key)
            if old_val:
                try:
                    outer_data = json.loads(old_val)
                    if isinstance(outer_data, dict) and 'value' in outer_data:
                        poss_val = outer_data['value']
                        old_data = (
                            poss_val if isinstance(poss_val, dict)
                            else json.loads(poss_val)
                        )
                    else:
                        old_data = outer_data if isinstance(outer_data, dict) else {}
                    old_path = old_data.get('path')
                    if old_path:
                        old_end = await self._extract_end_time_from_path(old_path, interval)
                        if old_end and new_end_time and new_end_time < old_end:
                            await self.async_logger.log(
                                logging.DEBUG,
                                f"Skipping KV update for {kv_key}; old record is newer."
                            )
                            continue
                except Exception as e:
                    await self.async_logger.log(logging.WARNING, f"Error parsing old KV for {kv_key}: {e}")
            
            kv_data = {
                'path': new_path,
                'last_processed': datetime.now(timezone.utc).isoformat(),
                'last_success': sensor_status[col].get('last_success'),
                'last_error': sensor_status[col].get('last_error'),
                'status': 'success' if not sensor_status[col].get('last_error') else 'error'
            }
            self.cloudflare.update_kv_record(kv_key, kv_data)
            await self.async_logger.log(logging.INFO, f"Updated KV record: {kv_key} => {new_path}")

    async def process_sensor_update(
        self,
        sensor: Dict[str,str],
        start_time: datetime,
        end_time: datetime,
        sensor_status: Dict[str, dict],
        interval: str,
        existing_data: Optional[List[EnvironmentalData]] = None
    ):
        """
        Generate and upload a single sensor's visualization for the given interval/time range.
        """
        try:
            col = sensor['column']
            data_service = self._get_data_service()
            data = existing_data if existing_data is not None else await data_service.get_sensor_data(
                col,
                start_date=start_time.astimezone(timezone.utc),
                end_date=end_time.astimezone(timezone.utc)
            )
            if not data:
                await self.async_logger.log(
                    logging.WARNING,
                    f"No data for {col} from {start_time} to {end_time}"
                )
                return
            
            start_pst = convert_to_pst(start_time)
            end_pst = convert_to_pst(end_time)
            image = self.generator.generate_visualization(
                data=data,
                column=col,
                color_scheme=sensor["color_scheme"],
                start_time=start_pst,
                end_time=end_pst,
                interval=interval
            )
            
            r2_key = self.get_r2_key(col, start_time, end_time, interval)
            await self._save_image(image, Path(r2_key))
            
            sensor_status[col] = {
                'last_success': end_pst.isoformat(),
                'last_error': None,
                'image_path': r2_key
            }
            
            del image
            if existing_data is None:
                del data

        except Exception as e:
            col = sensor["column"]
            await self.async_logger.log(logging.ERROR, f"Error processing {col}: {e}")
            sensor_status[col]["last_error"] = str(e)

    async def process_interval_updates(
        self,
        start_date: datetime,
        end_date: datetime,
        sensor_status: Dict[str,dict],
        interval: str,
        existing_data: Optional[Dict[str, List[EnvironmentalData]]] = None
    ):
        """
        For 'daily' and 'hourly': chunk the data in steps (1 day or 1 hour) and produce each PNG.

        For 'minute' backfill: produce *day-by-day cumulative* PNGs from the earliest record
        to each day. This ensures a comprehensive history of minute-resolution data.

        Once backfill is complete, `_run_minute_updates` produces a single all-time
        minute chart every hour in normal operation.
        """
        start_pst = convert_to_pst(start_date)
        end_pst = convert_to_pst(end_date)
        await self.async_logger.log(logging.INFO, f"Processing {interval} from {start_pst} to {end_pst}")

        if interval in ("daily","hourly"):
            sensor_data = existing_data or {}
            if not existing_data:
                tasks = []
                for s in self.sensors:
                    col = s["column"]
                    tasks.append(
                        self._get_data_service().get_sensor_data(
                            col,
                            start_date=start_date.astimezone(timezone.utc),
                            end_date=end_date.astimezone(timezone.utc)
                        )
                    )
                results = await asyncio.gather(*tasks)
                sensor_data = {
                    s['column']: r
                    for s,r in zip(self.sensors, results)
                    if r
                }
            
            if not sensor_data:
                await self.async_logger.log(
                    logging.WARNING,
                    f"No data found for {interval} from {start_date} to {end_date}"
                )
                return
            
            # Step size: daily -> 1 day, hourly -> 1 hour
            if interval == 'daily':
                step = timedelta(days=1)
                current = start_pst.replace(hour=0, minute=0)
            else:  # hourly
                step = timedelta(hours=1)
                current = start_pst.replace(minute=0)
            
            total = max(0, int((end_pst - current)/step)) * len(self.sensors)
            pm = ProgressManager(total, f"Generating {interval} visuals")
            async with pm.progress_bar() as pbar:
                while current < end_pst:
                    next_time = min(current + step, end_pst)
                    
                    futs = []
                    for s in self.sensors:
                        col = s["column"]
                        if col not in sensor_data:
                            continue
                        subset = [
                            x for x in sensor_data[col]
                            if current <= convert_to_pst(x.time) <= next_time
                        ]
                        if subset:
                            earliest = subset[0].time
                            latest = subset[-1].time
                            futs.append(
                                self.process_sensor_update(
                                    s,
                                    earliest,
                                    latest,
                                    sensor_status,
                                    interval,
                                    existing_data=subset
                                )
                            )
                        pbar.update(1)
                    
                    if futs:
                        await asyncio.gather(*futs)
                    
                    current = next_time

            await self._update_kv_records(interval, sensor_status)
            await self.async_logger.log(logging.INFO, f"Completed {interval} updates")

        else:
            # minute => day-by-day cumulative backfill
            first_db_record = await self._get_data_service().get_sensor_data(self.sensors[0]["column"], limit=1)
            if not first_db_record:
                await self.async_logger.log(logging.WARNING, "No data in DB for minute interval.")
                return
            
            earliest_db_time = first_db_record[0].time
            earliest_pst = convert_to_pst(earliest_db_time)
            
            # We'll produce a cumulative PNG for each day from earliest -> end_date
            total_days = (end_pst.date() - earliest_pst.date()).days + 1
            total_ops = total_days * len(self.sensors)
            
            pm = ProgressManager(total_ops, "Generating minute cumulative per-day")
            async with pm.progress_bar() as pbar:
                day_cursor = earliest_pst.replace(hour=0, minute=0)
                while day_cursor <= end_pst:
                    # For each day, generate a chart from earliest_db_time -> that day's end
                    day_end = day_cursor.replace(hour=23, minute=59, second=59)
                    if day_end > end_pst:
                        day_end = end_pst
                    
                    tasks = []
                    for s in self.sensors:
                        col = s["column"]
                        tasks.append(
                            self._get_data_service().get_sensor_data(
                                col,
                                start_date=earliest_db_time.astimezone(timezone.utc),
                                end_date=day_end.astimezone(timezone.utc)
                            )
                        )
                    results = await asyncio.gather(*tasks)
                    
                    futs = []
                    for s, sensor_data_ in zip(self.sensors, results):
                        if sensor_data_:
                            futs.append(
                                self.process_sensor_update(
                                    s,
                                    earliest_db_time,
                                    day_end,
                                    sensor_status,
                                    'minute',
                                    existing_data=sensor_data_
                                )
                            )
                        pbar.update(1)
                    if futs:
                        await asyncio.gather(*futs)
                    
                    day_cursor += timedelta(days=1)
            
            await self._update_kv_records('minute', sensor_status)
            await self.async_logger.log(logging.INFO, "Completed minute (cumulative) daily stepping updates")

    async def process_daily_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict[str, dict]):
        """Helper to process daily updates in a standard chunked way."""
        await self.process_interval_updates(start_date, end_date, sensor_status, 'daily')

    async def process_hourly_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict[str, dict]):
        """Helper to process hourly updates in a standard chunked way."""
        await self.process_interval_updates(start_date, end_date, sensor_status, 'hourly')

    async def process_minute_updates(
        self,
        start_date: datetime,
        end_date: datetime,
        sensor_status: Dict[str, dict],
        existing_data: Optional[Dict]=None
    ):
        """
        'minute' backfill: day-by-day cumulative from earliest record.

        Normal runtime (in _run_minute_updates): single all-time minute chart every hour,
        after the hourly interval completes, so that minute charts are also updated hourly.
        """
        await self.process_interval_updates(start_date, end_date, sensor_status, 'minute', existing_data)

    async def generate_visualization(
        self,
        sensor: str,
        start_time: datetime,
        end_time: datetime,
        color_scheme: str = 'base'
    ) -> Optional[Path]:
        """
        Convenience method to produce a single PNG for [start_time, end_time] locally.
        """
        try:
            data = await self._get_data_service().get_sensor_data(
                sensor,
                start_date=start_time.astimezone(timezone.utc),
                end_date=end_time.astimezone(timezone.utc)
            )
            if data:
                start_pst = convert_to_pst(start_time)
                end_pst = convert_to_pst(end_time)
                image = self.generator.generate_visualization(
                    data=data,
                    column=sensor,
                    color_scheme=color_scheme,
                    start_time=start_pst,
                    end_time=end_pst
                )
                out_name = f"{sensor}_{start_pst:%Y%m%d_%H%M}_{end_pst:%Y%m%d_%H%M}.png"
                out_path = self.output_dir / out_name
                image.save(out_path)
                return out_path
            else:
                logger.warning(f"No data found for {sensor} from {start_time} to {end_time}")
        except Exception as e:
            logger.error(f"Error generating single visualization for {sensor}: {e}")
            raise
        return None

    async def _find_latest_image(self, sensor: str, interval: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Looks up the last-known KV record, parses out the path, and tries
        to compute a start_time and end_time from its filename.
        """
        try:
            kv_key = self.get_kv_key(sensor, interval)
            val = self.cloudflare.get_kv_record(kv_key)
            if not val:
                return None
            
            outer_data = json.loads(val)
            if isinstance(outer_data, dict) and 'value' in outer_data:
                poss_val = outer_data['value']
                kv_data = poss_val if isinstance(poss_val, dict) else json.loads(poss_val)
            else:
                kv_data = outer_data if isinstance(outer_data, dict) else {}
            
            if kv_data.get('status') == 'error':
                await self.async_logger.log(
                    logging.WARNING,
                    f"{sensor} {interval} is in error state, ignoring."
                )
                return None
            
            path = kv_data.get('path')
            if not path:
                return None
            end_time = await self._extract_end_time_from_path(path, interval)
            if not end_time:
                return None
            
            if interval == 'daily':
                start_time = end_time.replace(hour=0, minute=0)
            elif interval == 'hourly':
                start_time = end_time.replace(minute=0)
            else:
                start_time = end_time
            return (start_time, end_time)
        except Exception as e:
            await self.async_logger.log(logging.ERROR, f"Error reading KV for {sensor}/{interval}: {e}")
            return None

    async def _extract_end_time_from_path(self, path: str, interval: str) -> Optional[datetime]:
        """
        For a given file path sensor/interval/YYYY-MM-DD_####_####.png, parse out the final minutes_offset
        to reconstruct the end_time. For daily, 1439 => 23:59; for hourly, e.g. 719 => 11:59; etc.
        """
        try:
            parts = path.split('/')
            if len(parts) != 3:
                return None
            
            filename = parts[2].replace('.png','')
            date_str, _ignored_days, mins_str = filename.split('_')
            
            base_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=ZoneInfo('America/Los_Angeles'))
            val = int(mins_str)
            
            if interval == 'daily':
                # 1439 => 23:59
                return base_date.replace(hour=23, minute=59)
            elif interval == 'hourly':
                # e.g. 719 => 11:59
                hour_ = val // 60
                return base_date.replace(hour=hour_, minute=59)
            else:
                # minute => e.g. 611 => 10:11
                hour_ = val // 60
                minute_ = val % 60
                return base_date.replace(hour=hour_, minute=minute_)
        except Exception:
            return None

    async def backfill(self):
        """
        One-time catch-up process to fill in historical images for daily/hourly/minute.
        - daily/hourly chunk by day/hour
        - minute is stepped day-by-day with a cumulative data set

        After backfill, runtime tasks generate fresh images at intervals.
        """
        await self.async_logger.log(logging.INFO, "Starting backfill process")
        try:
            now_utc = datetime.now(timezone.utc)
            now_pst = convert_to_pst(now_utc)
            midnight = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
            
            first_data = await self._get_data_service().get_sensor_data(self.sensors[0]['column'], limit=1)
            if not first_data:
                await self.async_logger.log(logging.WARNING, "No data in DB.")
                return
            
            first_point = first_data[0].time
            first_point_pst = convert_to_pst(first_point)
            
            await self.async_logger.log(logging.INFO, f"First data point: {first_point_pst.isoformat()}")
            await self.async_logger.log(logging.INFO, f"Current time: {now_pst.isoformat()}")
            
            # Read the latest known images from KV to see where we left off
            latest_map = {}
            for s in self.sensors:
                c = s['column']
                latest_map[c] = {}
                for iv in ['daily','hourly','minute']:
                    pair = await self._find_latest_image(c, iv)
                    latest_map[c][iv] = pair
            
            # These are the earliest possible start times
            start_dates = {
                'daily': first_point_pst.replace(hour=0, minute=0, second=0, microsecond=0),
                'hourly': first_point_pst.replace(minute=0, second=0, microsecond=0),
                'minute': first_point_pst
            }
            # If we have existing KV images, push the start dates forward
            for s in self.sensors:
                c = s['column']
                for iv in ['daily','hourly','minute']:
                    pair = latest_map[c][iv]
                    if pair:
                        _, end_t = pair
                        if iv == 'daily':
                            next_start = end_t.replace(hour=0, minute=0) + timedelta(days=1)
                        elif iv == 'hourly':
                            next_start = end_t.replace(minute=0) + timedelta(hours=1)
                        else:
                            next_start = end_t + timedelta(minutes=1)
                        start_dates[iv] = max(start_dates[iv], next_start)
            
            sensor_status = {s['column']: {'last_success': None,'last_error':None} for s in self.sensors}
            
            # Daily chunk backfill up to most recent midnight
            if start_dates['daily'] < midnight:
                await self.async_logger.log(logging.INFO, "=== Starting Daily Processing ===")
                await self.process_daily_updates(start_dates['daily'], midnight, sensor_status)
                await self.async_logger.log(logging.INFO, "=== Daily done ===")
            
            # Hourly chunk backfill up to current time
            if start_dates['hourly'] < now_pst:
                await self.async_logger.log(logging.INFO, "=== Starting Hourly Processing ===")
                await self.process_hourly_updates(start_dates['hourly'], now_pst, sensor_status)
                await self.async_logger.log(logging.INFO, "=== Hourly done ===")
            
            # Minute day-by-day backfill
            if start_dates['minute'] < now_pst:
                await self.async_logger.log(logging.INFO, "=== Starting Minute Processing ===")
                await self.process_minute_updates(
                    start_date=start_dates['minute'],
                    end_date=now_pst,
                    sensor_status=sensor_status
                )
                await self.async_logger.log(logging.INFO, "=== Minute done ===")
            
            await self.async_logger.log(logging.INFO, "=== Backfill Complete ===")

        except Exception as e:
            await self.async_logger.error("Error during backfill", exc_info=e)
            raise

    async def run(self):
        """
        Called after the initial backfill is finished. Starts continuous loops for
        daily/hourly/minute intervals, each performing regular updates.

        - Daily runs once per midnight
        - Hourly runs once per hour
        - Minute also runs once per hour (slightly offset after hourly),
          generating a single all-time minute chart.
        """
        await self.async_logger.start()
        await self.async_logger.log(logging.INFO, "Starting visualization service...")
        try:
            await self.backfill()
            await self.async_logger.log(logging.INFO, "Starting continuous updates")
            
            daily_task  = asyncio.create_task(self._run_daily_updates())
            hourly_task = asyncio.create_task(self._run_hourly_updates())
            minute_task = asyncio.create_task(self._run_minute_updates())
            
            await asyncio.gather(daily_task, hourly_task, minute_task)
        finally:
            await self.async_logger.stop()

    async def _get_last_processed_time(self, interval: str) -> Dict[str, Optional[datetime]]:
        """
        Check the KV store for each sensor's 'latest_{interval}' record, parse
        'last_processed' or the path's end_time, and return a dict of sensor -> latest time.
        If a sensor has no record or is in error state, its time will be None.
        """
        sensor_times = {}
        try:
            for s in self.sensors:
                c = s['column']
                kv_key = self.get_kv_key(c, interval)
                kv_val = self.cloudflare.get_kv_record(kv_key)
                sensor_times[c] = None
                
                if kv_val:
                    try:
                        outer_data = json.loads(kv_val)
                        if isinstance(outer_data, dict) and 'value' in outer_data:
                            outer_data = outer_data['value']
                        
                        kv_data = (
                            outer_data if isinstance(outer_data, dict)
                            else json.loads(outer_data)
                        )
                        
                        if kv_data.get('status') == 'error':
                            await self.async_logger.log(
                                logging.WARNING,
                                f"{c} {interval} is in error state."
                            )
                            continue
                            
                        if kv_data.get('last_processed'):
                            dt = datetime.fromisoformat(kv_data['last_processed'])
                            sensor_times[c] = convert_to_pst(dt)
                        else:
                            # If there's no last_processed, use the file path
                            p = kv_data.get('path')
                            if p:
                                e_time = await self._extract_end_time_from_path(p, interval)
                                if e_time:
                                    sensor_times[c] = e_time
                    except Exception as e:
                        await self.async_logger.log(
                            logging.WARNING,
                            f"Error parsing KV for {c}/{interval}: {e}"
                        )
                else:
                    await self.async_logger.log(
                        logging.INFO,
                        f"No KV record found for {c} in {interval} interval"
                    )
            
        except Exception as e:
            await self.async_logger.log(
                logging.ERROR,
                f"Error in _get_last_processed_time({interval}): {e}"
            )
        return sensor_times

    async def _run_daily_updates(self):
        """
        Runs every midnight PST. We compare what the last 'processed time'
        was in the KV store for each sensor, then chunk from there until the current midnight,
        generating daily images as needed.
        """
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                midn = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
                
                sensor_times = await self._get_last_processed_time('daily')
                next_midn = midn + timedelta(days=1)
                until = min(next_midn, now_pst)
                
                # Process each sensor that needs updating
                for s in self.sensors:
                    col = s['column']
                    last_proc = sensor_times.get(col)
                    
                    if not last_proc:
                        start_time = midn
                    else:
                        start_time = last_proc.replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    if start_time < until:
                        st = {col: {'last_success':None,'last_error':None}}
                        await self.process_daily_updates(start_time, until, st)
                        await self._update_kv_records('daily', st)
                
                # Sleep until the next midnight
                wait_secs = max(0, (next_midn - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO, f"Daily loop sleeps {int(wait_secs/3600)} h")
                await asyncio.sleep(wait_secs)
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in daily loop: {e}")
                await asyncio.sleep(60)

    async def _run_hourly_updates(self):
        """
        Runs every hour on the hour PST. We see what's processed for each sensor,
        produce new hourly images if needed, then sleep.
        """
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                hr_top = now_pst.replace(minute=0, second=0, microsecond=0)
                
                sensor_times = await self._get_last_processed_time('hourly')
                next_hour = hr_top + timedelta(hours=1)
                until = min(next_hour, now_pst)
                
                # Process each sensor that needs updating
                for s in self.sensors:
                    col = s['column']
                    last_proc = sensor_times.get(col)
                    
                    if not last_proc:
                        start_time = hr_top
                    else:
                        start_time = last_proc.replace(minute=0, second=0, microsecond=0)
                    
                    if start_time < until:
                        st = {col: {'last_success':None,'last_error':None}}
                        await self.process_hourly_updates(start_time, until, st)
                        await self._update_kv_records('hourly', st)
                
                # Sleep until the next hour
                wait_secs = max(0, (next_hour - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO, f"Hourly loop sleeps {int(wait_secs/60)} min")
                await asyncio.sleep(wait_secs)
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in hourly loop: {e}")
                await asyncio.sleep(60)

    async def _run_minute_updates(self):
        """
        Runs every hour (offset by +5 minutes from the top of the hour),
        generating a single all-time minute chart from earliest record to 'now'
        for each sensor that needs updating.
        """
        while True:
            try:
                now = datetime.now(timezone.utc)
                # Round down to hour
                now_pst = convert_to_pst(now)
                hr_top = now_pst.replace(minute=0, second=0, microsecond=0)
                
                # We offset minute updates by +5 minutes so it runs after hourly is done
                run_time = hr_top + timedelta(minutes=5)
                if now_pst < run_time:
                    wait_secs = (run_time - now_pst).total_seconds()
                    await self.async_logger.log(logging.INFO, f"Minute updates waiting ~{int(wait_secs)}s after hourly.")
                    await asyncio.sleep(wait_secs)

                # Re-check time after waiting
                now_2 = datetime.now(timezone.utc)
                now_2_pst = convert_to_pst(now_2)
                
                sensor_times = await self._get_last_processed_time('minute')
                data_service = self._get_data_service()
                
                # Process each sensor that needs updating
                for s in self.sensors:
                    col = s['column']
                    last_proc = sensor_times.get(col)
                    
                    # Get earliest record for this sensor
                    first_db_record = await data_service.get_sensor_data(col, limit=1)
                    if not first_db_record:
                        continue
                    
                    earliest_db_time = first_db_record[0].time
                    
                    # If we have a last processed time and it's after the earliest record,
                    # and it's within the last hour, skip this sensor
                    if (last_proc and 
                        last_proc > convert_to_pst(earliest_db_time) and
                        (now_2_pst - last_proc).total_seconds() < 3600):
                        continue
                    
                    st = {col: {'last_success':None,'last_error':None}}
                    await self.process_minute_updates(earliest_db_time, now_2_pst, st)
                    await self._update_kv_records('minute', st)
                
                # Sleep for 1 hour until next iteration
                next_cycle = hr_top + timedelta(hours=1)
                wait_secs = max(0, (next_cycle - now_2_pst).total_seconds())
                await self.async_logger.log(logging.INFO, f"Minute updates sleep {int(wait_secs/60)} min until next hour.")
                await asyncio.sleep(wait_secs)
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in minute loop: {e}")
                await asyncio.sleep(60)

async def start_service(
    db_path: str = "data/environmental.db",
    output_dir: str = "data/visualizations",
    scale_factor: int = 4
):
    logger.info("Starting visualization service with config:")
    logger.info(f"DB path: {db_path}")
    logger.info(f"Output dir: {output_dir}")
    
    logger.info("Syncing data from Supabase...")
    await migrate_data(SUPABASE_URL, SUPABASE_KEY, db_path=db_path)
    logger.info("Data sync complete")

    sensors = [
        {'column': 'temperature','color_scheme': 'redblue'},
        {'column': 'humidity',   'color_scheme': 'cyan'},
        {'column': 'pressure',   'color_scheme': 'green'},
        {'column': 'light',      'color_scheme': 'base'},
        {'column': 'uv',         'color_scheme': 'purple'},
        {'column': 'gas',        'color_scheme': 'yellow'}
    ]

    config = {
        'db_path': db_path,
        'output_dir': output_dir,
        'scale_factor': scale_factor,
        'max_workers': None,
        'batch_size': 30,
        # daily/hourly => chunked
        # minute => day-by-day backfill, then every hour offset by +5 min
        'minute_chunk_interval': 'hourly'
    }

    svc = VisualizationService(sensors, config)
    await svc.initialize()
    await svc.run()

if __name__ == "__main__":
    asyncio.run(start_service())
