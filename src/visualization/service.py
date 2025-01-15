"""Service for generating and managing environmental data visualizations."""

import asyncio
from datetime import datetime, timezone, timedelta, time
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
    """Asynchronous logger that batches messages."""
    
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

                # Attempt logging
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
            if level == logging.DEBUG:
                debug_logger.debug(msg)
            else:
                self.logger.log(level, msg)
            return
        try:
            self.queue.put_nowait((level, msg))
        except asyncio.QueueFull:
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
    """Manages progress bars and optionally suppresses logging."""
    
    def __init__(self, total: int, desc: str, disable_logging: bool = True):
        self.total = total
        self.desc = desc
        self.disable_logging = disable_logging
        self.async_logger = None
    
    @contextlib.asynccontextmanager
    async def progress_bar(self):
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
    """Service for generating and managing environmental data visualizations."""
    
    def __init__(self, sensors: List[Dict[str, str]], config: Dict[str, any]):
        self.sensors = sensors
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.data_service = DataService(config['db_path'])
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

    async def initialize(self):
        """Initialize async services."""
        await self.async_logger.start()
        await self.data_service.initialize()
        logger.info("Initialized async components of visualization service")

    def get_r2_key(self, sensor: str, start_time: datetime, end_time: datetime, interval: str) -> str:
        """
        Format: sensor/interval/YYYY-MM-DD_daysOffset_minutesOffset.png
        We'll keep this original style, ignoring day offsets in _extract_end_time_from_path.
        """
        start_pst = convert_to_pst(start_time)
        end_pst = convert_to_pst(end_time)
        day_diff = (end_pst.date() - start_pst.date()).days
        
        if interval == 'daily':
            minutes_offset = 1439
        elif interval == 'hourly':
            minutes_offset = end_pst.hour * 60 + 59
        else:  # minute
            minutes_offset = end_pst.hour * 60 + end_pst.minute
        
        return (
            f"{sensor}/{interval}/"
            f"{end_pst.strftime('%Y-%m-%d')}_{day_diff:04d}_{minutes_offset:04d}.png"
        )

    def get_kv_key(self, sensor: str, interval: str) -> str:
        return f"latest_{sensor}_{interval}"

    async def _save_image(self, image, path: Path) -> str:
        """
        Saves a Pillow image to R2 using cloudflare_service.py
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
        After generating images, update each sensor's KV record if it's newer than existing.
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
                            await self.async_logger.log(logging.DEBUG,
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
        """Generate a single sensor's visualization for the given interval."""
        try:
            col = sensor['column']
            data = existing_data if existing_data is not None else await self.data_service.get_sensor_data(
                col,
                start_date=start_time.astimezone(timezone.utc),
                end_date=end_time.astimezone(timezone.utc)
            )
            if not data:
                await self.async_logger.log(logging.WARNING,
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
        For daily/hourly, we do normal chunking.
        For minute, we do a single big "cumulative" fetch from earliest to end_date,
        then produce exactly one image that includes everything.
        """
        start_pst = convert_to_pst(start_date)
        end_pst = convert_to_pst(end_date)
        await self.async_logger.log(logging.INFO, f"Processing {interval} from {start_pst} to {end_pst}")

        if interval in ("daily","hourly"):
            # old chunk logic
            sensor_data = existing_data or {}
            if not existing_data:
                tasks = []
                for s in self.sensors:
                    col = s["column"]
                    tasks.append(
                        self.data_service.get_sensor_data(
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
                await self.async_logger.log(logging.WARNING,
                    f"No data found for {interval} from {start_date} to {end_date}"
                )
                return
            
            # chunk for daily or hourly
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
                        # filter for [current, next_time]
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
            await self.async_logger.log(logging.INFO,
                f"Completed {interval} updates"
            )

        else:
            # minute => single "all time" approach
            # 1) get earliest data from DB for each sensor
            tasks = []
            for s in self.sensors:
                earliest_data = await self.data_service.get_sensor_data(
                    s["column"], limit=1
                )
                if earliest_data:
                    earliest_time = earliest_data[0].time
                    # if earliest_time is after end_date => skip
                    if earliest_time <= end_date:
                        tasks.append(
                            self.data_service.get_sensor_data(
                                s["column"],
                                start_date=earliest_time,
                                end_date=end_date.astimezone(timezone.utc)
                            )
                        )
                    else:
                        tasks.append(None)
                else:
                    tasks.append(None)
            
            results = await asyncio.gather(*tasks)
            
            # match them up
            sensor_data = {}
            i=0
            for s in self.sensors:
                col = s["column"]
                # results[i] might be None
                if i < len(results) and results[i]:
                    sensor_data[col] = results[i]
                i+=1
            
            if not sensor_data:
                await self.async_logger.log(logging.WARNING,
                    f"No data found for minute interval up to {end_date}"
                )
                return
            
            # produce exactly ONE big minute image for each sensor, from earliest->end
            # so each sensor has earliest_data[0].time => end_date
            # We'll process them as a "batch" so we get a one-liner progress
            total = len(self.sensors)
            pm = ProgressManager(total, "Generating minute cumulative visuals")
            async with pm.progress_bar() as pbar:
                futs = []
                for s in self.sensors:
                    col = s["column"]
                    if col not in sensor_data:
                        pbar.update(1)
                        continue
                    all_data = sensor_data[col]
                    # earliest & latest from the entire dataset
                    earliest = all_data[0].time
                    latest   = all_data[-1].time
                    futs.append(
                        self.process_sensor_update(
                            s,
                            earliest,
                            latest,
                            sensor_status,
                            'minute',
                            existing_data=all_data
                        )
                    )
                    pbar.update(1)
                
                if futs:
                    await asyncio.gather(*futs)

            await self._update_kv_records('minute', sensor_status)
            await self.async_logger.log(logging.INFO,
                "Completed minute (cumulative) updates"
            )

    async def process_daily_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict[str, dict]):
        await self.process_interval_updates(start_date, end_date, sensor_status, 'daily')

    async def process_hourly_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict[str, dict]):
        await self.process_interval_updates(start_date, end_date, sensor_status, 'hourly')

    async def process_minute_updates(self, start_date: datetime, end_date: datetime, sensor_status: Dict[str, dict], existing_data: Optional[Dict]=None):
        await self.process_interval_updates(start_date, end_date, sensor_status, 'minute', existing_data)

    async def generate_visualization(
        self,
        sensor: str,
        start_time: datetime,
        end_time: datetime,
        color_scheme: str = 'base'
    ) -> Optional[Path]:
        """
        A quick method to produce a single PNG for [start_time, end_time].
        """
        try:
            data = await self.data_service.get_sensor_data(
                sensor,
                start_date=start_time.astimezone(timezone.utc),
                end_date=end_time.astimezone(timezone.utc)
            )
            if data:
                start_pst = convert_to_pst(start_time)
                end_pst   = convert_to_pst(end_time)
                image = self.generator.generate_visualization(
                    data=data,
                    column=sensor,
                    color_scheme=color_scheme,
                    start_time=start_pst,
                    end_time=end_pst
                )
                # save locally
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
        Tries to parse the last known record from KV, returning (start_time, end_time).
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
                await self.async_logger.log(logging.WARNING,
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
        Ignores days_offset; only uses final minutes_offset for daily/hourly/minute.
        e.g. "2025-01-15_0000_1439.png" => base_date=2025-01-15, minutes=1439 => 23:59
        """
        try:
            parts = path.split('/')
            if len(parts) != 3:
                return None
            
            filename = parts[2].replace('.png','')
            date_str, ignored_days, mins_str = filename.split('_')
            
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
                # minute => 611 => 10:11
                hour_ = val // 60
                minute_ = val % 60
                return base_date.replace(hour=hour_, minute=minute_)
        except Exception:
            return None

    async def backfill(self):
        """One-time catch up on daily/hourly/minute intervals."""
        await self.async_logger.log(logging.INFO, "Starting backfill process")
        try:
            now_utc = datetime.now(timezone.utc)
            now_pst = convert_to_pst(now_utc)
            midnight = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
            
            first_data = await self.data_service.get_sensor_data(self.sensors[0]['column'], limit=1)
            if not first_data:
                await self.async_logger.log(logging.WARNING, "No data in DB.")
                return
            
            first_point = first_data[0].time
            first_point_pst = convert_to_pst(first_point)
            
            await self.async_logger.log(logging.INFO, f"First data point: {first_point_pst.isoformat()}")
            await self.async_logger.log(logging.INFO, f"Current time: {now_pst.isoformat()}")
            
            # read existing images
            latest_map = {}
            for s in self.sensors:
                c = s['column']
                latest_map[c] = {}
                for iv in ['daily','hourly','minute']:
                    pair = await self._find_latest_image(c, iv)
                    latest_map[c][iv] = pair
            
            # default starts
            start_dates = {
                'daily': first_point_pst.replace(hour=0, minute=0, second=0, microsecond=0),
                'hourly': first_point_pst.replace(minute=0, second=0, microsecond=0),
                'minute': first_point_pst
            }
            # incorporate last known images
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
            
            # do it
            sensor_status = {s['column']: {'last_success': None,'last_error':None} for s in self.sensors}
            
            if start_dates['daily'] < midnight:
                await self.async_logger.log(logging.INFO, "=== Starting Daily Processing ===")
                await self.process_daily_updates(start_dates['daily'], midnight, sensor_status)
                await self.async_logger.log(logging.INFO, "=== Daily done ===")
            
            if start_dates['hourly'] < now_pst:
                await self.async_logger.log(logging.INFO, "=== Starting Hourly Processing ===")
                await self.process_hourly_updates(start_dates['hourly'], now_pst, sensor_status)
                await self.async_logger.log(logging.INFO, "=== Hourly done ===")
            
            if start_dates['minute'] < now_pst:
                await self.async_logger.log(logging.INFO, "=== Starting Minute Processing ===")
                # minute fetch
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
        Once backfill is done, start daily/hourly/minute loops.
        """
        await self.async_logger.start()
        await self.async_logger.log(logging.INFO, "Starting visualization service...")
        try:
            await self.backfill()
            await self.async_logger.log(logging.INFO, "Starting continuous updates")
            
            daily_task = asyncio.create_task(self._run_daily_updates())
            hourly_task= asyncio.create_task(self._run_hourly_updates())
            minute_task= asyncio.create_task(self._run_minute_updates())
            
            await asyncio.gather(daily_task, hourly_task, minute_task)
        finally:
            await self.async_logger.stop()

    async def _get_last_processed_time(self, interval: str) -> Optional[datetime]:
        """
        Read KV for each sensor's latest_{interval}, parse 'last_processed' or path. Return earliest among them.
        """
        earliest_time = None
        missing = []
        try:
            for s in self.sensors:
                c = s['column']
                kv_key = self.get_kv_key(c, interval)
                kv_val = self.cloudflare.get_kv_record(kv_key)
                if kv_val:
                    try:
                        outer_data = json.loads(kv_val)
                        if isinstance(outer_data, dict) and 'value' in outer_data:
                            outer_data = outer_data['value']
                        
                        kv_data = (
                            outer_data if isinstance(outer_data, dict)
                            else json.loads(outer_data)
                        )
                        if kv_data.get('last_processed'):
                            dt = datetime.fromisoformat(kv_data['last_processed'])
                            dt_pst = convert_to_pst(dt)
                            if earliest_time is None or dt_pst < earliest_time:
                                earliest_time = dt_pst
                        else:
                            # parse path
                            p = kv_data.get('path')
                            if p:
                                e_time = await self._extract_end_time_from_path(p, interval)
                                if e_time and (earliest_time is None or e_time < earliest_time):
                                    earliest_time = e_time
                    except Exception:
                        pass
                else:
                    missing.append(c)
            
            if missing:
                await self.async_logger.log(logging.INFO, 
                    f"No KV records found for sensors {', '.join(missing)} in {interval} interval"
                )
                return None
        except Exception as e:
            await self.async_logger.log(logging.ERROR, f"Error in _get_last_processed_time({interval}): {e}")
            return None
        return earliest_time

    async def _run_daily_updates(self):
        """Runs daily updates each midnight minus a small buffer."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                midn = now_pst.replace(hour=0, minute=0, second=0, microsecond=0)
                
                last_proc = await self._get_last_processed_time('daily')
                if not last_proc:
                    start_time = midn
                else:
                    start_time = last_proc.replace(hour=0, minute=0, second=0, microsecond=0)
                
                next_midn = midn + timedelta(days=1)
                until = min(next_midn, now_pst - timedelta(minutes=5))
                
                if start_time < until:
                    st = {s['column']: {'last_success':None,'last_error':None} for s in self.sensors}
                    await self.process_daily_updates(start_time, until, st)
                    await self._update_kv_records('daily', st)
                
                wait_secs = max(0,(next_midn - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO,f"Daily loop sleeps {int(wait_secs/3600)} h")
                await asyncio.sleep(wait_secs)
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in daily loop: {e}")
                await asyncio.sleep(60)

    async def _run_hourly_updates(self):
        """Runs hourly updates every hour on the hour, minus 5 minutes buffer."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                hr_top = now_pst.replace(minute=0, second=0, microsecond=0)
                
                last_proc = await self._get_last_processed_time('hourly')
                if not last_proc:
                    start_time = hr_top
                else:
                    start_time = last_proc.replace(minute=0, second=0, microsecond=0)
                
                next_hour = hr_top + timedelta(hours=1)
                until = min(next_hour, now_pst - timedelta(minutes=5))
                
                if start_time < until:
                    st = {s['column']: {'last_success':None,'last_error':None} for s in self.sensors}
                    await self.process_hourly_updates(start_time, until, st)
                    await self._update_kv_records('hourly', st)
                
                wait_secs = max(0,(next_hour - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO,f"Hourly loop sleeps {int(wait_secs/60)} min")
                await asyncio.sleep(wait_secs)
            except Exception as e:
                await self.async_logger.log(logging.ERROR, f"Error in hourly loop: {e}")
                await asyncio.sleep(60)

    async def _run_minute_updates(self):
        """
        For minute intervals, run every 5 minutes. We fetch from last known or 5 minutes prior,
        then call `process_minute_updates` which does a SINGLE big all-time fetch and single image 
        if you keep the code above. Or, if you prefer chunking, you'd revert it.
        """
        UPDATE_INTERVAL = timedelta(minutes=5)
        while True:
            try:
                now = datetime.now(timezone.utc)
                now_pst = convert_to_pst(now)
                block_5 = (now_pst.minute // 5)*5
                cur_time= now_pst.replace(minute=block_5, second=0, microsecond=0)
                
                last_proc = await self._get_last_processed_time('minute')
                if not last_proc:
                    start_time = cur_time - UPDATE_INTERVAL
                else:
                    st_5 = (last_proc.minute // 5)*5
                    start_time = last_proc.replace(minute=st_5, second=0, microsecond=0)
                
                nxt = cur_time + UPDATE_INTERVAL
                until = min(nxt, now_pst - timedelta(seconds=30))
                
                if start_time < until:
                    st = {s['column']: {'last_success':None,'last_error':None} for s in self.sensors}
                    await self.process_minute_updates(start_time, until, st)
                    await self._update_kv_records('minute', st)
                
                wait_secs= max(0,(nxt - now_pst).total_seconds())
                await self.async_logger.log(logging.INFO, f"Minute updates sleep {int(wait_secs)}s")
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
        # minute => single cumulative
        'minute_chunk_interval': 'hourly'
    }

    svc = VisualizationService(sensors, config)
    await svc.initialize()
    await svc.run()

if __name__ == "__main__":
    asyncio.run(start_service())
