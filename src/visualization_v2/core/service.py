"""Main visualization service for environmental sensor data."""

from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta, timezone
import asyncio
import structlog
from zoneinfo import ZoneInfo
from pathlib import Path

from models.config import ServiceConfig
from models.sensor import SensorType, Sensor
from models.visualization import (
    VisualizationRequest,
    VisualizationResult,
    BatchVisualizationRequest,
    Interval
)
from data import SensorDataRepository, DataCache
from generators import HeatmapGenerator
from .processor import DataProcessor
from .timezone_utils import (
    convert_request_times_for_query,
    ensure_pst,
    PST_TZ,
    UTC_TZ,
    get_pst_day_boundaries,
    pst_to_utc
)
from core.progress import BackfillProgressTracker
from upload.manager import UploadManager

logger = structlog.get_logger()

# Timezone constants
PST_TZ = ZoneInfo("America/Los_Angeles")
UTC_TZ = ZoneInfo("UTC")


class VisualizationService:
    """Main service for generating environmental data visualizations."""
    
    def __init__(self, config: ServiceConfig):
        """Initialize the service."""
        self.config = config
        self.repository = SensorDataRepository(config.database)
        self.cache = DataCache(
            max_size=config.processing.cache_size,
            ttl_seconds=3600  # 1 hour cache TTL
        )
        self.generator = HeatmapGenerator(config.output)
        self.processor = DataProcessor(
            repository=self.repository,
            cache=self.cache,
            config=config.processing
        )
        self.progress_tracker = BackfillProgressTracker(
            config.progress_tracking,
            config.output
        )
        self.upload_manager = UploadManager(config.upload)
        
        # Service state
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info(
            "Visualization service initialized",
            sensors=[s.type.value for s in config.sensors],
            output_dir=str(config.output.base_dir),
            db_path=str(config.database.path),
            progress_log=str(config.progress_tracking.log_file),
            upload_enabled=config.upload.enabled
        )
    
    async def start(self) -> None:
        """Start the service."""
        if self._running:
            logger.warning("Service is already running")
            return
            
        self._running = True
        
        try:
            # Initialize components
            await self.repository.initialize()
            await self.progress_tracker.initialize()
            
            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._data_sync_task()),
                asyncio.create_task(self._cache_cleanup_task()),
                asyncio.create_task(self._daily_generation_task()),
                asyncio.create_task(self._hourly_generation_task()),
                asyncio.create_task(self._cumulative_generation_task()),
            ]
            
            logger.info("Visualization service started")
            
        except Exception as e:
            self._running = False
            logger.error("Failed to start service", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the service."""
        if not self._running:
            return
            
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close connections
        await self.repository.close()
        
        logger.info("Visualization service stopped")
    
    async def generate_single(
        self,
        sensor_type: SensorType,
        start_time: datetime,
        end_time: datetime,
        interval: Interval,
        color_scheme: Optional[str] = None
    ) -> VisualizationResult:
        """Generate a single visualization."""
        # Find sensor config
        sensor = self._get_sensor_config(sensor_type)
        if not sensor:
            raise ValueError(f"Sensor {sensor_type.value} not configured")

        # Convert timezone using centralized utility
        start_pst, end_pst, start_utc, end_utc = convert_request_times_for_query(
            start_time, end_time
        )

        # For daily visualizations, always expand to full PST day boundaries
        # This ensures we get data for the complete 24-hour period (1440 minutes)
        if interval == Interval.DAILY:
            # Use the date from the start time to determine which day
            pst_date = start_pst.date()
            full_day_start_pst, full_day_end_pst = get_pst_day_boundaries(pst_date)
            
            # Convert full day boundaries to UTC for database query
            full_day_start_utc = pst_to_utc(full_day_start_pst) 
            full_day_end_utc = pst_to_utc(full_day_end_pst)
            
            logger.debug(
                "Expanded daily visualization to full PST day",
                requested_start_pst=start_pst.isoformat(),
                requested_end_pst=end_pst.isoformat(),
                full_day_start_pst=full_day_start_pst.isoformat(),
                full_day_end_pst=full_day_end_pst.isoformat(),
                full_day_start_utc=full_day_start_utc.isoformat(),
                full_day_end_utc=full_day_end_utc.isoformat()
            )
            
            # Use full day boundaries for both visualization and data query
            start_pst, end_pst = full_day_start_pst, full_day_end_pst
            start_utc, end_utc = full_day_start_utc, full_day_end_utc

        # Create request with PST times for visualization
        request = VisualizationRequest(
            sensor_type=sensor_type,
            start_time=start_pst,
            end_time=end_pst,
            interval=interval,
            color_scheme=color_scheme or sensor.color_scheme
        )

        # Get data using UTC times for database query
        data = await self.processor.get_sensor_data(
            sensor_type, start_utc, end_utc
        )

        logger.debug(
            "Retrieved data for visualization",
            sensor_type=sensor_type.value,
            data_points=len(data),
            interval=interval.value,
            time_range_hours=(end_utc - start_utc).total_seconds() / 3600
        )

        # Generate visualization
        result = await self.generator.generate(request, data)
        
        # Upload to external services if enabled and generation was successful
        if self.upload_manager.is_enabled() and result.success:
            logger.debug(
                "Starting upload process for generated visualization",
                sensor_type=sensor_type.value,
                interval=interval.value,
                output_path=str(result.output_path)
            )
            
            upload_result = await self.upload_manager.upload_visualization(result)
            
            if upload_result.success:
                logger.info(
                    "Visualization uploaded successfully",
                    sensor_type=sensor_type.value,
                    interval=interval.value,
                    cid=upload_result.cid,
                    kv_key=upload_result.kv_key,
                    local_deleted=upload_result.local_file_deleted
                )
            else:
                logger.error(
                    "Failed to upload visualization",
                    sensor_type=sensor_type.value,
                    interval=interval.value,
                    error=upload_result.error_message
                )

        return result
    
    async def generate_batch(
        self,
        request: BatchVisualizationRequest
    ) -> List[VisualizationResult]:
        """Generate multiple visualizations."""
        results = []
        
        # Temporarily update output directory for batch processing
        original_output_dir = self.config.output.base_dir
        self.config.output.base_dir = request.output_dir
        
        try:
            # Create individual requests
            requests = []
            for sensor_type in request.sensor_types:
                sensor = self._get_sensor_config(sensor_type)
                if not sensor:
                    logger.warning(f"Sensor {sensor_type.value} not configured, skipping")
                    continue
                    
                for interval in request.intervals:
                    viz_request = VisualizationRequest(
                        sensor_type=sensor_type,
                        start_time=request.start_time,
                        end_time=request.end_time,
                        interval=interval,
                        color_scheme=sensor.color_scheme,
                        # Don't set output_path for batch - let generator create proper filename
                    )
                    requests.append(viz_request)
            
            # Process requests in parallel batches
            batch_size = self.config.processing.batch_size
            for i in range(0, len(requests), batch_size):
                batch = requests[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[self._process_single_request(req) for req in batch],
                    return_exceptions=True
                )
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error("Batch processing error", error=str(result))
                    else:
                        results.append(result)
            
            # Upload all successful results if upload is enabled
            if self.upload_manager.is_enabled():
                successful_results = [r for r in results if r.success and r.output_path]
                
                if successful_results:
                    logger.info(
                        "Starting batch upload for generated visualizations",
                        count=len(successful_results)
                    )
                    
                    upload_results = await self.upload_manager.upload_multiple(
                        successful_results,
                        max_concurrent=3
                    )
                    
                    # Log summary
                    successful_uploads = sum(1 for r in upload_results if r.success)
                    failed_uploads = len(upload_results) - successful_uploads
                    
                    logger.info(
                        "Batch upload completed",
                        total=len(upload_results),
                        successful=successful_uploads,
                        failed=failed_uploads
                    )
                    
                    if failed_uploads > 0:
                        logger.warning(
                            "Some uploads failed during batch processing",
                            failed_count=failed_uploads
                        )
                else:
                    logger.debug("No successful visualizations to upload")
            
            logger.info(
                "Batch generation completed",
                total_requests=len(requests),
                successful=sum(1 for r in results if r.success),
                failed=sum(1 for r in results if not r.success)
            )
            
        finally:
            # Restore original output directory
            self.config.output.base_dir = original_output_dir
            
        return results
    
    async def backfill_historical(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        intervals: Optional[List[Interval]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive historical visualizations with resume support."""
        logger.info("Starting comprehensive historical backfill with resume support")
        
        # Default to all intervals if none specified
        if intervals is None:
            intervals = [Interval.DAILY, Interval.HOURLY, Interval.CUMULATIVE]
            
        logger.info(
            "Backfill configuration", 
            intervals=[i.value for i in intervals],
            sensors=len(self.config.sensors),
            progress_log=str(self.config.progress_tracking.log_file),
            resume_enabled=self.config.progress_tracking.resume_enabled
        )
        
        # Determine date range
        if not start_date:
            start_date = await self.repository.get_earliest_timestamp()
            if not start_date:
                raise ValueError("No data available for backfill")
        
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        # Ensure we're working with UTC timezone for database queries
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        logger.info(
            "Backfill date range determined",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        # Convert to PST for proper day/hour boundary calculations
        start_date_pst = start_date.astimezone(PST_TZ)
        end_date_pst = end_date.astimezone(PST_TZ)
        
        # Generate all visualization tasks
        all_tasks = []
        daily_count = 0
        hourly_count = 0
        cumulative_count = 0
        
        # 1. DAILY: One image per PST day
        if Interval.DAILY in intervals:
            current_day_pst = start_date_pst.replace(hour=0, minute=0, second=0, microsecond=0)
            while current_day_pst < end_date_pst:
                next_day_pst = current_day_pst + timedelta(days=1)
                
                for sensor in self.config.sensors:
                    task = {
                        'sensor_type': sensor.type,
                        'start_time': current_day_pst,
                        'end_time': min(next_day_pst, end_date_pst),
                        'interval': Interval.DAILY,
                        'color_scheme': sensor.color_scheme
                    }
                    all_tasks.append(task)
                    daily_count += 1
                
                current_day_pst = next_day_pst
        
        # 2. HOURLY: One image per PST hour
        if Interval.HOURLY in intervals:
            current_hour_pst = start_date_pst.replace(minute=0, second=0, microsecond=0)
            while current_hour_pst < end_date_pst:
                next_hour_pst = current_hour_pst + timedelta(hours=1)
                
                for sensor in self.config.sensors:
                    task = {
                        'sensor_type': sensor.type,
                        'start_time': current_hour_pst,
                        'end_time': min(next_hour_pst, end_date_pst),
                        'interval': Interval.HOURLY,
                        'color_scheme': sensor.color_scheme
                    }
                    all_tasks.append(task)
                    hourly_count += 1
                
                current_hour_pst = next_hour_pst
        
        # 3. CUMULATIVE: Every hour with expanding windows
        if Interval.CUMULATIVE in intervals:
            current_time_pst = start_date_pst.replace(minute=0, second=0, microsecond=0)
            while current_time_pst < end_date_pst:
                next_time_pst = current_time_pst + timedelta(hours=1)
                
                for sensor in self.config.sensors:
                    task = {
                        'sensor_type': sensor.type,
                        'start_time': start_date_pst.replace(hour=0, minute=0, second=0, microsecond=0),  # Always from PST midnight
                        'end_time': min(next_time_pst, end_date_pst),  # Expand the window
                        'interval': Interval.CUMULATIVE,
                        'color_scheme': sensor.color_scheme
                    }
                    all_tasks.append(task)
                    cumulative_count += 1
                
                current_time_pst = next_time_pst
        
        initial_total = len(all_tasks)
        logger.info(
            "Generated initial backfill tasks",
            total_tasks=initial_total,
            daily=daily_count,
            hourly=hourly_count,
            cumulative=cumulative_count,
            sensors=len(self.config.sensors)
        )
        
        # Filter out already completed tasks if resume is enabled
        if self.config.progress_tracking.resume_enabled:
            logger.info("Filtering out completed tasks for resume functionality")
            pending_tasks, skipped_count = await self.progress_tracker.filter_pending_tasks(all_tasks)
            
            logger.info(
                "Resume filtering completed",
                initial_tasks=initial_total,
                pending_tasks=len(pending_tasks),
                skipped_tasks=skipped_count
            )
        else:
            pending_tasks = all_tasks
            skipped_count = 0
            logger.info("Resume functionality disabled, processing all tasks")
        
        total_tasks = len(pending_tasks)
        
        if total_tasks == 0:
            logger.info("No pending tasks found - backfill already complete")
            progress_summary = self.progress_tracker.get_progress_summary()
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_generated": initial_total,
                "successful": progress_summary["completed_tasks"],
                "failed": progress_summary["failed_tasks"],
                "skipped": skipped_count,
                "total_data_points": 0,
                "total_processing_time": 0.0,
                "message": "All visualizations already exist"
            }
        
        # Process pending tasks in batches
        batch_size = self.config.processing.batch_size
        results = []
        successful = 0
        failed = 0
        total_data_points = 0
        total_processing_time = 0.0
        
        logger.info(f"Starting batch processing of {total_tasks} pending tasks")
        
        for i in range(0, total_tasks, batch_size):
            batch = pending_tasks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_tasks + batch_size - 1) // batch_size
            
            logger.info(
                f"Processing batch {batch_num}/{total_batches}",
                batch_size=len(batch),
                progress=f"{i + len(batch)}/{total_tasks}",
                completed_so_far=successful,
                failed_so_far=failed
            )
            
            # Process batch with progress tracking
            batch_results = await self._process_batch_with_tracking(batch)
            
            for task, result in batch_results:
                results.append(result)
                if isinstance(result, Exception):
                    logger.error("Batch processing error", error=str(result))
                    failed += 1
                    # Mark as failed in progress tracker
                    await self.progress_tracker.mark_task_completed(
                        task['sensor_type'],
                        task['interval'],
                        task['start_time'],
                        task['end_time'],
                        Path("failed"),  # Placeholder path
                        success=False,
                        error_message=str(result)
                    )
                else:
                    if result.success:
                        successful += 1
                        total_data_points += result.data_points
                        total_processing_time += result.processing_time_seconds
                        # Mark as completed in progress tracker
                        await self.progress_tracker.mark_task_completed(
                            task['sensor_type'],
                            task['interval'],
                            task['start_time'],
                            task['end_time'],
                            result.output_path,
                            success=True
                        )
                    else:
                        failed += 1
                        # Mark as failed in progress tracker
                        await self.progress_tracker.mark_task_completed(
                            task['sensor_type'],
                            task['interval'],
                            task['start_time'],
                            task['end_time'],
                            result.output_path or Path("failed"),
                            success=False,
                            error_message=result.error_message
                        )
        
        # Compile final statistics
        progress_summary = self.progress_tracker.get_progress_summary()
        stats = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_generated": total_tasks,
            "successful": successful,
            "failed": failed,
            "skipped": skipped_count,
            "total_data_points": total_data_points,
            "total_processing_time": total_processing_time,
            "breakdown": {
                "daily": daily_count,
                "hourly": hourly_count,
                "cumulative": cumulative_count
            },
            "progress_summary": progress_summary
        }
        
        logger.info("Comprehensive historical backfill completed", **stats)
        return stats
    
    async def _process_batch_with_tracking(self, batch: List[Dict]) -> List[tuple]:
        """Process a batch of tasks with progress tracking."""
        # Create tasks for concurrent execution
        tasks = []
        for task_spec in batch:
            coro = self.generate_single(
                task_spec['sensor_type'],
                task_spec['start_time'],
                task_spec['end_time'],
                task_spec['interval'],
                task_spec['color_scheme']
            )
            tasks.append((task_spec, coro))
        
        # Execute batch concurrently
        results = await asyncio.gather(
            *[coro for _, coro in tasks],
            return_exceptions=True
        )
        
        # Pair results with original task specs
        return list(zip([task_spec for task_spec, _ in tasks], results))
    
    async def _process_single_request(self, request: VisualizationRequest) -> VisualizationResult:
        """Process a single visualization request."""
        try:
            # Get data
            data = await self.processor.get_sensor_data(
                request.sensor_type,
                request.start_time,
                request.end_time
            )
            
            # Generate visualization
            return await self.generator.generate(request, data)
            
        except Exception as e:
            logger.error(
                "Failed to process visualization request",
                sensor_type=request.sensor_type.value,
                interval=request.interval.value,
                error=str(e)
            )
            return VisualizationResult(
                request=request,
                output_path=None,
                success=False,
                error_message=str(e)
            )
    
    def _get_sensor_config(self, sensor_type: SensorType) -> Optional[Sensor]:
        """Get sensor configuration."""
        for sensor in self.config.sensors:
            if sensor.type == sensor_type:
                return sensor
        return None
    
    async def _data_sync_task(self) -> None:
        """Background task for syncing data."""
        while self._running:
            try:
                await asyncio.sleep(self.config.update_interval)
                # Data sync would happen here if we were writing to the DB
                # For read-only operation, this is mainly for cache management
                logger.debug("Data sync check completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in data sync task", error=str(e))
    
    async def _cache_cleanup_task(self) -> None:
        """Background task for cache cleanup."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                await self.cache.cleanup_expired()
                
                # Log cache stats periodically
                stats = await self.cache.get_stats()
                logger.debug("Cache stats", **stats)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cache cleanup task", error=str(e))
    
    async def _daily_generation_task(self) -> None:
        """Background task for daily visualizations."""
        while self._running:
            try:
                await asyncio.sleep(self.config.daily_check_interval)
                # Check if daily visualizations need updating
                logger.debug("Daily generation check completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in daily generation task", error=str(e))
    
    async def _hourly_generation_task(self) -> None:
        """Background task for hourly visualizations."""
        while self._running:
            try:
                await asyncio.sleep(self.config.hourly_check_interval)
                # Check if hourly visualizations need updating
                logger.debug("Hourly generation check completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in hourly generation task", error=str(e))
    
    async def _cumulative_generation_task(self) -> None:
        """Background task for cumulative visualizations."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cumulative_check_interval)
                # Check if cumulative visualizations need updating
                logger.debug("Cumulative generation check completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cumulative generation task", error=str(e)) 