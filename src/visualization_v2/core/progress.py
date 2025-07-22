"""Progress tracking for backfill operations."""

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple
import structlog

from models.sensor import SensorType
from models.visualization import Interval
from models.config import ProgressTrackingConfig, OutputConfig

logger = structlog.get_logger()


class ProgressEntry:
    """Represents a single progress entry for a generated visualization."""
    
    def __init__(
        self,
        sensor_type: SensorType,
        interval: Interval,
        start_time: datetime,
        end_time: datetime,
        output_path: Path,
        completed_at: datetime,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        self.sensor_type = sensor_type
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.output_path = output_path
        self.completed_at = completed_at
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensor_type": self.sensor_type.value,
            "interval": self.interval.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "output_path": str(self.output_path),
            "completed_at": self.completed_at.isoformat(),
            "success": self.success,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ProgressEntry":
        """Create from dictionary loaded from JSON."""
        return cls(
            sensor_type=SensorType(data["sensor_type"]),
            interval=Interval(data["interval"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            output_path=Path(data["output_path"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            success=data.get("success", True),
            error_message=data.get("error_message")
        )
    
    def get_task_key(self) -> str:
        """Generate unique key for this task."""
        return f"{self.sensor_type.value}_{self.interval.value}_{self.start_time.isoformat()}_{self.end_time.isoformat()}"


class BackfillProgressTracker:
    """Tracks progress of backfill operations for resumability."""
    
    def __init__(
        self,
        config: ProgressTrackingConfig,
        output_config: OutputConfig
    ):
        self.config = config
        self.output_config = output_config
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize progress tracker by loading existing progress."""
        await self._load_existing_progress()
    
    async def _load_existing_progress(self) -> None:
        """Load existing progress from log file."""
        if not self.config.log_file.exists():
            logger.info("No existing progress log found")
            return
        
        logger.info(f"Loading existing progress from {self.config.log_file}")
        
        try:
            with open(self.config.log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        entry = ProgressEntry.from_dict(data)
                        task_key = entry.get_task_key()
                        
                        if entry.success:
                            self.completed_tasks.add(task_key)
                        else:
                            self.failed_tasks.add(task_key)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
            
            logger.info(
                "Progress loaded",
                completed_tasks=len(self.completed_tasks),
                failed_tasks=len(self.failed_tasks)
            )
            
        except Exception as e:
            logger.error(f"Failed to load progress log: {e}")
            # Continue with empty progress rather than failing
    
    async def is_task_completed(
        self,
        sensor_type: SensorType,
        interval: Interval,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """Check if a task has already been completed successfully."""
        task_key = self._get_task_key(sensor_type, interval, start_time, end_time)
        
        # Check in-memory tracking first
        if task_key in self.completed_tasks:
            return True
        
        # Optionally check if the output file exists
        if self.config.check_existing_files:
            expected_path = self._get_expected_output_path(
                sensor_type, interval, start_time, end_time
            )
            if expected_path.exists():
                logger.debug(f"Found existing file: {expected_path}")
                # Mark as completed in memory to avoid future file checks
                self.completed_tasks.add(task_key)
                return True
        
        return False
    
    async def mark_task_completed(
        self,
        sensor_type: SensorType,
        interval: Interval,
        start_time: datetime,
        end_time: datetime,
        output_path: Path,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Mark a task as completed and log to file."""
        async with self._lock:
            entry = ProgressEntry(
                sensor_type=sensor_type,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                output_path=output_path,
                completed_at=datetime.now(timezone.utc),
                success=success,
                error_message=error_message
            )
            
            # Update in-memory tracking
            task_key = entry.get_task_key()
            if success:
                self.completed_tasks.add(task_key)
                self.failed_tasks.discard(task_key)  # Remove from failed if it was there
            else:
                self.failed_tasks.add(task_key)
                self.completed_tasks.discard(task_key)  # Remove from completed if it was there
            
            # Append to log file
            await self._append_to_log(entry)
    
    async def _append_to_log(self, entry: ProgressEntry) -> None:
        """Append progress entry to log file."""
        try:
            with open(self.config.log_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to progress log: {e}")
    
    def _get_task_key(
        self,
        sensor_type: SensorType,
        interval: Interval,
        start_time: datetime,
        end_time: datetime
    ) -> str:
        """Generate unique key for a task."""
        return f"{sensor_type.value}_{interval.value}_{start_time.isoformat()}_{end_time.isoformat()}"
    
    def _get_expected_output_path(
        self,
        sensor_type: SensorType,
        interval: Interval,
        start_time: datetime,
        end_time: datetime
    ) -> Path:
        """Get expected output path for a visualization."""
        from core.timezone_utils import utc_to_pst
        
        # Convert to PST for filename generation
        start_pst = utc_to_pst(start_time)
        
        if interval == Interval.DAILY:
            filename = f"{start_pst.strftime('%Y-%m-%d')}_daily.png"
        elif interval == Interval.HOURLY:
            filename = f"{start_pst.strftime('%Y-%m-%d_%H')}_hourly.png"
        else:  # CUMULATIVE
            filename = f"{start_pst.strftime('%Y-%m-%d_%H')}_cumulative.png"
        
        return (self.output_config.base_dir / 
                sensor_type.value / 
                interval.value / 
                filename)
    
    def get_progress_summary(self) -> Dict[str, int]:
        """Get summary of current progress."""
        return {
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tracked": len(self.completed_tasks) + len(self.failed_tasks)
        }
    
    async def filter_pending_tasks(
        self,
        all_tasks: List[Dict]
    ) -> Tuple[List[Dict], int]:
        """Filter out already completed tasks from a list of tasks."""
        pending_tasks = []
        skipped_count = 0
        
        for task in all_tasks:
            if await self.is_task_completed(
                task['sensor_type'],
                task['interval'],
                task['start_time'],
                task['end_time']
            ):
                skipped_count += 1
                logger.debug(
                    "Skipping completed task",
                    sensor=task['sensor_type'].value,
                    interval=task['interval'].value,
                    start=task['start_time'].isoformat()
                )
            else:
                pending_tasks.append(task)
        
        logger.info(
            "Task filtering completed",
            total_tasks=len(all_tasks),
            pending_tasks=len(pending_tasks),
            skipped_tasks=skipped_count
        )
        
        return pending_tasks, skipped_count
    
    async def cleanup_log(self, keep_days: int = 30) -> None:
        """Clean up old entries from the log file (optional maintenance)."""
        # This could be implemented to remove very old entries
        # For now, we'll keep all entries for complete history
        pass 