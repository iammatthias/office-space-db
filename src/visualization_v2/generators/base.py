"""Base visualization generator."""

from typing import List, Optional
from datetime import datetime, time as dt_time
from pathlib import Path
from abc import ABC, abstractmethod
import structlog
from PIL import Image
import time

from models.sensor import SensorData, SensorType
from models.visualization import VisualizationRequest, VisualizationResult, Interval
from models.config import OutputConfig
from .colors import ColorScheme
from core.timezone_utils import ensure_pst

logger = structlog.get_logger()


class BaseVisualizationGenerator(ABC):
    """Base class for visualization generators."""
    
    def __init__(self, output_config: OutputConfig):
        """Initialize the generator."""
        self.output_config = output_config
        
    @abstractmethod
    async def generate(
        self,
        request: VisualizationRequest,
        data: List[SensorData]
    ) -> VisualizationResult:
        """Generate a visualization from the given data."""
        pass
    
    def _create_output_path(
        self,
        sensor_type: SensorType,
        interval: Interval,
        start_time: datetime,
        end_time: datetime,
        custom_path: Optional[Path] = None
    ) -> Path:
        """Create output path for the visualization."""
        if custom_path:
            return custom_path
            
        # Convert to PST for filename generation
        # All filenames should be based on PST times for consistency
        start_pst = ensure_pst(start_time)
        end_pst = ensure_pst(end_time)
        
        # Create filename based on interval
        if interval == Interval.DAILY:
            filename = f"{start_pst.strftime('%Y-%m-%d')}_daily.png"
        elif interval == Interval.HOURLY:
            filename = f"{start_pst.strftime('%Y-%m-%d_%H')}_hourly.png"
        else:  # CUMULATIVE
            # For cumulative intervals, include end date if it spans multiple days
            start_date = start_pst.date()
            end_date = end_pst.date()
            
            if start_date == end_date:
                # Single day
                filename = f"{start_pst.strftime('%Y-%m-%d_%H-%M')}_{end_pst.strftime('%H-%M')}_cumulative.png"
            else:
                # Multiple days - include both dates
                filename = f"{start_pst.strftime('%Y-%m-%d_%H-%M')}_to_{end_pst.strftime('%Y-%m-%d_%H-%M')}_cumulative.png"
        
        # Get output directory
        output_dir = self.output_config.get_interval_dir(sensor_type.value, interval.value)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir / filename
    
    def _save_image(self, image: Image.Image, path: Path) -> None:
        """Save image to disk with proper compression."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_kwargs = {
            'format': self.output_config.image_format,
            'optimize': True,
        }
        
        if self.output_config.image_format.upper() == 'PNG':
            save_kwargs['compress_level'] = self.output_config.compression_level
            
        image.save(path, **save_kwargs)
        logger.debug("Image saved", path=str(path), size=image.size)
    
    def _get_data_range(self, data: List[SensorData]) -> tuple[float, float]:
        """Get min and max values from the data."""
        if not data:
            return 0.0, 1.0
            
        values = [point.value for point in data]
        return min(values), max(values)
    
    def _convert_to_pst(self, dt: datetime) -> datetime:
        """Convert datetime to PST timezone."""
        from zoneinfo import ZoneInfo
        pst_tz = ZoneInfo("America/Los_Angeles")
        
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
            
        return dt.astimezone(pst_tz) 