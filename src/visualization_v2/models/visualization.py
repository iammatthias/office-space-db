"""Visualization models."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from .sensor import SensorType


class Interval(str, Enum):
    """Visualization intervals."""
    CUMULATIVE = "cumulative"
    HOURLY = "hourly"
    DAILY = "daily"


class VisualizationRequest(BaseModel):
    """Request for generating a visualization."""
    sensor_type: SensorType
    start_time: datetime
    end_time: datetime
    interval: Interval
    color_scheme: Optional[str] = None
    output_path: Optional[Path] = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class VisualizationResult(BaseModel):
    """Result of a visualization generation."""
    request: VisualizationRequest
    output_path: Path
    success: bool
    error_message: Optional[str] = None
    data_points: int = 0
    processing_time_seconds: float = 0.0
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class BatchVisualizationRequest(BaseModel):
    """Request for generating multiple visualizations."""
    sensor_types: List[SensorType]
    start_time: datetime
    end_time: datetime
    intervals: List[Interval]
    output_dir: Path
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True 