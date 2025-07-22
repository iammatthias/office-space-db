"""
Environmental Data Visualization Service V2

A modular, efficient visualization service built with modern Python practices.
"""

from .core import VisualizationService
from .models import ServiceConfig, SensorType, Interval

__version__ = "0.1.0"
__all__ = ["VisualizationService", "ServiceConfig", "SensorType", "Interval"]
