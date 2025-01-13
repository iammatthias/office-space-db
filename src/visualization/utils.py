"""
Utility functions for environmental data visualization.
"""

from typing import Tuple, List, Dict, Optional, NamedTuple
from datetime import datetime, timezone, timedelta
import numpy as np
from zoneinfo import ZoneInfo
from .color_schemes import COLOR_SCHEMES


class Range(NamedTuple):
    """Range of values, matching React implementation."""
    min: float
    max: float


def convert_to_pst(time: datetime) -> datetime:
    """Convert UTC time to PST, ensuring timezone awareness."""
    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    return time.astimezone(ZoneInfo("America/Los_Angeles"))


def get_day_boundaries(dt: datetime) -> Tuple[datetime, datetime]:
    """Get PST midnight boundaries for a given datetime."""
    pst_dt = convert_to_pst(dt)
    start = pst_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to RGB tuple with the same precision as Three.js Color."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))


def get_color_for(value: float, min_val: float, max_val: float, color_scheme: str) -> Tuple[float, float, float]:
    """
    Get RGB color for a value based on the color scheme and value range.
    Ensures smooth transitions at edges to prevent banding.
    """
    # Handle edge case where min and max are the same
    if max_val == min_val:
        normalized_value = 0.5
    else:
        # Simple linear normalization with clamping
        normalized_value = (value - min_val) / (max_val - min_val)
        normalized_value = max(0.0, min(1.0, normalized_value))
    
    scheme = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['redblue'])
    scheme_max = len(scheme) - 1
    
    # Scale to color range, ensuring we never hit the exact edges
    float_index = normalized_value * (scheme_max - 0.001)  # Prevent hitting last index
    lower_index = int(float_index)
    upper_index = min(lower_index + 1, scheme_max)
    
    # Get interpolation fraction
    fraction = float_index - lower_index
    
    # Get colors and interpolate
    lower_color = hex_to_rgb(scheme[lower_index])
    upper_color = hex_to_rgb(scheme[upper_index])
    
    return tuple(
        lower_color[i] * (1 - fraction) + upper_color[i] * fraction
        for i in range(3)
    )


class EnvironmentalData:
    """Class to hold environmental data points."""
    def __init__(self, time: datetime, value: float):
        """
        Initialize with UTC time, converts to PST internally.
        
        Args:
            time: UTC timestamp
            value: sensor value
        """
        self.time = convert_to_pst(time)
        self.value = value


def organize_data_by_day(data: List[EnvironmentalData]) -> List[List[EnvironmentalData]]:
    """
    Organize data points by day in PST.
    Matches the React implementation exactly.
    """
    day_map = {}
    
    for entry in data:
        date = entry.time.date().isoformat()
        if date not in day_map:
            day_map[date] = []
        day_map[date].append(entry)
    
    # Sort each day's data by time in ascending order
    for day_data in day_map.values():
        day_data.sort(key=lambda x: x.time)
    
    # Convert to list of lists in chronological order
    sorted_days = sorted(day_map.items(), key=lambda x: x[0])
    return [day_data for _, day_data in sorted_days]


def get_range(data: List[EnvironmentalData]) -> Range:
    """Get the min and max values from the data."""
    if not data:
        return Range(0.0, 1.0)
    values = [entry.value for entry in data]
    return Range(min(values), max(values))


def create_minute_map(day_data: List[EnvironmentalData], day_start: datetime) -> Dict[int, float]:
    """
    Create a map of minute -> value for a day's data.
    For the first day, fill all minutes before first data point with the first value.
    """
    minute_map = {}
    
    if not day_data:
        return minute_map
    
    # Fill in actual data points
    for entry in day_data:
        if entry.time >= day_start:
            minutes = entry.time.hour * 60 + entry.time.minute
            minute_map[minutes] = entry.value
    
    # For first day, fill missing minutes with first value
    if day_start.date() == convert_to_pst(day_data[0].time).date():
        first_value = day_data[0].value
        first_minute = min(minute_map.keys())
        for minute in range(first_minute):
            minute_map[minute] = first_value
            
    return minute_map


def interpolate_value(minute: int, 
                     minute_map: Dict[int, float],
                     first_minute: int,
                     last_minute: int,
                     min_val: float,
                     max_val: float) -> float:
    """
    Interpolate value for a specific minute.
    - Uses first known value for any minutes before first data point
    - Uses last known value for any minutes after last data point
    - Interpolates between known values for any gaps in between
    """
    # If we have the exact minute, use it
    if minute in minute_map:
        return minute_map[minute]
    
    # Get first and last known values
    first_value = minute_map[first_minute]
    last_value = minute_map[last_minute]
    
    # Before first data point, use first known value
    if minute < first_minute:
        return first_value
        
    # After last data point, use last known value
    if minute > last_minute:
        return last_value
    
    # Find nearest known values before and after current minute
    before_minute = minute - 1
    after_minute = minute + 1
    before_value = None
    after_value = None
    
    # Search backwards for before value
    while before_minute >= first_minute:
        if before_minute in minute_map:
            before_value = minute_map[before_minute]
            break
        before_minute -= 1
    
    # Search forwards for after value
    while after_minute <= last_minute:
        if after_minute in minute_map:
            after_value = minute_map[after_minute]
            break
        after_minute += 1
    
    # Between two known values, interpolate
    if before_value is not None and after_value is not None:
        range_minutes = after_minute - before_minute
        weight = (minute - before_minute) / range_minutes
        return before_value + (after_value - before_value) * weight
    
    # This shouldn't happen given the above checks, but just in case
    return first_value 