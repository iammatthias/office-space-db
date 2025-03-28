"""
Main visualization generator for environmental data.
Matches the React/Three.js implementation exactly.
"""

import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from .utils import (
    EnvironmentalData,
    organize_data_by_day,
    get_range,
    create_minute_map,
    interpolate_value,
    Range,
    get_day_boundaries,
    convert_to_pst
)
from .color_schemes import COLOR_SCHEMES
import math
import logging

logger = logging.getLogger(__name__)

MINUTES_IN_DAY = 1440  # 1px per minute
BASE_HEIGHT = 1680     # Fixed height for all visualizations
SCALE_FACTOR = 4      # Default scale factor for final output

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_color_for(value: float, min_value: float, max_value: float, scheme: str) -> Tuple[int, int, int]:
    """Get color for a value using the specified color scheme."""
    # Normalize the value to a 0-1 scale
    normalized = (value - min_value) / (max_value - min_value) if max_value > min_value else 0.5
    normalized = min(max(normalized, 0.0), 1.0)
    
    # Get the appropriate color scheme
    colors = COLOR_SCHEMES.get(scheme, COLOR_SCHEMES['redblue'])
    
    # Calculate the floating point index and interpolate between colors
    float_index = normalized * (len(colors) - 1)
    lower_index = int(float_index)
    upper_index = min(lower_index + 1, len(colors) - 1)
    
    # Get the two colors to interpolate between
    color1 = hex_to_rgb(colors[lower_index])
    color2 = hex_to_rgb(colors[upper_index])
    
    # Calculate the interpolation weight
    weight = float_index - lower_index
    
    # Interpolate between the two colors
    return tuple(
        int(c1 * (1 - weight) + c2 * weight)
        for c1, c2 in zip(color1, color2)
    )

class VisualizationGenerator:
    """Generator for environmental data visualizations."""
    
    def __init__(self, scale_factor: int = 4):
        """Initialize the visualization generator."""
        self.width = MINUTES_IN_DAY  # 1440 minutes in a day
        self.height = BASE_HEIGHT    # Fixed height for all visualizations
        self.scale_factor = scale_factor

    def _calculate_row_boundaries(self, num_rows: int) -> List[Tuple[int, int]]:
        """Calculate the exact pixel boundaries for each row.
        Returns a list of (start, end) tuples for each row that precisely
        divides the total height while handling fractional pixels.
        """
        row_boundaries = []
        for row in range(num_rows):
            # Calculate exact floating point positions
            start_float = (row * self.height) / num_rows
            end_float = ((row + 1) * self.height) / num_rows
            # Round to nearest pixel
            start = round(start_float)
            end = round(end_float)
            # Ensure no gaps between rows
            if row > 0:
                start = row_boundaries[-1][1]
            row_boundaries.append((start, end))
        return row_boundaries

    def generate_visualization(
        self,
        data: List[EnvironmentalData],
        column: str,
        color_scheme: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        interval: str = 'daily'
    ) -> Image:
        """Generate a visualization for a time period."""
        # Ensure start_time is in PST
        start_time = convert_to_pst(start_time)
        if end_time is None:
            # Default to end of the day if not specified
            end_time = start_time + timedelta(days=1)
        end_time = convert_to_pst(end_time)
        
        # Calculate number of days needed (partial days count as full days)
        start_date = start_time.date()
        end_date = end_time.date()
        if end_time.time() > time(0, 0):  # If end time is not midnight, we need one more day
            end_date += timedelta(days=1)
        num_rows = (end_date - start_date).days
        
        # Create image with fixed dimensions
        image = Image.new('RGB', (self.width, self.height))
        pixels = image.load()
        
        if not data:
            logger.warning(f"No data points found for {column} between {start_time} and {end_time}")
            return image.resize(
                (self.width * self.scale_factor, self.height * self.scale_factor),
                Image.NEAREST
            )
        
        # Get value range for normalization from all data
        values = [point.value for point in data]
        min_value = min(values)
        max_value = max(values)
        
        # Create minute map for each day and track closest values
        day_minute_maps = {}
        closest_values = {}  # Store closest values for each row
        
        for point in data:
            # Convert point time to PST and get its position
            point_time = convert_to_pst(point.time)
            # Calculate row based on days since start
            row = (point_time.date() - start_date).days
            # Calculate minute within the day (0-1439)
            minute = point_time.hour * 60 + point_time.minute
            
            if row not in day_minute_maps:
                day_minute_maps[row] = {}
            day_minute_maps[row][minute] = point.value
            
            # Update closest values for the row
            if row not in closest_values:
                closest_values[row] = {'before': None, 'after': None}
            closest_values[row]['after'] = point.value
            
            # Update 'before' value for next row if this is the last point of current row
            if minute == MINUTES_IN_DAY - 1 and row + 1 < num_rows:
                if row + 1 not in closest_values:
                    closest_values[row + 1] = {'before': None, 'after': None}
                closest_values[row + 1]['before'] = point.value
        
        # Calculate exact row boundaries
        row_boundaries = self._calculate_row_boundaries(num_rows)
        
        # Fill pixels for each day
        for row in range(num_rows):
            minute_map = day_minute_maps.get(row, {})
            data_minutes = sorted(minute_map.keys()) if minute_map else []
            row_start, row_end = row_boundaries[row]
            
            if data_minutes:
                # Get the closest values for interpolation at row boundaries
                row_closest = closest_values.get(row, {'before': None, 'after': None})
                prev_row_last = closest_values.get(row - 1, {}).get('after') if row > 0 else None
                next_row_first = closest_values.get(row + 1, {}).get('before') if row < num_rows - 1 else None
                
                first_data_minute = data_minutes[0]
                last_data_minute = data_minutes[-1]
                first_value = minute_map[first_data_minute]
                last_value = minute_map[last_data_minute]
                
                # Fill each minute in the row
                for minute in range(MINUTES_IN_DAY):
                    if minute in minute_map:
                        value = minute_map[minute]
                    else:
                        # Handle gaps differently based on position
                        if minute < first_data_minute:
                            # Use closest value: either last value from previous row or first value in current row
                            value = prev_row_last if prev_row_last is not None else first_value
                        elif minute > last_data_minute:
                            # Use closest value: either first value from next row or last value in current row
                            value = next_row_first if next_row_first is not None else last_value
                        else:
                            # Find closest values for interpolation within the row
                            before_value = None
                            after_value = None
                            before_minute = None
                            after_minute = None
                            
                            # Find previous value
                            for m in reversed(data_minutes):
                                if m < minute:
                                    before_minute = m
                                    before_value = minute_map[m]
                                    break
                            
                            # Find next value
                            for m in data_minutes:
                                if m > minute:
                                    after_minute = m
                                    after_value = minute_map[m]
                                    break
                            
                            # Interpolate between closest known values
                            if before_value is not None and after_value is not None:
                                range_size = after_minute - before_minute
                                weight = (minute - before_minute) / range_size
                                value = before_value + (after_value - before_value) * weight
                            else:
                                # Use the closest available value
                                value = before_value if before_value is not None else after_value
                    
                    # Get color and fill the column for this row
                    if value is not None:
                        color = get_color_for(value, min_value, max_value, color_scheme)
                        for y in range(row_start, row_end):
                            pixels[minute, y] = color
            else:
                # Handle completely empty rows by interpolating from surrounding rows
                prev_row_value = closest_values.get(row - 1, {}).get('after') if row > 0 else None
                next_row_value = closest_values.get(row + 1, {}).get('before') if row < num_rows - 1 else None
                
                if prev_row_value is not None or next_row_value is not None:
                    value = prev_row_value if prev_row_value is not None else next_row_value
                    color = get_color_for(value, min_value, max_value, color_scheme)
                    for minute in range(MINUTES_IN_DAY):
                        for y in range(row_start, row_end):
                            pixels[minute, y] = color
        
        # Scale the image to final dimensions
        return image.resize(
            (self.width * self.scale_factor, self.height * self.scale_factor),
            Image.NEAREST
        ) 