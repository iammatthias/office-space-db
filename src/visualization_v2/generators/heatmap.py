"""Heatmap visualization generator."""

import time
from datetime import datetime, timedelta, time as dt_time, date
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw
import structlog
import numpy as np

from models.sensor import SensorData, SensorType
from models.visualization import VisualizationRequest, VisualizationResult, Interval
from .base import BaseVisualizationGenerator
from .colors import ColorScheme, get_color_for_value
from core.timezone_utils import ensure_pst, utc_to_pst, PST_TZ, UTC_TZ

logger = structlog.get_logger()

# Constants matching the original implementation
MINUTES_IN_DAY = 1440  # 1px per minute
BASE_HEIGHT = 1680     # Fixed height for all visualizations


class HeatmapGenerator(BaseVisualizationGenerator):
    """Generator for heatmap-style visualizations."""
    
    async def generate(
        self,
        request: VisualizationRequest,
        data: List[SensorData]
    ) -> VisualizationResult:
        """Generate a heatmap visualization."""
        start_time = time.time()
        
        try:
            # Create output path
            output_path = self._create_output_path(
                request.sensor_type,
                request.interval,
                request.start_time,
                request.end_time,
                request.output_path
            )
            
            # Ensure we're working with PST times for proper day/hour boundaries
            # Request times should already be PST from the service layer
            start_pst = ensure_pst(request.start_time)
            end_pst = ensure_pst(request.end_time)

            logger.debug(
                "Generating heatmap",
                sensor_type=request.sensor_type.value,
                interval=request.interval.value,
                start_pst=start_pst.isoformat(),
                end_pst=end_pst.isoformat(),
                data_points=len(data)
            )
            
            # Determine visualization parameters based on interval type
            if request.interval == Interval.CUMULATIVE:
                # Cumulative: Each row = 1 day, columns = minutes (PST-based)
                # Show all days from start to end, each row represents a complete day
                start_date = start_pst.date()
                end_date = end_pst.date()
                num_rows = (end_date - start_date).days + 1
                visualization_type = "cumulative"
                
                logger.debug(
                    "Cumulative setup",
                    start_date=str(start_date),
                    end_date=str(end_date), 
                    num_rows=num_rows
                )
                
            elif request.interval == Interval.DAILY:
                # Daily: Single day stretched to full height (PST midnight-to-midnight)
                start_date = start_pst.date()
                num_rows = 1
                visualization_type = "daily"
                
                logger.debug(
                    "Daily setup",
                    date=str(start_date),
                    pst_day_start=start_pst.replace(hour=0, minute=0, second=0).isoformat(),
                    pst_day_end=start_pst.replace(hour=23, minute=59, second=59).isoformat()
                )
                
            else:  # HOURLY
                # Hourly: Single hour positioned within day (PST-based)
                start_date = start_pst.date()  
                num_rows = 1
                visualization_type = "hourly"
                
                logger.debug(
                    "Hourly setup",
                    date=str(start_date),
                    hour=start_pst.hour,
                    pst_hour_start=start_pst.isoformat(),
                    pst_hour_end=end_pst.isoformat()
                )
            
            # Create image with fixed dimensions
            width = MINUTES_IN_DAY
            height = BASE_HEIGHT
            image = Image.new('RGB', (width, height), color=(0, 0, 0))  # Start with black
            pixels = image.load()
            
            if not data:
                logger.warning(
                    "No data points for visualization",
                    sensor_type=request.sensor_type.value,
                    start_time=str(start_pst),
                    end_time=str(end_pst)
                )
                # Save empty black image
                scaled_image = image.resize(
                    (width * self.output_config.scale_factor, height * self.output_config.scale_factor),
                    Image.NEAREST
                )
                self._save_image(scaled_image, output_path)
                
                return VisualizationResult(
                    request=request,
                    output_path=output_path,
                    success=True,
                    data_points=0,
                    processing_time_seconds=time.time() - start_time
                )
            
            # Get value range for normalization
            min_value, max_value = self._get_data_range(data)
            
            # Convert color scheme string to enum
            color_scheme = ColorScheme(request.color_scheme or "base")
            
            # Fill the visualization based on type
            self._fill_visualization(
                pixels, data, request.interval, start_pst, end_pst, start_date, 
                num_rows, min_value, max_value, color_scheme
            )
            
            # Scale the image to final dimensions
            scaled_image = image.resize(
                (width * self.output_config.scale_factor, height * self.output_config.scale_factor),
                Image.NEAREST
            )
            
            # Save the image
            self._save_image(scaled_image, output_path)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Heatmap generated successfully",
                sensor_type=request.sensor_type.value,
                interval=request.interval.value,
                output_path=str(output_path),
                data_points=len(data),
                processing_time=f"{processing_time:.2f}s",
                visualization_type=visualization_type
            )
            
            return VisualizationResult(
                request=request,
                output_path=output_path,
                success=True,
                data_points=len(data),
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            logger.error(
                "Failed to generate heatmap",
                sensor_type=request.sensor_type.value,
                interval=request.interval.value,
                error=str(e)
            )
            
            return VisualizationResult(
                request=request,
                output_path=None,
                success=False,
                error_message=str(e),
                data_points=len(data) if data else 0,
                processing_time_seconds=time.time() - start_time
            )
    
    def _fill_visualization(
        self,
        pixels,
        data: List[SensorData],
        interval: Interval,
        start_pst: datetime,
        end_pst: datetime,
        start_date: date,
        num_rows: int,
        min_value: float,
        max_value: float,
        color_scheme: ColorScheme
    ) -> None:
        """Fill the visualization based on interval type."""
        
        # Convert UTC-stored data to PST for proper visualization organization
        # Database stores UTC timestamps, but visualization should be PST-based
        data_map = {}  # {day_offset: {minute: value}}
        
        # Calculate end date for proper filtering
        if interval == Interval.CUMULATIVE:
            # For cumulative, we want all days from start to end
            end_date = start_date + timedelta(days=num_rows - 1)
        else:
            # For daily/hourly, we're dealing with a single day
            end_date = start_date
        
        processed_points = 0
        
        for point in data:
            # Convert UTC timestamp to PST for proper day/minute mapping using utility
            point_pst = utc_to_pst(point.timestamp)
            
            # Filter data points based on the visualization type
            if interval == Interval.CUMULATIVE:
                # For cumulative: include all data within the date range
                if point_pst.date() < start_date or point_pst.date() > end_date:
                    continue
            elif interval == Interval.DAILY:
                # For daily: only include data from the specific day (PST midnight-to-midnight)
                if point_pst.date() != start_date:
                    continue
            else:  # HOURLY
                # For hourly: only include data from the specific hour
                if (point_pst.date() != start_date or 
                    point_pst.hour < start_pst.hour or 
                    point_pst.hour >= end_pst.hour):
                    continue
                
            day_offset = (point_pst.date() - start_date).days
            minute = point_pst.hour * 60 + point_pst.minute
            
            if 0 <= minute < MINUTES_IN_DAY and 0 <= day_offset < num_rows:  # Valid minute and day
                if day_offset not in data_map:
                    data_map[day_offset] = {}
                data_map[day_offset][minute] = point.value
                processed_points += 1
        
        logger.debug(
            "Data mapping completed",
            processed_points=processed_points,
            days_with_data=len(data_map),
            interval=interval.value
        )
        
        # Calculate row boundaries
        row_boundaries = []
        for row in range(num_rows):
            start_y = int((row * BASE_HEIGHT) / num_rows)
            end_y = int(((row + 1) * BASE_HEIGHT) / num_rows)
            row_boundaries.append((start_y, end_y))
        
        if interval == Interval.CUMULATIVE:
            self._fill_cumulative(pixels, data_map, num_rows, min_value, max_value, color_scheme, row_boundaries)
        elif interval == Interval.DAILY:
            self._fill_daily(pixels, data_map.get(0, {}), min_value, max_value, color_scheme, row_boundaries[0])
        else:  # HOURLY
            # For hourly, use the PST hour from the request time
            hour_start = start_pst.hour
            self._fill_hourly(pixels, data_map.get(0, {}), hour_start, min_value, max_value, color_scheme, row_boundaries[0])
    
    def _fill_cumulative(
        self,
        pixels,
        data_map: Dict[int, Dict[int, float]], 
        num_rows: int,
        min_value: float,
        max_value: float,
        color_scheme: ColorScheme,
        row_boundaries: List[Tuple[int, int]]
    ) -> None:
        """Fill cumulative visualization: each row is a day, each column is a minute."""
        
        # Get all available data for fallback colors
        all_values = []
        for day_data in data_map.values():
            all_values.extend(day_data.values())
        
        if not all_values:
            return  # No data at all
        
        # Use median as fallback color for areas with no data
        fallback_value = sorted(all_values)[len(all_values) // 2]
        
        for row in range(num_rows):
            day_data = data_map.get(row, {})
            start_y, end_y = row_boundaries[row]
            
            if not day_data:
                # No data for this day - use fallback color for entire day
                color = get_color_for_value(fallback_value, min_value, max_value, color_scheme)
                for minute in range(MINUTES_IN_DAY):
                    for y in range(start_y, end_y):
                        pixels[minute, y] = color
                continue
            
            # Get sorted minutes for this day
            minutes = sorted(day_data.keys())
            
            # Fill each minute column for this row
            for minute in range(MINUTES_IN_DAY):
                value = None
                
                if minute in day_data:
                    # Exact data point
                    value = day_data[minute]
                else:
                    # Always interpolate or extrapolate - never leave black
                    if minute < minutes[0]:
                        # Before first data point - use first value
                        value = day_data[minutes[0]]
                    elif minute > minutes[-1]:
                        # After last data point - use last value
                        value = day_data[minutes[-1]]
                    else:
                        # Between data points - interpolate
                        prev_minute = max(m for m in minutes if m <= minute)
                        next_minute = min(m for m in minutes if m >= minute)
                        
                        if prev_minute == next_minute:
                            value = day_data[prev_minute]
                        else:
                            # Linear interpolation
                            prev_value = day_data[prev_minute]
                            next_value = day_data[next_minute]
                            weight = (minute - prev_minute) / (next_minute - prev_minute)
                            value = prev_value + weight * (next_value - prev_value)
                
                # Always fill with a color - never leave black
                if value is None:
                    value = fallback_value
                    
                color = get_color_for_value(value, min_value, max_value, color_scheme)
                # Fill the column for this row
                for y in range(start_y, end_y):
                    pixels[minute, y] = color
    
    def _fill_daily(
        self,
        pixels,
        day_data: Dict[int, float],
        min_value: float,
        max_value: float, 
        color_scheme: ColorScheme,
        row_bounds: Tuple[int, int]
    ) -> None:
        """Fill daily visualization: single day stretched to full height."""
        
        start_y, end_y = row_bounds
        
        if not day_data:
            logger.warning("No data for daily visualization")
            return
        
        minutes = sorted(day_data.keys())
        
        # Use median value as fallback for areas with no data
        fallback_value = sorted(day_data.values())[len(day_data.values()) // 2]
        
        # Fill each minute column across the full height
        for minute in range(MINUTES_IN_DAY):
            value = None
            
            if minute in day_data:
                value = day_data[minute]
            else:
                # Always interpolate/extrapolate - never leave black
                if minute < minutes[0]:
                    value = day_data[minutes[0]]
                elif minute > minutes[-1]:
                    value = day_data[minutes[-1]]
                else:
                    # Interpolate between surrounding points
                    prev_minute = max(m for m in minutes if m <= minute)
                    next_minute = min(m for m in minutes if m >= minute)
                    
                    if prev_minute == next_minute:
                        value = day_data[prev_minute]
                    else:
                        prev_value = day_data[prev_minute]
                        next_value = day_data[next_minute]
                        weight = (minute - prev_minute) / (next_minute - prev_minute)
                        value = prev_value + weight * (next_value - prev_value)
            
            # Always fill with a color - never leave black
            if value is None:
                value = fallback_value
                
            color = get_color_for_value(value, min_value, max_value, color_scheme)
            # Fill entire height for this minute
            for y in range(start_y, end_y):
                pixels[minute, y] = color
    
    def _fill_hourly(
        self,
        pixels,
        day_data: Dict[int, float],
        hour_start: int,
        min_value: float,
        max_value: float,
        color_scheme: ColorScheme, 
        row_bounds: Tuple[int, int]
    ) -> None:
        """Fill hourly visualization: hour slice positioned in day with fill colors."""
        
        start_y, end_y = row_bounds
        
        # Extract data for the specific hour
        hour_start_minute = hour_start * 60
        hour_end_minute = hour_start_minute + 60
        
        if not day_data:
            return
        
        # Get all data points in the hour
        hour_data = {m: v for m, v in day_data.items() 
                    if hour_start_minute <= m < hour_end_minute}
        
        if not hour_data:
            # No data for this hour - use nearest available data from the day
            day_minutes = sorted(day_data.keys())
            if day_minutes:
                # Find closest minute to the hour
                hour_center = hour_start_minute + 30
                closest_minute = min(day_minutes, key=lambda m: abs(m - hour_center))
                fallback_value = day_data[closest_minute]
            else:
                return
        else:
            hour_minutes = sorted(hour_data.keys())
            first_value = hour_data[hour_minutes[0]]
            last_value = hour_data[hour_minutes[-1]]
            
            # Use median of hour data as fallback
            fallback_value = sorted(hour_data.values())[len(hour_data.values()) // 2]
        
        # Fill all minutes in the day (0-1439)
        for minute in range(MINUTES_IN_DAY):
            value = None
            
            if hour_start_minute <= minute < hour_end_minute:
                # Within the target hour
                if hour_data and minute in hour_data:
                    value = hour_data[minute]
                elif hour_data:
                    # Interpolate within hour
                    if minute < hour_minutes[0]:
                        value = first_value
                    elif minute > hour_minutes[-1]:
                        value = last_value
                    else:
                        # Find surrounding values
                        prev_minute = max(m for m in hour_minutes if m <= minute)
                        next_minute = min(m for m in hour_minutes if m >= minute)
                        
                        if prev_minute == next_minute:
                            value = hour_data[prev_minute]
                        else:
                            prev_value = hour_data[prev_minute]
                            next_value = hour_data[next_minute]
                            weight = (minute - prev_minute) / (next_minute - prev_minute)
                            value = prev_value + weight * (next_value - prev_value)
                else:
                    # No hour data, use fallback
                    value = fallback_value
            else:
                # Outside the hour - use first/last value as fill or fallback
                if hour_data:
                    if minute < hour_start_minute:
                        value = first_value  # Fill before with first hour value
                    else:  # minute >= hour_end_minute
                        value = last_value   # Fill after with last hour value
                else:
                    value = fallback_value
            
            # Always fill with a color - never leave black
            if value is None:
                value = fallback_value
                
            color = get_color_for_value(value, min_value, max_value, color_scheme)
            # Fill entire height for this minute  
            for y in range(start_y, end_y):
                pixels[minute, y] = color 