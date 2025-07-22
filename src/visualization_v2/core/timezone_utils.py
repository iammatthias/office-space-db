"""Timezone utilities for the visualization service.

This module provides centralized timezone handling for the visualization service.
All sensor data is stored in UTC but visualizations are PST-based to match
the local sensor timezone.
"""

from datetime import datetime, date, time as dt_time
from typing import Tuple
from zoneinfo import ZoneInfo
import structlog

logger = structlog.get_logger()

# Timezone constants
PST_TZ = ZoneInfo("America/Los_Angeles")  # Pacific Standard/Daylight Time
UTC_TZ = ZoneInfo("UTC")


def ensure_pst(dt: datetime) -> datetime:
    """Ensure a datetime is in PST timezone.
    
    Args:
        dt: Datetime that may be naive or in any timezone
        
    Returns:
        Datetime converted to PST timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetimes are already PST
        return dt.replace(tzinfo=PST_TZ)
    else:
        return dt.astimezone(PST_TZ)


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is in UTC timezone.
    
    Args:
        dt: Datetime that may be naive or in any timezone
        
    Returns:
        Datetime converted to UTC timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetimes are UTC (for database queries)
        return dt.replace(tzinfo=UTC_TZ)
    else:
        return dt.astimezone(UTC_TZ)


def pst_to_utc(pst_dt: datetime) -> datetime:
    """Convert PST datetime to UTC for database queries.
    
    Args:
        pst_dt: Datetime in PST (may be naive or timezone-aware)
        
    Returns:
        Datetime converted to UTC
    """
    # First ensure it's PST
    pst_aware = ensure_pst(pst_dt)
    # Then convert to UTC
    return pst_aware.astimezone(UTC_TZ)


def utc_to_pst(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to PST for visualization.
    
    Args:
        utc_dt: Datetime in UTC (may be naive or timezone-aware)
        
    Returns:
        Datetime converted to PST
    """
    # First ensure it's UTC
    utc_aware = ensure_utc(utc_dt)
    # Then convert to PST
    return utc_aware.astimezone(PST_TZ)


def get_pst_day_boundaries(pst_date: date) -> Tuple[datetime, datetime]:
    """Get PST day boundaries (midnight to midnight).
    
    Args:
        pst_date: Date in PST timezone
        
    Returns:
        Tuple of (day_start_pst, day_end_pst) both in PST timezone
    """
    day_start = datetime.combine(pst_date, dt_time.min, tzinfo=PST_TZ)
    day_end = datetime.combine(pst_date, dt_time.max, tzinfo=PST_TZ)
    return day_start, day_end


def get_pst_hour_boundaries(pst_datetime: datetime) -> Tuple[datetime, datetime]:
    """Get PST hour boundaries.
    
    Args:
        pst_datetime: Datetime in PST timezone
        
    Returns:
        Tuple of (hour_start_pst, hour_end_pst) both in PST timezone
    """
    pst_aware = ensure_pst(pst_datetime)
    hour_start = pst_aware.replace(minute=0, second=0, microsecond=0)
    hour_end = pst_aware.replace(minute=59, second=59, microsecond=999999)
    return hour_start, hour_end


def convert_request_times_for_query(start_time: datetime, end_time: datetime) -> Tuple[datetime, datetime, datetime, datetime]:
    """Convert visualization request times to appropriate formats.
    
    This function handles the core timezone conversion logic:
    1. User requests are assumed to be PST (local sensor time)
    2. Database queries need UTC times
    3. Visualizations use PST times for proper day/hour mapping
    
    Args:
        start_time: Request start time (assumed PST if naive)
        end_time: Request end time (assumed PST if naive)
        
    Returns:
        Tuple of (start_pst, end_pst, start_utc, end_utc)
    """
    # Convert to PST for visualization
    start_pst = ensure_pst(start_time)
    end_pst = ensure_pst(end_time)
    
    # Convert to UTC for database queries
    start_utc = pst_to_utc(start_pst)
    end_utc = pst_to_utc(end_pst)
    
    logger.debug(
        "Timezone conversion for request",
        original_start=start_time.isoformat(),
        original_end=end_time.isoformat(),
        start_pst=start_pst.isoformat(),
        end_pst=end_pst.isoformat(),
        start_utc=start_utc.isoformat(),
        end_utc=end_utc.isoformat()
    )
    
    return start_pst, end_pst, start_utc, end_utc


def log_timezone_conversion(operation: str, **kwargs) -> None:
    """Log timezone conversion operations for debugging.
    
    Args:
        operation: Description of the operation being performed
        **kwargs: Additional logging data
    """
    logger.debug(f"Timezone conversion: {operation}", **kwargs) 