"""
Helper functions for the visualization service.
"""

from datetime import datetime, timedelta
from typing import Iterator


def hourly_range(start: datetime, end: datetime) -> Iterator[datetime]:
    """Generate hourly timestamps between start and end dates."""
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        yield current
        current += timedelta(hours=1)


def get_next_hour(dt: datetime) -> datetime:
    """Get the start of the next hour."""
    return (dt + timedelta(hours=1)).replace(
        minute=0, second=0, microsecond=0
    )


def format_timestamp(dt: datetime, format: str = "%Y%m%d_%H%M") -> str:
    """Format timestamp for filenames."""
    return dt.strftime(format) 