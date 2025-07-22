"""Sensor data models."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SensorType(str, Enum):
    """Available sensor types."""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    UV = "uv"
    GAS = "gas"


class Sensor(BaseModel):
    """Sensor configuration."""
    type: SensorType
    color_scheme: str = Field(default="base", description="Color scheme for visualization")
    display_name: Optional[str] = None
    unit: Optional[str] = None
    
    @property
    def name(self) -> str:
        """Get the sensor name."""
        return self.display_name or self.type.value
    
    @property
    def db_column(self) -> str:
        """Get the database column name for this sensor."""
        column_map = {
            SensorType.TEMPERATURE: "temp",
            SensorType.HUMIDITY: "hum", 
            SensorType.PRESSURE: "pressure",
            SensorType.LIGHT: "lux",
            SensorType.UV: "uv",
            SensorType.GAS: "gas",
        }
        return column_map[self.type]


class SensorData(BaseModel):
    """Individual sensor data point."""
    timestamp: datetime
    value: float
    sensor_type: SensorType
    
    class Config:
        """Pydantic config."""
        frozen = True  # Make immutable 