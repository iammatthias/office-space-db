"""Configuration models for the visualization service."""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .sensor import SensorType, Sensor


def _load_env_file():
    """Load environment variables from .env file in common locations."""
    env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path(__file__).parent.parent.parent.parent / "config" / ".env",  # ../../../config/.env
        Path(__file__).parent.parent / ".env",  # Parent directory
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
            break


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    path: Path = Field(default_factory=lambda: Path("data/environmental.db"))
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    query_timeout: int = Field(default=60, description="Query timeout in seconds")
    max_connections: int = Field(default=10, description="Maximum number of connections")
    
    def model_post_init(self, __context) -> None:
        """Ensure database path is absolute."""
        if not self.path.is_absolute():
            # Make relative to the current working directory
            self.path = Path.cwd() / self.path


class OutputConfig(BaseModel):
    """Output configuration."""
    base_dir: Path = Field(default_factory=lambda: Path("visualizations"))
    scale_factor: int = Field(default=4, description="Image scaling factor")
    image_format: str = "PNG"
    compression_level: int = Field(default=6, description="PNG compression level (0-9)")
    
    def model_post_init(self, __context) -> None:
        """Ensure output directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sensor_dir(self, sensor_type: str) -> Path:
        """Get output directory for a sensor."""
        return self.base_dir / sensor_type
    
    def get_interval_dir(self, sensor_type: str, interval: str) -> Path:
        """Get output directory for a sensor/interval combination."""
        return self.get_sensor_dir(sensor_type) / interval


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    batch_size: int = 10
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    max_concurrent: int = 4
    max_workers: int = Field(default=4, description="Maximum worker threads")
    memory_limit_mb: int = Field(default=2048, description="Memory limit in MB")


class ProgressTrackingConfig(BaseModel):
    """Progress tracking configuration for backfill operations."""
    log_file: Path = Field(default_factory=lambda: Path("backfill_progress.jsonl"))
    check_existing_files: bool = True
    resume_enabled: bool = True
    
    def model_post_init(self, __context) -> None:
        """Ensure progress log directory exists."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


class PinataConfig(BaseModel):
    """Pinata IPFS configuration."""
    jwt: str = Field(description="Pinata JWT token")
    group_id: Optional[str] = Field(default=None, description="Pinata group ID")
    base_url: str = Field(default="https://uploads.pinata.cloud/v3/files")
    
    @classmethod
    def from_env(cls) -> "PinataConfig":
        """Create from environment variables."""
        import os
        _load_env_file()  # Load .env file
        return cls(
            jwt=os.getenv("PINATA_JWT", ""),
            group_id=os.getenv("PINATA_GROUP_ID")
        )


class CloudflareKVConfig(BaseModel):
    """Cloudflare KV configuration."""
    api_token: str = Field(description="Cloudflare API token")
    namespace_id: str = Field(description="KV namespace ID")
    base_url: str = Field(default="https://api.cloudflare.com/client/v4")
    
    @classmethod
    def from_env(cls) -> "CloudflareKVConfig":
        """Create from environment variables."""
        import os
        _load_env_file()  # Load .env file
        return cls(
            api_token=os.getenv("CLOUDFLARE_API_TOKEN", ""),
            namespace_id=os.getenv("KV_NAMESPACE_ID", "")
        )


class UploadConfig(BaseModel):
    """Upload configuration for external services."""
    enabled: bool = Field(default=False, description="Enable uploads to external services")
    delete_local: bool = Field(default=True, description="Delete local files after successful upload")
    pinata: Optional[PinataConfig] = Field(default=None)
    cloudflare_kv: Optional[CloudflareKVConfig] = Field(default=None)
    
    @classmethod
    def from_env(cls) -> "UploadConfig":
        """Create from environment variables with auto-detection."""
        import os
        _load_env_file()  # Load .env file
        
        # Check if required environment variables exist
        pinata_jwt = os.getenv("PINATA_JWT")
        cf_token = os.getenv("CLOUDFLARE_API_TOKEN")
        kv_namespace = os.getenv("KV_NAMESPACE_ID")
        
        enabled = bool(pinata_jwt and cf_token and kv_namespace)
        
        return cls(
            enabled=enabled,
            pinata=PinataConfig.from_env() if pinata_jwt else None,
            cloudflare_kv=CloudflareKVConfig.from_env() if (cf_token and kv_namespace) else None
        )


class ServiceConfig(BaseModel):
    """Complete service configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    progress_tracking: ProgressTrackingConfig = Field(default_factory=ProgressTrackingConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig.from_env)
    sensors: List[Sensor] = Field(default_factory=list)
    
    # Service intervals (in seconds)
    update_interval: int = Field(default=300, description="Data update interval")
    daily_check_interval: int = Field(default=3600, description="Daily check interval")
    hourly_check_interval: int = Field(default=300, description="Hourly check interval")
    cumulative_check_interval: int = Field(default=60, description="Cumulative check interval")
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    @classmethod
    def create_default(cls) -> "ServiceConfig":
        """Create default configuration with all sensors."""
        return cls(
            sensors=[
                Sensor(type=SensorType.TEMPERATURE, color_scheme="redblue"),
                Sensor(type=SensorType.HUMIDITY, color_scheme="cyan"),
                Sensor(type=SensorType.PRESSURE, color_scheme="green"),
                Sensor(type=SensorType.LIGHT, color_scheme="base"),
                Sensor(type=SensorType.UV, color_scheme="purple"),
                Sensor(type=SensorType.GAS, color_scheme="yellow"),
            ]
        ) 