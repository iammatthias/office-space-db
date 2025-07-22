"""Main CLI interface."""

import asyncio
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List
import sys
import typer
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from models.config import ServiceConfig
from models.sensor import SensorType
from models.visualization import Interval
from core import VisualizationService

console = Console()
app = typer.Typer(help="Environmental data visualization service")

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


@app.command()
def run(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path"
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db-path",
        help="Database file path"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for images"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level"
    )
):
    """Run the visualization service continuously."""
    # Load configuration
    config = _load_config(config_file, db_path, output_dir, log_level)
    
    # Setup logging
    _setup_logging(config.log_level, config.log_file)
    
    console.print(f"[green]Starting visualization service...[/green]")
    console.print(f"Database: {config.database.path}")
    console.print(f"Output: {config.output.base_dir}")
    console.print(f"Sensors: {', '.join(s.type.value for s in config.sensors)}")
    
    # Run the service
    asyncio.run(_run_service(config))


@app.command()
def generate(
    sensor: SensorType = typer.Argument(..., help="Sensor type"),
    start: str = typer.Argument(..., help="Start time (ISO format)"),
    end: str = typer.Argument(..., help="End time (ISO format)"),
    interval: Interval = typer.Option(Interval.DAILY, "--interval", "-i", help="Time interval"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Database file path"),
    color_scheme: Optional[str] = typer.Option(None, "--color", help="Color scheme")
):
    """Generate a single visualization."""
    # Parse dates
    try:
        start_time = datetime.fromisoformat(start)
        end_time = datetime.fromisoformat(end)
    except ValueError as e:
        console.print(f"[red]Error parsing dates: {e}[/red]")
        raise typer.Exit(1)
    
    # Load configuration
    config = _load_config(db_path=db_path)
    _setup_logging(config.log_level, config.log_file)
    
    console.print(f"[green]Generating {sensor.value} visualization...[/green]")
    
    # Run generation
    asyncio.run(_generate_single(config, sensor, start_time, end_time, interval, output, color_scheme))


@app.command()
def backfill(
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Start date for backfill (ISO format)"
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date", 
        help="End date for backfill (ISO format)"
    ),
    sensors: Optional[List[SensorType]] = typer.Option(
        None,
        "--sensor",
        help="Specific sensors to backfill (can be repeated)"
    ),
    intervals: Optional[List[Interval]] = typer.Option(
        None,
        "--interval",
        help="Specific intervals to generate (daily, hourly, cumulative). Can be repeated. Default: all intervals"
    ),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Database file path"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output directory"),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Enable resume functionality (skip already completed visualizations)"
    ),
    check_files: bool = typer.Option(
        True,
        "--check-files/--no-check-files",
        help="Check for existing output files before generating"
    ),
    progress_log: Optional[Path] = typer.Option(
        None,
        "--progress-log",
        help="Custom path for progress log file"
    )
):
    """Generate historical visualizations with resume support."""
    # Parse dates
    start_dt = None
    end_dt = None
    
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
        except ValueError as e:
            console.print(f"[red]Error parsing start date: {e}[/red]")
            raise typer.Exit(1)
    
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
        except ValueError as e:
            console.print(f"[red]Error parsing end date: {e}[/red]")
            raise typer.Exit(1)
    
    # Load configuration
    config = _load_config(db_path=db_path, output_dir=output_dir)
    
    # Override progress tracking settings from CLI options
    config.progress_tracking.resume_enabled = resume
    config.progress_tracking.check_existing_files = check_files
    if progress_log:
        config.progress_tracking.log_file = progress_log
    
    _setup_logging(config.log_level, config.log_file)
    
    # Filter sensors and intervals if specified
    if sensors:
        config.sensors = [s for s in config.sensors if s.type in sensors]
    
    # Show what will be processed
    intervals_str = ", ".join([i.value for i in intervals]) if intervals else "all (daily, hourly, cumulative)"
    sensors_str = ", ".join([s.type.value for s in config.sensors])
    
    console.print(f"[blue]Intervals:[/blue] {intervals_str}")
    console.print(f"[blue]Sensors:[/blue] {sensors_str}")
    console.print(f"[blue]Resume enabled:[/blue] {resume}")
    console.print(f"[blue]Check existing files:[/blue] {check_files}")
    console.print(f"[blue]Progress log:[/blue] {config.progress_tracking.log_file}")
    console.print(f"[green]Starting historical backfill...[/green]")
    
    # Run backfill
    asyncio.run(_run_backfill(config, start_dt, end_dt, intervals))


@app.command()
def progress(
    progress_log: Optional[Path] = typer.Option(
        None,
        "--progress-log",
        help="Path to progress log file"
    ),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Database file path"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output directory")
):
    """Show backfill progress status and statistics."""
    config = _load_config(db_path=db_path, output_dir=output_dir)
    
    if progress_log:
        config.progress_tracking.log_file = progress_log
    
    _setup_logging("WARNING", None)  # Quiet logging for progress display
    
    asyncio.run(_show_progress(config))


@app.command()
def stats(
    sensor: Optional[SensorType] = typer.Option(None, "--sensor", help="Specific sensor"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Database file path")
):
    """Show database statistics."""
    config = _load_config(db_path=db_path)
    _setup_logging("WARNING", None)  # Quiet logging for stats
    
    asyncio.run(_show_stats(config, sensor))


@app.command()
def sync(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path"
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db-path",
        help="Database file path"
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Perform full sync (otherwise incremental)"
    ),
    batch_size: int = typer.Option(
        10000,
        "--batch-size",
        help="Batch size for full sync"
    )
):
    """Sync data from Supabase to local database."""
    console.print("[green]Starting Supabase sync...[/green]")
    
    # Load configuration
    config = _load_config(config_file, db_path, None, None)
    
    # Run the sync
    asyncio.run(_run_sync(config, full, batch_size))


@app.command()
def test_upload(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path"
    ),
):
    """Test the upload functionality with a sample image."""
    # Load configuration with upload settings
    config = _load_config(config_file)
    _setup_logging(config.log_level, config.log_file)
    
    # Check individual environment variables
    env_vars = {
        "PINATA_JWT": os.getenv("PINATA_JWT"),
        "CLOUDFLARE_API_TOKEN": os.getenv("CLOUDFLARE_API_TOKEN"),
        "CLOUDFLARE_ACCOUNT_ID": os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        "KV_NAMESPACE_ID": os.getenv("KV_NAMESPACE_ID")
    }
    
    console.print("[green]Environment Variable Status:[/green]")
    all_set = True
    for var_name, var_value in env_vars.items():
        if var_value:
            console.print(f"✓ {var_name}: [green]Set[/green]")
        else:
            console.print(f"✗ {var_name}: [red]Missing[/red]")
            all_set = False
    
    if not all_set:
        console.print("\n[red]Upload functionality is not enabled[/red]")
        console.print("Please set the missing environment variables in your config/.env file")
        raise typer.Exit(1)
    
    console.print(f"\n[green]Configuration Status:[/green]")
    console.print(f"Upload enabled: {'✓' if config.upload.enabled else '✗'}")
    console.print(f"Pinata configured: {'✓' if config.upload.pinata else '✗'}")
    console.print(f"Cloudflare configured: {'✓' if config.upload.cloudflare_kv else '✗'}")
    
    if not config.upload.enabled:
        console.print("[red]Upload configuration failed to initialize properly[/red]")
        raise typer.Exit(1)
    
    console.print("\n[green]Testing upload functionality...[/green]")
    
    # Run the upload demo
    asyncio.run(_test_upload(config))


async def _run_service(config: ServiceConfig) -> None:
    """Run the service continuously."""
    service = VisualizationService(config)
    
    try:
        await service.start()
        console.print("[green]Service started successfully![/green]")
        console.print("Press Ctrl+C to stop...")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping service...[/yellow]")
    finally:
        await service.stop()
        console.print("[green]Service stopped.[/green]")


async def _generate_single(
    config: ServiceConfig,
    sensor: SensorType,
    start_time: datetime,
    end_time: datetime,
    interval: Interval,
    output: Optional[Path],
    color_scheme: Optional[str]
) -> None:
    """Generate a single visualization."""
    service = VisualizationService(config)
    
    try:
        await service.start()
        
        with Progress() as progress:
            task = progress.add_task("Generating visualization...", total=1)
            
            result = await service.generate_single(
                sensor, start_time, end_time, interval, color_scheme
            )
            
            progress.update(task, completed=1)
        
        if result.success:
            console.print(f"[green]✓ Generated successfully![/green]")
            console.print(f"Output: {result.output_path}")
            console.print(f"Data points: {result.data_points}")
            console.print(f"Processing time: {result.processing_time_seconds:.2f}s")
        else:
            console.print(f"[red]✗ Generation failed: {result.error_message}[/red]")
            raise typer.Exit(1)
            
    finally:
        await service.stop()


async def _run_backfill(
    config: ServiceConfig,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    intervals: Optional[List[Interval]] = None
) -> None:
    """Run historical backfill."""
    service = VisualizationService(config)
    
    try:
        await service.start()
        
        console.print("Starting backfill process...")
        stats = await service.backfill_historical(start_date, end_date, intervals)
        
        # Display results
        table = Table(title="Backfill Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Date Range", f"{stats['start_date']} to {stats['end_date']}")
        table.add_row("Total Generated", str(stats['total_generated']))
        table.add_row("Successful", str(stats['successful']))
        table.add_row("Failed", str(stats['failed']))
        table.add_row("Skipped (Resume)", str(stats.get('skipped', 0)))
        table.add_row("Data Points", str(stats['total_data_points']))
        table.add_row("Processing Time", f"{stats['total_processing_time']:.2f}s")
        
        # Add progress summary if available
        if 'progress_summary' in stats:
            progress_summary = stats['progress_summary']
            table.add_row("Total Tracked", str(progress_summary['total_tracked']))
            
        console.print(table)
        
        if stats['failed'] > 0:
            console.print(f"[yellow]Warning: {stats['failed']} visualizations failed[/yellow]")
            
        if stats.get('skipped', 0) > 0:
            console.print(f"[blue]Info: {stats['skipped']} visualizations were skipped (already completed)[/blue]")
        
        # Show message if provided
        if 'message' in stats:
            console.print(f"[blue]{stats['message']}[/blue]")
            
    finally:
        await service.stop()


async def _show_progress(config: ServiceConfig) -> None:
    """Show backfill progress status."""
    service = VisualizationService(config)
    
    try:
        await service.start()
        
        # Get progress summary
        progress_summary = service.progress_tracker.get_progress_summary()
        
        table = Table(title="Backfill Progress Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Progress Log", str(config.progress_tracking.log_file))
        table.add_row("Log Exists", "Yes" if config.progress_tracking.log_file.exists() else "No")
        table.add_row("Resume Enabled", str(config.progress_tracking.resume_enabled))
        table.add_row("Check Files", str(config.progress_tracking.check_existing_files))
        table.add_row("Completed Tasks", str(progress_summary['completed_tasks']))
        table.add_row("Failed Tasks", str(progress_summary['failed_tasks']))
        table.add_row("Total Tracked", str(progress_summary['total_tracked']))
        
        console.print(table)
        
        if progress_summary['failed_tasks'] > 0:
            console.print(f"[yellow]Warning: {progress_summary['failed_tasks']} failed tasks in log[/yellow]")
            console.print("[dim]You may want to retry failed tasks by running backfill again[/dim]")
            
    finally:
        await service.stop()


async def _show_stats(config: ServiceConfig, sensor: Optional[SensorType]) -> None:
    """Show database statistics."""
    service = VisualizationService(config)
    
    try:
        await service.start()
        
        if sensor:
            # Show stats for specific sensor
            stats = await service.processor.get_data_statistics(sensor)
            
            table = Table(title=f"Statistics for {sensor.value}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Sensor Type", stats['sensor_type'])
            table.add_row("Data Points", str(stats['count']))
            table.add_row("Earliest", stats['earliest_timestamp'] or "No data")
            table.add_row("Latest", stats['latest_timestamp'] or "No data")
            table.add_row("Date Range (days)", str(stats['date_range_days']))
            
            console.print(table)
        else:
            # Show stats for all sensors
            table = Table(title="Database Statistics")
            table.add_column("Sensor", style="cyan")
            table.add_column("Data Points", style="green")
            table.add_column("Date Range (days)", style="yellow")
            
            for sensor_config in config.sensors:
                stats = await service.processor.get_data_statistics(sensor_config.type)
                table.add_row(
                    stats['sensor_type'],
                    str(stats['count']),
                    str(stats['date_range_days'])
                )
            
            console.print(table)
            
    finally:
        await service.stop()


async def _run_sync(config: ServiceConfig, full: bool, batch_size: int) -> None:
    """Run the sync operation."""
    from data.sync import DataSynchronizer
    
    try:
        synchronizer = DataSynchronizer(config.database)
        
        # Check sync status first
        status = await synchronizer.get_sync_status()
        console.print(f"Sync status: {status}")
        
        if not status.get('sync_enabled', False):
            console.print("[red]Sync is disabled - check Supabase credentials[/red]")
            return
        
        # Perform sync
        if full:
            console.print(f"[blue]Performing full sync with batch size {batch_size}[/blue]")
            result = await synchronizer.full_sync(batch_size)
        else:
            console.print("[blue]Performing incremental sync[/blue]")
            result = await synchronizer.sync_from_supabase()
        
        # Display results
        if "Success" in result.get("status", ""):
            console.print(f"[green]✓ {result['status']}[/green]")
            console.print(f"Records synced: {result.get('synced_records', 0)}")
            if result.get('skipped_records'):
                console.print(f"Records skipped: {result.get('skipped_records', 0)}")
        else:
            console.print(f"[red]✗ {result['status']}[/red]")
            
    except Exception as e:
        console.print(f"[red]Sync failed: {str(e)}[/red]")


async def _test_upload(config: ServiceConfig) -> None:
    """Test upload functionality."""
    from datetime import datetime, timedelta
    import tempfile
    from PIL import Image
    from upload.manager import UploadManager
    
    # Create a test image
    image = Image.new('RGB', (100, 100), color='red')
    temp_file = Path(tempfile.mktemp(suffix='.png'))
    image.save(temp_file, 'PNG')
    
    console.print(f"Created test image: {temp_file}")
    
    try:
        # Create mock visualization result
        from models.visualization import VisualizationRequest, VisualizationResult
        from models.sensor import SensorType
        
        mock_request = VisualizationRequest(
            sensor_type=SensorType.TEMPERATURE,
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            interval=Interval.DAILY,
            color_scheme="redblue"
        )
        
        mock_result = VisualizationResult(
            request=mock_request,
            output_path=temp_file,
            success=True,
            data_points=1440,
            processing_time_seconds=1.5
        )
        
        # Test upload
        upload_manager = UploadManager(config.upload)
        
        with Progress() as progress:
            task = progress.add_task("Uploading test image...", total=1)
            
            upload_result = await upload_manager.upload_visualization(mock_result)
            
            progress.update(task, completed=1)
        
        if upload_result.success:
            console.print("[green]✓ Upload test successful![/green]")
            console.print(f"CID: {upload_result.cid}")
            console.print(f"IPFS URL: {upload_result.ipfs_url}")
            console.print(f"KV Key: {upload_result.kv_key}")
            console.print(f"Local file deleted: {upload_result.local_file_deleted}")
        else:
            console.print("[red]✗ Upload test failed[/red]")
            console.print(f"Error: {upload_result.error_message}")
            if upload_result.pinata_error:
                console.print(f"Pinata Error: {upload_result.pinata_error}")
            if upload_result.cloudflare_error:
                console.print(f"Cloudflare Error: {upload_result.cloudflare_error}")
            
    finally:
        # Clean up test file if it still exists
        if temp_file.exists():
            temp_file.unlink()
            console.print("Cleaned up test file")


def _load_config(
    config_file: Optional[Path] = None,
    db_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    log_level: Optional[str] = None
) -> ServiceConfig:
    """Load service configuration."""
    # Start with default config
    config = ServiceConfig.create_default()
    
    # Override with CLI options
    if db_path:
        config.database.path = db_path
    if output_dir:
        config.output.base_dir = output_dir
    if log_level:
        config.log_level = log_level
    
    # TODO: Load from config file if provided
    
    return config


def _setup_logging(log_level: str, log_file: Optional[Path]) -> None:
    """Setup logging configuration."""
    import logging
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level)
    
    # TODO: Configure file logging if log_file is provided


def main() -> None:
    """Main entry point."""
    app() 