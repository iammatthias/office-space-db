# Office Space Environmental Monitoring System

A comprehensive environmental monitoring system that tracks office space conditions using multiple sensors and provides data visualization through a modern CLI interface.

## System Overview

This system consists of two main components:

1. **Sensor Service**: Collects environmental data from hardware sensors
2. **Visualization Service**: Processes and visualizes the collected data

## Sensors

- ICM20948: 9-DoF Motion Tracking
- BME280: Temperature, Humidity, Pressure
- LTR390: UV Light
- TSL25911: Ambient Light
- SGP40: VOC Air Quality

## Quick Start

### Prerequisites

- Python 3.11+
- Supabase account and credentials

### Installation

```bash
# Install system dependencies
pip3 install -r requirements.txt

# Install visualization service (recommended)
cd src/visualization_v2
uv sync

# Configure environment
cp config/.env.example config/.env
# Edit config/.env with your Supabase credentials
```

## Visualization CLI

The `viz-service` command provides a powerful interface for generating and managing environmental data visualizations.

### Available Commands

#### Generate Single Visualization

```bash
# Generate a single visualization for specific sensor and time range
viz-service generate SENSOR START_TIME END_TIME [OPTIONS]

# Examples:
viz-service generate temperature 2025-01-02T00:00:00 2025-01-03T00:00:00 --interval daily
viz-service generate humidity 2025-01-01T10:00:00 2025-01-01T11:00:00 --interval hourly
viz-service generate pressure 2025-01-01T00:00:00 2025-01-01T05:00:00 --interval cumulative
```

**Options:**

- `--interval, -i`: Time interval (daily, hourly, cumulative) [default: daily]
- `--output, -o`: Output file path
- `--db-path`: Database file path
- `--color`: Color scheme override

#### Historical Backfill with Resume Support

```bash
# Generate historical visualizations for date ranges
viz-service backfill [OPTIONS]

# Examples:
viz-service backfill --start-date 2025-01-01 --end-date 2025-01-31
viz-service backfill --sensor temperature --sensor humidity --interval daily
viz-service backfill --start-date 2025-01-15 --intervals daily hourly

# Resume functionality (automatically enabled)
viz-service backfill --start-date 2025-01-01 --end-date 2025-01-31  # Will skip completed ones
viz-service backfill --no-resume  # Disable resume, regenerate everything
```

**Options:**

- `--start-date`: Start date for backfill (ISO format)
- `--end-date`: End date for backfill (ISO format)
- `--sensor`: Specific sensors to process (repeatable)
- `--interval`: Specific intervals to generate (repeatable)
- `--db-path`: Database file path
- `--output-dir`: Output directory
- `--resume/--no-resume`: Enable/disable resume functionality [default: enabled]
- `--check-files/--no-check-files`: Check for existing output files [default: enabled]
- `--progress-log`: Custom path for progress log file

#### Progress Tracking

```bash
# Check backfill progress and status
viz-service progress [OPTIONS]

# Examples:
viz-service progress                           # Show current progress
viz-service progress --progress-log my-log.jsonl  # Use custom log file
```

**Options:**

- `--progress-log`: Path to progress log file
- `--db-path`: Database file path
- `--output-dir`: Output directory

#### Database Statistics

```bash
# View database statistics and data availability
viz-service stats [OPTIONS]

# Examples:
viz-service stats                    # All sensors
viz-service stats --sensor temperature  # Specific sensor
```

**Options:**

- `--sensor`: Show stats for specific sensor
- `--db-path`: Database file path

#### Data Synchronization

```bash
# Sync data from Supabase to local database
viz-service sync [OPTIONS]

# Examples:
viz-service sync                    # Incremental sync
viz-service sync --full            # Full sync
viz-service sync --full --batch-size 5000
```

**Options:**

- `--config, -c`: Configuration file path
- `--db-path`: Database file path
- `--full`: Perform full sync (vs incremental)
- `--batch-size`: Batch size for full sync [default: 10000]

#### Run Service (Continuous Mode)

```bash
# Run the visualization service continuously
viz-service run [OPTIONS]

# Example:
viz-service run --config config/viz-config.yaml --log-level DEBUG
```

**Options:**

- `--config, -c`: Configuration file path
- `--db-path`: Database file path
- `--output-dir`: Output directory for images
- `--log-level`: Logging level [default: INFO]

## Resume Functionality

The backfill command includes intelligent resume functionality to handle interruptions:

### Features

- **Automatic Progress Tracking**: Each completed visualization is logged with timestamp and status
- **File Existence Checking**: Verifies output files exist before attempting to regenerate
- **Smart Resume**: Automatically skips completed visualizations when rerunning backfill
- **Error Recovery**: Failed visualizations can be retried on subsequent runs
- **Progress Persistence**: Progress is saved to a JSON Lines log file for durability

### Progress Log Format

```jsonl
{"sensor_type": "temperature", "interval": "daily", "start_time": "2025-01-01T00:00:00", "end_time": "2025-01-02T00:00:00", "output_path": "/path/to/output.png", "completed_at": "2025-01-20T10:30:00Z", "success": true}
{"sensor_type": "humidity", "interval": "hourly", "start_time": "2025-01-01T10:00:00", "end_time": "2025-01-01T11:00:00", "output_path": "/path/to/output.png", "completed_at": "2025-01-20T10:31:00Z", "success": false, "error_message": "No data available"}
```

### Usage Examples

```bash
# Start a large backfill
viz-service backfill --start-date 2024-01-01 --end-date 2024-12-31

# If interrupted (Ctrl+C), simply run again - it will resume from where it left off
viz-service backfill --start-date 2024-01-01 --end-date 2024-12-31

# Check progress status
viz-service progress

# Force regeneration (disable resume)
viz-service backfill --no-resume --start-date 2024-01-01 --end-date 2024-01-31

# Use custom progress log location
viz-service backfill --progress-log /custom/path/backfill.jsonl --start-date 2024-01-01
```

### Supported Sensors

| Sensor        | Type     | Color Scheme | Description          |
| ------------- | -------- | ------------ | -------------------- |
| `temperature` | BME280   | Red-blue     | Temperature readings |
| `humidity`    | BME280   | Cyan         | Humidity percentage  |
| `pressure`    | BME280   | Green        | Atmospheric pressure |
| `light`       | TSL25911 | Base         | Ambient light levels |
| `uv`          | LTR390   | Purple       | UV light intensity   |
| `gas`         | SGP40    | Yellow       | VOC air quality      |

### Visualization Types

- **Daily**: Full day visualization (PST midnight-to-midnight)
- **Hourly**: Single hour positioned within day context
- **Cumulative**: Multi-day rows with hourly snapshots

### Timezone Handling

The system uses PST (Pacific Standard Time) for all user-facing operations:

- Input times are interpreted as PST
- Database queries automatically convert to UTC
- Output filenames use PST timestamps
- Visualizations respect PST day/hour boundaries

## Sensor Service Setup

### Installation (Raspberry Pi)

```bash
# Install service
sudo cp systemd/sensor-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sensor-service
sudo systemctl start sensor-service
```

### Development

```bash
# Run sensor service manually
python3 src/sensor_service.py
```

### Monitoring

```bash
# View logs
sudo journalctl -u sensor-service -f

# Check status
sudo systemctl status sensor-service
```

## Output Structure

Visualizations are organized in a clear directory structure:

```
output_dir/
├── temperature/
│   ├── daily/
│   ├── hourly/
│   └── cumulative/
├── humidity/
├── pressure/
├── light/
├── uv/
└── gas/
```

Images are generated as high-resolution PNG files (5760×6720 pixels) suitable for display and analysis.

## Advanced Usage

### Batch Processing with Resume

```bash
# Generate multiple sensor types for the same period
viz-service backfill \
  --start-date 2025-01-01 \
  --end-date 2025-01-07 \
  --sensor temperature \
  --sensor humidity \
  --sensor pressure \
  --interval daily \
  --interval hourly

# If this gets interrupted, running the same command again will resume
# from where it left off automatically
```

### Progress Management

```bash
# Check what's been completed
viz-service progress

# Run backfill with custom progress log
viz-service backfill \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --progress-log /path/to/my-backfill.jsonl

# View progress for custom log
viz-service progress --progress-log /path/to/my-backfill.jsonl
```

### Performance Optimization

The service includes several performance optimizations:

- LRU caching with 60%+ hit rates
- SQLite WAL mode with connection pooling
- Parallel batch processing
- Intelligent memory management
- Resume functionality prevents wasted regeneration

### Configuration Options

- Database path customization
- Output directory configuration
- Color scheme overrides
- Logging level control
- Timezone preference settings
- Progress log file location
- Resume behavior control

## Troubleshooting

### Common Issues

1. **Missing data**: Check sync status with `viz-service stats`
2. **Permission errors**: Ensure output directory is writable
3. **Database issues**: Verify database path and permissions
4. **Sync problems**: Check Supabase credentials in `.env`
5. **Interrupted backfill**: Use `viz-service progress` to check status, then rerun the same backfill command

### Debugging

```bash
# Enable debug logging
viz-service run --log-level DEBUG

# Check data availability
viz-service stats --sensor temperature

# Test single visualization
viz-service generate temperature 2025-01-01T00:00:00 2025-01-02T00:00:00

# Check backfill progress
viz-service progress

# Force regeneration if needed
viz-service backfill --no-resume --start-date 2025-01-01 --end-date 2025-01-02
```

### Resume Functionality Issues

```bash
# Progress log corrupted or missing
viz-service progress  # Check status
rm backfill_progress.jsonl  # Remove if needed
viz-service backfill --start-date 2025-01-01  # Start fresh

# Files exist but not in progress log
viz-service backfill --check-files  # Will detect existing files

# Disable resume completely
viz-service backfill --no-resume --no-check-files
```
