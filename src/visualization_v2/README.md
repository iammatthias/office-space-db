# Visualization Service V2

A modular, efficient environmental data visualization service built with modern Python practices.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data access, image generation, and service orchestration
- **Efficient Processing**: Optimized for handling large datasets with intelligent caching and batching
- **Multiple Intervals**: Generates cumulative (multi-day), hourly, and daily visualizations
- **Long-running Service**: Designed for continuous background operation with automatic data updates
- **External Upload**: Automatic upload to Pinata IPFS with Cloudflare KV indexing
- **Local Development**: Outputs images locally for testing (upload integration available but optional)
- **Timezone Handling**: Proper PST-based visualization with UTC database queries

## Architecture

```
visualization_v2/
├── core/           # Core business logic and service orchestration
│   ├── service.py       # Main visualization service
│   ├── processor.py     # Data processing and caching
│   └── timezone_utils.py # Centralized timezone handling
├── data/           # Data access layer with caching and repository patterns
├── generators/     # Image generation with multiple visualization types
├── models/         # Type-safe data models and configuration
├── upload/         # External upload services (Pinata IPFS + Cloudflare KV)
└── cli/            # Command line interface with rich output
```

## Upload Integration

The service now includes automatic upload functionality for generated visualizations:

### Features

- **Pinata IPFS**: Uploads images to decentralized IPFS storage
- **Cloudflare KV**: Stores IPFS CIDs with metadata for fast retrieval
- **Automatic Cleanup**: Optionally deletes local files after successful upload
- **Batch Operations**: Efficient batch uploads with concurrency control
- **Error Handling**: Comprehensive error handling and retry logic

### Configuration

Set the following environment variables to enable uploads:

```bash
# Pinata Configuration
PINATA_JWT=your_pinata_jwt_token
PINATA_GROUP_ID=your_pinata_group_id  # Optional

# Cloudflare Configuration
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
KV_NAMESPACE_ID=your_kv_namespace_id

# Upload Settings (optional)
DELETE_LOCAL_FILES=true  # Delete local files after upload (default: true)
```

### KV Structure

Images are stored in Cloudflare KV with keys like:

- `temperature:daily:latest`
- `humidity:hourly:latest`
- `pressure:cumulative:latest`

Each key contains:

```json
{
  "ipfs_url": "ipfs://QmHash...",
  "updated_at": "1640995200",
  "metadata": {
    "filename": "2025-01-15_daily.png",
    "sensor_type": "temperature",
    "interval": "daily",
    "data_points": 1440,
    "processing_time": 1.23,
    "start_time": "2025-01-15T00:00:00-08:00",
    "end_time": "2025-01-16T00:00:00-08:00"
  }
}
```

### Testing Upload Functionality

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
vim .env

# Run upload demo
cd src/visualization_v2
python upload/demo.py
```

## Timezone Handling

The service correctly handles timezone conversions between PST (sensor local time) and UTC (database storage):

- **Input**: User requests are assumed to be PST (Pacific Standard Time)
- **Database**: All sensor data is stored in UTC
- **Output**: Visualizations are PST-based for proper day/hour boundaries

### Key Behaviors

- **Daily**: Represents PST midnight-to-midnight (24-hour period in local sensor time)
- **Hourly**: 1-hour slice correctly positioned within PST day boundaries
- **Cumulative**: Hourly snapshots (not 5-minute) with new rows for each PST day

## Installation

```bash
# Install with upload dependencies
cd src/visualization_v2
uv sync

# Copy environment template
cp .env.example .env

# Configure your environment variables
vim .env
```

## Quick Start

```bash
# Generate single visualization (with automatic upload if configured)
uv run viz-service generate temperature 2025-01-15T00:00:00 2025-01-16T00:00:00

# Historical backfill with upload
uv run viz-service backfill --start-date 2025-01-01 --end-date 2025-01-31

# Run continuous service
uv run viz-service run
```

## Usage

### Command Line Interface

```

```
