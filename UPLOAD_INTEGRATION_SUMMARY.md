# Upload Integration Summary

This document summarizes the new upload functionality added to the visualization service for automatic uploading to Pinata IPFS and Cloudflare KV storage.

## Features Added

### 1. Configuration System

- **Environment-based Configuration**: Automatically detects and enables uploads when required env vars are set
- **Flexible Settings**: Optional group IDs, configurable cleanup behavior, and service URLs
- **Validation**: Comprehensive validation of configuration before attempting operations

### 2. Pinata IPFS Integration

- **File Upload**: Upload PNG images to Pinata IPFS storage
- **Async Operations**: Non-blocking uploads with proper timeout handling
- **Error Handling**: Detailed error reporting for debugging
- **Batch Support**: Concurrent uploads with semaphore-based rate limiting

### 3. Cloudflare KV Integration

- **Structured Storage**: Organized key-value storage for image metadata
- **Metadata Rich**: Stores IPFS URLs with detailed visualization metadata
- **Key Convention**: Structured keys like `temperature:daily:latest`
- **JSON Values**: Complete metadata including processing time, data points, etc.

### 4. Upload Manager

- **Orchestrated Workflow**: Manages the complete upload process (Pinata → Cloudflare KV → Cleanup)
- **Atomic Operations**: Ensures consistency between IPFS and KV storage
- **Batch Processing**: Efficient batch uploads for multiple files
- **Configurable Cleanup**: Optional deletion of local files after successful upload

### 5. Service Integration

- **Automatic Integration**: Seamlessly integrated into existing visualization workflow
- **Single & Batch**: Works with both individual and batch visualization generation
- **Error Isolation**: Upload failures don't affect visualization generation
- **Comprehensive Logging**: Detailed logging at all stages

## File Structure

```
src/visualization_v2/
├── upload/
│   ├── __init__.py          # Module exports
│   ├── pinata.py            # Pinata IPFS client
│   ├── cloudflare_kv.py     # Cloudflare KV client
│   ├── manager.py           # Upload orchestration
│   └── demo.py              # Standalone test script
├── models/
│   └── config.py            # Updated with upload configuration
├── core/
│   └── service.py           # Updated with upload integration
├── cli/
│   └── main.py              # Added test-upload command
└── pyproject.toml           # Updated dependencies
```

## Environment Variables

Set these environment variables to enable upload functionality:

```bash
# Required for Pinata IPFS
PINATA_JWT=your_pinata_jwt_token

# Required for Cloudflare KV
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
KV_NAMESPACE_ID=your_kv_namespace_id

# Optional settings
PINATA_GROUP_ID=your_pinata_group_id        # Optional
DELETE_LOCAL_FILES=true                     # Default: true
```

## Usage Examples

### 1. Test Upload Functionality

```bash
# Set your environment variables first
export PINATA_JWT="your_jwt_here"
export CLOUDFLARE_API_TOKEN="your_token_here"
export CLOUDFLARE_ACCOUNT_ID="your_account_id_here"
export KV_NAMESPACE_ID="your_namespace_id_here"

# Test upload with CLI command
cd src/visualization_v2
uv run viz-service test-upload
```

### 2. Generate with Automatic Upload

```bash
# Single visualization with upload
uv run viz-service generate temperature 2025-01-15T00:00:00 2025-01-16T00:00:00 --interval daily

# Batch backfill with upload
uv run viz-service backfill --start-date 2025-01-01 --end-date 2025-01-31
```

### 3. Standalone Test Script

```bash
# Run the demo script directly
cd src/visualization_v2
python upload/demo.py
```

## Cloudflare KV Structure

### Key Format

Keys follow the pattern: `{sensor_type}:{interval}:latest`

Examples:

- `temperature:daily:latest`
- `humidity:hourly:latest`
- `pressure:cumulative:latest`

### Value Format

Values are JSON objects containing:

```json
{
  "ipfs_url": "ipfs://QmHash123...",
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

## API Examples

### Upload a Single File

```python
from upload.manager import UploadManager
from models.config import UploadConfig

# Initialize with environment configuration
upload_config = UploadConfig.from_env()
upload_manager = UploadManager(upload_config)

# Upload a visualization result
upload_result = await upload_manager.upload_visualization(viz_result)

if upload_result.success:
    print(f"IPFS URL: {upload_result.ipfs_url}")
    print(f"KV Key: {upload_result.kv_key}")
```

### Batch Upload

```python
# Upload multiple results
results = [viz_result1, viz_result2, viz_result3]
upload_results = await upload_manager.upload_multiple(results, max_concurrent=3)

successful = sum(1 for r in upload_results if r.success)
print(f"Successfully uploaded: {successful}/{len(results)}")
```

### Direct Pinata Upload

```python
from upload.pinata import PinataUploader
from models.config import PinataConfig

async with PinataUploader(PinataConfig.from_env()) as pinata:
    result = await pinata.upload_file(Path("image.png"))
    if result.success:
        print(f"CID: {result.cid}")
```

### Direct Cloudflare KV Operations

```python
from upload.cloudflare_kv import CloudflareKVClient
from models.config import CloudflareKVConfig
from models.sensor import SensorType
from models.visualization import Interval

async with CloudflareKVClient(CloudflareKVConfig.from_env()) as kv:
    # Store latest image
    result = await kv.set_latest_image(
        SensorType.TEMPERATURE,
        Interval.DAILY,
        "ipfs://QmHash123...",
        {"filename": "test.png"}
    )

    # Retrieve latest image
    get_result = await kv.get_latest_image(
        SensorType.TEMPERATURE,
        Interval.DAILY
    )
```

## Error Handling

The system includes comprehensive error handling:

- **Network Errors**: HTTP timeouts, connection failures
- **API Errors**: Invalid tokens, quota limits, service unavailable
- **File Errors**: Missing files, permission issues
- **Configuration Errors**: Missing environment variables, invalid settings

All errors are logged with structured data for debugging.

## Performance Features

- **Concurrent Uploads**: Configurable concurrency limits for batch operations
- **Timeout Management**: Appropriate timeouts for different operations
- **Async Operations**: Non-blocking I/O throughout the system
- **Connection Pooling**: Efficient HTTP client usage
- **Error Recovery**: Graceful handling of temporary failures

## Dependencies Added

- `requests>=2.31.0` - HTTP client for API calls
- `cloudflare>=3.0.0` - Cloudflare API client
- `httpx>=0.27.0` - Modern async HTTP client

## Testing

The integration includes multiple testing approaches:

1. **CLI Test Command**: `uv run viz-service test-upload`
2. **Demo Script**: `python upload/demo.py`
3. **Integration Tests**: Automatic upload during normal visualization generation

## Logging

All upload operations are logged with structured data:

- **Debug Level**: Detailed operation steps
- **Info Level**: Success confirmations with key details
- **Warning Level**: Non-critical failures
- **Error Level**: Critical failures with full error details

## Security Considerations

- Environment variables for sensitive tokens
- No token storage in code or logs
- Secure HTTP connections (HTTPS)
- Configurable service endpoints

## Next Steps

Consider these enhancements for the future:

1. **Retry Logic**: Automatic retry for transient failures
2. **Progress Tracking**: Upload progress for large files
3. **Compression**: Optional image compression before upload
4. **Webhooks**: Notification system for upload events
5. **Metrics**: Upload success/failure metrics collection

This integration provides a robust, production-ready system for automatically uploading visualization images to distributed storage while maintaining all existing functionality.
