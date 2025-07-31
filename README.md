# TVHeadend M3U Sync - Python Version

A Python tool to synchronize M3U playlist files with TVHeadend server with improved features.

## Key Features

### 🔄 **Smart Channel Synchronization - Preserves UUIDs!**
- **Channels and muxes are UPDATED, not recreated** - existing channel UUIDs are preserved
- **No channel duplication** - updates existing channels instead of creating new ones
- **Seamless re-sync** - run multiple times without losing channel configurations

### 📡 **Complete M3U Management**
- Parse M3U playlist files with robust error handling
- **Preserves ALL original M3U content** - custom lines, comments, and EXT-X directives remain untouched
- Only adds/updates TVH-UUID tags without discarding any existing content
- Create, update, and delete muxes based on M3U content
- Automatic backup creation before modifications
- **Environment variable support** for secure credential management
- **Configuration file support** for easy deployment
- **Improved error handling** with retry logic and timeouts
- **Enhanced logging** with structured output and sensitive data masking
- **Graceful shutdown** handling

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Environment Variables (Recommended)
```bash
# Set secure configuration via environment variables
export TVH_URL="http://localhost:9981"
export TVH_USERNAME="admin"
export TVH_PASSWORD="your_password"
export TVH_NETWORK="IPTV Network"
export TVH_M3U_FILE="/path/to/playlist.m3u"

# Run sync
python tvheadend_m3u_sync.py sync
```

### Command Line Interface
```bash
python tvheadend_m3u_sync.py [options]
```

**Main Options (required):**
- `-m, --m3u-file <file>` - Path to M3U playlist file (or use TVH_M3U_FILE env var)
- `-n, --network-name <name>` - TVHeadend network name (or use TVH_NETWORK env var, auto-generated from M3U filename if not provided)
- `--url <url>` - TVHeadend server URL (or use TVH_URL env var)

**Authentication Options:**
- `-u, --username` - Username (prefer TVH_USERNAME env var)
- `-p, --password` - Password (prefer TVH_PASSWORD env var)
- `--basic-auth` - Use basic auth instead of digest

**Global Options:**
- `-v, --verbose` - Enable debug logging
- `--version` - Show version information
- `--config <file>` - Use configuration file
- `--log-file <file>` - Log to file instead of console
- `--json-log` - Use JSON log format for structured logging
- `--timeout <seconds>` - HTTP request timeout (default: 30)
- `--dry-run` - Preview changes without applying them



### Configuration Methods (Priority Order)

1. **Environment Variables** (Most Secure)
2. **Configuration File** 
3. **Command Line Arguments** (Least Secure for passwords)

## Examples

### Environment Variables (Production Recommended)
```bash
# Set configuration securely
export TVH_URL="http://192.168.1.100:9981"
export TVH_USERNAME="admin"
export TVH_PASSWORD="password123"
export TVH_NETWORK="IPTV Network"
export TVH_M3U_FILE="/app/data/playlist.m3u"

# Sync with verbose logging
python tvheadend_m3u_sync.py --verbose

# Dry run preview
python tvheadend_m3u_sync.py --dry-run --verbose
```

### Configuration File
```bash
# Create config.ini (copy from config.ini.example)
python tvheadend_m3u_sync.py --config config.ini -m playlist.m3u -n "IPTV Network"
```

### Command Line (Development)
```bash
# Full sync with digest auth (flexible argument order)
python tvheadend_m3u_sync.py -m playlist.m3u -n "IPTV Network" --url http://192.168.1.100:9981 -u admin -p password123 -v

# Alternative order - arguments can be in any order
python tvheadend_m3u_sync.py --url http://192.168.1.100:9981 -u admin -m playlist.m3u -n "IPTV Network" -p password123

# Auto-generated network name from M3U filename
python tvheadend_m3u_sync.py -m my_channels.m3u --url http://tvheadend:9981  # Network: "my channels"

# Sync with structured JSON logging
python tvheadend_m3u_sync.py -m playlist.m3u -n "IPTV Network" --json-log --log-file sync.log


```



## Configuration

### Environment Variables
```bash
TVH_URL          # TVHeadend server URL
TVH_USERNAME     # Username for authentication  
TVH_PASSWORD     # Password for authentication
TVH_NETWORK      # Network name for IPTV channels
TVH_M3U_FILE     # Path to M3U playlist file
TVH_AUTH_TYPE    # "digest" (default) or "basic"
```

### Configuration File (config.ini)
```ini
[tvheadend]
url = http://localhost:9981
username = admin
password = your_password_here
network_name = IPTV Network
m3u_file = /path/to/playlist.m3u
auth_type = digest

[logging]
level = INFO
file = /var/log/tvheadend_m3u_sync.log
json_format = false

[network]
timeout = 30
max_retries = 3
```

### Authentication Methods
1. **Digest Authentication (Default)**: More secure, recommended
2. **Basic Authentication**: Legacy method, less secure

### Security Best Practices
- ✅ Use environment variables for credentials
- ✅ Use configuration files for non-sensitive settings
- ❌ Avoid passwords in command line arguments (visible in process list)

## Logging

### Standard Logging
```
2025-07-15 10:30:15 - tvheadend_m3u_sync - INFO - Starting TVHeadend M3U Sync v1.0.0
2025-07-15 10:30:15 - tvheadend_m3u_sync - INFO - Found 150 entries in M3U file
2025-07-15 10:30:16 - tvheadend_m3u_sync - INFO - Using network: IPTV Network (UUID: abc123)
2025-07-15 10:30:17 - tvheadend_m3u_sync - INFO - Synchronization completed successfully
```

### JSON Structured Logging
```bash
python tvheadend_m3u_sync.py --json-log
```
```json
{"timestamp": "2024-01-15T10:30:15.123456", "level": "INFO", "logger": "tvheadend_m3u_sync", "message": "Starting sync", "module": "tvheadend_m3u_sync", "function": "sync_mode", "line": 123}
```

### Log Levels
- **INFO**: Important operations and results
- **DEBUG**: Detailed operation steps (use `-v`)
- **WARNING**: Non-critical issues  
- **ERROR**: Critical errors that stop execution

### Security Features
- Automatic masking of sensitive data (passwords, tokens)
- Safe logging of authentication without exposing credentials

## Project Structure

```
.
├── tvheadend_m3u_sync.py     # Main application
├── requirements.txt          # Dependencies  
├── config.ini.example        # Example configuration
└── README.md                 # This documentation
```

## M3U File Format Support

The script processes M3U files for TVHeadend compatibility:

**✅ Processed (EXTINF entries):**
```
#EXTINF:-1 TVH-UUID="abc123", CHANNEL NAME
#EXT-X-STREAM-INF:BANDWIDTH=7830000,CODECS="h264,aac,eac3"
pipe://ffmpeg
```

**How it works:**
1. **EXTINF line** → Extracts channel name and TVH-UUID
2. **EXT-X-STREAM-INF** → Completely ignored
3. **Next non-comment line** → Used as URL/command for the channel

**❌ Ignored (EXT-X directives):**
```
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:10
#EXT-X-STREAM-INF:... (ignored, not converted)
```

**Supported URLs/Commands:**
- HTTP/HTTPS streams: `http://example.com/stream`
- Pipe commands: `pipe://ffmpeg`
- RTMP/RTSP: `rtmp://server/stream`
- File paths: `/path/to/file.ts`

## Notes

- TVH-UUID tags are added to M3U entries for tracking
- Old muxes not found in the M3U file will be deleted from TVHeadend
- Authentication supports Digest Auth (secure) and Basic Auth (legacy)
- **Single file** - no import problems, easy to distribute
- **Professional logging** with timestamps and levels
- **argparse** for proper command-line argument handling
- **Simplified authentication** - username/password with digest or basic auth
- **Error handling** with meaningful messages and exit codes
- **Smart M3U parsing** - only processes EXTINF entries, ignores EXT-X-* directives
- **Dry run mode** - test changes before applying them
- **User-Agent** - proper HTTP identification

## Troubleshooting

### Debug Mode
```bash
python tvheadend_m3u_sync.py --verbose --log-file debug.log
```

### Common Issues

**Authentication Errors**
```bash
# Check credentials and use environment variables
export TVH_USERNAME="your_username"
export TVH_PASSWORD="your_password"
python tvheadend_m3u_sync.py --verbose
```

**Timeout Issues**
```bash
# Increase timeout for slow networks
python tvheadend_m3u_sync.py --timeout 60
```

**Configuration Problems**
```bash
# Validate configuration
python tvheadend_m3u_sync.py --config config.ini --dry-run
```



### Error Codes
- `0`: Success
- `1`: General error
- `128 + N`: Killed by signal N (e.g., 130 for Ctrl+C)

### Health Checks
The application includes graceful shutdown handling for reliable operation.

## Credits

This Python version is based on the original [.NET C# implementation by hagaygo](https://github.com/hagaygo/tvheadendm3usync). 

### v1.0 Enhancements by Claude (Anthropic)
- 🔒 **Security**: Environment variables, credential masking
- ⚙️ **Configuration**: Config file support, validation  
- 📊 **Logging**: Structured logging, JSON format, error handling
- 🔄 **Reliability**: Retry logic, graceful shutdown, timeout handling

## License

This project is licensed under the WTFPL (Do What The Fuck You Want To Public License) Version 2.

See the [LICENSE](LICENSE) file for details.

**TL;DR**: You can do whatever you want with this code. No restrictions, no warranties, no bullshit.
