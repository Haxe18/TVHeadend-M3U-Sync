#!/usr/bin/env python3
"""
TVHeadend M3U Sync Tool

Synchronizes M3U playlist files with TVHeadend server.
Python conversion of the original .NET C# implementation by hagaygo.

Original project: https://github.com/hagaygo/tvheadendm3usync
Python conversion by Claude (Anthropic's AI assistant)

License: WTFPL (Do What The Fuck You Want To Public License) Version 2
"""

import sys
import os
import json
import base64
import shutil
import logging
import argparse
import signal
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple, Union
from urllib.parse import urljoin, urlparse
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import configparser

__version__ = "1.0.0"

# Global flag for graceful shutdown
_shutdown_requested = False


# ==================== SIGNAL HANDLING ====================

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    global _shutdown_requested
    logger = logging.getLogger('tvheadend_m3u_sync')
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True


def check_shutdown():
    """Check if shutdown was requested"""
    if _shutdown_requested:
        logger = logging.getLogger('tvheadend_m3u_sync')
        logger.info("Shutdown requested, exiting...")
        sys.exit(128 + signal.SIGTERM)


# ==================== CONFIGURATION MANAGEMENT ====================

class Config:
    """Configuration manager for sensitive data and settings"""

    def __init__(self):
        self.config_file = None
        self.settings = {}

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, str]:
        """Load configuration from file or environment variables"""
        config = {}

        # Try to load from config file
        if config_path and os.path.exists(config_path):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            if 'tvheadend' in parser:
                config.update(dict(parser['tvheadend']))

        # Override with environment variables (more secure)
        env_mappings = {
            'TVH_URL': 'url',
            'TVH_USERNAME': 'username',
            'TVH_PASSWORD': 'password',
            'TVH_NETWORK': 'network_name',
            'TVH_M3U_FILE': 'm3u_file',
            'TVH_AUTH_TYPE': 'auth_type'
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                config[config_key] = value

        self.settings = config
        return config

    def get(self, key: str, default: str = None) -> Optional[str]:
        """Get configuration value"""
        return self.settings.get(key, default)

    def mask_sensitive_data(self, data: str) -> str:
        """Mask sensitive information for logging"""
        if not data or len(data) < 4:
            return "***"
        return data[:2] + "*" * (len(data) - 4) + data[-2:]


# ==================== LOGGING SETUP ====================

def setup_logging(verbose: bool = False, log_file: Optional[str] = None, json_format: bool = False):
    """Setup logging configuration with improved formatting and options"""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    if json_format:
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_entry)
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup file handler if requested
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        logger.addHandler(handler)

    # Reduce noise from external libraries
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


# ==================== DATA MODELS ====================

@dataclass
class ModelBase:
    """Base model class for TVHeadEnd objects"""
    uuid: Optional[str] = None
    name: Optional[str] = None
    enabled: bool = False


@dataclass
class Entry:
    """M3U playlist entry"""
    url: Optional[str] = None
    xtinf: Optional[str] = None
    name: Optional[str] = None
    tvh_uuid: Optional[str] = None


@dataclass
class Network(ModelBase):
    """TVHeadEnd network configuration"""
    pass


@dataclass
class Mux(ModelBase):
    """TVHeadEnd mux configuration"""
    network_name: Optional[str] = None
    network_uuid: Optional[str] = None
    url: Optional[str] = None


@dataclass
class Service(ModelBase):
    """TVHeadEnd service configuration"""
    pass


# ==================== M3U PARSER ====================

class Parser:
    """M3U playlist parser"""

    HEADER = "#EXTM3U"
    TVH_UUID_KEY = "TVH-UUID"
    TVH_UUID = f'{TVH_UUID_KEY}="'

    @staticmethod
    def get_entries(file_path: str) -> List[Entry]:
        """Parse M3U file and return list of entries"""
        logger = logging.getLogger('tvheadend_m3u_sync')
        logger.debug(f"Parsing M3U file: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            header = file.readline().strip()
            if header != Parser.HEADER:
                raise ValueError("Expected #EXTM3U as file header")

            entries = []
            last_xtinf = None

            # Statistics for logging
            extinf_count = 0
            extx_count = 0
            skipped_lines = 0

            for line_num, line in enumerate(file, start=2):  # Start at 2 because we already read header
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Handle #EXTINF lines (standard M3U entries)
                if line.startswith("#EXTINF:"):
                    last_xtinf = line
                    extinf_count += 1
                    logger.debug(f"Found EXTINF at line {line_num}: {line[:50]}...")

                # Skip ALL #EXT-X-* directives (including EXT-X-STREAM-INF)
                elif line.startswith("#EXT-X-"):
                    extx_count += 1
                    if line.startswith("#EXT-X-STREAM-INF"):
                        logger.debug(f"Ignoring EXT-X-STREAM-INF at line {line_num} (using previous EXTINF)")
                    else:
                        logger.debug(f"Skipping EXT-X directive at line {line_num}: {line[:30]}...")
                    continue

                # Skip other comments
                elif line.startswith("#"):
                    skipped_lines += 1
                    logger.debug(f"Skipping comment at line {line_num}: {line[:50]}...")
                    continue

                # Process URL/command lines (only if we have a preceding EXTINF)
                elif last_xtinf is not None and line:
                    # Accept any non-comment line as URL/command
                    idx = last_xtinf.rfind(",")
                    if idx > 0:
                        name = last_xtinf[idx + 1:].strip()
                        tvh_uuid = Parser._get_tvh_uuid(last_xtinf)

                        entry = Entry(
                            url=line,
                            xtinf=last_xtinf,
                            name=name,
                            tvh_uuid=tvh_uuid
                        )
                        entries.append(entry)
                        logger.debug(f"Added entry: {entry.name} -> {entry.url[:50]}...")
                    else:
                        logger.warning(f"Invalid EXTINF format at line {line_num-1}: {last_xtinf}")

                    last_xtinf = None

                # URL without preceding EXTINF
                elif line and not line.startswith("#"):
                    logger.warning(f"Found URL without EXTINF at line {line_num}: {line[:50]}...")
                    skipped_lines += 1

            # Log parsing summary
            logger.info(f"M3U parsing complete: {len(entries)} EXTINF entries processed for TVHeadEnd")
            if extx_count > 0:
                logger.debug(f"Ignored {extx_count} EXT-X directives (not processed)")
            if skipped_lines > 0:
                logger.debug(f"Skipped {skipped_lines} other lines")

            return entries

    @staticmethod
    def _get_tvh_uuid(xtinf: str) -> str:
        """Extract TVH UUID from XTINF line"""
        idx = xtinf.find(Parser.TVH_UUID)
        if idx > 0:
            uuid_part = xtinf[idx + len(Parser.TVH_UUID):]
            uuid = uuid_part.split('"')[0]
            return uuid
        return None

    @staticmethod
    def parse_tags(tags: str) -> List[Tuple[str, str]]:
        """Parse tag string into list of key-value pairs"""
        result = []
        while "=" in tags:
            try:
                key = tags[:tags.index("=")]
                tags = tags[len(key) + 2:]  # skip = and "

                value = tags[:tags.index('"')]
                tags = tags[len(value) + 1:].strip()

                result.append((key, value))
            except (ValueError, IndexError):
                break  # Malformed tag, stop parsing

        return result

    @staticmethod
    def write_file(m3u_file: str, entries: List[Entry], dry_run: bool = False):
        """Write entries back to M3U file with backup, preserving all original content"""
        logger = logging.getLogger('tvheadend_m3u_sync')

        if dry_run:
            logger.info(f"[DRY RUN] Would update M3U file: {m3u_file}")
            logger.debug(f"[DRY RUN] Would update UUIDs for {len([e for e in entries if e.tvh_uuid])} entries")
            for entry in entries:
                if entry.tvh_uuid:
                    logger.debug(f"[DRY RUN] Would update UUID for: {entry.name} -> {entry.tvh_uuid}")
            return

        # Create backup
        backup_name = f"{m3u_file}.backup{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.debug(f"Creating backup: {backup_name}")
        shutil.copy2(m3u_file, backup_name)

        # Read original file line by line and preserve all content
        logger.debug(f"Preserving original content and updating UUIDs for {len(entries)} entries")
        
        with open(m3u_file, 'r', encoding='utf-8') as original_file:
            original_lines = original_file.readlines()

        # Process the file, updating only EXTINF lines that need UUID updates
        with open(m3u_file, 'w', encoding='utf-8') as file:
            i = 0
            while i < len(original_lines):
                line = original_lines[i].strip()
                
                # Check if this is an EXTINF line that matches one of our entries
                if line.startswith("#EXTINF:"):
                    # Find matching entry
                    matching_entry = None
                    for entry in entries:
                        if entry.xtinf.strip() == line:
                            matching_entry = entry
                            break
                    
                    # If we found a matching entry and it has a UUID to add/update
                    if matching_entry and matching_entry.tvh_uuid:
                        idx = line.rfind(",")
                        if idx > 0:
                            x = line[:idx]
                            current_uuid = Parser._get_tvh_uuid(line)
                            
                            # Add or update UUID
                            if current_uuid is None:
                                x = f'{x} {Parser.TVH_UUID}{matching_entry.tvh_uuid}"'
                            else:
                                x = x.replace(f'{Parser.TVH_UUID}{current_uuid}"', f'{Parser.TVH_UUID}{matching_entry.tvh_uuid}"')
                            
                            updated_line = f"{x},{matching_entry.name}\n"
                            file.write(updated_line)
                        else:
                            # Malformed EXTINF, write as-is
                            file.write(original_lines[i])
                    else:
                        # No matching entry or no UUID to update, write original line
                        file.write(original_lines[i])
                else:
                    # Not an EXTINF line, write as-is (preserves comments, EXT-X directives, URLs, etc.)
                    file.write(original_lines[i])
                
                i += 1

        logger.info(f"M3U file updated: {m3u_file} (all original content preserved)")





# ==================== TVHEADEND CLIENT ====================

class AuthenticationType(Enum):
    BASIC = "basic"
    DIGEST = "digest"


# Removed GridRequestParameters - using dict directly


class TVHClient:
    """TVHeadend HTTP API client with improved error handling and retry logic"""

    def __init__(self, base_url: str, dry_run: bool = False, timeout: int = 30):
        self.base_url = base_url
        self.username = ""
        self.password = ""
        self.authentication_type = AuthenticationType.DIGEST
        self.dry_run = dry_run
        self.timeout = timeout
        self.session = requests.Session()

        # Set custom User-Agent
        self.session.headers.update({
            'User-Agent': f'TVHeadendM3USync/{__version__}'
        })

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _get_auth_header(self) -> Dict[str, str]:
        """Generate authentication header - for digest auth, this is handled by requests"""
        if self.authentication_type == AuthenticationType.BASIC:
            credentials = f"{self.username}:{self.password}"
            encoded_credentials = base64.b64encode(credentials.encode('iso-8859-1')).decode('ascii')
            return {"Authorization": f"Basic {encoded_credentials}"}
        elif self.authentication_type == AuthenticationType.DIGEST:
            # Digest auth doesn't use authorization header
            return {}
        else:
            raise NotImplementedError("Authentication type not implemented")

    def _create_params(self, start: int = 0, limit: int = None) -> Dict[str, str]:
        """Create request parameters"""
        params = {"start": str(start)}
        if limit is not None:
            params["limit"] = str(limit)
        return params

    def _post_request(self, url_path: str, data: Dict[str, str] = None) -> requests.Response:
        """Make POST request to TVHeadend API with improved error handling"""
        url = urljoin(self.base_url, url_path)
        headers = self._get_auth_header()
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        # Setup authentication
        auth = None
        if self.authentication_type == AuthenticationType.DIGEST and self.username:
            from requests.auth import HTTPDigestAuth
            auth = HTTPDigestAuth(self.username, self.password)

        logger = logging.getLogger('tvheadend_m3u_sync')

        # Mask sensitive information for logging
        config = Config()
        masked_headers = {k: config.mask_sensitive_data(v) if 'auth' in k.lower() else v 
                         for k, v in headers.items()}

        logger.debug(f"POST request to: {url}")
        logger.debug(f"Auth type: {self.authentication_type.value}")
        logger.debug(f"Headers: {masked_headers}")
        if data and logger.isEnabledFor(logging.DEBUG):
            # Mask sensitive data in request body
            masked_data = {k: config.mask_sensitive_data(str(v)) if any(sensitive in k.lower() 
                              for sensitive in ['password', 'token', 'auth']) else v 
                          for k, v in data.items()}
            logger.debug(f"Data: {masked_data}")

        try:
            check_shutdown()  # Check for graceful shutdown
            response = self.session.post(url, data=data, headers=headers, auth=auth, timeout=self.timeout)
            return response
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout} seconds to {url}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed to {url}: {e}")
            raise

    def _get_json(self, response: requests.Response) -> Dict:
        """Parse JSON response"""
        response.raise_for_status()
        return response.json()

    def create_new_iptv_network(self, name: str):
        """Create new IPTV network"""
        logger = logging.getLogger('tvheadend_m3u_sync')

        if self.dry_run:
            logger.info(f"[DRY RUN] Would create new IPTV network: {name}")
            return

        conf_data = {
            "enabled": True,
            "max_timeout": 10,
            "networkname": name,
            "pnetworkname": name,
            "bouquet": True,
            "scan_create": 1,
            "max_streams": 0
        }

        data = {
            "class": "iptv_network",
            "conf": json.dumps(conf_data)
        }

        response = self._post_request("api/mpegts/network/create", data)
        response.raise_for_status()
        logger.info(f"Created new IPTV network: {name}")

    def add_mux(self, network: Network, entry: Entry) -> str:
        """Add new mux to network"""
        logger = logging.getLogger('tvheadend_m3u_sync')

        if self.dry_run:
            fake_uuid = f"dry-run-uuid-{hash(entry.url) % 100000}"
            logger.info(f"[DRY RUN] Would create new mux: {entry.name} -> {entry.url[:50]}...")
            logger.debug(f"[DRY RUN] Would assign UUID: {fake_uuid}")
            return fake_uuid
        conf_data = {
            "Enabled": True, "epg": False, "iptv_url": entry.url, "iptv_muxname": entry.name, "scan_state": 0
        }

        data = {
            "uuid": network.uuid,
            "conf": json.dumps(conf_data)
        }

        response = self._post_request("api/mpegts/network/mux_create", data)
        result = self._get_json(response)
        logger.info(f"Created new mux: {entry.name} -> {entry.url[:50]}...")
        return result["uuid"]

    def update_mux(self, mux: Mux):
        """Update existing mux"""
        logger = logging.getLogger('tvheadend_m3u_sync')

        if self.dry_run:
            logger.info(f"[DRY RUN] Would update mux: {mux.name} (UUID: {mux.uuid})")
            logger.debug(f"[DRY RUN] Would set URL to: {mux.url}")
            return

        node_data = {
            "uuid": mux.uuid,
            "iptv_muxname": mux.name,
            "iptv_url": mux.url
        }

        data = {
            "node": json.dumps(node_data)
        }


        response = self._post_request("api/idnode/save", data)
        response.raise_for_status()
        logger.info(f"Updated mux: {mux.name} (UUID: {mux.uuid})")


    def delete_mux(self, mux: Mux):
        """Delete mux"""
        logger = logging.getLogger('tvheadend_m3u_sync')


        if self.dry_run:
            logger.info(f"[DRY RUN] Would delete mux: {mux.name} -> {mux.url}")
            logger.debug(f"[DRY RUN] Would delete UUID: {mux.uuid}")
            return


        uuid_array = [mux.uuid]


        data = {
            "uuid": json.dumps(uuid_array)
        }


        response = self._post_request("api/idnode/delete", data)
        response.raise_for_status()
        logger.info(f"Deleted mux: {mux.name} -> {mux.url}")


    def get_networks(self) -> List[Network]:
        """Get list of networks"""
        logger = logging.getLogger('tvheadend_m3u_sync')


        params = self._create_params()


        try:
            response = self._post_request("api/mpegts/network/grid", params)
            data = self._get_json(response)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                logger.error("Authentication failed - check your credentials")
                logger.debug(f"HTTP 400 Error: {e.response.text}")
            raise


        networks = []
        for entry in data.get("entries", []):
            network = Network(
                uuid=entry.get("uuid"),
                name=entry.get("networkname"),
                enabled=entry.get("enabled", False)
            )
            networks.append(network)


        return networks


    def get_muxes(self) -> List[Mux]:
        """Get list of muxes"""
        params = self._create_params(start=0, limit=10000)  # Large enough limit


        response = self._post_request("api/mpegts/mux/grid", params)
        data = self._get_json(response)


        muxes = []
        for entry in data.get("entries", []):
            mux = Mux(
                uuid=entry.get("uuid"),
                name=entry.get("name"),
                enabled=entry.get("enabled", False),
                network_name=entry.get("networkname"),
                network_uuid=entry.get("network_uuid"),
                url=entry.get("iptv_url")
            )
            muxes.append(mux)


        return muxes


# ==================== ARGUMENT PARSING ====================

def parse_args():
    """Parse command line arguments with improved security and configuration options"""
    parser = argparse.ArgumentParser(
        description='TVHeadend M3U Sync - Synchronize M3U playlist files with TVHeadend server\n\nUsage: python tvheadend_m3u_sync.py -m <m3u_file> -n <network_name> --url <tvheadend_url>',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full synchronization with environment variables (recommended for security)
  export TVH_URL="http://tvheadend:9981"
  export TVH_USERNAME="admin"
  export TVH_PASSWORD="password"
  %(prog)s -m playlist.m3u -n "IPTV Network"

  # Auto-generated network name from M3U filename
  %(prog)s -m my_channels.m3u --url http://tvheadend:9981  # Network: "my channels"

  # Full synchronization with config file
  %(prog)s --config /path/to/config.ini -m playlist.m3u -n "IPTV Network"

  # Full synchronization with command line (flexible order)
  %(prog)s -m playlist.m3u -n "IPTV Network" --url http://tvheadend:9981 -u admin -p password

  # Dry run preview
  %(prog)s --dry-run -m playlist.m3u -n "IPTV Network" --url http://localhost:9981

  # Arguments can be in any order
  %(prog)s --url http://localhost:9981 -m channels.m3u -n "My IPTV"
        """
    )


    # Main options
    parser.add_argument('-m', '--m3u-file', help='Path to M3U playlist file (or use TVH_M3U_FILE env var)')
    parser.add_argument('-n', '--network-name', help='TVHeadend network name (or use TVH_NETWORK env var)')
    parser.add_argument('--url', help='TVHeadend server URL (or use TVH_URL env var)')

    # Authentication options
    parser.add_argument('-u', '--username', help='TVHeadend username (prefer TVH_USERNAME env var)')
    parser.add_argument('-p', '--password', help='TVHeadend password (prefer TVH_PASSWORD env var)')
    parser.add_argument('--basic-auth', action='store_true', help='Use basic authentication instead of digest')

    # Global options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-file', help='Log to file instead of console')
    parser.add_argument('--json-log', action='store_true', help='Use JSON log format')
    parser.add_argument('--timeout', type=int, default=30, help='HTTP request timeout in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying them')


    args = parser.parse_args()

    # Validate required parameters
    if not args.m3u_file and not args.config:
        parser.print_help()
        sys.exit(1)

    return args


# ==================== MAIN PROGRAM ====================

def main():
    """Main program entry point with improved error handling and configuration"""
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    try:
        args = parse_args()


        # Setup logging with new options
        setup_logging(
            verbose=args.verbose, 
            log_file=getattr(args, 'log_file', None),
            json_format=getattr(args, 'json_log', False)
        )
        logger = logging.getLogger('tvheadend_m3u_sync')


        logger.info(f"Starting TVHeadend M3U Sync v{__version__}")


        # Load configuration
        config = Config()
        config.load_config(getattr(args, 'config', None))


        sync_mode(args, config)


    except KeyboardInterrupt:
        logger = logging.getLogger('tvheadend_m3u_sync')
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except SystemExit:
        raise  # Re-raise SystemExit to preserve exit codes
    except Exception as e:
        logger = logging.getLogger('tvheadend_m3u_sync')
        logger.error(f"Unexpected error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        # Cleanup logging handlers
        logging.shutdown()





def sync_mode(args, config: Config):
    """Handle synchronization with configuration support"""
    logger = logging.getLogger('tvheadend_m3u_sync')


    # Get configuration values with fallback order: CLI args -> config file -> env vars
    m3u_file = getattr(args, 'm3u_file', None) or config.get('m3u_file')
    network_name = getattr(args, 'network_name', None) or config.get('network_name')
    url = getattr(args, 'url', None) or config.get('url')
    username = getattr(args, 'username', None) or config.get('username', '')
    password = getattr(args, 'password', None) or config.get('password', '')

    # Auto-generate network name from M3U filename if not provided
    if not network_name and m3u_file:
        # Extract filename without extension and replace underscores with spaces
        m3u_basename = os.path.basename(m3u_file)
        network_name = os.path.splitext(m3u_basename)[0].replace('_', ' ')
        logger.debug(f"Auto-generated network name from M3U file: '{m3u_basename}' -> '{network_name}'")


    # Validate required parameters
    if not m3u_file:
        logger.error("M3U file path is required (provide via argument, config file, or TVH_M3U_FILE env var)")
        sys.exit(1)


    if not network_name:
        logger.error("Network name is required (provide via argument, config file, or TVH_NETWORK env var)")
        sys.exit(1)


    if not url:
        logger.error("TVHeadend URL is required (provide via argument, config file, or TVH_URL env var)")
        sys.exit(1)


    # Validate M3U file exists
    if not os.path.exists(m3u_file):
        logger.error(f"M3U File ({m3u_file}) not found")
        sys.exit(1)


    # Validate URL
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
    except Exception:
        logger.error(f"Invalid URL ({url})")
        sys.exit(1)


    # Log configuration (with masked sensitive data)
    if username:
        logger.debug(f"Using username: {config.mask_sensitive_data(username)}")
    if password:
        logger.debug(f"Using password: {config.mask_sensitive_data(password)}")


    mode_info = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"{mode_info}Starting sync: {m3u_file} -> {url} (network: {network_name})")


    # Parse M3U file
    logger.debug("Parsing M3U file...")
    try:
        m3u_entries = Parser.get_entries(m3u_file)
        logger.info(f"Found {len(m3u_entries)} entries in M3U file")
    except Exception as e:
        logger.error(f"Failed to parse M3U file: {e}")
        sys.exit(1)


    # Initialize TVHeadend client
    logger.debug("Initializing TVHeadend client...")
    try:
        client = TVHClient(url, dry_run=args.dry_run, timeout=getattr(args, 'timeout', 30))


        # Configure authentication
        if username or password:
            if hasattr(args, 'basic_auth') and args.basic_auth:
                logger.debug("Using basic authentication")
                client.authentication_type = AuthenticationType.BASIC
            else:
                logger.debug("Using digest authentication")
                client.authentication_type = AuthenticationType.DIGEST
            client.username = username
            client.password = password
        else:
            logger.debug("No authentication configured")
            client.authentication_type = AuthenticationType.DIGEST


        # Get or create network
        logger.debug(f"Getting/creating network: {network_name}")
        current_network = get_network(network_name, client)
        logger.info(f"Using network: {current_network.name} (UUID: {current_network.uuid})")
    except Exception as e:
        logger.error(f"Failed to initialize TVHeadend client: {e}")
        sys.exit(1)


    # Scan existing muxes and update them according to M3U file info
    logger.debug("Fetching existing muxes...")
    muxes = client.get_muxes()
    network_muxes = [mux for mux in muxes if mux.network_uuid == current_network.uuid]
    logger.info(f"Found {len(network_muxes)} existing muxes in network")


    # Update existing muxes
    updated_count = 0
    for mux in network_muxes:
        match = next((entry for entry in m3u_entries if entry.tvh_uuid == mux.uuid), None)
        if match:
            needs_update = False


            if mux.url != match.url:
                logger.info(f"Mux {mux.name} URL changed: {mux.url} -> {match.url}")
                mux.url = match.url
                needs_update = True


            if mux.name != match.name:
                logger.info(f"Mux name changed: {mux.name} -> {match.name}")
                mux.name = match.name
                needs_update = True


            if needs_update:
                logger.debug(f"Updating mux: {mux.uuid}")
                client.update_mux(mux)
                updated_count += 1


    if updated_count > 0:
        logger.info(f"Updated {updated_count} existing muxes")


    # M3U update if needed - only update mux UUID tag on the correct entry
    # for future sync, normally should happen only on first run for the entry
    update_m3u_file = False
    created_count = 0


    for entry in m3u_entries:
        current_mux = next((mux for mux in network_muxes if mux.url == entry.url), None)


        if current_mux is None:
            logger.info(f"Creating new mux: {entry.name} ({entry.url})")
            uuid = client.add_mux(current_network, entry)
            entry.tvh_uuid = uuid
            update_m3u_file = True
            created_count += 1
        else:
            if entry.name != current_mux.name:
                entry.tvh_uuid = current_mux.uuid
                logger.info(f"Mux name changed: {current_mux.name} -> {entry.name} (UUID: {entry.tvh_uuid})")
                current_mux.name = entry.name
                client.update_mux(current_mux)
                update_m3u_file = True


    if created_count > 0:
        logger.info(f"Created {created_count} new muxes")


    logger.info(f"Finished analyzing {len(m3u_entries)} M3U entries")


    if update_m3u_file:
        if args.dry_run:
            logger.info("[DRY RUN] Would update M3U file with UUIDs")
        else:
            logger.info("Updating M3U file with UUIDs...")
        Parser.write_file(m3u_file, m3u_entries, dry_run=args.dry_run)
        if not args.dry_run:
            logger.info("M3U file updated successfully")
    else:
        logger.info("No M3U file update needed")


    # Find muxes for the network which have URLs not found in M3U file
    # Should only happen if user removed entry from M3U
    muxes = [mux for mux in client.get_muxes() if mux.network_uuid == current_network.uuid]


    if muxes:
        deleted_count = 0
        for mux in muxes:
            if not any(entry.url == mux.url for entry in m3u_entries):
                if deleted_count == 0:
                    logger.info(f"Deleting old muxes from network {current_network.name}...")
                logger.info(f"Deleting old mux: {mux.name} ({mux.url})")
                client.delete_mux(mux)
                deleted_count += 1


        if deleted_count == 0:
            logger.info("No muxes for deletion")
        else:
            logger.info(f"Deleted {deleted_count} old muxes")


    if args.dry_run:
        logger.info("[DRY RUN] Synchronization simulation completed - no actual changes made")
    else:
        logger.info("Synchronization completed successfully")


def get_network(network_name_to_sync: str, client: TVHClient) -> Network:
    """Get existing network or create new one"""
    logger = logging.getLogger('tvheadend_m3u_sync')


    logger.debug("Fetching existing networks...")
    networks = client.get_networks()
    logger.debug(f"Found {len(networks)} networks")


    work_network = next((network for network in networks if network.name == network_name_to_sync), None)


    if work_network is None:
        if client.dry_run:
            logger.info(f"[DRY RUN] Network '{network_name_to_sync}' not found, would create new IPTV network")
            return Network(
                name=network_name_to_sync,
                uuid=f"dry-run-network-{hash(network_name_to_sync) % 100000}",
                enabled=False
            )
        else:
            logger.info(f"Network '{network_name_to_sync}' not found, creating new IPTV network...")
            client.create_new_iptv_network(network_name_to_sync)


            # Refetch and find the new network
            networks = client.get_networks()
            work_network = next(network for network in networks if network.name == network_name_to_sync)
            logger.info(f"Created new network: {work_network.name}")
    else:
        logger.debug(f"Using existing network: {work_network.name}")


    return work_network


if __name__ == "__main__":
    main()
