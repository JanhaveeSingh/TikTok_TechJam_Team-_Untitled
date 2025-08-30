#!/usr/bin/env python3
"""
Parser Configuration and Switching System
Supports both OmniParser and CLIP Parser with automatic fallback
"""

import os
import requests
import time
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ParserConfig:
    """Configuration manager for different UI parsers"""
    
    def __init__(self):
        # Parser type selection
        self.parser_type = os.getenv("PARSER_TYPE", "auto")  # "omni", "clip", "auto"
        
        # Parser URIs
        self.omni_uri = os.getenv("OMNI_URI", "http://127.0.0.1:8000")
        self.clip_uri = os.getenv("CLIP_URI", "http://127.0.0.1:8002")  # CLIP Parser port
        
        # Parser preferences and fallback
        self.primary_parser = os.getenv("PRIMARY_PARSER", "omni")  # Primary choice
        self.fallback_parser = os.getenv("FALLBACK_PARSER", "clip")  # Fallback choice
        
        # Performance settings
        self.timeout = int(os.getenv("PARSER_TIMEOUT", 30))
        self.max_retries = int(os.getenv("PARSER_MAX_RETRIES", 2))
        
        # Cache for parser availability
        self._parser_status = {}
        self._last_health_check = 0
        self._health_check_interval = 60  # Check every 60 seconds
        
    def check_parser_health(self, parser_uri: str) -> bool:
        """Check if a parser service is healthy and available"""
        try:
            response = requests.get(
                f"{parser_uri}/health", 
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_available_parsers(self) -> Dict[str, bool]:
        """Get status of all available parsers"""
        current_time = time.time()
        
        # Use cached status if recent
        if (current_time - self._last_health_check) < self._health_check_interval:
            return self._parser_status
        
        # Check parser availability
        self._parser_status = {
            "omni": self.check_parser_health(self.omni_uri),
            "clip": self.check_parser_health(self.clip_uri)
        }
        
        self._last_health_check = current_time
        return self._parser_status
    
    def get_active_parser(self) -> Tuple[str, str]:
        """
        Get the active parser type and URI based on configuration and availability
        Returns: (parser_type, parser_uri)
        """
        available_parsers = self.get_available_parsers()
        
        # Auto mode: choose best available parser
        if self.parser_type == "auto":
            # Try primary parser first
            if available_parsers.get(self.primary_parser, False):
                parser_type = self.primary_parser
            # Fallback to secondary parser
            elif available_parsers.get(self.fallback_parser, False):
                parser_type = self.fallback_parser
            else:
                # No parsers available - use primary as default
                parser_type = self.primary_parser
                print(f"⚠️ No parsers available, using {parser_type} as default")
        
        # Specific parser requested
        else:
            parser_type = self.parser_type
            if not available_parsers.get(parser_type, False):
                print(f"⚠️ Requested parser '{parser_type}' not available")
        
        # Get URI for selected parser
        parser_uri = self.omni_uri if parser_type == "omni" else self.clip_uri
        
        return parser_type, parser_uri
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get comprehensive parser information"""
        active_parser, active_uri = self.get_active_parser()
        available_parsers = self.get_available_parsers()
        
        return {
            "active_parser": active_parser,
            "active_uri": active_uri,
            "available_parsers": available_parsers,
            "configuration": {
                "parser_type": self.parser_type,
                "primary_parser": self.primary_parser,
                "fallback_parser": self.fallback_parser,
                "timeout": self.timeout,
                "max_retries": self.max_retries
            }
        }

# Global parser config instance
parser_config = ParserConfig()

def get_parser_config() -> ParserConfig:
    """Get the global parser configuration instance"""
    return parser_config

def switch_parser(parser_type: str) -> bool:
    """
    Switch to a different parser
    Args:
        parser_type: "omni", "clip", or "auto"
    Returns:
        True if switch was successful
    """
    if parser_type not in ["omni", "clip", "auto"]:
        print(f"❌ Invalid parser type: {parser_type}")
        return False
    
    parser_config.parser_type = parser_type
    active_parser, active_uri = parser_config.get_active_parser()
    
    print(f"🔄 Switched to parser: {active_parser} ({active_uri})")
    return True

def get_current_parser() -> Tuple[str, str]:
    """Get current active parser info"""
    return parser_config.get_active_parser()

# Compatibility functions for existing code
def get_omni_uri() -> str:
    """Get OmniParser URI"""
    return parser_config.omni_uri

def get_clip_uri() -> str:
    """Get CLIP Parser URI"""
    return parser_config.clip_uri

if __name__ == "__main__":
    # Test the parser configuration
    print("🧪 Testing Parser Configuration")
    print("=" * 50)
    
    config = get_parser_config()
    info = config.get_parser_info()
    
    print(f"Active Parser: {info['active_parser']}")
    print(f"Active URI: {info['active_uri']}")
    print(f"Available Parsers: {info['available_parsers']}")
    print(f"Configuration: {info['configuration']}")
