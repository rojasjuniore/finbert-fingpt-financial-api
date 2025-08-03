"""
Logging configuration for FinBERT API
"""

import sys
import json
from typing import Dict, Any
from loguru import logger
from .config import get_settings

settings = get_settings()


class JSONFormatter:
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: Dict[str, Any]) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"]
        }
        
        # Add extra fields if present
        if "extra" in record:
            log_entry.update(record["extra"])
            
        return json.dumps(log_entry)


def setup_logging():
    """Configure application logging"""
    
    # Remove default logger
    logger.remove()
    
    # Configure format based on settings
    if settings.log_format.lower() == "json":
        # JSON format for production - use built-in serialization
        logger.add(
            sys.stdout,
            format="{time} | {level} | {name}:{function}:{line} | {message}",
            level=settings.log_level.upper(),
            serialize=True  # Use built-in JSON serialization
        )
    else:
        # Human-readable format for development
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=settings.log_level.upper()
        )
    
    # Add file logging for production
    if settings.environment.lower() == "production":
        logger.add(
            "logs/finbert-api.log",
            rotation="10 MB",
            retention="7 days",
            format=JSONFormatter().format if settings.log_format.lower() == "json" else None,
            level=settings.log_level.upper()
        )
    
    return logger


# Initialize logging
app_logger = setup_logging()