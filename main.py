"""
Entry point for FinBERT API
"""

import uvicorn
from app.main import app
from app.core.config import get_settings
from app.core.logging import setup_logging

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Get settings
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
        log_config=None,  # Use our custom logging
    )