"""
Main FastAPI application for FinBERT sentiment analysis API
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from loguru import logger

from .core.config import get_settings
from .core.logging import setup_logging
from .api.routes import router
from .api.finnhub_routes import router as finnhub_router
from .services.finbert_service import finbert_service
from .services.fingpt_service import fingpt_service
from .utils.middleware import LoggingMiddleware, RateLimitMiddleware
from .utils.exceptions import (
    http_exception_handler,
    general_exception_handler,
    validation_exception_handler
)
from . import __version__, __description__

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FinBERT API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Model: {settings.model_name}")
    
    # Load the FinBERT model
    logger.info("Loading FinBERT model...")
    finbert_loaded = await finbert_service.load_model()
    
    if finbert_loaded:
        logger.success("FinBERT model loaded successfully!")
    else:
        logger.error("Failed to load FinBERT model!")
    
    # Load the FinGPT model
    logger.info("Loading FinGPT model...")
    fingpt_loaded = await fingpt_service.load_model()
    
    if fingpt_loaded:
        logger.success("FinGPT model loaded successfully!")
    else:
        logger.warning("Failed to load FinGPT model! FinGPT features will be disabled.")
        # You might want to exit here or set a flag
    
    # Log startup completion
    logger.info("FinBERT API startup completed")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down FinBERT API...")
    logger.info("FinBERT API shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="FinBERT API",
    description=__description__,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    openapi_url="/openapi.json" if settings.environment != "production" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)

# Add rate limiting if configured
if settings.rate_limit:
    app.add_middleware(RateLimitMiddleware, calls_per_minute=settings.rate_limit)

# Add exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["sentiment-analysis"])
app.include_router(finnhub_router, prefix="/api/v1", tags=["enhanced-analysis"])

# Health check endpoint at root
@app.get("/", include_in_schema=False)
async def root_health_check():
    """Simple health check at root"""
    return {
        "service": "LLMs API",
        "version": __version__,
        "status": "healthy" if finbert_service.is_loaded() else "starting",
        "finbert_loaded": finbert_service.is_loaded(),
        "fingpt_loaded": fingpt_service.is_loaded(),
        "docs": "/docs" if settings.environment != "production" else "disabled",
        "api": "/api/v1"
    }


# Additional health endpoint for load balancers
@app.get("/health", include_in_schema=False)
async def simple_health():
    """Simple health endpoint for load balancers"""
    if finbert_service.is_loaded():
        return {"status": "healthy"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "model not loaded"}
        )


if __name__ == "__main__":
    import uvicorn
    
    # Setup logging
    setup_logging()
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
        log_config=None,  # Use our custom logging
    )