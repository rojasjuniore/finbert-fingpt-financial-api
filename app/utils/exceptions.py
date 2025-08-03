"""
Custom exception handlers for FinBERT API
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
from datetime import datetime

from ..models.responses import ErrorResponse


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "url": str(request.url),
            "method": request.method
        }
    )
    
    error_response = ErrorResponse(
        error="HTTPException",
        message=exc.detail,
        details={
            "status_code": exc.status_code,
            "request_id": request_id
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception: {type(exc).__name__} - {str(exc)}",
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "url": str(request.url),
            "method": request.method
        }
    )
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An internal server error occurred",
        details={
            "exception_type": type(exc).__name__,
            "request_id": request_id
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle validation exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"Validation error: {str(exc)}",
        extra={
            "request_id": request_id,
            "validation_error": str(exc),
            "url": str(request.url),
            "method": request.method
        }
    )
    
    error_response = ErrorResponse(
        error="ValidationError",
        message="Request validation failed",
        details={
            "validation_errors": str(exc),
            "request_id": request_id
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.dict()
    )