"""
Custom middleware for FinBERT API
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log request
        start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
            }
        )
        
        # Add request ID to state
        request.state.request_id = request_id
        
        try:
            # Process request with enhanced error handling
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time,
                }
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Enhanced error logging with safe string formatting
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Safely format the error message to avoid KeyError issues
            logger.error(
                "Request failed with error: {} - Type: {}",
                error_msg.replace('{', '{{').replace('}', '}}'),  # Escape braces to prevent format errors
                error_type,
                extra={
                    "request_id": request_id,
                    "error": error_msg,
                    "error_type": error_type,
                    "process_time": process_time,
                    "url": str(request.url),
                    "method": request.method
                }
            )
            
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, calls_per_minute: int = 100):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.client_requests = {}
        self.window_size = 60  # 1 minute
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self.calls_per_minute is None:
            return await call_next(request)
        
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip]
                if current_time - req_time < self.window_size
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.calls_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                headers={"Content-Type": "application/json"}
            )
        
        # Add current request
        self.client_requests[client_ip].append(current_time)
        
        return await call_next(request)