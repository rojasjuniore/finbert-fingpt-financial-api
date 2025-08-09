"""
ASGI error handling utilities to prevent EndOfStream and WouldBlock errors
"""

import asyncio
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from loguru import logger
from anyio import WouldBlock, EndOfStream


class ASGIStreamMiddleware(BaseHTTPMiddleware):
    """Middleware to handle ASGI stream communication errors"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except (WouldBlock, EndOfStream) as e:
            logger.warning(
                "ASGI stream communication error: {} - Handling gracefully",
                type(e).__name__,
                extra={
                    "error_type": type(e).__name__,
                    "url": str(request.url),
                    "method": request.method,
                    "client_ip": request.client.host if request.client else "unknown"
                }
            )
            
            # Return a proper response for stream errors
            return JSONResponse(
                status_code=503,
                content={
                    "error": "ServiceTemporarilyUnavailable",
                    "message": "Service temporarily unavailable due to communication error",
                    "details": {"error_type": "stream_communication"}
                }
            )
            
        except asyncio.CancelledError:
            logger.info("Request cancelled by client")
            return JSONResponse(
                status_code=499,
                content={
                    "error": "ClientDisconnected", 
                    "message": "Client disconnected"
                }
            )
            
        except Exception as e:
            logger.error(
                "Unexpected error in ASGI handler: {} - {}",
                type(e).__name__,
                str(e).replace('{', '{{').replace('}', '}}'),
                extra={
                    "error_type": type(e).__name__,
                    "url": str(request.url),
                    "method": request.method
                }
            )
            raise