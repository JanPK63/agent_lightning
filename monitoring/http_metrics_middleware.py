"""
FastAPI Middleware for HTTP Metrics Collection
Automatically instruments FastAPI applications with Prometheus HTTP metrics
"""

import time
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .metrics import get_metrics


class HTTPMetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that automatically collects HTTP metrics for all endpoints.

    This middleware tracks:
    - HTTP request count by method, endpoint, and status code
    - HTTP request duration by method and endpoint
    - Error rates by error type and component
    """

    def __init__(self, app, service_name: str = "api"):
        super().__init__(app)
        self.service_name = service_name
        self.metrics = get_metrics(service_name)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each HTTP request and collect metrics.

        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler

        Returns:
            Response: FastAPI response object
        """
        start_time = time.time()

        # Extract request details
        method = request.method
        endpoint = request.url.path
        client_ip = self._get_client_ip(request)

        try:
            # Process the request
            response = await call_next(request)

            # Calculate response time
            duration = time.time() - start_time

            # Record successful metrics
            self._record_success_metrics(method, endpoint, response.status_code, duration)

            return response

        except Exception as exc:
            # Calculate response time for errors
            duration = time.time() - start_time

            # Record error metrics
            self._record_error_metrics(method, endpoint, exc, duration)

            # Re-raise the exception (let FastAPI handle it)
            raise exc

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for other proxy headers
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        client_host = getattr(request.client, 'host', None)
        return client_host or "unknown"

    def _record_success_metrics(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record metrics for successful requests."""
        try:
            # Record HTTP request metrics
            self.metrics.record_http_request(method, endpoint, status_code, duration)

            # Log successful request
            self.metrics.logger.info(
                f"HTTP {method} {endpoint} -> {status_code} ({duration:.3f}s)"
            )

        except Exception as e:
            # Don't let metrics collection break the response
            self.metrics.logger.error(f"Failed to record success metrics: {e}")

    def _record_error_metrics(self, method: str, endpoint: str, exc: Exception, duration: float):
        """Record metrics for failed requests."""
        try:
            # Determine error type
            error_type = type(exc).__name__

            # Record error metrics
            self.metrics.record_service_error(error_type, "http_middleware")

            # Log error
            self.metrics.logger.warning(
                f"HTTP {method} {endpoint} failed: {error_type} ({duration:.3f}s)"
            )

        except Exception as e:
            # Don't let metrics collection break error handling
            self.metrics.logger.error(f"Failed to record error metrics: {e}")


def add_http_metrics_middleware(app, service_name: str = "api"):
    """
    Convenience function to add HTTP metrics middleware to a FastAPI app.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for metrics labeling

    Returns:
        FastAPI app with middleware added
    """
    app.add_middleware(HTTPMetricsMiddleware, service_name=service_name)
    return app


# Example usage in a FastAPI application:
"""
from fastapi import FastAPI
from monitoring.http_metrics_middleware import add_http_metrics_middleware

app = FastAPI(title="My API")
app = add_http_metrics_middleware(app, service_name="my_service")

@app.get("/health")
async def health():
    return {"status": "healthy"}
"""