"""
Rate Limiting Middleware for FastAPI Endpoints and Service Methods

This module provides comprehensive rate limiting functionality for
Agent Lightning, integrating with the existing rate limiter
infrastructure. It includes:

- FastAPI middleware for automatic endpoint rate limiting
- Function decorators for service method rate limiting
- Proper error responses with rate limit information
- Flexible configuration options
- Integration with Redis and in-memory fallbacks
"""

import time
import logging
import os
from typing import Optional, Any, Callable
from functools import wraps
from dataclasses import dataclass, field

from fastapi import Request, HTTPException, Response
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint
)

from shared.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    create_rate_limiter
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimitMiddlewareConfig:
    """Configuration for rate limiting middleware"""

    # Rate limiting settings
    max_requests: int = 100
    window_seconds: int = 60
    prefix: str = "api"

    # Identifier extraction
    identifier_header: str = "X-API-Key"  # Primary identifier header
    fallback_to_ip: bool = True  # Use client IP if header not present

    # Error response settings
    error_message: str = "Rate limit exceeded"
    include_headers: bool = True  # Include rate limit headers in response

    # Redis settings (optional)
    redis_client: Optional[Any] = None

    # Advanced settings
    exclude_paths: list = field(
        default_factory=lambda: ["/health", "/metrics"]
    )
    exclude_methods: list = field(default_factory=lambda: ["OPTIONS"])

    # Service method decorator settings
    service_max_requests: int = 1000
    service_window_seconds: int = 60


def load_rate_limit_config_from_env() -> RateLimitMiddlewareConfig:
    """
    Load rate limiting configuration from environment variables.

    Environment Variables:
        RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
        RATE_LIMIT_MAX_REQUESTS: Max requests per window (default: 100)
        RATE_LIMIT_WINDOW_SECONDS: Time window in seconds (default: 60)
        RATE_LIMIT_SERVICE_MAX_REQUESTS: Max requests for service methods
        RATE_LIMIT_SERVICE_WINDOW_SECONDS: Service window in seconds
        RATE_LIMIT_IDENTIFIER_HEADER: Header for rate limiting identifier
        RATE_LIMIT_FALLBACK_TO_IP: Use client IP if header not present
        RATE_LIMIT_INCLUDE_HEADERS: Include rate limit headers in response
        RATE_LIMIT_ERROR_MESSAGE: Error message for rate limit exceeded
        RATE_LIMIT_PREFIX: Redis key prefix (default: api)
        RATE_LIMIT_EXCLUDE_PATHS: Comma-separated paths to exclude
        RATE_LIMIT_EXCLUDE_METHODS: Comma-separated methods to exclude

    Returns:
        RateLimitMiddlewareConfig: Configuration loaded from environment
    """
    # Helper function to get boolean from env
    def get_bool_env(key: str, default: bool) -> bool:
        value = os.getenv(key, str(default).lower())
        return value.lower() in ('true', '1', 'yes', 'on')

    # Helper function to get int from env
    def get_int_env(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}, using default: {default}"
            )
            return default

    # Helper function to get list from env
    def get_list_env(key: str, default: str) -> list:
        value = os.getenv(key, default)
        if not value.strip():
            return []
        return [item.strip() for item in value.split(',') if item.strip()]

    # Load configuration from environment
    config = RateLimitMiddlewareConfig(
        max_requests=get_int_env('RATE_LIMIT_MAX_REQUESTS', 100),
        window_seconds=get_int_env('RATE_LIMIT_WINDOW_SECONDS', 60),
        prefix=os.getenv('RATE_LIMIT_PREFIX', 'api'),
        identifier_header=os.getenv(
            'RATE_LIMIT_IDENTIFIER_HEADER', 'X-API-Key'
        ),
        fallback_to_ip=get_bool_env('RATE_LIMIT_FALLBACK_TO_IP', True),
        error_message=os.getenv(
            'RATE_LIMIT_ERROR_MESSAGE', 'Rate limit exceeded'
        ),
        include_headers=get_bool_env('RATE_LIMIT_INCLUDE_HEADERS', True),
        exclude_paths=get_list_env(
            'RATE_LIMIT_EXCLUDE_PATHS', '/health,/metrics'
        ),
        exclude_methods=get_list_env(
            'RATE_LIMIT_EXCLUDE_METHODS', 'OPTIONS'
        ),
        service_max_requests=get_int_env(
            'RATE_LIMIT_SERVICE_MAX_REQUESTS', 1000
        ),
        service_window_seconds=get_int_env(
            'RATE_LIMIT_SERVICE_WINDOW_SECONDS', 60
        ),
    )

    # Check if rate limiting is enabled
    if not get_bool_env('RATE_LIMIT_ENABLED', True):
        logger.info("Rate limiting is disabled via environment variable")
        # Return a disabled configuration
        config.max_requests = float('inf')  # Effectively disable rate limiting
        config.service_max_requests = float('inf')

    logger.info(
        f"Rate limiting configuration loaded from environment: "
        f"{config.max_requests} requests per {config.window_seconds}s"
    )

    return config


class RateLimitExceededException(HTTPException):
    """Custom exception for rate limit exceeded"""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        remaining: Optional[int] = None,
        limit: Optional[int] = None
    ):
        super().__init__(status_code=429, detail=detail)
        self.retry_after = retry_after
        self.remaining = remaining
        self.limit = limit


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic rate limiting of endpoints

    This middleware automatically applies rate limiting to all routes based on
    configurable identifiers (API key headers or client IP addresses).
    """

    def __init__(self, config: RateLimitMiddlewareConfig):
        super().__init__(None)  # app will be set by FastAPI
        self.config = config
        self.rate_limiter = self._create_rate_limiter()

        logger.info(
            f"Rate limiting middleware initialized: "
            f"{config.max_requests} requests per {config.window_seconds}s"
        )

    def _create_rate_limiter(self) -> RateLimiter:
        """Create rate limiter instance based on configuration"""
        rate_config = RateLimitConfig(
            max_requests=self.config.max_requests,
            window_seconds=self.config.window_seconds,
            prefix=self.config.prefix
        )

        return create_rate_limiter(
            redis_client=self.config.redis_client,
            config=rate_config
        )

    def _extract_identifier(self, request: Request) -> str:
        """Extract identifier from request (API key header or IP)"""
        # Try to get identifier from header first
        if self.config.identifier_header:
            identifier = request.headers.get(self.config.identifier_header)
            if identifier:
                return f"header:{identifier}"

        # Fallback to client IP if enabled
        if self.config.fallback_to_ip:
            client_ip = self._get_client_ip(request)
            return f"ip:{client_ip}"

        # Default fallback
        return f"unknown:{request.url.path}"

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP from request"""
        # Check for forwarded headers (common in proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in case of multiple
            return forwarded_for.split(",")[0].strip()

        # Check for other proxy headers
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to request.client.host
        return getattr(request.client, 'host', 'unknown')

    def _should_skip_rate_limit(self, request: Request) -> bool:
        """Check if request should skip rate limiting"""
        # Skip excluded paths
        if any(path in request.url.path for path in self.config.exclude_paths):
            return True

        # Skip excluded methods
        if request.method in self.config.exclude_methods:
            return True

        return False

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request through rate limiting middleware"""

        # Skip rate limiting for certain requests
        if self._should_skip_rate_limit(request):
            return await call_next(request)

        # Extract identifier
        identifier = self._extract_identifier(request)

        # Check rate limit
        if not self.rate_limiter.is_allowed(identifier):
            # Get rate limit information
            remaining = self.rate_limiter.get_remaining(identifier)
            reset_time = self.rate_limiter.get_reset_time(identifier)
            retry_after = max(1, int(reset_time - time.time()))

            # Create error response
            detail = self.config.error_message
            if self.config.include_headers:
                detail += f". Retry after {retry_after} seconds."

            # Raise custom exception
            raise RateLimitExceededException(
                detail=detail,
                retry_after=retry_after,
                remaining=remaining,
                limit=self.config.max_requests
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers if enabled
        if self.config.include_headers:
            remaining = self.rate_limiter.get_remaining(identifier)
            reset_time = self.rate_limiter.get_reset_time(identifier)
            retry_after = max(1, int(reset_time - time.time()))

            response.headers["X-RateLimit-Limit"] = str(
                self.config.max_requests
            )
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(reset_time))
            response.headers["Retry-After"] = str(retry_after)

        return response


# Exception handler for rate limit exceeded
def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitExceededException
):
    """Handle RateLimitExceededException and return proper response"""
    headers = {}

    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    if exc.remaining is not None and exc.limit is not None:
        headers["X-RateLimit-Limit"] = str(exc.limit)
        headers["X-RateLimit-Remaining"] = str(exc.remaining)
        headers["X-RateLimit-Reset"] = str(
            int(time.time() + (exc.retry_after or 60))
        )

    return Response(
        content='{"detail": "' + exc.detail + '"}',
        status_code=429,
        media_type="application/json",
        headers=headers
    )


# Function decorators for service methods
def rate_limited(
    max_requests: Optional[int] = None,
    window_seconds: Optional[int] = None,
    identifier_func: Optional[Callable] = None,
    rate_limiter: Optional[RateLimiter] = None
):
    """
    Decorator for rate limiting service methods

    Args:
        max_requests: Maximum requests allowed in window (default: 1000)
        window_seconds: Time window in seconds (default: 60)
        identifier_func: Function to extract identifier from method arguments
        rate_limiter: Custom rate limiter instance (default: creates new one)

    Example:
        @rate_limited(max_requests=50, window_seconds=300)
        def process_data(self, user_id: str, data: dict):
            # Method will be rate limited per user_id
            pass

        @rate_limited(
            identifier_func=lambda self, user_id, **kwargs: f"custom:{user_id}"
        )
        def custom_method(self, user_id: str):
            # Custom identifier extraction
            pass
    """
    def decorator(func: Callable):
        # Create rate limiter if not provided
        limiter = rate_limiter
        if limiter is None:
            config = RateLimitConfig(
                max_requests=max_requests or 1000,
                window_seconds=window_seconds or 60,
                prefix=f"service:{func.__name__}"
            )
            limiter = create_rate_limiter(config=config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract identifier
            identifier = _extract_method_identifier(
                func, args, kwargs, identifier_func
            )

            # Check rate limit
            if not limiter.is_allowed(identifier):
                remaining = limiter.get_remaining(identifier)
                reset_time = limiter.get_reset_time(identifier)
                retry_after = max(1, int(reset_time - time.time()))

                raise RateLimitExceededException(
                    detail=(
                        f"Service rate limit exceeded for {func.__name__}"
                    ),
                    retry_after=retry_after,
                    remaining=remaining,
                    limit=max_requests or 1000
                )

            # Call original function
            return func(*args, **kwargs)

        # Store rate limiter on wrapper for testing/debugging
        wrapper.rate_limiter = limiter
        return wrapper

    return decorator


def _extract_method_identifier(
    func: Callable,
    args: tuple,
    kwargs: dict,
    identifier_func: Optional[Callable]
) -> str:
    """Extract identifier from method arguments"""
    if identifier_func:
        # Use custom identifier function
        try:
            return identifier_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error in custom identifier function: {e}")
            return f"error:{func.__name__}"

    # Default identifier extraction
    if args and hasattr(args[0], '__class__'):
        # Assume first argument is self/instance
        instance = args[0]
        class_name = instance.__class__.__name__

        # Look for common identifier patterns in kwargs
        for key in ['user_id', 'user', 'id', 'identifier', 'key']:
            if key in kwargs:
                return f"{class_name}:{func.__name__}:{kwargs[key]}"

        # Look for identifier in args (skip self)
        if len(args) > 1:
            # Try second argument as identifier
            return f"{class_name}:{func.__name__}:{args[1]}"

        # Fallback to class and method name
        return f"{class_name}:{func.__name__}"

    # Fallback for module-level functions
    return f"module:{func.__module__}:{func.__name__}"


# Async version of the decorator
def async_rate_limited(
    max_requests: Optional[int] = None,
    window_seconds: Optional[int] = None,
    identifier_func: Optional[Callable] = None,
    rate_limiter: Optional[RateLimiter] = None
):
    """
    Async version of rate_limited decorator for async service methods

    Args:
        max_requests: Maximum requests allowed in window (default: 1000)
        window_seconds: Time window in seconds (default: 60)
        identifier_func: Function to extract identifier from method arguments
        rate_limiter: Custom rate limiter instance (default: creates new one)
    """
    def decorator(func: Callable):
        # Create rate limiter if not provided
        limiter = rate_limiter
        if limiter is None:
            config = RateLimitConfig(
                max_requests=max_requests or 1000,
                window_seconds=window_seconds or 60,
                prefix=f"async_service:{func.__name__}"
            )
            limiter = create_rate_limiter(config=config)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract identifier
            identifier = _extract_method_identifier(
                func, args, kwargs, identifier_func
            )

            # Check rate limit
            if not limiter.is_allowed(identifier):
                remaining = limiter.get_remaining(identifier)
                reset_time = limiter.get_reset_time(identifier)
                retry_after = max(1, int(reset_time - time.time()))

                raise RateLimitExceededException(
                    detail=(
                        f"Async rate limit exceeded for {func.__name__}"
                    ),
                    retry_after=retry_after,
                    remaining=remaining,
                    limit=max_requests or 1000
                )

            # Call original async function
            return await func(*args, **kwargs)

        # Store rate limiter on wrapper for testing/debugging
        wrapper.rate_limiter = limiter
        return wrapper

    return decorator


# Utility functions for common use cases
def create_endpoint_rate_limiter(
    max_requests: int = 100,
    window_seconds: int = 60,
    redis_client: Optional[Any] = None,
    identifier_header: str = "X-API-Key",
    fallback_to_ip: bool = True
) -> RateLimitingMiddleware:
    """
    Create a rate limiting middleware for FastAPI endpoints

    Args:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        redis_client: Optional Redis client for distributed rate limiting
        identifier_header: Header to use for rate limiting identifier
        fallback_to_ip: Whether to fallback to client IP if header not present

    Returns:
        Configured RateLimitingMiddleware instance
    """
    config = RateLimitMiddlewareConfig(
        max_requests=max_requests,
        window_seconds=window_seconds,
        identifier_header=identifier_header,
        fallback_to_ip=fallback_to_ip,
        redis_client=redis_client
    )

    return RateLimitingMiddleware(config)


def create_service_rate_limiter(
    max_requests: int = 1000,
    window_seconds: int = 60,
    redis_client: Optional[Any] = None
) -> RateLimiter:
    """
    Create a rate limiter for service methods

    Args:
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        redis_client: Optional Redis client for distributed rate limiting

    Returns:
        Configured RateLimiter instance
    """
    config = RateLimitConfig(
        max_requests=max_requests,
        window_seconds=window_seconds,
        prefix="service"
    )

    return create_rate_limiter(redis_client=redis_client, config=config)


# Example usage and integration helpers
def setup_rate_limiting(
    app,
    endpoint_config: Optional[RateLimitMiddlewareConfig] = None,
    exception_handler: bool = True,
    use_env_config: bool = True
):
    """
    Setup rate limiting for a FastAPI application

    Args:
        app: FastAPI application instance
        endpoint_config: Configuration for endpoint rate limiting
        exception_handler: Whether to add exception handler for rate limits
        use_env_config: Whether to load config from environment variables
    """
    if endpoint_config is None:
        if use_env_config:
            # Load configuration from environment variables
            endpoint_config = load_rate_limit_config_from_env()
        else:
            endpoint_config = RateLimitMiddlewareConfig()

    # Add middleware
    app.add_middleware(RateLimitingMiddleware, config=endpoint_config)

    # Add exception handler
    if exception_handler:
        app.add_exception_handler(
            RateLimitExceededException,
            rate_limit_exception_handler
        )

    logger.info("Rate limiting setup completed for FastAPI application")


# Export key classes and functions
__all__ = [
    "RateLimitingMiddleware",
    "RateLimitMiddlewareConfig",
    "RateLimitExceededException",
    "rate_limited",
    "async_rate_limited",
    "create_endpoint_rate_limiter",
    "create_service_rate_limiter",
    "setup_rate_limiting",
    "load_rate_limit_config_from_env",
    "rate_limit_exception_handler"
]