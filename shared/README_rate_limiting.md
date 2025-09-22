# Rate Limiting Middleware

This module provides comprehensive rate limiting functionality for Agent Lightning, integrating seamlessly with FastAPI applications and service methods.

## Features

- **FastAPI Middleware**: Automatic rate limiting for all endpoints
- **Function Decorators**: Rate limiting for individual service methods
- **Redis Integration**: Distributed rate limiting with Redis backend
- **Flexible Configuration**: Customizable limits, identifiers, and exclusions
- **Proper Error Responses**: HTTP 429 responses with rate limit headers
- **Fallback Support**: Graceful degradation when Redis is unavailable

## Quick Start

### 1. Environment-Based Setup (Recommended)

```python
from fastapi import FastAPI
from shared.rate_limiting_middleware import setup_rate_limiting

app = FastAPI()

# Setup rate limiting with environment configuration
setup_rate_limiting(app)  # Automatically loads from environment variables

@app.get("/api/data")
async def get_data():
    return {"message": "Hello World"}
```

### 2. Programmatic Setup

```python
from fastapi import FastAPI
from shared.rate_limiting_middleware import setup_rate_limiting, RateLimitMiddlewareConfig

app = FastAPI()

# Custom configuration
config = RateLimitMiddlewareConfig(
    max_requests=50,
    window_seconds=300
)

# Setup with custom config (overrides environment)
setup_rate_limiting(app, config, use_env_config=False)

@app.get("/api/data")
async def get_data():
    return {"message": "Hello World"}
```

### 2. Service Method Rate Limiting

```python
from shared.rate_limiting_middleware import rate_limited

class MyService:
    @rate_limited(max_requests=10, window_seconds=60)
    def process_data(self, user_id: str, data: dict):
        # This method is rate limited to 10 calls per minute per user
        return {"processed": True, "data": data}
```

## Configuration

### Environment-Based Configuration

The rate limiting middleware supports configuration via environment variables for easy deployment and configuration management:

```bash
# Enable/disable rate limiting
RATE_LIMIT_ENABLED=true

# Rate limiting settings
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_PREFIX=api

# Identifier extraction
RATE_LIMIT_IDENTIFIER_HEADER=X-API-Key
RATE_LIMIT_FALLBACK_TO_IP=true

# Error handling
RATE_LIMIT_ERROR_MESSAGE=Rate limit exceeded
RATE_LIMIT_INCLUDE_HEADERS=true

# Exclusions
RATE_LIMIT_EXCLUDE_PATHS=/health,/metrics
RATE_LIMIT_EXCLUDE_METHODS=OPTIONS

# Service method settings
RATE_LIMIT_SERVICE_MAX_REQUESTS=1000
RATE_LIMIT_SERVICE_WINDOW_SECONDS=60
```

### Programmatic Configuration

```python
from shared.rate_limiting_middleware import RateLimitMiddlewareConfig

config = RateLimitMiddlewareConfig(
    max_requests=100,        # Max requests per window
    window_seconds=60,       # Time window in seconds
    prefix="api",            # Redis key prefix
    identifier_header="X-API-Key",  # Header for rate limiting identifier
    fallback_to_ip=True,     # Use client IP if header not present
    exclude_paths=["/health", "/metrics"],  # Paths to exclude
    exclude_methods=["OPTIONS"],  # HTTP methods to exclude
)
```

### Automatic Environment Loading

The `setup_rate_limiting` function automatically loads configuration from environment variables:

```python
from shared.rate_limiting_middleware import setup_rate_limiting

# Automatically loads from environment variables
setup_rate_limiting(app)

# Or disable environment loading
setup_rate_limiting(app, use_env_config=False)
```

## Usage Examples

### FastAPI Middleware

```python
from fastapi import FastAPI
from shared.rate_limiting_middleware import (
    RateLimitingMiddleware,
    RateLimitMiddlewareConfig
)

app = FastAPI()

# Create configuration
config = RateLimitMiddlewareConfig(
    max_requests=50,
    window_seconds=300,  # 5 minutes
    identifier_header="Authorization"
)

# Add middleware
app.add_middleware(RateLimitingMiddleware, config=config)

# Your routes will now be automatically rate limited
@app.get("/api/users")
async def get_users():
    return {"users": []}
```

### Function Decorators

```python
from shared.rate_limiting_middleware import rate_limited, async_rate_limited

class UserService:
    @rate_limited(max_requests=5, window_seconds=60)
    def create_user(self, user_data: dict):
        # Rate limited to 5 calls per minute
        return {"user_id": "123", "created": True}

    @rate_limited(
        max_requests=100,
        window_seconds=3600,
        identifier_func=lambda self, user_id, **kwargs: f"user:{user_id}"
    )
    def get_user_profile(self, user_id: str):
        # Rate limited per user (100 calls per hour)
        return {"user_id": user_id, "profile": {}}

    @async_rate_limited(max_requests=20, window_seconds=60)
    async def async_operation(self, data: dict):
        # Async rate limiting
        return {"result": "async_processed"}
```

### Custom Identifier Functions

```python
@rate_limited(
    max_requests=10,
    window_seconds=300,
    identifier_func=lambda self, tenant_id, user_id, **kwargs: f"tenant:{tenant_id}:user:{user_id}"
)
def tenant_user_operation(self, tenant_id: str, user_id: str):
    # Rate limited per tenant+user combination
    pass
```

## Redis Integration

For distributed deployments, use Redis for rate limiting:

```python
import redis
from shared.rate_limiting_middleware import create_endpoint_rate_limiter

# Create Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Create rate limiter with Redis
middleware = create_endpoint_rate_limiter(
    max_requests=100,
    window_seconds=60,
    redis_client=redis_client
)

app.add_middleware(RateLimitingMiddleware, config=middleware.config)
```

## Error Handling

The middleware automatically returns HTTP 429 responses with appropriate headers:

```python
# Response headers include:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 95
# X-RateLimit-Reset: 1640995200
# Retry-After: 60
```

For service methods, rate limit exceptions are raised:

```python
from shared.rate_limiting_middleware import RateLimitExceededException

try:
    result = my_service.rate_limited_method()
except RateLimitExceededException as e:
    print(f"Rate limit exceeded: {e.detail}")
    print(f"Retry after: {e.retry_after} seconds")
```

## Advanced Configuration

### Setup Utility

```python
from shared.rate_limiting_middleware import setup_rate_limiting

# Quick setup with defaults
setup_rate_limiting(app)

# Advanced setup
config = RateLimitMiddlewareConfig(...)
setup_rate_limiting(app, config, exception_handler=True)
```

### Service Rate Limiter

```python
from shared.rate_limiting_middleware import create_service_rate_limiter

# Create rate limiter for service methods
limiter = create_service_rate_limiter(
    max_requests=1000,
    window_seconds=60,
    redis_client=redis_client
)

# Use with custom decorator
@rate_limited(rate_limiter=limiter)
def my_method():
    pass
```

## Rate Limit Headers

When rate limits are exceeded, the following headers are included:

- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when the limit resets
- `Retry-After`: Seconds until the limit resets

## Best Practices

1. **Choose Appropriate Limits**: Balance security with usability
2. **Use Redis for Production**: Ensures consistency across multiple instances
3. **Exclude Health Checks**: Don't rate limit monitoring endpoints
4. **Monitor Rate Limits**: Track rate limit hits for capacity planning
5. **Use Descriptive Identifiers**: Make identifiers meaningful for debugging

## Testing

Run the included tests to verify functionality:

```bash
PYTHONPATH=/path/to/project python tests/test_rate_limiting_middleware.py
```

## Integration with Existing Services

The middleware integrates seamlessly with existing Agent Lightning services:

```python
# In your service file
from shared.rate_limiting_middleware import rate_limited

class MyService:
    def __init__(self):
        # Service initialization
        pass

    @rate_limited(max_requests=50, window_seconds=300)
    def expensive_operation(self, data):
        # Rate limited expensive operation
        return self._process_data(data)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the module is in your Python path
2. **Redis Connection**: Check Redis connectivity for distributed setups
3. **Header Not Found**: Verify identifier headers are being sent correctly
4. **Rate Limits Too Strict**: Adjust limits based on your use case

### Debugging

Enable debug logging to see rate limiting decisions:

```python
import logging
logging.getLogger('shared.rate_limiting_middleware').setLevel(logging.DEBUG)
```

## API Reference

### Classes

- `RateLimitingMiddleware`: FastAPI middleware for endpoint rate limiting
- `RateLimitMiddlewareConfig`: Configuration for rate limiting
- `RateLimitExceededException`: Exception raised when rate limits are exceeded

### Functions

- `rate_limited()`: Decorator for synchronous service methods
- `async_rate_limited()`: Decorator for asynchronous service methods
- `create_endpoint_rate_limiter()`: Factory for endpoint rate limiters
- `create_service_rate_limiter()`: Factory for service rate limiters
- `setup_rate_limiting()`: Utility for quick FastAPI integration
- `load_rate_limit_config_from_env()`: Load configuration from environment variables

### Configuration Options

See `RateLimitMiddlewareConfig` for all available options including Redis settings, identifier extraction, path exclusions, and error handling preferences.