"""
Basic Tests for Rate Limiting Middleware

This module contains basic unit tests for the rate limiting
middleware functionality. Run these tests to verify the
implementation works correctly.
"""

from shared.rate_limiting_middleware import (
    RateLimitingMiddleware,
    RateLimitMiddlewareConfig,
    RateLimitExceededException,
    rate_limited,
    create_endpoint_rate_limiter,
    create_service_rate_limiter
)


def test_middleware_creation():
    """Test middleware can be created with default config"""
    config = RateLimitMiddlewareConfig()
    middleware = RateLimitingMiddleware(config)

    assert middleware.config == config
    assert middleware.rate_limiter is not None
    # Ensure middleware is properly initialized
    assert hasattr(middleware, '_extract_identifier')
    print("✓ Middleware creation test passed")


def test_middleware_with_custom_config():
    """Test middleware with custom configuration"""
    config = RateLimitMiddlewareConfig(
        max_requests=50,
        window_seconds=120,
        identifier_header="X-Custom-Key",
        fallback_to_ip=False
    )
    middleware = RateLimitingMiddleware(config)

    assert config.max_requests == 50
    assert config.window_seconds == 120
    assert config.identifier_header == "X-Custom-Key"
    assert config.fallback_to_ip is False
    # Ensure middleware was created successfully
    assert middleware is not None
    print("✓ Custom config test passed")


def test_decorator_creation():
    """Test decorator can be created"""
    @rate_limited(max_requests=10, window_seconds=60)
    def test_function():
        return "test"

    assert hasattr(test_function, 'rate_limiter')
    assert test_function.rate_limiter is not None
    print("✓ Decorator creation test passed")


def test_decorator_with_allowed_request():
    """Test decorator allows requests within limit"""
    @rate_limited(max_requests=5, window_seconds=60)
    def test_function():
        return "allowed"

    # Should allow multiple calls within limit
    for i in range(3):
        result = test_function()
        assert result == "allowed"

    print("✓ Decorator allowed requests test passed")


def test_decorator_with_exceeded_limit():
    """Test decorator raises exception when limit exceeded"""
    @rate_limited(max_requests=2, window_seconds=1)
    def test_function():
        return "test"

    # Use up the limit
    test_function()
    test_function()

    # Next call should raise exception
    try:
        test_function()
        assert False, "Expected RateLimitExceededException"
    except RateLimitExceededException:
        print("✓ Decorator exceeded limit test passed")


def test_create_endpoint_rate_limiter():
    """Test endpoint rate limiter creation"""
    middleware = create_endpoint_rate_limiter(
        max_requests=25,
        window_seconds=30,
        identifier_header="X-Test-Key"
    )

    assert isinstance(middleware, RateLimitingMiddleware)
    assert middleware.config.max_requests == 25
    assert middleware.config.window_seconds == 30
    assert middleware.config.identifier_header == "X-Test-Key"
    print("✓ Endpoint rate limiter creation test passed")


def test_create_service_rate_limiter():
    """Test service rate limiter creation"""
    limiter = create_service_rate_limiter(
        max_requests=100,
        window_seconds=300
    )

    assert limiter is not None
    # Test that it works
    assert limiter.is_allowed("test_id") is True
    print("✓ Service rate limiter creation test passed")


def test_rate_limit_exceeded_exception():
    """Test exception creation"""
    exc = RateLimitExceededException(
        detail="Custom message",
        retry_after=30,
        remaining=5,
        limit=10
    )

    assert exc.detail == "Custom message"
    assert exc.retry_after == 30
    assert exc.remaining == 5
    assert exc.limit == 10
    assert exc.status_code == 429
    print("✓ Exception creation test passed")


def run_all_tests():
    """Run all tests"""
    print("Running Rate Limiting Middleware Tests...")
    print("=" * 50)

    test_middleware_creation()
    test_middleware_with_custom_config()
    test_decorator_creation()
    test_decorator_with_allowed_request()
    test_decorator_with_exceeded_limit()
    test_create_endpoint_rate_limiter()
    test_create_service_rate_limiter()
    test_rate_limit_exceeded_exception()

    print("=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()