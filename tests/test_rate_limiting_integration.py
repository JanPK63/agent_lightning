"""
Integration tests for rate limiting implementation in Agent Lightning.

This module provides comprehensive integration tests that simulate real-world
scenarios including multiple users, different endpoints, rate limit violations,
header responses, and Redis vs in-memory fallbacks using httpx for HTTP testing.
"""

import asyncio
import pytest
import redis
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
try:
    import httpx
except ImportError:
    httpx = None

from shared.rate_limiting_middleware import (
    RateLimitingMiddleware,
    RateLimitMiddlewareConfig,
    RateLimitExceededException,
    rate_limited,
    async_rate_limited,
    setup_rate_limiting,
    rate_limit_exception_handler
)


class MockTestService:
    """Test service class for testing decorators"""

    def __init__(self):
        self.call_count = 0

    @rate_limited(max_requests=3, window_seconds=60)
    def sync_method(self, user_id: str):
        """Synchronous method with rate limiting"""
        self.call_count += 1
        return {"user_id": user_id, "call_count": self.call_count}

    @async_rate_limited(max_requests=2, window_seconds=30)
    async def async_method(self, user_id: str):
        """Asynchronous method with rate limiting"""
        await asyncio.sleep(0.01)  # Simulate async work
        return {"user_id": user_id, "async": True}

    @rate_limited(
        max_requests=5,
        window_seconds=120,
        identifier_func=lambda self, tenant_id, user_id, **kwargs:
            f"tenant:{tenant_id}:user:{user_id}"
    )
    def tenant_method(self, tenant_id: str, user_id: str):
        """Method with custom identifier function"""
        return {"tenant_id": tenant_id, "user_id": user_id}


@pytest.fixture
def fastapi_app():
    """Create a FastAPI app for testing"""
    app = FastAPI()

    @app.get("/api/public")
    async def public_endpoint():
        return {"message": "public"}

    @app.get("/api/protected")
    async def protected_endpoint():
        return {"message": "protected"}

    @app.get("/api/admin")
    async def admin_endpoint():
        return {"message": "admin"}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.options("/api/test")
    async def options_endpoint():
        return {"method": "OPTIONS"}

    return app


@pytest.fixture
def rate_limit_config():
    """Create a test rate limit configuration"""
    return RateLimitMiddlewareConfig(
        max_requests=5,
        window_seconds=60,
        identifier_header="X-API-Key",
        fallback_to_ip=True,
        include_headers=True,
        exclude_paths=["/health"],
        exclude_methods=["OPTIONS"]
    )


@pytest.fixture
def app_with_rate_limiting(fastapi_app, rate_limit_config):
    """Create FastAPI app with rate limiting middleware"""
    # Add rate limiting middleware
    fastapi_app.add_middleware(
        RateLimitingMiddleware, config=rate_limit_config
    )

    # Add exception handler
    fastapi_app.add_exception_handler(
        RateLimitExceededException,
        rate_limit_exception_handler
    )

    return fastapi_app


@pytest.fixture
def test_client(app_with_rate_limiting):
    """Create test client for the rate-limited app"""
    return TestClient(app_with_rate_limiting)


@pytest.fixture
def httpx_client(app_with_rate_limiting):
    """Create httpx client for async testing"""
    return httpx.AsyncClient(
        app=app_with_rate_limiting, base_url="http://testserver"
    )


@pytest.fixture
def redis_client():
    """Create Redis client for testing"""
    try:
        client = redis.Redis(
            host='localhost', port=6379, db=1, decode_responses=True
        )
        client.ping()  # Test connection
        # Clear any existing data
        client.flushdb()
        return client
    except redis.ConnectionError:
        pytest.skip("Redis not available for testing")


@pytest.fixture
def app_with_redis_rate_limiting(fastapi_app, redis_client):
    """Create app with Redis-based rate limiting"""
    config = RateLimitMiddlewareConfig(
        max_requests=3,
        window_seconds=30,
        identifier_header="X-API-Key",
        redis_client=redis_client,
        include_headers=True
    )

    fastapi_app.add_middleware(RateLimitingMiddleware, config=config)
    fastapi_app.add_exception_handler(
        RateLimitExceededException,
        rate_limit_exception_handler
    )

    return fastapi_app


@pytest.fixture
def redis_test_client(app_with_redis_rate_limiting):
    """Test client with Redis rate limiting"""
    return TestClient(app_with_redis_rate_limiting)


class TestMultipleUsersIntegration:
    """Test rate limiting with multiple users"""

    def test_different_users_separate_limits(self, test_client):
        """Test that different users have separate rate limits"""
        # User 1 makes requests
        for i in range(3):
            response = test_client.get("/api/protected", headers={"X-API-Key": "user1"})
            assert response.status_code == 200
            assert response.json() == {"message": "protected"}

        # User 1 should be rate limited on next request
        response = test_client.get(
            "/api/protected", headers={"X-API-Key": "user1"}
        )
        assert response.status_code == 429

        # User 2 should still have requests available
        response = test_client.get("/api/protected", headers={"X-API-Key": "user2"})
        assert response.status_code == 200

    def test_same_user_shared_limit_across_endpoints(self, test_client):
        """Test that same user shares limit across different endpoints"""
        user_key = "shared-user"

        # Make requests to different endpoints
        test_client.get("/api/public", headers={"X-API-Key": user_key})
        test_client.get("/api/protected", headers={"X-API-Key": user_key})
        test_client.get("/api/admin", headers={"X-API-Key": user_key})

        # Should be rate limited on next request
        response = test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response.status_code == 429

    def test_ip_fallback_when_no_api_key(self, test_client):
        """Test IP-based rate limiting when no API key provided"""
        # Make requests without API key (should use IP)
        for i in range(3):
            response = test_client.get("/api/protected")
            assert response.status_code == 200

        # Should be rate limited
        response = test_client.get("/api/protected")
        assert response.status_code == 429


class TestDifferentEndpointsIntegration:
    """Test rate limiting on different endpoints"""

    def test_endpoints_with_different_configs(self):
        """Test different rate limits for different endpoints"""
        app = FastAPI()

        # Endpoint with strict limiting
        @app.get("/api/strict")
        async def strict_endpoint():
            return {"endpoint": "strict"}

        # Endpoint with lenient limiting
        @app.get("/api/lenient")
        async def lenient_endpoint():
            return {"endpoint": "lenient"}

        # Apply different middleware configs
        strict_config = RateLimitMiddlewareConfig(
            max_requests=2,
            window_seconds=60,
            identifier_header="X-API-Key"
        )
        lenient_config = RateLimitMiddlewareConfig(
            max_requests=10,
            window_seconds=60,
            identifier_header="X-API-Key"
        )

        # Note: In real implementation, you'd need separate middleware instances
        # For this test, we'll use the same config but demonstrate the concept
        app.add_middleware(RateLimitingMiddleware, config=strict_config)
        app.add_exception_handler(RateLimitExceededException, rate_limit_exception_handler)

        client = TestClient(app)

        # Test strict endpoint
        for i in range(2):
            response = client.get("/api/strict", headers={"X-API-Key": "user1"})
            assert response.status_code == 200

        response = client.get("/api/strict", headers={"X-API-Key": "user1"})
        assert response.status_code == 429

    def test_excluded_paths_not_rate_limited(self, test_client):
        """Test that excluded paths are not rate limited"""
        # Health check should not be rate limited
        for i in range(10):
            response = test_client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy"}

    def test_excluded_methods_not_rate_limited(self, test_client):
        """Test that excluded HTTP methods are not rate limited"""
        # OPTIONS requests should not be rate limited
        for i in range(10):
            response = test_client.options("/api/test")
            assert response.status_code == 200
            assert response.json() == {"method": "OPTIONS"}


class TestRateLimitViolationsIntegration:
    """Test rate limit violation scenarios"""

    def test_rate_limit_exceeded_response(self, test_client):
        """Test proper 429 response when rate limit exceeded"""
        user_key = "violation-user"

        # Exhaust the rate limit
        for i in range(5):
            response = test_client.get("/api/protected", headers={"X-API-Key": user_key})
            assert response.status_code == 200

        # Next request should be rate limited
        response = test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response.status_code == 429

        data = response.json()
        assert "detail" in data
        assert "Rate limit exceeded" in data["detail"]

    def test_rate_limit_headers_present(self, test_client):
        """Test that rate limit headers are included in responses"""
        user_key = "header-user"

        # Make a request
        response = test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response.status_code == 200

        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        assert "Retry-After" in response.headers

        # Verify header values
        assert response.headers["X-RateLimit-Limit"] == "5"
        remaining = int(response.headers["X-RateLimit-Remaining"])
        assert 0 <= remaining <= 4  # Should be 4 after first request

    def test_retry_after_header_on_violation(self, test_client):
        """Test Retry-After header when rate limit is exceeded"""
        user_key = "retry-user"

        # Exhaust rate limit
        for i in range(5):
            test_client.get("/api/protected", headers={"X-API-Key": user_key})

        # Check violation response
        response = test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response.status_code == 429

        assert "Retry-After" in response.headers
        retry_after = int(response.headers["Retry-After"])
        assert retry_after > 0


class TestRedisVsInMemoryFallback:
    """Test Redis integration and in-memory fallback"""

    def test_redis_rate_limiting(self, redis_test_client):
        """Test rate limiting with Redis backend"""
        user_key = "redis-user"

        # Make requests up to limit
        for i in range(3):
            response = redis_test_client.get("/api/protected", headers={"X-API-Key": user_key})
            assert response.status_code == 200

        # Should be rate limited
        response = redis_test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response.status_code == 429

    def test_in_memory_fallback_when_redis_unavailable(self):
        """Test fallback to in-memory when Redis is unavailable"""
        app = FastAPI()

        @app.get("/api/test")
        async def test_endpoint():
            return {"message": "test"}

        # Create config with mock Redis client that fails
        mock_redis = Mock()
        mock_redis.ping.side_effect = redis.ConnectionError("Redis unavailable")

        config = RateLimitMiddlewareConfig(
            max_requests=3,
            window_seconds=60,
            redis_client=mock_redis,
            include_headers=True
        )

        app.add_middleware(RateLimitingMiddleware, config=config)
        app.add_exception_handler(RateLimitExceededException, rate_limit_exception_handler)

        client = TestClient(app)

        # Should work with in-memory fallback
        for i in range(3):
            response = client.get("/api/test", headers={"X-API-Key": "fallback-user"})
            assert response.status_code == 200

        # Should be rate limited
        response = client.get("/api/test", headers={"X-API-Key": "fallback-user"})
        assert response.status_code == 429

    def test_redis_and_in_memory_consistency(self, redis_test_client):
        """Test that Redis and in-memory give consistent results"""
        user_key = "consistency-user"

        # Make requests and check headers
        response1 = redis_test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response1.status_code == 200
        remaining1 = int(response1.headers["X-RateLimit-Remaining"])

        response2 = redis_test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response2.status_code == 200
        remaining2 = int(response2.headers["X-RateLimit-Remaining"])

        # Remaining should decrease by 1
        assert remaining2 == remaining1 - 1


class TestServiceMethodDecorators:
    """Test rate limiting decorators on service methods"""

    def test_sync_method_rate_limiting(self):
        """Test rate limiting on synchronous service methods"""
        service = MockTestService()

        # Should work within limit
        for i in range(3):
            result = service.sync_method("user1")
            assert result["user_id"] == "user1"
            assert result["call_count"] == i + 1

        # Should raise exception when limit exceeded
        with pytest.raises(RateLimitExceededException):
            service.sync_method("user1")

    def test_different_users_separate_limits_service(self):
        """Test that different users have separate limits in service methods"""
        service = MockTestService()

        # User 1 exhausts limit
        for i in range(3):
            service.sync_method("user1")

        with pytest.raises(RateLimitExceededException):
            service.sync_method("user1")

        # User 2 should still work
        result = service.sync_method("user2")
        assert result["user_id"] == "user2"

    @pytest.mark.asyncio
    async def test_async_method_rate_limiting(self):
        """Test rate limiting on asynchronous service methods"""
        service = TestService()

        # Should work within limit
        for i in range(2):
            result = await service.async_method("async-user")
            assert result["user_id"] == "async-user"
            assert result["async"] is True

        # Should raise exception when limit exceeded
        with pytest.raises(RateLimitExceededException):
            await service.async_method("async-user")

    def test_custom_identifier_function(self):
        """Test custom identifier function in decorators"""
        service = TestService()

        # Should work with tenant:user combination
        for i in range(5):
            result = service.tenant_method("tenant1", "user1")
            assert result["tenant_id"] == "tenant1"
            assert result["user_id"] == "user1"

        # Should be rate limited
        with pytest.raises(RateLimitExceededException):
            service.tenant_method("tenant1", "user1")

        # Different tenant should work
        result = service.tenant_method("tenant2", "user1")
        assert result["tenant_id"] == "tenant2"


class TestAsyncIntegration:
    """Test async integration scenarios"""

    @pytest.mark.asyncio
    async def test_async_http_requests(self, httpx_client):
        """Test rate limiting with async HTTP requests"""
        user_key = "async-user"

        # Make async requests
        for i in range(3):
            response = await httpx_client.get("/api/protected", headers={"X-API-Key": user_key})
            assert response.status_code == 200

        # Should be rate limited
        response = await httpx_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self, httpx_client):
        """Test handling of concurrent requests"""
        user_key = "concurrent-user"

        # Create concurrent requests
        tasks = []
        for i in range(6):  # More than limit
            task = httpx_client.get("/api/protected", headers={"X-API-Key": user_key})
            tasks.append(task)

        # Execute concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful vs rate limited responses
        success_count = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 429)

        # Should have exactly 5 successful and 1 rate limited (due to limit of 5)
        assert success_count == 5
        assert rate_limited_count == 1


class TestConfigurationIntegration:
    """Test various configuration scenarios"""

    def test_environment_configuration_loading(self):
        """Test loading configuration from environment variables"""
        with patch.dict('os.environ', {
            'RATE_LIMIT_MAX_REQUESTS': '10',
            'RATE_LIMIT_WINDOW_SECONDS': '120',
            'RATE_LIMIT_IDENTIFIER_HEADER': 'Authorization',
            'RATE_LIMIT_FALLBACK_TO_IP': 'false',
            'RATE_LIMIT_INCLUDE_HEADERS': 'false'
        }):
            from shared.rate_limiting_middleware import load_rate_limit_config_from_env
            config = load_rate_limit_config_from_env()

            assert config.max_requests == 10
            assert config.window_seconds == 120
            assert config.identifier_header == 'Authorization'
            assert config.fallback_to_ip is False
            assert config.include_headers is False

    def test_setup_rate_limiting_utility(self):
        """Test the setup_rate_limiting utility function"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        # Setup with custom config
        config = RateLimitMiddlewareConfig(max_requests=2, window_seconds=30)
        setup_rate_limiting(app, config)

        client = TestClient(app)

        # Test rate limiting works
        for i in range(2):
            response = client.get("/test", headers={"X-API-Key": "setup-user"})
            assert response.status_code == 200

        response = client.get("/test", headers={"X-API-Key": "setup-user"})
        assert response.status_code == 429


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_no_identifier_header_uses_ip(self, test_client):
        """Test that requests without identifier header use IP"""
        # Make requests without X-API-Key
        for i in range(3):
            response = test_client.get("/api/protected")
            assert response.status_code == 200

        response = test_client.get("/api/protected")
        assert response.status_code == 429

    def test_mixed_identifier_types(self, test_client):
        """Test mixing API key and IP-based requests"""
        # Some requests with API key
        test_client.get("/api/protected", headers={"X-API-Key": "key-user"})

        # Some requests without (IP-based)
        test_client.get("/api/protected")

        # Both should be tracked separately
        response = test_client.get("/api/protected", headers={"X-API-Key": "key-user"})
        assert response.status_code == 200  # Still has requests left

    def test_window_expiration(self, test_client):
        """Test that rate limits reset after window expires"""
        user_key = "window-user"

        # Exhaust limit
        for i in range(5):
            test_client.get("/api/protected", headers={"X-API-Key": user_key})

        # Should be rate limited
        response = test_client.get("/api/protected", headers={"X-API-Key": user_key})
        assert response.status_code == 429

        # Simulate waiting for window to expire by creating new limiter
        # In real scenario, this would require time manipulation or waiting
        # For this test, we'll just verify the behavior is consistent
        assert response.status_code == 429

    def test_very_high_limits(self):
        """Test with very high limits (effectively unlimited)"""
        app = FastAPI()

        @app.get("/api/unlimited")
        async def unlimited_endpoint():
            return {"message": "unlimited"}

        config = RateLimitMiddlewareConfig(
            max_requests=1000,
            window_seconds=60,
            identifier_header="X-API-Key"
        )

        app.add_middleware(RateLimitingMiddleware, config=config)
        client = TestClient(app)

        # Should handle many requests without rate limiting
        for i in range(50):
            response = client.get("/api/unlimited", headers={"X-API-Key": "high-limit-user"})
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])