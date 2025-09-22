"""
Unit tests for rate limiting functionality

Tests both in-memory and Redis-based sliding window rate limiters
"""

import pytest
import time
import unittest.mock as mock

from shared.rate_limiter import (
    InMemoryRateLimiter,
    RedisSlidingWindowRateLimiter,
    RateLimitConfig,
    create_rate_limiter
)


class TestInMemoryRateLimiter:
    """Test in-memory rate limiter implementation"""

    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality"""
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=1)

        # Should allow first 3 requests
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True

        # Fourth request should be denied
        assert limiter.is_allowed("user1") is False

    def test_window_sliding(self):
        """Test that window slides over time"""
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=0.1)

        # Use up the limit
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False  # Should be denied

        # Wait for window to slide
        time.sleep(0.11)

        # Should allow again
        assert limiter.is_allowed("user1") is True

    def test_different_identifiers(self):
        """Test that different identifiers are tracked separately"""
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=1)

        # User1 uses up limit
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False

        # User2 should still be allowed
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user2") is False

    def test_get_remaining(self):
        """Test getting remaining requests"""
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=1)

        # Initially should have full limit
        assert limiter.get_remaining("user1") == 3

        # After one request
        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 2

        # After two more
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 0

    def test_get_reset_time(self):
        """Test getting reset time"""
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=0.1)

        # Before any requests
        reset_time = limiter.get_reset_time("user1")
        assert reset_time > time.time()

        # After request
        limiter.is_allowed("user1")
        reset_time = limiter.get_reset_time("user1")
        assert reset_time > time.time()
        assert reset_time <= time.time() + 0.1


class TestRedisSlidingWindowRateLimiter:
    """Test Redis-based rate limiter"""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis_mock = mock.MagicMock()
        redis_mock.ping.return_value = True

        # Mock pipeline
        pipeline_mock = mock.MagicMock()
        pipeline_mock.__enter__ = mock.MagicMock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = mock.MagicMock(return_value=None)
        pipeline_mock.zremrangebyscore.return_value = None
        pipeline_mock.zcount.return_value = None
        # zremrangebyscore result, zcount result
        pipeline_mock.execute.return_value = [0, 1]

        redis_mock.pipeline.return_value = pipeline_mock
        redis_mock.zadd.return_value = 1
        redis_mock.zrange.return_value = []
        redis_mock.expire.return_value = True
        return redis_mock

    @pytest.fixture
    def config(self):
        """Rate limit configuration"""
        return RateLimitConfig(
            max_requests=3,
            window_seconds=60,
            prefix="test_ratelimit"
        )

    def test_initialization(self, mock_redis, config):
        """Test proper initialization"""
        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        assert limiter.redis == mock_redis
        assert limiter.config == config
        assert limiter.redis_available is True
        assert limiter.fallback_limiter is not None

    def test_redis_connection_failure(self, config):
        """Test behavior when Redis is unavailable"""
        mock_redis = mock.MagicMock()
        mock_redis.ping.side_effect = Exception("Connection failed")

        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        assert limiter.redis_available is False
        # Should still work with fallback
        assert limiter.fallback_limiter is not None

    def test_is_allowed_basic(self, mock_redis, config):
        """Test basic allow/deny logic"""
        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Get the pipeline mock
        pipeline_mock = mock_redis.pipeline.return_value

        assert limiter.is_allowed("user1") is True

        # Verify Redis calls - now on pipeline
        pipeline_mock.zremrangebyscore.assert_called_once()
        pipeline_mock.zcount.assert_called_once()
        mock_redis.zadd.assert_called_once()

    def test_is_allowed_deny(self, mock_redis, config):
        """Test denying request when limit exceeded"""
        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Get the pipeline mock and set it to return count at limit
        pipeline_mock = mock_redis.pipeline.return_value
        pipeline_mock.execute.return_value = [0, 3]  # At limit

        assert limiter.is_allowed("user1") is False

        # Should not add to Redis when denied
        mock_redis.zadd.assert_not_called()

    def test_fallback_on_redis_error(self, config):
        """Test fallback to in-memory when Redis fails"""
        mock_redis = mock.MagicMock()
        mock_redis.ping.return_value = True

        # Make pipeline operations fail
        pipeline_mock = mock.MagicMock()
        pipeline_mock.__enter__ = mock.MagicMock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = mock.MagicMock(return_value=None)
        pipeline_mock.execute.side_effect = Exception("Redis error")
        mock_redis.pipeline.return_value = pipeline_mock

        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Should use fallback
        assert limiter.is_allowed("user1") is True  # Fallback allows
        assert limiter.is_allowed("user1") is True

    def test_get_remaining(self, mock_redis, config):
        """Test getting remaining requests"""
        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        mock_redis.zremrangebyscore.return_value = 0
        mock_redis.zcount.return_value = 1  # 1 request in window

        remaining = limiter.get_remaining("user1")
        assert remaining == 2  # 3 - 1

    def test_get_reset_time(self, mock_redis, config):
        """Test getting reset time"""
        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Mock having requests in window
        mock_redis.zrange.return_value = [(b'timestamp', 1234567890.0)]

        reset_time = limiter.get_reset_time("user1")
        assert reset_time == 1234567890.0 + 60  # timestamp + window

    def test_get_reset_time_no_requests(self, mock_redis, config):
        """Test reset time when no requests in window"""
        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        mock_redis.zrange.return_value = []

        reset_time = limiter.get_reset_time("user1")
        assert reset_time > time.time()
        assert reset_time <= time.time() + 60

    def test_key_generation(self, mock_redis, config):
        """Test Redis key generation"""
        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        key = limiter._get_key("user123")
        assert key == "test_ratelimit:user123"

    def test_no_fallback_config(self):
        """Test behavior without fallback configured"""
        config = RateLimitConfig(
            max_requests=3,
            window_seconds=60,
            fallback_enabled=False
        )

        mock_redis = mock.MagicMock()
        mock_redis.ping.return_value = True

        # Make pipeline operations fail
        pipeline_mock = mock.MagicMock()
        pipeline_mock.__enter__ = mock.MagicMock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = mock.MagicMock(return_value=None)
        pipeline_mock.execute.side_effect = Exception("Redis error")
        mock_redis.pipeline.return_value = pipeline_mock

        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Should allow on Redis failure when no fallback
        assert limiter.is_allowed("user1") is True


class TestRateLimitConfig:
    """Test configuration dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = RateLimitConfig(max_requests=10, window_seconds=30)

        assert config.max_requests == 10
        assert config.window_seconds == 30
        assert config.prefix == "ratelimit"
        assert config.fallback_enabled is True
        assert config.fallback_max_requests == 1000


class TestCreateRateLimiter:
    """Test factory function"""

    def test_create_in_memory(self):
        """Test creating in-memory rate limiter"""
        limiter = create_rate_limiter()

        assert isinstance(limiter, InMemoryRateLimiter)

    def test_create_redis(self):
        """Test creating Redis rate limiter"""
        mock_redis = mock.MagicMock()
        mock_redis.ping.return_value = True

        config = RateLimitConfig(max_requests=5, window_seconds=30)
        limiter = create_rate_limiter(mock_redis, config)

        assert isinstance(limiter, RedisSlidingWindowRateLimiter)
        assert limiter.config == config


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""

    def test_burst_traffic_simulation(self):
        """Simulate burst traffic pattern"""
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=0.2)

        # Simulate burst
        allowed_count = 0
        for i in range(10):
            if limiter.is_allowed("api_user"):
                allowed_count += 1

        assert allowed_count == 5

        # Wait for window to reset
        time.sleep(0.21)

        # Should allow again
        assert limiter.is_allowed("api_user") is True

    def test_multiple_users_concurrent(self):
        """Test multiple users with different patterns"""
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=0.1)

        users = ["user1", "user2", "user3"]

        # Each user makes requests
        for user in users:
            assert limiter.is_allowed(user) is True
            assert limiter.is_allowed(user) is True
            assert limiter.is_allowed(user) is False  # Third denied

        # Different user should be allowed
        assert limiter.is_allowed("user4") is True

    def test_window_boundary_behavior(self):
        """Test behavior at window boundaries"""
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=0.1)

        # Fill the window
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False

        # Wait just enough for one request to expire
        time.sleep(0.05)  # Half window

        # Should still be denied (sliding window)
        assert limiter.is_allowed("user1") is False

        # Wait for full window
        time.sleep(0.06)
        assert limiter.is_allowed("user1") is True


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_redis_connection_timeout(self):
        """Test handling Redis connection timeouts"""
        config = RateLimitConfig(max_requests=3, window_seconds=60)

        mock_redis = mock.MagicMock()
        mock_redis.ping.side_effect = TimeoutError("Connection timeout")

        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Should fallback gracefully
        assert limiter.redis_available is False
        assert limiter.is_allowed("user1") is True  # Fallback allows

    def test_redis_operation_failure(self):
        """Test handling Redis operation failures during requests"""
        config = RateLimitConfig(max_requests=3, window_seconds=60)

        mock_redis = mock.MagicMock()
        mock_redis.ping.return_value = True

        # Make pipeline operations fail
        pipeline_mock = mock.MagicMock()
        pipeline_mock.__enter__ = mock.MagicMock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = mock.MagicMock(return_value=None)
        pipeline_mock.execute.side_effect = ConnectionError("Operation failed")
        mock_redis.pipeline.return_value = pipeline_mock

        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Should handle error and use fallback
        result = limiter.is_allowed("user1")
        assert result is True  # Fallback allows

    def test_fallback_disabled_redis_failure(self):
        """Test behavior when fallback disabled and Redis fails"""
        config = RateLimitConfig(
            max_requests=3,
            window_seconds=60,
            fallback_enabled=False
        )

        mock_redis = mock.MagicMock()
        mock_redis.ping.return_value = True

        # Make pipeline operations fail
        pipeline_mock = mock.MagicMock()
        pipeline_mock.__enter__ = mock.MagicMock(return_value=pipeline_mock)
        pipeline_mock.__exit__ = mock.MagicMock(return_value=None)
        pipeline_mock.execute.side_effect = Exception("Redis error")
        mock_redis.pipeline.return_value = pipeline_mock

        limiter = RedisSlidingWindowRateLimiter(mock_redis, config)

        # Should allow requests when Redis fails and no fallback
        assert limiter.is_allowed("user1") is True


if __name__ == "__main__":
    pytest.main([__file__])