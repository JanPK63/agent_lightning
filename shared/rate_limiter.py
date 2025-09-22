"""
Sliding Window Rate Limiter for Agent Lightning

This module provides rate limiting functionality using a sliding window
algorithm implemented with Redis sorted sets for distributed systems.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass
import redis

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    max_requests: int
    window_seconds: int
    prefix: str = "ratelimit"
    fallback_enabled: bool = True
    fallback_max_requests: int = 1000  # Higher limit for fallback


class RateLimiter(ABC):
    """Abstract base class for rate limiters"""

    @abstractmethod
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier

        Args:
            identifier: Unique identifier (e.g., user ID, IP address)

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        pass

    @abstractmethod
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests allowed in current window

        Args:
            identifier: Unique identifier

        Returns:
            Number of remaining requests allowed
        """
        pass

    @abstractmethod
    def get_reset_time(self, identifier: str) -> float:
        """Get time when rate limit resets (Unix timestamp)

        Args:
            identifier: Unique identifier

        Returns:
            Unix timestamp when the rate limit resets
        """
        pass


class InMemoryRateLimiter(RateLimiter):
    """In-memory rate limiter for development/testing or as fallback"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed using in-memory storage"""
        now = time.time()
        window_start = now - self.window_seconds

        # Get or create request list for identifier
        if identifier not in self.requests:
            self.requests[identifier] = []

        request_times = self.requests[identifier]

        # Remove old requests outside window
        request_times[:] = [t for t in request_times if t > window_start]

        # Check if under limit
        if len(request_times) < self.max_requests:
            request_times.append(now)
            return True

        return False

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests in current window"""
        now = time.time()
        window_start = now - self.window_seconds

        if identifier not in self.requests:
            return self.max_requests

        request_times = self.requests[identifier]
        # Remove old requests outside window
        valid_requests = [t for t in request_times if t > window_start]

        return max(0, self.max_requests - len(valid_requests))

    def get_reset_time(self, identifier: str) -> float:
        """Get reset time for rate limit"""
        if identifier not in self.requests or not self.requests[identifier]:
            return time.time() + self.window_seconds

        # Reset time is when the oldest request in window expires
        oldest_request = min(self.requests[identifier])
        return oldest_request + self.window_seconds


class RedisSlidingWindowRateLimiter(RateLimiter):
    """Redis-based sliding window rate limiter using sorted sets"""

    def __init__(self, redis_client: redis.Redis, config: RateLimitConfig):
        """Initialize Redis sliding window rate limiter

        Args:
            redis_client: Redis client instance
            config: Rate limiting configuration
        """
        self.redis = redis_client
        self.config = config
        self.fallback_limiter = InMemoryRateLimiter(
            config.fallback_max_requests,
            config.window_seconds
        ) if config.fallback_enabled else None

        # Test Redis connection
        self.redis_available = self._check_redis_connection()

    def _check_redis_connection(self) -> bool:
        """Check if Redis is available"""
        try:
            self.redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed, using fallback: {e}")
            return False

    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for identifier"""
        return f"{self.config.prefix}:{identifier}"

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed using Redis sliding window"""
        if not self.redis_available and self.fallback_limiter:
            logger.debug("Using fallback rate limiter")
            return self.fallback_limiter.is_allowed(identifier)
        elif not self.redis_available:
            logger.warning("Redis unavailable and no fallback configured")
            return True  # Allow all requests if no fallback

        try:
            now = time.time()
            key = self._get_key(identifier)
            window_start = now - self.config.window_seconds

            # Use Redis pipeline for atomic operations
            with self.redis.pipeline() as pipe:
                # Remove old entries
                pipe.zremrangebyscore(key, '-inf', window_start)
                # Count remaining entries
                pipe.zcount(key, window_start, '+inf')
                # Execute pipeline
                results = pipe.execute()

            count = results[1]  # zcount result

            if count < self.config.max_requests:
                # Add current request
                self.redis.zadd(key, {str(now): now})
                # Set expiration on key to auto-cleanup
                self.redis.expire(key, self.config.window_seconds * 2)
                return True

            return False

        except Exception as e:
            logger.error(f"Redis error in is_allowed: {e}")
            if self.fallback_limiter:
                logger.debug("Falling back to in-memory limiter")
                return self.fallback_limiter.is_allowed(identifier)
            return True  # Allow on Redis failure if no fallback

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests allowed in current window"""
        if not self.redis_available and self.fallback_limiter:
            return self.fallback_limiter.get_remaining(identifier)
        elif not self.redis_available:
            return self.config.max_requests

        try:
            now = time.time()
            key = self._get_key(identifier)
            window_start = now - self.config.window_seconds

            # Clean old entries and count
            with self.redis.pipeline() as pipe:
                pipe.zremrangebyscore(key, '-inf', window_start)
                pipe.zcount(key, window_start, '+inf')
                results = pipe.execute()

            count = results[1]  # zcount result
            return max(0, self.config.max_requests - count)

        except Exception as e:
            logger.error(f"Redis error in get_remaining: {e}")
            if self.fallback_limiter:
                return self.fallback_limiter.get_remaining(identifier)
            return self.config.max_requests

    def get_reset_time(self, identifier: str) -> float:
        """Get time when rate limit resets"""
        if not self.redis_available and self.fallback_limiter:
            return self.fallback_limiter.get_reset_time(identifier)
        elif not self.redis_available:
            return time.time() + self.config.window_seconds

        try:
            now = time.time()
            key = self._get_key(identifier)

            # Get the oldest timestamp in the current window
            oldest_scores = self.redis.zrange(key, 0, 0, withscores=True)

            if oldest_scores:
                oldest_time = oldest_scores[0][1]
                return oldest_time + self.config.window_seconds
            else:
                # No requests in window, reset at end of window
                return now + self.config.window_seconds

        except Exception as e:
            logger.error(f"Redis error in get_reset_time: {e}")
            if self.fallback_limiter:
                return self.fallback_limiter.get_reset_time(identifier)
            return time.time() + self.config.window_seconds


def create_rate_limiter(
    redis_client: Optional[redis.Redis] = None,
    config: Optional[RateLimitConfig] = None
) -> RateLimiter:
    """Factory function to create rate limiter

    Args:
        redis_client: Redis client (if None, uses in-memory fallback)
        config: Rate limit configuration

    Returns:
        RateLimiter instance
    """
    if config is None:
        config = RateLimitConfig(
            max_requests=100,
            window_seconds=60,
            prefix="ratelimit"
        )

    if redis_client is None:
        logger.info("No Redis client provided, using in-memory rate limiter")
        return InMemoryRateLimiter(config.max_requests, config.window_seconds)

    return RedisSlidingWindowRateLimiter(redis_client, config)