"""
Redis Cache Manager for Agent Lightning
Handles caching, sessions, and pub/sub messaging
"""

import redis
import json
import os
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.client = None
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600):
        """Set cache with TTL"""
        if not self.client:
            return False
        try:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            return self.client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.client:
            return None
        try:
            value = self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except:
                    return value
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def delete_cache(self, key: str):
        """Delete cache key"""
        if not self.client:
            return False
        try:
            return self.client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def set_session(self, session_id: str, data: dict, ttl: int = 86400):
        """Store session data"""
        return self.set_cache(f"session:{session_id}", data, ttl)
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        return self.get_cache(f"session:{session_id}")
    
    def health_check(self) -> bool:
        """Check Redis health"""
        if not self.client:
            return False
        try:
            return self.client.ping()
        except:
            return False

# Global Redis manager
redis_manager = RedisManager()