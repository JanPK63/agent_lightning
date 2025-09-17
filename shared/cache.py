"""
Redis Cache Manager for Agent Lightning
Provides caching, pub/sub, and distributed locking capabilities
"""

import redis
import json
import pickle
import hashlib
import uuid
import logging
from typing import Any, Optional, List, Dict, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager
import os

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages Redis cache operations, pub/sub, and distributed locking"""
    
    def __init__(self, host: str = None, port: int = None, db: int = 0, 
                 password: str = None, max_connections: int = 50):
        """Initialize Redis cache manager
        
        Args:
            host: Redis host (default: localhost)
            port: Redis port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password (optional)
            max_connections: Maximum connection pool size
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.db = db
        self.password = password or os.getenv('REDIS_PASSWORD', None)
        
        # Create connection pool
        pool_kwargs = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'max_connections': max_connections,
            'socket_keepalive': True,
            'decode_responses': False
        }
        
        # Add socket keepalive options only on Linux
        import platform
        if platform.system() == 'Linux':
            pool_kwargs['socket_keepalive_options'] = {
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 3,  # TCP_KEEPCNT
            }
        
        if self.password:
            pool_kwargs['password'] = self.password
        
        self.pool = redis.ConnectionPool(**pool_kwargs)
        self.redis_client = redis.Redis(connection_pool=self.pool)
        self.pubsub = None
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    # ==================== Cache Operations ====================
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if successful
        """
        try:
            serialized = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache key
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if TTL was set
        """
        try:
            return bool(self.redis_client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get remaining TTL for key
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -2
    
    # ==================== Batch Operations ====================
    
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values
        
        Args:
            keys: List of cache keys
            
        Returns:
            List of values (None for missing keys)
        """
        try:
            values = self.redis_client.mget(keys)
            return [pickle.loads(v) if v else None for v in values]
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            return [None] * len(keys)
    
    def mset(self, mapping: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set multiple values
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            pipe = self.redis_client.pipeline()
            for key, value in mapping.items():
                serialized = pickle.dumps(value)
                pipe.setex(key, ttl, serialized)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern
        
        Args:
            pattern: Key pattern (e.g., "agent:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache delete_pattern error for {pattern}: {e}")
            return 0
    
    # ==================== Cache Patterns ====================
    
    def cache_aside(self, key: str, loader: Callable, ttl: int = 3600) -> Any:
        """Cache-aside pattern (lazy loading)
        
        Args:
            key: Cache key
            loader: Function to load data if cache miss
            ttl: Time to live in seconds
            
        Returns:
            Cached or loaded value
        """
        # Try cache first
        value = self.get(key)
        if value is not None:
            logger.debug(f"Cache hit for {key}")
            return value
        
        # Load from source
        logger.debug(f"Cache miss for {key}, loading...")
        value = loader()
        
        # Store in cache
        if value is not None:
            self.set(key, value, ttl)
        
        return value
    
    def write_through(self, key: str, value: Any, writer: Callable, ttl: int = 3600) -> Any:
        """Write-through pattern
        
        Args:
            key: Cache key
            value: Value to write
            writer: Function to write to persistent storage
            ttl: Time to live in seconds
            
        Returns:
            Written value
        """
        # Write to persistent storage first
        result = writer(value)
        
        # Update cache
        self.set(key, result, ttl)
        
        return result
    
    # ==================== Pub/Sub Operations ====================
    
    def publish(self, channel: str, message: dict) -> int:
        """Publish message to channel
        
        Args:
            channel: Channel name
            message: Message dictionary
            
        Returns:
            Number of subscribers that received the message
        """
        try:
            serialized = json.dumps(message, default=str)
            return self.redis_client.publish(channel, serialized)
        except Exception as e:
            logger.error(f"Publish error to channel {channel}: {e}")
            return 0
    
    def subscribe(self, channels: List[str]) -> redis.client.PubSub:
        """Subscribe to channels
        
        Args:
            channels: List of channel names
            
        Returns:
            PubSub object for listening
        """
        try:
            if not self.pubsub:
                self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe(*channels)
            logger.info(f"Subscribed to channels: {channels}")
            return self.pubsub
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
            raise
    
    def unsubscribe(self, channels: List[str] = None):
        """Unsubscribe from channels
        
        Args:
            channels: List of channels to unsubscribe (None for all)
        """
        try:
            if self.pubsub:
                if channels:
                    self.pubsub.unsubscribe(*channels)
                else:
                    self.pubsub.unsubscribe()
                logger.info(f"Unsubscribed from channels: {channels or 'all'}")
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
    
    def listen(self, timeout: int = None) -> Any:
        """Listen for messages
        
        Args:
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            Message or None if timeout
        """
        try:
            if not self.pubsub:
                raise ValueError("Not subscribed to any channels")
            
            message = self.pubsub.get_message(timeout=timeout)
            if message and message['type'] == 'message':
                message['data'] = json.loads(message['data'])
            return message
        except Exception as e:
            logger.error(f"Listen error: {e}")
            return None
    
    # ==================== Distributed Locking ====================
    
    @contextmanager
    def lock(self, resource: str, timeout: int = 30, blocking: bool = True, 
             blocking_timeout: int = 5):
        """Distributed lock context manager
        
        Args:
            resource: Resource to lock
            timeout: Lock timeout in seconds
            blocking: Whether to wait for lock
            blocking_timeout: How long to wait for lock
            
        Yields:
            Lock identifier
        """
        lock_key = f"lock:{resource}"
        identifier = str(uuid.uuid4())
        acquired = False
        
        try:
            # Try to acquire lock
            if blocking:
                start_time = datetime.now()
                while (datetime.now() - start_time).seconds < blocking_timeout:
                    acquired = self.redis_client.set(
                        lock_key, identifier, nx=True, ex=timeout
                    )
                    if acquired:
                        break
                    # Short sleep before retry
                    import time
                    time.sleep(0.1)
            else:
                acquired = self.redis_client.set(
                    lock_key, identifier, nx=True, ex=timeout
                )
            
            if acquired:
                logger.debug(f"Acquired lock for {resource}")
                yield identifier
            else:
                raise TimeoutError(f"Failed to acquire lock for {resource}")
                
        finally:
            if acquired:
                # Release lock only if we own it
                current = self.redis_client.get(lock_key)
                if current and current.decode() == identifier:
                    self.redis_client.delete(lock_key)
                    logger.debug(f"Released lock for {resource}")
    
    def acquire_lock(self, resource: str, timeout: int = 30) -> Optional[str]:
        """Acquire distributed lock
        
        Args:
            resource: Resource to lock
            timeout: Lock timeout in seconds
            
        Returns:
            Lock identifier if acquired, None otherwise
        """
        lock_key = f"lock:{resource}"
        identifier = str(uuid.uuid4())
        
        acquired = self.redis_client.set(
            lock_key, identifier, nx=True, ex=timeout
        )
        
        if acquired:
            logger.debug(f"Acquired lock for {resource}")
            return identifier
        return None
    
    def release_lock(self, resource: str, identifier: str) -> bool:
        """Release distributed lock
        
        Args:
            resource: Resource to unlock
            identifier: Lock identifier
            
        Returns:
            True if lock was released
        """
        lock_key = f"lock:{resource}"
        
        # Use Lua script for atomic check-and-delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        try:
            result = self.redis_client.eval(lua_script, 1, lock_key, identifier)
            if result:
                logger.debug(f"Released lock for {resource}")
            return bool(result)
        except Exception as e:
            logger.error(f"Release lock error for {resource}: {e}")
            return False
    
    # ==================== Cache Warming ====================
    
    def warm_cache(self, data_loader: Callable[[], List[tuple]]) -> int:
        """Pre-populate cache with data
        
        Args:
            data_loader: Function returning list of (key, value, ttl) tuples
            
        Returns:
            Number of keys warmed
        """
        try:
            count = 0
            for key, value, ttl in data_loader():
                if self.set(key, value, ttl):
                    count += 1
            logger.info(f"Warmed {count} cache keys")
            return count
        except Exception as e:
            logger.error(f"Cache warming error: {e}")
            return 0
    
    # ==================== Utility Methods ====================
    
    def clear_cache(self) -> bool:
        """Clear all cache data (use with caution!)
        
        Returns:
            True if successful
        """
        try:
            self.redis_client.flushdb()
            logger.warning("Cleared all cache data")
            return True
        except Exception as e:
            logger.error(f"Clear cache error: {e}")
            return False
    
    def get_info(self) -> dict:
        """Get Redis server info
        
        Returns:
            Server information dictionary
        """
        try:
            return self.redis_client.info()
        except Exception as e:
            logger.error(f"Get info error: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check Redis health
        
        Returns:
            True if Redis is healthy
        """
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def close(self):
        """Close Redis connections"""
        try:
            if self.pubsub:
                self.pubsub.close()
            self.pool.disconnect()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Close error: {e}")

# Global cache manager instance
cache_manager = None

def get_cache() -> CacheManager:
    """Get global cache manager instance
    
    Returns:
        CacheManager instance
    """
    global cache_manager
    if not cache_manager:
        cache_manager = CacheManager()
    return cache_manager