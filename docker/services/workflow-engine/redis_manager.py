"""
Enterprise Redis Manager
Production-grade Redis operations with connection pooling and monitoring
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
import redis.asyncio as redis
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RedisManager:
    """Enterprise Redis manager with connection pooling and monitoring"""
    
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        self.pool = None
        self.client = None
        
    async def initialize(self):
        """Initialize Redis connection pool"""
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            logger.info("Redis connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def set_async(self, key: str, value: str, ttl: int = None):
        """Set key-value pair with optional TTL"""
        try:
            if ttl:
                await self.client.setex(key, ttl, value)
            else:
                await self.client.set(key, value)
        except Exception as e:
            logger.error(f"Redis SET failed for key {key}: {e}")
            raise
    
    async def get_async(self, key: str) -> Optional[str]:
        """Get value by key"""
        try:
            result = await self.client.get(key)
            return result.decode('utf-8') if result else None
        except Exception as e:
            logger.error(f"Redis GET failed for key {key}: {e}")
            return None
    
    async def delete_async(self, key: str):
        """Delete key"""
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE failed for key {key}: {e}")
    
    async def exists_async(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS failed for key {key}: {e}")
            return False
    
    async def set_json(self, key: str, data: Dict[str, Any], ttl: int = None):
        """Set JSON data"""
        try:
            json_str = json.dumps(data, default=str)
            await self.set_async(key, json_str, ttl)
        except Exception as e:
            logger.error(f"Redis SET JSON failed for key {key}: {e}")
            raise
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON data"""
        try:
            json_str = await self.get_async(key)
            return json.loads(json_str) if json_str else None
        except Exception as e:
            logger.error(f"Redis GET JSON failed for key {key}: {e}")
            return None
    
    async def publish(self, channel: str, message: str):
        """Publish message to channel"""
        try:
            await self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Redis PUBLISH failed for channel {channel}: {e}")
    
    async def subscribe(self, channel: str):
        """Subscribe to channel"""
        try:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            logger.error(f"Redis SUBSCRIBE failed for channel {channel}: {e}")
            return None
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCREMENT failed for key {key}: {e}")
            return 0
    
    async def set_hash(self, key: str, field: str, value: str):
        """Set hash field"""
        try:
            await self.client.hset(key, field, value)
        except Exception as e:
            logger.error(f"Redis HSET failed for key {key}, field {field}: {e}")
    
    async def get_hash(self, key: str, field: str) -> Optional[str]:
        """Get hash field"""
        try:
            result = await self.client.hget(key, field)
            return result.decode('utf-8') if result else None
        except Exception as e:
            logger.error(f"Redis HGET failed for key {key}, field {field}: {e}")
            return None
    
    async def get_all_hash(self, key: str) -> Dict[str, str]:
        """Get all hash fields"""
        try:
            result = await self.client.hgetall(key)
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in result.items()}
        except Exception as e:
            logger.error(f"Redis HGETALL failed for key {key}: {e}")
            return {}
    
    async def add_to_list(self, key: str, value: str):
        """Add value to list"""
        try:
            await self.client.lpush(key, value)
        except Exception as e:
            logger.error(f"Redis LPUSH failed for key {key}: {e}")
    
    async def get_list(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """Get list values"""
        try:
            result = await self.client.lrange(key, start, end)
            return [item.decode('utf-8') for item in result]
        except Exception as e:
            logger.error(f"Redis LRANGE failed for key {key}: {e}")
            return []
    
    async def add_to_set(self, key: str, value: str):
        """Add value to set"""
        try:
            await self.client.sadd(key, value)
        except Exception as e:
            logger.error(f"Redis SADD failed for key {key}: {e}")
    
    async def get_set_members(self, key: str) -> List[str]:
        """Get set members"""
        try:
            result = await self.client.smembers(key)
            return [item.decode('utf-8') for item in result]
        except Exception as e:
            logger.error(f"Redis SMEMBERS failed for key {key}: {e}")
            return []
    
    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        try:
            result = await self.client.keys(pattern)
            return [key.decode('utf-8') for key in result]
        except Exception as e:
            logger.error(f"Redis KEYS failed for pattern {pattern}: {e}")
            return []
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server info"""
        try:
            info = await self.client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Redis INFO failed: {e}")
            return {}
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()


# Global Redis manager instance
redis_manager = RedisManager()