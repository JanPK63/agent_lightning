"""
Cache decorators and integration helpers for Agent Lightning
Provides easy-to-use caching decorators for functions and methods
"""

import functools
import hashlib
import json
import logging
from typing import Any, Callable, Optional, Union
from datetime import datetime

from shared.cache import CacheManager, get_cache
from shared.events import EventPublisher, EventChannel

logger = logging.getLogger(__name__)

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate cache key from function arguments
    
    Args:
        prefix: Key prefix
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Cache key string
    """
    # Create a unique key from arguments
    key_data = {
        'args': args,
        'kwargs': kwargs
    }
    
    # Convert to JSON and hash
    key_json = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_json.encode()).hexdigest()
    
    return f"{prefix}:{key_hash}"

def cached(ttl: int = 3600, prefix: str = None, 
          key_func: Callable = None, invalidate_on: list = None):
    """Cache decorator for functions
    
    Args:
        ttl: Time to live in seconds (default: 1 hour)
        prefix: Cache key prefix (default: function name)
        key_func: Custom function to generate cache key
        invalidate_on: List of event channels that invalidate cache
    
    Example:
        @cached(ttl=600, prefix="agent")
        def get_agent(agent_id):
            return db.query(Agent).get(agent_id)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache instance
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_prefix = prefix or f"{func.__module__}.{func.__name__}"
                cache_key = generate_cache_key(key_prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss for {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        
        # Store cache metadata
        wrapper._cache_ttl = ttl
        wrapper._cache_prefix = prefix
        wrapper._cache_invalidate_on = invalidate_on or []
        
        # Add cache invalidation method
        def invalidate(*args, **kwargs):
            cache = get_cache()
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_prefix = prefix or f"{func.__module__}.{func.__name__}"
                cache_key = generate_cache_key(key_prefix, *args, **kwargs)
            cache.delete(cache_key)
            logger.debug(f"Invalidated cache for {cache_key}")
        
        wrapper.invalidate = invalidate
        
        return wrapper
    return decorator

def cache_aside(loader: Callable, ttl: int = 3600, prefix: str = None):
    """Cache-aside pattern decorator
    
    Args:
        loader: Function to load data if cache miss
        ttl: Time to live in seconds
        prefix: Cache key prefix
    
    Example:
        @cache_aside(loader=load_from_db, ttl=600)
        def get_user(user_id):
            pass  # Will use loader if cache miss
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key_prefix = prefix or f"{func.__module__}.{func.__name__}"
            cache_key = generate_cache_key(key_prefix, *args, **kwargs)
            
            # Use cache-aside pattern
            return cache.cache_aside(cache_key, lambda: loader(*args, **kwargs), ttl)
        
        return wrapper
    return decorator

def write_through(writer: Callable, ttl: int = 3600, prefix: str = None):
    """Write-through cache pattern decorator
    
    Args:
        writer: Function to write to persistent storage
        ttl: Time to live in seconds
        prefix: Cache key prefix
    
    Example:
        @write_through(writer=save_to_db, ttl=600)
        def update_user(user_id, data):
            return data  # Will write through to DB and cache
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key_prefix = prefix or f"{func.__module__}.{func.__name__}"
            cache_key = generate_cache_key(key_prefix, *args, **kwargs)
            
            # Get the value to write
            value = func(*args, **kwargs)
            
            # Use write-through pattern
            return cache.write_through(cache_key, value, 
                                      lambda v: writer(*args, **kwargs), ttl)
        
        return wrapper
    return decorator

def invalidate_cache(patterns: Union[str, list]):
    """Decorator to invalidate cache after function execution
    
    Args:
        patterns: Cache key pattern(s) to invalidate
    
    Example:
        @invalidate_cache("agent:*")
        def delete_agent(agent_id):
            db.delete(Agent, agent_id)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidate cache patterns
            cache = get_cache()
            if isinstance(patterns, str):
                cache.delete_pattern(patterns)
                logger.debug(f"Invalidated cache pattern: {patterns}")
            else:
                for pattern in patterns:
                    cache.delete_pattern(pattern)
                    logger.debug(f"Invalidated cache pattern: {pattern}")
            
            return result
        
        return wrapper
    return decorator

def cache_lock(resource: str = None, timeout: int = 30):
    """Distributed lock decorator
    
    Args:
        resource: Resource to lock (default: function name)
        timeout: Lock timeout in seconds
    
    Example:
        @cache_lock(resource="critical_operation", timeout=60)
        def critical_operation():
            # Only one instance can run at a time
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Determine resource name
            lock_resource = resource or f"{func.__module__}.{func.__name__}"
            
            # Use lock context manager
            with cache.lock(lock_resource, timeout=timeout):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def rate_limit(max_calls: int, period: int, key_func: Callable = None):
    """Rate limiting decorator using Redis
    
    Args:
        max_calls: Maximum number of calls
        period: Time period in seconds
        key_func: Function to generate rate limit key
    
    Example:
        @rate_limit(max_calls=10, period=60)  # 10 calls per minute
        def api_endpoint(user_id):
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate rate limit key
            if key_func:
                rate_key = f"rate:{key_func(*args, **kwargs)}"
            else:
                rate_key = f"rate:{func.__module__}.{func.__name__}"
            
            # Check rate limit
            current = cache.redis_client.incr(rate_key)
            
            if current == 1:
                # First call, set expiry
                cache.redis_client.expire(rate_key, period)
            
            if current > max_calls:
                raise Exception(f"Rate limit exceeded: {max_calls} calls per {period} seconds")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def memoize(ttl: int = 3600):
    """Simple memoization decorator
    
    Args:
        ttl: Time to live in seconds
    
    Example:
        @memoize(ttl=300)
        def expensive_calculation(x, y):
            return x ** y
    """
    return cached(ttl=ttl)

class CachedProperty:
    """Cached property descriptor
    
    Example:
        class MyClass:
            @CachedProperty(ttl=600)
            def expensive_property(self):
                return calculate_expensive_value()
    """
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.func = None
    
    def __call__(self, func):
        self.func = func
        return self
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        cache = get_cache()
        cache_key = f"{objtype.__name__}.{self.func.__name__}:{id(obj)}"
        
        # Try cache
        value = cache.get(cache_key)
        if value is not None:
            return value
        
        # Calculate value
        value = self.func(obj)
        cache.set(cache_key, value, self.ttl)
        
        return value

# ==================== Cache Warming ====================

def warm_agent_cache():
    """Warm cache with agent data"""
    from shared.database import get_db
    from shared.models import Agent
    
    cache = get_cache()
    
    def load_agents():
        with get_db() as db:
            agents = db.query(Agent).all()
            for agent in agents:
                yield f"agent:{agent.id}", agent.to_dict(), 3600
                
            # Also cache agent list
            agent_list = [a.to_dict() for a in agents]
            yield "agents:all", agent_list, 300
    
    return cache.warm_cache(load_agents)

def warm_knowledge_cache():
    """Warm cache with knowledge base data"""
    from shared.database import get_db
    from shared.models import Knowledge
    
    cache = get_cache()
    
    def load_knowledge():
        with get_db() as db:
            # Get recent, high-usage knowledge items
            knowledge_items = db.query(Knowledge)\
                .filter(Knowledge.usage_count > 5)\
                .order_by(Knowledge.usage_count.desc())\
                .limit(100)\
                .all()
            
            for item in knowledge_items:
                key = f"knowledge:{item.agent_id}:{item.id}"
                yield key, item.to_dict(), 21600  # 6 hours
    
    return cache.warm_cache(load_knowledge)

# ==================== Cache Management ====================

class CacheManager:
    """High-level cache management utilities"""
    
    @staticmethod
    def clear_agent_cache(agent_id: str = None):
        """Clear agent-related cache
        
        Args:
            agent_id: Specific agent ID (None for all)
        """
        cache = get_cache()
        
        if agent_id:
            cache.delete(f"agent:{agent_id}")
            cache.delete_pattern(f"task:*:{agent_id}:*")
            cache.delete_pattern(f"knowledge:{agent_id}:*")
        else:
            cache.delete_pattern("agent:*")
            cache.delete_pattern("task:*")
            cache.delete_pattern("knowledge:*")
        
        # Invalidate agent list
        cache.delete("agents:all")
    
    @staticmethod
    def clear_task_cache(task_id: str = None):
        """Clear task-related cache
        
        Args:
            task_id: Specific task ID (None for all)
        """
        cache = get_cache()
        
        if task_id:
            cache.delete(f"task:{task_id}")
            cache.delete(f"task:result:{task_id}")
        else:
            cache.delete_pattern("task:*")
    
    @staticmethod
    def get_cache_stats() -> dict:
        """Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        cache = get_cache()
        info = cache.get_info()
        
        return {
            'used_memory': info.get('used_memory_human', 'N/A'),
            'connected_clients': info.get('connected_clients', 0),
            'total_commands': info.get('total_commands_processed', 0),
            'keyspace_hits': info.get('keyspace_hits', 0),
            'keyspace_misses': info.get('keyspace_misses', 0),
            'hit_ratio': calculate_hit_ratio(
                info.get('keyspace_hits', 0),
                info.get('keyspace_misses', 0)
            )
        }

def calculate_hit_ratio(hits: int, misses: int) -> float:
    """Calculate cache hit ratio
    
    Args:
        hits: Number of cache hits
        misses: Number of cache misses
        
    Returns:
        Hit ratio as percentage
    """
    total = hits + misses
    if total == 0:
        return 0.0
    return (hits / total) * 100