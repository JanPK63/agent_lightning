# Sliding Window Rate Limiting Design for Agent Lightning

## Overview

This document outlines the design for implementing sliding window rate limiting in the Agent Lightning framework. The sliding window algorithm provides smoother rate limiting compared to fixed windows by using a rolling time window to track requests.

## Algorithm Description

The sliding window rate limiting algorithm maintains a sorted collection of request timestamps for each identifier (e.g., user ID, IP address, API key). For each incoming request:

1. Remove timestamps older than the current window duration
2. Count the remaining timestamps in the current window
3. If the count is less than the maximum allowed requests, allow the request and add the current timestamp
4. Otherwise, deny the request

This approach provides more accurate rate limiting than fixed windows, as it doesn't suffer from the "boundary problem" where requests cluster at window boundaries.

## Data Structures

### Redis-based Implementation (Recommended for Distributed Systems)

- **Primary Structure**: Redis Sorted Sets
  - Key format: `ratelimit:{identifier}:{window_type}`
  - Members: Request timestamps (stored as strings for precision)
  - Scores: Timestamps as floating-point numbers for sorting and range queries

**Operations**:
- Add request: `ZADD key timestamp timestamp`
- Count requests in window: `ZCOUNT key (current_time - window) +inf`
- Clean old entries: `ZREMRANGEBYSCORE key -inf (current_time - window)`

### In-Memory Implementation (For Single Instance/Development)

- **Primary Structure**: Dictionary of sorted lists or deques
  - Key: Identifier string
  - Value: Sorted list of timestamps (as floats)

**Operations**:
- Add request: Append timestamp and maintain sort
- Count requests: Binary search for timestamps within window
- Clean old entries: Remove timestamps outside window during count operation

## Storage Options Comparison

### Redis Storage
**Pros**:
- Distributed across multiple services/instances
- Persistent across service restarts
- Atomic operations prevent race conditions
- Scalable with Redis clustering

**Cons**:
- Network latency for Redis operations
- Additional infrastructure dependency
- Potential single point of failure (mitigated by Redis clustering)

**Best For**: Production distributed deployments

### In-Memory Storage
**Pros**:
- Lowest latency
- No external dependencies
- Simple implementation

**Cons**:
- Not shared across instances
- Lost on service restart
- Race conditions in multi-threaded environments

**Best For**: Single instance, development, or as fallback

## Distributed Systems Considerations

### Clock Synchronization
- All instances must use NTP-synchronized clocks
- Timestamps should use UTC to avoid timezone issues
- Consider using Redis TIME command for server-side timestamp generation

### Race Conditions and Atomicity
- Redis operations are atomic, preventing concurrent modification issues
- Use Redis transactions (MULTI/EXEC) for complex operations if needed
- In-memory implementation requires locks for thread safety

### Consistency and Scalability
- Eventual consistency is acceptable for rate limiting (slight over/under counting)
- Redis can handle high throughput with proper configuration
- Consider Redis sharding for very high-scale deployments

### Failure Handling
- **Redis Failure**: Fallback to in-memory with reduced limits or allow-all mode
- **Network Issues**: Implement timeouts and retry logic
- **Instance Failure**: Rate limits reset on restart (acceptable for most use cases)

### Cleanup Strategy
- Implement background cleanup job to remove expired entries
- Use Redis EXPIRE on keys to auto-cleanup inactive identifiers
- Balance cleanup frequency with memory usage

## Integration Points in Agent Lightning

Based on the Agent Lightning architecture, rate limiting should be integrated at:

1. **LLM API Calls**: Rate limit requests to external providers (OpenAI, Anthropic)
2. **User-Facing APIs**: Rate limit incoming requests to agent services
3. **Internal Service Communication**: Rate limit inter-service calls in distributed setup
4. **Agent Execution**: Rate limit agent action executions to prevent resource exhaustion
5. **Training Pipeline**: Rate limit batch processing and RL updates

## Implementation Outline

### Core Components

#### RateLimiter Interface
```python
class RateLimiter(ABC):
    @abstractmethod
    def is_allowed(self, identifier: str) -> bool:
        pass
    
    @abstractmethod
    def get_remaining(self, identifier: str) -> int:
        pass
    
    @abstractmethod
    def get_reset_time(self, identifier: str) -> float:
        pass
```

#### RedisSlidingWindowRateLimiter
```python
class RedisSlidingWindowRateLimiter(RateLimiter):
    def __init__(self, redis_client, max_requests: int, window_seconds: int, prefix: str = "ratelimit"):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds
        self.prefix = prefix
    
    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        key = f"{self.prefix}:{identifier}"
        
        # Clean old entries
        self.redis.zremrangebyscore(key, '-inf', now - self.window)
        
        # Count current requests
        count = self.redis.zcount(key, now - self.window, '+inf')
        
        if count < self.max_requests:
            # Add current request
            self.redis.zadd(key, {str(now): now})
            return True
        return False
    
    def get_remaining(self, identifier: str) -> int:
        # Implementation for remaining count
        pass
    
    def get_reset_time(self, identifier: str) -> float:
        # Implementation for reset time
        pass
```

### Configuration

```yaml
rate_limiting:
  enabled: true
  storage: redis  # or 'memory'
  redis_url: ${REDIS_URL}
  default_limits:
    user_requests: 
      max_requests: 100
      window_seconds: 60
    api_calls:
      max_requests: 1000
      window_seconds: 3600
    agent_executions:
      max_requests: 50
      window_seconds: 60
```

### Usage Examples

#### Basic Usage
```python
limiter = RedisSlidingWindowRateLimiter(redis_client, max_requests=10, window_seconds=60)

if limiter.is_allowed("user123"):
    # Process the request
    process_request()
else:
    # Return rate limit exceeded response
    return Response(status=429, body="Rate limit exceeded")
```

#### Integration with FastAPI
```python
from fastapi import Request, HTTPException
from starlette.responses import JSONResponse

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    limiter = get_limiter_for_endpoint(request.url.path)
    
    if not limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": limiter.get_reset_time(client_ip)}
        )
    
    response = await call_next(request)
    return response
```

#### LLM API Rate Limiting
```python
class LLMClient:
    def __init__(self, api_client, rate_limiter):
        self.api_client = api_client
        self.rate_limiter = rate_limiter
    
    async def call_llm(self, prompt: str) -> str:
        if not self.rate_limiter.is_allowed("llm_calls"):
            raise RateLimitExceeded("LLM API rate limit exceeded")
        
        return await self.api_client.generate(prompt)
```

## Testing Strategy

### Unit Tests
- Test core rate limiting logic with mock storage
- Test edge cases (exact window boundaries, concurrent requests)
- Test cleanup mechanisms

### Integration Tests
- Test with actual Redis instance
- Test distributed scenarios with multiple instances
- Test failure scenarios (Redis unavailable)

### Load Tests
- Simulate high request rates
- Test memory usage over time
- Test cleanup performance

### Distributed Tests
- Test consistency across multiple instances
- Test clock skew scenarios
- Test failover scenarios

## Monitoring and Observability

### Metrics to Collect
- Total requests allowed/denied
- Current window size per identifier
- Redis connection health
- Cleanup job performance
- Rate limit violation patterns

### Alerts
- High denial rates (>90% of requests)
- Redis connection failures
- Memory usage spikes
- Clock skew detection

### Logging
- Rate limit violations with identifier and endpoint
- Configuration changes
- Cleanup job execution times

## Deployment Considerations

### Configuration Management
- Environment-specific rate limits
- Gradual rollout with feature flags
- A/B testing different limit configurations

### Scaling
- Horizontal scaling with Redis clustering
- Instance-specific limits vs global limits
- Auto-scaling based on request patterns

### Security
- Rate limiting based on authenticated user ID vs IP
- Protection against identifier spoofing
- Integration with existing auth/rbac systems

## Conclusion

This sliding window rate limiting design provides a robust, scalable solution suitable for Agent Lightning's distributed architecture. The Redis-based implementation ensures consistency across services while providing the smooth rate limiting characteristics needed for AI agent workflows. The design supports multiple integration points and can be easily extended for future requirements.