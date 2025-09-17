# Solution Architecture: Redis Cache & Event Bus

**Document ID:** SA-002  
**Date:** 2025-09-05  
**Status:** For Review  
**Priority:** Critical  
**Author:** System Architect  
**Dependencies:** SA-001 (Database Persistence Layer)  

## Executive Summary

This solution architecture introduces Redis as a dual-purpose component serving as both a high-performance cache layer and an event bus for real-time inter-service communication. Redis will reduce database load, improve response times, and enable event-driven architecture across all microservices.

## Problem Statement

### Current Issues
- **Database Load:** Every request hits PostgreSQL directly
- **No Service Communication:** Services cannot notify each other of changes
- **Session Management:** Sessions stored in database causing overhead
- **No Real-time Updates:** Dashboard doesn't receive live updates
- **Repeated Queries:** Same data fetched multiple times

### Business Impact
- Slower response times (200-500ms vs target 50ms)
- Database becomes bottleneck at scale
- No real-time collaboration features
- Higher infrastructure costs
- Poor user experience with stale data

## Proposed Solution

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Main   │  │  Agent   │  │ Workflow │  │    AI    │       │
│  │Dashboard │  │ Designer │  │  Engine  │  │  Model   │  ...  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │               │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        └──────┬──────┴──────┬──────┴──────┬──────┘
               │             │             │
         ┌─────▼─────────────▼─────────────▼─────┐
         │         Redis Cluster (6379)          │
         │  ┌─────────────────────────────────┐  │
         │  │      Cache Layer                │  │
         │  ├─────────────────────────────────┤  │
         │  │  • Agent Data (TTL: 1h)        │  │
         │  │  • Task Results (TTL: 15m)     │  │
         │  │  • Knowledge Base (TTL: 6h)    │  │
         │  │  • Sessions (TTL: 24h)         │  │
         │  │  • Metrics (TTL: 5m)           │  │
         │  └─────────────────────────────────┘  │
         │  ┌─────────────────────────────────┐  │
         │  │      Pub/Sub Event Bus         │  │
         │  ├─────────────────────────────────┤  │
         │  │  Channels:                     │  │
         │  │  • agent:*                     │  │
         │  │  • task:*                      │  │
         │  │  • workflow:*                  │  │
         │  │  • metrics:*                   │  │
         │  │  • system:*                    │  │
         │  └─────────────────────────────────┘  │
         └────────────────┬───────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │        PostgreSQL Database         │
         └────────────────────────────────────┘
```

### Cache Strategy

#### 1. Cache Patterns

```python
# Cache-Aside Pattern (Lazy Loading)
def get_agent(agent_id: str) -> Agent:
    # Try cache first
    cached = cache.get(f"agent:{agent_id}")
    if cached:
        return cached
    
    # Load from database
    agent = db.query(Agent).get(agent_id)
    
    # Store in cache
    cache.set(f"agent:{agent_id}", agent, ttl=3600)
    return agent

# Write-Through Pattern
def update_agent(agent_id: str, data: dict) -> Agent:
    # Update database
    agent = db.update(Agent, agent_id, data)
    
    # Update cache
    cache.set(f"agent:{agent_id}", agent, ttl=3600)
    
    # Publish update event
    cache.publish("agent:updated", {"id": agent_id})
    return agent

# Cache Invalidation
def delete_agent(agent_id: str):
    # Delete from database
    db.delete(Agent, agent_id)
    
    # Invalidate cache
    cache.delete(f"agent:{agent_id}")
    
    # Publish delete event
    cache.publish("agent:deleted", {"id": agent_id})
```

#### 2. Cache Keys Structure

| Entity | Key Pattern | TTL | Example |
|--------|------------|-----|---------|
| Agent | `agent:{id}` | 1 hour | `agent:researcher` |
| Agent List | `agents:all` | 5 min | `agents:all` |
| Task | `task:{id}` | 15 min | `task:uuid-123` |
| Task Results | `task:result:{id}` | 1 hour | `task:result:uuid-123` |
| Knowledge | `knowledge:{agent}:{id}` | 6 hours | `knowledge:researcher:kb-001` |
| Session | `session:{token}` | 24 hours | `session:jwt-xyz` |
| Metrics | `metrics:{service}:{name}` | 5 min | `metrics:api:latency` |
| Workflow | `workflow:{id}` | 30 min | `workflow:wf-456` |
| Lock | `lock:{resource}` | 30 sec | `lock:agent:researcher` |

### Event Bus Architecture

#### 1. Event Channels

```yaml
Event Channels:
  # Agent Events
  agent:created    # New agent registered
  agent:updated    # Agent configuration changed
  agent:deleted    # Agent removed
  agent:status     # Agent status change (idle/busy/error)
  
  # Task Events  
  task:created     # New task assigned
  task:started     # Task execution began
  task:completed   # Task finished successfully
  task:failed      # Task failed with error
  task:progress    # Task progress update
  
  # Workflow Events
  workflow:started   # Workflow execution started
  workflow:step      # Workflow step completed
  workflow:completed # Workflow finished
  workflow:failed    # Workflow failed
  
  # System Events
  system:health      # Health check updates
  system:alert       # System alerts
  system:metrics     # Performance metrics
  
  # WebSocket Events
  ws:connect        # Client connected
  ws:disconnect     # Client disconnected
  ws:message        # WebSocket message
```

#### 2. Event Message Format

```json
{
  "event_id": "evt_1234567890",
  "timestamp": "2025-09-05T10:30:00Z",
  "channel": "task:completed",
  "service": "agent_designer",
  "data": {
    "task_id": "task_uuid",
    "agent_id": "researcher",
    "result": {...},
    "duration_ms": 1250
  },
  "metadata": {
    "correlation_id": "req_abc123",
    "user_id": "user_456",
    "version": "1.0"
  }
}
```

### Implementation Components

#### 1. Redis Cache Manager

```python
# shared/cache.py
import redis
import json
import pickle
from typing import Any, Optional, List
import hashlib

class CacheManager:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False,
            connection_pool=redis.ConnectionPool(
                max_connections=50,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 1,  # TCP_KEEPINTVL
                    3: 3,  # TCP_KEEPCNT
                }
            )
        )
        self.pubsub = self.redis_client.pubsub()
    
    # Cache Operations
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = self.redis_client.get(key)
        if value:
            return pickle.loads(value)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache value with TTL"""
        serialized = pickle.dumps(value)
        self.redis_client.setex(key, ttl, serialized)
    
    def delete(self, key: str):
        """Delete cache key"""
        self.redis_client.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.redis_client.exists(key)
    
    # Batch Operations
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values"""
        values = self.redis_client.mget(keys)
        return [pickle.loads(v) if v else None for v in values]
    
    def mset(self, mapping: dict, ttl: int = 3600):
        """Set multiple values"""
        serialized = {k: pickle.dumps(v) for k, v in mapping.items()}
        pipe = self.redis_client.pipeline()
        for key, value in serialized.items():
            pipe.setex(key, ttl, value)
        pipe.execute()
    
    # Pattern Operations
    def delete_pattern(self, pattern: str):
        """Delete all keys matching pattern"""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
    
    # Pub/Sub Operations
    def publish(self, channel: str, message: dict):
        """Publish message to channel"""
        self.redis_client.publish(channel, json.dumps(message))
    
    def subscribe(self, channels: List[str]):
        """Subscribe to channels"""
        self.pubsub.subscribe(*channels)
        return self.pubsub
    
    # Distributed Locking
    def acquire_lock(self, resource: str, timeout: int = 30) -> bool:
        """Acquire distributed lock"""
        lock_key = f"lock:{resource}"
        identifier = str(uuid.uuid4())
        return self.redis_client.set(
            lock_key, identifier, 
            nx=True, ex=timeout
        )
    
    def release_lock(self, resource: str):
        """Release distributed lock"""
        self.redis_client.delete(f"lock:{resource}")
    
    # Cache Warming
    def warm_cache(self, data_loader):
        """Pre-populate cache with data"""
        for key, value, ttl in data_loader():
            self.set(key, value, ttl)
```

#### 2. Event Publisher

```python
# shared/events.py
from datetime import datetime
import uuid

class EventPublisher:
    def __init__(self, cache_manager):
        self.cache = cache_manager
        
    def emit(self, channel: str, data: dict, metadata: dict = None):
        """Emit event to channel"""
        event = {
            "event_id": f"evt_{uuid.uuid4().hex}",
            "timestamp": datetime.utcnow().isoformat(),
            "channel": channel,
            "data": data,
            "metadata": metadata or {}
        }
        self.cache.publish(channel, event)
        
        # Also store recent events
        self.cache.redis_client.lpush(
            f"events:{channel}:history",
            json.dumps(event)
        )
        self.cache.redis_client.ltrim(
            f"events:{channel}:history", 0, 99
        )
```

## Implementation Plan

### Phase 1: Redis Setup (Day 1)
1. Install Redis server
2. Configure Redis cluster mode
3. Set up persistence (RDB + AOF)
4. Configure memory policies
5. Set up Redis Sentinel for HA

### Phase 2: Cache Layer (Day 2)
1. Implement CacheManager class
2. Add cache decorators
3. Integrate with database layer
4. Implement cache warming
5. Add cache metrics

### Phase 3: Event Bus (Day 3)
1. Implement EventPublisher
2. Create event subscribers
3. Add event handlers
4. Implement event replay
5. Add dead letter queue

### Phase 4: Service Integration (Days 4-5)
1. Update all services to use cache
2. Add event publishing
3. Implement event handlers
4. Add cache invalidation
5. Performance testing

## Technical Specifications

### Redis Configuration

```conf
# redis.conf
port 6379
bind 0.0.0.0
protected-mode yes
requirepass your_redis_password

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Memory Management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Connection Pool
timeout 300
tcp-keepalive 60
tcp-backlog 511

# Cluster Configuration
cluster-enabled no  # Enable for production
cluster-config-file nodes.conf
cluster-node-timeout 5000
```

### Performance Requirements
- Cache hit ratio: > 80%
- Cache response time: < 5ms
- Event latency: < 10ms
- Throughput: 50,000 ops/sec
- Memory usage: < 2GB

### Security Considerations
- Password authentication required
- SSL/TLS for connections
- ACL for user permissions
- Network isolation
- Regular security updates

## Rollback Plan

### Rollback Strategy
1. Feature flag for cache usage
2. Fallback to direct database queries
3. Event bus can be disabled
4. Gradual rollback per service
5. Data consistency checks

### Rollback Procedure
```bash
# 1. Disable cache in environment
export USE_CACHE=false

# 2. Clear all cache data
redis-cli FLUSHALL

# 3. Restart services
systemctl restart agent-lightning-*

# 4. Monitor for issues
tail -f /var/log/agent-lightning/*.log
```

## Success Metrics

### Technical Metrics
- ✅ 80%+ cache hit ratio
- ✅ <50ms average response time
- ✅ Zero message loss
- ✅ 99.9% uptime
- ✅ <2GB memory usage

### Business Metrics
- ✅ 5x faster response times
- ✅ Real-time updates working
- ✅ 50% reduction in database load
- ✅ Improved user experience
- ✅ Lower infrastructure costs

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cache inconsistency | Medium | Medium | TTL strategy, invalidation |
| Memory overflow | Low | High | Memory limits, eviction policy |
| Redis failure | Low | High | Redis Sentinel, replicas |
| Event message loss | Low | Medium | Persistence, acknowledgments |
| Network partitioning | Low | High | Cluster mode, monitoring |

## Cost Analysis

### Infrastructure Costs
- Redis instance: ~$50/month (2GB RAM)
- Redis replicas: ~$100/month (2x replicas)
- Monitoring: ~$20/month
- **Total: ~$170/month**

### Benefits
- Reduced database load: -$200/month
- Improved performance: +20% user retention
- Reduced latency: Better user experience
- **ROI: Positive within first month**

## Monitoring & Observability

### Key Metrics to Monitor
```python
# Cache Metrics
cache_hits_total
cache_misses_total
cache_hit_ratio
cache_evictions_total
cache_memory_usage_bytes

# Event Bus Metrics
events_published_total
events_consumed_total
event_processing_duration_seconds
event_errors_total

# Redis Metrics
redis_connected_clients
redis_memory_used_bytes
redis_ops_per_second
redis_keyspace_hits
redis_keyspace_misses
```

## Approval

**Review Checklist:**
- [ ] Cache strategy appropriate for use cases
- [ ] Event architecture supports requirements
- [ ] Performance targets are achievable
- [ ] Security measures are adequate
- [ ] Cost-benefit analysis is positive

**Sign-off Required From:**
- [ ] Technical Lead
- [ ] DevOps Team
- [ ] Security Team
- [ ] Product Owner

---

**Next Steps After Approval:**
1. Provision Redis infrastructure
2. Implement cache manager
3. Add event bus functionality
4. Begin service integration

**Related Documents:**
- SA-001: Database Persistence Layer
- SYSTEM_ARCHITECTURE_ANALYSIS.md
- Next: SA-003 Microservices Integration