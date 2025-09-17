# Solution Architecture: Integration Hub Service

**Document ID:** SA-005  
**Date:** 2025-09-06  
**Status:** For Review  
**Priority:** High  
**Author:** System Architect  
**Dependencies:** SA-001 (Database), SA-002 (Redis), SA-003 (Microservices)  

## Executive Summary

This solution architecture details the integration of the Integration Hub service with the shared PostgreSQL database and Redis event bus. The Integration Hub is responsible for managing external API connections, webhooks, data transformations, and third-party service integrations.

## Problem Statement

### Current Issues

- **In-Memory Configuration:** API credentials and configurations lost on restart
- **No Integration History:** No record of API calls or webhook events
- **Limited Rate Limiting:** Cannot track API usage across service instances
- **No Credential Vault:** Insecure storage of API keys in memory
- **Poor Error Recovery:** Failed integrations not retried automatically

### Business Impact

- Security risk from credential exposure
- Lost webhook events during downtime
- API rate limit violations causing service disruptions
- No audit trail for compliance
- Manual intervention required for failed integrations

## Proposed Solution

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Hub Service                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Integration Manager                      │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │   API    │  │ Webhook  │  │   Data   │             │   │
│  │  │ Gateway  │  │ Handler  │  │Transform │             │   │
│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘             │   │
│  │        │             │              │                   │   │
│  └────────┼─────────────┼──────────────┼───────────────────┘   │
│           │             │              │                        │
│  ┌────────▼─────────────▼──────────────▼────────────────────┐   │
│  │              Data Access Layer (DAL)                     │   │
│  │                                                          │   │
│  │  - Integration configs                                   │   │
│  │  - API credentials (encrypted)                           │   │
│  │  - Webhook subscriptions                                 │   │
│  │  - Integration history                                   │   │
│  └────────┬──────────────────────────┬──────────────────────┘   │
│           │                          │                          │
└───────────┼──────────────────────────┼──────────────────────────┘
            │                          │
     ┌──────▼──────┐            ┌──────▼──────┐
     │  PostgreSQL │            │    Redis    │
     │             │            │             │
     │ • Configs   │            │ • Rate Limits│
     │ • History   │            │ • API Cache  │
     │ • Webhooks  │            │ • Events     │
     └─────────────┘            └─────────────┘
```

### Database Schema Extensions

```sql
-- Integration configurations
CREATE TABLE integrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'api', 'webhook', 'database', 'message_queue'
    provider VARCHAR(50), -- 'stripe', 'github', 'slack', etc.
    config JSONB NOT NULL,
    credentials BYTEA, -- Encrypted credentials
    status VARCHAR(20) DEFAULT 'active',
    rate_limit INTEGER,
    rate_window INTEGER, -- seconds
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API call history
CREATE TABLE integration_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    integration_id UUID REFERENCES integrations(id),
    method VARCHAR(10),
    endpoint TEXT,
    request_data JSONB,
    response_data JSONB,
    status_code INTEGER,
    latency_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Webhook subscriptions
CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    integration_id UUID REFERENCES integrations(id),
    event_type VARCHAR(100),
    callback_url TEXT,
    secret VARCHAR(255),
    active BOOLEAN DEFAULT true,
    last_triggered TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Webhook events
CREATE TABLE webhook_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id UUID REFERENCES webhooks(id),
    event_data JSONB,
    status VARCHAR(20), -- 'pending', 'processed', 'failed'
    retry_count INTEGER DEFAULT 0,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Key Components

#### 1. Secure Credential Management

```python
import cryptography.fernet

class CredentialVault:
    def __init__(self):
        # Use environment variable for encryption key
        self.cipher = Fernet(os.getenv('ENCRYPTION_KEY'))
    
    def encrypt_credentials(self, credentials: dict) -> bytes:
        """Encrypt API credentials before storage"""
        json_str = json.dumps(credentials)
        return self.cipher.encrypt(json_str.encode())
    
    def decrypt_credentials(self, encrypted: bytes) -> dict:
        """Decrypt credentials for use"""
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
```

#### 2. Rate Limiting with Redis

```python
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(self, integration_id: str, 
                               limit: int, window: int) -> bool:
        """Check if API call is within rate limits"""
        key = f"rate_limit:{integration_id}"
        
        # Use sliding window algorithm
        now = time.time()
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(key, 0, now - window)
        pipeline.zadd(key, {str(uuid.uuid4()): now})
        pipeline.zcount(key, now - window, now)
        pipeline.expire(key, window)
        results = pipeline.execute()
        
        count = results[2]
        return count <= limit
```

#### 3. Webhook Processing Queue

```python
class WebhookProcessor:
    def __init__(self, dal: DataAccessLayer):
        self.dal = dal
        self.queue_key = "webhook:queue"
    
    async def process_webhook(self, webhook_id: str, event_data: dict):
        """Process incoming webhook with retry logic"""
        # Store event
        event = self.dal.create_webhook_event({
            'webhook_id': webhook_id,
            'event_data': event_data,
            'status': 'pending'
        })
        
        # Queue for processing
        self.redis.lpush(self.queue_key, event['id'])
        
        # Process async
        await self._process_event(event['id'])
```

#### 4. API Response Caching

```python
class APICache:
    def __init__(self, cache_manager):
        self.cache = cache_manager
    
    def cache_response(self, endpoint: str, params: dict, 
                      response: dict, ttl: int = 300):
        """Cache API responses to reduce external calls"""
        cache_key = f"api:{endpoint}:{hash(frozenset(params.items()))}"
        self.cache.set(cache_key, response, ttl=ttl)
    
    def get_cached(self, endpoint: str, params: dict) -> Optional[dict]:
        """Get cached response if available"""
        cache_key = f"api:{endpoint}:{hash(frozenset(params.items()))}"
        return self.cache.get(cache_key)
```

### Implementation Features

#### 1. Integration Configuration Management

```python
class IntegrationManager:
    def create_integration(self, config: dict) -> dict:
        """Create new integration with encrypted credentials"""
        # Encrypt sensitive data
        if 'credentials' in config:
            encrypted = self.vault.encrypt_credentials(config['credentials'])
            config['credentials'] = encrypted
        
        # Store in database
        integration = self.dal.create_integration(config)
        
        # Emit event
        self.event_bus.emit(EventChannel.INTEGRATION_CREATED, {
            'integration_id': integration['id'],
            'type': config['type']
        })
        
        return integration
```

#### 2. Automatic Retry Logic

```python
class RetryManager:
    async def retry_failed_calls(self):
        """Retry failed API calls with exponential backoff"""
        failed_calls = self.dal.get_failed_integrations()
        
        for call in failed_calls:
            retry_count = call['retry_count']
            if retry_count < MAX_RETRIES:
                delay = 2 ** retry_count  # Exponential backoff
                await asyncio.sleep(delay)
                await self.execute_integration(call['id'])
```

#### 3. Data Transformation Pipeline

```python
class DataTransformer:
    def transform(self, data: dict, mapping: dict) -> dict:
        """Transform data between different API formats"""
        result = {}
        for target_field, source_path in mapping.items():
            value = self._extract_value(data, source_path)
            if value is not None:
                result[target_field] = value
        return result
```

## Success Metrics

### Technical Metrics

- ✅ 100% credential encryption at rest
- ✅ Zero webhook events lost
- ✅ < 50ms integration response time (cached)
- ✅ 99.9% webhook delivery success rate
- ✅ Automatic retry for failed integrations

### Business Metrics

- ✅ Complete integration audit trail
- ✅ 90% reduction in API rate limit violations
- ✅ Zero credential exposure incidents
- ✅ 80% reduction in manual intervention
- ✅ Real-time integration monitoring

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| API service downtime | Medium | High | Circuit breaker pattern, cached responses |
| Credential compromise | Low | Critical | Encryption, access controls, rotation |
| Webhook flooding | Low | Medium | Rate limiting, queue management |
| Data transformation errors | Medium | Medium | Schema validation, error handling |

## Testing Strategy

### Unit Tests

- Credential encryption/decryption
- Rate limiting logic
- Data transformation rules
- Webhook signature validation

### Integration Tests

- End-to-end API calls
- Webhook processing pipeline
- Retry mechanism
- Cache invalidation

### Security Tests

- Credential vault security
- API authentication
- Webhook signature verification
- Rate limit enforcement

## Migration Plan

1. Deploy integrated service alongside existing
2. Migrate integration configurations
3. Encrypt and store credentials
4. Enable webhook processing
5. Monitor and optimize
6. Decommission old service

## Approval

**Review Checklist:**
- [ ] Security measures adequate
- [ ] Rate limiting implemented
- [ ] Retry logic comprehensive
- [ ] Monitoring in place
- [ ] Testing coverage sufficient

**Sign-off Required From:**
- [ ] Security Team
- [ ] Integration Team
- [ ] Database Team
- [ ] DevOps Team

---

**Next Steps After Approval:**
1. Implement credential vault
2. Create integration tables
3. Deploy integrated service
4. Migrate existing integrations
5. Monitor performance

**Related Documents:**
- SA-001: Database Persistence Layer
- SA-002: Redis Cache & Event Bus  
- SA-003: Microservices Integration
- SA-004: Workflow Engine Integration
- Next: SA-006 AI Model Service Integration