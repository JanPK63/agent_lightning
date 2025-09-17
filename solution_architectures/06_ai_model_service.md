# Solution Architecture: AI Model Service Integration

**Document ID:** SA-006  
**Date:** 2025-09-06  
**Status:** For Review  
**Priority:** Critical  
**Author:** System Architect  
**Dependencies:** SA-001 (Database), SA-002 (Redis), SA-003 (Microservices)  

## Executive Summary

This solution architecture details the integration of the AI Model service with the shared PostgreSQL database and Redis event bus. The AI Model service is the core intelligence engine of the Lightning system, managing model configurations, serving inference requests, and maintaining conversation history.

## Problem Statement

### Current Issues

- **In-Memory Model Configs:** Model configurations lost on service restart
- **No Conversation History:** Previous interactions not persisted
- **Isolated Inference:** Cannot share model results across services
- **No Performance Metrics:** Cannot track model usage and performance
- **Resource Waste:** Multiple services loading same models

### Business Impact

- Lost fine-tuning configurations during updates
- Cannot provide contextual responses based on history
- Inefficient resource utilization
- No visibility into model performance
- Cannot scale horizontally

## Proposed Solution

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                     AI Model Service                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 AI Model Manager                         │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │  Model   │  │Inference │  │ Context  │             │   │
│  │  │ Registry │  │  Engine  │  │ Manager  │             │   │
│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘             │   │
│  │        │             │              │                   │   │
│  └────────┼─────────────┼──────────────┼───────────────────┘   │
│           │             │              │                        │
│  ┌────────▼─────────────▼──────────────▼────────────────────┐   │
│  │              Data Access Layer (DAL)                     │   │
│  │                                                          │   │
│  │  - Model configurations                                  │   │
│  │  - Conversation history                                  │   │
│  │  - Inference cache                                       │   │
│  │  - Performance metrics                                   │   │
│  └────────┬──────────────────────────┬──────────────────────┘   │
│           │                          │                          │
└───────────┼──────────────────────────┼──────────────────────────┘
            │                          │
     ┌──────▼──────┐            ┌──────▼──────┐
     │  PostgreSQL │            │    Redis    │
     │             │            │             │
     │ • Models    │            │ • Cache     │
     │ • History   │            │ • Sessions  │
     │ • Metrics   │            │ • Events    │
     └─────────────┘            └─────────────┘
```

### Database Schema Extensions

```sql
-- AI Model configurations
CREATE TABLE ai_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    provider VARCHAR(50) NOT NULL, -- 'openai', 'anthropic', 'local'
    model_id VARCHAR(100) NOT NULL, -- 'gpt-4', 'claude-3', etc.
    config JSONB NOT NULL, -- model-specific configuration
    capabilities JSONB, -- supported features
    context_window INTEGER DEFAULT 4096,
    max_tokens INTEGER DEFAULT 2048,
    temperature FLOAT DEFAULT 0.7,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation history
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    agent_id UUID REFERENCES agents(id),
    user_id VARCHAR(255),
    model_id UUID REFERENCES ai_models(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Individual messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    tokens_used INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Inference results cache
CREATE TABLE inference_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_hash VARCHAR(64) UNIQUE NOT NULL,
    model_id UUID REFERENCES ai_models(id),
    response TEXT NOT NULL,
    tokens_used INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Model performance metrics
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ai_models(id),
    metric_type VARCHAR(50), -- 'latency', 'tokens', 'error_rate'
    value FLOAT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_inference_cache_hash ON inference_cache(prompt_hash);
CREATE INDEX idx_model_metrics_model ON model_metrics(model_id);
```

### Key Components

#### 1. Model Registry

```python
class ModelRegistry:
    def __init__(self, dal: DataAccessLayer):
        self.dal = dal
        self.models = {}
        self.load_models()
    
    def register_model(self, config: dict) -> dict:
        """Register new AI model configuration"""
        model = self.dal.create_ai_model(config)
        
        # Cache model config
        self.models[model['id']] = model
        
        # Emit event
        self.dal.event_bus.emit(EventChannel.MODEL_REGISTERED, {
            'model_id': model['id'],
            'name': model['name']
        })
        
        return model
    
    def get_best_model(self, task_type: str) -> dict:
        """Select best model for task based on capabilities"""
        # Query models with matching capabilities
        suitable_models = self.dal.get_models_by_capability(task_type)
        
        if not suitable_models:
            return self.get_default_model()
        
        # Select based on performance metrics
        return self.select_by_performance(suitable_models)
```

#### 2. Context Manager

```python
class ContextManager:
    def __init__(self, dal: DataAccessLayer, cache):
        self.dal = dal
        self.cache = cache
        self.active_contexts = {}
    
    def get_context(self, session_id: str, limit: int = 10) -> list:
        """Get conversation context for session"""
        # Check cache first
        cache_key = f"context:{session_id}"
        context = self.cache.get(cache_key)
        
        if not context:
            # Load from database
            context = self.dal.get_conversation_history(
                session_id, limit=limit
            )
            # Cache for 5 minutes
            self.cache.set(cache_key, context, ttl=300)
        
        return context
    
    def add_message(self, session_id: str, message: dict):
        """Add message to conversation history"""
        # Save to database
        self.dal.add_message(session_id, message)
        
        # Update cache
        cache_key = f"context:{session_id}"
        self.cache.delete(cache_key)  # Invalidate cache
        
        # Emit event
        self.dal.event_bus.emit(EventChannel.MESSAGE_ADDED, {
            'session_id': session_id,
            'role': message['role']
        })
```

#### 3. Inference Engine

```python
class InferenceEngine:
    def __init__(self, dal: DataAccessLayer, cache):
        self.dal = dal
        self.cache = cache
        self.providers = {}
        self.initialize_providers()
    
    async def inference(self, request: dict) -> dict:
        """Perform model inference with caching"""
        # Check cache
        prompt_hash = hashlib.sha256(
            request['prompt'].encode()
        ).hexdigest()
        
        cached = self.dal.get_cached_inference(prompt_hash)
        if cached and not request.get('no_cache'):
            return cached
        
        # Get model
        model = self.dal.get_model(request['model_id'])
        provider = self.providers[model['provider']]
        
        # Perform inference
        start_time = time.time()
        response = await provider.complete(
            model=model['model_id'],
            prompt=request['prompt'],
            **model['config']
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Cache result
        self.dal.cache_inference({
            'prompt_hash': prompt_hash,
            'model_id': model['id'],
            'response': response['text'],
            'tokens_used': response['tokens'],
            'latency_ms': latency_ms
        })
        
        # Record metrics
        self.dal.record_model_metric(
            model['id'], 'latency', latency_ms
        )
        
        return response
```

#### 4. Streaming Support

```python
class StreamingInference:
    async def stream_inference(self, request: dict):
        """Stream model responses for real-time interaction"""
        model = self.dal.get_model(request['model_id'])
        provider = self.providers[model['provider']]
        
        async for chunk in provider.stream_complete(
            model=model['model_id'],
            prompt=request['prompt'],
            **model['config']
        ):
            # Emit chunk via WebSocket
            await self.websocket_manager.send(
                request['session_id'],
                {
                    'type': 'stream_chunk',
                    'content': chunk,
                    'model_id': model['id']
                }
            )
            
            yield chunk
```

### Implementation Features

#### 1. Multi-Provider Support

```python
class ProviderManager:
    def __init__(self):
        self.providers = {
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider(),
            'local': LocalModelProvider(),
            'huggingface': HuggingFaceProvider()
        }
    
    def get_provider(self, provider_name: str):
        return self.providers.get(provider_name)
```

#### 2. Token Management

```python
class TokenManager:
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for text using model tokenizer"""
        # Implementation varies by model
        pass
    
    def truncate_to_limit(self, messages: list, limit: int) -> list:
        """Truncate conversation to fit context window"""
        total_tokens = 0
        truncated = []
        
        for message in reversed(messages):
            tokens = self.count_tokens(message['content'])
            if total_tokens + tokens > limit:
                break
            truncated.insert(0, message)
            total_tokens += tokens
        
        return truncated
```

#### 3. Performance Monitoring

```python
class ModelMonitor:
    async def track_performance(self, model_id: str, metrics: dict):
        """Track model performance metrics"""
        # Record to database
        self.dal.record_model_metrics({
            'model_id': model_id,
            'latency': metrics['latency'],
            'tokens_per_second': metrics['tps'],
            'error_rate': metrics['errors']
        })
        
        # Alert on degradation
        if metrics['latency'] > LATENCY_THRESHOLD:
            await self.alert_performance_issue(model_id)
```

## Success Metrics

### Technical Metrics

- ✅ 100% conversation history persistence
- ✅ < 50ms inference cache hit latency
- ✅ Zero model config loss on restart
- ✅ Horizontal scaling capability
- ✅ 90% cache hit rate for repeated queries

### Business Metrics

- ✅ Complete audit trail of AI interactions
- ✅ 50% reduction in inference costs via caching
- ✅ Improved response quality with context
- ✅ Real-time performance monitoring
- ✅ Multi-model A/B testing capability

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| API rate limits | Medium | High | Implement backoff, use multiple keys |
| Context overflow | Medium | Medium | Smart truncation, summarization |
| Model API changes | Low | High | Version pinning, adapter pattern |
| Cache invalidation issues | Low | Medium | TTL strategy, manual invalidation |

## Testing Strategy

### Unit Tests

- Model registration and selection
- Context management and truncation
- Token counting accuracy
- Cache operations

### Integration Tests

- End-to-end inference flow
- Multi-provider support
- Streaming responses
- Performance under load

### Performance Tests

- Inference latency benchmarks
- Cache effectiveness
- Concurrent request handling
- Memory usage optimization

## Migration Plan

1. Deploy integrated service on port 8105
2. Migrate model configurations
3. Import conversation history
4. Enable caching layer
5. Switch traffic gradually
6. Monitor and optimize
7. Decommission old service

## Approval

**Review Checklist:**
- [ ] Database schema adequate
- [ ] Caching strategy optimal
- [ ] Multi-provider support complete
- [ ] Performance metrics comprehensive
- [ ] Testing coverage sufficient

**Sign-off Required From:**
- [ ] AI Team Lead
- [ ] Data Team
- [ ] Infrastructure Team
- [ ] Security Team

---

**Next Steps After Approval:**
1. Create database tables
2. Implement integrated service
3. Migrate existing configurations
4. Test with multiple providers
5. Deploy and monitor

**Related Documents:**
- SA-001: Database Persistence Layer
- SA-002: Redis Cache & Event Bus  
- SA-003: Microservices Integration
- SA-004: Workflow Engine Integration
- SA-005: Integration Hub Service
- Next: SA-007 Authentication & Authorization Service