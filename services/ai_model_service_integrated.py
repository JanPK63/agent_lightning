#!/usr/bin/env python3
"""
AI Model Orchestration Microservice - Integrated with Shared Database
Handles AI model routing, inference, conversation history, and caching
Using shared PostgreSQL and Redis for persistence and performance
Based on SA-006: AI Model Service Integration
"""

import os
import sys
import json
import asyncio
import time
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Types of AI models"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies for model selection"""
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_FIRST = "quality_first"
    RANDOM = "random"
    WEIGHTED = "weighted"


# Constants
MAX_CONTEXT_TOKENS = 4096
CACHE_TTL = 3600  # 1 hour
CONVERSATION_HISTORY_LIMIT = 50


# Pydantic Models
class ModelCreate(BaseModel):
    """Create new AI model configuration"""
    name: str = Field(description="Model name")
    provider: ModelProvider = Field(description="Model provider")
    model_id: str = Field(description="Provider's model ID")
    model_type: ModelType = Field(description="Type of model")
    config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    context_window: int = Field(default=4096, description="Context window size")
    max_tokens: int = Field(default=2048, description="Maximum output tokens")
    temperature: float = Field(default=0.7, description="Default temperature")


class InferenceRequest(BaseModel):
    """Model inference request"""
    prompt: str = Field(description="Input prompt")
    model_id: Optional[str] = Field(default=None, description="Specific model ID")
    model_type: Optional[ModelType] = Field(default=ModelType.CHAT, description="Type of model to use")
    session_id: Optional[str] = Field(default=None, description="Session for context")
    agent_id: Optional[str] = Field(default=None, description="Agent making request")
    temperature: Optional[float] = Field(default=None, description="Temperature override")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens override")
    stream: bool = Field(default=False, description="Stream response")
    use_cache: bool = Field(default=True, description="Use cached responses")
    include_context: bool = Field(default=True, description="Include conversation context")


class ConversationMessage(BaseModel):
    """Conversation message"""
    role: str = Field(description="Message role: user, assistant, system")
    content: str = Field(description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ModelRegistry:
    """Manages model configurations and selection"""
    
    def __init__(self, dal: DataAccessLayer, cache):
        self.dal = dal
        self.cache = cache
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load model configurations from database"""
        # For now, use default models (in production, load from DB)
        self.models = {
            "gpt-4": {
                "id": "gpt-4",
                "name": "GPT-4",
                "provider": ModelProvider.OPENAI,
                "model_type": ModelType.CHAT,
                "context_window": 8192,
                "max_tokens": 4096,
                "capabilities": ["chat", "code", "reasoning"]
            },
            "claude-3": {
                "id": "claude-3",
                "name": "Claude 3",
                "provider": ModelProvider.ANTHROPIC,
                "model_type": ModelType.CHAT,
                "context_window": 100000,
                "max_tokens": 4096,
                "capabilities": ["chat", "code", "analysis"]
            },
            "llama-2": {
                "id": "llama-2",
                "name": "Llama 2",
                "provider": ModelProvider.LOCAL,
                "model_type": ModelType.CHAT,
                "context_window": 4096,
                "max_tokens": 2048,
                "capabilities": ["chat", "general"]
            }
        }
        logger.info(f"Loaded {len(self.models)} AI models")
    
    def register_model(self, config: dict) -> dict:
        """Register new AI model configuration"""
        model_id = config.get('model_id', str(uuid.uuid4()))
        
        # Store in memory (in production, save to DB)
        self.models[model_id] = config
        
        # Cache model config
        cache_key = f"model:{model_id}"
        self.cache.set(cache_key, config, ttl=None)  # No expiry
        
        # Emit event
        self.dal.event_bus.emit(EventChannel.SYSTEM_ALERT, {
            'type': 'model_registered',
            'model_id': model_id,
            'name': config['name']
        })
        
        logger.info(f"Registered model {model_id}: {config['name']}")
        return config
    
    def get_model(self, model_id: str) -> Optional[dict]:
        """Get specific model configuration"""
        if model_id in self.models:
            return self.models[model_id]
        
        # Check cache
        cache_key = f"model:{model_id}"
        return self.cache.get(cache_key)
    
    def get_best_model(self, model_type: ModelType, capabilities: List[str] = None) -> dict:
        """Select best model for task based on type and capabilities"""
        suitable_models = []
        
        for model_id, model in self.models.items():
            if model['model_type'] == model_type:
                if not capabilities or any(cap in model.get('capabilities', []) for cap in capabilities):
                    suitable_models.append(model)
        
        if not suitable_models:
            # Return default model
            return list(self.models.values())[0] if self.models else None
        
        # Select based on performance (simplified - in production use metrics)
        return suitable_models[0]


class ContextManager:
    """Manages conversation context and history"""
    
    def __init__(self, dal: DataAccessLayer, cache):
        self.dal = dal
        self.cache = cache
        self.active_contexts = {}
    
    def get_context(self, session_id: str, limit: int = 10) -> List[dict]:
        """Get conversation context for session"""
        if not session_id:
            return []
        
        # Check cache first
        cache_key = f"context:{session_id}"
        context = self.cache.get(cache_key)
        
        if not context:
            # In production, load from database
            # For now, return empty context
            context = self.active_contexts.get(session_id, [])
            
            # Cache for 5 minutes
            if context:
                self.cache.set(cache_key, context, ttl=300)
        
        # Limit context size
        return context[-limit:] if len(context) > limit else context
    
    def add_message(self, session_id: str, message: dict):
        """Add message to conversation history"""
        if not session_id:
            return
        
        # Update in-memory context
        if session_id not in self.active_contexts:
            self.active_contexts[session_id] = []
        
        self.active_contexts[session_id].append({
            **message,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Limit history size
        if len(self.active_contexts[session_id]) > CONVERSATION_HISTORY_LIMIT:
            self.active_contexts[session_id] = self.active_contexts[session_id][-CONVERSATION_HISTORY_LIMIT:]
        
        # Invalidate cache
        cache_key = f"context:{session_id}"
        self.cache.delete(cache_key)
        
        # In production, save to database
        # self.dal.add_message(session_id, message)
        
        # Emit event
        self.dal.event_bus.emit(EventChannel.SYSTEM_ALERT, {
            'type': 'message_added',
            'session_id': session_id,
            'role': message['role']
        })
    
    def clear_context(self, session_id: str):
        """Clear conversation context"""
        if session_id in self.active_contexts:
            del self.active_contexts[session_id]
        
        cache_key = f"context:{session_id}"
        self.cache.delete(cache_key)


class InferenceEngine:
    """Handles model inference with caching and optimization"""
    
    def __init__(self, dal: DataAccessLayer, cache, model_registry: ModelRegistry):
        self.dal = dal
        self.cache = cache
        self.model_registry = model_registry
        self.providers = self._initialize_providers()
    
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize model provider interfaces"""
        # In production, initialize actual provider clients
        return {
            ModelProvider.OPENAI: self._mock_openai_provider,
            ModelProvider.ANTHROPIC: self._mock_anthropic_provider,
            ModelProvider.LOCAL: self._mock_local_provider
        }
    
    async def inference(self, request: dict, context: List[dict] = None) -> dict:
        """Perform model inference with caching"""
        # Generate cache key
        prompt_hash = hashlib.sha256(
            f"{request['prompt']}:{request.get('model_id', '')}".encode()
        ).hexdigest()
        
        # Check cache if enabled
        if request.get('use_cache', True):
            cache_key = f"inference:{prompt_hash}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for inference: {prompt_hash[:8]}")
                self.dal.record_metric('inference_cache_hit', 1)
                return cached
        
        # Get model
        model_id = request.get('model_id')
        if not model_id:
            model = self.model_registry.get_best_model(
                request.get('model_type', ModelType.CHAT)
            )
            model_id = model['id'] if model else 'gpt-4'
        else:
            model = self.model_registry.get_model(model_id)
        
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Build messages with context
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": request['prompt']})
        
        # Perform inference
        start_time = time.time()
        provider = self.providers.get(model['provider'])
        
        if not provider:
            raise ValueError(f"Provider {model['provider']} not supported")
        
        response = await provider(
            model=model,
            messages=messages,
            temperature=request.get('temperature', 0.7),
            max_tokens=request.get('max_tokens', 2048)
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Prepare result
        result = {
            'model_id': model_id,
            'response': response['content'],
            'tokens_used': response.get('tokens', 100),
            'latency_ms': latency_ms,
            'cached': False
        }
        
        # Cache result
        if request.get('use_cache', True):
            cache_key = f"inference:{prompt_hash}"
            self.cache.set(cache_key, result, ttl=CACHE_TTL)
        
        # Record metrics
        self.dal.record_metric('inference_latency', latency_ms, {
            'model': model_id,
            'provider': model['provider']
        })
        self.dal.record_metric('tokens_used', result['tokens_used'], {
            'model': model_id
        })
        
        return result
    
    async def stream_inference(self, request: dict, context: List[dict] = None):
        """Stream model responses for real-time interaction"""
        model_id = request.get('model_id', 'gpt-4')
        model = self.model_registry.get_model(model_id)
        
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Build messages with context
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": request['prompt']})
        
        # Simulate streaming response
        response_text = f"This is a streaming response to: {request['prompt'][:50]}..."
        words = response_text.split()
        
        for word in words:
            chunk = {
                'type': 'stream_chunk',
                'content': word + " ",
                'model_id': model_id
            }
            yield json.dumps(chunk) + "\n"
            await asyncio.sleep(0.1)  # Simulate streaming delay
        
        # Final chunk
        yield json.dumps({
            'type': 'stream_end',
            'model_id': model_id,
            'tokens_used': len(words) * 2
        }) + "\n"
    
    async def _mock_openai_provider(self, model: dict, messages: List[dict], **kwargs) -> dict:
        """Mock OpenAI provider for testing"""
        await asyncio.sleep(0.3)  # Simulate API latency
        
        last_message = messages[-1]['content'] if messages else "Hello"
        return {
            'content': f"[OpenAI {model['name']}] Response to: {last_message[:100]}...",
            'tokens': len(last_message.split()) * 3
        }
    
    async def _mock_anthropic_provider(self, model: dict, messages: List[dict], **kwargs) -> dict:
        """Mock Anthropic provider for testing"""
        await asyncio.sleep(0.2)  # Simulate API latency
        
        last_message = messages[-1]['content'] if messages else "Hello"
        return {
            'content': f"[Anthropic {model['name']}] Response to: {last_message[:100]}...",
            'tokens': len(last_message.split()) * 3
        }
    
    async def _mock_local_provider(self, model: dict, messages: List[dict], **kwargs) -> dict:
        """Mock local model provider for testing"""
        await asyncio.sleep(0.1)  # Simulate local inference
        
        last_message = messages[-1]['content'] if messages else "Hello"
        return {
            'content': f"[Local {model['name']}] Response to: {last_message[:100]}...",
            'tokens': len(last_message.split()) * 2
        }


class TokenManager:
    """Manages token counting and context truncation"""
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens for text using model tokenizer"""
        # Simplified token counting (in production use tiktoken or model-specific tokenizer)
        return len(text.split()) * 1.3  # Rough approximation
    
    def truncate_to_limit(self, messages: List[dict], limit: int = MAX_CONTEXT_TOKENS) -> List[dict]:
        """Truncate conversation to fit context window"""
        total_tokens = 0
        truncated = []
        
        # Keep system message if present
        if messages and messages[0].get('role') == 'system':
            system_message = messages[0]
            total_tokens = self.count_tokens(system_message['content'])
            truncated.append(system_message)
            messages = messages[1:]
        
        # Add messages from newest to oldest
        for message in reversed(messages):
            tokens = self.count_tokens(message['content'])
            if total_tokens + tokens > limit:
                break
            truncated.insert(len(truncated) if truncated and truncated[0].get('role') == 'system' else 0, message)
            total_tokens += tokens
        
        return truncated


class AIModelService:
    """Main AI Model Service - Integrated with shared database"""
    
    def __init__(self):
        self.app = FastAPI(title="AI Model Service (Integrated)", version="2.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("ai_model")
        self.cache = get_cache()
        self.model_registry = ModelRegistry(self.dal, self.cache)
        self.context_manager = ContextManager(self.dal, self.cache)
        self.inference_engine = InferenceEngine(self.dal, self.cache, self.model_registry)
        self.token_manager = TokenManager()
        
        # Metrics
        self.metrics = {
            "total_inferences": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "average_latency": 0
        }
        
        logger.info("✅ Connected to shared database and cache")
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = self.dal.health_check()
            return {
                "service": "ai_model",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "models_loaded": len(self.model_registry.models),
                "active_sessions": len(self.context_manager.active_contexts),
                "metrics": self.metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/models")
        async def create_model(model: ModelCreate):
            """Register new AI model"""
            try:
                model_config = {
                    "name": model.name,
                    "provider": model.provider,
                    "model_id": model.model_id,
                    "model_type": model.model_type,
                    "config": model.config,
                    "capabilities": model.capabilities,
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                    "temperature": model.temperature
                }
                
                registered = self.model_registry.register_model(model_config)
                return registered
                
            except Exception as e:
                logger.error(f"Failed to register model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def list_models():
            """List all available models"""
            return {
                "models": list(self.model_registry.models.values()),
                "count": len(self.model_registry.models)
            }
        
        @self.app.get("/models/{model_id}")
        async def get_model(model_id: str):
            """Get specific model details"""
            model = self.model_registry.get_model(model_id)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            return model
        
        @self.app.post("/inference")
        async def inference(request: InferenceRequest):
            """Perform model inference"""
            try:
                # Get context if session provided
                context = []
                if request.session_id and request.include_context:
                    context = self.context_manager.get_context(request.session_id)
                    
                    # Truncate context to fit model window
                    if context:
                        context = self.token_manager.truncate_to_limit(context)
                
                # Prepare inference request
                inference_request = {
                    "prompt": request.prompt,
                    "model_id": request.model_id,
                    "model_type": request.model_type,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "use_cache": request.use_cache
                }
                
                # Handle streaming
                if request.stream:
                    return StreamingResponse(
                        self.inference_engine.stream_inference(inference_request, context),
                        media_type="application/x-ndjson"
                    )
                
                # Regular inference
                result = await self.inference_engine.inference(inference_request, context)
                
                # Update metrics
                self.metrics["total_inferences"] += 1
                if result.get('cached'):
                    self.metrics["cache_hits"] += 1
                self.metrics["total_tokens"] += result.get('tokens_used', 0)
                
                # Update average latency
                current_avg = self.metrics["average_latency"]
                count = self.metrics["total_inferences"]
                self.metrics["average_latency"] = (
                    (current_avg * (count - 1) + result['latency_ms']) / count
                )
                
                # Add to conversation history
                if request.session_id:
                    self.context_manager.add_message(request.session_id, {
                        "role": "user",
                        "content": request.prompt
                    })
                    self.context_manager.add_message(request.session_id, {
                        "role": "assistant",
                        "content": result['response']
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/sessions/{session_id}/history")
        async def get_session_history(session_id: str, limit: int = 50):
            """Get conversation history for session"""
            context = self.context_manager.get_context(session_id, limit=limit)
            return {
                "session_id": session_id,
                "messages": context,
                "count": len(context)
            }
        
        @self.app.delete("/sessions/{session_id}/history")
        async def clear_session_history(session_id: str):
            """Clear conversation history for session"""
            self.context_manager.clear_context(session_id)
            return {"message": f"History cleared for session {session_id}"}
        
        @self.app.post("/sessions/{session_id}/messages")
        async def add_message(session_id: str, message: ConversationMessage):
            """Add message to conversation history"""
            self.context_manager.add_message(session_id, {
                "role": message.role,
                "content": message.content,
                "metadata": message.metadata
            })
            return {"message": "Message added to history"}
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics"""
            # Get additional metrics from DAL
            db_metrics = {}
            try:
                # In production, query actual metrics from database
                pass
            except:
                pass
            
            return {
                "service_metrics": self.metrics,
                "cache_stats": {
                    "size": len(self.cache.redis_client.keys("inference:*")),
                    "hit_rate": (
                        self.metrics["cache_hits"] / self.metrics["total_inferences"]
                        if self.metrics["total_inferences"] > 0 else 0
                    )
                },
                "database_metrics": db_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_agent_request(event):
            """Handle agent inference requests"""
            agent_id = event.data.get('agent_id')
            request = event.data.get('request')
            logger.info(f"Received inference request from agent {agent_id}")
        
        def on_workflow_task(event):
            """Handle workflow task requiring AI inference"""
            task_id = event.data.get('task_id')
            prompt = event.data.get('prompt')
            logger.info(f"Processing workflow task {task_id}")
        
        # Register handlers
        self.dal.event_bus.on(EventChannel.TASK_CREATED, on_workflow_task)
        self.dal.event_bus.on(EventChannel.SYSTEM_ALERT, on_agent_request)
        
        logger.info("Event handlers registered for cross-service communication")
    
    async def startup(self):
        """Startup tasks"""
        logger.info("AI Model Service (Integrated) starting up...")
        
        # Verify connections
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")
        
        # Load models from database (if any)
        # In production, would load persisted model configs
        
        logger.info(f"AI Model Service ready with {len(self.model_registry.models)} models")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("AI Model Service shutting down...")
        
        # Save active contexts to database
        # In production, persist conversation history
        
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    import uuid
    
    service = AIModelService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("AI_MODEL_PORT", 8105))
    logger.info(f"Starting AI Model Service (Integrated) on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()