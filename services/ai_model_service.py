#!/usr/bin/env python3
"""
AI Model Orchestration Microservice
Handles AI model routing, load balancing, and inference across multiple providers
Based on the architecture from technical_architecture.md
"""

import os
import sys
import json
import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging
import hashlib
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class ModelStatus(str, Enum):
    """Model availability status"""
    AVAILABLE = "available"
    BUSY = "busy"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"


# Pydantic Models
class ModelConfig(BaseModel):
    """Model configuration"""
    model_id: str
    provider: ModelProvider
    model_type: ModelType
    name: str
    version: Optional[str] = None
    max_tokens: int = 4096
    temperature_range: Tuple[float, float] = (0.0, 2.0)
    cost_per_1k_tokens: float = 0.001
    latency_ms: int = 1000
    quality_score: float = 0.9
    capabilities: List[str] = Field(default_factory=list)
    rate_limit: Optional[Dict[str, Any]] = None


class ModelInstance(BaseModel):
    """Individual model instance"""
    instance_id: str = Field(default_factory=lambda: str(os.urandom(8).hex()))
    config: ModelConfig
    status: ModelStatus = ModelStatus.AVAILABLE
    current_load: int = 0
    max_concurrent: int = 10
    total_requests: int = 0
    total_tokens: int = 0
    average_latency_ms: float = 0
    last_used: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 1.0


class InferenceRequest(BaseModel):
    """AI inference request"""
    model_type: ModelType
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: bool = False
    provider_preference: Optional[ModelProvider] = None
    requirements: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """AI inference response"""
    success: bool
    content: Optional[str] = None
    model_used: Optional[str] = None
    provider: Optional[ModelProvider] = None
    usage: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchInferenceRequest(BaseModel):
    """Batch inference request"""
    requests: List[InferenceRequest]
    parallel: bool = True
    max_parallel: int = 5
    callback_url: Optional[str] = None


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_id: str
    provider: ModelProvider
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_processed: int = 0
    average_latency_ms: float = 0
    total_cost: float = 0
    uptime_percentage: float = 100.0
    last_error: Optional[str] = None
    last_used: Optional[datetime] = None


class ModelLoadBalancer:
    """Load balancer for model selection"""
    
    def __init__(self):
        self.round_robin_counters: Dict[ModelType, int] = defaultdict(int)
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.cost_tracker: Dict[str, float] = defaultdict(float)
    
    def select_model(self, models: List[ModelInstance], request: InferenceRequest,
                    strategy: LoadBalancingStrategy) -> Optional[ModelInstance]:
        """Select best model based on strategy"""
        
        # Filter available models
        available = [m for m in models 
                    if m.status == ModelStatus.AVAILABLE 
                    and m.current_load < m.max_concurrent]
        
        if not available:
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            counter = self.round_robin_counters[request.model_type]
            selected = available[counter % len(available)]
            self.round_robin_counters[request.model_type] += 1
            return selected
        
        elif strategy == LoadBalancingStrategy.LEAST_LATENCY:
            return min(available, key=lambda m: m.average_latency_ms)
        
        elif strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            return min(available, key=lambda m: m.config.cost_per_1k_tokens)
        
        elif strategy == LoadBalancingStrategy.QUALITY_FIRST:
            return max(available, key=lambda m: m.config.quality_score)
        
        elif strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available)
        
        elif strategy == LoadBalancingStrategy.WEIGHTED:
            # Weight by quality and availability
            weights = [m.config.quality_score * (1 - m.current_load/m.max_concurrent) 
                      for m in available]
            return random.choices(available, weights=weights)[0]
        
        return available[0]


class CostTracker:
    """Track AI model usage costs"""
    
    def __init__(self):
        self.costs_by_provider: Dict[str, float] = defaultdict(float)
        self.costs_by_model: Dict[str, float] = defaultdict(float)
        self.daily_costs: Dict[str, float] = defaultdict(float)
        self.monthly_costs: Dict[str, float] = defaultdict(float)
    
    def track_cost(self, model: ModelInstance, tokens: int):
        """Track cost for model usage"""
        cost = (tokens / 1000) * model.config.cost_per_1k_tokens
        
        self.costs_by_provider[model.config.provider.value] += cost
        self.costs_by_model[model.config.model_id] += cost
        
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        self.daily_costs[today] += cost
        self.monthly_costs[month] += cost
        
        return cost
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost report"""
        return {
            "by_provider": dict(self.costs_by_provider),
            "by_model": dict(self.costs_by_model),
            "today": self.daily_costs.get(datetime.now().strftime("%Y-%m-%d"), 0),
            "this_month": self.monthly_costs.get(datetime.now().strftime("%Y-%m"), 0)
        }


class AIModelService:
    """Main AI Model Orchestration Service"""
    
    def __init__(self):
        self.app = FastAPI(title="AI Model Orchestration Service", version="1.0.0")
        
        # Model registry
        self.models: Dict[str, ModelInstance] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Components
        self.load_balancer = ModelLoadBalancer()
        self.cost_tracker = CostTracker()
        
        # Metrics
        self.metrics: Dict[str, ModelMetrics] = {}
        self.total_inferences = 0
        self.cache: Dict[str, Tuple[InferenceResponse, datetime]] = {}
        
        # Configuration
        self.default_strategy = LoadBalancingStrategy.WEIGHTED
        self.cache_ttl = 300  # 5 minutes
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        self._register_default_models()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_event_handlers(self):
        """Setup startup and shutdown handlers"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize service"""
            asyncio.create_task(self._monitor_models())
            asyncio.create_task(self._cleanup_cache())
            logger.info("AI Model Orchestration Service started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            logger.info("AI Model Orchestration Service shut down")
    
    def _register_default_models(self):
        """Register default AI models"""
        
        # OpenAI Models
        gpt4 = ModelConfig(
            model_id="gpt-4",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.CHAT,
            name="GPT-4",
            version="latest",
            max_tokens=8192,
            cost_per_1k_tokens=0.03,
            latency_ms=2000,
            quality_score=0.95,
            capabilities=["chat", "reasoning", "code", "analysis"]
        )
        self.model_configs["gpt-4"] = gpt4
        self.models["gpt-4-instance-1"] = ModelInstance(config=gpt4)
        
        gpt35 = ModelConfig(
            model_id="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.CHAT,
            name="GPT-3.5 Turbo",
            version="latest",
            max_tokens=4096,
            cost_per_1k_tokens=0.001,
            latency_ms=800,
            quality_score=0.85,
            capabilities=["chat", "basic_reasoning"]
        )
        self.model_configs["gpt-3.5-turbo"] = gpt35
        self.models["gpt-35-instance-1"] = ModelInstance(config=gpt35)
        
        # Anthropic Models
        claude3 = ModelConfig(
            model_id="claude-3-opus",
            provider=ModelProvider.ANTHROPIC,
            model_type=ModelType.CHAT,
            name="Claude 3 Opus",
            version="20240229",
            max_tokens=200000,
            cost_per_1k_tokens=0.015,
            latency_ms=1500,
            quality_score=0.96,
            capabilities=["chat", "reasoning", "code", "analysis", "vision"]
        )
        self.model_configs["claude-3-opus"] = claude3
        self.models["claude-3-instance-1"] = ModelInstance(config=claude3)
        
        # Google Models
        gemini = ModelConfig(
            model_id="gemini-pro",
            provider=ModelProvider.GOOGLE,
            model_type=ModelType.CHAT,
            name="Gemini Pro",
            version="latest",
            max_tokens=32768,
            cost_per_1k_tokens=0.00025,
            latency_ms=1200,
            quality_score=0.88,
            capabilities=["chat", "multimodal"]
        )
        self.model_configs["gemini-pro"] = gemini
        self.models["gemini-instance-1"] = ModelInstance(config=gemini)
        
        # Embedding Models
        ada = ModelConfig(
            model_id="text-embedding-ada-002",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.EMBEDDING,
            name="Ada Embedding",
            max_tokens=8191,
            cost_per_1k_tokens=0.0001,
            latency_ms=100,
            quality_score=0.9,
            capabilities=["embedding", "similarity"]
        )
        self.model_configs["text-embedding-ada-002"] = ada
        self.models["ada-instance-1"] = ModelInstance(config=ada)
        
        logger.info(f"Registered {len(self.models)} model instances")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check"""
            available_models = sum(1 for m in self.models.values() 
                                 if m.status == ModelStatus.AVAILABLE)
            return {
                "status": "healthy",
                "service": "ai_model_orchestration",
                "timestamp": datetime.now().isoformat(),
                "available_models": available_models,
                "total_models": len(self.models)
            }
        
        # Inference endpoints
        @self.app.post("/api/v1/inference", response_model=InferenceResponse)
        async def inference(request: InferenceRequest):
            """Execute AI model inference"""
            return await self._execute_inference(request)
        
        @self.app.post("/api/v1/inference/batch", response_model=List[InferenceResponse])
        async def batch_inference(request: BatchInferenceRequest, background_tasks: BackgroundTasks):
            """Execute batch inference"""
            if request.parallel:
                # Execute in parallel with concurrency limit
                semaphore = asyncio.Semaphore(request.max_parallel)
                
                async def limited_inference(req):
                    async with semaphore:
                        return await self._execute_inference(req)
                
                tasks = [limited_inference(req) for req in request.requests]
                results = await asyncio.gather(*tasks)
            else:
                # Execute sequentially
                results = []
                for req in request.requests:
                    result = await self._execute_inference(req)
                    results.append(result)
            
            # Send callback if configured
            if request.callback_url:
                background_tasks.add_task(self._send_callback, request.callback_url, results)
            
            return results
        
        # Model management
        @self.app.get("/api/v1/models")
        async def list_models(model_type: Optional[ModelType] = None,
                             provider: Optional[ModelProvider] = None):
            """List available models"""
            models = list(self.models.values())
            
            if model_type:
                models = [m for m in models if m.config.model_type == model_type]
            if provider:
                models = [m for m in models if m.config.provider == provider]
            
            return [
                {
                    "instance_id": m.instance_id,
                    "model_id": m.config.model_id,
                    "name": m.config.name,
                    "provider": m.config.provider.value,
                    "type": m.config.model_type.value,
                    "status": m.status.value,
                    "load": f"{m.current_load}/{m.max_concurrent}",
                    "success_rate": f"{m.success_rate:.2%}"
                }
                for m in models
            ]
        
        @self.app.post("/api/v1/models/register")
        async def register_model(config: ModelConfig):
            """Register a new model"""
            model_id = f"{config.model_id}-{len(self.models)+1}"
            instance = ModelInstance(
                instance_id=model_id,
                config=config
            )
            self.models[model_id] = instance
            self.model_configs[config.model_id] = config
            
            logger.info(f"Registered model: {model_id}")
            return {"message": "Model registered", "instance_id": model_id}
        
        @self.app.delete("/api/v1/models/{instance_id}")
        async def unregister_model(instance_id: str):
            """Unregister a model"""
            if instance_id not in self.models:
                raise HTTPException(status_code=404, detail="Model not found")
            
            del self.models[instance_id]
            logger.info(f"Unregistered model: {instance_id}")
            return {"message": "Model unregistered"}
        
        # Metrics and cost tracking
        @self.app.get("/api/v1/metrics")
        async def get_metrics():
            """Get service metrics"""
            model_metrics = []
            for model in self.models.values():
                model_metrics.append({
                    "model_id": model.config.model_id,
                    "provider": model.config.provider.value,
                    "total_requests": model.total_requests,
                    "total_tokens": model.total_tokens,
                    "average_latency_ms": model.average_latency_ms,
                    "success_rate": model.success_rate,
                    "status": model.status.value
                })
            
            return {
                "total_inferences": self.total_inferences,
                "models": model_metrics,
                "cache_size": len(self.cache),
                "cost_report": self.cost_tracker.get_cost_report()
            }
        
        @self.app.get("/api/v1/costs")
        async def get_costs():
            """Get cost tracking information"""
            return self.cost_tracker.get_cost_report()
        
        # Load balancing configuration
        @self.app.post("/api/v1/config/strategy")
        async def set_load_balancing_strategy(strategy: LoadBalancingStrategy):
            """Set default load balancing strategy"""
            self.default_strategy = strategy
            logger.info(f"Load balancing strategy set to: {strategy.value}")
            return {"strategy": strategy.value}
        
        @self.app.get("/api/v1/config")
        async def get_config():
            """Get service configuration"""
            return {
                "default_strategy": self.default_strategy.value,
                "cache_ttl": self.cache_ttl,
                "total_models": len(self.models),
                "providers": list(set(m.config.provider.value for m in self.models.values()))
            }
    
    async def _execute_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Execute model inference"""
        try:
            # Check cache
            cache_key = self._get_cache_key(request)
            if cache_key in self.cache:
                cached_response, cached_time = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_ttl:
                    logger.info(f"Cache hit for inference request")
                    return cached_response
            
            # Select model based on requirements and strategy
            eligible_models = [m for m in self.models.values() 
                             if m.config.model_type == request.model_type]
            
            if request.provider_preference:
                eligible_models = [m for m in eligible_models 
                                 if m.config.provider == request.provider_preference]
            
            if not eligible_models:
                return InferenceResponse(
                    success=False,
                    error="No eligible models available"
                )
            
            # Select model using load balancer
            model = self.load_balancer.select_model(
                eligible_models, request, self.default_strategy
            )
            
            if not model:
                return InferenceResponse(
                    success=False,
                    error="All models are busy or unavailable"
                )
            
            # Update model load
            model.current_load += 1
            model.last_used = datetime.now()
            
            try:
                # Simulate inference (in production, would call actual model API)
                start_time = time.time()
                await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate latency
                
                # Generate response
                if request.prompt:
                    response_text = f"Response from {model.config.name}: {request.prompt[:50]}..."
                elif request.messages:
                    response_text = f"Chat response from {model.config.name}"
                else:
                    response_text = f"Generated by {model.config.name}"
                
                # Calculate metrics
                tokens_used = len(response_text.split()) * 2  # Simple estimation
                latency_ms = int((time.time() - start_time) * 1000)
                cost = self.cost_tracker.track_cost(model, tokens_used)
                
                # Update model metrics
                model.total_requests += 1
                model.total_tokens += tokens_used
                model.average_latency_ms = (
                    (model.average_latency_ms * (model.total_requests - 1) + latency_ms) 
                    / model.total_requests
                )
                
                # Create response
                response = InferenceResponse(
                    success=True,
                    content=response_text,
                    model_used=model.config.model_id,
                    provider=model.config.provider,
                    usage={
                        "prompt_tokens": tokens_used // 2,
                        "completion_tokens": tokens_used // 2,
                        "total_tokens": tokens_used
                    },
                    latency_ms=latency_ms,
                    cost=cost,
                    metadata={
                        "instance_id": model.instance_id,
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens
                    }
                )
                
                # Cache response
                self.cache[cache_key] = (response, datetime.now())
                
                # Update global metrics
                self.total_inferences += 1
                
                return response
                
            finally:
                # Release model load
                model.current_load -= 1
                
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return InferenceResponse(
                success=False,
                error=str(e)
            )
    
    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "model_type": request.model_type.value,
            "prompt": request.prompt,
            "messages": str(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _monitor_models(self):
        """Monitor model health and availability"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            for model in self.models.values():
                # Simulate health checks
                if model.error_count > 5:
                    model.status = ModelStatus.ERROR
                elif model.current_load >= model.max_concurrent * 0.9:
                    model.status = ModelStatus.BUSY
                else:
                    model.status = ModelStatus.AVAILABLE
                
                # Update success rate
                if model.total_requests > 0:
                    model.success_rate = (model.total_requests - model.error_count) / model.total_requests
    
    async def _cleanup_cache(self):
        """Cleanup expired cache entries"""
        while True:
            await asyncio.sleep(60)  # Clean every minute
            
            now = datetime.now()
            expired_keys = []
            
            for key, (response, cached_time) in self.cache.items():
                if (now - cached_time).seconds >= self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    async def _send_callback(self, callback_url: str, results: List[InferenceResponse]):
        """Send callback with results"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(
                    callback_url,
                    json=[r.dict() for r in results]
                )
            logger.info(f"Sent callback to {callback_url}")
        except Exception as e:
            logger.error(f"Callback error: {e}")


def create_service():
    """Create and return the service instance"""
    return AIModelService()


if __name__ == "__main__":
    import uvicorn
    
    print("AI Model Orchestration Microservice")
    print("=" * 60)
    
    service = create_service()
    
    print("\n🤖 Starting AI Model Service on port 8005")
    print("\nEndpoints:")
    print("  • POST /api/v1/inference - Single inference")
    print("  • POST /api/v1/inference/batch - Batch inference")
    print("  • GET  /api/v1/models - List models")
    print("  • GET  /api/v1/metrics - Service metrics")
    print("  • GET  /api/v1/costs - Cost tracking")
    
    print("\n🧠 Available Models:")
    print("  • GPT-4 (OpenAI)")
    print("  • GPT-3.5 Turbo (OpenAI)")
    print("  • Claude 3 Opus (Anthropic)")
    print("  • Gemini Pro (Google)")
    print("  • Text Embedding Ada (OpenAI)")
    
    print("\n⚖️ Load Balancing Strategies:")
    print("  • Round Robin")
    print("  • Least Latency")
    print("  • Cost Optimized")
    print("  • Quality First")
    print("  • Weighted")
    
    uvicorn.run(service.app, host="0.0.0.0", port=8005, reload=False)