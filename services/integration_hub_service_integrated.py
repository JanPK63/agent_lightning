#!/usr/bin/env python3
"""
Integration Hub Microservice - Integrated with Shared Database
Manages external integrations, APIs, webhooks, and data transformations
Using shared PostgreSQL and Redis for configuration and history
Based on SA-005: Integration Hub Service
"""

import os
import sys
import json
import asyncio
import uuid
import time
import hashlib
import hmac
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
import aiohttp
from cryptography.fernet import Fernet

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
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


class IntegrationType(str, Enum):
    """Types of integrations"""
    API = "api"
    WEBHOOK = "webhook"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"


class WebhookStatus(str, Enum):
    """Webhook processing status"""
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRY = "retry"


# Constants
MAX_RETRIES = 3
RETRY_DELAY = 5
DEFAULT_CACHE_TTL = 300  # 5 minutes


# Pydantic models
class IntegrationCreate(BaseModel):
    """Create new integration"""
    name: str = Field(description="Integration name")
    type: IntegrationType = Field(description="Integration type")
    provider: Optional[str] = Field(default=None, description="Provider name")
    config: Dict[str, Any] = Field(description="Integration configuration")
    credentials: Optional[Dict[str, str]] = Field(default=None, description="API credentials")
    rate_limit: Optional[int] = Field(default=None, description="Rate limit")
    rate_window: Optional[int] = Field(default=3600, description="Rate window in seconds")


class WebhookSubscribe(BaseModel):
    """Subscribe to webhook"""
    integration_id: str = Field(description="Integration ID")
    event_type: str = Field(description="Event type to subscribe")
    callback_url: str = Field(description="Callback URL")
    secret: Optional[str] = Field(default=None, description="Webhook secret")


class APIRequest(BaseModel):
    """API request model"""
    integration_id: str = Field(description="Integration ID")
    method: str = Field(default="GET", description="HTTP method")
    endpoint: str = Field(description="API endpoint")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Request data")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Additional headers")


class DataTransform(BaseModel):
    """Data transformation request"""
    source_data: Dict[str, Any] = Field(description="Source data")
    mapping: Dict[str, str] = Field(description="Field mapping")
    target_format: Optional[str] = Field(default="json", description="Target format")


class CredentialVault:
    """Secure credential management"""
    
    def __init__(self):
        # Get or generate encryption key
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            logger.warning("Generated new encryption key - set ENCRYPTION_KEY env var for production")
        
        self.cipher = Fernet(key if isinstance(key, bytes) else key.encode())
    
    def encrypt_credentials(self, credentials: dict) -> bytes:
        """Encrypt API credentials before storage"""
        json_str = json.dumps(credentials)
        return self.cipher.encrypt(json_str.encode())
    
    def decrypt_credentials(self, encrypted: bytes) -> dict:
        """Decrypt credentials for use"""
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())


class RateLimiter:
    """API rate limiting with Redis"""
    
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


class APICache:
    """Cache API responses"""
    
    def __init__(self, cache_manager):
        self.cache = cache_manager
    
    def cache_response(self, endpoint: str, params: dict, 
                      response: dict, ttl: int = DEFAULT_CACHE_TTL):
        """Cache API responses to reduce external calls"""
        # Create stable cache key
        param_str = json.dumps(params, sort_keys=True)
        cache_key = f"api:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"
        self.cache.set(cache_key, response, ttl=ttl)
    
    def get_cached(self, endpoint: str, params: dict) -> Optional[dict]:
        """Get cached response if available"""
        param_str = json.dumps(params, sort_keys=True)
        cache_key = f"api:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"
        return self.cache.get(cache_key)


class IntegrationHubService:
    """Main Integration Hub Service - Integrated with shared database"""
    
    def __init__(self):
        self.app = FastAPI(title="Integration Hub Service (Integrated)", version="2.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("integration_hub")
        self.cache = get_cache()
        self.vault = CredentialVault()
        self.rate_limiter = RateLimiter(self.cache.redis_client)
        self.api_cache = APICache(self.cache)
        
        # HTTP session for API calls
        self.http_session = None
        
        # Webhook processing queue
        self.webhook_queue = []
        
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
                "service": "integration_hub",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "webhook_queue": len(self.webhook_queue),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/integrations")
        async def create_integration(integration: IntegrationCreate):
            """Create new integration"""
            try:
                # Prepare integration data
                integration_data = integration.dict()
                
                # Encrypt credentials if provided
                if integration.credentials:
                    encrypted = self.vault.encrypt_credentials(integration.credentials)
                    integration_data['credentials'] = encrypted
                    
                # Store in database (using generic storage since we don't have integration table yet)
                # In production, would extend models.py with Integration model
                with self.dal.db.get_db() as session:
                    # For now, store as JSON in cache
                    integration_id = str(uuid.uuid4())
                    integration_data['id'] = integration_id
                    integration_data['created_at'] = datetime.utcnow().isoformat()
                    
                    # Store in cache as temporary solution
                    self.cache.set(f"integration:{integration_id}", integration_data, ttl=86400)
                    
                    # Emit event
                    self.dal.event_bus.emit(EventChannel.SYSTEM_EVENT, {
                        'type': 'integration_created',
                        'integration_id': integration_id,
                        'name': integration.name
                    })
                    
                    logger.info(f"Created integration {integration_id}: {integration.name}")
                    return integration_data
                    
            except Exception as e:
                logger.error(f"Failed to create integration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/integrations")
        async def list_integrations():
            """List all integrations"""
            try:
                # Get all integrations from cache (temporary solution)
                integrations = []
                for key in self.cache.redis_client.keys("integration:*"):
                    integration = self.cache.get(key)
                    if integration:
                        # Don't expose encrypted credentials
                        if 'credentials' in integration:
                            integration['credentials'] = '***encrypted***'
                        integrations.append(integration)
                
                return {
                    "integrations": integrations,
                    "count": len(integrations)
                }
            except Exception as e:
                logger.error(f"Failed to list integrations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/call")
        async def make_api_call(request: APIRequest, background_tasks: BackgroundTasks):
            """Make external API call"""
            try:
                # Get integration config
                integration = self.cache.get(f"integration:{request.integration_id}")
                if not integration:
                    raise HTTPException(status_code=404, detail="Integration not found")
                
                # Check cache first
                cached_response = self.api_cache.get_cached(
                    request.endpoint, 
                    request.data or {}
                )
                if cached_response:
                    logger.info(f"Returning cached response for {request.endpoint}")
                    return cached_response
                
                # Check rate limit
                if integration.get('rate_limit'):
                    allowed = await self.rate_limiter.check_rate_limit(
                        request.integration_id,
                        integration['rate_limit'],
                        integration.get('rate_window', 3600)
                    )
                    if not allowed:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Decrypt credentials if needed
                headers = request.headers or {}
                if integration.get('credentials'):
                    creds = self.vault.decrypt_credentials(integration['credentials'])
                    # Add auth headers based on provider
                    if 'api_key' in creds:
                        headers['Authorization'] = f"Bearer {creds['api_key']}"
                
                # Make API call
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=request.method,
                        url=request.endpoint,
                        json=request.data,
                        headers=headers
                    ) as response:
                        response_data = await response.json()
                        latency_ms = int((time.time() - start_time) * 1000)
                        
                        # Record in history
                        history_entry = {
                            'integration_id': request.integration_id,
                            'method': request.method,
                            'endpoint': request.endpoint,
                            'request_data': request.data,
                            'response_data': response_data,
                            'status_code': response.status,
                            'latency_ms': latency_ms,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        
                        # Store history in cache (temporary)
                        history_key = f"api_history:{uuid.uuid4()}"
                        self.cache.set(history_key, history_entry, ttl=86400)
                        
                        # Cache successful responses
                        if response.status == 200:
                            self.api_cache.cache_response(
                                request.endpoint,
                                request.data or {},
                                response_data
                            )
                        
                        # Record metrics
                        self.dal.record_metric('api_call_latency', latency_ms, {
                            'integration': integration['name'],
                            'endpoint': request.endpoint
                        })
                        
                        return {
                            'status': response.status,
                            'data': response_data,
                            'latency_ms': latency_ms
                        }
                        
            except aiohttp.ClientError as e:
                logger.error(f"API call failed: {e}")
                raise HTTPException(status_code=503, detail=f"External API error: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to make API call: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/webhooks/subscribe")
        async def subscribe_webhook(subscription: WebhookSubscribe):
            """Subscribe to webhook events"""
            try:
                webhook_id = str(uuid.uuid4())
                webhook_data = {
                    'id': webhook_id,
                    'integration_id': subscription.integration_id,
                    'event_type': subscription.event_type,
                    'callback_url': subscription.callback_url,
                    'secret': subscription.secret or str(uuid.uuid4()),
                    'active': True,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Store webhook subscription
                self.cache.set(f"webhook:{webhook_id}", webhook_data, ttl=None)
                
                logger.info(f"Created webhook subscription {webhook_id}")
                return webhook_data
                
            except Exception as e:
                logger.error(f"Failed to create webhook subscription: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/webhooks/{webhook_id}/receive")
        async def receive_webhook(
            webhook_id: str,
            request: Request,
            background_tasks: BackgroundTasks,
            x_signature: Optional[str] = Header(None)
        ):
            """Receive incoming webhook"""
            try:
                # Get webhook config
                webhook = self.cache.get(f"webhook:{webhook_id}")
                if not webhook:
                    raise HTTPException(status_code=404, detail="Webhook not found")
                
                # Get request body
                body = await request.body()
                data = json.loads(body)
                
                # Verify signature if secret is set
                if webhook.get('secret') and x_signature:
                    expected_sig = hmac.new(
                        webhook['secret'].encode(),
                        body,
                        hashlib.sha256
                    ).hexdigest()
                    
                    if not hmac.compare_digest(expected_sig, x_signature):
                        raise HTTPException(status_code=401, detail="Invalid signature")
                
                # Store webhook event
                event_id = str(uuid.uuid4())
                event_data = {
                    'id': event_id,
                    'webhook_id': webhook_id,
                    'event_data': data,
                    'status': WebhookStatus.PENDING.value,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                self.cache.set(f"webhook_event:{event_id}", event_data, ttl=86400)
                
                # Queue for processing
                self.webhook_queue.append(event_id)
                background_tasks.add_task(self._process_webhook_event, event_id)
                
                logger.info(f"Received webhook event {event_id} for {webhook_id}")
                return {"status": "accepted", "event_id": event_id}
                
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                logger.error(f"Failed to process webhook: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/transform")
        async def transform_data(transform: DataTransform):
            """Transform data between formats"""
            try:
                result = {}
                
                for target_field, source_path in transform.mapping.items():
                    # Simple path extraction (in production would be more sophisticated)
                    value = self._extract_value(transform.source_data, source_path)
                    if value is not None:
                        result[target_field] = value
                
                return {
                    "transformed_data": result,
                    "format": transform.target_format
                }
                
            except Exception as e:
                logger.error(f"Failed to transform data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/history/api")
        async def get_api_history(integration_id: Optional[str] = None):
            """Get API call history"""
            try:
                history = []
                for key in self.cache.redis_client.keys("api_history:*"):
                    entry = self.cache.get(key)
                    if entry:
                        if not integration_id or entry.get('integration_id') == integration_id:
                            history.append(entry)
                
                # Sort by timestamp
                history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                return {
                    "history": history[:100],  # Limit to last 100
                    "count": len(history)
                }
                
            except Exception as e:
                logger.error(f"Failed to get API history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _extract_value(self, data: dict, path: str) -> Any:
        """Extract value from nested dict using dot notation"""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    async def _process_webhook_event(self, event_id: str):
        """Process webhook event asynchronously"""
        try:
            event = self.cache.get(f"webhook_event:{event_id}")
            if not event:
                logger.error(f"Webhook event {event_id} not found")
                return
            
            webhook = self.cache.get(f"webhook:{event['webhook_id']}")
            if not webhook:
                logger.error(f"Webhook {event['webhook_id']} not found")
                return
            
            # Process based on event type
            logger.info(f"Processing webhook event {event_id} of type {webhook['event_type']}")
            
            # Emit event for other services
            self.dal.event_bus.emit(EventChannel.SYSTEM_EVENT, {
                'type': 'webhook_received',
                'webhook_id': event['webhook_id'],
                'event_type': webhook['event_type'],
                'data': event['event_data']
            })
            
            # Update status
            event['status'] = WebhookStatus.PROCESSED.value
            event['processed_at'] = datetime.utcnow().isoformat()
            self.cache.set(f"webhook_event:{event_id}", event, ttl=86400)
            
            # Remove from queue
            if event_id in self.webhook_queue:
                self.webhook_queue.remove(event_id)
                
        except Exception as e:
            logger.error(f"Error processing webhook event {event_id}: {e}")
            # Implement retry logic here
    
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_workflow_event(event):
            """Handle workflow events that may trigger integrations"""
            workflow_id = event.data.get('workflow_id')
            event_type = event.data.get('type')
            logger.info(f"Received workflow event {event_type} for {workflow_id}")
        
        def on_task_event(event):
            """Handle task events that may require external API calls"""
            task_id = event.data.get('task_id')
            logger.info(f"Received task event for {task_id}")
        
        # Register handlers
        self.dal.event_bus.on(EventChannel.WORKFLOW_COMPLETED, on_workflow_event)
        self.dal.event_bus.on(EventChannel.TASK_CREATED, on_task_event)
        
        logger.info("Event handlers registered for cross-service communication")
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Integration Hub Service (Integrated) starting up...")
        
        # Verify connections
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")
        
        # Create HTTP session
        self.http_session = aiohttp.ClientSession()
        
        logger.info("Integration Hub ready")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Integration Hub Service shutting down...")
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = IntegrationHubService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("INTEGRATION_HUB_PORT", 8004))
    logger.info(f"Starting Integration Hub Service (Integrated) on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()