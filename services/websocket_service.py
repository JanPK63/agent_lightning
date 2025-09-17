#!/usr/bin/env python3
"""
WebSocket Service for Real-time Updates
Provides real-time communication for all microservices
"""

import os
import sys
import json
import asyncio
import uuid
import time
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from enum import Enum
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as aioredis
from jose import jwt, JWTError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of real-time events"""
    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DEPLOYED = "agent.deployed"
    AGENT_STATUS = "agent.status"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_PROGRESS = "workflow.progress"
    
    # Task events
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    # Integration events
    INTEGRATION_TRIGGERED = "integration.triggered"
    INTEGRATION_COMPLETED = "integration.completed"
    INTEGRATION_ERROR = "integration.error"
    
    # AI Model events
    INFERENCE_STARTED = "inference.started"
    INFERENCE_COMPLETED = "inference.completed"
    MODEL_SWITCHED = "model.switched"
    
    # System events
    SYSTEM_ALERT = "system.alert"
    SYSTEM_METRIC = "system.metric"
    SERVICE_HEALTH = "service.health"
    
    # Collaboration events
    USER_JOINED = "user.joined"
    USER_LEFT = "user.left"
    USER_TYPING = "user.typing"
    DOCUMENT_UPDATED = "document.updated"
    
    # Custom events
    CUSTOM = "custom"


class Channel(str, Enum):
    """WebSocket channels/rooms"""
    GLOBAL = "global"
    ORGANIZATION = "organization"
    PROJECT = "project"
    WORKFLOW = "workflow"
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    channel: Channel
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BroadcastRequest(BaseModel):
    """Request to broadcast a message"""
    type: EventType
    channel: Channel
    channel_id: str
    data: Dict[str, Any]
    sender: Optional[str] = None


class SubscriptionRequest(BaseModel):
    """Request to subscribe to a channel"""
    channel: Channel
    channel_id: str
    filters: Optional[Dict[str, Any]] = None


class ConnectionManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        # Active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Channel subscriptions
        self.subscriptions: Dict[str, Set[str]] = {}  # channel -> set of client_ids
        
        # Client metadata
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Message history (last 100 per channel)
        self.message_history: Dict[str, List[WebSocketMessage]] = {}
        
        # Redis for pub/sub across multiple instances
        self.redis_client: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
    
    async def init_redis(self):
        """Initialize Redis connection for pub/sub"""
        try:
            self.redis_client = None  # Disable Redis for now
            return
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379",
                decode_responses=True
            )
            self.pubsub = self.redis_client.pubsub()
            logger.info("Redis pub/sub initialized")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using in-memory only.")
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Dict[str, Any] = None):
        """Accept and register a new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = metadata or {}
        
        # Subscribe to user's personal channel
        user_channel = f"{Channel.USER}:{client_id}"
        await self.subscribe(client_id, Channel.USER, client_id)
        
        # Send connection confirmation
        await self.send_personal_message(
            WebSocketMessage(
                type=EventType.USER_JOINED,
                channel=Channel.SYSTEM,
                data={"client_id": client_id, "status": "connected"}
            ),
            client_id
        )
        
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove a connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Remove from all subscriptions
            for channel_key in list(self.subscriptions.keys()):
                if client_id in self.subscriptions[channel_key]:
                    self.subscriptions[channel_key].remove(client_id)
                    if not self.subscriptions[channel_key]:
                        del self.subscriptions[channel_key]
            
            # Clean up metadata
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]
            
            logger.info(f"Client {client_id} disconnected")
    
    async def subscribe(self, client_id: str, channel: Channel, channel_id: str):
        """Subscribe a client to a channel"""
        channel_key = f"{channel.value}:{channel_id}"
        
        if channel_key not in self.subscriptions:
            self.subscriptions[channel_key] = set()
            
            # Subscribe to Redis channel if available
            if self.pubsub:
                await self.pubsub.subscribe(channel_key)
        
        self.subscriptions[channel_key].add(client_id)
        
        # Send recent message history
        if channel_key in self.message_history:
            for message in self.message_history[channel_key][-10:]:  # Last 10 messages
                await self.send_personal_message(message, client_id)
        
        logger.info(f"Client {client_id} subscribed to {channel_key}")
    
    async def unsubscribe(self, client_id: str, channel: Channel, channel_id: str):
        """Unsubscribe a client from a channel"""
        channel_key = f"{channel.value}:{channel_id}"
        
        if channel_key in self.subscriptions and client_id in self.subscriptions[channel_key]:
            self.subscriptions[channel_key].remove(client_id)
            
            if not self.subscriptions[channel_key]:
                del self.subscriptions[channel_key]
                
                # Unsubscribe from Redis channel if no more clients
                if self.pubsub:
                    await self.pubsub.unsubscribe(channel_key)
        
        logger.info(f"Client {client_id} unsubscribed from {channel_key}")
    
    async def send_personal_message(self, message: WebSocketMessage, client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                # Convert datetime to ISO format for JSON serialization
                msg_dict = message.dict()
                if 'timestamp' in msg_dict and hasattr(msg_dict['timestamp'], 'isoformat'):
                    msg_dict['timestamp'] = msg_dict['timestamp'].isoformat()
                await websocket.send_json(msg_dict)
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: WebSocketMessage, channel: Channel, channel_id: str):
        """Broadcast a message to all subscribers of a channel"""
        channel_key = f"{channel.value}:{channel_id}"
        
        # Store in history
        if channel_key not in self.message_history:
            self.message_history[channel_key] = []
        self.message_history[channel_key].append(message)
        
        # Keep only last 100 messages
        if len(self.message_history[channel_key]) > 100:
            self.message_history[channel_key] = self.message_history[channel_key][-100:]
        
        # Publish to Redis for other instances
        if self.redis_client:
            await self.redis_client.publish(channel_key, json.dumps(message.dict()))
        
        # Send to local subscribers
        if channel_key in self.subscriptions:
            disconnected_clients = []
            
            for client_id in self.subscriptions[channel_key]:
                if client_id in self.active_connections:
                    try:
                        msg_dict = message.dict()
                        if 'timestamp' in msg_dict and hasattr(msg_dict['timestamp'], 'isoformat'):
                            msg_dict['timestamp'] = msg_dict['timestamp'].isoformat()
                        await self.active_connections[client_id].send_json(msg_dict)
                    except Exception as e:
                        logger.error(f"Error broadcasting to {client_id}: {e}")
                        disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    async def broadcast_to_organization(self, message: WebSocketMessage, org_id: str):
        """Broadcast to all users in an organization"""
        await self.broadcast(message, Channel.ORGANIZATION, org_id)
    
    async def broadcast_global(self, message: WebSocketMessage):
        """Broadcast to all connected clients"""
        await self.broadcast(message, Channel.GLOBAL, "all")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "total_channels": len(self.subscriptions),
            "channels": {
                channel_key: len(clients) 
                for channel_key, clients in self.subscriptions.items()
            },
            "message_history_size": sum(
                len(messages) for messages in self.message_history.values()
            )
        }


class WebSocketService:
    """Main WebSocket Service"""
    
    def __init__(self):
        self.app = FastAPI(title="WebSocket Service", version="1.0.0")
        self.manager = ConnectionManager()
        
        # JWT configuration
        self.jwt_secret = os.getenv("JWT_SECRET", "change-this-secret-key-in-production")
        self.jwt_algorithm = "HS256"
        
        # Metrics
        self.metrics = {
            "total_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0
        }
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
    
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
            await self.manager.init_redis()
            asyncio.create_task(self._redis_listener())
            logger.info("WebSocket Service started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            if self.manager.redis_client:
                await self.manager.redis_client.close()
            logger.info("WebSocket Service shut down")
    
    async def _redis_listener(self):
        """Listen for Redis pub/sub messages"""
        if not self.manager.pubsub:
            return
        
        try:
            while True:
                message = await self.manager.pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    # Forward to local subscribers
                    channel_key = message['channel']
                    if channel_key in self.manager.subscriptions:
                        data = json.loads(message['data'])
                        ws_message = WebSocketMessage(**data)
                        
                        for client_id in self.manager.subscriptions[channel_key]:
                            await self.manager.send_personal_message(ws_message, client_id)
                
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check"""
            stats = self.manager.get_connection_stats()
            return {
                "status": "healthy",
                "service": "websocket",
                "timestamp": datetime.now().isoformat(),
                "connections": stats["total_connections"],
                "metrics": self.metrics
            }
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """Main WebSocket endpoint"""
            # Extract token from query params or headers
            token = websocket.query_params.get("token")
            
            # Verify authentication (simplified)
            user_data = await self._verify_token(token) if token else None
            
            # Connect
            await self.manager.connect(websocket, client_id, metadata=user_data)
            self.metrics["total_connections"] += 1
            
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_json()
                    self.metrics["messages_received"] += 1
                    
                    # Handle different message types
                    if data.get("action") == "subscribe":
                        await self.manager.subscribe(
                            client_id,
                            Channel(data["channel"]),
                            data["channel_id"]
                        )
                    
                    elif data.get("action") == "unsubscribe":
                        await self.manager.unsubscribe(
                            client_id,
                            Channel(data["channel"]),
                            data["channel_id"]
                        )
                    
                    elif data.get("action") == "broadcast":
                        message = WebSocketMessage(
                            type=EventType(data.get("type", EventType.CUSTOM)),
                            channel=Channel(data["channel"]),
                            data=data.get("data", {}),
                            sender=client_id
                        )
                        await self.manager.broadcast(
                            message,
                            Channel(data["channel"]),
                            data["channel_id"]
                        )
                        self.metrics["messages_sent"] += 1
                    
                    elif data.get("action") == "ping":
                        # Respond to ping
                        await websocket.send_json({"action": "pong", "timestamp": time.time()})
                        
            except WebSocketDisconnect:
                self.manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"WebSocket error for {client_id}: {e}")
                self.metrics["errors"] += 1
                self.manager.disconnect(client_id)
        
        @self.app.post("/api/v1/broadcast")
        async def broadcast_message(request: BroadcastRequest):
            """HTTP endpoint to broadcast messages"""
            message = WebSocketMessage(
                type=request.type,
                channel=request.channel,
                data=request.data,
                sender=request.sender
            )
            
            await self.manager.broadcast(message, request.channel, request.channel_id)
            self.metrics["messages_sent"] += 1
            
            return {"status": "broadcasted", "message_id": message.id}
        
        @self.app.get("/api/v1/connections")
        async def get_connections():
            """Get connection statistics"""
            return self.manager.get_connection_stats()
        
        @self.app.get("/api/v1/history/{channel}/{channel_id}")
        async def get_message_history(channel: Channel, channel_id: str, limit: int = 50):
            """Get message history for a channel"""
            channel_key = f"{channel.value}:{channel_id}"
            
            if channel_key in self.manager.message_history:
                messages = self.manager.message_history[channel_key][-limit:]
                return {"messages": [m.dict() for m in messages]}
            
            return {"messages": []}
    
    async def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return {
                "user_id": payload.get("sub"),
                "org_id": payload.get("org"),
                "roles": payload.get("roles", [])
            }
        except JWTError:
            return None


def create_service():
    """Create and return the service instance"""
    return WebSocketService()


if __name__ == "__main__":
    import uvicorn
    
    print("WebSocket Service")
    print("=" * 60)
    
    service = create_service()
    
    print("\n🔌 Starting WebSocket Service on port 8007")
    print("\nFeatures:")
    print("  • Real-time bidirectional communication")
    print("  • Channel-based pub/sub messaging")
    print("  • Redis pub/sub for horizontal scaling")
    print("  • Message history and replay")
    print("  • JWT authentication support")
    
    print("\nEndpoints:")
    print("  • WS   /ws/{client_id} - WebSocket connection")
    print("  • POST /api/v1/broadcast - HTTP broadcast")
    print("  • GET  /api/v1/connections - Connection stats")
    print("  • GET  /api/v1/history/{channel}/{channel_id} - Message history")
    
    print("\nEvent Types:")
    print("  • Agent events (created, updated, deployed)")
    print("  • Workflow events (started, completed, progress)")
    print("  • Integration events (triggered, completed)")
    print("  • System events (alerts, metrics, health)")
    
    uvicorn.run(service.app, host="0.0.0.0", port=8007, reload=False)