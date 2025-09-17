#!/usr/bin/env python3
"""
WebSocket Service - Integrated with Redis Event Bus
Provides real-time bidirectional communication with event routing
Using Redis pub/sub for scalable event distribution
Based on SA-008: WebSocket Service Integration
"""

import os
import sys
import json
import asyncio
import uuid
import time
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as aioredis
from jose import jwt, JWTError

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared components
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types"""
    # Client -> Server
    AUTHENTICATE = "authenticate"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"
    SEND_MESSAGE = "send_message"
    PING = "ping"
    
    # Server -> Client
    AUTHENTICATED = "authenticated"
    EVENT = "event"
    MESSAGE = "message"
    ERROR = "error"
    ROOM_JOINED = "room_joined"
    ROOM_LEFT = "room_left"
    MEMBER_JOINED = "member_joined"
    MEMBER_LEFT = "member_left"
    PONG = "pong"
    CONNECTION_ID = "connection_id"


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


# Constants
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
JWT_ALGORITHM = "HS256"
HEARTBEAT_INTERVAL = 30  # seconds
CONNECTION_TIMEOUT = 60  # seconds
SERVER_ID = os.getenv("SERVER_ID", f"ws-{uuid.uuid4().hex[:8]}")


class WebSocketAuth:
    """Handles WebSocket authentication"""
    
    def __init__(self, secret_key: str = JWT_SECRET_KEY, algorithm: str = JWT_ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    async def authenticate_connection(self, token: str) -> Optional[dict]:
        """Authenticate WebSocket connection with JWT token"""
        try:
            # Verify JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "roles": payload.get("roles", []),
                "authenticated": True
            }
        except JWTError as e:
            logger.error(f"WebSocket auth failed: {e}")
            return None
    
    def check_permission(self, user_roles: List[str], action: str) -> bool:
        """Check if user has permission for action"""
        permission_map = {
            "subscribe_all": ["admin"],
            "create_room": ["admin", "developer"],
            "broadcast": ["admin", "developer", "operator"],
            "join_private_room": ["admin", "developer"]
        }
        
        required_roles = permission_map.get(action, [])
        if not required_roles:  # No specific permission required
            return True
        
        return any(role in required_roles for role in user_roles)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.connections: Dict[str, dict] = {}  # connection_id -> connection info
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.connection_key = f"connections:{SERVER_ID}"
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        
        # Store connection locally
        self.connections[connection_id] = {
            "websocket": websocket,
            "user_id": None,
            "authenticated": False,
            "connected_at": datetime.utcnow().isoformat(),
            "last_activity": time.time(),
            "rooms": set(),
            "subscriptions": set()
        }
        
        # Send connection ID to client
        await websocket.send_json({
            "type": MessageType.CONNECTION_ID.value,
            "connection_id": connection_id
        })
        
        logger.info(f"WebSocket connection established: {connection_id}")
    
    async def authenticate(self, connection_id: str, user_data: dict):
        """Mark connection as authenticated"""
        if connection_id in self.connections:
            conn = self.connections[connection_id]
            conn["authenticated"] = True
            conn["user_id"] = user_data["user_id"]
            conn["user_data"] = user_data
            
            # Track user connections
            user_id = user_data["user_id"]
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
            # Store in Redis for cluster awareness
            self.redis.hset(self.connection_key, connection_id, json.dumps({
                "user_id": user_id,
                "server_id": SERVER_ID,
                "connected_at": conn["connected_at"],
                "metadata": {
                    "email": user_data.get("email"),
                    "roles": user_data.get("roles", [])
                }
            }))
            
            # Add to user's connections in Redis
            user_key = f"user:{user_id}:connections"
            self.redis.sadd(user_key, connection_id)
            
            logger.info(f"Connection {connection_id} authenticated for user {user_id}")
    
    async def disconnect(self, connection_id: str):
        """Handle connection disconnect"""
        if connection_id not in self.connections:
            return
        
        conn = self.connections[connection_id]
        user_id = conn.get("user_id")
        
        # Leave all rooms
        for room in list(conn["rooms"]):
            await self.leave_room(connection_id, room)
        
        # Clean up user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
            
            # Remove from Redis
            user_key = f"user:{user_id}:connections"
            self.redis.srem(user_key, connection_id)
        
        # Remove from Redis connection registry
        self.redis.hdel(self.connection_key, connection_id)
        
        # Remove local connection
        del self.connections[connection_id]
        
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: dict):
        """Send message to specific connection"""
        if connection_id in self.connections:
            conn = self.connections[connection_id]
            try:
                await conn["websocket"].send_json(message)
                conn["last_activity"] = time.time()
            except Exception as e:
                logger.error(f"Failed to send to {connection_id}: {e}")
                await self.disconnect(connection_id)
    
    async def broadcast_to_user(self, user_id: str, message: dict):
        """Send message to all connections of a user"""
        if user_id in self.user_connections:
            for conn_id in self.user_connections[user_id]:
                await self.send_to_connection(conn_id, message)
    
    async def broadcast_to_all(self, message: dict, exclude: Set[str] = None):
        """Broadcast message to all authenticated connections"""
        exclude = exclude or set()
        
        for conn_id, conn in self.connections.items():
            if conn_id not in exclude and conn["authenticated"]:
                await self.send_to_connection(conn_id, message)
    
    async def join_room(self, connection_id: str, room_id: str):
        """Add connection to room"""
        if connection_id in self.connections:
            self.connections[connection_id]["rooms"].add(room_id)
            
            # Add to Redis room
            room_key = f"room:{room_id}:members"
            self.redis.sadd(room_key, connection_id)
            
            logger.info(f"Connection {connection_id} joined room {room_id}")
    
    async def leave_room(self, connection_id: str, room_id: str):
        """Remove connection from room"""
        if connection_id in self.connections:
            self.connections[connection_id]["rooms"].discard(room_id)
            
            # Remove from Redis room
            room_key = f"room:{room_id}:members"
            self.redis.srem(room_key, connection_id)
            
            logger.info(f"Connection {connection_id} left room {room_id}")
    
    async def broadcast_to_room(self, room_id: str, message: dict, exclude: Set[str] = None):
        """Send message to all connections in a room"""
        exclude = exclude or set()
        
        # Get room members from Redis
        room_key = f"room:{room_id}:members"
        members = self.redis.smembers(room_key)
        
        # Send to local connections
        for member_id in members:
            if member_id not in exclude and member_id in self.connections:
                await self.send_to_connection(member_id, message)
        
        # Publish to Redis for other servers
        self.redis.publish(f"room:{room_id}", json.dumps({
            "server_id": SERVER_ID,
            "message": message,
            "exclude": list(exclude)
        }))


class EventRouter:
    """Routes events from Redis to WebSocket connections"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.redis = None
        self.pubsub = None
        self.running = False
    
    async def start(self):
        """Start event routing"""
        # Create separate Redis connection for pub/sub
        self.redis = await aioredis.from_url(
            "redis://localhost:6379",
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )
        self.pubsub = self.redis.pubsub()
        
        # Subscribe to event channels
        await self.setup_subscriptions()
        
        # Start listening
        self.running = True
        asyncio.create_task(self.event_listener())
        
        logger.info("Event router started")
    
    async def stop(self):
        """Stop event routing"""
        self.running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
    
    async def setup_subscriptions(self):
        """Subscribe to Redis event channels"""
        # Subscribe to all EventChannel channels
        for channel in EventChannel:
            await self.pubsub.subscribe(channel.value)
            logger.info(f"Subscribed to channel: {channel.value}")
        
        # Subscribe to room broadcast channel
        await self.pubsub.subscribe(f"room:*")
        
        # Subscribe to server-specific channel
        await self.pubsub.subscribe(f"server:{SERVER_ID}")
    
    async def event_listener(self):
        """Listen for Redis events and route to connections"""
        try:
            async for message in self.pubsub.listen():
                if not self.running:
                    break
                
                if message['type'] == 'message':
                    channel = message['channel']
                    
                    try:
                        data = json.loads(message['data']) if isinstance(message['data'], str) else message['data']
                        await self.route_event(channel, data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in channel {channel}: {message['data']}")
                    except Exception as e:
                        logger.error(f"Error routing event from {channel}: {e}")
                        
        except Exception as e:
            logger.error(f"Event listener error: {e}")
    
    async def route_event(self, channel: str, data: dict):
        """Route event to appropriate connections"""
        # Map Redis EventChannel to WebSocket EventType
        event_mapping = {
            EventChannel.AGENT_CREATED.value: EventType.AGENT_CREATED,
            EventChannel.AGENT_STATUS.value: EventType.AGENT_STATUS,
            EventChannel.WORKFLOW_STARTED.value: EventType.WORKFLOW_STARTED,
            EventChannel.WORKFLOW_COMPLETED.value: EventType.WORKFLOW_COMPLETED,
            EventChannel.TASK_CREATED.value: EventType.TASK_STARTED,
            EventChannel.TASK_COMPLETED.value: EventType.TASK_COMPLETED,
            EventChannel.TASK_FAILED.value: EventType.TASK_FAILED,
            EventChannel.SYSTEM_ALERT.value: EventType.SYSTEM_ALERT
        }
        
        # Get event type
        event_type = event_mapping.get(channel)
        if not event_type:
            # Check if it's a room message
            if channel.startswith("room:"):
                await self.handle_room_broadcast(channel, data)
            return
        
        # Route to subscribed connections
        message = {
            "type": MessageType.EVENT.value,
            "event": event_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for conn_id, conn in self.connection_manager.connections.items():
            if conn["authenticated"] and event_type.value in conn["subscriptions"]:
                await self.connection_manager.send_to_connection(conn_id, message)
    
    async def handle_room_broadcast(self, channel: str, data: dict):
        """Handle room broadcast from another server"""
        if data.get("server_id") == SERVER_ID:
            return  # Ignore our own broadcasts
        
        room_id = channel.replace("room:", "")
        message = data.get("message", {})
        exclude = set(data.get("exclude", []))
        
        # Send to local connections in this room
        for conn_id, conn in self.connection_manager.connections.items():
            if room_id in conn["rooms"] and conn_id not in exclude:
                await self.connection_manager.send_to_connection(conn_id, message)


class HeartbeatManager:
    """Manages connection heartbeats"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.running = False
    
    async def start(self):
        """Start heartbeat monitoring"""
        self.running = True
        asyncio.create_task(self.heartbeat_loop())
        logger.info("Heartbeat manager started")
    
    async def stop(self):
        """Stop heartbeat monitoring"""
        self.running = False
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats and check connection health"""
        while self.running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            
            now = time.time()
            disconnected = []
            
            for conn_id, conn in self.connection_manager.connections.items():
                # Check for timeout
                if now - conn["last_activity"] > CONNECTION_TIMEOUT:
                    disconnected.append(conn_id)
                    continue
                
                # Send ping
                try:
                    await conn["websocket"].send_json({"type": MessageType.PING.value})
                except:
                    disconnected.append(conn_id)
            
            # Clean up dead connections
            for conn_id in disconnected:
                await self.connection_manager.disconnect(conn_id)


class WebSocketService:
    """Main WebSocket Service"""
    
    def __init__(self):
        self.app = FastAPI(title="WebSocket Service (Integrated)", version="2.0.0")
        
        # Initialize components
        self.cache = get_cache()
        self.connection_manager = ConnectionManager(self.cache.redis_client)
        self.auth = WebSocketAuth()
        self.event_router = EventRouter(self.connection_manager)
        self.heartbeat_manager = HeartbeatManager(self.connection_manager)
        
        logger.info(f"✅ WebSocket Service initialized with server ID: {SERVER_ID}")
        
        self._setup_middleware()
        self._setup_routes()
    
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
            return {
                "service": "websocket",
                "status": "healthy",
                "server_id": SERVER_ID,
                "connections": len(self.connection_manager.connections),
                "authenticated": sum(1 for c in self.connection_manager.connections.values() if c["authenticated"]),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(None)):
            """Main WebSocket endpoint"""
            connection_id = str(uuid.uuid4())
            
            # Accept connection
            await self.connection_manager.connect(websocket, connection_id)
            
            # Auto-authenticate if token provided
            if token:
                user_data = await self.auth.authenticate_connection(token)
                if user_data:
                    await self.connection_manager.authenticate(connection_id, user_data)
                    await websocket.send_json({
                        "type": MessageType.AUTHENTICATED.value,
                        "user": {
                            "id": user_data["user_id"],
                            "email": user_data["email"],
                            "roles": user_data["roles"]
                        }
                    })
            
            try:
                # Message handling loop
                while True:
                    # Receive message
                    data = await websocket.receive_json()
                    
                    # Update activity
                    if connection_id in self.connection_manager.connections:
                        self.connection_manager.connections[connection_id]["last_activity"] = time.time()
                    
                    # Handle message
                    await self.handle_message(connection_id, data)
                    
            except WebSocketDisconnect:
                await self.connection_manager.disconnect(connection_id)
            except Exception as e:
                logger.error(f"WebSocket error for {connection_id}: {e}")
                await self.connection_manager.disconnect(connection_id)
    
    async def handle_message(self, connection_id: str, data: dict):
        """Handle incoming WebSocket message"""
        message_type = data.get("type")
        conn = self.connection_manager.connections.get(connection_id)
        
        if not conn:
            return
        
        # Handle authentication
        if message_type == MessageType.AUTHENTICATE.value:
            token = data.get("token")
            if token:
                user_data = await self.auth.authenticate_connection(token)
                if user_data:
                    await self.connection_manager.authenticate(connection_id, user_data)
                    await self.connection_manager.send_to_connection(connection_id, {
                        "type": MessageType.AUTHENTICATED.value,
                        "user": {
                            "id": user_data["user_id"],
                            "email": user_data["email"],
                            "roles": user_data["roles"]
                        }
                    })
                else:
                    await self.connection_manager.send_to_connection(connection_id, {
                        "type": MessageType.ERROR.value,
                        "error": "Authentication failed"
                    })
            return
        
        # Check authentication for other operations
        if not conn["authenticated"]:
            await self.connection_manager.send_to_connection(connection_id, {
                "type": MessageType.ERROR.value,
                "error": "Not authenticated"
            })
            return
        
        # Handle different message types
        if message_type == MessageType.SUBSCRIBE.value:
            events = data.get("events", [])
            conn["subscriptions"].update(events)
            logger.info(f"Connection {connection_id} subscribed to: {events}")
            
        elif message_type == MessageType.UNSUBSCRIBE.value:
            events = data.get("events", [])
            conn["subscriptions"].difference_update(events)
            logger.info(f"Connection {connection_id} unsubscribed from: {events}")
            
        elif message_type == MessageType.JOIN_ROOM.value:
            room_id = data.get("room_id")
            if room_id:
                await self.connection_manager.join_room(connection_id, room_id)
                await self.connection_manager.send_to_connection(connection_id, {
                    "type": MessageType.ROOM_JOINED.value,
                    "room_id": room_id
                })
                
                # Notify room members
                await self.connection_manager.broadcast_to_room(room_id, {
                    "type": MessageType.MEMBER_JOINED.value,
                    "room_id": room_id,
                    "user_id": conn["user_id"]
                }, exclude={connection_id})
                
        elif message_type == MessageType.LEAVE_ROOM.value:
            room_id = data.get("room_id")
            if room_id:
                await self.connection_manager.leave_room(connection_id, room_id)
                await self.connection_manager.send_to_connection(connection_id, {
                    "type": MessageType.ROOM_LEFT.value,
                    "room_id": room_id
                })
                
                # Notify room members
                await self.connection_manager.broadcast_to_room(room_id, {
                    "type": MessageType.MEMBER_LEFT.value,
                    "room_id": room_id,
                    "user_id": conn["user_id"]
                }, exclude={connection_id})
                
        elif message_type == MessageType.SEND_MESSAGE.value:
            # Check permission
            user_roles = conn.get("user_data", {}).get("roles", [])
            if not self.auth.check_permission(user_roles, "broadcast"):
                await self.connection_manager.send_to_connection(connection_id, {
                    "type": MessageType.ERROR.value,
                    "error": "Permission denied"
                })
                return
            
            # Broadcast message
            target = data.get("target", "all")
            message = data.get("message", {})
            
            if target == "all":
                await self.connection_manager.broadcast_to_all(message, exclude={connection_id})
            elif target.startswith("room:"):
                room_id = target.replace("room:", "")
                await self.connection_manager.broadcast_to_room(room_id, message, exclude={connection_id})
            elif target.startswith("user:"):
                user_id = target.replace("user:", "")
                await self.connection_manager.broadcast_to_user(user_id, message)
                
        elif message_type == MessageType.PING.value:
            await self.connection_manager.send_to_connection(connection_id, {
                "type": MessageType.PONG.value
            })
    
    async def startup(self):
        """Startup tasks"""
        logger.info("WebSocket Service (Integrated) starting up...")
        
        # Start event router
        await self.event_router.start()
        
        # Start heartbeat manager
        await self.heartbeat_manager.start()
        
        logger.info("WebSocket Service ready")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("WebSocket Service shutting down...")
        
        # Stop components
        await self.heartbeat_manager.stop()
        await self.event_router.stop()
        
        # Close all connections
        for conn_id in list(self.connection_manager.connections.keys()):
            await self.connection_manager.disconnect(conn_id)


def main():
    """Main entry point"""
    import uvicorn
    
    service = WebSocketService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("WEBSOCKET_PORT", 8107))
    logger.info(f"Starting WebSocket Service (Integrated) on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()