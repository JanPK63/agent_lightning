# Solution Architecture: WebSocket Service Integration

**Document ID:** SA-008  
**Date:** 2025-09-06  
**Status:** For Review  
**Priority:** High  
**Author:** System Architect  
**Dependencies:** SA-001 (Database), SA-002 (Redis), SA-007 (Auth)  

## Executive Summary

This solution architecture details the integration of the WebSocket service with the shared Redis event bus and authentication system. The WebSocket service provides real-time bidirectional communication between clients and the Lightning system, enabling live updates, streaming responses, and collaborative features.

## Problem Statement

### Current Issues

- **No Event Broadcasting:** Events not propagated to connected clients
- **Isolated Connections:** WebSocket sessions not integrated with auth
- **No Persistence:** Connection state lost on service restart
- **Limited Scalability:** Cannot distribute connections across instances
- **No Room Management:** Cannot group connections for targeted messaging

### Business Impact

- Poor real-time user experience
- Cannot build collaborative features
- No live monitoring capabilities
- Limited system responsiveness
- Cannot scale WebSocket connections

## Proposed Solution

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    WebSocket Service                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Connection Manager                        │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │  Session │  │   Room   │  │  Event   │             │   │
│  │  │  Handler │  │  Manager │  │  Router  │             │   │
│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘             │   │
│  │        │             │              │                   │   │
│  └────────┼─────────────┼──────────────┼───────────────────┘   │
│           │             │              │                        │
│  ┌────────▼─────────────▼──────────────▼────────────────────┐   │
│  │              Redis Pub/Sub & Event Bus                   │   │
│  │                                                          │   │
│  │  - Event subscriptions                                   │   │
│  │  - Message broadcasting                                  │   │
│  │  - Connection registry                                   │   │
│  │  - Room memberships                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└───────────────────────────────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Other Services │
                    │                 │
                    │ • Agent Designer│
                    │ • Workflow Eng. │
                    │ • AI Model     │
                    │ • Auth Service │
                    └─────────────────┘
```

### Redis Schema for WebSocket

```python
# Connection Registry
connections:{server_id} = {
    "{connection_id}": {
        "user_id": "uuid",
        "session_id": "uuid",
        "connected_at": "timestamp",
        "rooms": ["room1", "room2"],
        "metadata": {}
    }
}

# Room Memberships
room:{room_id}:members = ["connection_id1", "connection_id2"]

# User Sessions (track all connections for a user)
user:{user_id}:connections = ["connection_id1", "connection_id2"]

# Event Subscriptions
subscriptions:{connection_id} = ["event_type1", "event_type2"]
```

### Key Components

#### 1. Connection Manager

```python
class ConnectionManager:
    def __init__(self, redis_client, server_id: str):
        self.redis = redis_client
        self.server_id = server_id
        self.connections = {}  # Local connection tracking
        self.connection_key = f"connections:{server_id}"
    
    async def register_connection(self, connection_id: str, 
                                 user_id: str, websocket):
        """Register new WebSocket connection"""
        # Store locally
        self.connections[connection_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "connected_at": datetime.utcnow().isoformat(),
            "rooms": set(),
            "subscriptions": set()
        }
        
        # Store in Redis
        self.redis.hset(self.connection_key, connection_id, json.dumps({
            "user_id": user_id,
            "session_id": str(uuid.uuid4()),
            "connected_at": datetime.utcnow().isoformat(),
            "server_id": self.server_id,
            "rooms": [],
            "metadata": {}
        }))
        
        # Add to user's connections
        user_key = f"user:{user_id}:connections"
        self.redis.sadd(user_key, connection_id)
        
    async def unregister_connection(self, connection_id: str):
        """Remove connection on disconnect"""
        if connection_id in self.connections:
            conn = self.connections[connection_id]
            
            # Leave all rooms
            for room in conn["rooms"]:
                await self.leave_room(connection_id, room)
            
            # Remove from user's connections
            user_key = f"user:{conn['user_id']}:connections"
            self.redis.srem(user_key, connection_id)
            
            # Remove from Redis
            self.redis.hdel(self.connection_key, connection_id)
            
            # Remove locally
            del self.connections[connection_id]
```

#### 2. Room Manager

```python
class RoomManager:
    def __init__(self, redis_client, connection_manager):
        self.redis = redis_client
        self.connection_manager = connection_manager
    
    async def join_room(self, connection_id: str, room_id: str):
        """Add connection to room"""
        # Update local tracking
        if connection_id in self.connection_manager.connections:
            self.connection_manager.connections[connection_id]["rooms"].add(room_id)
        
        # Update Redis
        room_key = f"room:{room_id}:members"
        self.redis.sadd(room_key, connection_id)
        
        # Notify room members
        await self.broadcast_to_room(room_id, {
            "type": "member_joined",
            "room_id": room_id,
            "connection_id": connection_id
        }, exclude=[connection_id])
    
    async def broadcast_to_room(self, room_id: str, message: dict, 
                               exclude: List[str] = None):
        """Send message to all room members"""
        room_key = f"room:{room_id}:members"
        members = self.redis.smembers(room_key)
        
        for member_id in members:
            if exclude and member_id in exclude:
                continue
            
            # Check if connection is on this server
            if member_id in self.connection_manager.connections:
                conn = self.connection_manager.connections[member_id]
                await conn["websocket"].send_json(message)
            else:
                # Publish to Redis for other servers
                self.redis.publish(f"ws:message:{member_id}", json.dumps(message))
```

#### 3. Event Router

```python
class EventRouter:
    def __init__(self, redis_client, connection_manager):
        self.redis = redis_client
        self.connection_manager = connection_manager
        self.event_handlers = {}
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        """Subscribe to Redis event channels"""
        pubsub = self.redis.pubsub()
        
        # Subscribe to all event channels
        for channel in EventChannel:
            pubsub.subscribe(channel.value)
        
        # Start listening in background
        asyncio.create_task(self.event_listener(pubsub))
    
    async def event_listener(self, pubsub):
        """Listen for Redis events and route to connections"""
        async for message in pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel']
                data = json.loads(message['data'])
                
                # Route to subscribed connections
                await self.route_event(channel, data)
    
    async def route_event(self, channel: str, data: dict):
        """Route event to subscribed connections"""
        for conn_id, conn in self.connection_manager.connections.items():
            if channel in conn["subscriptions"]:
                await conn["websocket"].send_json({
                    "type": "event",
                    "channel": channel,
                    "data": data
                })
    
    async def subscribe_connection(self, connection_id: str, 
                                  event_types: List[str]):
        """Subscribe connection to event types"""
        if connection_id in self.connection_manager.connections:
            conn = self.connection_manager.connections[connection_id]
            conn["subscriptions"].update(event_types)
            
            # Store in Redis
            sub_key = f"subscriptions:{connection_id}"
            self.redis.sadd(sub_key, *event_types)
```

#### 4. Authentication Integration

```python
class WebSocketAuth:
    def __init__(self, jwt_manager):
        self.jwt_manager = jwt_manager
    
    async def authenticate_connection(self, token: str) -> Optional[dict]:
        """Authenticate WebSocket connection"""
        try:
            # Verify JWT token
            payload = self.jwt_manager.verify_token(token)
            
            return {
                "user_id": payload.get("sub"),
                "email": payload.get("email"),
                "roles": payload.get("roles", [])
            }
        except Exception as e:
            logger.error(f"WebSocket auth failed: {e}")
            return None
    
    def check_permission(self, user_roles: List[str], 
                        action: str) -> bool:
        """Check if user can perform action"""
        permission_map = {
            "subscribe_all": ["admin"],
            "create_room": ["admin", "developer"],
            "broadcast": ["admin", "developer", "operator"]
        }
        
        required_roles = permission_map.get(action, [])
        return any(role in required_roles for role in user_roles)
```

### Implementation Features

#### 1. Message Types

```python
class MessageType(str, Enum):
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
```

#### 2. Scaling with Redis Pub/Sub

```python
class ClusterManager:
    def __init__(self, redis_client, server_id: str):
        self.redis = redis_client
        self.server_id = server_id
        self.setup_cluster_communication()
    
    def setup_cluster_communication(self):
        """Setup inter-server communication"""
        pubsub = self.redis.pubsub()
        
        # Subscribe to server-specific channel
        pubsub.subscribe(f"server:{self.server_id}")
        
        # Subscribe to broadcast channel
        pubsub.subscribe("broadcast:all")
        
        asyncio.create_task(self.cluster_listener(pubsub))
    
    async def broadcast_to_cluster(self, message: dict):
        """Broadcast message to all servers"""
        self.redis.publish("broadcast:all", json.dumps(message))
    
    async def send_to_server(self, server_id: str, message: dict):
        """Send message to specific server"""
        self.redis.publish(f"server:{server_id}", json.dumps(message))
```

#### 3. Heartbeat & Connection Health

```python
class HeartbeatManager:
    def __init__(self, connection_manager, timeout: int = 60):
        self.connection_manager = connection_manager
        self.timeout = timeout
        self.last_ping = {}
    
    async def start_heartbeat(self):
        """Start heartbeat monitoring"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            await self.check_connections()
    
    async def check_connections(self):
        """Check connection health"""
        now = time.time()
        
        for conn_id in list(self.connection_manager.connections.keys()):
            conn = self.connection_manager.connections.get(conn_id)
            if not conn:
                continue
            
            # Send ping
            try:
                await conn["websocket"].send_json({"type": "ping"})
                self.last_ping[conn_id] = now
            except:
                # Connection dead, remove it
                await self.connection_manager.unregister_connection(conn_id)
            
            # Check for timeout
            last = self.last_ping.get(conn_id, now)
            if now - last > self.timeout:
                await self.connection_manager.unregister_connection(conn_id)
```

## Success Metrics

### Technical Metrics

- ✅ < 10ms message latency within cluster
- ✅ Support 10,000+ concurrent connections
- ✅ Zero message loss during failover
- ✅ Automatic reconnection handling
- ✅ Real-time event propagation

### Business Metrics

- ✅ Enhanced user experience with live updates
- ✅ Enabled collaborative features
- ✅ Real-time monitoring capabilities
- ✅ Reduced server load from polling
- ✅ Improved system responsiveness

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Connection overload | Medium | High | Connection limits, rate limiting |
| Message flooding | Low | Medium | Rate limiting, message throttling |
| Memory leaks | Low | High | Connection cleanup, monitoring |
| Network partitions | Low | High | Heartbeat, automatic reconnection |

## Testing Strategy

### Unit Tests

- Connection management
- Room operations
- Event routing
- Message handling

### Integration Tests

- End-to-end messaging
- Multi-server communication
- Authentication flow
- Event propagation

### Load Tests

- 10,000 concurrent connections
- Message throughput
- Room scaling
- Cluster failover

## Migration Plan

1. Deploy integrated WebSocket service
2. Configure Redis pub/sub channels
3. Integrate with Auth service
4. Connect to event bus
5. Test real-time features
6. Enable client connections
7. Monitor and optimize

## Approval

**Review Checklist:**
- [ ] Scalability design adequate
- [ ] Authentication integrated
- [ ] Event routing comprehensive
- [ ] Room management complete
- [ ] Testing coverage sufficient

**Sign-off Required From:**
- [ ] Frontend Team
- [ ] Backend Team
- [ ] Infrastructure Team
- [ ] Security Team

---

**Next Steps After Approval:**
1. Implement connection manager
2. Setup Redis pub/sub
3. Integrate authentication
4. Deploy WebSocket service
5. Test real-time features

**Related Documents:**
- SA-001: Database Persistence Layer
- SA-002: Redis Cache & Event Bus  
- SA-007: Authentication & Authorization Service
- Previous: All other SA documents