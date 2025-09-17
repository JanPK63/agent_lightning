"""
Event-driven architecture components for Agent Lightning
Provides event publishing, subscription, and handling
"""

import json
import uuid
import logging
import threading
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from shared.cache import CacheManager, get_cache

logger = logging.getLogger(__name__)

class EventChannel(Enum):
    """Event channel definitions"""
    # Agent events
    AGENT_CREATED = "agent:created"
    AGENT_UPDATED = "agent:updated"
    AGENT_DELETED = "agent:deleted"
    AGENT_STATUS = "agent:status"
    
    # Task events
    TASK_CREATED = "task:created"
    TASK_STARTED = "task:started"
    TASK_COMPLETED = "task:completed"
    TASK_FAILED = "task:failed"
    TASK_PROGRESS = "task:progress"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow:started"
    WORKFLOW_STEP = "workflow:step"
    WORKFLOW_COMPLETED = "workflow:completed"
    WORKFLOW_FAILED = "workflow:failed"

    # Knowledge events
    KNOWLEDGE_ADDED = "knowledge:added"
    KNOWLEDGE_UPDATED = "knowledge:updated"
    KNOWLEDGE_DELETED = "knowledge:deleted"
    
    # System events
    SYSTEM_HEALTH = "system:health"
    SYSTEM_ALERT = "system:alert"
    SYSTEM_METRICS = "system:metrics"
    
    # WebSocket events
    WS_CONNECT = "ws:connect"
    WS_DISCONNECT = "ws:disconnect"
    WS_MESSAGE = "ws:message"

@dataclass
class Event:
    """Event data structure"""
    event_id: str
    timestamp: str
    channel: str
    service: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        """Convert event to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        """Create event from dictionary"""
        return cls(**data)

class EventPublisher:
    """Publishes events to Redis pub/sub channels"""
    
    def __init__(self, service_name: str, cache: CacheManager = None):
        """Initialize event publisher
        
        Args:
            service_name: Name of the publishing service
            cache: Cache manager instance
        """
        self.service_name = service_name
        self.cache = cache or get_cache()
    
    def emit(self, channel: EventChannel, data: dict, 
             metadata: dict = None) -> bool:
        """Emit event to channel
        
        Args:
            channel: Event channel
            data: Event data
            metadata: Optional metadata
            
        Returns:
            True if event was published
        """
        try:
            event = Event(
                event_id=f"evt_{uuid.uuid4().hex}",
                timestamp=datetime.utcnow().isoformat(),
                channel=channel.value if isinstance(channel, EventChannel) else channel,
                service=self.service_name,
                data=data,
                metadata=metadata or {}
            )
            
            # Publish event
            subscribers = self.cache.publish(event.channel, event.to_dict())
            
            # Store in event history
            history_key = f"events:{event.channel}:history"
            self.cache.redis_client.lpush(history_key, json.dumps(event.to_dict()))
            self.cache.redis_client.ltrim(history_key, 0, 99)  # Keep last 100 events
            
            logger.debug(f"Published event to {event.channel} ({subscribers} subscribers)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to emit event to {channel}: {e}")
            return False
    
    def emit_batch(self, events: List[tuple]) -> int:
        """Emit multiple events
        
        Args:
            events: List of (channel, data, metadata) tuples
            
        Returns:
            Number of events published
        """
        count = 0
        for channel, data, metadata in events:
            if self.emit(channel, data, metadata):
                count += 1
        return count

class EventSubscriber:
    """Subscribes to and handles events from Redis pub/sub channels"""
    
    def __init__(self, service_name: str, cache: CacheManager = None):
        """Initialize event subscriber
        
        Args:
            service_name: Name of the subscribing service
            cache: Cache manager instance
        """
        self.service_name = service_name
        self.cache = cache or get_cache()
        self.handlers: Dict[str, List[Callable]] = {}
        self.listener_thread = None
        self.running = False
    
    def register_handler(self, channel: EventChannel, 
                        handler: Callable[[Event], None]):
        """Register event handler for channel
        
        Args:
            channel: Event channel
            handler: Handler function
        """
        channel_name = channel.value if isinstance(channel, EventChannel) else channel
        
        if channel_name not in self.handlers:
            self.handlers[channel_name] = []
        
        self.handlers[channel_name].append(handler)
        logger.info(f"Registered handler for {channel_name}")
    
    def unregister_handler(self, channel: EventChannel, 
                          handler: Callable[[Event], None]):
        """Unregister event handler
        
        Args:
            channel: Event channel
            handler: Handler function to remove
        """
        channel_name = channel.value if isinstance(channel, EventChannel) else channel
        
        if channel_name in self.handlers:
            try:
                self.handlers[channel_name].remove(handler)
                logger.info(f"Unregistered handler for {channel_name}")
            except ValueError:
                pass
    
    def start(self, channels: List[EventChannel] = None):
        """Start listening for events
        
        Args:
            channels: List of channels to subscribe (None for all registered)
        """
        if self.running:
            logger.warning("Event subscriber already running")
            return
        
        # Subscribe to channels
        if channels:
            channel_names = [c.value if isinstance(c, EventChannel) else c 
                           for c in channels]
        else:
            channel_names = list(self.handlers.keys())
        
        if not channel_names:
            logger.warning("No channels to subscribe to")
            return
        
        self.cache.subscribe(channel_names)
        self.running = True
        
        # Start listener thread
        self.listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self.listener_thread.start()
        logger.info(f"Started event subscriber for channels: {channel_names}")
    
    def stop(self):
        """Stop listening for events"""
        if not self.running:
            return
        
        self.running = False
        self.cache.unsubscribe()
        
        if self.listener_thread:
            self.listener_thread.join(timeout=5)
        
        logger.info("Stopped event subscriber")
    
    def _listen_loop(self):
        """Main event listening loop"""
        while self.running:
            try:
                message = self.cache.listen(timeout=1)
                
                if message and message['type'] == 'message':
                    self._handle_message(message)
                    
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
                # Continue listening unless stopped
                if self.running:
                    import time
                    time.sleep(1)
    
    def _handle_message(self, message: dict):
        """Handle received message
        
        Args:
            message: Redis pub/sub message
        """
        try:
            channel = message['channel'].decode() if isinstance(message['channel'], bytes) else message['channel']
            event_data = message['data']
            
            # Parse event
            event = Event.from_dict(event_data)
            
            # Call registered handlers
            if channel in self.handlers:
                for handler in self.handlers[channel]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler for {channel}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")

class EventBus:
    """Central event bus for the application"""
    
    def __init__(self, service_name: str, cache: CacheManager = None):
        """Initialize event bus
        
        Args:
            service_name: Name of the service
            cache: Cache manager instance
        """
        self.service_name = service_name
        self.cache = cache or get_cache()
        self.publisher = EventPublisher(service_name, cache)
        self.subscriber = EventSubscriber(service_name, cache)
    
    def emit(self, channel: EventChannel, data: dict, 
             metadata: dict = None) -> bool:
        """Emit event
        
        Args:
            channel: Event channel
            data: Event data
            metadata: Optional metadata
            
        Returns:
            True if event was published
        """
        return self.publisher.emit(channel, data, metadata)
    
    def on(self, channel: EventChannel, handler: Callable[[Event], None]):
        """Register event handler
        
        Args:
            channel: Event channel
            handler: Handler function
        """
        self.subscriber.register_handler(channel, handler)
    
    def off(self, channel: EventChannel, handler: Callable[[Event], None]):
        """Unregister event handler
        
        Args:
            channel: Event channel
            handler: Handler function
        """
        self.subscriber.unregister_handler(channel, handler)
    
    def start(self):
        """Start event bus"""
        self.subscriber.start()
    
    def stop(self):
        """Stop event bus"""
        self.subscriber.stop()

# ==================== Event Handlers ====================

def log_event_handler(event: Event):
    """Default event logging handler"""
    logger.info(f"Event received: {event.channel} from {event.service}")
    logger.debug(f"Event data: {event.data}")

def metrics_event_handler(event: Event):
    """Handler for metrics events"""
    if event.channel == EventChannel.SYSTEM_METRICS.value:
        # Store metrics in cache for dashboard
        cache = get_cache()
        for metric_name, value in event.data.items():
            key = f"metrics:{event.service}:{metric_name}"
            cache.set(key, value, ttl=300)  # 5 minutes

def alert_event_handler(event: Event):
    """Handler for system alerts"""
    if event.channel == EventChannel.SYSTEM_ALERT.value:
        severity = event.data.get('severity', 'info')
        message = event.data.get('message', 'Unknown alert')
        
        if severity in ['critical', 'error']:
            logger.error(f"ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")

# ==================== Decorators ====================

def emit_event(channel: EventChannel, data_func: Callable = None):
    """Decorator to emit events after function execution
    
    Args:
        channel: Event channel
        data_func: Function to generate event data from result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Generate event data
            if data_func:
                event_data = data_func(result)
            else:
                event_data = {'result': str(result)}
            
            # Emit event
            publisher = EventPublisher(func.__module__)
            publisher.emit(channel, event_data)
            
            return result
        return wrapper
    return decorator

def handle_event(channel: EventChannel):
    """Decorator to mark function as event handler
    
    Args:
        channel: Event channel to handle
    """
    def decorator(func):
        func._event_channel = channel
        func._is_event_handler = True
        return func
    return decorator

# ==================== Utility Functions ====================

def get_event_history(channel: EventChannel, limit: int = 10) -> List[Event]:
    """Get recent events from channel
    
    Args:
        channel: Event channel
        limit: Maximum number of events
        
    Returns:
        List of recent events
    """
    cache = get_cache()
    channel_name = channel.value if isinstance(channel, EventChannel) else channel
    history_key = f"events:{channel_name}:history"
    
    try:
        raw_events = cache.redis_client.lrange(history_key, 0, limit - 1)
        events = []
        
        for raw_event in raw_events:
            event_data = json.loads(raw_event)
            events.append(Event.from_dict(event_data))
        
        return events
        
    except Exception as e:
        logger.error(f"Failed to get event history: {e}")
        return []

def clear_event_history(channel: EventChannel = None):
    """Clear event history
    
    Args:
        channel: Event channel (None for all)
    """
    cache = get_cache()
    
    if channel:
        channel_name = channel.value if isinstance(channel, EventChannel) else channel
        cache.delete(f"events:{channel_name}:history")
    else:
        cache.delete_pattern("events:*:history")