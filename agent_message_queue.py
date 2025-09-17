#!/usr/bin/env python3
"""
Inter-Agent Message Queue System for Agent Lightning
Provides reliable, asynchronous message passing between agents
"""

import os
import sys
import json
import asyncio
import pickle
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import logging
import hashlib
from asyncio import PriorityQueue, Queue
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_communication_protocol import (
    AgentMessage, 
    Performative, 
    MessagePriority,
    ConversationState
)
from agent_collaboration import CollaborativeTask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueueType(Enum):
    """Types of message queues"""
    FIFO = "fifo"                   # First In First Out
    LIFO = "lifo"                   # Last In First Out (Stack)
    PRIORITY = "priority"           # Priority-based
    TOPIC = "topic"                 # Topic-based pub-sub
    BROADCAST = "broadcast"         # Broadcast to all
    ROUND_ROBIN = "round_robin"     # Round-robin distribution
    WORK_STEALING = "work_stealing" # Work-stealing queue


class DeliveryMode(Enum):
    """Message delivery guarantees"""
    AT_MOST_ONCE = "at_most_once"     # Fire and forget
    AT_LEAST_ONCE = "at_least_once"   # Retry until acknowledged
    EXACTLY_ONCE = "exactly_once"      # Deduplication + acknowledgment


class MessageStatus(Enum):
    """Status of a message in the queue"""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueuedMessage:
    """Message wrapper for queue management"""
    message: AgentMessage
    queue_time: datetime = field(default_factory=datetime.now)
    delivery_attempts: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    acknowledgment_id: str = field(default_factory=lambda: str(hashlib.md5(str(datetime.now()).encode()).hexdigest()))
    expires_at: Optional[datetime] = None
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def should_retry(self) -> bool:
        """Check if message should be retried"""
        return (
            self.delivery_attempts < self.max_retries and
            self.status in [MessageStatus.FAILED, MessageStatus.PENDING] and
            not self.is_expired()
        )
    
    def to_priority_tuple(self) -> Tuple[int, datetime, 'QueuedMessage']:
        """Convert to tuple for priority queue"""
        # Lower priority value = higher priority
        priority_value = self.message.priority.value
        return (priority_value, self.queue_time, self)


class MessageQueue:
    """Base class for message queues"""
    
    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.max_size = max_size
        self.message_count = 0
        self.subscribers: Set[str] = set()
        self.metrics = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'messages_expired': 0
        }
    
    async def enqueue(self, message: QueuedMessage) -> bool:
        """Add message to queue"""
        raise NotImplementedError
    
    async def dequeue(self, agent_id: str) -> Optional[QueuedMessage]:
        """Get message from queue"""
        raise NotImplementedError
    
    def subscribe(self, agent_id: str):
        """Subscribe agent to queue"""
        self.subscribers.add(agent_id)
    
    def unsubscribe(self, agent_id: str):
        """Unsubscribe agent from queue"""
        self.subscribers.discard(agent_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        return {
            'name': self.name,
            'subscribers': len(self.subscribers),
            'message_count': self.message_count,
            **self.metrics
        }


class FIFOMessageQueue(MessageQueue):
    """First In First Out message queue"""
    
    def __init__(self, name: str, max_size: int = 1000):
        super().__init__(name, max_size)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.processing: Dict[str, QueuedMessage] = {}
    
    async def enqueue(self, message: QueuedMessage) -> bool:
        """Add message to FIFO queue"""
        try:
            if self.queue.full():
                logger.warning(f"Queue {self.name} is full")
                return False
            
            await self.queue.put(message)
            self.message_count += 1
            self.metrics['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error enqueueing message: {e}")
            return False
    
    async def dequeue(self, agent_id: str) -> Optional[QueuedMessage]:
        """Get message from FIFO queue"""
        try:
            if agent_id not in self.subscribers:
                return None
            
            message = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            
            if message.is_expired():
                self.metrics['messages_expired'] += 1
                message.status = MessageStatus.EXPIRED
                return None
            
            message.status = MessageStatus.PROCESSING
            message.delivery_attempts += 1
            self.processing[message.acknowledgment_id] = message
            
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error dequeueing message: {e}")
            return None
    
    async def acknowledge(self, ack_id: str) -> bool:
        """Acknowledge message delivery"""
        if ack_id in self.processing:
            message = self.processing.pop(ack_id)
            message.status = MessageStatus.ACKNOWLEDGED
            self.metrics['messages_delivered'] += 1
            self.message_count -= 1
            return True
        return False
    
    async def negative_acknowledge(self, ack_id: str) -> bool:
        """Negative acknowledge - requeue message"""
        if ack_id in self.processing:
            message = self.processing.pop(ack_id)
            
            if message.should_retry():
                message.status = MessageStatus.PENDING
                await self.enqueue(message)
                return True
            else:
                message.status = MessageStatus.FAILED
                self.metrics['messages_failed'] += 1
                self.message_count -= 1
        return False


class PriorityMessageQueue(MessageQueue):
    """Priority-based message queue"""
    
    def __init__(self, name: str, max_size: int = 1000):
        super().__init__(name, max_size)
        self.queue: PriorityQueue = PriorityQueue(maxsize=max_size)
        self.processing: Dict[str, QueuedMessage] = {}
    
    async def enqueue(self, message: QueuedMessage) -> bool:
        """Add message to priority queue"""
        try:
            if self.queue.full():
                logger.warning(f"Priority queue {self.name} is full")
                return False
            
            await self.queue.put(message.to_priority_tuple())
            self.message_count += 1
            self.metrics['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error enqueueing to priority queue: {e}")
            return False
    
    async def dequeue(self, agent_id: str) -> Optional[QueuedMessage]:
        """Get highest priority message"""
        try:
            if agent_id not in self.subscribers:
                return None
            
            priority, queue_time, message = await asyncio.wait_for(
                self.queue.get(), timeout=0.1
            )
            
            if message.is_expired():
                self.metrics['messages_expired'] += 1
                message.status = MessageStatus.EXPIRED
                return None
            
            message.status = MessageStatus.PROCESSING
            message.delivery_attempts += 1
            self.processing[message.acknowledgment_id] = message
            
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error dequeueing from priority queue: {e}")
            return None


class TopicMessageQueue(MessageQueue):
    """Topic-based publish-subscribe queue"""
    
    def __init__(self, name: str, max_size: int = 1000):
        super().__init__(name, max_size)
        self.topics: Dict[str, asyncio.Queue] = {}
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.processing: Dict[str, QueuedMessage] = {}
    
    def create_topic(self, topic: str):
        """Create a new topic"""
        if topic not in self.topics:
            self.topics[topic] = asyncio.Queue(maxsize=self.max_size)
            logger.info(f"Created topic: {topic}")
    
    def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe agent to specific topic"""
        self.create_topic(topic)
        self.topic_subscribers[topic].add(agent_id)
        self.subscribe(agent_id)
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """Unsubscribe agent from topic"""
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(agent_id)
    
    async def enqueue(self, message: QueuedMessage, topic: str = "default") -> bool:
        """Publish message to topic"""
        try:
            self.create_topic(topic)
            
            if self.topics[topic].full():
                logger.warning(f"Topic queue {topic} is full")
                return False
            
            # Clone message for each subscriber
            for subscriber in self.topic_subscribers[topic]:
                await self.topics[topic].put(message)
            
            self.message_count += 1
            self.metrics['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error publishing to topic {topic}: {e}")
            return False
    
    async def dequeue(self, agent_id: str, topic: str = "default") -> Optional[QueuedMessage]:
        """Get message from subscribed topic"""
        try:
            if topic not in self.topics or agent_id not in self.topic_subscribers[topic]:
                return None
            
            message = await asyncio.wait_for(
                self.topics[topic].get(), timeout=0.1
            )
            
            if message.is_expired():
                self.metrics['messages_expired'] += 1
                message.status = MessageStatus.EXPIRED
                return None
            
            message.status = MessageStatus.PROCESSING
            message.delivery_attempts += 1
            self.processing[message.acknowledgment_id] = message
            
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error dequeueing from topic {topic}: {e}")
            return None


class WorkStealingQueue(MessageQueue):
    """Work-stealing queue for load balancing"""
    
    def __init__(self, name: str, max_size: int = 1000):
        super().__init__(name, max_size)
        self.agent_queues: Dict[str, deque] = {}
        self.global_queue: deque = deque(maxlen=max_size)
        self.processing: Dict[str, QueuedMessage] = {}
        self.agent_loads: Dict[str, int] = defaultdict(int)
        self.steal_threshold = 2  # Steal if difference > threshold
    
    def subscribe(self, agent_id: str):
        """Subscribe agent and create personal queue"""
        super().subscribe(agent_id)
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = deque(maxlen=self.max_size // len(self.subscribers) if self.subscribers else 100)
    
    async def enqueue(self, message: QueuedMessage) -> bool:
        """Add message to least loaded agent's queue"""
        try:
            if not self.subscribers:
                # No subscribers, add to global queue
                if len(self.global_queue) >= self.max_size:
                    return False
                self.global_queue.append(message)
            else:
                # Initialize loads for new subscribers
                for subscriber in self.subscribers:
                    if subscriber not in self.agent_loads:
                        self.agent_loads[subscriber] = 0
                
                # Find least loaded agent
                if self.agent_loads:
                    min_agent = min(self.agent_loads.keys(), key=lambda a: self.agent_loads[a])
                    
                    if len(self.agent_queues[min_agent]) >= self.agent_queues[min_agent].maxlen:
                        # Try global queue
                        if len(self.global_queue) >= self.max_size:
                            return False
                        self.global_queue.append(message)
                    else:
                        self.agent_queues[min_agent].append(message)
                        self.agent_loads[min_agent] += 1
                else:
                    # Fallback to global queue
                    if len(self.global_queue) >= self.max_size:
                        return False
                    self.global_queue.append(message)
            
            self.message_count += 1
            self.metrics['messages_sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error in work-stealing enqueue: {e}")
            return False
    
    async def dequeue(self, agent_id: str) -> Optional[QueuedMessage]:
        """Get message, stealing from other queues if necessary"""
        try:
            if agent_id not in self.subscribers:
                return None
            
            message = None
            
            # Try own queue first
            if agent_id in self.agent_queues and self.agent_queues[agent_id]:
                message = self.agent_queues[agent_id].popleft()
                self.agent_loads[agent_id] -= 1
            
            # Try global queue
            elif self.global_queue:
                message = self.global_queue.popleft()
            
            # Try stealing from overloaded agents
            else:
                for other_agent, other_queue in self.agent_queues.items():
                    if other_agent != agent_id and len(other_queue) > self.steal_threshold:
                        # Steal from the back (oldest tasks)
                        message = other_queue.pop()
                        self.agent_loads[other_agent] -= 1
                        logger.debug(f"Agent {agent_id} stole work from {other_agent}")
                        break
            
            if message:
                if message.is_expired():
                    self.metrics['messages_expired'] += 1
                    message.status = MessageStatus.EXPIRED
                    return None
                
                message.status = MessageStatus.PROCESSING
                message.delivery_attempts += 1
                self.processing[message.acknowledgment_id] = message
                return message
            
            return None
            
        except Exception as e:
            logger.error(f"Error in work-stealing dequeue: {e}")
            return None


class MessageQueueManager:
    """Central manager for all message queues"""
    
    def __init__(self):
        self.queues: Dict[str, MessageQueue] = {}
        self.agent_subscriptions: Dict[str, List[str]] = defaultdict(list)
        self.dead_letter_queue = FIFOMessageQueue("dead_letter", max_size=5000)
        self.message_history: deque = deque(maxlen=10000)
        self.running = False
        self.background_tasks = []
    
    def create_queue(
        self,
        name: str,
        queue_type: QueueType = QueueType.FIFO,
        max_size: int = 1000
    ) -> MessageQueue:
        """Create a new message queue"""
        if name in self.queues:
            logger.warning(f"Queue {name} already exists")
            return self.queues[name]
        
        if queue_type == QueueType.FIFO:
            queue = FIFOMessageQueue(name, max_size)
        elif queue_type == QueueType.PRIORITY:
            queue = PriorityMessageQueue(name, max_size)
        elif queue_type == QueueType.TOPIC:
            queue = TopicMessageQueue(name, max_size)
        elif queue_type == QueueType.WORK_STEALING:
            queue = WorkStealingQueue(name, max_size)
        else:
            queue = FIFOMessageQueue(name, max_size)
        
        self.queues[name] = queue
        logger.info(f"Created {queue_type.value} queue: {name}")
        return queue
    
    def delete_queue(self, name: str) -> bool:
        """Delete a message queue"""
        if name in self.queues:
            del self.queues[name]
            # Clean up subscriptions
            for agent_id in list(self.agent_subscriptions.keys()):
                if name in self.agent_subscriptions[agent_id]:
                    self.agent_subscriptions[agent_id].remove(name)
            logger.info(f"Deleted queue: {name}")
            return True
        return False
    
    async def send_message(
        self,
        message: AgentMessage,
        queue_name: str = "default",
        delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE,
        ttl_seconds: Optional[int] = None,
        topic: Optional[str] = None
    ) -> bool:
        """Send a message to a queue"""
        
        # Create default queue if not exists
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        queue = self.queues[queue_name]
        
        # Create queued message
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        queued_msg = QueuedMessage(
            message=message,
            delivery_mode=delivery_mode,
            expires_at=expires_at
        )
        
        # Send to appropriate queue
        if isinstance(queue, TopicMessageQueue) and topic:
            success = await queue.enqueue(queued_msg, topic)
        else:
            success = await queue.enqueue(queued_msg)
        
        # Record in history
        self.message_history.append({
            'timestamp': datetime.now(),
            'queue': queue_name,
            'message_id': message.message_id,
            'sender': message.sender,
            'receiver': message.receiver,
            'success': success
        })
        
        return success
    
    async def receive_message(
        self,
        agent_id: str,
        queue_name: str = "default",
        topic: Optional[str] = None,
        timeout: float = 1.0
    ) -> Optional[AgentMessage]:
        """Receive a message from a queue"""
        
        if queue_name not in self.queues:
            return None
        
        queue = self.queues[queue_name]
        
        if agent_id not in queue.subscribers:
            return None
        
        # Receive from appropriate queue
        if isinstance(queue, TopicMessageQueue) and topic:
            queued_msg = await queue.dequeue(agent_id, topic)
        else:
            queued_msg = await queue.dequeue(agent_id)
        
        if queued_msg:
            return queued_msg.message
        
        return None
    
    def subscribe(self, agent_id: str, queue_name: str, topic: Optional[str] = None):
        """Subscribe agent to a queue"""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        queue = self.queues[queue_name]
        
        if isinstance(queue, TopicMessageQueue) and topic:
            queue.subscribe_to_topic(agent_id, topic)
        else:
            queue.subscribe(agent_id)
        
        if queue_name not in self.agent_subscriptions[agent_id]:
            self.agent_subscriptions[agent_id].append(queue_name)
        
        logger.info(f"Agent {agent_id} subscribed to queue {queue_name}" + 
                   (f" topic {topic}" if topic else ""))
    
    def unsubscribe(self, agent_id: str, queue_name: str, topic: Optional[str] = None):
        """Unsubscribe agent from a queue"""
        if queue_name in self.queues:
            queue = self.queues[queue_name]
            
            if isinstance(queue, TopicMessageQueue) and topic:
                queue.unsubscribe_from_topic(agent_id, topic)
            else:
                queue.unsubscribe(agent_id)
            
            if queue_name in self.agent_subscriptions[agent_id]:
                self.agent_subscriptions[agent_id].remove(queue_name)
    
    async def acknowledge_message(self, queue_name: str, ack_id: str) -> bool:
        """Acknowledge message delivery"""
        if queue_name in self.queues:
            queue = self.queues[queue_name]
            if hasattr(queue, 'acknowledge'):
                return await queue.acknowledge(ack_id)
        return False
    
    async def negative_acknowledge_message(self, queue_name: str, ack_id: str) -> bool:
        """Negative acknowledge - retry or dead letter"""
        if queue_name in self.queues:
            queue = self.queues[queue_name]
            if hasattr(queue, 'negative_acknowledge'):
                return await queue.negative_acknowledge(ack_id)
        return False
    
    async def start_background_processing(self):
        """Start background tasks for queue management"""
        self.running = True
        
        # Start dead letter processing
        task = asyncio.create_task(self._process_dead_letters())
        self.background_tasks.append(task)
        
        # Start metrics collection
        task = asyncio.create_task(self._collect_metrics())
        self.background_tasks.append(task)
        
        logger.info("Started background queue processing")
    
    async def stop_background_processing(self):
        """Stop background tasks"""
        self.running = False
        
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        logger.info("Stopped background queue processing")
    
    async def _process_dead_letters(self):
        """Process dead letter queue"""
        while self.running:
            try:
                # Check for failed messages to move to dead letter queue
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error processing dead letters: {e}")
    
    async def _collect_metrics(self):
        """Collect queue metrics periodically"""
        while self.running:
            try:
                metrics = self.get_all_metrics()
                # Could send to monitoring system here
                await asyncio.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all queues"""
        return {
            'queues': {
                name: queue.get_metrics()
                for name, queue in self.queues.items()
            },
            'total_queues': len(self.queues),
            'total_subscribers': len(self.agent_subscriptions),
            'message_history_size': len(self.message_history)
        }
    
    def get_queue_status(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific queue"""
        if queue_name in self.queues:
            queue = self.queues[queue_name]
            return {
                'name': queue_name,
                'type': queue.__class__.__name__,
                'subscribers': list(queue.subscribers),
                'message_count': queue.message_count,
                'metrics': queue.get_metrics()
            }
        return None


# Global message queue manager instance
message_queue_manager = MessageQueueManager()


# Example usage and testing
async def test_message_queue():
    """Test the message queue system"""
    print("\n" + "="*60)
    print("Testing Inter-Agent Message Queue System")
    print("="*60)
    
    # Create manager
    manager = MessageQueueManager()
    
    # Create different types of queues
    fifo_queue = manager.create_queue("tasks", QueueType.FIFO)
    priority_queue = manager.create_queue("urgent", QueueType.PRIORITY)
    topic_queue = manager.create_queue("events", QueueType.TOPIC)
    work_queue = manager.create_queue("workload", QueueType.WORK_STEALING)
    
    print(f"\nCreated {len(manager.queues)} queues")
    
    # Subscribe agents
    agents = ["agent_1", "agent_2", "agent_3"]
    for agent in agents:
        manager.subscribe(agent, "tasks")
        manager.subscribe(agent, "urgent")
        manager.subscribe(agent, "workload")
    
    # Subscribe to topics
    manager.subscribe("agent_1", "events", "analysis")
    manager.subscribe("agent_2", "events", "processing")
    manager.subscribe("agent_3", "events", "analysis")
    
    print(f"Subscribed {len(agents)} agents to queues")
    
    # Test FIFO queue
    print("\n--- Testing FIFO Queue ---")
    msg1 = AgentMessage(
        performative=Performative.REQUEST,
        sender="coordinator",
        receiver="agent_1",
        content={"task": "Process data"}
    )
    
    success = await manager.send_message(msg1, "tasks")
    print(f"Sent message to FIFO queue: {success}")
    
    received = await manager.receive_message("agent_1", "tasks")
    if received:
        print(f"Agent 1 received: {received.content}")
    
    # Test Priority queue
    print("\n--- Testing Priority Queue ---")
    
    # Send high priority message
    high_priority = AgentMessage(
        performative=Performative.REQUEST,
        sender="coordinator",
        receiver="any",
        content={"task": "Critical task"},
        priority=MessagePriority.CRITICAL
    )
    
    # Send normal priority message
    normal_priority = AgentMessage(
        performative=Performative.INFORM,
        sender="coordinator",
        receiver="any",
        content={"info": "Regular update"},
        priority=MessagePriority.NORMAL
    )
    
    await manager.send_message(normal_priority, "urgent")
    await manager.send_message(high_priority, "urgent")
    
    # Should receive high priority first
    received = await manager.receive_message("agent_2", "urgent")
    if received:
        print(f"Received (should be critical): {received.content}")
    
    # Test Topic queue
    print("\n--- Testing Topic Queue ---")
    
    analysis_msg = AgentMessage(
        performative=Performative.INFORM,
        sender="data_source",
        receiver="broadcast",
        content={"data": "Analysis results"}
    )
    
    await manager.send_message(analysis_msg, "events", topic="analysis")
    
    # Both subscribers to "analysis" should receive
    for agent in ["agent_1", "agent_3"]:
        received = await manager.receive_message(agent, "events", topic="analysis")
        if received:
            print(f"{agent} received from topic: {received.content}")
    
    # Test Work-Stealing queue
    print("\n--- Testing Work-Stealing Queue ---")
    
    # Send multiple tasks
    for i in range(5):
        task_msg = AgentMessage(
            performative=Performative.REQUEST,
            sender="coordinator",
            receiver="any",
            content={"task": f"Task {i+1}"}
        )
        await manager.send_message(task_msg, "workload")
    
    # Agents take work
    for agent in agents:
        received = await manager.receive_message(agent, "workload")
        if received:
            print(f"{agent} took work: {received.content}")
    
    # Get metrics
    print("\n--- Queue Metrics ---")
    metrics = manager.get_all_metrics()
    for queue_name, queue_metrics in metrics['queues'].items():
        print(f"{queue_name}: {queue_metrics['message_count']} messages, "
              f"{queue_metrics['messages_sent']} sent, "
              f"{queue_metrics['messages_delivered']} delivered")
    
    print("\nMessage queue system test completed!")
    return manager


if __name__ == "__main__":
    print("Inter-Agent Message Queue System")
    print("="*60)
    
    # Run test
    manager = asyncio.run(test_message_queue())
    
    print("\n✅ Inter-Agent Message Queue System ready!")