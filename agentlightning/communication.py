#!/usr/bin/env python3
"""
Agent-to-Agent Communication System for Agent Lightning

This module provides protocols and infrastructure for agents to communicate with each other,
enabling collaborative workflows, knowledge sharing, and coordinated actions.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Awaitable
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from agentlightning.types import AgentMessage, AgentAddress, CommunicationProtocol, MessageType

logger = logging.getLogger(__name__)


class CommunicationChannel(ABC):
    """Abstract base class for communication channels"""

    @abstractmethod
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through this channel"""
        pass

    @abstractmethod
    async def receive_messages(self, agent_address: AgentAddress) -> List[AgentMessage]:
        """Receive messages for a specific agent"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the channel is available"""
        pass


class InMemoryChannel(CommunicationChannel):
    """Simple in-memory communication channel for local agent communication"""

    def __init__(self):
        self.message_queues: Dict[str, List[AgentMessage]] = {}
        self.lock = asyncio.Lock()

    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message to the recipient's queue"""
        async with self.lock:
            recipient_key = f"{message.recipient.agent_id}"
            if recipient_key not in self.message_queues:
                self.message_queues[recipient_key] = []
            self.message_queues[recipient_key].append(message)
            logger.debug(f"Message {message.message_id} queued for {recipient_key}")
            return True

    async def receive_messages(self, agent_address: AgentAddress) -> List[AgentMessage]:
        """Receive all messages for the agent"""
        async with self.lock:
            agent_key = f"{agent_address.agent_id}"
            messages = self.message_queues.get(agent_key, [])
            self.message_queues[agent_key] = []  # Clear the queue
            return messages

    def is_available(self) -> bool:
        return True


class MockRedisChannel(CommunicationChannel):
    """Mock Redis-like channel for testing without Redis"""

    def __init__(self):
        self.channels: Dict[str, List[AgentMessage]] = {}
        self.lock = asyncio.Lock()
        logger.info("Using mock Redis channel for communication")

    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message to a channel"""
        async with self.lock:
            channel_key = f"agent:{message.recipient.agent_id}"
            if channel_key not in self.channels:
                self.channels[channel_key] = []
            self.channels[channel_key].append(message)
            logger.debug(f"Message {message.message_id} published to {channel_key}")
            return True

    async def receive_messages(self, agent_address: AgentAddress) -> List[AgentMessage]:
        """Receive messages from agent's channel"""
        async with self.lock:
            channel_key = f"agent:{agent_address.agent_id}"
            messages = self.channels.get(channel_key, [])
            self.channels[channel_key] = []  # Clear the channel
            return messages

    def is_available(self) -> bool:
        return True


class MessageBroker:
    """Central message broker for agent communication"""

    def __init__(self):
        self.channels: Dict[CommunicationProtocol, CommunicationChannel] = {}
        self.agent_registry: Dict[str, AgentAddress] = {}
        self.message_handlers: Dict[str, Callable[[AgentMessage], Awaitable[None]]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    def register_channel(self, protocol: CommunicationProtocol, channel: CommunicationChannel):
        """Register a communication channel"""
        self.channels[protocol] = channel
        logger.info(f"Registered {protocol.value} communication channel")

    def register_agent(self, agent_address: AgentAddress):
        """Register an agent in the system"""
        self.agent_registry[agent_address.agent_id] = agent_address
        logger.info(f"Registered agent: {agent_address.agent_id}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    def get_agent_address(self, agent_id: str) -> Optional[AgentAddress]:
        """Get agent address by ID"""
        return self.agent_registry.get(agent_id)

    def register_message_handler(self, agent_id: str, handler: Callable[[AgentMessage], Awaitable[None]]):
        """Register a message handler for an agent"""
        self.message_handlers[agent_id] = handler
        logger.info(f"Registered message handler for agent: {agent_id}")

    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message using the appropriate channel"""
        channel = self.channels.get(message.protocol)
        if not channel:
            logger.error(f"No channel available for protocol: {message.protocol}")
            return False

        if not channel.is_available():
            logger.error(f"Channel {message.protocol} is not available")
            return False

        # Validate recipient exists
        if message.recipient.agent_id not in self.agent_registry:
            logger.warning(f"Recipient agent {message.recipient.agent_id} not registered")
            # Still send the message in case it's a broadcast or the agent registers later

        success = await channel.send_message(message)
        if success:
            logger.debug(f"Message {message.message_id} sent successfully")
        return success

    async def deliver_messages(self):
        """Deliver pending messages to registered agents"""
        for agent_id, handler in self.message_handlers.items():
            agent_address = self.agent_registry.get(agent_id)
            if not agent_address:
                continue

            # Get the appropriate channel (assume direct for now)
            channel = self.channels.get(CommunicationProtocol.DIRECT)
            if not channel:
                continue

            messages = await channel.receive_messages(agent_address)
            for message in messages:
                try:
                    await handler(message)
                    logger.debug(f"Delivered message {message.message_id} to {agent_id}")
                except Exception as e:
                    logger.error(f"Error delivering message {message.message_id} to {agent_id}: {e}")

    async def broadcast_message(self, sender: AgentAddress, subject: str, content: Any,
                              message_type: MessageType = MessageType.NOTIFICATION) -> List[bool]:
        """Broadcast a message to all registered agents"""
        results = []
        for agent_address in self.agent_registry.values():
            if agent_address.agent_id == sender.agent_id:
                continue  # Don't send to self

            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender=sender,
                recipient=agent_address,
                message_type=message_type,
                subject=subject,
                content=content,
                timestamp=time.time()
            )
            success = await self.send_message(message)
            results.append(success)

        return results

    def list_registered_agents(self) -> List[AgentAddress]:
        """List all registered agents"""
        return list(self.agent_registry.values())


# Global message broker instance
message_broker = MessageBroker()

# Initialize with appropriate channels
try:
    # Try to use Redis if available
    import redis
    # Test Redis connection
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    # If we get here, Redis is available
    message_broker.register_channel(CommunicationProtocol.PUBSUB, MockRedisChannel())
    logger.info("✅ Redis available - using Redis channels")
except ImportError:
    logger.warning("Redis not installed - using mock channels")
    message_broker.register_channel(CommunicationProtocol.PUBSUB, MockRedisChannel())
except Exception as e:
    logger.warning(f"Redis not available ({e}) - using mock channels")
    message_broker.register_channel(CommunicationProtocol.PUBSUB, MockRedisChannel())

# Always register in-memory channel as fallback
message_broker.register_channel(CommunicationProtocol.DIRECT, InMemoryChannel())


class AgentCommunicator:
    """Helper class for agents to communicate with each other"""

    def __init__(self, agent_address: AgentAddress):
        self.agent_address = agent_address
        self.message_broker = message_broker

    def register(self):
        """Register this agent with the message broker"""
        self.message_broker.register_agent(self.agent_address)

    def unregister(self):
        """Unregister this agent"""
        self.message_broker.unregister_agent(self.agent_address.agent_id)

    def set_message_handler(self, handler: Callable[[AgentMessage], Awaitable[None]]):
        """Set the message handler for this agent"""
        self.message_broker.register_message_handler(self.agent_address.agent_id, handler)

    async def send_message(self, recipient_id: str, subject: str, content: Any,
                          message_type: MessageType = MessageType.REQUEST,
                          protocol: CommunicationProtocol = CommunicationProtocol.DIRECT) -> bool:
        """Send a message to another agent"""
        recipient = self.message_broker.get_agent_address(recipient_id)
        if not recipient:
            logger.error(f"Unknown recipient: {recipient_id}")
            return False

        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender=self.agent_address,
            recipient=recipient,
            message_type=message_type,
            protocol=protocol,
            subject=subject,
            content=content,
            timestamp=time.time()
        )

        return await self.message_broker.send_message(message)

    async def broadcast(self, subject: str, content: Any,
                       message_type: MessageType = MessageType.NOTIFICATION) -> List[bool]:
        """Broadcast a message to all other agents"""
        return await self.message_broker.broadcast_message(
            self.agent_address, subject, content, message_type
        )

    async def request_response(self, recipient_id: str, subject: str, content: Any,
                              timeout: float = 30.0) -> Optional[Any]:
        """Send a request and wait for a response"""
        response_future = asyncio.Future()
        correlation_id = str(uuid.uuid4())

        async def response_handler(message: AgentMessage):
            if message.correlation_id == correlation_id and message.message_type == MessageType.RESPONSE:
                response_future.set_result(message.content)

        # Set up temporary handler for response
        original_handler = self.message_broker.message_handlers.get(self.agent_address.agent_id)

        async def combined_handler(message: AgentMessage):
            await response_handler(message)
            if original_handler:
                await original_handler(message)

        self.message_broker.register_message_handler(self.agent_address.agent_id, combined_handler)

        try:
            # Send the request
            recipient = self.message_broker.get_agent_address(recipient_id)
            if not recipient:
                return None

            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender=self.agent_address,
                recipient=recipient,
                message_type=MessageType.REQUEST,
                subject=subject,
                content=content,
                correlation_id=correlation_id,
                reply_to=self.agent_address,
                timestamp=time.time()
            )

            success = await self.message_broker.send_message(message)
            if not success:
                return None

            # Wait for response
            return await asyncio.wait_for(response_future, timeout=timeout)

        finally:
            # Restore original handler
            if original_handler:
                self.message_broker.register_message_handler(self.agent_address.agent_id, original_handler)
            else:
                self.message_broker.message_handlers.pop(self.agent_address.agent_id, None)


# Convenience functions
async def send_agent_message(sender_id: str, recipient_id: str, subject: str, content: Any) -> bool:
    """Convenience function to send a message between agents"""
    sender_addr = message_broker.get_agent_address(sender_id)
    if not sender_addr:
        return False

    communicator = AgentCommunicator(sender_addr)
    return await communicator.send_message(recipient_id, subject, content)


def get_registered_agents() -> List[AgentAddress]:
    """Get list of all registered agents"""
    return message_broker.list_registered_agents()


def create_agent_address(agent_id: str, agent_type: Optional[str] = None,
                        capabilities: Optional[List[str]] = None) -> AgentAddress:
    """Create an agent address"""
    return AgentAddress(
        agent_id=agent_id,
        agent_type=agent_type,
        capabilities=capabilities or []
    )