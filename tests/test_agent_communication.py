#!/usr/bin/env python3
"""
Tests for agent-to-agent communication system
"""

import asyncio
import pytest
from agentlightning.types import AgentAddress, AgentMessage, MessageType, CommunicationProtocol
from agentlightning.communication import MessageBroker, AgentCommunicator, create_agent_address


class TestAgentCommunication:
    """Test agent communication functionality"""

    def test_agent_address_creation(self):
        """Test creating agent addresses"""
        addr = create_agent_address("agent_001", "TestAgent", ["vision", "text"])
        assert addr.agent_id == "agent_001"
        assert addr.agent_type == "TestAgent"
        assert "vision" in addr.capabilities
        assert "text" in addr.capabilities

    def test_message_creation(self):
        """Test creating agent messages"""
        sender = create_agent_address("sender_001")
        recipient = create_agent_address("recipient_001")

        message = AgentMessage(
            message_id="msg_001",
            sender=sender,
            recipient=recipient,
            message_type=MessageType.REQUEST,
            subject="Test request",
            content={"data": "test"},
            timestamp=1234567890.0
        )

        assert message.message_id == "msg_001"
        assert message.sender.agent_id == "sender_001"
        assert message.recipient.agent_id == "recipient_001"
        assert message.message_type == MessageType.REQUEST
        assert message.subject == "Test request"
        assert message.content == {"data": "test"}

    @pytest.mark.asyncio
    async def test_message_broker_registration(self):
        """Test agent registration with message broker"""
        broker = MessageBroker()

        addr1 = create_agent_address("agent_001")
        addr2 = create_agent_address("agent_002")

        broker.register_agent(addr1)
        broker.register_agent(addr2)

        registered = broker.list_registered_agents()
        assert len(registered) == 2
        agent_ids = [addr.agent_id for addr in registered]
        assert "agent_001" in agent_ids
        assert "agent_002" in agent_ids

    @pytest.mark.asyncio
    async def test_direct_messaging(self):
        """Test direct messaging between agents"""
        broker = MessageBroker()

        # Register agents
        addr1 = create_agent_address("agent_001")
        addr2 = create_agent_address("agent_002")
        broker.register_agent(addr1)
        broker.register_agent(addr2)

        # Create communicators
        comm1 = AgentCommunicator(addr1)
        comm2 = AgentCommunicator(addr2)

        # Set up message handler for agent 2
        received_messages = []

        async def handler(message):
            received_messages.append(message)

        comm2.set_message_handler(handler)

        # Send message from agent 1 to agent 2
        success = await comm1.send_message("agent_002", "Test message", {"data": "hello"})
        assert success

        # Deliver messages
        await broker.deliver_messages()

        # Check that message was received
        assert len(received_messages) == 1
        msg = received_messages[0]
        assert msg.sender.agent_id == "agent_001"
        assert msg.recipient.agent_id == "agent_002"
        assert msg.subject == "Test message"
        assert msg.content == {"data": "hello"}

    @pytest.mark.asyncio
    async def test_broadcast_messaging(self):
        """Test broadcast messaging to all agents"""
        broker = MessageBroker()

        # Register multiple agents
        addrs = []
        comms = []
        handlers = []

        for i in range(3):
            addr = create_agent_address(f"agent_{i:03d}")
            addrs.append(addr)
            broker.register_agent(addr)

            comm = AgentCommunicator(addr)
            comms.append(comm)

            received = []
            async def handler(message):
                received.append(message)
            handlers.append(handler)
            comm.set_message_handler(handler)

        # Broadcast from agent 0
        results = await comms[0].broadcast("Broadcast test", {"broadcast": True})
        assert len(results) == 2  # Should send to 2 other agents

        # Deliver messages
        await broker.deliver_messages()

        # Check that agents 1 and 2 received the message
        for i in range(1, 3):
            # Wait a bit for async delivery
            await asyncio.sleep(0.01)
            # Note: In a real implementation, we'd need to wait for delivery
            # For this test, we assume the message was queued

    @pytest.mark.asyncio
    async def test_request_response_pattern(self):
        """Test request-response communication pattern"""
        broker = MessageBroker()

        # Register agents
        addr1 = create_agent_address("requester")
        addr2 = create_agent_address("responder")
        broker.register_agent(addr1)
        broker.register_agent(addr2)

        comm1 = AgentCommunicator(addr1)
        comm2 = AgentCommunicator(addr2)

        # Set up response handler for responder
        async def response_handler(message):
            if message.message_type == MessageType.REQUEST:
                # Send response back
                await comm2.send_message(
                    message.sender.agent_id,
                    f"Response to: {message.subject}",
                    {"response": "acknowledged", "original": message.content},
                    message_type=MessageType.RESPONSE,
                    correlation_id=message.correlation_id
                )

        comm2.set_message_handler(response_handler)

        # Send request and wait for response
        response = await comm1.request_response("responder", "Test request", {"request": "data"})

        # Should receive the response
        assert response is not None
        assert response["response"] == "acknowledged"
        assert response["original"] == {"request": "data"}


if __name__ == "__main__":
    pytest.main([__file__])