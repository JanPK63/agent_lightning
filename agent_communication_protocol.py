#!/usr/bin/env python3
"""
Agent Communication Protocol for Multi-Agent Collaboration
Implements FIPA ACL-compliant messaging with extensions for Agent Lightning
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib


class Performative(Enum):
    """FIPA ACL performative types for agent communication"""
    # Information passing
    INFORM = "inform"              # Share information
    INFORM_IF = "inform-if"        # Conditional information
    INFORM_REF = "inform-ref"      # Reference to information
    
    # Requesting actions
    REQUEST = "request"            # Request an action
    REQUEST_WHEN = "request-when"  # Conditional request
    REQUEST_WHENEVER = "request-whenever"  # Persistent request
    
    # Querying
    QUERY_IF = "query-if"          # Yes/no question
    QUERY_REF = "query-ref"        # Request for object
    
    # Negotiation
    CFP = "cfp"                    # Call for proposals
    PROPOSE = "propose"            # Submit proposal
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"
    
    # Action performing
    AGREE = "agree"                # Agree to perform
    REFUSE = "refuse"              # Refuse to perform
    FAILURE = "failure"            # Action failed
    
    # Subscription
    SUBSCRIBE = "subscribe"        # Subscribe to updates
    CANCEL = "cancel"              # Cancel subscription
    
    # Custom for Agent Lightning
    DELEGATE = "delegate"          # Delegate subtask
    COLLABORATE = "collaborate"    # Request collaboration
    SYNC = "sync"                  # Synchronization request
    REPORT = "report"              # Status report


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class ConversationState(Enum):
    """States for conversation tracking"""
    INITIATED = "initiated"
    WAITING_RESPONSE = "waiting_response"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentMessage:
    """
    Standard agent communication message structure
    Based on FIPA ACL with extensions
    """
    # Required fields
    performative: Performative
    sender: str                    # Agent ID
    receiver: str                   # Agent ID or 'broadcast'
    content: Any                    # Message content
    
    # Optional fields
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reply_to: Optional[str] = None
    reply_by: Optional[datetime] = None
    language: str = "json"
    encoding: str = "utf-8"
    ontology: str = "agent-lightning"
    protocol: str = "task-sharing"
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize message to JSON"""
        data = asdict(self)
        # Convert enums to values
        data['performative'] = self.performative.value
        data['priority'] = self.priority.value
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        if self.reply_by:
            data['reply_by'] = self.reply_by.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        # Convert values back to enums
        data['performative'] = Performative(data['performative'])
        data['priority'] = MessagePriority(data['priority'])
        # Convert ISO format to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('reply_by'):
            data['reply_by'] = datetime.fromisoformat(data['reply_by'])
        return cls(**data)
    
    def create_reply(self, performative: Performative, content: Any) -> 'AgentMessage':
        """Create a reply message"""
        return AgentMessage(
            performative=performative,
            sender=self.receiver,
            receiver=self.sender,
            content=content,
            conversation_id=self.conversation_id,
            reply_to=self.message_id,
            language=self.language,
            ontology=self.ontology,
            protocol=self.protocol
        )


@dataclass
class TaskSharingProtocol:
    """Protocol for sharing tasks between agents"""
    
    @dataclass
    class TaskAnnouncement:
        """Structure for task announcements"""
        task_id: str
        task_type: str
        description: str
        requirements: Dict[str, Any]
        deadline: Optional[datetime]
        reward: Optional[float]
        complexity: int  # 1-10
        dependencies: List[str] = field(default_factory=list)
        
    @dataclass
    class TaskProposal:
        """Structure for task proposals from agents"""
        agent_id: str
        task_id: str
        estimated_time: timedelta
        confidence: float  # 0.0 to 1.0
        approach: str
        cost: Optional[float] = None
        
    @dataclass
    class TaskAssignment:
        """Structure for task assignments"""
        task_id: str
        assigned_to: str
        deadline: datetime
        resources: Dict[str, Any]
        monitoring_interval: timedelta
        
    @dataclass
    class TaskReport:
        """Structure for task progress reports"""
        task_id: str
        agent_id: str
        status: str  # "in_progress", "completed", "failed", "blocked"
        progress: float  # 0.0 to 1.0
        results: Optional[Any] = None
        issues: List[str] = field(default_factory=list)


class ConversationManager:
    """Manages multi-turn conversations between agents"""
    
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}
        self.message_history: List[AgentMessage] = []
        self.conversation_handlers: Dict[str, Callable] = {}
        
    def start_conversation(self, initial_message: AgentMessage) -> str:
        """Start a new conversation"""
        conv_id = initial_message.conversation_id
        self.conversations[conv_id] = {
            'state': ConversationState.INITIATED,
            'participants': [initial_message.sender, initial_message.receiver],
            'messages': [initial_message],
            'started': datetime.now(),
            'last_activity': datetime.now()
        }
        return conv_id
    
    def add_message(self, message: AgentMessage):
        """Add message to conversation"""
        conv_id = message.conversation_id
        if conv_id in self.conversations:
            self.conversations[conv_id]['messages'].append(message)
            self.conversations[conv_id]['last_activity'] = datetime.now()
            
            # Add participants if new
            if message.sender not in self.conversations[conv_id]['participants']:
                self.conversations[conv_id]['participants'].append(message.sender)
        else:
            self.start_conversation(message)
        
        self.message_history.append(message)
    
    def get_conversation_state(self, conversation_id: str) -> ConversationState:
        """Get current conversation state"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]['state']
        return ConversationState.FAILED
    
    def update_conversation_state(self, conversation_id: str, state: ConversationState):
        """Update conversation state"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['state'] = state
    
    def get_conversation_history(self, conversation_id: str) -> List[AgentMessage]:
        """Get all messages in a conversation"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]['messages']
        return []
    
    def cleanup_old_conversations(self, max_age: timedelta):
        """Remove old completed/failed conversations"""
        now = datetime.now()
        to_remove = []
        
        for conv_id, conv in self.conversations.items():
            if conv['state'] in [ConversationState.COMPLETED, ConversationState.FAILED]:
                if now - conv['last_activity'] > max_age:
                    to_remove.append(conv_id)
        
        for conv_id in to_remove:
            del self.conversations[conv_id]


class MessageRouter:
    """Routes messages between agents"""
    
    def __init__(self):
        self.agent_queues: Dict[str, asyncio.Queue] = {}
        self.broadcast_subscribers: List[str] = []
        self.topic_subscribers: Dict[str, List[str]] = defaultdict(list)
        self.message_filters: Dict[str, Callable] = {}
        
    async def register_agent(self, agent_id: str):
        """Register an agent for message routing"""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = asyncio.Queue()
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]
        
        # Remove from subscribers
        if agent_id in self.broadcast_subscribers:
            self.broadcast_subscribers.remove(agent_id)
        
        for subscribers in self.topic_subscribers.values():
            if agent_id in subscribers:
                subscribers.remove(agent_id)
    
    async def route_message(self, message: AgentMessage):
        """Route message to appropriate recipient(s)"""
        # Handle broadcast messages
        if message.receiver == 'broadcast':
            for agent_id in self.broadcast_subscribers:
                if agent_id != message.sender:  # Don't send to self
                    await self._deliver_message(agent_id, message)
        
        # Handle direct messages
        elif message.receiver in self.agent_queues:
            await self._deliver_message(message.receiver, message)
        
        # Handle topic-based routing
        if 'topic' in message.metadata:
            topic = message.metadata['topic']
            if topic in self.topic_subscribers:
                for agent_id in self.topic_subscribers[topic]:
                    if agent_id != message.sender:
                        await self._deliver_message(agent_id, message)
    
    async def _deliver_message(self, agent_id: str, message: AgentMessage):
        """Deliver message to agent's queue"""
        if agent_id in self.agent_queues:
            # Apply filters if any
            if agent_id in self.message_filters:
                if not self.message_filters[agent_id](message):
                    return  # Message filtered out
            
            await self.agent_queues[agent_id].put(message)
    
    async def get_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Get message for agent"""
        if agent_id in self.agent_queues:
            try:
                if timeout:
                    return await asyncio.wait_for(
                        self.agent_queues[agent_id].get(),
                        timeout=timeout
                    )
                else:
                    return await self.agent_queues[agent_id].get()
            except asyncio.TimeoutError:
                return None
        return None
    
    def subscribe_to_broadcast(self, agent_id: str):
        """Subscribe agent to broadcast messages"""
        if agent_id not in self.broadcast_subscribers:
            self.broadcast_subscribers.append(agent_id)
    
    def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe agent to topic"""
        if agent_id not in self.topic_subscribers[topic]:
            self.topic_subscribers[topic].append(agent_id)
    
    def set_message_filter(self, agent_id: str, filter_func: Callable):
        """Set message filter for agent"""
        self.message_filters[agent_id] = filter_func


class ProtocolValidator:
    """Validates messages against protocol specifications"""
    
    @staticmethod
    def validate_message(message: AgentMessage) -> tuple[bool, List[str]]:
        """Validate message structure and content"""
        errors = []
        
        # Check required fields
        if not message.sender:
            errors.append("Missing sender")
        if not message.receiver:
            errors.append("Missing receiver")
        if message.content is None:
            errors.append("Missing content")
        
        # Validate performative-specific requirements
        if message.performative == Performative.CFP:
            if not isinstance(message.content, dict):
                errors.append("CFP content must be a dictionary")
            elif 'task' not in message.content:
                errors.append("CFP must contain 'task' in content")
        
        elif message.performative == Performative.PROPOSE:
            if not isinstance(message.content, dict):
                errors.append("PROPOSE content must be a dictionary")
            elif 'proposal' not in message.content:
                errors.append("PROPOSE must contain 'proposal' in content")
        
        # Check conversation consistency
        if message.reply_to:
            # Should have same conversation_id as the message it's replying to
            pass  # Would need message history to validate
        
        # Check deadline if specified
        if message.reply_by and message.reply_by < datetime.now():
            errors.append("Reply deadline has already passed")
        
        return len(errors) == 0, errors


# Example usage functions
async def example_task_sharing():
    """Example of task sharing protocol"""
    
    # Create router and conversation manager
    router = MessageRouter()
    conv_manager = ConversationManager()
    
    # Register agents
    await router.register_agent("coordinator")
    await router.register_agent("agent_1")
    await router.register_agent("agent_2")
    
    # Coordinator announces task
    task_announcement = TaskSharingProtocol.TaskAnnouncement(
        task_id="task_001",
        task_type="code_analysis",
        description="Analyze Python codebase for security vulnerabilities",
        requirements={"language": "python", "expertise": "security"},
        deadline=datetime.now() + timedelta(hours=2),
        complexity=7,
        reward=100.0
    )
    
    cfp_message = AgentMessage(
        performative=Performative.CFP,
        sender="coordinator",
        receiver="broadcast",
        content={"task": asdict(task_announcement)}
    )
    
    # Route the CFP
    await router.route_message(cfp_message)
    conv_manager.add_message(cfp_message)
    
    print(f"Task announced: {task_announcement.task_id}")
    
    # Agents respond with proposals
    proposal_1 = TaskSharingProtocol.TaskProposal(
        agent_id="agent_1",
        task_id="task_001",
        estimated_time=timedelta(hours=1.5),
        confidence=0.85,
        approach="Static analysis with Bandit and custom rules"
    )
    
    propose_message = cfp_message.create_reply(
        performative=Performative.PROPOSE,
        content={"proposal": asdict(proposal_1)}
    )
    
    await router.route_message(propose_message)
    conv_manager.add_message(propose_message)
    
    print(f"Proposal received from {proposal_1.agent_id}")


if __name__ == "__main__":
    print("Agent Communication Protocol Module")
    print("=" * 60)
    
    # Test message creation and serialization
    msg = AgentMessage(
        performative=Performative.REQUEST,
        sender="agent_1",
        receiver="agent_2",
        content={"action": "analyze", "target": "codebase"}
    )
    
    print("\nSample Message:")
    print(f"  From: {msg.sender}")
    print(f"  To: {msg.receiver}")
    print(f"  Type: {msg.performative.value}")
    print(f"  Content: {msg.content}")
    
    # Test serialization
    json_msg = msg.to_json()
    print(f"\nSerialized: {json_msg[:100]}...")
    
    # Test deserialization
    restored = AgentMessage.from_json(json_msg)
    print(f"\nDeserialized Successfully: {restored.message_id == msg.message_id}")
    
    # Run async example
    print("\n" + "=" * 60)
    print("Running Task Sharing Example...")
    asyncio.run(example_task_sharing())
    
    print("\n✅ Agent Communication Protocol ready for integration!")