#!/usr/bin/env python3
"""
Agent Collaboration System for Agent Lightning
Enables multiple AI agents to work together on complex tasks
"""

import os
import sys
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_communication_protocol import (
    AgentMessage, Performative, MessageRouter, 
    ConversationManager, TaskSharingProtocol
)
from agent_config import AgentConfigManager
from enhanced_production_api import EnhancedAgentService
from shared_memory_system import SharedMemorySystem
from knowledge_manager import KnowledgeManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborationMode(Enum):
    """Different modes of agent collaboration"""
    MASTER_WORKER = "master_worker"      # Hierarchical with coordinator
    PEER_TO_PEER = "peer_to_peer"        # Direct agent communication
    BLACKBOARD = "blackboard"            # Shared knowledge space
    CONTRACT_NET = "contract_net"        # Task bidding system
    PIPELINE = "pipeline"                # Sequential processing
    ENSEMBLE = "ensemble"                # Parallel with voting


class TaskComplexity(Enum):
    """Task complexity levels"""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    VERY_COMPLEX = 5


class AgentRole(Enum):
    """Roles agents can play in collaboration"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"
    AGGREGATOR = "aggregator"
    MONITOR = "monitor"


@dataclass
class CollaborativeTask:
    """Represents a task that requires collaboration"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    complexity: TaskComplexity = TaskComplexity.MODERATE
    required_capabilities: List[str] = field(default_factory=list)
    subtasks: List['CollaborativeTask'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # task_ids
    deadline: Optional[datetime] = None
    assigned_agents: Dict[str, str] = field(default_factory=dict)  # agent_id: role
    status: str = "pending"
    results: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_subtask(self, subtask: 'CollaborativeTask'):
        """Add a subtask"""
        self.subtasks.append(subtask)
        
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (dependencies met)"""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)


@dataclass
class CollaborationSession:
    """Represents a collaboration session between agents"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: CollaborativeTask = None
    mode: CollaborationMode = CollaborationMode.MASTER_WORKER
    participating_agents: List[str] = field(default_factory=list)
    coordinator: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "active"  # active, completed, failed
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    results: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class CollaborativeAgent(ABC):
    """Base class for collaborative agents"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.current_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        self.message_queue = asyncio.Queue()
        
    @abstractmethod
    async def process_task(self, task: CollaborativeTask) -> Any:
        """Process a task"""
        pass
        
    @abstractmethod
    async def handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        pass
        
    def can_handle_task(self, task: CollaborativeTask) -> bool:
        """Check if agent can handle the task"""
        required = set(task.required_capabilities)
        available = set(self.capabilities)
        return required.issubset(available)


class CollaborationOrchestrator:
    """Main orchestrator for multi-agent collaboration"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.agents: Dict[str, CollaborativeAgent] = {}
        self.message_router = MessageRouter()
        self.conversation_manager = ConversationManager()
        self.memory_system = SharedMemorySystem()
        self.config_manager = AgentConfigManager()
        self.knowledge_manager = KnowledgeManager()
        self.agent_service = EnhancedAgentService()
        
        # Collaboration patterns
        self.collaboration_patterns = {
            CollaborationMode.MASTER_WORKER: MasterWorkerPattern(),
            CollaborationMode.PEER_TO_PEER: PeerToPeerPattern(),
            CollaborationMode.BLACKBOARD: BlackboardPattern(),
            CollaborationMode.CONTRACT_NET: ContractNetPattern(),
            CollaborationMode.PIPELINE: PipelinePattern(),
            CollaborationMode.ENSEMBLE: EnsemblePattern()
        }
        
        # Performance tracking
        self.performance_metrics = defaultdict(lambda: {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_time': 0,
            'collaboration_efficiency': 0.0
        })
    
    async def initialize(self):
        """Initialize the orchestrator"""
        # Load all available agents
        await self._load_agents()
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents")
    
    async def _load_agents(self):
        """Load all available agents from configuration"""
        agent_names = self.config_manager.list_agents()
        for name in agent_names:
            config = self.config_manager.get_agent(name)
            if config:
                # Create agent wrapper for collaboration
                agent = SpecializedCollaborativeAgent(
                    agent_id=name,
                    config=config,
                    agent_service=self.agent_service
                )
                self.agents[name] = agent
                await self.message_router.register_agent(name)
    
    async def start_collaboration(
        self, 
        task: CollaborativeTask,
        mode: CollaborationMode = CollaborationMode.MASTER_WORKER,
        agents: Optional[List[str]] = None
    ) -> CollaborationSession:
        """Start a new collaboration session"""
        
        # Create session
        session = CollaborationSession(
            task=task,
            mode=mode,
            participating_agents=agents or []
        )
        
        # Select agents if not specified
        if not session.participating_agents:
            session.participating_agents = await self._select_agents_for_task(task)
        
        # Select collaboration pattern
        pattern = self.collaboration_patterns[mode]
        
        # Initialize pattern
        await pattern.initialize(session, self.agents, self.message_router)
        
        # Store session
        self.sessions[session.session_id] = session
        
        # Start collaboration
        asyncio.create_task(self._execute_collaboration(session, pattern))
        
        logger.info(f"Started collaboration session {session.session_id} with mode {mode.value}")
        return session
    
    async def _select_agents_for_task(self, task: CollaborativeTask) -> List[str]:
        """Select best agents for a task based on capabilities"""
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            if agent.can_handle_task(task):
                # Calculate agent score based on various factors
                score = self._calculate_agent_score(agent, task)
                suitable_agents.append((agent_id, score))
        
        # Sort by score and select top agents
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        selected = [agent_id for agent_id, _ in suitable_agents[:5]]  # Top 5 agents
        
        logger.info(f"Selected agents for task: {selected}")
        return selected
    
    def _calculate_agent_score(self, agent: CollaborativeAgent, task: CollaborativeTask) -> float:
        """Calculate agent suitability score for a task"""
        score = 0.0
        
        # Capability match
        required = set(task.required_capabilities)
        available = set(agent.capabilities)
        overlap = len(required.intersection(available))
        score += overlap * 10
        
        # Current workload (prefer less loaded agents)
        score -= len(agent.current_tasks) * 2
        
        # Past performance
        if agent.agent_id in self.performance_metrics:
            metrics = self.performance_metrics[agent.agent_id]
            score += metrics['collaboration_efficiency'] * 5
        
        return score
    
    async def _execute_collaboration(self, session: CollaborationSession, pattern: 'CollaborationPattern'):
        """Execute the collaboration using selected pattern"""
        try:
            # Record start in memory
            self.memory_system.add_conversation(
                agent="collaboration_orchestrator",
                user_query=f"Start collaboration: {session.task.description}",
                agent_response=f"Session {session.session_id} started with {len(session.participating_agents)} agents",
                metadata={'session_id': session.session_id, 'mode': session.mode.value}
            )
            
            # Execute pattern
            result = await pattern.execute(session.task)
            
            # Update session
            session.results = result
            session.status = "completed"
            session.end_time = datetime.now()
            
            # Calculate metrics
            duration = (session.end_time - session.start_time).total_seconds()
            session.metrics = {
                'duration': duration,
                'agents_used': len(session.participating_agents),
                'subtasks_completed': len([t for t in session.task.subtasks if t.status == 'completed'])
            }
            
            # Update performance metrics
            for agent_id in session.participating_agents:
                self.performance_metrics[agent_id]['tasks_completed'] += 1
                
            logger.info(f"Collaboration session {session.session_id} completed successfully")
            
        except Exception as e:
            session.status = "failed"
            session.end_time = datetime.now()
            logger.error(f"Collaboration session {session.session_id} failed: {e}")
            
            # Update failure metrics
            for agent_id in session.participating_agents:
                self.performance_metrics[agent_id]['tasks_failed'] += 1
    
    async def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get status of a collaboration session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return {
                'session_id': session_id,
                'status': session.status,
                'task': session.task.description,
                'mode': session.mode.value,
                'agents': session.participating_agents,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'metrics': session.metrics,
                'results': session.results
            }
        return None


class CollaborationPattern(ABC):
    """Base class for collaboration patterns"""
    
    @abstractmethod
    async def initialize(self, session: CollaborationSession, agents: Dict, router: MessageRouter):
        """Initialize the pattern"""
        pass
    
    @abstractmethod
    async def execute(self, task: CollaborativeTask) -> Any:
        """Execute the collaboration pattern"""
        pass


class MasterWorkerPattern(CollaborationPattern):
    """Hierarchical master-worker collaboration pattern"""
    
    def __init__(self):
        self.master: Optional[str] = None
        self.workers: List[str] = []
        self.router: Optional[MessageRouter] = None
        self.task_assignments: Dict[str, str] = {}  # task_id: worker_id
    
    async def initialize(self, session: CollaborationSession, agents: Dict, router: MessageRouter):
        """Initialize master-worker pattern"""
        self.router = router
        
        # Select master (first agent or designated coordinator)
        if session.coordinator:
            self.master = session.coordinator
        else:
            self.master = session.participating_agents[0]
        
        # Rest are workers
        self.workers = [a for a in session.participating_agents if a != self.master]
        
        logger.info(f"Master-Worker initialized: Master={self.master}, Workers={self.workers}")
    
    async def execute(self, task: CollaborativeTask) -> Any:
        """Execute master-worker collaboration"""
        # Master decomposes task
        subtasks = await self._decompose_task(task)
        
        # Assign subtasks to workers
        assignments = await self._assign_tasks(subtasks, self.workers)
        
        # Execute in parallel
        results = await asyncio.gather(*[
            self._execute_subtask(subtask, worker)
            for subtask, worker in assignments
        ])
        
        # Aggregate results
        final_result = await self._aggregate_results(results, task)
        
        return final_result
    
    async def _decompose_task(self, task: CollaborativeTask) -> List[CollaborativeTask]:
        """Decompose task into subtasks"""
        # Simple decomposition for now
        if not task.subtasks:
            # Create subtasks based on task complexity
            num_subtasks = min(task.complexity.value, len(self.workers))
            for i in range(num_subtasks):
                subtask = CollaborativeTask(
                    description=f"Subtask {i+1} of {task.description}",
                    complexity=TaskComplexity.SIMPLE,
                    required_capabilities=task.required_capabilities
                )
                task.add_subtask(subtask)
        
        return task.subtasks
    
    async def _assign_tasks(
        self, 
        subtasks: List[CollaborativeTask], 
        workers: List[str]
    ) -> List[Tuple[CollaborativeTask, str]]:
        """Assign subtasks to workers"""
        assignments = []
        worker_idx = 0
        
        for subtask in subtasks:
            # Round-robin assignment for now
            worker = workers[worker_idx % len(workers)]
            assignments.append((subtask, worker))
            self.task_assignments[subtask.task_id] = worker
            worker_idx += 1
        
        return assignments
    
    async def _execute_subtask(self, subtask: CollaborativeTask, worker: str) -> Any:
        """Execute a subtask on a worker"""
        # Send task to worker
        message = AgentMessage(
            performative=Performative.REQUEST,
            sender=self.master,
            receiver=worker,
            content={
                'action': 'execute_task',
                'task': asdict(subtask)
            }
        )
        
        await self.router.route_message(message)
        
        # Wait for response (simplified)
        # In real implementation, would wait for actual response
        await asyncio.sleep(1)  # Simulate processing
        
        return {
            'task_id': subtask.task_id,
            'worker': worker,
            'result': f"Result from {worker} for {subtask.description}"
        }
    
    async def _aggregate_results(self, results: List[Any], original_task: CollaborativeTask) -> Any:
        """Aggregate results from workers"""
        return {
            'task': original_task.description,
            'pattern': 'master-worker',
            'subtask_results': results,
            'aggregated': True
        }


class PeerToPeerPattern(CollaborationPattern):
    """Peer-to-peer collaboration pattern"""
    
    async def initialize(self, session: CollaborationSession, agents: Dict, router: MessageRouter):
        """Initialize peer-to-peer pattern"""
        self.agents = session.participating_agents
        self.router = router
        
        # All agents subscribe to broadcast
        for agent_id in self.agents:
            self.router.subscribe_to_broadcast(agent_id)
    
    async def execute(self, task: CollaborativeTask) -> Any:
        """Execute peer-to-peer collaboration"""
        # Broadcast task to all peers
        message = AgentMessage(
            performative=Performative.CFP,
            sender="orchestrator",
            receiver="broadcast",
            content={'task': asdict(task)}
        )
        
        await self.router.route_message(message)
        
        # Peers negotiate and collaborate directly
        # Simplified for now
        return {
            'task': task.description,
            'pattern': 'peer-to-peer',
            'participants': self.agents
        }


class BlackboardPattern(CollaborationPattern):
    """Blackboard collaboration pattern with shared knowledge"""
    
    def __init__(self):
        self.blackboard: Dict[str, Any] = {}
        self.contributors: List[str] = []
    
    async def initialize(self, session: CollaborationSession, agents: Dict, router: MessageRouter):
        """Initialize blackboard pattern"""
        self.contributors = session.participating_agents
        self.router = router
        
        # Initialize blackboard with task
        self.blackboard['task'] = asdict(session.task)
        self.blackboard['knowledge'] = []
        self.blackboard['hypotheses'] = []
        self.blackboard['solutions'] = []
    
    async def execute(self, task: CollaborativeTask) -> Any:
        """Execute blackboard collaboration"""
        # Each agent contributes to blackboard
        for agent_id in self.contributors:
            contribution = await self._get_contribution(agent_id, task)
            self._update_blackboard(agent_id, contribution)
        
        # Synthesize final solution
        solution = self._synthesize_solution()
        
        return {
            'task': task.description,
            'pattern': 'blackboard',
            'blackboard_state': self.blackboard,
            'solution': solution
        }
    
    async def _get_contribution(self, agent_id: str, task: CollaborativeTask) -> Any:
        """Get contribution from an agent"""
        # Simulate agent analysis
        return {
            'agent': agent_id,
            'analysis': f"Analysis from {agent_id}",
            'suggestions': [f"Suggestion from {agent_id}"]
        }
    
    def _update_blackboard(self, agent_id: str, contribution: Any):
        """Update blackboard with contribution"""
        self.blackboard['knowledge'].append(contribution)
    
    def _synthesize_solution(self) -> Any:
        """Synthesize final solution from blackboard"""
        return {
            'synthesized': True,
            'knowledge_items': len(self.blackboard['knowledge']),
            'contributors': self.contributors
        }


class ContractNetPattern(CollaborationPattern):
    """Contract net protocol for task distribution"""
    
    async def initialize(self, session: CollaborationSession, agents: Dict, router: MessageRouter):
        """Initialize contract net pattern"""
        self.agents = session.participating_agents
        self.router = router
        self.bids: Dict[str, Any] = {}
    
    async def execute(self, task: CollaborativeTask) -> Any:
        """Execute contract net collaboration"""
        # Announce task (CFP)
        await self._announce_task(task)
        
        # Collect bids
        await self._collect_bids(task)
        
        # Award contracts
        winners = self._award_contracts(task)
        
        # Execute contracts
        results = await self._execute_contracts(winners, task)
        
        return {
            'task': task.description,
            'pattern': 'contract-net',
            'bids': self.bids,
            'winners': winners,
            'results': results
        }
    
    async def _announce_task(self, task: CollaborativeTask):
        """Announce task to all agents"""
        message = AgentMessage(
            performative=Performative.CFP,
            sender="manager",
            receiver="broadcast",
            content={'task': asdict(task)},
            reply_by=datetime.now() + timedelta(seconds=5)
        )
        await self.router.route_message(message)
    
    async def _collect_bids(self, task: CollaborativeTask):
        """Collect bids from agents"""
        # Simulate bid collection
        for agent_id in self.agents:
            self.bids[agent_id] = {
                'confidence': 0.5 + (hash(agent_id) % 50) / 100,
                'time_estimate': 10 + (hash(agent_id) % 20),
                'cost': 100 + (hash(agent_id) % 100)
            }
    
    def _award_contracts(self, task: CollaborativeTask) -> List[str]:
        """Award contracts to best bidders"""
        # Sort by confidence and select top bidders
        sorted_bids = sorted(
            self.bids.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        # Award to top 2 bidders
        winners = [agent_id for agent_id, _ in sorted_bids[:2]]
        return winners
    
    async def _execute_contracts(self, winners: List[str], task: CollaborativeTask) -> Any:
        """Execute awarded contracts"""
        results = []
        for agent_id in winners:
            results.append({
                'agent': agent_id,
                'completed': True,
                'output': f"Contract executed by {agent_id}"
            })
        return results


class PipelinePattern(CollaborationPattern):
    """Pipeline pattern for sequential processing"""
    
    async def initialize(self, session: CollaborationSession, agents: Dict, router: MessageRouter):
        """Initialize pipeline pattern"""
        self.pipeline = session.participating_agents
        self.router = router
    
    async def execute(self, task: CollaborativeTask) -> Any:
        """Execute pipeline collaboration"""
        result = {'initial': task.description}
        
        # Process through pipeline
        for i, agent_id in enumerate(self.pipeline):
            result = await self._process_stage(agent_id, result, i)
        
        return {
            'task': task.description,
            'pattern': 'pipeline',
            'pipeline': self.pipeline,
            'final_result': result
        }
    
    async def _process_stage(self, agent_id: str, input_data: Any, stage: int) -> Any:
        """Process a pipeline stage"""
        # Simulate stage processing
        return {
            'stage': stage,
            'agent': agent_id,
            'input': input_data,
            'output': f"Processed by {agent_id}"
        }


class EnsemblePattern(CollaborationPattern):
    """Ensemble pattern with voting/consensus"""
    
    async def initialize(self, session: CollaborationSession, agents: Dict, router: MessageRouter):
        """Initialize ensemble pattern"""
        self.agents = session.participating_agents
        self.router = router
    
    async def execute(self, task: CollaborativeTask) -> Any:
        """Execute ensemble collaboration"""
        # Get predictions from all agents
        predictions = await asyncio.gather(*[
            self._get_prediction(agent_id, task)
            for agent_id in self.agents
        ])
        
        # Aggregate predictions
        final_result = self._aggregate_predictions(predictions)
        
        return {
            'task': task.description,
            'pattern': 'ensemble',
            'predictions': predictions,
            'consensus': final_result
        }
    
    async def _get_prediction(self, agent_id: str, task: CollaborativeTask) -> Any:
        """Get prediction from an agent"""
        return {
            'agent': agent_id,
            'prediction': f"Prediction from {agent_id}",
            'confidence': 0.5 + (hash(agent_id) % 50) / 100
        }
    
    def _aggregate_predictions(self, predictions: List[Any]) -> Any:
        """Aggregate predictions using voting or averaging"""
        # Simple majority voting for now
        return {
            'method': 'majority_vote',
            'participants': len(predictions),
            'result': 'Aggregated result'
        }


class SpecializedCollaborativeAgent(CollaborativeAgent):
    """Wrapper for specialized agents to work in collaboration"""
    
    def __init__(self, agent_id: str, config, agent_service):
        capabilities = [k for k, v in config.capabilities.__dict__.items() if v]
        super().__init__(agent_id, capabilities)
        self.config = config
        self.agent_service = agent_service
    
    async def process_task(self, task: CollaborativeTask) -> Any:
        """Process task using the specialized agent"""
        # Convert CollaborativeTask to agent request
        from production_api import AgentRequest
        
        request = AgentRequest(
            task=task.description,
            agent_id=self.agent_id,
            context=task.metadata
        )
        
        # Execute through agent service
        response = await self.agent_service.process_task_with_knowledge(request)
        
        return response.result
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        if message.performative == Performative.REQUEST:
            # Process task request
            if message.content.get('action') == 'execute_task':
                task_data = message.content.get('task')
                task = CollaborativeTask(**task_data)
                result = await self.process_task(task)
                
                # Send response
                reply = message.create_reply(
                    performative=Performative.INFORM,
                    content={'result': result}
                )
                await self.message_queue.put(reply)
        
        elif message.performative == Performative.CFP:
            # Respond to call for proposals
            proposal = {
                'agent_id': self.agent_id,
                'capabilities': self.capabilities,
                'confidence': 0.8
            }
            
            reply = message.create_reply(
                performative=Performative.PROPOSE,
                content={'proposal': proposal}
            )
            await self.message_queue.put(reply)


# API Integration functions
async def create_collaborative_task(
    description: str,
    complexity: int = 3,
    required_capabilities: List[str] = None,
    deadline_hours: int = 24
) -> CollaborativeTask:
    """Create a collaborative task"""
    task = CollaborativeTask(
        description=description,
        complexity=TaskComplexity(min(max(complexity, 1), 5)),
        required_capabilities=required_capabilities or [],
        deadline=datetime.now() + timedelta(hours=deadline_hours) if deadline_hours else None
    )
    return task


async def start_collaboration_session(
    task: CollaborativeTask,
    mode: str = "master_worker",
    agents: List[str] = None
) -> Dict:
    """Start a collaboration session"""
    orchestrator = CollaborationOrchestrator()
    await orchestrator.initialize()
    
    collaboration_mode = CollaborationMode[mode.upper()]
    session = await orchestrator.start_collaboration(task, collaboration_mode, agents)
    
    return {
        'session_id': session.session_id,
        'task': task.description,
        'mode': mode,
        'agents': session.participating_agents,
        'status': session.status
    }


if __name__ == "__main__":
    print("Agent Collaboration System")
    print("=" * 60)
    
    async def test_collaboration():
        # Create a test task
        task = await create_collaborative_task(
            description="Analyze and optimize a Python codebase for performance",
            complexity=4,
            required_capabilities=["code_analysis", "optimization", "python"],
            deadline_hours=2
        )
        
        print(f"\nTask Created: {task.task_id}")
        print(f"Description: {task.description}")
        print(f"Complexity: {task.complexity.name}")
        
        # Initialize orchestrator
        orchestrator = CollaborationOrchestrator()
        await orchestrator.initialize()
        
        # Start collaboration
        session = await orchestrator.start_collaboration(
            task,
            CollaborationMode.MASTER_WORKER
        )
        
        print(f"\nCollaboration Session: {session.session_id}")
        print(f"Mode: {session.mode.value}")
        print(f"Agents: {session.participating_agents}")
        
        # Simulate waiting for completion
        await asyncio.sleep(2)
        
        # Get status
        status = await orchestrator.get_session_status(session.session_id)
        print(f"\nSession Status: {json.dumps(status, indent=2, default=str)}")
    
    # Run test
    asyncio.run(test_collaboration())
    
    print("\n✅ Agent Collaboration System ready for integration!")