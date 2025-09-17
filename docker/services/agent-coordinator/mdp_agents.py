"""
Enterprise MDP Agents Module
Production-grade Markov Decision Process agents with state management
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid
from datetime import datetime


class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class AgentAction(Enum):
    START_TASK = "start_task"
    CONTINUE_TASK = "continue_task"
    PAUSE_TASK = "pause_task"
    COMPLETE_TASK = "complete_task"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class MDPTransition:
    """MDP state transition"""
    agent_id: str
    from_state: AgentState
    to_state: AgentState
    action: AgentAction
    reward: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class MDPAgent:
    """Enterprise MDP Agent with state management"""
    
    def __init__(self, agent_id: str, capabilities: List[str] = None):
        self.agent_id = agent_id
        self.capabilities = capabilities or []
        self.current_state = AgentState.IDLE
        self.transitions: List[MDPTransition] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "total_reward": 0.0,
            "success_rate": 1.0,
            "avg_execution_time": 0.0
        }
    
    def transition(self, action: AgentAction, new_state: AgentState, reward: float = 0.0) -> MDPTransition:
        """Execute state transition"""
        transition = MDPTransition(
            agent_id=self.agent_id,
            from_state=self.current_state,
            to_state=new_state,
            action=action,
            reward=reward
        )
        
        self.transitions.append(transition)
        self.current_state = new_state
        self.performance_metrics["total_reward"] += reward
        
        if new_state == AgentState.COMPLETED:
            self.performance_metrics["tasks_completed"] += 1
        
        return transition
    
    def get_state(self) -> AgentState:
        """Get current agent state"""
        return self.current_state
    
    def get_performance(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return self.performance_metrics.copy()


def create_transition_batch(transitions: List[MDPTransition]) -> Dict[str, Any]:
    """Create batch of transitions for training"""
    return {
        "transitions": [
            {
                "agent_id": t.agent_id,
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "action": t.action.value,
                "reward": t.reward,
                "timestamp": t.timestamp.isoformat(),
                "metadata": t.metadata
            }
            for t in transitions
        ],
        "batch_size": len(transitions),
        "created_at": datetime.utcnow().isoformat()
    }