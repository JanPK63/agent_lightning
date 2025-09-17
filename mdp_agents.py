"""
MDP Agent Definitions for Agent Lightning Framework
Implements agents as Markov Decision Processes with states, actions, and rewards
Based on the Agent Lightning paper architecture
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
import os
from openai import OpenAI
import anthropic

@dataclass
class AgentState:
    """
    State representation for MDP following Agent Lightning's unified data interface
    State is a snapshot of agent execution including semantic variables
    """
    current_input: str  # Current task or query
    role: str  # Agent's role (e.g., "Research Specialist", "Writer")
    history: List[Dict] = None  # Previous interactions
    context: Dict = None  # Additional context
    hierarchy_level: str = "low"  # "high" for planning, "low" for execution
    timestamp: float = None
    memory_context: List[Dict] = None  # Long-term memory
    semantic_variables: Dict = None  # Key variables that evolve during execution
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for serialization"""
        return {
            "current_input": self.current_input,
            "role": self.role,
            "history": self.history or [],
            "context": self.context or {},
            "hierarchy_level": self.hierarchy_level,
            "timestamp": self.timestamp or time.time(),
            "memory_context": self.memory_context or [],
            "semantic_variables": self.semantic_variables or {}
        }

@dataclass
class AgentAction:
    """
    Action representation - entire token sequence from single LLM invocation
    Following Agent Lightning's approach where one action = one LLM call output
    """
    content: str  # The generated response/action
    action_type: str  # "plan", "execute", "communicate", "review"
    confidence: float  # Confidence score for the action
    reasoning: str = None  # Optional reasoning for the action
    tokens: List[str] = None  # Token-level breakdown if needed
    
    def to_dict(self) -> Dict:
        """Convert action to dictionary"""
        return {
            "content": self.content,
            "action_type": self.action_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "tokens": self.tokens
        }

@dataclass
class MDPTransition:
    """
    Transition representation for RL training
    Each transition contains (state, action, reward, next_state)
    """
    state: Dict
    action: Dict
    reward: float
    next_state: Dict
    done: bool = False
    info: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert transition to dictionary for training"""
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "info": self.info or {}
        }

class MDPAgent:
    """
    Agent modeled as Partially Observable MDP following Agent Lightning framework
    Handles state transitions, action generation, and reward computation
    """
    
    def __init__(self, 
                 role: str, 
                 model: str = "gpt-4o",
                 hierarchy_level: str = "low",
                 agent_type: str = "default"):
        """
        Initialize MDP Agent
        
        Args:
            role: Agent's role defining its behavior
            model: LLM model to use (gpt-4o, claude-3-opus, etc.)
            hierarchy_level: "high" for planning, "low" for execution
            agent_type: Type of agent (researcher, writer, reviewer, optimizer)
        """
        self.role = role
        self.model = model
        self.hierarchy_level = hierarchy_level
        self.agent_type = agent_type
        
        # Memory components
        self.memory = []  # Long-term memory
        self.episode_memory = []  # Short-term episode memory
        self.semantic_memory = {}  # Semantic knowledge storage
        
        # Initialize LLM clients
        self._init_llm_clients()
        
        # RL components for value estimation
        self.state_values = {}  # V(s) values
        self.q_values = {}  # Q(s,a) values
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        
    def _init_llm_clients(self):
        """Initialize LLM clients based on available API keys"""
        self.openai_client = None
        self.anthropic_client = None
        
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
        if os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
    
    def observe(self, execution_state: Dict) -> AgentState:
        """
        Create observation (state) from current execution context
        Maps execution state to agent's observation space
        """
        # Extract semantic variables from execution state
        semantic_vars = execution_state.get("semantic_variables", {})
        
        # Build agent state from execution context
        state = AgentState(
            current_input=execution_state.get("input", ""),
            role=self.role,
            history=self.episode_memory[-5:] if self.episode_memory else [],
            context=execution_state.get("context", {}),
            hierarchy_level=self.hierarchy_level,
            timestamp=time.time(),
            memory_context=self._retrieve_relevant_memories(
                execution_state.get("input", "")
            ),
            semantic_variables=semantic_vars
        )
        
        return state
    
    def act(self, state: AgentState) -> Tuple[AgentAction, MDPTransition]:
        """
        Generate action based on current state
        Returns both the action and the transition for RL training
        
        This follows Agent Lightning's approach where:
        - Input (observation) is part of the state
        - Output (entire LLM response) is the action
        - Transition captures the state change
        """
        # Build prompt from state
        prompt = self._build_context_aware_prompt(state)
        
        # Generate action using appropriate LLM
        action_content, confidence = self._generate_action(prompt, state)
        
        # Classify action type
        action_type = self._classify_action_type(action_content, state)
        
        # Create action object
        action = AgentAction(
            content=action_content,
            action_type=action_type,
            confidence=confidence,
            reasoning=f"{self.role} at {state.hierarchy_level} level"
        )
        
        # Create next state (simplified - in practice would involve environment)
        next_state = self._compute_next_state(state, action)
        
        # Compute immediate reward (can be sparse or shaped)
        reward = self._compute_reward(state, action, next_state)
        
        # Create transition for RL training
        transition = MDPTransition(
            state=state.to_dict(),
            action=action.to_dict(),
            reward=reward,
            next_state=next_state.to_dict(),
            done=self._is_terminal_state(next_state),
            info={
                "agent_role": self.role,
                "hierarchy_level": self.hierarchy_level,
                "timestamp": time.time()
            }
        )
        
        # Update episode memory
        self._update_memory(state, action, reward)
        
        return action, transition
    
    def _build_context_aware_prompt(self, state: AgentState) -> str:
        """
        Build context-aware prompt incorporating state information
        Following Agent Lightning's flexible context construction
        """
        prompt_parts = []
        
        # Role and hierarchy instructions
        if state.hierarchy_level == "high":
            prompt_parts.append(
                f"You are a {self.role} working on high-level planning. "
                f"Break down the task into actionable steps."
            )
        else:
            prompt_parts.append(
                f"You are a {self.role} executing specific tasks. "
                f"Focus on concrete implementation."
            )
        
        # Add memory context if available
        if state.memory_context:
            prompt_parts.append("\nRelevant past experience:")
            for mem in state.memory_context[:3]:
                if isinstance(mem, dict):
                    summary = mem.get("summary", "")[:100]
                    prompt_parts.append(f"- {summary}")
        
        # Add current context
        if state.context:
            prompt_parts.append(f"\nContext: {state.context}")
        
        # Add task
        prompt_parts.append(f"\nTask: {state.current_input}")
        
        # Add specific instructions based on agent type
        if self.agent_type == "researcher":
            prompt_parts.append("\nProvide thorough research and analysis.")
        elif self.agent_type == "writer":
            prompt_parts.append("\nGenerate clear, well-structured content.")
        elif self.agent_type == "reviewer":
            prompt_parts.append("\nReview and provide constructive feedback.")
        
        return "\n".join(prompt_parts)
    
    def _generate_action(self, prompt: str, state: AgentState) -> Tuple[str, float]:
        """
        Generate action using LLM based on model type
        Returns action content and confidence score
        """
        if "gpt" in self.model and self.openai_client:
            return self._generate_openai_action(prompt, state)
        elif "claude" in self.model and self.anthropic_client:
            return self._generate_anthropic_action(prompt, state)
        else:
            # Fallback for testing
            return self._generate_mock_action(prompt, state)
    
    def _generate_openai_action(self, prompt: str, state: AgentState) -> Tuple[str, float]:
        """Generate action using OpenAI model"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a {self.role}. Be precise and actionable."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            confidence = self._estimate_confidence(content, state)
            
            return content, confidence
            
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return "Error generating response", 0.1
    
    def _generate_anthropic_action(self, prompt: str, state: AgentState) -> Tuple[str, float]:
        """Generate action using Anthropic model"""
        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                system=f"You are a {self.role}. Provide detailed analysis.",
                max_tokens=500,
                temperature=0.7
            )
            
            content = response.content[0].text
            confidence = self._estimate_confidence(content, state)
            
            return content, confidence
            
        except Exception as e:
            print(f"Anthropic generation error: {e}")
            return "Error generating response", 0.1
    
    def _generate_mock_action(self, prompt: str, state: AgentState) -> Tuple[str, float]:
        """Mock action generation for testing"""
        action = f"Mock {self.role} action for: {state.current_input[:50]}"
        confidence = 0.5
        return action, confidence
    
    def _estimate_confidence(self, action_content: str, state: AgentState) -> float:
        """
        Estimate confidence score for generated action
        Used for credit assignment in hierarchical RL
        """
        confidence = 0.5  # Base confidence
        
        # Detailed responses get higher confidence
        if len(action_content) > 200:
            confidence += 0.15
        
        # Structured responses indicate higher quality
        if any(marker in action_content.lower() 
               for marker in ["first", "second", "step", "1.", "2."]):
            confidence += 0.2
        
        # Context utilization
        if state.memory_context:
            confidence += 0.1
        
        # Hierarchy level alignment
        if state.hierarchy_level == "high" and "plan" in action_content.lower():
            confidence += 0.05
        elif state.hierarchy_level == "low" and any(
            word in action_content.lower() 
            for word in ["implement", "execute", "create"]
        ):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _classify_action_type(self, action_content: str, state: AgentState) -> str:
        """Classify the type of action based on content and state"""
        content_lower = action_content.lower()
        
        if state.hierarchy_level == "high":
            if any(word in content_lower for word in ["plan", "strategy", "steps"]):
                return "plan"
            elif any(word in content_lower for word in ["review", "evaluate"]):
                return "review"
        else:
            if any(word in content_lower for word in ["implement", "create", "build"]):
                return "execute"
            elif any(word in content_lower for word in ["communicate", "coordinate"]):
                return "communicate"
        
        return "execute"  # Default
    
    def _compute_next_state(self, state: AgentState, action: AgentAction) -> AgentState:
        """
        Compute next state after action execution
        In practice, this would involve environment feedback
        """
        next_state = AgentState(
            current_input=state.current_input,  # Could be updated based on action
            role=self.role,
            history=state.history + [{"action": action.to_dict()}] if state.history else [{"action": action.to_dict()}],
            context={**state.context, "last_action": action.action_type},
            hierarchy_level=state.hierarchy_level,
            timestamp=time.time(),
            memory_context=state.memory_context,
            semantic_variables=state.semantic_variables
        )
        
        return next_state
    
    def _compute_reward(self, state: AgentState, action: AgentAction, 
                       next_state: AgentState) -> float:
        """
        Compute immediate reward for the transition
        Can be sparse (only at episode end) or shaped (intermediate rewards)
        """
        reward = 0.0
        
        # Confidence-based reward shaping
        reward += action.confidence * 0.1
        
        # Action type rewards
        if action.action_type == "plan" and state.hierarchy_level == "high":
            reward += 0.1  # Appropriate action for context
        elif action.action_type == "execute" and state.hierarchy_level == "low":
            reward += 0.1
        
        # Length penalty for overly verbose responses
        if len(action.content) > 1000:
            reward -= 0.05
        
        return reward
    
    def _is_terminal_state(self, state: AgentState) -> bool:
        """Check if state is terminal (episode complete)"""
        # Simplified - in practice would check task completion
        return False
    
    def _retrieve_relevant_memories(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve k most relevant memories for current context
        Simple implementation - could use embeddings for similarity
        """
        if not self.memory:
            return []
        
        # Simple recency-based retrieval
        return self.memory[-k:]
    
    def _update_memory(self, state: AgentState, action: AgentAction, reward: float):
        """Update agent's memory with new experience"""
        memory_entry = {
            "state": state.to_dict(),
            "action": action.to_dict(),
            "reward": reward,
            "timestamp": time.time(),
            "summary": f"{action.action_type}: {action.content[:100]}"
        }
        
        # Update episode memory
        self.episode_memory.append(memory_entry)
        
        # Store in long-term memory if significant
        if reward > 0.5 or action.confidence > 0.7:
            self.memory.append(memory_entry)
            
            # Limit memory size
            if len(self.memory) > 1000:
                self.memory = self.memory[-1000:]
    
    def update_q_values(self, transition: MDPTransition):
        """
        Update Q-values using the transition (for RL training)
        Implements basic Q-learning update
        """
        state_key = self._state_to_key(transition.state)
        action_key = transition.action.get("action_type", "unknown")
        sa_key = f"{state_key}_{action_key}"
        
        # Get current Q-value
        current_q = self.q_values.get(sa_key, 0.0)
        
        # Get max Q-value for next state
        next_state_key = self._state_to_key(transition.next_state)
        next_q_values = [
            v for k, v in self.q_values.items() 
            if k.startswith(next_state_key)
        ]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        if transition.done:
            target = transition.reward
        else:
            target = transition.reward + self.discount_factor * max_next_q
        
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_values[sa_key] = new_q
        
        # Update state value
        state_q_values = [
            v for k, v in self.q_values.items() 
            if k.startswith(state_key)
        ]
        if state_q_values:
            self.state_values[state_key] = max(state_q_values)
    
    def _state_to_key(self, state: Dict) -> str:
        """Convert state to string key for value storage"""
        role = state.get("role", "unknown")
        level = state.get("hierarchy_level", "low")
        task = state.get("current_input", "")[:30]
        return f"{role}_{level}_{task}"
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_memory = []


def create_transition_batch(transitions: List[MDPTransition]) -> Dict:
    """
    Create a batch of transitions for training
    Following Agent Lightning's approach to batch processing
    """
    batch = {
        "states": [t.state for t in transitions],
        "actions": [t.action for t in transitions],
        "rewards": [t.reward for t in transitions],
        "next_states": [t.next_state for t in transitions],
        "dones": [t.done for t in transitions],
        "infos": [t.info for t in transitions]
    }
    return batch


# Example usage and testing
if __name__ == "__main__":
    print("Testing MDP Agent Implementation for Agent Lightning")
    print("-" * 50)
    
    # Create a research agent
    researcher = MDPAgent(
        role="Research Specialist",
        model="gpt-4o",
        hierarchy_level="high",
        agent_type="researcher"
    )
    
    # Create execution state (what the agent observes)
    execution_state = {
        "input": "Research the latest advances in quantum computing applications",
        "context": {"domain": "technology", "depth": "comprehensive"},
        "semantic_variables": {
            "task_type": "research",
            "status": "initiated"
        }
    }
    
    # Agent observes and creates state
    state = researcher.observe(execution_state)
    print(f"Agent Role: {researcher.role}")
    print(f"Hierarchy Level: {state.hierarchy_level}")
    print(f"Current Input: {state.current_input}\n")
    
    # Agent acts based on state
    action, transition = researcher.act(state)
    print(f"Action Type: {action.action_type}")
    print(f"Action Confidence: {action.confidence:.2f}")
    print(f"Action Content: {action.content[:200]}...")
    print(f"Immediate Reward: {transition.reward:.2f}\n")
    
    # Update Q-values with transition
    researcher.update_q_values(transition)
    
    # Create a batch for training
    transitions = [transition]  # In practice, would have many
    batch = create_transition_batch(transitions)
    print(f"Created training batch with {len(transitions)} transition(s)")
    
    print("\n✅ MDP Agent implementation complete!")
    print("Ready for integration with Agent Lightning training pipeline")