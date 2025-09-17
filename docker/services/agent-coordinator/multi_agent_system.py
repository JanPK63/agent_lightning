"""
Multi-Agent System Implementation for Agent Lightning
Orchestrates multiple specialized agents with MARL coordination
Based on Agent Lightning's unified data interface and hierarchical RL
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np
from mdp_agents import MDPAgent, MDPTransition, AgentState, AgentAction, create_transition_batch


class MultiAgentSystem:
    """
    Multi-agent system with specialized roles following Agent Lightning architecture
    Supports cooperative and competitive agent interactions
    """
    
    def __init__(self, coordination_type: str = "cooperative"):
        """
        Initialize multi-agent system
        
        Args:
            coordination_type: "cooperative" or "competitive"
        """
        self.coordination_type = coordination_type
        
        # Create specialized agents with different roles and models
        self.agents = self._initialize_agents()
        
        # Shared state for multi-agent coordination
        self.shared_state = {
            "task": None,
            "context": {},
            "results": {},
            "semantic_variables": {},
            "communication_buffer": []
        }
        
        # Multi-agent RL components
        self.shared_reward = (coordination_type == "cooperative")
        self.communication_enabled = True
        self.orchestration_mode = "sequential"  # or "parallel"
        
        # Trajectory storage for training
        self.trajectories = []
        self.episode_count = 0
        
    def _initialize_agents(self) -> Dict[str, MDPAgent]:
        """
        Initialize specialized agents with different capabilities
        Following Agent Lightning's multi-agent examples
        """
        agents = {
            "researcher": MDPAgent(
                role="Research Specialist",
                model="gpt-4o",
                hierarchy_level="high",
                agent_type="researcher"
            ),
            "writer": MDPAgent(
                role="Content Writer",
                model="gpt-4o",
                hierarchy_level="low",
                agent_type="writer"
            ),
            "reviewer": MDPAgent(
                role="Quality Reviewer",
                model="claude-3-opus" if "claude" in ["claude-3-opus"] else "gpt-4o",
                hierarchy_level="high",
                agent_type="reviewer"
            ),
            "optimizer": MDPAgent(
                role="Performance Optimizer",
                model="gpt-3.5-turbo",
                hierarchy_level="low",
                agent_type="optimizer"
            )
        }
        
        print(f"✅ Initialized {len(agents)} specialized agents:")
        for name, agent in agents.items():
            print(f"   - {name}: {agent.role} ({agent.model})")
        
        return agents
    
    def orchestrate(self, task: Dict) -> Tuple[Dict, List[MDPTransition]]:
        """
        Orchestrate multi-agent collaboration for a task
        Returns final result and all transitions for RL training
        
        This follows Agent Lightning's approach where:
        - Each agent generates transitions
        - Transitions are collected for training
        - Rewards can be shared or individual
        """
        # Reset shared state for new task
        self.shared_state = {
            "task": task,
            "context": task.get("context", {}),
            "results": {},
            "semantic_variables": task.get("semantic_variables", {}),
            "communication_buffer": []
        }
        
        transitions = []
        
        # Phase 1: Research (High-level planning)
        print("\n📚 Phase 1: Research")
        research_transitions = self._execute_research_phase(task)
        transitions.extend(research_transitions)
        
        # Phase 2: Content Creation (Low-level execution)
        print("\n✍️ Phase 2: Content Creation")
        writing_transitions = self._execute_writing_phase()
        transitions.extend(writing_transitions)
        
        # Phase 3: Review and Optimization
        print("\n🔍 Phase 3: Review and Optimization")
        review_transitions = self._execute_review_phase()
        transitions.extend(review_transitions)
        
        # Phase 4: Final Optimization (if needed)
        if self._needs_optimization():
            print("\n⚡ Phase 4: Optimization")
            optimization_transitions = self._execute_optimization_phase()
            transitions.extend(optimization_transitions)
        
        # Calculate final shared reward if cooperative
        if self.shared_reward:
            final_reward = self._calculate_shared_reward()
            print(f"\n🎯 Shared Reward: {final_reward:.3f}")
            
            # Update all transitions with shared reward component
            for transition in transitions:
                transition.reward = 0.7 * transition.reward + 0.3 * final_reward
        
        # Store episode for analysis
        self.trajectories.append({
            "task": task,
            "transitions": transitions,
            "results": self.shared_state["results"],
            "episode_id": self.episode_count
        })
        self.episode_count += 1
        
        return self.shared_state["results"], transitions
    
    def _execute_research_phase(self, task: Dict) -> List[MDPTransition]:
        """Execute research phase with researcher agent"""
        transitions = []
        researcher = self.agents["researcher"]
        
        # Create execution state for researcher
        execution_state = {
            "input": f"Research: {task.get('query', task.get('input', ''))}",
            "context": self.shared_state["context"],
            "semantic_variables": self.shared_state["semantic_variables"]
        }
        
        # Researcher observes and acts
        state = researcher.observe(execution_state)
        action, transition = researcher.act(state)
        
        # Store research results
        self.shared_state["results"]["research"] = {
            "content": action.content,
            "confidence": action.confidence,
            "timestamp": time.time()
        }
        
        # Update semantic variables
        self.shared_state["semantic_variables"]["research_complete"] = True
        
        # Add to communication buffer for other agents
        if self.communication_enabled:
            self._broadcast_to_agents(
                sender="researcher",
                message=f"Research findings: {action.content[:200]}..."
            )
        
        transitions.append(transition)
        print(f"   ✓ Research completed (confidence: {action.confidence:.2f})")
        
        return transitions
    
    def _execute_writing_phase(self) -> List[MDPTransition]:
        """Execute writing phase based on research"""
        transitions = []
        writer = self.agents["writer"]
        
        # Build context from research results
        research_content = self.shared_state["results"].get("research", {}).get("content", "")
        
        execution_state = {
            "input": f"Write content based on research: {research_content[:500]}...\nOriginal task: {self.shared_state['task'].get('query', '')}",
            "context": {
                **self.shared_state["context"],
                "research_available": True
            },
            "semantic_variables": self.shared_state["semantic_variables"]
        }
        
        # Writer observes and acts
        state = writer.observe(execution_state)
        action, transition = writer.act(state)
        
        # Store writing results
        self.shared_state["results"]["content"] = {
            "text": action.content,
            "confidence": action.confidence,
            "timestamp": time.time()
        }
        
        # Update semantic variables
        self.shared_state["semantic_variables"]["content_created"] = True
        
        # Broadcast to other agents
        if self.communication_enabled:
            self._broadcast_to_agents(
                sender="writer",
                message=f"Content created: {len(action.content)} characters"
            )
        
        transitions.append(transition)
        print(f"   ✓ Content written (confidence: {action.confidence:.2f})")
        
        return transitions
    
    def _execute_review_phase(self) -> List[MDPTransition]:
        """Execute review phase to evaluate quality"""
        transitions = []
        reviewer = self.agents["reviewer"]
        
        # Prepare content for review
        content = self.shared_state["results"].get("content", {}).get("text", "")
        research = self.shared_state["results"].get("research", {}).get("content", "")
        
        execution_state = {
            "input": f"Review this content:\n{content}\n\nBased on research:\n{research[:300]}...",
            "context": {
                **self.shared_state["context"],
                "phase": "review"
            },
            "semantic_variables": self.shared_state["semantic_variables"]
        }
        
        # Reviewer observes and acts
        state = reviewer.observe(execution_state)
        action, transition = reviewer.act(state)
        
        # Store review results
        self.shared_state["results"]["review"] = {
            "feedback": action.content,
            "confidence": action.confidence,
            "timestamp": time.time(),
            "quality_score": self._extract_quality_score(action.content)
        }
        
        # Update semantic variables
        self.shared_state["semantic_variables"]["review_complete"] = True
        
        transitions.append(transition)
        print(f"   ✓ Review completed (confidence: {action.confidence:.2f})")
        
        return transitions
    
    def _execute_optimization_phase(self) -> List[MDPTransition]:
        """Execute optimization phase if needed"""
        transitions = []
        optimizer = self.agents["optimizer"]
        
        # Get review feedback
        review = self.shared_state["results"].get("review", {})
        content = self.shared_state["results"].get("content", {}).get("text", "")
        
        execution_state = {
            "input": f"Optimize based on feedback:\n{review.get('feedback', '')}\n\nContent:\n{content[:500]}...",
            "context": self.shared_state["context"],
            "semantic_variables": self.shared_state["semantic_variables"]
        }
        
        # Optimizer observes and acts
        state = optimizer.observe(execution_state)
        action, transition = optimizer.act(state)
        
        # Store optimization results
        self.shared_state["results"]["optimized_content"] = {
            "text": action.content,
            "confidence": action.confidence,
            "timestamp": time.time()
        }
        
        transitions.append(transition)
        print(f"   ✓ Optimization completed (confidence: {action.confidence:.2f})")
        
        return transitions
    
    def _needs_optimization(self) -> bool:
        """Determine if optimization phase is needed"""
        review = self.shared_state["results"].get("review", {})
        quality_score = review.get("quality_score", 1.0)
        
        # Optimize if quality score is below threshold
        return quality_score < 0.8
    
    def _extract_quality_score(self, review_content: str) -> float:
        """Extract quality score from review content"""
        # Simple heuristic - in practice would use more sophisticated analysis
        positive_words = ["excellent", "good", "great", "perfect", "accurate"]
        negative_words = ["poor", "incorrect", "wrong", "bad", "unclear"]
        
        content_lower = review_content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count + negative_count == 0:
            return 0.7  # Neutral
        
        score = positive_count / (positive_count + negative_count)
        return score
    
    def _calculate_shared_reward(self) -> float:
        """
        Calculate shared reward for cooperative multi-agent system
        Following Agent Lightning's MARL approach
        """
        reward = 0.0
        
        # Task completion bonus
        if all(key in self.shared_state["results"] 
               for key in ["research", "content", "review"]):
            reward += 0.3
        
        # Quality bonus from review
        review = self.shared_state["results"].get("review", {})
        quality_score = review.get("quality_score", 0.5)
        reward += quality_score * 0.3
        
        # Collaboration bonus (all agents participated)
        agent_participation = len(self.shared_state["results"])
        if agent_participation >= 3:
            reward += 0.2
        
        # Efficiency penalty for verbosity
        total_content_length = sum(
            len(str(result.get("content", result.get("text", ""))))
            for result in self.shared_state["results"].values()
        )
        if total_content_length > 5000:
            reward -= 0.1
        
        # Communication bonus
        if len(self.shared_state["communication_buffer"]) > 2:
            reward += 0.1
        
        return np.clip(reward, 0.0, 1.0)
    
    def _broadcast_to_agents(self, sender: str, message: str):
        """
        Broadcast message from one agent to others
        Enables inter-agent communication
        """
        communication_entry = {
            "sender": sender,
            "message": message,
            "timestamp": time.time()
        }
        
        self.shared_state["communication_buffer"].append(communication_entry)
        
        # In a more sophisticated system, other agents would process these messages
        # and potentially adjust their behavior
    
    def get_training_batch(self, batch_size: int = 32) -> Dict:
        """
        Get a batch of transitions for training
        Following Agent Lightning's batch processing approach
        """
        if not self.trajectories:
            return None
        
        # Collect all transitions
        all_transitions = []
        for trajectory in self.trajectories:
            all_transitions.extend(trajectory["transitions"])
        
        # Sample batch
        if len(all_transitions) < batch_size:
            batch_transitions = all_transitions
        else:
            indices = np.random.choice(len(all_transitions), batch_size, replace=False)
            batch_transitions = [all_transitions[i] for i in indices]
        
        return create_transition_batch(batch_transitions)
    
    def reset(self):
        """Reset the multi-agent system for new episode"""
        self.shared_state = {
            "task": None,
            "context": {},
            "results": {},
            "semantic_variables": {},
            "communication_buffer": []
        }
        
        # Reset each agent's episode memory
        for agent in self.agents.values():
            agent.reset_episode()


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Multi-Agent System for Agent Lightning")
    print("=" * 60)
    
    # Create multi-agent system
    mas = MultiAgentSystem(coordination_type="cooperative")
    
    # Define a sample task
    task = {
        "query": "Explain the applications of quantum computing in drug discovery",
        "context": {
            "domain": "science",
            "audience": "technical",
            "length": "comprehensive"
        },
        "semantic_variables": {
            "task_type": "explanation",
            "complexity": "high"
        }
    }
    
    print(f"\n📋 Task: {task['query']}")
    print("-" * 60)
    
    # Orchestrate multi-agent collaboration
    results, transitions = mas.orchestrate(task)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    for agent_name, result in results.items():
        print(f"\n{agent_name.upper()}:")
        if "content" in result:
            print(f"  Content: {result['content'][:150]}...")
        elif "text" in result:
            print(f"  Text: {result['text'][:150]}...")
        elif "feedback" in result:
            print(f"  Feedback: {result['feedback'][:150]}...")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
    
    print(f"\n📈 Collected {len(transitions)} transitions for RL training")
    
    # Get training batch
    batch = mas.get_training_batch(batch_size=16)
    if batch:
        print(f"📦 Created training batch with {len(batch['states'])} samples")
    
    print("\n✅ Multi-Agent System test complete!")
    print("Ready for MARL optimization with Agent Lightning")