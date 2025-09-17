"""
Reward Functions for Agent Lightning
Comprehensive scoring system for different task types and agent behaviors
Following Agent Lightning's approach to reward shaping and credit assignment
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import re
import json
from difflib import SequenceMatcher
from sklearn.metrics import f1_score, precision_score, recall_score
import ast
import math


class RewardType(Enum):
    """Types of rewards"""
    SPARSE = "sparse"  # Only at episode end
    DENSE = "dense"    # At each step
    SHAPED = "shaped"  # With intermediate shaping
    HIERARCHICAL = "hierarchical"  # Different for high/low level


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    reward_type: RewardType = RewardType.SHAPED
    success_reward: float = 1.0
    failure_penalty: float = -0.5
    step_penalty: float = -0.01
    efficiency_bonus: float = 0.1
    quality_weight: float = 0.7
    speed_weight: float = 0.3
    cooperation_bonus: float = 0.2
    exploration_bonus: float = 0.05


class RewardCalculator:
    """
    Main reward calculator for Agent Lightning
    Implements various reward functions for different task types
    """
    
    def __init__(self, config: RewardConfig = None):
        """
        Initialize reward calculator
        
        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()
        
        # Task-specific reward functions
        self.task_reward_functions = {
            "math": self.calculate_math_reward,
            "code": self.calculate_code_reward,
            "text": self.calculate_text_reward,
            "qa": self.calculate_qa_reward,
            "rag": self.calculate_rag_reward,
            "tool_use": self.calculate_tool_use_reward,
            "multi_agent": self.calculate_multi_agent_reward,
            "sql": self.calculate_sql_reward
        }
        
        # Quality metrics
        self.quality_metrics = {
            "accuracy": self.calculate_accuracy,
            "completeness": self.calculate_completeness,
            "coherence": self.calculate_coherence,
            "efficiency": self.calculate_efficiency
        }
        
        print(f"💰 Reward Calculator initialized with {self.config.reward_type.value} rewards")
    
    def calculate_reward(self,
                        action: str,
                        ground_truth: Optional[str] = None,
                        task_type: str = "general",
                        metadata: Dict[str, Any] = None) -> float:
        """
        Main reward calculation function
        
        Args:
            action: Agent's action/output
            ground_truth: Expected result
            task_type: Type of task
            metadata: Additional context
            
        Returns:
            Reward value
        """
        metadata = metadata or {}
        
        # Get task-specific reward function
        reward_fn = self.task_reward_functions.get(
            task_type,
            self.calculate_general_reward
        )
        
        # Calculate base reward
        base_reward = reward_fn(action, ground_truth, metadata)
        
        # Apply reward shaping if configured
        if self.config.reward_type == RewardType.SHAPED:
            shaped_reward = self.apply_reward_shaping(base_reward, action, metadata)
        else:
            shaped_reward = base_reward
        
        # Apply hierarchical scaling if needed
        if self.config.reward_type == RewardType.HIERARCHICAL:
            hierarchy_level = metadata.get("hierarchy_level", "low")
            if hierarchy_level == "high":
                shaped_reward *= 1.5  # Higher rewards for high-level decisions
        
        # Clip reward to reasonable range
        return np.clip(shaped_reward, -1.0, 1.0)
    
    def calculate_math_reward(self, action: str, ground_truth: str, 
                             metadata: Dict) -> float:
        """Calculate reward for mathematical tasks"""
        if not ground_truth:
            return 0.0
        
        # Extract numerical answer
        action_number = self.extract_number(action)
        truth_number = self.extract_number(ground_truth)
        
        if action_number is None or truth_number is None:
            return self.config.failure_penalty
        
        # Exact match gets full reward
        if abs(action_number - truth_number) < 1e-6:
            reward = self.config.success_reward
        else:
            # Partial credit based on relative error
            error = abs(action_number - truth_number) / (abs(truth_number) + 1e-6)
            reward = max(0, self.config.success_reward * (1 - error))
        
        # Bonus for showing work
        if self.contains_reasoning(action):
            reward += self.config.efficiency_bonus
        
        return reward
    
    def calculate_code_reward(self, action: str, ground_truth: str,
                             metadata: Dict) -> float:
        """Calculate reward for code generation tasks"""
        reward = 0.0
        
        # Check syntax validity
        if self.is_valid_code(action):
            reward += 0.3
        else:
            return self.config.failure_penalty
        
        # Check functional correctness if test cases provided
        test_cases = metadata.get("test_cases", [])
        if test_cases:
            passed = self.run_test_cases(action, test_cases)
            reward += 0.5 * (passed / len(test_cases))
        
        # Code quality metrics
        if ground_truth:
            # Structural similarity
            similarity = self.code_similarity(action, ground_truth)
            reward += 0.2 * similarity
        
        # Efficiency bonus for concise code
        if len(action) < len(ground_truth) * 1.5:
            reward += self.config.efficiency_bonus
        
        return reward
    
    def calculate_text_reward(self, action: str, ground_truth: str,
                             metadata: Dict) -> float:
        """Calculate reward for text generation tasks"""
        if not ground_truth:
            # Use quality metrics only
            return self.calculate_text_quality(action, metadata)
        
        # Semantic similarity
        similarity = SequenceMatcher(None, action.lower(), ground_truth.lower()).ratio()
        
        # Quality metrics from metadata
        quality_metrics = metadata.get("quality_metrics", {})
        completeness = quality_metrics.get("completeness", similarity)
        accuracy = quality_metrics.get("accuracy", similarity)
        coherence = quality_metrics.get("coherence", 0.8)
        
        # Weighted combination
        reward = (
            0.3 * similarity +
            0.3 * completeness +
            0.2 * accuracy +
            0.2 * coherence
        )
        
        # Length penalty/bonus
        ideal_length = metadata.get("ideal_length", 200)
        length_ratio = len(action) / ideal_length
        if 0.8 <= length_ratio <= 1.2:
            reward += self.config.efficiency_bonus
        elif length_ratio > 2.0 or length_ratio < 0.3:
            reward -= 0.1
        
        return reward
    
    def calculate_qa_reward(self, action: str, ground_truth: str,
                           metadata: Dict) -> float:
        """Calculate reward for question answering tasks"""
        if not ground_truth:
            return 0.0
        
        # Exact match
        if action.strip().lower() == ground_truth.strip().lower():
            return self.config.success_reward
        
        # F1 score for partial credit
        action_tokens = set(action.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not truth_tokens:
            return 0.0
        
        precision = len(action_tokens & truth_tokens) / len(action_tokens) if action_tokens else 0
        recall = len(action_tokens & truth_tokens) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1 * self.config.success_reward
    
    def calculate_rag_reward(self, action: str, ground_truth: str,
                            metadata: Dict) -> float:
        """Calculate reward for retrieval-augmented generation"""
        # Base QA reward
        qa_reward = self.calculate_qa_reward(action, ground_truth, metadata)
        
        # Retrieval quality bonus
        retrieval_quality = metadata.get("retrieval_quality", {})
        relevance = retrieval_quality.get("relevance", 0.5)
        coverage = retrieval_quality.get("coverage", 0.5)
        
        retrieval_bonus = 0.2 * (relevance + coverage) / 2
        
        # Grounding bonus - reward for using retrieved information
        if metadata.get("used_retrieval", False):
            grounding_bonus = 0.1
        else:
            grounding_bonus = -0.1  # Penalty for not using retrieval
        
        return qa_reward + retrieval_bonus + grounding_bonus
    
    def calculate_tool_use_reward(self, action: str, ground_truth: str,
                                 metadata: Dict) -> float:
        """Calculate reward for tool usage tasks"""
        reward = 0.0
        
        # Correct tool selection
        expected_tool = metadata.get("expected_tool")
        used_tool = metadata.get("used_tool")
        
        if expected_tool and used_tool:
            if expected_tool == used_tool:
                reward += 0.3
            else:
                reward -= 0.2
        
        # Correct parameters
        if metadata.get("correct_parameters", False):
            reward += 0.3
        
        # Successful execution
        if metadata.get("execution_success", False):
            reward += 0.4
        else:
            reward -= 0.2
        
        return reward
    
    def calculate_multi_agent_reward(self, action: str, ground_truth: str,
                                    metadata: Dict) -> float:
        """Calculate reward for multi-agent coordination"""
        # Individual agent performance
        agent_rewards = metadata.get("agent_rewards", {})
        if agent_rewards:
            individual_reward = np.mean(list(agent_rewards.values()))
        else:
            individual_reward = self.calculate_general_reward(action, ground_truth, metadata)
        
        # Coordination bonus
        coordination_metrics = metadata.get("coordination", {})
        communication = coordination_metrics.get("communication_quality", 0.5)
        synchronization = coordination_metrics.get("synchronization", 0.5)
        consensus = coordination_metrics.get("consensus_level", 0.5)
        
        coordination_bonus = self.config.cooperation_bonus * np.mean([
            communication, synchronization, consensus
        ])
        
        # Efficiency bonus for parallel execution
        if metadata.get("parallel_execution", False):
            efficiency_bonus = self.config.efficiency_bonus
        else:
            efficiency_bonus = 0
        
        return individual_reward + coordination_bonus + efficiency_bonus
    
    def calculate_sql_reward(self, action: str, ground_truth: str,
                            metadata: Dict) -> float:
        """Calculate reward for SQL generation tasks"""
        reward = 0.0
        
        # Syntax validity
        if self.is_valid_sql(action):
            reward += 0.2
        else:
            return self.config.failure_penalty
        
        # Execution success
        if metadata.get("execution_success", False):
            reward += 0.3
        
        # Result correctness
        if metadata.get("correct_results", False):
            reward += 0.5
        elif metadata.get("partial_results", False):
            reward += 0.25
        
        # Query efficiency (simplified)
        if ground_truth and len(action) <= len(ground_truth) * 1.2:
            reward += self.config.efficiency_bonus
        
        return reward
    
    def calculate_general_reward(self, action: str, ground_truth: str,
                                metadata: Dict) -> float:
        """General reward calculation fallback"""
        if not ground_truth:
            # Basic quality checks
            if len(action) > 10:
                return 0.3
            return 0.1
        
        # Simple similarity
        similarity = SequenceMatcher(None, action, ground_truth).ratio()
        return similarity * self.config.success_reward
    
    def apply_reward_shaping(self, base_reward: float, action: str,
                            metadata: Dict) -> float:
        """Apply reward shaping for better learning"""
        shaped_reward = base_reward
        
        # Step penalty to encourage efficiency
        steps = metadata.get("num_steps", 0)
        shaped_reward += self.config.step_penalty * steps
        
        # Exploration bonus
        if metadata.get("novel_action", False):
            shaped_reward += self.config.exploration_bonus
        
        # Confidence-based shaping
        confidence = metadata.get("confidence", 0.5)
        if confidence > 0.8:
            shaped_reward *= 1.1
        elif confidence < 0.3:
            shaped_reward *= 0.9
        
        # Time-based shaping
        time_taken = metadata.get("time_taken", 0)
        max_time = metadata.get("max_time", 60)
        if time_taken > 0 and max_time > 0:
            time_bonus = self.config.efficiency_bonus * (1 - time_taken / max_time)
            shaped_reward += max(0, time_bonus)
        
        return shaped_reward
    
    # Utility functions
    def extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text"""
        # Look for patterns like "x = 5", "answer: 10", etc.
        patterns = [
            r'(?:x\s*=\s*|answer\s*(?:is|:)\s*)([+-]?\d+\.?\d*)',
            r'([+-]?\d+\.?\d*)(?:\s*(?:is|are)\s+the\s+answer)',
            r'^\s*([+-]?\d+\.?\d*)\s*$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Try to find any number
        numbers = re.findall(r'[+-]?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])  # Return last number found
            except ValueError:
                pass
        
        return None
    
    def contains_reasoning(self, text: str) -> bool:
        """Check if text contains reasoning steps"""
        reasoning_markers = [
            "step", "first", "then", "next", "finally",
            "because", "therefore", "thus", "solve"
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in reasoning_markers)
    
    def is_valid_code(self, code: str) -> bool:
        """Check if code is syntactically valid Python"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def is_valid_sql(self, sql: str) -> bool:
        """Basic SQL syntax validation"""
        sql_upper = sql.upper().strip()
        valid_starts = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
        return any(sql_upper.startswith(start) for start in valid_starts)
    
    def code_similarity(self, code1: str, code2: str) -> float:
        """Calculate structural similarity between code snippets"""
        # Remove whitespace and comments for comparison
        clean1 = re.sub(r'#.*', '', code1).replace(' ', '').replace('\n', '')
        clean2 = re.sub(r'#.*', '', code2).replace(' ', '').replace('\n', '')
        
        return SequenceMatcher(None, clean1, clean2).ratio()
    
    def run_test_cases(self, code: str, test_cases: List[Tuple]) -> int:
        """Run test cases on code (simplified)"""
        # In practice, would execute code safely in sandbox
        # For now, return mock results
        return len(test_cases) // 2
    
    def calculate_text_quality(self, text: str, metadata: Dict) -> float:
        """Calculate text quality without ground truth"""
        quality = 0.5  # Base quality
        
        # Length appropriateness
        length = len(text.split())
        if 50 <= length <= 500:
            quality += 0.1
        
        # Sentence structure
        sentences = text.split('.')
        if 3 <= len(sentences) <= 20:
            quality += 0.1
        
        # Vocabulary diversity
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio > 0.5:
            quality += 0.1
        
        # Check for common quality indicators
        quality_indicators = ["therefore", "however", "moreover", "specifically"]
        if any(indicator in text.lower() for indicator in quality_indicators):
            quality += 0.1
        
        return min(quality, 1.0)
    
    # Quality metric calculations
    def calculate_accuracy(self, prediction: str, ground_truth: str) -> float:
        """Calculate accuracy metric"""
        if not ground_truth:
            return 0.5
        
        return 1.0 if prediction.strip() == ground_truth.strip() else 0.0
    
    def calculate_completeness(self, response: str, requirements: List[str]) -> float:
        """Calculate completeness based on requirements"""
        if not requirements:
            return 1.0 if len(response) > 10 else 0.5
        
        covered = sum(1 for req in requirements if req.lower() in response.lower())
        return covered / len(requirements)
    
    def calculate_coherence(self, text: str) -> float:
        """Calculate text coherence (simplified)"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Check for logical connectors
        connectors = ["therefore", "however", "moreover", "because", "thus"]
        connector_count = sum(1 for conn in connectors if conn in text.lower())
        
        coherence = min(1.0, 0.5 + 0.1 * connector_count)
        return coherence
    
    def calculate_efficiency(self, action: str, metadata: Dict) -> float:
        """Calculate efficiency metric"""
        # Time efficiency
        time_taken = metadata.get("time_taken", 0)
        max_time = metadata.get("max_time", 60)
        time_efficiency = 1 - (time_taken / max_time) if max_time > 0 else 0.5
        
        # Length efficiency
        ideal_length = metadata.get("ideal_length", len(action))
        length_ratio = len(action) / ideal_length if ideal_length > 0 else 1
        length_efficiency = 1 - abs(1 - length_ratio)
        
        return (time_efficiency + length_efficiency) / 2


class HierarchicalRewardCalculator(RewardCalculator):
    """
    Hierarchical reward calculator for multi-level agent systems
    Implements credit assignment for hierarchical RL
    """
    
    def __init__(self, config: RewardConfig = None):
        super().__init__(config)
        self.high_level_multiplier = 1.5
        self.low_level_multiplier = 1.0
    
    def calculate_hierarchical_reward(self,
                                     transitions: List[Dict],
                                     final_reward: float) -> List[float]:
        """
        Assign credit to transitions in hierarchical structure
        
        Args:
            transitions: List of transitions
            final_reward: Episode final reward
            
        Returns:
            List of rewards for each transition
        """
        rewards = []
        
        # Group transitions by hierarchy level
        high_level_transitions = []
        low_level_transitions = []
        
        for i, transition in enumerate(transitions):
            level = transition.get("info", {}).get("hierarchy_level", "low")
            if level == "high":
                high_level_transitions.append(i)
            else:
                low_level_transitions.append(i)
        
        # Calculate credit assignment
        for i, transition in enumerate(transitions):
            if i in high_level_transitions:
                # High-level actions get credit for overall strategy
                credit = final_reward * self.high_level_multiplier
                credit /= max(1, len(high_level_transitions))
            else:
                # Low-level actions get credit for execution
                credit = final_reward * self.low_level_multiplier
                credit /= max(1, len(low_level_transitions))
            
            # Add immediate reward if available
            immediate = transition.get("reward", 0)
            total_reward = 0.7 * credit + 0.3 * immediate
            
            rewards.append(total_reward)
        
        return rewards


class CuriosityReward:
    """
    Intrinsic curiosity reward for exploration
    Encourages agents to explore novel states
    """
    
    def __init__(self, state_dim: int = 768):
        self.state_dim = state_dim
        self.state_history = []
        self.novelty_threshold = 0.5
    
    def calculate_curiosity_reward(self, state: np.ndarray) -> float:
        """Calculate curiosity bonus based on state novelty"""
        if len(self.state_history) == 0:
            self.state_history.append(state)
            return 0.1  # Small bonus for first state
        
        # Calculate novelty as distance to nearest previous state
        min_distance = float('inf')
        for prev_state in self.state_history[-100:]:  # Check last 100 states
            distance = np.linalg.norm(state - prev_state)
            min_distance = min(min_distance, distance)
        
        # Add to history
        self.state_history.append(state)
        
        # Calculate reward based on novelty
        if min_distance > self.novelty_threshold:
            return 0.2  # High novelty bonus
        else:
            return 0.05 * (min_distance / self.novelty_threshold)


# Example usage
if __name__ == "__main__":
    print("💰 Testing Reward Functions")
    print("=" * 60)
    
    # Initialize reward calculator
    config = RewardConfig(
        reward_type=RewardType.SHAPED,
        success_reward=1.0,
        failure_penalty=-0.5
    )
    calculator = RewardCalculator(config)
    
    # Test math reward
    print("\n📐 Testing Math Reward:")
    math_action = "Let me solve: 3x + 7 = 22\nSubtract 7: 3x = 15\nDivide by 3: x = 5"
    math_truth = "x = 5"
    math_reward = calculator.calculate_reward(
        math_action, math_truth, "math", {}
    )
    print(f"   Action: {math_action[:50]}...")
    print(f"   Ground truth: {math_truth}")
    print(f"   Reward: {math_reward:.3f}")
    
    # Test code reward
    print("\n💻 Testing Code Reward:")
    code_action = "def is_prime(n):\n    if n <= 1: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"
    code_reward = calculator.calculate_reward(
        code_action, "", "code", {"test_cases": [(2, True), (4, False)]}
    )
    print(f"   Code valid: {calculator.is_valid_code(code_action)}")
    print(f"   Reward: {code_reward:.3f}")
    
    # Test text reward
    print("\n📝 Testing Text Reward:")
    text_action = "Reinforcement learning is a machine learning paradigm where agents learn through interaction."
    text_truth = "RL is a type of machine learning where agents learn by interacting with environments."
    text_reward = calculator.calculate_reward(
        text_action, text_truth, "text", {"ideal_length": 100}
    )
    print(f"   Similarity: {SequenceMatcher(None, text_action, text_truth).ratio():.3f}")
    print(f"   Reward: {text_reward:.3f}")
    
    # Test multi-agent reward
    print("\n👥 Testing Multi-Agent Reward:")
    multi_reward = calculator.calculate_reward(
        "Collaborative solution", 
        "Expected solution",
        "multi_agent",
        {
            "agent_rewards": {"agent1": 0.8, "agent2": 0.7, "agent3": 0.9},
            "coordination": {"communication_quality": 0.8, "consensus_level": 0.9},
            "parallel_execution": True
        }
    )
    print(f"   Reward: {multi_reward:.3f}")
    
    # Test hierarchical rewards
    print("\n🏗️ Testing Hierarchical Rewards:")
    hier_calculator = HierarchicalRewardCalculator(config)
    test_transitions = [
        {"info": {"hierarchy_level": "high"}, "reward": 0.1},
        {"info": {"hierarchy_level": "low"}, "reward": 0.2},
        {"info": {"hierarchy_level": "low"}, "reward": 0.15},
        {"info": {"hierarchy_level": "high"}, "reward": 0.3}
    ]
    hier_rewards = hier_calculator.calculate_hierarchical_reward(
        test_transitions, final_reward=0.8
    )
    print(f"   Transitions: {len(test_transitions)}")
    print(f"   Assigned rewards: {[f'{r:.3f}' for r in hier_rewards]}")
    
    # Test curiosity reward
    print("\n🔍 Testing Curiosity Reward:")
    curiosity = CuriosityReward(state_dim=10)
    
    # Novel state
    novel_state = np.random.randn(10)
    novel_reward = curiosity.calculate_curiosity_reward(novel_state)
    print(f"   Novel state reward: {novel_reward:.3f}")
    
    # Similar state
    similar_state = novel_state + np.random.randn(10) * 0.1
    similar_reward = curiosity.calculate_curiosity_reward(similar_state)
    print(f"   Similar state reward: {similar_reward:.3f}")
    
    print("\n✅ Reward functions test complete!")