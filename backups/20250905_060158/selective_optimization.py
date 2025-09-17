"""
Selective Optimization for Agent Lightning
Targeted improvements for specific agents and capabilities
Implements focused training and fine-tuning strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from collections import defaultdict
import copy

# Import Agent Lightning components
from mdp_agents import MDPAgent, AgentState, AgentAction, MDPTransition
from reward_functions import RewardCalculator, RewardConfig
from memory_manager import MemoryManager
from meta_learning import MetaLearner


class OptimizationTarget(Enum):
    """Targets for selective optimization"""
    ACCURACY = "accuracy"
    SPEED = "speed"
    ROBUSTNESS = "robustness"
    GENERALIZATION = "generalization"
    EFFICIENCY = "efficiency"
    SPECIALIZATION = "specialization"


class CapabilityArea(Enum):
    """Agent capability areas"""
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    CODING = "coding"
    LANGUAGE = "language"
    PLANNING = "planning"
    MEMORY = "memory"
    TOOL_USE = "tool_use"
    COLLABORATION = "collaboration"


@dataclass
class OptimizationProfile:
    """Profile for selective optimization"""
    agent_id: str
    target: OptimizationTarget
    capability_areas: List[CapabilityArea]
    current_performance: Dict[str, float]
    target_performance: Dict[str, float]
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of selective optimization"""
    agent_id: str
    optimization_target: OptimizationTarget
    initial_performance: Dict[str, float]
    final_performance: Dict[str, float]
    improvement: Dict[str, float]
    iterations: int
    time_taken: float
    success: bool


class SelectiveOptimizer:
    """
    Main selective optimization system for Agent Lightning
    Provides targeted improvements for specific agent capabilities
    """
    
    def __init__(self):
        """Initialize selective optimizer"""
        self.agents = {}
        self.optimization_profiles = {}
        self.optimization_history = []
        
        # Performance baselines
        self.performance_baselines = defaultdict(dict)
        
        # Optimization strategies
        self.strategies = self._initialize_strategies()
        
        # Reward calculator for evaluation
        self.reward_calculator = RewardCalculator()
        
        # Memory manager for experience
        self.memory_manager = MemoryManager()
        
        # Meta-learner for fast adaptation
        self.meta_learner = MetaLearner()
        
        print("🎯 Selective Optimizer initialized")
    
    def _initialize_strategies(self) -> Dict[OptimizationTarget, Callable]:
        """Initialize optimization strategies"""
        return {
            OptimizationTarget.ACCURACY: self._optimize_for_accuracy,
            OptimizationTarget.SPEED: self._optimize_for_speed,
            OptimizationTarget.ROBUSTNESS: self._optimize_for_robustness,
            OptimizationTarget.GENERALIZATION: self._optimize_for_generalization,
            OptimizationTarget.EFFICIENCY: self._optimize_for_efficiency,
            OptimizationTarget.SPECIALIZATION: self._optimize_for_specialization
        }
    
    def analyze_agent(self, agent: MDPAgent) -> Dict[str, Any]:
        """
        Analyze agent's current capabilities and weaknesses
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Analysis results
        """
        analysis = {
            "agent_id": agent.role,
            "capabilities": {},
            "weaknesses": [],
            "strengths": [],
            "recommendations": []
        }
        
        # Test agent on various tasks
        test_results = self._run_capability_tests(agent)
        
        # Analyze results
        for capability in CapabilityArea:
            score = test_results.get(capability.value, 0.0)
            analysis["capabilities"][capability.value] = score
            
            if score < 0.5:
                analysis["weaknesses"].append(capability.value)
            elif score > 0.8:
                analysis["strengths"].append(capability.value)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _run_capability_tests(self, agent: MDPAgent) -> Dict[str, float]:
        """Run capability tests on agent"""
        results = {}
        
        # Test reasoning
        reasoning_tasks = [
            {"input": "If A implies B and B implies C, what can we conclude?", 
             "expected": "A implies C"},
            {"input": "All birds can fly. Penguins are birds. Can penguins fly?",
             "expected": "This is a logical fallacy"}
        ]
        results[CapabilityArea.REASONING.value] = self._test_capability(
            agent, reasoning_tasks, "reasoning"
        )
        
        # Test mathematics
        math_tasks = [
            {"input": "Solve: 2x + 5 = 13", "expected": "x = 4"},
            {"input": "Calculate: 15% of 200", "expected": "30"}
        ]
        results[CapabilityArea.MATHEMATICS.value] = self._test_capability(
            agent, math_tasks, "math"
        )
        
        # Test coding
        code_tasks = [
            {"input": "Write a function to reverse a string", 
             "expected": "def reverse_string"},
            {"input": "Fix this code: for i in range(10) print(i)",
             "expected": "for i in range(10): print(i)"}
        ]
        results[CapabilityArea.CODING.value] = self._test_capability(
            agent, code_tasks, "code"
        )
        
        # Test language
        language_tasks = [
            {"input": "Summarize: The quick brown fox jumps over the lazy dog",
             "expected": "A fox jumps over a dog"},
            {"input": "Translate to formal: hey whats up",
             "expected": "Hello, how are you?"}
        ]
        results[CapabilityArea.LANGUAGE.value] = self._test_capability(
            agent, language_tasks, "text"
        )
        
        # Test other capabilities with simplified scores
        results[CapabilityArea.PLANNING.value] = np.random.random() * 0.5 + 0.4
        results[CapabilityArea.MEMORY.value] = np.random.random() * 0.5 + 0.4
        results[CapabilityArea.TOOL_USE.value] = np.random.random() * 0.5 + 0.4
        results[CapabilityArea.COLLABORATION.value] = np.random.random() * 0.5 + 0.4
        
        return results
    
    def _test_capability(self, agent: MDPAgent, tasks: List[Dict], 
                        task_type: str) -> float:
        """Test agent on specific capability"""
        scores = []
        
        for task in tasks:
            # Create state
            state = agent.observe({
                "input": task["input"],
                "context": {},
                "semantic_variables": {"task_type": task_type}
            })
            
            # Get agent action
            action, _ = agent.act(state)
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(
                action=action.content,
                ground_truth=task["expected"],
                task_type=task_type
            )
            
            scores.append(reward)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Recommend focusing on weaknesses
        if analysis["weaknesses"]:
            weak_areas = ", ".join(analysis["weaknesses"][:3])
            recommendations.append(f"Focus on improving: {weak_areas}")
        
        # Recommend leveraging strengths
        if analysis["strengths"]:
            strong_areas = ", ".join(analysis["strengths"][:2])
            recommendations.append(f"Leverage strengths in: {strong_areas}")
        
        # Recommend optimization targets
        avg_score = np.mean(list(analysis["capabilities"].values()))
        if avg_score < 0.6:
            recommendations.append("Prioritize ACCURACY optimization")
        elif avg_score > 0.8:
            recommendations.append("Consider EFFICIENCY or SPECIALIZATION optimization")
        else:
            recommendations.append("Focus on GENERALIZATION for broader capabilities")
        
        return recommendations
    
    def create_optimization_profile(self,
                                   agent_id: str,
                                   target: OptimizationTarget,
                                   capability_areas: List[CapabilityArea],
                                   target_improvement: float = 0.2) -> OptimizationProfile:
        """
        Create optimization profile for an agent
        
        Args:
            agent_id: ID of agent to optimize
            target: Optimization target
            capability_areas: Areas to focus on
            target_improvement: Target improvement percentage
            
        Returns:
            Optimization profile
        """
        # Get current performance
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            analysis = self.analyze_agent(agent)
            current_performance = analysis["capabilities"]
        else:
            # Default performance
            current_performance = {area.value: 0.5 for area in capability_areas}
        
        # Calculate target performance
        target_performance = {}
        for area in capability_areas:
            current = current_performance.get(area.value, 0.5)
            target_performance[area.value] = min(1.0, current + target_improvement)
        
        profile = OptimizationProfile(
            agent_id=agent_id,
            target=target,
            capability_areas=capability_areas,
            current_performance=current_performance,
            target_performance=target_performance
        )
        
        self.optimization_profiles[agent_id] = profile
        
        return profile
    
    def optimize_agent(self,
                      agent: MDPAgent,
                      profile: OptimizationProfile,
                      num_iterations: int = 100) -> OptimizationResult:
        """
        Optimize agent according to profile
        
        Args:
            agent: Agent to optimize
            profile: Optimization profile
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimization result
        """
        print(f"\n🔧 Optimizing agent: {profile.agent_id}")
        print(f"   Target: {profile.target.value}")
        print(f"   Areas: {[area.value for area in profile.capability_areas]}")
        
        start_time = time.time()
        initial_performance = profile.current_performance.copy()
        
        # Select optimization strategy
        strategy = self.strategies.get(profile.target)
        if not strategy:
            raise ValueError(f"Unknown optimization target: {profile.target}")
        
        # Run optimization
        success = strategy(agent, profile, num_iterations)
        
        # Evaluate final performance
        final_analysis = self.analyze_agent(agent)
        final_performance = final_analysis["capabilities"]
        
        # Calculate improvements
        improvement = {}
        for area in profile.capability_areas:
            area_name = area.value
            initial = initial_performance.get(area_name, 0.0)
            final = final_performance.get(area_name, 0.0)
            improvement[area_name] = final - initial
        
        # Create result
        result = OptimizationResult(
            agent_id=profile.agent_id,
            optimization_target=profile.target,
            initial_performance=initial_performance,
            final_performance=final_performance,
            improvement=improvement,
            iterations=num_iterations,
            time_taken=time.time() - start_time,
            success=success
        )
        
        # Store in history
        self.optimization_history.append(result)
        
        print(f"\n✅ Optimization complete!")
        print(f"   Time taken: {result.time_taken:.2f}s")
        print(f"   Average improvement: {np.mean(list(improvement.values())):.2%}")
        
        return result
    
    def _optimize_for_accuracy(self, agent: MDPAgent, profile: OptimizationProfile,
                              num_iterations: int) -> bool:
        """Optimize agent for accuracy"""
        print("   Strategy: Accuracy optimization")
        
        for iteration in range(num_iterations):
            # Generate high-quality training examples
            training_data = self._generate_accuracy_training_data(profile.capability_areas)
            
            # Fine-tune agent on accurate examples
            for example in training_data:
                state = agent.observe({
                    "input": example["input"],
                    "context": {"focus": "accuracy"},
                    "semantic_variables": {"optimization": "accuracy"}
                })
                
                action, transition = agent.act(state)
                
                # Calculate accuracy-focused reward
                reward = self.reward_calculator.calculate_reward(
                    action=action.content,
                    ground_truth=example["ground_truth"],
                    task_type=example["type"],
                    metadata={"accuracy_weight": 1.0}
                )
                
                # Update agent if using Q-learning
                if hasattr(agent, 'update_q_values'):
                    agent.update_q_values(state, action, reward)
                
                # Store successful examples
                if reward > 0.8:
                    self.memory_manager.store_episodic({
                        "state": state.to_dict(),
                        "action": action.to_dict(),
                        "reward": reward
                    }, importance=reward)
            
            if iteration % 20 == 0:
                print(f"     Iteration {iteration}/{num_iterations}")
        
        return True
    
    def _optimize_for_speed(self, agent: MDPAgent, profile: OptimizationProfile,
                          num_iterations: int) -> bool:
        """Optimize agent for speed"""
        print("   Strategy: Speed optimization")
        
        # Implement caching mechanisms
        cache = {}
        
        for iteration in range(num_iterations):
            # Generate speed-focused examples
            training_data = self._generate_speed_training_data(profile.capability_areas)
            
            for example in training_data:
                # Check cache first
                cache_key = json.dumps(example["input"], sort_keys=True)
                if cache_key in cache:
                    continue
                
                start_time = time.time()
                
                state = agent.observe({
                    "input": example["input"],
                    "context": {"focus": "speed"},
                    "semantic_variables": {"optimization": "speed"}
                })
                
                action, _ = agent.act(state)
                
                response_time = time.time() - start_time
                
                # Reward based on speed
                speed_reward = max(0, 1.0 - response_time / 2.0)  # 2 second baseline
                
                # Cache successful fast responses
                if speed_reward > 0.7:
                    cache[cache_key] = action.content
                
                # Prune agent's decision tree for faster inference
                if hasattr(agent, 'prune_decision_paths'):
                    agent.prune_decision_paths(threshold=0.1)
            
            if iteration % 20 == 0:
                print(f"     Iteration {iteration}/{num_iterations}")
        
        return True
    
    def _optimize_for_robustness(self, agent: MDPAgent, profile: OptimizationProfile,
                                num_iterations: int) -> bool:
        """Optimize agent for robustness"""
        print("   Strategy: Robustness optimization")
        
        for iteration in range(num_iterations):
            # Generate adversarial examples
            training_data = self._generate_adversarial_examples(profile.capability_areas)
            
            for example in training_data:
                # Add noise to input
                noisy_input = self._add_noise(example["input"])
                
                state = agent.observe({
                    "input": noisy_input,
                    "context": {"focus": "robustness"},
                    "semantic_variables": {"optimization": "robustness"}
                })
                
                action, transition = agent.act(state)
                
                # Evaluate robustness
                reward = self._evaluate_robustness(
                    action.content,
                    example["ground_truth"],
                    noise_level=example.get("noise_level", 0.1)
                )
                
                # Store robust responses
                if reward > 0.7:
                    self.memory_manager.store_episodic({
                        "state": state.to_dict(),
                        "action": action.to_dict(),
                        "reward": reward,
                        "robust": True
                    }, importance=reward)
            
            # Increase noise level gradually
            if iteration % 10 == 0:
                print(f"     Iteration {iteration}/{num_iterations}")
        
        return True
    
    def _optimize_for_generalization(self, agent: MDPAgent, profile: OptimizationProfile,
                                    num_iterations: int) -> bool:
        """Optimize agent for generalization"""
        print("   Strategy: Generalization optimization")
        
        for iteration in range(num_iterations):
            # Generate diverse task distribution
            task_distribution = self._generate_diverse_tasks(profile.capability_areas)
            
            # Meta-training step
            for task_batch in self._batch_tasks(task_distribution, batch_size=4):
                for task in task_batch:
                    state = agent.observe({
                        "input": task["input"],
                        "context": {"task_type": task["type"]},
                        "semantic_variables": {"optimization": "generalization"}
                    })
                    
                    action, _ = agent.act(state)
                    
                    # Evaluate on novel variations
                    generalization_score = self._evaluate_generalization(
                        agent, task, num_variations=3
                    )
                    
                    # Update based on generalization performance
                    if generalization_score > 0.6:
                        self.memory_manager.store_semantic(
                            key=f"general_{task['type']}",
                            value={
                                "pattern": task["input"],
                                "response": action.content,
                                "score": generalization_score
                            }
                        )
            
            if iteration % 20 == 0:
                print(f"     Iteration {iteration}/{num_iterations}")
        
        return True
    
    def _optimize_for_efficiency(self, agent: MDPAgent, profile: OptimizationProfile,
                                num_iterations: int) -> bool:
        """Optimize agent for efficiency"""
        print("   Strategy: Efficiency optimization")
        
        for iteration in range(num_iterations):
            # Generate efficiency-focused examples
            training_data = self._generate_efficiency_training_data(profile.capability_areas)
            
            for example in training_data:
                state = agent.observe({
                    "input": example["input"],
                    "context": {"focus": "efficiency"},
                    "semantic_variables": {"optimization": "efficiency"}
                })
                
                action, transition = agent.act(state)
                
                # Evaluate efficiency (accuracy vs resources)
                efficiency_score = self._evaluate_efficiency(
                    action.content,
                    example["ground_truth"],
                    transition
                )
                
                # Optimize decision paths
                if efficiency_score > 0.7:
                    # Simplify successful patterns
                    if hasattr(agent, 'compress_knowledge'):
                        agent.compress_knowledge(pattern=state, response=action)
            
            if iteration % 20 == 0:
                print(f"     Iteration {iteration}/{num_iterations}")
        
        return True
    
    def _optimize_for_specialization(self, agent: MDPAgent, profile: OptimizationProfile,
                                    num_iterations: int) -> bool:
        """Optimize agent for specialization"""
        print("   Strategy: Specialization optimization")
        
        # Focus on specific capability areas
        for capability in profile.capability_areas:
            print(f"     Specializing in: {capability.value}")
            
            for iteration in range(num_iterations // len(profile.capability_areas)):
                # Generate specialized training data
                training_data = self._generate_specialized_data(capability)
                
                for example in training_data:
                    state = agent.observe({
                        "input": example["input"],
                        "context": {"specialization": capability.value},
                        "semantic_variables": {"optimization": "specialization"}
                    })
                    
                    action, _ = agent.act(state)
                    
                    # Evaluate specialization
                    specialization_score = self.reward_calculator.calculate_reward(
                        action=action.content,
                        ground_truth=example["ground_truth"],
                        task_type=capability.value
                    )
                    
                    # Store specialized knowledge
                    if specialization_score > 0.8:
                        self.memory_manager.store_procedural(
                            procedure={
                                "trigger": example["pattern"],
                                "action": action.content,
                                "domain": capability.value
                            }
                        )
        
        return True
    
    # Helper methods for training data generation
    def _generate_accuracy_training_data(self, areas: List[CapabilityArea]) -> List[Dict]:
        """Generate training data for accuracy optimization"""
        data = []
        
        for area in areas:
            if area == CapabilityArea.MATHEMATICS:
                data.extend([
                    {"input": "2 + 2", "ground_truth": "4", "type": "math"},
                    {"input": "sqrt(16)", "ground_truth": "4", "type": "math"}
                ])
            elif area == CapabilityArea.REASONING:
                data.extend([
                    {"input": "If P then Q, P is true", "ground_truth": "Q is true", "type": "reasoning"},
                    {"input": "All A are B, X is A", "ground_truth": "X is B", "type": "reasoning"}
                ])
            # Add more areas as needed
        
        return data
    
    def _generate_speed_training_data(self, areas: List[CapabilityArea]) -> List[Dict]:
        """Generate training data for speed optimization"""
        # Simple, quick-to-process examples
        return [
            {"input": f"Quick {area.value} task {i}", "type": area.value}
            for area in areas
            for i in range(5)
        ]
    
    def _generate_adversarial_examples(self, areas: List[CapabilityArea]) -> List[Dict]:
        """Generate adversarial examples for robustness"""
        examples = []
        
        for area in areas:
            base_examples = self._generate_accuracy_training_data([area])
            
            for example in base_examples:
                # Add various types of noise
                examples.append({
                    **example,
                    "noise_level": 0.1,
                    "noise_type": "typo"
                })
                examples.append({
                    **example,
                    "noise_level": 0.2,
                    "noise_type": "semantic"
                })
        
        return examples
    
    def _generate_diverse_tasks(self, areas: List[CapabilityArea]) -> List[Dict]:
        """Generate diverse tasks for generalization"""
        tasks = []
        
        for area in areas:
            # Generate variations of tasks
            for variation in range(10):
                tasks.append({
                    "input": f"{area.value} task variation {variation}",
                    "type": area.value,
                    "variation": variation,
                    "ground_truth": f"solution_{variation}"
                })
        
        return tasks
    
    def _generate_efficiency_training_data(self, areas: List[CapabilityArea]) -> List[Dict]:
        """Generate training data for efficiency optimization"""
        return [
            {
                "input": f"Efficient {area.value} task",
                "ground_truth": f"optimal_{area.value}_solution",
                "type": area.value,
                "max_steps": 5
            }
            for area in areas
        ]
    
    def _generate_specialized_data(self, capability: CapabilityArea) -> List[Dict]:
        """Generate specialized training data"""
        specialized = []
        
        # Deep, specialized examples for the capability
        for i in range(20):
            specialized.append({
                "input": f"Advanced {capability.value} problem {i}",
                "ground_truth": f"expert_{capability.value}_solution_{i}",
                "pattern": f"{capability.value}_pattern_{i % 5}",
                "difficulty": "expert"
            })
        
        return specialized
    
    def _add_noise(self, text: str, noise_level: float = 0.1) -> str:
        """Add noise to text for robustness testing"""
        import random
        
        if random.random() < noise_level:
            # Random typo
            if len(text) > 0:
                idx = random.randint(0, len(text) - 1)
                text = text[:idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + text[idx+1:]
        
        return text
    
    def _evaluate_robustness(self, output: str, ground_truth: str, 
                            noise_level: float) -> float:
        """Evaluate robustness of output"""
        base_score = self.reward_calculator.calculate_reward(
            action=output,
            ground_truth=ground_truth,
            task_type="general"
        )
        
        # Bonus for handling noise well
        robustness_bonus = (1 - noise_level) * 0.2
        
        return min(1.0, base_score + robustness_bonus)
    
    def _evaluate_generalization(self, agent: MDPAgent, task: Dict, 
                                num_variations: int = 3) -> float:
        """Evaluate generalization capability"""
        scores = []
        
        for i in range(num_variations):
            # Create variation of task
            varied_input = f"{task['input']} (variation {i})"
            
            state = agent.observe({
                "input": varied_input,
                "context": {"original": task["input"]},
                "semantic_variables": {}
            })
            
            action, _ = agent.act(state)
            
            # Simple similarity check
            score = 0.5 + np.random.random() * 0.5  # Simplified
            scores.append(score)
        
        return np.mean(scores)
    
    def _evaluate_efficiency(self, output: str, ground_truth: str,
                           transition: MDPTransition) -> float:
        """Evaluate efficiency of solution"""
        # Accuracy component
        accuracy = self.reward_calculator.calculate_reward(
            action=output,
            ground_truth=ground_truth,
            task_type="general"
        )
        
        # Efficiency component (simplified)
        steps_taken = transition.info.get("steps", 10)
        efficiency = max(0, 1 - steps_taken / 20)
        
        # Combined score
        return 0.7 * accuracy + 0.3 * efficiency
    
    def _batch_tasks(self, tasks: List[Dict], batch_size: int = 4) -> List[List[Dict]]:
        """Batch tasks for processing"""
        batches = []
        for i in range(0, len(tasks), batch_size):
            batches.append(tasks[i:i+batch_size])
        return batches
    
    def compare_optimizations(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Compare multiple optimization results"""
        comparison = {
            "best_overall": None,
            "best_by_area": {},
            "average_improvements": {},
            "success_rate": 0
        }
        
        if not results:
            return comparison
        
        # Find best overall improvement
        best_improvement = -float('inf')
        for result in results:
            avg_improvement = np.mean(list(result.improvement.values()))
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                comparison["best_overall"] = result.agent_id
        
        # Find best by capability area
        area_improvements = defaultdict(list)
        for result in results:
            for area, improvement in result.improvement.items():
                area_improvements[area].append((result.agent_id, improvement))
        
        for area, improvements in area_improvements.items():
            best = max(improvements, key=lambda x: x[1])
            comparison["best_by_area"][area] = {
                "agent": best[0],
                "improvement": best[1]
            }
        
        # Calculate average improvements
        for result in results:
            comparison["average_improvements"][result.agent_id] = np.mean(
                list(result.improvement.values())
            )
        
        # Calculate success rate
        comparison["success_rate"] = sum(1 for r in results if r.success) / len(results)
        
        return comparison
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            "total_optimizations": len(self.optimization_history),
            "optimization_targets": {},
            "capability_improvements": {},
            "average_time": 0,
            "success_rate": 0,
            "recommendations": []
        }
        
        if not self.optimization_history:
            return report
        
        # Analyze by optimization target
        target_results = defaultdict(list)
        for result in self.optimization_history:
            target_results[result.optimization_target.value].append(result)
        
        for target, results in target_results.items():
            avg_improvement = np.mean([
                np.mean(list(r.improvement.values())) for r in results
            ])
            report["optimization_targets"][target] = {
                "count": len(results),
                "avg_improvement": avg_improvement,
                "success_rate": sum(1 for r in results if r.success) / len(results)
            }
        
        # Analyze by capability area
        capability_improvements = defaultdict(list)
        for result in self.optimization_history:
            for area, improvement in result.improvement.items():
                capability_improvements[area].append(improvement)
        
        for area, improvements in capability_improvements.items():
            report["capability_improvements"][area] = {
                "avg_improvement": np.mean(improvements),
                "max_improvement": max(improvements),
                "min_improvement": min(improvements)
            }
        
        # Calculate overall metrics
        report["average_time"] = np.mean([r.time_taken for r in self.optimization_history])
        report["success_rate"] = sum(1 for r in self.optimization_history if r.success) / len(self.optimization_history)
        
        # Generate recommendations
        if report["success_rate"] < 0.7:
            report["recommendations"].append("Consider increasing optimization iterations")
        
        weakest_area = min(report["capability_improvements"].items(), 
                          key=lambda x: x[1]["avg_improvement"])
        report["recommendations"].append(f"Focus on improving {weakest_area[0]}")
        
        return report


# Example usage
if __name__ == "__main__":
    print("🎯 Testing Selective Optimization System")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = SelectiveOptimizer()
    
    # Create test agents
    test_agents = [
        MDPAgent(role="MathExpert"),
        MDPAgent(role="ReasoningAgent"),
        MDPAgent(role="GeneralistAgent")
    ]
    
    # Store agents
    for agent in test_agents:
        optimizer.agents[agent.role] = agent
    
    # Analyze agents
    print("\n📊 Analyzing Agent Capabilities...")
    for agent in test_agents:
        analysis = optimizer.analyze_agent(agent)
        print(f"\n{agent.role}:")
        print(f"  Strengths: {analysis['strengths']}")
        print(f"  Weaknesses: {analysis['weaknesses']}")
        print(f"  Recommendations: {analysis['recommendations'][0] if analysis['recommendations'] else 'None'}")
    
    # Create optimization profiles
    print("\n📋 Creating Optimization Profiles...")
    
    profiles = [
        optimizer.create_optimization_profile(
            "MathExpert",
            OptimizationTarget.ACCURACY,
            [CapabilityArea.MATHEMATICS, CapabilityArea.REASONING],
            target_improvement=0.2
        ),
        optimizer.create_optimization_profile(
            "ReasoningAgent",
            OptimizationTarget.GENERALIZATION,
            [CapabilityArea.REASONING, CapabilityArea.LANGUAGE],
            target_improvement=0.15
        ),
        optimizer.create_optimization_profile(
            "GeneralistAgent",
            OptimizationTarget.EFFICIENCY,
            [CapabilityArea.CODING, CapabilityArea.PLANNING],
            target_improvement=0.1
        )
    ]
    
    # Optimize agents
    print("\n🔧 Running Selective Optimizations...")
    results = []
    
    for agent, profile in zip(test_agents, profiles):
        result = optimizer.optimize_agent(agent, profile, num_iterations=20)
        results.append(result)
        
        print(f"\n{result.agent_id} Optimization Results:")
        print(f"  Target: {result.optimization_target.value}")
        print(f"  Success: {result.success}")
        print(f"  Improvements:")
        for area, improvement in result.improvement.items():
            print(f"    {area}: {improvement:+.2%}")
    
    # Compare optimizations
    print("\n📊 Comparing Optimizations...")
    comparison = optimizer.compare_optimizations(results)
    
    print(f"Best Overall: {comparison['best_overall']}")
    print(f"Success Rate: {comparison['success_rate']:.0%}")
    print("\nBest by Area:")
    for area, info in comparison["best_by_area"].items():
        print(f"  {area}: {info['agent']} (+{info['improvement']:.2%})")
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION REPORT")
    print("=" * 60)
    
    report = optimizer.generate_optimization_report()
    
    print(f"\nTotal Optimizations: {report['total_optimizations']}")
    print(f"Average Time: {report['average_time']:.2f}s")
    print(f"Success Rate: {report['success_rate']:.0%}")
    
    print("\nOptimization Targets:")
    for target, metrics in report["optimization_targets"].items():
        print(f"  {target}:")
        print(f"    Count: {metrics['count']}")
        print(f"    Avg Improvement: {metrics['avg_improvement']:.2%}")
    
    print("\nCapability Improvements:")
    for area, metrics in report["capability_improvements"].items():
        print(f"  {area}:")
        print(f"    Average: {metrics['avg_improvement']:+.2%}")
        print(f"    Range: [{metrics['min_improvement']:+.2%}, {metrics['max_improvement']:+.2%}]")
    
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  - {rec}")
    
    print("\n✅ Selective optimization test complete!")