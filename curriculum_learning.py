"""
Curriculum Learning for Agent Lightning
Implements progressive task difficulty to improve learning efficiency
Following the Agent Lightning approach to structured learning progression
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import random
from collections import deque
import time

# Import Agent Lightning components
from mdp_agents import MDPAgent, AgentState, AgentAction, MDPTransition
from reward_functions import RewardCalculator, RewardConfig
from memory_manager import MemoryManager


class DifficultyLevel(Enum):
    """Task difficulty levels for curriculum"""
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5
    MASTER = 6


class TaskCategory(Enum):
    """Categories of tasks for curriculum"""
    MATH = "math"
    TEXT = "text"
    CODE = "code"
    REASONING = "reasoning"
    MULTI_AGENT = "multi_agent"
    TOOL_USE = "tool_use"


@dataclass
class CurriculumTask:
    """Represents a task in the curriculum"""
    task_id: str
    category: TaskCategory
    difficulty: DifficultyLevel
    content: Dict[str, Any]
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    max_attempts: int = 3
    time_limit: float = 60.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class LearningProgress:
    """Tracks agent's learning progress"""
    agent_id: str
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    current_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER
    performance_history: List[float] = field(default_factory=list)
    skill_levels: Dict[TaskCategory, float] = field(default_factory=dict)
    total_reward: float = 0.0
    learning_rate: float = 0.0


class CurriculumLearning:
    """
    Main curriculum learning system for Agent Lightning
    Manages progressive task difficulty and skill development
    """
    
    def __init__(self,
                 initial_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
                 progression_threshold: float = 0.7,
                 regression_threshold: float = 0.3,
                 window_size: int = 10):
        """
        Initialize curriculum learning system
        
        Args:
            initial_difficulty: Starting difficulty level
            progression_threshold: Performance threshold to advance
            regression_threshold: Performance threshold to regress
            window_size: Window for performance averaging
        """
        self.current_difficulty = initial_difficulty
        self.progression_threshold = progression_threshold
        self.regression_threshold = regression_threshold
        self.window_size = window_size
        
        # Task curriculum database
        self.curriculum_tasks = self._build_curriculum()
        
        # Learning progress tracking
        self.agent_progress = {}
        
        # Performance metrics
        self.performance_window = deque(maxlen=window_size)
        self.global_performance = []
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Memory for experience replay
        self.memory_manager = MemoryManager()
        
        print(f"📚 Curriculum Learning initialized")
        print(f"   Starting difficulty: {initial_difficulty.name}")
        print(f"   Progression threshold: {progression_threshold:.2f}")
        print(f"   Tasks in curriculum: {len(self.curriculum_tasks)}")
    
    def _build_curriculum(self) -> Dict[str, List[CurriculumTask]]:
        """Build the task curriculum with progressive difficulty"""
        curriculum = {level: [] for level in DifficultyLevel}
        
        # BEGINNER level tasks
        curriculum[DifficultyLevel.BEGINNER].extend([
            CurriculumTask(
                task_id="math_beginner_1",
                category=TaskCategory.MATH,
                difficulty=DifficultyLevel.BEGINNER,
                content={
                    "question": "What is 5 + 3?",
                    "answer": "8",
                    "type": "arithmetic"
                },
                success_criteria={"accuracy": 1.0}
            ),
            CurriculumTask(
                task_id="text_beginner_1",
                category=TaskCategory.TEXT,
                difficulty=DifficultyLevel.BEGINNER,
                content={
                    "instruction": "Complete the sentence: The cat is ___",
                    "acceptable_answers": ["sleeping", "happy", "cute", "playful"],
                    "type": "completion"
                },
                success_criteria={"validity": 0.8}
            )
        ])
        
        # EASY level tasks
        curriculum[DifficultyLevel.EASY].extend([
            CurriculumTask(
                task_id="math_easy_1",
                category=TaskCategory.MATH,
                difficulty=DifficultyLevel.EASY,
                content={
                    "question": "Solve for x: 2x + 4 = 10",
                    "answer": "3",
                    "type": "algebra"
                },
                prerequisites=["math_beginner_1"],
                success_criteria={"accuracy": 0.9}
            ),
            CurriculumTask(
                task_id="reasoning_easy_1",
                category=TaskCategory.REASONING,
                difficulty=DifficultyLevel.EASY,
                content={
                    "question": "If all roses are flowers and some flowers are red, can we conclude all roses are red?",
                    "answer": "No",
                    "type": "logical_reasoning"
                },
                success_criteria={"correctness": 1.0}
            )
        ])
        
        # MEDIUM level tasks
        curriculum[DifficultyLevel.MEDIUM].extend([
            CurriculumTask(
                task_id="code_medium_1",
                category=TaskCategory.CODE,
                difficulty=DifficultyLevel.MEDIUM,
                content={
                    "instruction": "Write a function to check if a number is prime",
                    "test_cases": [(2, True), (4, False), (17, True)],
                    "type": "implementation"
                },
                prerequisites=["math_easy_1"],
                success_criteria={"test_pass_rate": 0.8, "syntax_valid": 1.0}
            ),
            CurriculumTask(
                task_id="multi_agent_medium_1",
                category=TaskCategory.MULTI_AGENT,
                difficulty=DifficultyLevel.MEDIUM,
                content={
                    "task": "Collaborate to write a short story",
                    "roles": ["plot_designer", "character_developer", "narrator"],
                    "type": "collaboration"
                },
                success_criteria={"coherence": 0.7, "completion": 1.0}
            )
        ])
        
        # HARD level tasks
        curriculum[DifficultyLevel.HARD].extend([
            CurriculumTask(
                task_id="math_hard_1",
                category=TaskCategory.MATH,
                difficulty=DifficultyLevel.HARD,
                content={
                    "question": "Find the derivative of f(x) = x^3 * sin(x)",
                    "answer": "3x^2 * sin(x) + x^3 * cos(x)",
                    "type": "calculus"
                },
                prerequisites=["math_easy_1", "math_medium_1"],
                success_criteria={"accuracy": 0.85}
            ),
            CurriculumTask(
                task_id="tool_use_hard_1",
                category=TaskCategory.TOOL_USE,
                difficulty=DifficultyLevel.HARD,
                content={
                    "task": "Use multiple tools to research and summarize a topic",
                    "tools": ["search", "calculator", "translator"],
                    "type": "complex_tool_use"
                },
                success_criteria={"tool_accuracy": 0.8, "result_quality": 0.7}
            )
        ])
        
        # EXPERT level tasks
        curriculum[DifficultyLevel.EXPERT].extend([
            CurriculumTask(
                task_id="reasoning_expert_1",
                category=TaskCategory.REASONING,
                difficulty=DifficultyLevel.EXPERT,
                content={
                    "question": "Solve the Tower of Hanoi with 4 disks, minimizing moves",
                    "optimal_moves": 15,
                    "type": "algorithmic_reasoning"
                },
                prerequisites=["reasoning_easy_1", "reasoning_hard_1"],
                success_criteria={"optimality": 0.9, "correctness": 1.0}
            )
        ])
        
        # MASTER level tasks
        curriculum[DifficultyLevel.MASTER].extend([
            CurriculumTask(
                task_id="multi_agent_master_1",
                category=TaskCategory.MULTI_AGENT,
                difficulty=DifficultyLevel.MASTER,
                content={
                    "task": "Design and implement a distributed system architecture",
                    "requirements": ["scalability", "fault_tolerance", "consistency"],
                    "type": "system_design"
                },
                prerequisites=["code_medium_1", "multi_agent_medium_1", "tool_use_hard_1"],
                success_criteria={"design_quality": 0.85, "implementation": 0.8}
            )
        ])
        
        return curriculum
    
    def get_next_task(self, agent_id: str) -> Optional[CurriculumTask]:
        """
        Get the next appropriate task for an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Next task or None if curriculum complete
        """
        # Get or create agent progress
        if agent_id not in self.agent_progress:
            self.agent_progress[agent_id] = LearningProgress(
                agent_id=agent_id,
                current_difficulty=self.current_difficulty
            )
        
        progress = self.agent_progress[agent_id]
        
        # Get available tasks at current difficulty
        available_tasks = self._get_available_tasks(progress)
        
        if not available_tasks:
            # Try to progress to next level
            if self._should_progress(progress):
                progress.current_difficulty = self._get_next_difficulty(progress.current_difficulty)
                available_tasks = self._get_available_tasks(progress)
            elif self._should_regress(progress):
                progress.current_difficulty = self._get_previous_difficulty(progress.current_difficulty)
                available_tasks = self._get_available_tasks(progress)
        
        if available_tasks:
            # Select task based on learning strategy
            task = self._select_task_strategically(available_tasks, progress)
            return task
        
        return None
    
    def _get_available_tasks(self, progress: LearningProgress) -> List[CurriculumTask]:
        """Get tasks available for the agent's current level"""
        available = []
        
        tasks_at_level = self.curriculum_tasks.get(progress.current_difficulty, [])
        
        for task in tasks_at_level:
            # Check if not already completed
            if task.task_id in progress.completed_tasks:
                continue
            
            # Check if prerequisites met
            if all(prereq in progress.completed_tasks for prereq in task.prerequisites):
                available.append(task)
        
        return available
    
    def _select_task_strategically(self, 
                                  tasks: List[CurriculumTask],
                                  progress: LearningProgress) -> CurriculumTask:
        """Select task using curriculum learning strategy"""
        # Strategy 1: Balance across categories
        category_scores = {}
        for category in TaskCategory:
            skill_level = progress.skill_levels.get(category, 0.0)
            category_scores[category] = 1.0 - skill_level  # Prioritize weaker skills
        
        # Score each task
        task_scores = []
        for task in tasks:
            score = category_scores.get(task.category, 0.5)
            
            # Bonus for variety
            if len(progress.completed_tasks) > 0:
                last_tasks = progress.completed_tasks[-3:]
                last_categories = [self._get_task_category(t) for t in last_tasks]
                if task.category not in last_categories:
                    score += 0.2
            
            task_scores.append((task, score))
        
        # Select based on scores with some randomness
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Use epsilon-greedy selection
        epsilon = 0.1
        if random.random() < epsilon:
            return random.choice(tasks)
        else:
            return task_scores[0][0]
    
    def _get_task_category(self, task_id: str) -> Optional[TaskCategory]:
        """Get category of a task by ID"""
        for tasks in self.curriculum_tasks.values():
            for task in tasks:
                if task.task_id == task_id:
                    return task.category
        return None
    
    def evaluate_performance(self,
                            agent_id: str,
                            task: CurriculumTask,
                            result: Dict[str, Any]) -> float:
        """
        Evaluate agent's performance on a task
        
        Args:
            agent_id: ID of the agent
            task: The completed task
            result: Task execution result
            
        Returns:
            Performance score (0-1)
        """
        progress = self.agent_progress[agent_id]
        
        # Calculate task-specific performance
        performance = 0.0
        criteria_met = 0
        
        for criterion, threshold in task.success_criteria.items():
            if criterion in result:
                score = result[criterion]
                if score >= threshold:
                    criteria_met += 1
                performance += score
        
        if task.success_criteria:
            performance /= len(task.success_criteria)
            success_rate = criteria_met / len(task.success_criteria)
        else:
            success_rate = 0.5
        
        # Update progress
        if success_rate >= 0.5:
            progress.completed_tasks.append(task.task_id)
        else:
            progress.failed_tasks.append(task.task_id)
        
        # Update performance history
        progress.performance_history.append(performance)
        self.performance_window.append(performance)
        
        # Update skill levels
        current_skill = progress.skill_levels.get(task.category, 0.0)
        alpha = 0.1  # Learning rate
        progress.skill_levels[task.category] = current_skill + alpha * (performance - current_skill)
        
        # Calculate learning rate
        if len(progress.performance_history) > 1:
            recent_performance = progress.performance_history[-10:]
            progress.learning_rate = np.mean(np.diff(recent_performance))
        
        return performance
    
    def _should_progress(self, progress: LearningProgress) -> bool:
        """Check if agent should progress to next difficulty"""
        if len(self.performance_window) < self.window_size:
            return False
        
        avg_performance = np.mean(self.performance_window)
        return avg_performance >= self.progression_threshold
    
    def _should_regress(self, progress: LearningProgress) -> bool:
        """Check if agent should regress to easier difficulty"""
        if len(self.performance_window) < self.window_size // 2:
            return False
        
        avg_performance = np.mean(self.performance_window)
        return avg_performance < self.regression_threshold
    
    def _get_next_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get next difficulty level"""
        levels = list(DifficultyLevel)
        current_idx = levels.index(current)
        if current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        return current
    
    def _get_previous_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get previous difficulty level"""
        levels = list(DifficultyLevel)
        current_idx = levels.index(current)
        if current_idx > 0:
            return levels[current_idx - 1]
        return current
    
    def generate_curriculum_batch(self,
                                 agent_id: str,
                                 batch_size: int = 10) -> List[CurriculumTask]:
        """
        Generate a batch of tasks for training
        
        Args:
            agent_id: ID of the agent
            batch_size: Number of tasks to generate
            
        Returns:
            List of curriculum tasks
        """
        batch = []
        
        for _ in range(batch_size):
            task = self.get_next_task(agent_id)
            if task:
                batch.append(task)
            else:
                # Curriculum complete or need to repeat
                break
        
        # If batch incomplete, add review tasks
        if len(batch) < batch_size:
            progress = self.agent_progress[agent_id]
            review_tasks = self._generate_review_tasks(progress, batch_size - len(batch))
            batch.extend(review_tasks)
        
        return batch
    
    def _generate_review_tasks(self,
                              progress: LearningProgress,
                              num_tasks: int) -> List[CurriculumTask]:
        """Generate review tasks from completed ones"""
        review_tasks = []
        
        if progress.completed_tasks:
            # Select tasks to review based on performance
            for _ in range(num_tasks):
                # Prefer tasks from weaker categories
                weak_categories = sorted(
                    progress.skill_levels.items(),
                    key=lambda x: x[1]
                )[:3]
                
                # Find completed tasks in weak categories
                for category, _ in weak_categories:
                    for task_id in progress.completed_tasks:
                        if self._get_task_category(task_id) == category:
                            # Find the original task
                            for tasks in self.curriculum_tasks.values():
                                for task in tasks:
                                    if task.task_id == task_id:
                                        # Create review version
                                        review_task = CurriculumTask(
                                            task_id=f"{task_id}_review",
                                            category=task.category,
                                            difficulty=task.difficulty,
                                            content=task.content,
                                            prerequisites=task.prerequisites,
                                            success_criteria=task.success_criteria,
                                            metadata={"is_review": True}
                                        )
                                        review_tasks.append(review_task)
                                        break
                            break
                
                if len(review_tasks) >= num_tasks:
                    break
        
        return review_tasks[:num_tasks]
    
    def adapt_difficulty_dynamically(self, agent_id: str):
        """Dynamically adapt difficulty based on performance"""
        progress = self.agent_progress[agent_id]
        
        # Calculate performance trend
        if len(progress.performance_history) >= 5:
            recent = progress.performance_history[-5:]
            older = progress.performance_history[-10:-5] if len(progress.performance_history) >= 10 else recent
            
            trend = np.mean(recent) - np.mean(older)
            
            # Adjust progression threshold based on trend
            if trend > 0.1:  # Improving rapidly
                self.progression_threshold = max(0.6, self.progression_threshold - 0.05)
            elif trend < -0.1:  # Struggling
                self.progression_threshold = min(0.8, self.progression_threshold + 0.05)
        
        # Adjust based on learning rate
        if progress.learning_rate > 0.05:  # Fast learner
            progress.current_difficulty = self._get_next_difficulty(progress.current_difficulty)
        elif progress.learning_rate < -0.02:  # Struggling
            progress.current_difficulty = self._get_previous_difficulty(progress.current_difficulty)
    
    def get_curriculum_report(self, agent_id: str) -> Dict[str, Any]:
        """Generate curriculum progress report"""
        if agent_id not in self.agent_progress:
            return {"error": f"No progress found for agent {agent_id}"}
        
        progress = self.agent_progress[agent_id]
        
        # Calculate statistics
        total_tasks = len(progress.completed_tasks) + len(progress.failed_tasks)
        success_rate = len(progress.completed_tasks) / total_tasks if total_tasks > 0 else 0
        
        # Skill assessment
        skill_assessment = {}
        for category in TaskCategory:
            skill_level = progress.skill_levels.get(category, 0.0)
            skill_assessment[category.value] = {
                "level": skill_level,
                "proficiency": self._get_proficiency_label(skill_level)
            }
        
        report = {
            "agent_id": agent_id,
            "current_difficulty": progress.current_difficulty.name,
            "completed_tasks": len(progress.completed_tasks),
            "failed_tasks": len(progress.failed_tasks),
            "success_rate": success_rate,
            "average_performance": np.mean(progress.performance_history) if progress.performance_history else 0,
            "learning_rate": progress.learning_rate,
            "skill_assessment": skill_assessment,
            "recommendations": self._generate_recommendations(progress)
        }
        
        return report
    
    def _get_proficiency_label(self, skill_level: float) -> str:
        """Convert skill level to proficiency label"""
        if skill_level >= 0.9:
            return "Expert"
        elif skill_level >= 0.7:
            return "Advanced"
        elif skill_level >= 0.5:
            return "Intermediate"
        elif skill_level >= 0.3:
            return "Beginner"
        else:
            return "Novice"
    
    def _generate_recommendations(self, progress: LearningProgress) -> List[str]:
        """Generate learning recommendations"""
        recommendations = []
        
        # Check weak areas
        weak_skills = [cat for cat, level in progress.skill_levels.items() if level < 0.5]
        if weak_skills:
            recommendations.append(f"Focus on improving: {', '.join([s.value for s in weak_skills])}")
        
        # Check learning rate
        if progress.learning_rate < 0:
            recommendations.append("Consider reviewing fundamentals before advancing")
        elif progress.learning_rate > 0.1:
            recommendations.append("Excellent progress! Ready for more challenging tasks")
        
        # Check failure rate
        if len(progress.failed_tasks) > len(progress.completed_tasks) * 0.3:
            recommendations.append("High failure rate detected. Consider additional practice")
        
        return recommendations
    
    def save_curriculum_state(self, filepath: str):
        """Save curriculum state to file"""
        state = {
            "agent_progress": {
                agent_id: {
                    "completed_tasks": prog.completed_tasks,
                    "failed_tasks": prog.failed_tasks,
                    "current_difficulty": prog.current_difficulty.value,
                    "performance_history": prog.performance_history,
                    "skill_levels": {k.value: v for k, v in prog.skill_levels.items()},
                    "total_reward": prog.total_reward,
                    "learning_rate": prog.learning_rate
                }
                for agent_id, prog in self.agent_progress.items()
            },
            "global_performance": self.global_performance,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 Curriculum state saved to {filepath}")
    
    def load_curriculum_state(self, filepath: str):
        """Load curriculum state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore agent progress
        for agent_id, prog_data in state["agent_progress"].items():
            progress = LearningProgress(agent_id=agent_id)
            progress.completed_tasks = prog_data["completed_tasks"]
            progress.failed_tasks = prog_data["failed_tasks"]
            progress.current_difficulty = DifficultyLevel(prog_data["current_difficulty"])
            progress.performance_history = prog_data["performance_history"]
            progress.skill_levels = {
                TaskCategory(k): v for k, v in prog_data["skill_levels"].items()
            }
            progress.total_reward = prog_data["total_reward"]
            progress.learning_rate = prog_data["learning_rate"]
            
            self.agent_progress[agent_id] = progress
        
        self.global_performance = state["global_performance"]
        
        print(f"📂 Curriculum state loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("📚 Testing Curriculum Learning System")
    print("=" * 60)
    
    # Initialize curriculum
    curriculum = CurriculumLearning(
        initial_difficulty=DifficultyLevel.BEGINNER,
        progression_threshold=0.7,
        regression_threshold=0.3
    )
    
    # Simulate agent learning
    agent_id = "test_agent_001"
    
    print(f"\n🎯 Starting curriculum for agent: {agent_id}")
    
    # Simulate 20 tasks
    for i in range(20):
        # Get next task
        task = curriculum.get_next_task(agent_id)
        
        if task:
            print(f"\n📝 Task {i+1}: {task.task_id}")
            print(f"   Category: {task.category.value}")
            print(f"   Difficulty: {task.difficulty.name}")
            
            # Simulate task execution with random performance
            result = {}
            for criterion in task.success_criteria.keys():
                # Simulate performance based on difficulty
                base_performance = 0.9 - (task.difficulty.value * 0.1)
                result[criterion] = max(0, min(1, base_performance + random.uniform(-0.2, 0.2)))
            
            # Evaluate performance
            performance = curriculum.evaluate_performance(agent_id, task, result)
            print(f"   Performance: {performance:.2%}")
            
            # Adapt difficulty
            if i % 5 == 4:  # Every 5 tasks
                curriculum.adapt_difficulty_dynamically(agent_id)
        else:
            print(f"\n✅ Curriculum complete or waiting for progression")
            break
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 CURRICULUM REPORT")
    print("=" * 60)
    
    report = curriculum.get_curriculum_report(agent_id)
    print(f"\nAgent: {report['agent_id']}")
    print(f"Current Difficulty: {report['current_difficulty']}")
    print(f"Completed Tasks: {report['completed_tasks']}")
    print(f"Failed Tasks: {report['failed_tasks']}")
    print(f"Success Rate: {report['success_rate']:.2%}")
    print(f"Average Performance: {report['average_performance']:.2%}")
    print(f"Learning Rate: {report['learning_rate']:.3f}")
    
    print("\n🎯 Skill Assessment:")
    for skill, assessment in report['skill_assessment'].items():
        print(f"  {skill}: {assessment['proficiency']} (level: {assessment['level']:.2f})")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Save state
    curriculum.save_curriculum_state("curriculum_state.json")
    
    print("\n✅ Curriculum learning test complete!")