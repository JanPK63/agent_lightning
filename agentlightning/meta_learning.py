#!/usr/bin/env python3
"""
Meta-learning capabilities for Agent Lightning

This module provides meta-learning algorithms and frameworks that enable agents
to learn how to learn, adapt their strategies, and improve performance over time.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict

from agentlightning.types import Rollout, Task, NamedResources

logger = logging.getLogger(__name__)


@dataclass
class LearningExperience:
    """Represents a learning experience for meta-learning"""
    task_id: str
    rollout: Rollout
    resources_used: NamedResources
    performance_metrics: Dict[str, float]
    timestamp: float
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class AdaptationStrategy:
    """Represents a strategy for adapting agent behavior"""
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    performance_history: List[float] = None
    usage_count: int = 0

    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []

    def record_performance(self, performance: float):
        """Record performance of this strategy"""
        self.performance_history.append(performance)
        self.usage_count += 1

    def get_average_performance(self) -> float:
        """Get average performance of this strategy"""
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)


class MetaLearner(ABC):
    """Abstract base class for meta-learners"""

    def __init__(self, learner_id: str):
        self.learner_id = learner_id
        self.experience_buffer: List[LearningExperience] = []
        self.strategies: Dict[str, AdaptationStrategy] = {}
        self.learning_rate = 0.01

    @abstractmethod
    def learn_from_experience(self, experience: LearningExperience):
        """Learn from a single experience"""
        pass

    @abstractmethod
    def adapt_strategy(self, task: Task, current_resources: NamedResources) -> NamedResources:
        """Adapt resources/strategy based on meta-learning"""
        pass

    @abstractmethod
    def predict_performance(self, task: Task, resources: NamedResources) -> float:
        """Predict performance for a given task-resource combination"""
        pass

    def add_experience(self, experience: LearningExperience):
        """Add experience to the buffer"""
        self.experience_buffer.append(experience)

        # Keep buffer size manageable
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]  # Keep last 500

    def get_similar_experiences(self, task: Task, limit: int = 5) -> List[LearningExperience]:
        """Find similar past experiences"""
        # Simple similarity based on task input type and metadata
        similar = []
        for exp in reversed(self.experience_buffer):  # Most recent first
            if len(similar) >= limit:
                break
            # Basic similarity check - can be enhanced
            if exp.task_id != task.rollout_id:  # Avoid same task
                similar.append(exp)
        return similar


class PromptOptimizer(MetaLearner):
    """Meta-learner that optimizes prompts based on past performance"""

    def __init__(self, learner_id: str):
        super().__init__(learner_id)
        self.prompt_templates: Dict[str, str] = {}
        self.prompt_performance: Dict[str, List[float]] = defaultdict(list)

    def learn_from_experience(self, experience: LearningExperience):
        """Learn from rollout experience to improve prompts"""
        self.add_experience(experience)

        # Extract prompt information from resources
        if 'prompt_template' in experience.resources:
            prompt_key = experience.resources['prompt_template'].template
            performance = experience.performance_metrics.get('final_reward', 0.0)
            self.prompt_performance[prompt_key].append(performance)

    def adapt_strategy(self, task: Task, current_resources: NamedResources) -> NamedResources:
        """Adapt prompt based on meta-learning"""
        adapted_resources = current_resources.copy()

        if 'prompt_template' in current_resources:
            # Find best performing similar prompts
            similar_experiences = self.get_similar_experiences(task)

            best_prompt = None
            best_performance = -float('inf')

            for exp in similar_experiences:
                if 'prompt_template' in exp.resources:
                    prompt_key = exp.resources['prompt_template'].template
                    avg_perf = sum(self.prompt_performance.get(prompt_key, [0])) / max(1, len(self.prompt_performance.get(prompt_key, [])))

                    if avg_perf > best_performance:
                        best_performance = avg_perf
                        best_prompt = prompt_key

            if best_prompt and best_performance > 0:
                # Adapt the prompt template
                adapted_resources['prompt_template'] = adapted_resources['prompt_template'].copy()
                adapted_resources['prompt_template'].template = best_prompt
                logger.info(f"Meta-learning adapted prompt for better performance: {best_performance:.3f}")

        return adapted_resources

    def predict_performance(self, task: Task, resources: NamedResources) -> float:
        """Predict performance based on prompt history"""
        if 'prompt_template' in resources:
            prompt_key = resources['prompt_template'].template
            performances = self.prompt_performance.get(prompt_key, [])
            if performances:
                return sum(performances) / len(performances)
        return 0.5  # Default neutral prediction


class ResourceAllocator(MetaLearner):
    """Meta-learner that optimizes resource allocation"""

    def __init__(self, learner_id: str):
        super().__init__(learner_id)
        self.resource_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def learn_from_experience(self, experience: LearningExperience):
        """Learn from resource usage patterns"""
        self.add_experience(experience)

        # Track performance per resource type and model
        for resource_name, resource in experience.resources.items():
            if hasattr(resource, 'model'):
                model_name = resource.model
                performance = experience.performance_metrics.get('final_reward', 0.0)
                self.resource_performance[resource_name][model_name].append(performance)

    def adapt_strategy(self, task: Task, current_resources: NamedResources) -> NamedResources:
        """Adapt resource allocation based on meta-learning"""
        adapted_resources = current_resources.copy()

        # Optimize LLM selection based on past performance
        if 'main_llm' in current_resources:
            similar_experiences = self.get_similar_experiences(task)

            best_model = None
            best_performance = -float('inf')

            for exp in similar_experiences:
                if 'main_llm' in exp.resources:
                    model_name = exp.resources['main_llm'].model
                    performances = self.resource_performance['main_llm'].get(model_name, [])
                    if performances:
                        avg_perf = sum(performances) / len(performances)
                        if avg_perf > best_performance:
                            best_performance = avg_perf
                            best_model = model_name

            if best_model and best_model != current_resources['main_llm'].model:
                adapted_resources['main_llm'] = adapted_resources['main_llm'].copy()
                adapted_resources['main_llm'].model = best_model
                logger.info(f"Meta-learning switched to better performing model: {best_model} (perf: {best_performance:.3f})")

        return adapted_resources

    def predict_performance(self, task: Task, resources: NamedResources) -> float:
        """Predict performance based on resource history"""
        total_performance = 0.0
        count = 0

        for resource_name, resource in resources.items():
            if hasattr(resource, 'model'):
                model_name = resource.model
                performances = self.resource_performance[resource_name].get(model_name, [])
                if performances:
                    total_performance += sum(performances) / len(performances)
                    count += 1

        return total_performance / max(1, count) if count > 0 else 0.5


class HyperparameterTuner(MetaLearner):
    """Meta-learner that tunes hyperparameters dynamically"""

    def __init__(self, learner_id: str):
        super().__init__(learner_id)
        self.param_performance: Dict[str, Dict[str, List[Tuple[Any, float]]]] = defaultdict(lambda: defaultdict(list))

    def learn_from_experience(self, experience: LearningExperience):
        """Learn from hyperparameter performance"""
        self.add_experience(experience)

        # Track performance per parameter setting
        for resource_name, resource in experience.resources.items():
            if hasattr(resource, 'sampling_parameters'):
                params = resource.sampling_parameters
                for param_name, param_value in params.items():
                    performance = experience.performance_metrics.get('final_reward', 0.0)
                    self.param_performance[resource_name][param_name].append((param_value, performance))

    def adapt_strategy(self, task: Task, current_resources: NamedResources) -> NamedResources:
        """Adapt hyperparameters based on meta-learning"""
        adapted_resources = current_resources.copy()

        for resource_name, resource in current_resources.items():
            if hasattr(resource, 'sampling_parameters'):
                adapted_resource = adapted_resources[resource_name].copy()
                adapted_resource.sampling_parameters = adapted_resource.sampling_parameters.copy()

                # Tune temperature based on past performance
                if 'temperature' in adapted_resource.sampling_parameters:
                    temp_performances = self.param_performance[resource_name]['temperature']

                    if len(temp_performances) >= 3:  # Need some data
                        # Find best temperature
                        best_temp = None
                        best_perf = -float('inf')

                        temp_counts = defaultdict(list)
                        for temp, perf in temp_performances:
                            temp_counts[temp].append(perf)

                        for temp, perfs in temp_counts.items():
                            avg_perf = sum(perfs) / len(perfs)
                            if avg_perf > best_perf:
                                best_perf = avg_perf
                                best_temp = temp

                        if best_temp is not None:
                            current_temp = adapted_resource.sampling_parameters['temperature']
                            if abs(best_temp - current_temp) > 0.1:  # Significant difference
                                adapted_resource.sampling_parameters['temperature'] = best_temp
                                logger.info(f"Meta-learning tuned temperature from {current_temp} to {best_temp} (perf: {best_perf:.3f})")

                adapted_resources[resource_name] = adapted_resource

        return adapted_resources

    def predict_performance(self, task: Task, resources: NamedResources) -> float:
        """Predict performance based on parameter history"""
        total_score = 0.0
        count = 0

        for resource_name, resource in resources.items():
            if hasattr(resource, 'sampling_parameters'):
                params = resource.sampling_parameters

                for param_name, param_value in params.items():
                    param_history = self.param_performance[resource_name][param_name]
                    if param_history:
                        # Find similar parameter values
                        similar_perfs = []
                        for hist_value, perf in param_history:
                            if abs(hist_value - param_value) < 0.1:  # Close values
                                similar_perfs.append(perf)

                        if similar_perfs:
                            total_score += sum(similar_perfs) / len(similar_perfs)
                            count += 1

        return total_score / max(1, count) if count > 0 else 0.5


class MetaLearningOrchestrator:
    """Orchestrates multiple meta-learners for comprehensive adaptation"""

    def __init__(self):
        self.learners: Dict[str, MetaLearner] = {}
        self.global_experience_buffer: List[LearningExperience] = []

    def add_learner(self, learner: MetaLearner):
        """Add a meta-learner to the orchestrator"""
        self.learners[learner.learner_id] = learner
        logger.info(f"Added meta-learner: {learner.learner_id}")

    def learn_from_rollout(self, task: Task, rollout: Rollout, resources: NamedResources):
        """Learn from a completed rollout"""
        experience = LearningExperience(
            task_id=task.rollout_id,
            rollout=rollout,
            resources_used=resources,
            performance_metrics={
                'final_reward': rollout.final_reward or 0.0,
                'triplets_count': len(rollout.triplets or []),
                'timestamp': time.time()
            },
            timestamp=time.time(),
            context={'task_mode': task.mode, 'resources_count': len(resources)}
        )

        self.global_experience_buffer.append(experience)

        # Distribute learning to all meta-learners
        for learner in self.learners.values():
            learner.learn_from_experience(experience)

        # Keep global buffer manageable
        if len(self.global_experience_buffer) > 2000:
            self.global_experience_buffer = self.global_experience_buffer[-1000:]

    def adapt_for_task(self, task: Task, current_resources: NamedResources) -> NamedResources:
        """Adapt resources for a new task using meta-learning"""
        adapted_resources = current_resources.copy()

        # Apply adaptations from all learners
        for learner in self.learners.values():
            try:
                adapted_resources = learner.adapt_strategy(task, adapted_resources)
            except Exception as e:
                logger.warning(f"Meta-learner {learner.learner_id} adaptation failed: {e}")

        return adapted_resources

    def predict_task_performance(self, task: Task, resources: NamedResources) -> Dict[str, float]:
        """Predict performance for a task-resource combination"""
        predictions = {}
        for learner_id, learner in self.learners.items():
            try:
                predictions[learner_id] = learner.predict_performance(task, resources)
            except Exception as e:
                logger.warning(f"Meta-learner {learner_id} prediction failed: {e}")
                predictions[learner_id] = 0.5

        # Overall prediction as weighted average
        if predictions:
            weights = {'prompt_optimizer': 0.4, 'resource_allocator': 0.4, 'hyperparameter_tuner': 0.2}
            weighted_sum = sum(predictions.get(k, 0.5) * weights.get(k, 0.33) for k in predictions.keys())
            predictions['overall'] = weighted_sum

        return predictions

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about meta-learning progress"""
        return {
            'total_experiences': len(self.global_experience_buffer),
            'learners': {
                learner_id: {
                    'experiences': len(learner.experience_buffer),
                    'strategies': len(learner.strategies)
                }
                for learner_id, learner in self.learners.items()
            }
        }


# Global meta-learning orchestrator
meta_orchestrator = MetaLearningOrchestrator()

# Initialize with default learners
meta_orchestrator.add_learner(PromptOptimizer("prompt_optimizer"))
meta_orchestrator.add_learner(ResourceAllocator("resource_allocator"))
meta_orchestrator.add_learner(HyperparameterTuner("hyperparameter_tuner"))


def enable_meta_learning_for_agent(agent) -> None:
    """Enable meta-learning capabilities for an agent"""
    original_training_rollout = agent.training_rollout

    async def meta_learning_training_rollout(task_input, rollout_id, resources):
        # Adapt resources using meta-learning
        adapted_resources = meta_orchestrator.adapt_for_task(task_input, resources)

        # Execute rollout with adapted resources
        result = await original_training_rollout(task_input, rollout_id, adapted_resources)

        # Learn from the experience (this would be called after rollout completion)
        # Note: This is a simplified version - in practice, learning would happen
        # after the rollout is fully processed and rewards are calculated

        return result

    # Replace the method
    agent.training_rollout = meta_learning_training_rollout
    logger.info(f"Enabled meta-learning for agent: {agent.agent_id}")