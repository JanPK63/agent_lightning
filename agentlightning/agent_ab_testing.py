#!/usr/bin/env python3
"""
A/B Testing Framework for Agent Configurations

This module provides comprehensive A/B testing capabilities for agent configurations,
including different prompts, hyperparameters, models, and architectures.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os

from agentlightning.types import Task, Rollout, NamedResources
from agentlightning.model_versioning import ABTest, model_registry

logger = logging.getLogger(__name__)


class ConfigurationType(Enum):
    """Types of agent configurations that can be A/B tested"""
    PROMPT = "prompt"
    HYPERPARAMETERS = "hyperparameters"
    MODEL = "model"
    ARCHITECTURE = "architecture"
    TRAINING_PARAMS = "training_params"
    RESOURCE_ALLOCATION = "resource_allocation"


@dataclass
class AgentConfiguration:
    """Represents an agent configuration for A/B testing"""
    config_id: str
    name: str
    description: str
    config_type: ConfigurationType

    # Configuration details
    prompt_template: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    architecture_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    resource_config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "config_id": self.config_id,
            "name": self.name,
            "description": self.description,
            "config_type": self.config_type.value,
            "prompt_template": self.prompt_template,
            "hyperparameters": self.hyperparameters,
            "model_config": self.model_config,
            "architecture_config": self.architecture_config,
            "training_config": self.training_config,
            "resource_config": self.resource_config,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfiguration':
        """Create from dictionary"""
        data_copy = data.copy()
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        data_copy["config_type"] = ConfigurationType(data["config_type"])
        return cls(**data_copy)


@dataclass
class AgentABTest:
    """A/B test for agent configurations"""
    test_id: str
    name: str
    description: str
    config_a: AgentConfiguration
    config_b: AgentConfiguration
    primary_metric: str
    secondary_metrics: List[str] = field(default_factory=list)
    test_duration_hours: int = 24
    traffic_split: float = 0.5  # Percentage of traffic to config B
    status: str = "running"  # running, completed, stopped

    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Test execution
    task_filter: Optional[Dict[str, Any]] = None  # Filter for tasks to test on
    min_samples: int = 100  # Minimum samples per configuration

    # Results
    config_a_results: List[Dict[str, Any]] = field(default_factory=list)
    config_b_results: List[Dict[str, Any]] = field(default_factory=list)
    statistical_significance: Optional[float] = None
    winner: Optional[str] = None  # "A", "B", or "tie"
    confidence_level: str = "medium"  # low, medium, high

    def add_result(self, config_id: str, metrics: Dict[str, Any],
                  task: Task, rollout: Optional[Rollout] = None):
        """Add a test result"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "config_id": config_id,
            "metrics": metrics,
            "task_id": task.rollout_id,
            "task_input": task.input,
            "rollout_id": rollout.rollout_id if rollout else None,
            "context": {
                "task_mode": task.mode,
                "task_metadata": task.metadata
            }
        }

        if config_id == self.config_a.config_id:
            self.config_a_results.append(result)
        elif config_id == self.config_b.config_id:
            self.config_b_results.append(result)

    def should_complete(self) -> bool:
        """Check if test should be completed"""
        min_samples_reached = (len(self.config_a_results) >= self.min_samples and
                              len(self.config_b_results) >= self.min_samples)

        time_expired = (datetime.now() - self.created_at).total_seconds() > (self.test_duration_hours * 3600)

        return min_samples_reached or time_expired

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if len(self.config_a_results) < 10 or len(self.config_b_results) < 10:
            return {"status": "insufficient_data"}

        # Extract primary metric
        a_values = [r["metrics"].get(self.primary_metric, 0) for r in self.config_a_results]
        b_values = [r["metrics"].get(self.primary_metric, 0) for r in self.config_b_results]

        if not a_values or not b_values:
            return {"status": "no_data"}

        # Statistical analysis
        a_mean = sum(a_values) / len(a_values)
        a_std = (sum((x - a_mean) ** 2 for x in a_values) / len(a_values)) ** 0.5

        b_mean = sum(b_values) / len(b_values)
        b_std = (sum((x - b_mean) ** 2 for x in b_values) / len(b_values)) ** 0.5

        # Simple t-test approximation
        pooled_std = ((a_std ** 2 / len(a_values)) + (b_std ** 2 / len(b_values))) ** 0.5
        t_stat = abs(a_mean - b_mean) / pooled_std if pooled_std > 0 else 0

        # Determine winner and significance
        improvement = ((b_mean - a_mean) / a_mean) * 100 if a_mean != 0 else 0

        if abs(improvement) < 1.0:  # Less than 1% difference
            winner = "tie"
            significance = 0.0
        elif improvement > 0:
            winner = "B"
            significance = min(t_stat / 2.0, 1.0)  # Rough significance estimate
        else:
            winner = "A"
            significance = min(t_stat / 2.0, 1.0)

        # Confidence level based on sample size and variance
        total_samples = len(a_values) + len(b_values)
        if total_samples > 1000 and significance > 0.95:
            confidence = "high"
        elif total_samples > 100 and significance > 0.8:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "status": "completed",
            "config_a_mean": a_mean,
            "config_b_mean": b_mean,
            "improvement_percent": improvement,
            "winner": winner,
            "statistical_significance": significance,
            "confidence_level": confidence,
            "sample_size_a": len(a_values),
            "sample_size_b": len(b_values),
            "secondary_metrics": self._analyze_secondary_metrics()
        }

    def _analyze_secondary_metrics(self) -> Dict[str, Any]:
        """Analyze secondary metrics"""
        results = {}

        for metric in self.secondary_metrics:
            a_values = [r["metrics"].get(metric, 0) for r in self.config_a_results if metric in r["metrics"]]
            b_values = [r["metrics"].get(metric, 0) for r in self.config_b_results if metric in r["metrics"]]

            if a_values and b_values:
                a_mean = sum(a_values) / len(a_values)
                b_mean = sum(b_values) / len(b_values)
                improvement = ((b_mean - a_mean) / a_mean) * 100 if a_mean != 0 else 0

                results[metric] = {
                    "config_a_mean": a_mean,
                    "config_b_mean": b_mean,
                    "improvement_percent": improvement
                }

        return results


class AgentABTestManager:
    """Manager for agent configuration A/B tests"""

    def __init__(self, storage_path: str = "./agent_ab_tests"):
        self.storage_path = storage_path
        self.tests: Dict[str, AgentABTest] = {}
        self.configurations: Dict[str, AgentConfiguration] = {}

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        # Load existing tests and configurations
        self._load_data()

    def create_configuration(self, name: str, description: str,
                           config_type: ConfigurationType, **config_data) -> AgentConfiguration:
        """Create a new agent configuration"""
        config_id = str(uuid.uuid4())

        config = AgentConfiguration(
            config_id=config_id,
            name=name,
            description=description,
            config_type=config_type,
            **config_data
        )

        self.configurations[config_id] = config
        self._save_configuration(config)

        logger.info(f"Created agent configuration: {config_id} ({name})")
        return config

    def start_ab_test(self, name: str, description: str,
                     config_a: AgentConfiguration, config_b: AgentConfiguration,
                     primary_metric: str, secondary_metrics: Optional[List[str]] = None,
                     test_duration_hours: int = 24, min_samples: int = 100) -> AgentABTest:
        """Start an A/B test between two configurations"""
        test_id = str(uuid.uuid4())

        ab_test = AgentABTest(
            test_id=test_id,
            name=name,
            description=description,
            config_a=config_a,
            config_b=config_b,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            test_duration_hours=test_duration_hours,
            min_samples=min_samples
        )

        self.tests[test_id] = ab_test
        self._save_test(ab_test)

        logger.info(f"Started A/B test: {test_id} ({name})")
        return ab_test

    def select_configuration_for_task(self, task: Task) -> Optional[AgentConfiguration]:
        """Select a configuration for a task based on active A/B tests"""
        # Find applicable A/B tests for this task
        applicable_tests = []

        for ab_test in self.tests.values():
            if ab_test.status != "running":
                continue

            # Check task filter
            if ab_test.task_filter:
                if not self._task_matches_filter(task, ab_test.task_filter):
                    continue

            applicable_tests.append(ab_test)

        if not applicable_tests:
            return None

        # For now, use the first applicable test
        # In production, you might want more sophisticated selection
        test = applicable_tests[0]

        # Simple traffic splitting
        import random
        if random.random() < test.traffic_split:
            return test.config_b
        else:
            return test.config_a

    def record_test_result(self, test_id: str, config_id: str,
                          metrics: Dict[str, Any], task: Task,
                          rollout: Optional[Rollout] = None):
        """Record a result for an A/B test"""
        if test_id in self.tests:
            ab_test = self.tests[test_id]
            ab_test.add_result(config_id, metrics, task, rollout)
            self._save_test(ab_test)

            # Check if test should complete
            if ab_test.should_complete():
                results = ab_test.analyze_results()
                ab_test.status = "completed"
                ab_test.completed_at = datetime.now()
                ab_test.winner = results.get("winner")
                ab_test.statistical_significance = results.get("statistical_significance")
                ab_test.confidence_level = results.get("confidence_level")

                self._save_test(ab_test)
                logger.info(f"A/B test {test_id} completed. Winner: {ab_test.winner}")

    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get results for an A/B test"""
        ab_test = self.tests.get(test_id)
        if not ab_test:
            return None

        if ab_test.status == "running":
            return {"status": "running", "samples_a": len(ab_test.config_a_results), "samples_b": len(ab_test.config_b_results)}

        return ab_test.analyze_results()

    def list_active_tests(self) -> List[AgentABTest]:
        """List all active A/B tests"""
        return [test for test in self.tests.values() if test.status == "running"]

    def list_configurations(self, config_type: Optional[ConfigurationType] = None) -> List[AgentConfiguration]:
        """List agent configurations"""
        configs = list(self.configurations.values())
        if config_type:
            configs = [c for c in configs if c.config_type == config_type]
        return configs

    def _task_matches_filter(self, task: Task, task_filter: Dict[str, Any]) -> bool:
        """Check if a task matches a filter"""
        # Simple filtering logic - can be extended
        for key, value in task_filter.items():
            if key == "mode" and task.mode != value:
                return False
            elif key == "metadata":
                # Check metadata matching
                for meta_key, meta_value in value.items():
                    if task.metadata.get(meta_key) != meta_value:
                        return False
        return True

    def _save_configuration(self, config: AgentConfiguration):
        """Save configuration to disk"""
        config_file = os.path.join(self.storage_path, "configurations", f"{config.config_id}.json")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    def _save_test(self, ab_test: AgentABTest):
        """Save A/B test to disk"""
        test_file = os.path.join(self.storage_path, "tests", f"{ab_test.test_id}.json")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)

        with open(test_file, 'w') as f:
            json.dump({
                "test_id": ab_test.test_id,
                "name": ab_test.name,
                "description": ab_test.description,
                "config_a": ab_test.config_a.to_dict(),
                "config_b": ab_test.config_b.to_dict(),
                "primary_metric": ab_test.primary_metric,
                "secondary_metrics": ab_test.secondary_metrics,
                "test_duration_hours": ab_test.test_duration_hours,
                "traffic_split": ab_test.traffic_split,
                "status": ab_test.status,
                "created_at": ab_test.created_at.isoformat(),
                "completed_at": ab_test.completed_at.isoformat() if ab_test.completed_at else None,
                "task_filter": ab_test.task_filter,
                "min_samples": ab_test.min_samples,
                "config_a_results": ab_test.config_a_results,
                "config_b_results": ab_test.config_b_results,
                "statistical_significance": ab_test.statistical_significance,
                "winner": ab_test.winner,
                "confidence_level": ab_test.confidence_level
            }, f, indent=2, default=str)

    def _load_data(self):
        """Load existing configurations and tests from disk"""
        # Load configurations
        config_dir = os.path.join(self.storage_path, "configurations")
        if os.path.exists(config_dir):
            for config_file in os.listdir(config_dir):
                if config_file.endswith('.json'):
                    file_path = os.path.join(config_dir, config_file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        config = AgentConfiguration.from_dict(data)
                        self.configurations[config.config_id] = config
                    except Exception as e:
                        logger.warning(f"Failed to load configuration {file_path}: {e}")

        # Load tests
        test_dir = os.path.join(self.storage_path, "tests")
        if os.path.exists(test_dir):
            for test_file in os.listdir(test_dir):
                if test_file.endswith('.json'):
                    file_path = os.path.join(test_dir, test_file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        # Reconstruct configurations
                        config_a = AgentConfiguration.from_dict(data["config_a"])
                        config_b = AgentConfiguration.from_dict(data["config_b"])

                        # Reconstruct test
                        ab_test = AgentABTest(
                            test_id=data["test_id"],
                            name=data["name"],
                            description=data["description"],
                            config_a=config_a,
                            config_b=config_b,
                            primary_metric=data["primary_metric"],
                            secondary_metrics=data.get("secondary_metrics", []),
                            test_duration_hours=data["test_duration_hours"],
                            traffic_split=data["traffic_split"],
                            status=data["status"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                            task_filter=data.get("task_filter"),
                            min_samples=data.get("min_samples", 100),
                            statistical_significance=data.get("statistical_significance"),
                            winner=data.get("winner"),
                            confidence_level=data.get("confidence_level", "medium")
                        )

                        ab_test.config_a_results = data.get("config_a_results", [])
                        ab_test.config_b_results = data.get("config_b_results", [])

                        self.tests[ab_test.test_id] = ab_test

                    except Exception as e:
                        logger.warning(f"Failed to load A/B test {file_path}: {e}")


# Global A/B test manager instance
ab_test_manager = AgentABTestManager()


def create_prompt_ab_test(name: str, description: str,
                         prompt_a: str, prompt_b: str,
                         primary_metric: str = "final_reward") -> AgentABTest:
    """Create an A/B test for different prompts"""
    config_a = ab_test_manager.create_configuration(
        name=f"{name}_config_a",
        description=f"Configuration A for {name}",
        config_type=ConfigurationType.PROMPT,
        prompt_template=prompt_a
    )

    config_b = ab_test_manager.create_configuration(
        name=f"{name}_config_b",
        description=f"Configuration B for {name}",
        config_type=ConfigurationType.PROMPT,
        prompt_template=prompt_b
    )

    return ab_test_manager.start_ab_test(
        name=name,
        description=description,
        config_a=config_a,
        config_b=config_b,
        primary_metric=primary_metric
    )


def create_hyperparameter_ab_test(name: str, description: str,
                                 params_a: Dict[str, Any], params_b: Dict[str, Any],
                                 primary_metric: str = "accuracy") -> AgentABTest:
    """Create an A/B test for different hyperparameters"""
    config_a = ab_test_manager.create_configuration(
        name=f"{name}_config_a",
        description=f"Hyperparameters A for {name}",
        config_type=ConfigurationType.HYPERPARAMETERS,
        hyperparameters=params_a
    )

    config_b = ab_test_manager.create_configuration(
        name=f"{name}_config_b",
        description=f"Hyperparameters B for {name}",
        config_type=ConfigurationType.HYPERPARAMETERS,
        hyperparameters=params_b
    )

    return ab_test_manager.start_ab_test(
        name=name,
        description=description,
        config_a=config_a,
        config_b=config_b,
        primary_metric=primary_metric
    )


def get_configuration_for_task(task: Task) -> Optional[AgentConfiguration]:
    """Get the appropriate configuration for a task"""
    return ab_test_manager.select_configuration_for_task(task)


def record_ab_test_result(test_id: str, config_id: str, metrics: Dict[str, Any],
                         task: Task, rollout: Optional[Rollout] = None):
    """Record a result for an A/B test"""
    ab_test_manager.record_test_result(test_id, config_id, metrics, task, rollout)