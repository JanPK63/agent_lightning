#!/usr/bin/env python3
"""
Model Versioning and Performance Tracking for Agent Lightning

This module provides comprehensive model versioning, performance tracking,
and management capabilities including:
- Semantic versioning for models
- Performance metrics tracking and comparison
- Model lineage and dependency tracking
- Automated model evaluation pipelines
- A/B testing framework for models
- Model registry and deployment management
- Performance degradation detection and alerting
"""

import os
import json
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import threading
import queue

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class EvaluationMetric(Enum):
    """Standard evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LOSS = "loss"
    PERPLEXITY = "perplexity"
    BLEU_SCORE = "bleu_score"
    ROUGE_SCORE = "rouge_score"
    CUSTOM = "custom"


@dataclass
class ModelVersion:
    """Represents a versioned model"""
    model_id: str
    version: str  # Semantic version (e.g., "1.2.3")
    status: ModelStatus = ModelStatus.TRAINING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Model metadata
    model_type: str = ""  # e.g., "llm", "rl_agent", "classifier"
    framework: str = ""  # e.g., "pytorch", "tensorflow", "transformers"
    architecture: str = ""  # e.g., "gpt-2", "ppo_agent"

    # Training metadata
    training_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_info: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)

    # Model artifacts
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None

    # Lineage and dependencies
    parent_version: Optional[str] = None
    child_versions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other model IDs this depends on

    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    ab_test_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.version:
            self.version = self._generate_initial_version()

    def _generate_initial_version(self) -> str:
        """Generate initial semantic version"""
        timestamp = int(time.time())
        return f"0.1.{timestamp}"

    def bump_version(self, bump_type: str = "patch") -> str:
        """Bump version according to semantic versioning"""
        major, minor, patch = map(int, self.version.split('.'))

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1

        self.version = f"{major}.{minor}.{patch}"
        self.updated_at = datetime.now()
        return self.version

    def update_performance(self, metrics: Dict[str, float], context: Optional[Dict[str, Any]] = None):
        """Update model performance metrics"""
        self.performance_metrics.update(metrics)
        self.updated_at = datetime.now()

        # Add to performance history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy(),
            "context": context or {}
        }
        self.performance_history.append(history_entry)

        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_performance_trend(self, metric: str, window: int = 10) -> Dict[str, Any]:
        """Get performance trend for a specific metric"""
        recent_history = self.performance_history[-window:]

        if not recent_history:
            return {"trend": "insufficient_data", "change": 0.0}

        values = [entry["metrics"].get(metric, 0) for entry in recent_history if metric in entry["metrics"]]

        if len(values) < 2:
            return {"trend": "insufficient_data", "change": 0.0}

        # Calculate trend
        start_value = values[0]
        end_value = values[-1]
        change = end_value - start_value
        trend = "improving" if change > 0 else "degrading" if change < 0 else "stable"

        return {
            "trend": trend,
            "change": change,
            "change_percent": (change / start_value) * 100 if start_value != 0 else 0,
            "start_value": start_value,
            "end_value": end_value,
            "samples": len(values)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "model_type": self.model_type,
            "framework": self.framework,
            "architecture": self.architecture,
            "training_config": self.training_config,
            "hyperparameters": self.hyperparameters,
            "training_data_info": self.training_data_info,
            "performance_metrics": self.performance_metrics,
            "evaluation_results": self.evaluation_results,
            "model_path": self.model_path,
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "parent_version": self.parent_version,
            "child_versions": self.child_versions,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "metadata": self.metadata,
            "performance_history": self.performance_history,
            "ab_test_results": self.ab_test_results
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        # Convert string dates back to datetime
        data_copy = data.copy()
        data_copy["created_at"] = datetime.fromisoformat(data["created_at"])
        data_copy["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data_copy["status"] = ModelStatus(data["status"])
        return cls(**data_copy)


@dataclass
class ABTest:
    """A/B test configuration and results"""
    test_id: str
    name: str
    description: str
    model_a: str  # Model version ID
    model_b: str  # Model version ID
    metric: str  # Primary metric to compare
    test_duration_hours: int = 24
    traffic_split: float = 0.5  # Percentage of traffic to model B
    status: str = "running"  # running, completed, stopped

    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Results
    model_a_results: List[Dict[str, Any]] = field(default_factory=list)
    model_b_results: List[Dict[str, Any]] = field(default_factory=list)
    statistical_significance: Optional[float] = None
    winner: Optional[str] = None  # "A", "B", or "tie"

    def add_result(self, model_version: str, metrics: Dict[str, Any], context: Dict[str, Any]):
        """Add a test result"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "context": context
        }

        if model_version == self.model_a:
            self.model_a_results.append(result)
        elif model_version == self.model_b:
            self.model_b_results.append(result)

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if len(self.model_a_results) < 10 or len(self.model_b_results) < 10:
            return {"status": "insufficient_data"}

        # Extract primary metric
        a_values = [r["metrics"].get(self.metric, 0) for r in self.model_a_results]
        b_values = [r["metrics"].get(self.metric, 0) for r in self.model_b_results]

        if not a_values or not b_values:
            return {"status": "no_data"}

        # Simple statistical analysis
        a_mean = sum(a_values) / len(a_values)
        b_mean = sum(b_values) / len(b_values)

        # Determine winner
        if abs(a_mean - b_mean) / max(a_mean, b_mean) < 0.05:  # 5% threshold
            winner = "tie"
        elif b_mean > a_mean:
            winner = "B"
        else:
            winner = "A"

        return {
            "status": "completed",
            "model_a_mean": a_mean,
            "model_b_mean": b_mean,
            "improvement": ((b_mean - a_mean) / a_mean) * 100 if a_mean != 0 else 0,
            "winner": winner,
            "confidence": "high" if len(a_values) > 50 else "medium"
        }


class ModelEvaluator(ABC):
    """Abstract base class for model evaluators"""

    @abstractmethod
    async def evaluate(self, model_version: ModelVersion, test_data: Any) -> Dict[str, Any]:
        """Evaluate a model version"""
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported evaluation metrics"""
        pass


class LLMModelEvaluator(ModelEvaluator):
    """Evaluator for LLM models"""

    async def evaluate(self, model_version: ModelVersion, test_data: Any) -> Dict[str, Any]:
        """Evaluate LLM model"""
        # This would integrate with the LLM providers
        # For now, return mock results
        return {
            "perplexity": 15.3,
            "accuracy": 0.87,
            "response_time": 0.234,
            "token_efficiency": 0.92
        }

    def get_supported_metrics(self) -> List[str]:
        return ["perplexity", "accuracy", "response_time", "token_efficiency"]


class RLModelEvaluator(ModelEvaluator):
    """Evaluator for RL agent models"""

    async def evaluate(self, model_version: ModelVersion, test_data: Any) -> Dict[str, Any]:
        """Evaluate RL model"""
        # This would run the agent in evaluation mode
        # For now, return mock results
        return {
            "average_reward": 125.7,
            "success_rate": 0.78,
            "episode_length": 45.2,
            "convergence_speed": 0.89
        }

    def get_supported_metrics(self) -> List[str]:
        return ["average_reward", "success_rate", "episode_length", "convergence_speed"]


class ModelRegistry:
    """Central registry for model versions and management"""

    def __init__(self, storage_path: str = "./model_registry"):
        self.storage_path = storage_path
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # model_id -> version -> ModelVersion
        self.ab_tests: Dict[str, ABTest] = {}
        self.evaluators: Dict[str, ModelEvaluator] = {}

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        # Register default evaluators
        self.register_evaluator("llm", LLMModelEvaluator())
        self.register_evaluator("rl_agent", RLModelEvaluator())

        # Load existing models
        self._load_registry()

    def register_evaluator(self, model_type: str, evaluator: ModelEvaluator):
        """Register a model evaluator"""
        self.evaluators[model_type] = evaluator

    def create_model_version(self, model_id: str, model_type: str,
                           framework: str, architecture: str,
                           parent_version: Optional[str] = None) -> ModelVersion:
        """Create a new model version"""
        if model_id not in self.models:
            self.models[model_id] = {}

        # Determine next version
        if parent_version and parent_version in self.models[model_id]:
            parent = self.models[model_id][parent_version]
            version = parent.bump_version("minor")
        else:
            # Find latest version
            versions = list(self.models[model_id].keys())
            if versions:
                latest = max(versions, key=lambda v: tuple(map(int, v.split('.'))))
                major, minor, patch = map(int, latest.split('.'))
                version = f"{major}.{minor}.{patch + 1}"
            else:
                version = "1.0.0"

        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            framework=framework,
            architecture=architecture,
            parent_version=parent_version
        )

        # Update parent-child relationships
        if parent_version and parent_version in self.models[model_id]:
            self.models[model_id][parent_version].child_versions.append(version)

        self.models[model_id][version] = model_version
        self._save_model_version(model_version)

        logger.info(f"Created model version: {model_id}@{version}")
        return model_version

    def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version"""
        return self.models.get(model_id, {}).get(version)

    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = self.models.get(model_id, {})
        if not versions:
            return None

        latest_version = max(versions.keys(), key=lambda v: tuple(map(int, v.split('.'))))
        return versions[latest_version]

    def list_model_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model"""
        return list(self.models.get(model_id, {}).values())

    def update_model_performance(self, model_id: str, version: str,
                               metrics: Dict[str, float], context: Optional[Dict[str, Any]] = None):
        """Update model performance metrics"""
        model_version = self.get_model_version(model_id, version)
        if model_version:
            model_version.update_performance(metrics, context)
            self._save_model_version(model_version)

    async def evaluate_model(self, model_id: str, version: str, test_data: Any) -> Dict[str, Any]:
        """Evaluate a model version"""
        model_version = self.get_model_version(model_id, version)
        if not model_version:
            raise ValueError(f"Model version not found: {model_id}@{version}")

        evaluator = self.evaluators.get(model_version.model_type)
        if not evaluator:
            raise ValueError(f"No evaluator available for model type: {model_version.model_type}")

        results = await evaluator.evaluate(model_version, test_data)

        # Update model with results
        model_version.evaluation_results.update(results)
        model_version.status = ModelStatus.READY
        self._save_model_version(model_version)

        return results

    def start_ab_test(self, test_id: str, name: str, description: str,
                     model_a: str, model_b: str, metric: str,
                     test_duration_hours: int = 24) -> ABTest:
        """Start an A/B test between two model versions"""
        ab_test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            model_a=model_a,
            model_b=model_b,
            metric=metric,
            test_duration_hours=test_duration_hours
        )

        self.ab_tests[test_id] = ab_test
        self._save_ab_test(ab_test)

        logger.info(f"Started A/B test: {test_id}")
        return ab_test

    def record_ab_test_result(self, test_id: str, model_version: str,
                            metrics: Dict[str, Any], context: Dict[str, Any]):
        """Record a result for an A/B test"""
        if test_id in self.ab_tests:
            self.ab_tests[test_id].add_result(model_version, metrics, context)
            self._save_ab_test(self.ab_tests[test_id])

    def get_ab_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results"""
        ab_test = self.ab_tests.get(test_id)
        if not ab_test:
            return None

        return ab_test.analyze_results()

    def detect_performance_degradation(self, model_id: str, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect performance degradation in models"""
        alerts = []

        for version, model_version in self.models.get(model_id, {}).items():
            if model_version.status == ModelStatus.DEPLOYED:
                for metric in model_version.performance_metrics.keys():
                    trend = model_version.get_performance_trend(metric)
                    if trend["trend"] == "degrading" and abs(trend["change_percent"]) > threshold * 100:
                        alerts.append({
                            "model_id": model_id,
                            "version": version,
                            "metric": metric,
                            "degradation": trend["change_percent"],
                            "severity": "high" if trend["change_percent"] > 20 else "medium"
                        })

        return alerts

    def _save_model_version(self, model_version: ModelVersion):
        """Save model version to disk"""
        model_dir = os.path.join(self.storage_path, model_version.model_id)
        os.makedirs(model_dir, exist_ok=True)

        version_file = os.path.join(model_dir, f"{model_version.version}.json")
        with open(version_file, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)

    def _save_ab_test(self, ab_test: ABTest):
        """Save A/B test to disk"""
        test_file = os.path.join(self.storage_path, "ab_tests", f"{ab_test.test_id}.json")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)

        with open(test_file, 'w') as f:
            json.dump({
                "test_id": ab_test.test_id,
                "name": ab_test.name,
                "description": ab_test.description,
                "model_a": ab_test.model_a,
                "model_b": ab_test.model_b,
                "metric": ab_test.metric,
                "test_duration_hours": ab_test.test_duration_hours,
                "status": ab_test.status,
                "created_at": ab_test.created_at.isoformat(),
                "completed_at": ab_test.completed_at.isoformat() if ab_test.completed_at else None,
                "model_a_results": ab_test.model_a_results,
                "model_b_results": ab_test.model_b_results,
                "statistical_significance": ab_test.statistical_significance,
                "winner": ab_test.winner
            }, f, indent=2)

    def _load_registry(self):
        """Load existing models from disk"""
        if not os.path.exists(self.storage_path):
            return

        # Load model versions
        for model_dir in os.listdir(self.storage_path):
            model_path = os.path.join(self.storage_path, model_dir)
            if os.path.isdir(model_path) and not model_dir.startswith('.'):
                model_id = model_dir
                self.models[model_id] = {}

                for version_file in os.listdir(model_path):
                    if version_file.endswith('.json'):
                        version_path = os.path.join(model_path, version_file)
                        try:
                            with open(version_path, 'r') as f:
                                data = json.load(f)
                            model_version = ModelVersion.from_dict(data)
                            self.models[model_id][model_version.version] = model_version
                        except Exception as e:
                            logger.warning(f"Failed to load model version {version_path}: {e}")

        # Load A/B tests
        ab_test_dir = os.path.join(self.storage_path, "ab_tests")
        if os.path.exists(ab_test_dir):
            for test_file in os.listdir(ab_test_dir):
                if test_file.endswith('.json'):
                    test_path = os.path.join(ab_test_dir, test_file)
                    try:
                        with open(test_path, 'r') as f:
                            data = json.load(f)
                        # Reconstruct ABTest object
                        ab_test = ABTest(
                            test_id=data["test_id"],
                            name=data["name"],
                            description=data["description"],
                            model_a=data["model_a"],
                            model_b=data["model_b"],
                            metric=data["metric"],
                            test_duration_hours=data["test_duration_hours"],
                            status=data["status"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                            statistical_significance=data.get("statistical_significance"),
                            winner=data.get("winner")
                        )
                        ab_test.model_a_results = data.get("model_a_results", [])
                        ab_test.model_b_results = data.get("model_b_results", [])
                        self.ab_tests[ab_test.test_id] = ab_test
                    except Exception as e:
                        logger.warning(f"Failed to load A/B test {test_path}: {e}")


# Global model registry instance
model_registry = ModelRegistry()


def create_model_version(model_id: str, model_type: str, framework: str,
                        architecture: str, parent_version: Optional[str] = None) -> ModelVersion:
    """Convenience function to create a model version"""
    return model_registry.create_model_version(model_id, model_type, framework, architecture, parent_version)


def get_model_version(model_id: str, version: str) -> Optional[ModelVersion]:
    """Convenience function to get a model version"""
    return model_registry.get_model_version(model_id, version)


def update_model_performance(model_id: str, version: str, metrics: Dict[str, float],
                           context: Optional[Dict[str, Any]] = None):
    """Convenience function to update model performance"""
    model_registry.update_model_performance(model_id, version, metrics, context)


async def evaluate_model(model_id: str, version: str, test_data: Any) -> Dict[str, Any]:
    """Convenience function to evaluate a model"""
    return await model_registry.evaluate_model(model_id, version, test_data)