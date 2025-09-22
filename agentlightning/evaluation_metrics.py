#!/usr/bin/env python3
"""
Comprehensive Offline Evaluation Metrics for Agent Lightning

This module provides extensive evaluation capabilities for agents including:
- Standard ML and NLP metrics
- Agent-specific performance metrics
- Statistical analysis and significance testing
- Comparative evaluation across configurations
- Automated metric collection and alerting
- Custom metric definitions and plugins
"""

import logging
import time
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import json
import os

from agentlightning.types import Task, Rollout, Triplet

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TASK_COMPLETION = "task_completion"
    RESPONSE_QUALITY = "response_quality"
    ROBUSTNESS = "robustness"
    RESOURCE_USAGE = "resource_usage"
    CUSTOM = "custom"


class MetricCategory(Enum):
    """Categories of metrics"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    BUSINESS = "business"


@dataclass
class MetricResult:
    """Result of a metric evaluation"""
    metric_name: str
    metric_type: MetricType
    category: MetricCategory
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "category": self.category.value,
            "value": self.value,
            "confidence_interval": self.confidence_interval,
            "sample_size": self.sample_size,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model/configuration"""
    entity_id: str  # model_id, config_id, etc.
    entity_type: str  # "model", "configuration", etc.
    metrics: List[MetricResult] = field(default_factory=list)
    overall_score: Optional[float] = None
    grade: str = "unknown"  # A, B, C, D, F
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_metric(self, metric: MetricResult):
        """Add a metric result"""
        self.metrics.append(metric)

    def get_metric(self, metric_name: str) -> Optional[MetricResult]:
        """Get a specific metric by name"""
        for metric in self.metrics:
            if metric.metric_name == metric_name:
                return metric
        return None

    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall score from metrics"""
        if not self.metrics:
            return 0.0

        if weights is None:
            # Default weights
            weights = {
                "accuracy": 0.3,
                "f1_score": 0.25,
                "latency": -0.2,  # Negative because lower is better
                "task_completion": 0.25
            }

        total_weight = 0.0
        weighted_sum = 0.0

        for metric in self.metrics:
            if metric.metric_name in weights:
                weight = weights[metric.metric_name]
                # Normalize latency (invert if negative weight)
                value = metric.value
                if weight < 0 and value > 0:
                    value = 1.0 / (1.0 + value)  # Convert latency to efficiency score

                weighted_sum += value * abs(weight)
                total_weight += abs(weight)

        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return self.overall_score

    def assign_grade(self) -> str:
        """Assign a grade based on overall score"""
        if self.overall_score is None:
            self.grade = "unknown"
        elif self.overall_score >= 0.9:
            self.grade = "A"
        elif self.overall_score >= 0.8:
            self.grade = "B"
        elif self.overall_score >= 0.7:
            self.grade = "C"
        elif self.overall_score >= 0.6:
            self.grade = "D"
        else:
            self.grade = "F"
        return self.grade


class MetricCalculator(ABC):
    """Abstract base class for metric calculators"""

    @abstractmethod
    def calculate(self, data: Any, **kwargs) -> MetricResult:
        """Calculate the metric"""
        pass

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Get the metric type"""
        pass

    @property
    @abstractmethod
    def category(self) -> MetricCategory:
        """Get the metric category"""
        pass


class AccuracyCalculator(MetricCalculator):
    """Calculate accuracy metric"""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.ACCURACY

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.QUALITY

    def calculate(self, predictions: List[Any], ground_truth: List[Any], **kwargs) -> MetricResult:
        """Calculate accuracy"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        accuracy = correct / len(predictions) if predictions else 0.0

        return MetricResult(
            metric_name="accuracy",
            metric_type=self.metric_type,
            category=self.category,
            value=accuracy,
            sample_size=len(predictions)
        )


class F1ScoreCalculator(MetricCalculator):
    """Calculate F1 score for binary classification"""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.F1_SCORE

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.QUALITY

    def calculate(self, predictions: List[Any], ground_truth: List[Any], **kwargs) -> MetricResult:
        """Calculate F1 score"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        # Assume binary classification with positive class = 1
        tp = sum(1 for p, gt in zip(predictions, ground_truth) if p == 1 and gt == 1)
        fp = sum(1 for p, gt in zip(predictions, ground_truth) if p == 1 and gt == 0)
        fn = sum(1 for p, gt in zip(predictions, ground_truth) if p == 0 and gt == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            metric_name="f1_score",
            metric_type=self.metric_type,
            category=self.category,
            value=f1,
            sample_size=len(predictions),
            metadata={
                "precision": precision,
                "recall": recall,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            }
        )


class LatencyCalculator(MetricCalculator):
    """Calculate latency metrics"""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.LATENCY

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.PERFORMANCE

    def calculate(self, latencies: List[float], **kwargs) -> MetricResult:
        """Calculate latency statistics"""
        if not latencies:
            return MetricResult(
                metric_name="latency",
                metric_type=self.metric_type,
                category=self.category,
                value=0.0,
                sample_size=0
            )

        mean_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        return MetricResult(
            metric_name="latency",
            metric_type=self.metric_type,
            category=self.category,
            value=mean_latency,
            sample_size=len(latencies),
            metadata={
                "p50_latency": np.percentile(latencies, 50),
                "p95_latency": p95_latency,
                "p99_latency": p99_latency,
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            }
        )


class TaskCompletionCalculator(MetricCalculator):
    """Calculate task completion rate"""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TASK_COMPLETION

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.PERFORMANCE

    def calculate(self, rollouts: List[Rollout], **kwargs) -> MetricResult:
        """Calculate task completion rate from rollouts"""
        if not rollouts:
            return MetricResult(
                metric_name="task_completion",
                metric_type=self.metric_type,
                category=self.category,
                value=0.0,
                sample_size=0
            )

        # Define completion criteria (customizable)
        completion_threshold = kwargs.get('completion_threshold', 0.5)

        completed = sum(1 for rollout in rollouts
                       if rollout.final_reward and rollout.final_reward >= completion_threshold)

        completion_rate = completed / len(rollouts)

        return MetricResult(
            metric_name="task_completion",
            metric_type=self.metric_type,
            category=self.category,
            value=completion_rate,
            sample_size=len(rollouts),
            metadata={
                "completed_tasks": completed,
                "total_tasks": len(rollouts),
                "completion_threshold": completion_threshold
            }
        )


class ResponseQualityCalculator(MetricCalculator):
    """Calculate response quality metrics"""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.RESPONSE_QUALITY

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.QUALITY

    def calculate(self, responses: List[str], criteria: Optional[Dict[str, Any]] = None, **kwargs) -> MetricResult:
        """Calculate response quality score"""
        if not responses:
            return MetricResult(
                metric_name="response_quality",
                metric_type=self.metric_type,
                category=self.category,
                value=0.0,
                sample_size=0
            )

        # Simple quality heuristics (can be enhanced with ML models)
        total_score = 0.0

        for response in responses:
            score = 0.0

            # Length appropriateness
            if 10 <= len(response) <= 500:
                score += 0.3

            # Has some structure (sentences, paragraphs)
            if '.' in response or '\n' in response:
                score += 0.3

            # Not too repetitive
            words = response.lower().split()
            unique_words = set(words)
            if len(unique_words) / len(words) > 0.3:
                score += 0.2

            # Contains some actionable content
            action_words = ['should', 'can', 'will', 'use', 'try', 'consider']
            if any(word in response.lower() for word in action_words):
                score += 0.2

            total_score += score

        avg_quality = total_score / len(responses)

        return MetricResult(
            metric_name="response_quality",
            metric_type=self.metric_type,
            category=self.category,
            value=avg_quality,
            sample_size=len(responses),
            metadata={
                "quality_components": ["length", "structure", "uniqueness", "actionability"],
                "scoring_method": "heuristic"
            }
        )


class RobustnessCalculator(MetricCalculator):
    """Calculate robustness metrics"""

    @property
    def metric_type(self) -> MetricType:
        return MetricType.ROBUSTNESS

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.ROBUSTNESS

    def calculate(self, results: List[Dict[str, Any]], **kwargs) -> MetricResult:
        """Calculate robustness from result patterns"""
        if not results:
            return MetricResult(
                metric_name="robustness",
                metric_type=self.metric_type,
                category=self.category,
                value=0.0,
                sample_size=0
            )

        # Analyze failure patterns
        failures = sum(1 for r in results if r.get('success', False) == False)
        total = len(results)

        # Calculate consistency (lower variance in performance)
        performances = [r.get('performance', 0.0) for r in results if 'performance' in r]
        consistency = 1.0 - (statistics.stdev(performances) / statistics.mean(performances)
                           if performances and statistics.mean(performances) > 0 else 1.0)

        # Recovery rate (successful recoveries after failures)
        recoveries = sum(1 for i, r in enumerate(results[:-1])
                        if not r.get('success', True) and results[i+1].get('success', False))

        robustness_score = (
            (1.0 - failures/total) * 0.4 +  # Low failure rate
            consistency * 0.4 +              # Consistent performance
            (recoveries / failures if failures > 0 else 1.0) * 0.2  # Recovery capability
        )

        return MetricResult(
            metric_name="robustness",
            metric_type=self.metric_type,
            category=self.category,
            value=robustness_score,
            sample_size=total,
            metadata={
                "failure_rate": failures/total,
                "consistency_score": consistency,
                "recovery_rate": recoveries / failures if failures > 0 else 1.0,
                "total_failures": failures
            }
        )


class EvaluationEngine:
    """Main evaluation engine for comprehensive offline evaluation"""

    def __init__(self):
        self.calculators: Dict[str, MetricCalculator] = {}
        self.custom_metrics: Dict[str, Callable] = {}
        self.evaluation_history: List[EvaluationResult] = []

        # Register built-in calculators
        self._register_calculator(AccuracyCalculator())
        self._register_calculator(F1ScoreCalculator())
        self._register_calculator(LatencyCalculator())
        self._register_calculator(TaskCompletionCalculator())
        self._register_calculator(ResponseQualityCalculator())
        self._register_calculator(RobustnessCalculator())

    def _register_calculator(self, calculator: MetricCalculator):
        """Register a metric calculator"""
        self.calculators[calculator.metric_type.value] = calculator

    def _compute_metric(self, calculator: MetricCalculator, metric_name: str,
                       evaluation_data: Dict[str, Any]) -> Optional[MetricResult]:
        """Compute a specific metric with appropriate data extraction"""
        try:
            # Extract data based on calculator type
            if isinstance(calculator, (AccuracyCalculator, F1ScoreCalculator)):
                predictions = evaluation_data.get('predictions', [])
                ground_truth = evaluation_data.get('ground_truth', [])
                if predictions and ground_truth:
                    return calculator.calculate(predictions, ground_truth)
                else:
                    logger.warning(f"No predictions/ground_truth data for {metric_name}")
                    return None

            elif isinstance(calculator, LatencyCalculator):
                latencies = evaluation_data.get('latencies', [])
                if latencies:
                    return calculator.calculate(latencies)
                else:
                    logger.warning(f"No latency data for {metric_name}")
                    return None

            elif isinstance(calculator, TaskCompletionCalculator):
                rollouts = evaluation_data.get('rollouts', [])
                if rollouts:
                    return calculator.calculate(rollouts)
                else:
                    logger.warning(f"No rollout data for {metric_name}")
                    return None

            elif isinstance(calculator, ResponseQualityCalculator):
                responses = evaluation_data.get('responses', [])
                if responses:
                    return calculator.calculate(responses)
                else:
                    logger.warning(f"No response data for {metric_name}")
                    return None

            elif isinstance(calculator, RobustnessCalculator):
                results = evaluation_data.get('results', [])
                if results:
                    return calculator.calculate(results)
                else:
                    logger.warning(f"No results data for {metric_name}")
                    return None

            else:
                # Fallback: try to call with evaluation_data directly
                return calculator.calculate(evaluation_data)

        except Exception as e:
            logger.error(f"Error computing metric {metric_name}: {e}")
            return None

    def add_custom_metric(self, name: str, calculator: Callable[[Any], MetricResult]):
        """Add a custom metric calculator"""
        self.custom_metrics[name] = calculator

    def evaluate_model(self, model_id: str, evaluation_data: Dict[str, Any],
                       metrics_to_compute: Optional[List[str]] = None) -> EvaluationResult:
        """Evaluate a model with comprehensive metrics"""
        result = EvaluationResult(
            entity_id=model_id,
            entity_type="model"
        )

        # Default metrics if none specified
        if metrics_to_compute is None:
            metrics_to_compute = ["accuracy", "f1_score", "latency", "task_completion",
                                "response_quality", "robustness"]

        # Compute each requested metric
        for metric_name in metrics_to_compute:
            try:
                if metric_name in self.calculators:
                    calculator = self.calculators[metric_name]
                    # Extract appropriate data for each calculator
                    metric_result = self._compute_metric(calculator, metric_name, evaluation_data)
                    if metric_result:
                        result.add_metric(metric_result)

                elif metric_name in self.custom_metrics:
                    metric_result = self.custom_metrics[metric_name](evaluation_data)
                    result.add_metric(metric_result)

                else:
                    logger.warning(f"Unknown metric: {metric_name}")

            except Exception as e:
                logger.error(f"Failed to compute metric {metric_name}: {e}")
                # Add failed metric with 0 value
                result.add_metric(MetricResult(
                    metric_name=metric_name,
                    metric_type=MetricType.CUSTOM,
                    category=MetricCategory.PERFORMANCE,
                    value=0.0,
                    metadata={"error": str(e)}
                ))

        # Calculate overall score and grade
        result.calculate_overall_score()
        result.assign_grade()

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        self.evaluation_history.append(result)
        return result

    def evaluate_configuration(self, config_id: str, evaluation_data: Dict[str, Any],
                             metrics_to_compute: Optional[List[str]] = None) -> EvaluationResult:
        """Evaluate an agent configuration"""
        result = EvaluationResult(
            entity_id=config_id,
            entity_type="configuration"
        )

        # Similar to model evaluation but may have different default metrics
        if metrics_to_compute is None:
            metrics_to_compute = ["task_completion", "response_quality", "latency", "robustness"]

        # Reuse the same evaluation logic
        return self.evaluate_model(config_id, evaluation_data, metrics_to_compute)

    def compare_evaluations(self, eval_a: EvaluationResult, eval_b: EvaluationResult,
                          primary_metric: str = "overall_score") -> Dict[str, Any]:
        """Compare two evaluation results"""
        comparison = {
            "entity_a": eval_a.entity_id,
            "entity_b": eval_b.entity_id,
            "primary_metric": primary_metric,
            "winner": None,
            "improvement": 0.0,
            "significant": False
        }

        metric_a = eval_a.get_metric(primary_metric)
        metric_b = eval_b.get_metric(primary_metric)

        if metric_a and metric_b:
            val_a = metric_a.value
            val_b = metric_b.value

            if val_b > val_a:
                comparison["winner"] = "B"
                comparison["improvement"] = ((val_b - val_a) / val_a) * 100 if val_a != 0 else 0
            elif val_a > val_b:
                comparison["winner"] = "A"
                comparison["improvement"] = ((val_a - val_b) / val_b) * 100 if val_b != 0 else 0
            else:
                comparison["winner"] = "tie"

            # Simple significance test (can be enhanced)
            if abs(val_a - val_b) > 0.05:  # 5% difference threshold
                comparison["significant"] = True

        return comparison

    def detect_anomalies(self, recent_evaluations: List[EvaluationResult],
                        metric_name: str, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical methods"""
        anomalies = []

        if len(recent_evaluations) < 5:
            return anomalies

        # Extract metric values
        values = []
        for eval_result in recent_evaluations:
            metric = eval_result.get_metric(metric_name)
            if metric:
                values.append(metric.value)

        if len(values) < 5:
            return anomalies

        # Calculate rolling statistics
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)

        # Check recent values for anomalies
        for i, eval_result in enumerate(recent_evaluations[-3:]):  # Check last 3
            metric = eval_result.get_metric(metric_name)
            if metric:
                z_score = abs(metric.value - mean) / stdev if stdev > 0 else 0
                if z_score > threshold:
                    anomalies.append({
                        "entity_id": eval_result.entity_id,
                        "metric_name": metric_name,
                        "value": metric.value,
                        "expected_range": (mean - stdev, mean + stdev),
                        "z_score": z_score,
                        "severity": "high" if z_score > 3 else "medium",
                        "timestamp": eval_result.timestamp.isoformat()
                    })

        return anomalies

    def generate_report(self, evaluation: EvaluationResult, format: str = "dict") -> Union[Dict, str]:
        """Generate a comprehensive evaluation report"""
        report = {
            "evaluation_summary": {
                "entity_id": evaluation.entity_id,
                "entity_type": evaluation.entity_type,
                "overall_score": evaluation.overall_score,
                "grade": evaluation.grade,
                "timestamp": evaluation.timestamp.isoformat()
            },
            "metrics": [metric.to_dict() for metric in evaluation.metrics],
            "recommendations": evaluation.recommendations,
            "insights": self._generate_insights(evaluation)
        }

        if format == "json":
            return json.dumps(report, indent=2)
        return report

    def _generate_recommendations(self, evaluation: EvaluationResult) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []

        # Check for poor performance metrics
        for metric in evaluation.metrics:
            if metric.value < 0.5:  # Threshold for concerning metrics
                if metric.metric_name == "latency":
                    recommendations.append("Consider optimizing model inference speed or using a smaller model")
                elif metric.metric_name == "accuracy":
                    recommendations.append("Model accuracy is low - consider additional training or data augmentation")
                elif metric.metric_name == "task_completion":
                    recommendations.append("Task completion rate needs improvement - review agent logic and training")
                elif metric.metric_name == "robustness":
                    recommendations.append("Agent robustness is concerning - add better error handling and recovery mechanisms")

        # Overall grade-based recommendations
        if evaluation.grade in ["D", "F"]:
            recommendations.append("Major improvements needed - consider fundamental changes to approach")
        elif evaluation.grade == "C":
            recommendations.append("Moderate improvements recommended - focus on key weak areas")

        if not recommendations:
            recommendations.append("Performance looks good - continue monitoring and incremental improvements")

        return recommendations

    def _generate_insights(self, evaluation: EvaluationResult) -> List[str]:
        """Generate insights from evaluation data"""
        insights = []

        # Find best and worst performing metrics
        if evaluation.metrics:
            sorted_metrics = sorted(evaluation.metrics, key=lambda m: m.value, reverse=True)
            best_metric = sorted_metrics[0]
            worst_metric = sorted_metrics[-1]

            insights.append(f"Best performing metric: {best_metric.metric_name} ({best_metric.value:.3f})")
            insights.append(f"Area for improvement: {worst_metric.metric_name} ({worst_metric.value:.3f})")

        # Check for trade-offs
        latency_metric = evaluation.get_metric("latency")
        accuracy_metric = evaluation.get_metric("accuracy")

        if latency_metric and accuracy_metric:
            if latency_metric.value > 1.0 and accuracy_metric.value > 0.8:
                insights.append("Good accuracy-latency trade-off achieved")
            elif latency_metric.value > 2.0 and accuracy_metric.value < 0.7:
                insights.append("Consider optimizing for speed vs accuracy balance")

        return insights


# Global evaluation engine instance
evaluation_engine = EvaluationEngine()


def evaluate_model_comprehensive(model_id: str, evaluation_data: Dict[str, Any]) -> EvaluationResult:
    """Convenience function for comprehensive model evaluation"""
    return evaluation_engine.evaluate_model(model_id, evaluation_data)


def evaluate_configuration_comprehensive(config_id: str, evaluation_data: Dict[str, Any]) -> EvaluationResult:
    """Convenience function for comprehensive configuration evaluation"""
    return evaluation_engine.evaluate_configuration(config_id, evaluation_data)


def compare_model_performance(eval_a: EvaluationResult, eval_b: EvaluationResult) -> Dict[str, Any]:
    """Convenience function to compare two evaluations"""
    return evaluation_engine.compare_evaluations(eval_a, eval_b)


def generate_evaluation_report(evaluation: EvaluationResult) -> Dict:
    """Convenience function to generate evaluation report"""
    return evaluation_engine.generate_report(evaluation)