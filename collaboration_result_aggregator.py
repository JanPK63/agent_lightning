#!/usr/bin/env python3
"""
Collaboration Result Aggregation System for Multi-Agent Collaboration
Collects, merges, validates, and synthesizes results from multiple agents
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, Counter
import statistics
import hashlib
import numpy as np
from abc import ABC, abstractmethod
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_collaboration import (
    CollaborativeTask,
    CollaborationMode,
    AgentRole,
    TaskComplexity
)
from agent_communication_protocol import AgentMessage, Performative

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultType(Enum):
    """Types of results from agents"""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    ANALYSIS = "analysis"
    DECISION = "decision"
    METRICS = "metrics"
    ERROR = "error"
    PARTIAL = "partial"


class AggregationStrategy(Enum):
    """Strategies for aggregating results"""
    CONCATENATE = auto()      # Simple concatenation
    MERGE = auto()             # Smart merging with deduplication
    CONSENSUS = auto()         # Voting/consensus based
    WEIGHTED = auto()          # Weighted by agent confidence
    HIERARCHICAL = auto()      # Hierarchical aggregation
    STATISTICAL = auto()       # Statistical aggregation
    ENSEMBLE = auto()          # Ensemble methods
    CUSTOM = auto()            # Custom aggregation logic


class ConflictResolution(Enum):
    """Strategies for resolving conflicts"""
    MAJORITY_VOTE = auto()
    WEIGHTED_VOTE = auto()
    EXPERT_OPINION = auto()
    AVERAGE = auto()
    MEDIAN = auto()
    MOST_RECENT = auto()
    HIGHEST_CONFIDENCE = auto()
    MANUAL_REVIEW = auto()


@dataclass
class AgentResult:
    """Result from a single agent"""
    agent_id: str
    role: AgentRole
    task_id: str
    result_type: ResultType
    content: Any
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def get_hash(self) -> str:
        """Get hash of result content for deduplication"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def is_valid(self) -> bool:
        """Check if result is valid"""
        return (
            self.result_type != ResultType.ERROR and
            self.content is not None and
            self.confidence > 0
        )


@dataclass
class AggregatedResult:
    """Aggregated result from multiple agents"""
    task_id: str
    aggregation_strategy: AggregationStrategy
    contributing_agents: List[str]
    final_result: Any
    confidence_score: float
    consensus_level: float
    conflicts_resolved: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    individual_results: List[AgentResult] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of aggregated result"""
        return {
            "task_id": self.task_id,
            "strategy": self.aggregation_strategy.name,
            "num_agents": len(self.contributing_agents),
            "confidence": self.confidence_score,
            "consensus": self.consensus_level,
            "conflicts": self.conflicts_resolved,
            "timestamp": self.timestamp.isoformat()
        }


class ResultAggregator(ABC):
    """Abstract base class for result aggregators"""
    
    @abstractmethod
    async def aggregate(self, results: List[AgentResult]) -> AggregatedResult:
        """Aggregate results from multiple agents"""
        pass
    
    @abstractmethod
    def can_aggregate(self, results: List[AgentResult]) -> bool:
        """Check if this aggregator can handle the given results"""
        pass


class TextAggregator(ResultAggregator):
    """Aggregator for text results"""
    
    def can_aggregate(self, results: List[AgentResult]) -> bool:
        """Check if all results are text"""
        return all(r.result_type == ResultType.TEXT for r in results)
    
    async def aggregate(self, results: List[AgentResult]) -> AggregatedResult:
        """Aggregate text results"""
        if not results:
            return None
        
        # Group similar texts
        text_groups = defaultdict(list)
        for result in results:
            text_hash = hashlib.md5(str(result.content).encode()).hexdigest()[:8]
            text_groups[text_hash].append(result)
        
        # Find consensus text
        if len(text_groups) == 1:
            # All agents produced same text
            consensus_text = results[0].content
            consensus_level = 1.0
        else:
            # Multiple different texts - use most common or merge
            largest_group = max(text_groups.values(), key=len)
            consensus_text = largest_group[0].content
            consensus_level = len(largest_group) / len(results)
            
            # If low consensus, merge unique parts
            if consensus_level < 0.5:
                all_texts = [r.content for r in results]
                consensus_text = self._merge_texts(all_texts)
        
        # Calculate average confidence
        avg_confidence = statistics.mean(r.confidence for r in results)
        
        return AggregatedResult(
            task_id=results[0].task_id,
            aggregation_strategy=AggregationStrategy.MERGE,
            contributing_agents=[r.agent_id for r in results],
            final_result=consensus_text,
            confidence_score=avg_confidence,
            consensus_level=consensus_level,
            individual_results=results
        )
    
    def _merge_texts(self, texts: List[str]) -> str:
        """Merge multiple text results"""
        # Simple merge - can be enhanced with NLP
        merged = []
        seen_sentences = set()
        
        for text in texts:
            sentences = text.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen_sentences:
                    merged.append(sentence)
                    seen_sentences.add(sentence)
        
        return '. '.join(merged)


class CodeAggregator(ResultAggregator):
    """Aggregator for code results"""
    
    def can_aggregate(self, results: List[AgentResult]) -> bool:
        """Check if all results are code"""
        return all(r.result_type == ResultType.CODE for r in results)
    
    async def aggregate(self, results: List[AgentResult]) -> AggregatedResult:
        """Aggregate code results"""
        if not results:
            return None
        
        # Extract code from results
        code_versions = []
        for result in results:
            if isinstance(result.content, dict) and 'code' in result.content:
                code_versions.append(result.content)
            elif isinstance(result.content, str):
                code_versions.append({'code': result.content, 'language': 'python'})
        
        # Find best code version
        if len(set(cv['code'] for cv in code_versions)) == 1:
            # All agents produced same code
            final_code = code_versions[0]
            consensus_level = 1.0
            conflicts = 0
        else:
            # Different code versions - merge or select best
            final_code = self._merge_code_versions(code_versions, results)
            consensus_level = self._calculate_code_consensus(code_versions)
            conflicts = len(code_versions) - 1
        
        # Calculate weighted confidence
        weighted_confidence = self._calculate_weighted_confidence(results)
        
        return AggregatedResult(
            task_id=results[0].task_id,
            aggregation_strategy=AggregationStrategy.WEIGHTED,
            contributing_agents=[r.agent_id for r in results],
            final_result=final_code,
            confidence_score=weighted_confidence,
            consensus_level=consensus_level,
            conflicts_resolved=conflicts,
            individual_results=results,
            metadata={'code_versions': len(code_versions)}
        )
    
    def _merge_code_versions(self, code_versions: List[Dict], results: List[AgentResult]) -> Dict:
        """Merge different code versions"""
        # For now, select the version from the most confident agent
        # In production, could use AST analysis for smart merging
        best_idx = 0
        best_confidence = 0
        
        for i, result in enumerate(results):
            if result.confidence > best_confidence:
                best_confidence = result.confidence
                best_idx = i
        
        return code_versions[best_idx] if best_idx < len(code_versions) else code_versions[0]
    
    def _calculate_code_consensus(self, code_versions: List[Dict]) -> float:
        """Calculate consensus level for code"""
        if not code_versions:
            return 0.0
        
        # Count unique code versions
        unique_codes = set(cv['code'] for cv in code_versions)
        
        # Find most common version
        code_counts = Counter(cv['code'] for cv in code_versions)
        most_common_count = code_counts.most_common(1)[0][1] if code_counts else 0
        
        return most_common_count / len(code_versions)
    
    def _calculate_weighted_confidence(self, results: List[AgentResult]) -> float:
        """Calculate weighted confidence based on agent roles"""
        role_weights = {
            AgentRole.SPECIALIST: 1.5,
            AgentRole.REVIEWER: 1.3,
            AgentRole.COORDINATOR: 1.2,
            AgentRole.WORKER: 1.0,
            AgentRole.AGGREGATOR: 1.1,
            AgentRole.MONITOR: 0.9
        }
        
        weighted_sum = 0
        weight_total = 0
        
        for result in results:
            weight = role_weights.get(result.role, 1.0)
            weighted_sum += result.confidence * weight
            weight_total += weight
        
        return weighted_sum / weight_total if weight_total > 0 else 0


class DataAggregator(ResultAggregator):
    """Aggregator for data/metrics results"""
    
    def can_aggregate(self, results: List[AgentResult]) -> bool:
        """Check if all results are data or metrics"""
        return all(r.result_type in [ResultType.DATA, ResultType.METRICS] for r in results)
    
    async def aggregate(self, results: List[AgentResult]) -> AggregatedResult:
        """Aggregate data results"""
        if not results:
            return None
        
        # Collect all data points
        all_data = []
        for result in results:
            if isinstance(result.content, (list, tuple)):
                all_data.extend(result.content)
            elif isinstance(result.content, dict):
                all_data.append(result.content)
            else:
                all_data.append(result.content)
        
        # Perform statistical aggregation
        if all(isinstance(d, (int, float)) for d in all_data):
            # Numerical data
            aggregated_data = {
                'mean': statistics.mean(all_data),
                'median': statistics.median(all_data),
                'std_dev': statistics.stdev(all_data) if len(all_data) > 1 else 0,
                'min': min(all_data),
                'max': max(all_data),
                'count': len(all_data)
            }
            strategy = AggregationStrategy.STATISTICAL
        else:
            # Mixed or non-numerical data
            aggregated_data = self._merge_data_structures(all_data)
            strategy = AggregationStrategy.MERGE
        
        # Calculate consensus
        consensus = self._calculate_data_consensus(results)
        
        return AggregatedResult(
            task_id=results[0].task_id,
            aggregation_strategy=strategy,
            contributing_agents=[r.agent_id for r in results],
            final_result=aggregated_data,
            confidence_score=statistics.mean(r.confidence for r in results),
            consensus_level=consensus,
            individual_results=results
        )
    
    def _merge_data_structures(self, data_list: List[Any]) -> Any:
        """Merge various data structures"""
        if not data_list:
            return None
        
        # If all are dicts, merge them
        if all(isinstance(d, dict) for d in data_list):
            merged = {}
            for data in data_list:
                for key, value in data.items():
                    if key not in merged:
                        merged[key] = value
                    elif isinstance(merged[key], list):
                        if value not in merged[key]:
                            merged[key].append(value)
                    elif merged[key] != value:
                        merged[key] = [merged[key], value]
            return merged
        
        # Otherwise return as list
        return data_list
    
    def _calculate_data_consensus(self, results: List[AgentResult]) -> float:
        """Calculate consensus for data results"""
        if len(results) <= 1:
            return 1.0
        
        # Compare result hashes
        hashes = [r.get_hash() for r in results]
        hash_counts = Counter(hashes)
        most_common_count = hash_counts.most_common(1)[0][1]
        
        return most_common_count / len(results)


class CollaborationResultAggregator:
    """Main aggregator for collaboration results"""
    
    def __init__(self):
        self.aggregators = {
            ResultType.TEXT: TextAggregator(),
            ResultType.CODE: CodeAggregator(),
            ResultType.DATA: DataAggregator(),
            ResultType.METRICS: DataAggregator(),
            ResultType.ANALYSIS: TextAggregator()  # Use text aggregator for analysis
        }
        
        self.aggregation_history: List[AggregatedResult] = []
        self.conflict_log: List[Dict[str, Any]] = []
        self.metrics = defaultdict(int)
    
    async def aggregate_results(
        self,
        results: List[AgentResult],
        strategy: Optional[AggregationStrategy] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.MAJORITY_VOTE
    ) -> AggregatedResult:
        """Aggregate results from multiple agents"""
        
        if not results:
            logger.warning("No results to aggregate")
            return None
        
        logger.info(f"Aggregating {len(results)} results from agents")
        
        # Group results by type
        results_by_type = defaultdict(list)
        for result in results:
            results_by_type[result.result_type].append(result)
        
        # Aggregate each type separately
        aggregated_by_type = {}
        for result_type, typed_results in results_by_type.items():
            if result_type in self.aggregators:
                aggregator = self.aggregators[result_type]
                if aggregator.can_aggregate(typed_results):
                    aggregated = await aggregator.aggregate(typed_results)
                    aggregated_by_type[result_type] = aggregated
                    self.metrics[f"aggregated_{result_type.value}"] += 1
        
        # Combine aggregated results
        if len(aggregated_by_type) == 1:
            # Single type - return as is
            final_result = list(aggregated_by_type.values())[0]
        else:
            # Multiple types - create composite result
            final_result = await self._create_composite_result(
                aggregated_by_type,
                strategy or AggregationStrategy.HIERARCHICAL
            )
        
        # Resolve any remaining conflicts
        if final_result and final_result.conflicts_resolved > 0:
            await self._resolve_conflicts(final_result, conflict_resolution)
        
        # Record in history
        if final_result:
            self.aggregation_history.append(final_result)
            self.metrics["total_aggregations"] += 1
        
        logger.info(f"Aggregation complete. Confidence: {final_result.confidence_score:.2f}, "
                   f"Consensus: {final_result.consensus_level:.2f}")
        
        return final_result
    
    async def _create_composite_result(
        self,
        aggregated_by_type: Dict[ResultType, AggregatedResult],
        strategy: AggregationStrategy
    ) -> AggregatedResult:
        """Create composite result from multiple types"""
        
        # Combine all contributing agents
        all_agents = set()
        all_individual_results = []
        
        for agg_result in aggregated_by_type.values():
            all_agents.update(agg_result.contributing_agents)
            all_individual_results.extend(agg_result.individual_results)
        
        # Create composite final result
        composite = {
            "components": {}
        }
        
        total_confidence = 0
        total_consensus = 0
        total_conflicts = 0
        
        for result_type, agg_result in aggregated_by_type.items():
            composite["components"][result_type.value] = agg_result.final_result
            total_confidence += agg_result.confidence_score
            total_consensus += agg_result.consensus_level
            total_conflicts += agg_result.conflicts_resolved
        
        # Calculate averages
        num_components = len(aggregated_by_type)
        avg_confidence = total_confidence / num_components if num_components > 0 else 0
        avg_consensus = total_consensus / num_components if num_components > 0 else 0
        
        return AggregatedResult(
            task_id=all_individual_results[0].task_id if all_individual_results else "unknown",
            aggregation_strategy=strategy,
            contributing_agents=list(all_agents),
            final_result=composite,
            confidence_score=avg_confidence,
            consensus_level=avg_consensus,
            conflicts_resolved=total_conflicts,
            individual_results=all_individual_results,
            metadata={"num_types": num_components}
        )
    
    async def _resolve_conflicts(
        self,
        result: AggregatedResult,
        resolution_strategy: ConflictResolution
    ):
        """Resolve conflicts in aggregated results"""
        
        if result.conflicts_resolved == 0:
            return
        
        logger.info(f"Resolving {result.conflicts_resolved} conflicts using {resolution_strategy.name}")
        
        # Log conflict
        self.conflict_log.append({
            "timestamp": datetime.now(),
            "task_id": result.task_id,
            "num_conflicts": result.conflicts_resolved,
            "resolution_strategy": resolution_strategy.name,
            "agents_involved": result.contributing_agents
        })
        
        # Apply resolution strategy
        if resolution_strategy == ConflictResolution.MAJORITY_VOTE:
            # Already handled in aggregators
            pass
        elif resolution_strategy == ConflictResolution.HIGHEST_CONFIDENCE:
            # Select result from most confident agent
            if result.individual_results:
                best_result = max(result.individual_results, key=lambda r: r.confidence)
                result.final_result = best_result.content
                result.confidence_score = best_result.confidence
        # Add more resolution strategies as needed
    
    def validate_results(self, results: List[AgentResult]) -> Tuple[List[AgentResult], List[str]]:
        """Validate agent results"""
        valid_results = []
        errors = []
        
        for result in results:
            if not result.agent_id:
                errors.append(f"Result missing agent_id")
            elif not result.task_id:
                errors.append(f"Result from {result.agent_id} missing task_id")
            elif result.result_type == ResultType.ERROR:
                errors.append(f"Error result from {result.agent_id}: {result.content}")
            elif not result.is_valid():
                errors.append(f"Invalid result from {result.agent_id}")
            else:
                valid_results.append(result)
        
        return valid_results, errors
    
    def get_aggregation_metrics(self) -> Dict[str, Any]:
        """Get aggregation metrics"""
        return {
            "total_aggregations": self.metrics["total_aggregations"],
            "by_type": {
                k.replace("aggregated_", ""): v
                for k, v in self.metrics.items()
                if k.startswith("aggregated_")
            },
            "total_conflicts": len(self.conflict_log),
            "history_size": len(self.aggregation_history)
        }
    
    def get_consensus_report(self, result: AggregatedResult) -> Dict[str, Any]:
        """Generate consensus report for aggregated result"""
        report = {
            "summary": result.get_summary(),
            "consensus_analysis": {
                "level": result.consensus_level,
                "interpretation": self._interpret_consensus(result.consensus_level),
                "agent_agreement": f"{int(result.consensus_level * 100)}%"
            },
            "confidence_analysis": {
                "score": result.confidence_score,
                "interpretation": self._interpret_confidence(result.confidence_score)
            },
            "participation": {
                "total_agents": len(result.contributing_agents),
                "agents": result.contributing_agents
            }
        }
        
        if result.conflicts_resolved > 0:
            report["conflicts"] = {
                "count": result.conflicts_resolved,
                "resolution": "Applied conflict resolution strategy"
            }
        
        return report
    
    def _interpret_consensus(self, level: float) -> str:
        """Interpret consensus level"""
        if level >= 0.9:
            return "Very High - Almost complete agreement"
        elif level >= 0.7:
            return "High - Strong agreement"
        elif level >= 0.5:
            return "Moderate - Majority agreement"
        elif level >= 0.3:
            return "Low - Significant disagreement"
        else:
            return "Very Low - Little agreement"
    
    def _interpret_confidence(self, score: float) -> str:
        """Interpret confidence score"""
        if score >= 0.9:
            return "Very High Confidence"
        elif score >= 0.7:
            return "High Confidence"
        elif score >= 0.5:
            return "Moderate Confidence"
        elif score >= 0.3:
            return "Low Confidence"
        else:
            return "Very Low Confidence"


# Example usage and testing
async def test_result_aggregation():
    """Test the result aggregation system"""
    print("\n" + "="*60)
    print("Testing Collaboration Result Aggregation System")
    print("="*60)
    
    # Create aggregator
    aggregator = CollaborationResultAggregator()
    
    # Create sample results from different agents
    results = [
        # Text results
        AgentResult(
            agent_id="agent_1",
            role=AgentRole.WORKER,
            task_id="task_001",
            result_type=ResultType.TEXT,
            content="The analysis shows positive trends in performance metrics.",
            confidence=0.85
        ),
        AgentResult(
            agent_id="agent_2",
            role=AgentRole.SPECIALIST,
            task_id="task_001",
            result_type=ResultType.TEXT,
            content="The analysis shows positive trends in performance metrics. Additional optimization is recommended.",
            confidence=0.92
        ),
        AgentResult(
            agent_id="agent_3",
            role=AgentRole.REVIEWER,
            task_id="task_001",
            result_type=ResultType.TEXT,
            content="The analysis confirms positive performance trends.",
            confidence=0.88
        )
    ]
    
    # Test text aggregation
    print("\n--- Testing Text Aggregation ---")
    text_result = await aggregator.aggregate_results(results[:3])
    if text_result:
        print(f"Final text: {text_result.final_result[:100]}...")
        print(f"Confidence: {text_result.confidence_score:.2f}")
        print(f"Consensus: {text_result.consensus_level:.2f}")
    
    # Code results
    code_results = [
        AgentResult(
            agent_id="agent_1",
            role=AgentRole.WORKER,
            task_id="task_002",
            result_type=ResultType.CODE,
            content={"code": "def process(data):\n    return data * 2", "language": "python"},
            confidence=0.75
        ),
        AgentResult(
            agent_id="agent_2",
            role=AgentRole.SPECIALIST,
            task_id="task_002",
            result_type=ResultType.CODE,
            content={"code": "def process(data):\n    # Optimized\n    return data * 2", "language": "python"},
            confidence=0.95
        )
    ]
    
    # Test code aggregation
    print("\n--- Testing Code Aggregation ---")
    code_result = await aggregator.aggregate_results(code_results)
    if code_result:
        print(f"Final code selected from: {code_result.contributing_agents}")
        print(f"Confidence: {code_result.confidence_score:.2f}")
        print(f"Code versions: {code_result.metadata.get('code_versions', 0)}")
    
    # Data results
    data_results = [
        AgentResult(
            agent_id="agent_1",
            role=AgentRole.WORKER,
            task_id="task_003",
            result_type=ResultType.METRICS,
            content=[85, 90, 88, 92],
            confidence=0.8
        ),
        AgentResult(
            agent_id="agent_2",
            role=AgentRole.MONITOR,
            task_id="task_003",
            result_type=ResultType.METRICS,
            content=[86, 89, 87, 93],
            confidence=0.85
        )
    ]
    
    # Test data aggregation
    print("\n--- Testing Data Aggregation ---")
    data_result = await aggregator.aggregate_results(data_results)
    if data_result:
        print(f"Aggregated metrics: {data_result.final_result}")
        print(f"Strategy: {data_result.aggregation_strategy.name}")
    
    # Mixed results
    mixed_results = results + code_results + data_results
    
    # Test mixed aggregation
    print("\n--- Testing Mixed Type Aggregation ---")
    mixed_result = await aggregator.aggregate_results(mixed_results)
    if mixed_result:
        print(f"Components: {list(mixed_result.final_result['components'].keys())}")
        print(f"Total agents: {len(mixed_result.contributing_agents)}")
        print(f"Overall confidence: {mixed_result.confidence_score:.2f}")
    
    # Generate consensus report
    print("\n--- Consensus Report ---")
    if text_result:
        report = aggregator.get_consensus_report(text_result)
        print(f"Consensus: {report['consensus_analysis']['interpretation']}")
        print(f"Confidence: {report['confidence_analysis']['interpretation']}")
        print(f"Participation: {report['participation']['total_agents']} agents")
    
    # Get metrics
    metrics = aggregator.get_aggregation_metrics()
    print(f"\n--- Aggregation Metrics ---")
    print(f"Total aggregations: {metrics['total_aggregations']}")
    print(f"By type: {metrics['by_type']}")
    
    return aggregator


if __name__ == "__main__":
    print("Collaboration Result Aggregation System")
    print("="*60)
    
    # Run test
    aggregator = asyncio.run(test_result_aggregation())
    
    print("\n✅ Collaboration Result Aggregation System ready!")