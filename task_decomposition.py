#!/usr/bin/env python3
"""
Advanced Task Decomposition Algorithm for Multi-Agent Collaboration
Intelligently breaks down complex tasks into manageable subtasks
"""

import os
import sys
import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_collaboration import CollaborativeTask, TaskComplexity


class DecompositionStrategy(Enum):
    """Different strategies for task decomposition"""
    FUNCTIONAL = "functional"      # Break by function/feature
    TEMPORAL = "temporal"          # Break by time sequence
    DATA_PARALLEL = "data_parallel"  # Break by data segments
    HIERARCHICAL = "hierarchical"  # Tree-based breakdown
    DOMAIN = "domain"              # Break by domain expertise
    HYBRID = "hybrid"              # Combination of strategies


class TaskType(Enum):
    """Types of tasks for better decomposition"""
    DEVELOPMENT = "development"
    ANALYSIS = "analysis"
    DESIGN = "design"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"


@dataclass
class TaskPattern:
    """Reusable pattern for common task types"""
    name: str
    task_type: TaskType
    typical_subtasks: List[str]
    required_capabilities: List[str]
    decomposition_strategy: DecompositionStrategy
    estimated_complexity: TaskComplexity


class TaskDecomposer:
    """Advanced task decomposition system"""
    
    def __init__(self):
        # Predefined patterns for common tasks
        self.task_patterns = self._initialize_patterns()
        
        # Keywords for task analysis
        self.task_keywords = self._initialize_keywords()
        
        # Capability mappings
        self.capability_mappings = self._initialize_capabilities()
        
        # Dependency patterns
        self.dependency_patterns = self._initialize_dependencies()
    
    def _initialize_patterns(self) -> Dict[str, TaskPattern]:
        """Initialize common task patterns"""
        patterns = {
            "web_application": TaskPattern(
                name="Web Application Development",
                task_type=TaskType.DEVELOPMENT,
                typical_subtasks=[
                    "Design database schema",
                    "Implement backend API",
                    "Create frontend UI",
                    "Implement authentication",
                    "Setup deployment pipeline",
                    "Write tests",
                    "Create documentation"
                ],
                required_capabilities=["database", "backend", "frontend", "deployment"],
                decomposition_strategy=DecompositionStrategy.FUNCTIONAL,
                estimated_complexity=TaskComplexity.COMPLEX
            ),
            
            "code_analysis": TaskPattern(
                name="Code Analysis",
                task_type=TaskType.ANALYSIS,
                typical_subtasks=[
                    "Parse code structure",
                    "Identify patterns",
                    "Check code quality",
                    "Find vulnerabilities",
                    "Generate metrics",
                    "Create report"
                ],
                required_capabilities=["code_analysis", "security", "reporting"],
                decomposition_strategy=DecompositionStrategy.TEMPORAL,
                estimated_complexity=TaskComplexity.MODERATE
            ),
            
            "api_development": TaskPattern(
                name="API Development",
                task_type=TaskType.DEVELOPMENT,
                typical_subtasks=[
                    "Design API endpoints",
                    "Implement request handlers",
                    "Add data validation",
                    "Implement authentication",
                    "Create API documentation",
                    "Write API tests"
                ],
                required_capabilities=["backend", "api", "documentation"],
                decomposition_strategy=DecompositionStrategy.FUNCTIONAL,
                estimated_complexity=TaskComplexity.MODERATE
            ),
            
            "performance_optimization": TaskPattern(
                name="Performance Optimization",
                task_type=TaskType.OPTIMIZATION,
                typical_subtasks=[
                    "Profile current performance",
                    "Identify bottlenecks",
                    "Optimize database queries",
                    "Implement caching",
                    "Optimize algorithms",
                    "Verify improvements"
                ],
                required_capabilities=["performance", "database", "optimization"],
                decomposition_strategy=DecompositionStrategy.TEMPORAL,
                estimated_complexity=TaskComplexity.COMPLEX
            ),
            
            "ml_model_development": TaskPattern(
                name="Machine Learning Model",
                task_type=TaskType.DEVELOPMENT,
                typical_subtasks=[
                    "Data collection and preparation",
                    "Feature engineering",
                    "Model selection",
                    "Training and validation",
                    "Hyperparameter tuning",
                    "Model deployment",
                    "Performance monitoring"
                ],
                required_capabilities=["data_science", "machine_learning", "python"],
                decomposition_strategy=DecompositionStrategy.TEMPORAL,
                estimated_complexity=TaskComplexity.VERY_COMPLEX
            ),
            
            "security_audit": TaskPattern(
                name="Security Audit",
                task_type=TaskType.ANALYSIS,
                typical_subtasks=[
                    "Vulnerability scanning",
                    "Code security review",
                    "Dependency audit",
                    "Access control review",
                    "Penetration testing",
                    "Security report generation"
                ],
                required_capabilities=["security", "testing", "reporting"],
                decomposition_strategy=DecompositionStrategy.DOMAIN,
                estimated_complexity=TaskComplexity.COMPLEX
            )
        }
        return patterns
    
    def _initialize_keywords(self) -> Dict[TaskType, List[str]]:
        """Initialize keywords for task type detection"""
        return {
            TaskType.DEVELOPMENT: ["build", "create", "implement", "develop", "code", "construct"],
            TaskType.ANALYSIS: ["analyze", "review", "examine", "inspect", "evaluate", "assess"],
            TaskType.DESIGN: ["design", "architect", "plan", "sketch", "prototype", "model"],
            TaskType.TESTING: ["test", "verify", "validate", "check", "ensure", "qa"],
            TaskType.DEPLOYMENT: ["deploy", "release", "launch", "publish", "distribute"],
            TaskType.OPTIMIZATION: ["optimize", "improve", "enhance", "speed up", "refine"],
            TaskType.DOCUMENTATION: ["document", "describe", "explain", "write docs", "readme"],
            TaskType.RESEARCH: ["research", "investigate", "explore", "study", "discover"],
            TaskType.DEBUGGING: ["debug", "fix", "troubleshoot", "resolve", "diagnose"],
            TaskType.REFACTORING: ["refactor", "restructure", "reorganize", "clean up", "modernize"]
        }
    
    def _initialize_capabilities(self) -> Dict[str, List[str]]:
        """Initialize capability mappings"""
        return {
            "frontend": ["html", "css", "javascript", "react", "vue", "angular", "ui", "ux"],
            "backend": ["api", "server", "database", "authentication", "nodejs", "python", "java"],
            "database": ["sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "schema"],
            "mobile": ["ios", "android", "react native", "flutter", "swift", "kotlin"],
            "security": ["vulnerability", "penetration", "encryption", "authentication", "audit"],
            "devops": ["docker", "kubernetes", "ci/cd", "deployment", "monitoring", "aws"],
            "data_science": ["analysis", "statistics", "machine learning", "data", "visualization"],
            "blockchain": ["smart contract", "web3", "ethereum", "solidity", "defi"]
        }
    
    def _initialize_dependencies(self) -> Dict[str, List[str]]:
        """Initialize common dependency patterns"""
        return {
            "database_first": ["database schema", "data model"],
            "api_after_db": ["api", "backend", "server"],
            "frontend_after_api": ["frontend", "ui", "interface"],
            "test_after_impl": ["test", "verify", "validate"],
            "deploy_last": ["deploy", "release", "launch"],
            "auth_early": ["authentication", "authorization", "security"]
        }
    
    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze task to understand its nature and requirements"""
        analysis = {
            "task_type": self._detect_task_type(task_description),
            "detected_capabilities": self._detect_capabilities(task_description),
            "estimated_complexity": self._estimate_complexity(task_description),
            "suggested_strategy": self._suggest_strategy(task_description),
            "matched_patterns": self._match_patterns(task_description)
        }
        return analysis
    
    def _detect_task_type(self, description: str) -> TaskType:
        """Detect the type of task from description"""
        description_lower = description.lower()
        scores = {}
        
        for task_type, keywords in self.task_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            scores[task_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return TaskType.DEVELOPMENT  # Default
    
    def _detect_capabilities(self, description: str) -> List[str]:
        """Detect required capabilities from task description"""
        description_lower = description.lower()
        detected = []
        
        for capability, keywords in self.capability_mappings.items():
            if any(keyword in description_lower for keyword in keywords):
                detected.append(capability)
        
        return detected
    
    def _estimate_complexity(self, description: str) -> TaskComplexity:
        """Estimate task complexity"""
        # Simple heuristic based on description length and keywords
        words = description.split()
        word_count = len(words)
        
        complexity_keywords = {
            "simple": ["simple", "basic", "trivial", "easy"],
            "complex": ["complex", "advanced", "sophisticated", "comprehensive"],
            "multi": ["multiple", "various", "several", "many"]
        }
        
        has_complex = any(word in description.lower() for word in complexity_keywords["complex"])
        has_multi = any(word in description.lower() for word in complexity_keywords["multi"])
        
        if word_count < 10 and not has_complex:
            return TaskComplexity.SIMPLE
        elif word_count < 20 and not has_multi:
            return TaskComplexity.MODERATE
        elif has_complex or word_count > 30:
            return TaskComplexity.VERY_COMPLEX
        elif has_multi or word_count > 20:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.MODERATE
    
    def _suggest_strategy(self, description: str) -> DecompositionStrategy:
        """Suggest decomposition strategy based on task"""
        description_lower = description.lower()
        
        # Check for specific indicators
        if any(word in description_lower for word in ["step", "phase", "first", "then", "finally"]):
            return DecompositionStrategy.TEMPORAL
        elif any(word in description_lower for word in ["feature", "component", "module"]):
            return DecompositionStrategy.FUNCTIONAL
        elif any(word in description_lower for word in ["data", "dataset", "records", "batch"]):
            return DecompositionStrategy.DATA_PARALLEL
        elif any(word in description_lower for word in ["different", "various", "multiple domains"]):
            return DecompositionStrategy.DOMAIN
        else:
            return DecompositionStrategy.HIERARCHICAL
    
    def _match_patterns(self, description: str) -> List[str]:
        """Match task description against known patterns"""
        matched = []
        description_lower = description.lower()
        
        pattern_keywords = {
            "web_application": ["web app", "website", "web application", "full stack"],
            "code_analysis": ["analyze code", "code review", "code analysis"],
            "api_development": ["api", "rest api", "graphql", "endpoint"],
            "performance_optimization": ["optimize", "performance", "speed up"],
            "ml_model_development": ["machine learning", "ml model", "ai model"],
            "security_audit": ["security", "audit", "vulnerability", "penetration"]
        }
        
        for pattern_name, keywords in pattern_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                matched.append(pattern_name)
        
        return matched
    
    def decompose_task(
        self,
        task: CollaborativeTask,
        strategy: Optional[DecompositionStrategy] = None,
        max_depth: int = 3,
        min_subtasks: int = 2,
        max_subtasks: int = 7
    ) -> CollaborativeTask:
        """
        Decompose a task into subtasks using specified strategy
        
        Args:
            task: The task to decompose
            strategy: Decomposition strategy to use (auto-detected if None)
            max_depth: Maximum depth of decomposition tree
            min_subtasks: Minimum number of subtasks
            max_subtasks: Maximum number of subtasks
        
        Returns:
            Task with subtasks added
        """
        # Analyze task if strategy not provided
        if not strategy:
            analysis = self.analyze_task(task.description)
            strategy = analysis["suggested_strategy"]
            
            # Update task with detected capabilities if not set
            if not task.required_capabilities:
                task.required_capabilities = analysis["detected_capabilities"]
        
        # Apply decomposition based on strategy
        if strategy == DecompositionStrategy.FUNCTIONAL:
            return self._decompose_functional(task, max_depth, min_subtasks, max_subtasks)
        elif strategy == DecompositionStrategy.TEMPORAL:
            return self._decompose_temporal(task, max_depth, min_subtasks, max_subtasks)
        elif strategy == DecompositionStrategy.DATA_PARALLEL:
            return self._decompose_data_parallel(task, max_depth, min_subtasks, max_subtasks)
        elif strategy == DecompositionStrategy.DOMAIN:
            return self._decompose_by_domain(task, max_depth, min_subtasks, max_subtasks)
        elif strategy == DecompositionStrategy.HIERARCHICAL:
            return self._decompose_hierarchical(task, max_depth, min_subtasks, max_subtasks)
        else:  # HYBRID
            return self._decompose_hybrid(task, max_depth, min_subtasks, max_subtasks)
    
    def _decompose_functional(
        self,
        task: CollaborativeTask,
        max_depth: int,
        min_subtasks: int,
        max_subtasks: int
    ) -> CollaborativeTask:
        """Decompose by functional components"""
        # Check if task matches a pattern
        matched_patterns = self._match_patterns(task.description)
        
        if matched_patterns and matched_patterns[0] in self.task_patterns:
            # Use pattern-based decomposition
            pattern = self.task_patterns[matched_patterns[0]]
            
            for i, subtask_desc in enumerate(pattern.typical_subtasks[:max_subtasks]):
                subtask = CollaborativeTask(
                    description=f"{subtask_desc} for {task.description}",
                    complexity=TaskComplexity.SIMPLE if i < 3 else TaskComplexity.MODERATE,
                    required_capabilities=self._get_subtask_capabilities(subtask_desc)
                )
                task.add_subtask(subtask)
        else:
            # Generic functional decomposition
            components = self._identify_functional_components(task.description)
            
            for i, component in enumerate(components[:max_subtasks]):
                subtask = CollaborativeTask(
                    description=f"Implement {component}",
                    complexity=TaskComplexity.MODERATE,
                    required_capabilities=self._detect_capabilities(component)
                )
                task.add_subtask(subtask)
        
        # Add dependencies between subtasks
        self._add_functional_dependencies(task)
        
        return task
    
    def _decompose_temporal(
        self,
        task: CollaborativeTask,
        max_depth: int,
        min_subtasks: int,
        max_subtasks: int
    ) -> CollaborativeTask:
        """Decompose by temporal sequence"""
        phases = [
            "Planning and Design",
            "Initial Implementation",
            "Core Development",
            "Testing and Validation",
            "Optimization",
            "Deployment and Monitoring"
        ]
        
        num_phases = min(max_subtasks, len(phases))
        
        for i, phase in enumerate(phases[:num_phases]):
            subtask = CollaborativeTask(
                description=f"{phase} phase of {task.description}",
                complexity=TaskComplexity.MODERATE,
                required_capabilities=task.required_capabilities
            )
            
            # Add dependencies (each phase depends on previous)
            if i > 0 and task.subtasks:
                subtask.dependencies.append(task.subtasks[i-1].task_id)
            
            task.add_subtask(subtask)
        
        return task
    
    def _decompose_data_parallel(
        self,
        task: CollaborativeTask,
        max_depth: int,
        min_subtasks: int,
        max_subtasks: int
    ) -> CollaborativeTask:
        """Decompose for parallel data processing"""
        # Estimate data segments
        num_segments = max(min_subtasks, min(max_subtasks, 4))
        
        for i in range(num_segments):
            subtask = CollaborativeTask(
                description=f"Process data segment {i+1}/{num_segments} for {task.description}",
                complexity=TaskComplexity.SIMPLE,
                required_capabilities=task.required_capabilities
            )
            # No dependencies - can run in parallel
            task.add_subtask(subtask)
        
        # Add aggregation task
        aggregation = CollaborativeTask(
            description=f"Aggregate results for {task.description}",
            complexity=TaskComplexity.SIMPLE,
            required_capabilities=["data_processing"]
        )
        
        # Aggregation depends on all segments
        for subtask in task.subtasks:
            aggregation.dependencies.append(subtask.task_id)
        
        task.add_subtask(aggregation)
        
        return task
    
    def _decompose_by_domain(
        self,
        task: CollaborativeTask,
        max_depth: int,
        min_subtasks: int,
        max_subtasks: int
    ) -> CollaborativeTask:
        """Decompose by domain expertise"""
        # Identify domains from capabilities
        domains = task.required_capabilities[:max_subtasks] if task.required_capabilities else []
        
        if not domains:
            # Fallback to generic domains
            domains = ["technical", "business", "user_experience"][:max_subtasks]
        
        for domain in domains:
            subtask = CollaborativeTask(
                description=f"{domain.title()} aspects of {task.description}",
                complexity=TaskComplexity.MODERATE,
                required_capabilities=[domain]
            )
            task.add_subtask(subtask)
        
        return task
    
    def _decompose_hierarchical(
        self,
        task: CollaborativeTask,
        max_depth: int,
        min_subtasks: int,
        max_subtasks: int,
        current_depth: int = 0
    ) -> CollaborativeTask:
        """Decompose hierarchically (tree structure)"""
        if current_depth >= max_depth:
            return task
        
        # Create subtasks based on complexity
        num_subtasks = min(max_subtasks, max(min_subtasks, task.complexity.value))
        
        for i in range(num_subtasks):
            subtask = CollaborativeTask(
                description=f"Subtask {i+1} of {task.description}",
                complexity=TaskComplexity(max(1, task.complexity.value - 1)),
                required_capabilities=task.required_capabilities
            )
            
            # Recursively decompose if complex enough
            if subtask.complexity.value > 2 and current_depth < max_depth - 1:
                self._decompose_hierarchical(
                    subtask, max_depth, 2, 3, current_depth + 1
                )
            
            task.add_subtask(subtask)
        
        return task
    
    def _decompose_hybrid(
        self,
        task: CollaborativeTask,
        max_depth: int,
        min_subtasks: int,
        max_subtasks: int
    ) -> CollaborativeTask:
        """Hybrid decomposition combining multiple strategies"""
        # First level: functional decomposition
        self._decompose_functional(task, 1, min_subtasks, max_subtasks)
        
        # Second level: temporal for complex subtasks
        for subtask in task.subtasks:
            if subtask.complexity.value >= 3:
                self._decompose_temporal(subtask, 1, 2, 3)
        
        return task
    
    def _identify_functional_components(self, description: str) -> List[str]:
        """Identify functional components from description"""
        components = []
        
        # Common components to look for
        component_keywords = {
            "database": ["database", "data", "storage"],
            "api": ["api", "interface", "endpoint"],
            "frontend": ["ui", "interface", "frontend", "display"],
            "backend": ["backend", "server", "logic"],
            "authentication": ["auth", "login", "security"],
            "testing": ["test", "validation", "quality"]
        }
        
        description_lower = description.lower()
        for component, keywords in component_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                components.append(component)
        
        # If no specific components found, use generic ones
        if not components:
            components = ["core functionality", "data layer", "interface layer"]
        
        return components
    
    def _get_subtask_capabilities(self, subtask_desc: str) -> List[str]:
        """Get capabilities for a specific subtask"""
        return self._detect_capabilities(subtask_desc)
    
    def _add_functional_dependencies(self, task: CollaborativeTask):
        """Add dependencies between functional subtasks"""
        if not task.subtasks:
            return
        
        # Common dependency patterns
        for i, subtask in enumerate(task.subtasks):
            desc_lower = subtask.description.lower()
            
            # Database/schema tasks should come first
            if any(word in desc_lower for word in ["database", "schema", "data model"]):
                # Move to front if not already
                if i > 0:
                    task.subtasks.insert(0, task.subtasks.pop(i))
            
            # API/backend depends on database
            elif any(word in desc_lower for word in ["api", "backend", "server"]):
                # Find database task
                for other in task.subtasks:
                    if "database" in other.description.lower() or "schema" in other.description.lower():
                        if other.task_id not in subtask.dependencies:
                            subtask.dependencies.append(other.task_id)
            
            # Frontend depends on API
            elif any(word in desc_lower for word in ["frontend", "ui", "interface"]):
                # Find API task
                for other in task.subtasks:
                    if "api" in other.description.lower() or "backend" in other.description.lower():
                        if other.task_id not in subtask.dependencies:
                            subtask.dependencies.append(other.task_id)
            
            # Testing depends on implementation
            elif any(word in desc_lower for word in ["test", "validation"]):
                # Depends on all non-test tasks
                for other in task.subtasks:
                    if "test" not in other.description.lower() and other.task_id != subtask.task_id:
                        if other.task_id not in subtask.dependencies:
                            subtask.dependencies.append(other.task_id)
            
            # Deployment comes last
            elif any(word in desc_lower for word in ["deploy", "release"]):
                # Depends on everything except itself
                for other in task.subtasks:
                    if other.task_id != subtask.task_id and "deploy" not in other.description.lower():
                        if other.task_id not in subtask.dependencies:
                            subtask.dependencies.append(other.task_id)
    
    def optimize_dependencies(self, task: CollaborativeTask) -> CollaborativeTask:
        """Optimize task dependencies to minimize critical path"""
        if not task.subtasks:
            return task
        
        # Build dependency graph
        graph = nx.DiGraph()
        
        # Add nodes
        for subtask in task.subtasks:
            graph.add_node(subtask.task_id, task=subtask)
        
        # Add edges (dependencies)
        for subtask in task.subtasks:
            for dep_id in subtask.dependencies:
                if graph.has_node(dep_id):
                    graph.add_edge(dep_id, subtask.task_id)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            # Remove cycles
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                # Remove the last edge in each cycle
                if len(cycle) > 1:
                    graph.remove_edge(cycle[-2], cycle[-1])
                    # Update task dependencies
                    for subtask in task.subtasks:
                        if subtask.task_id == cycle[-1]:
                            if cycle[-2] in subtask.dependencies:
                                subtask.dependencies.remove(cycle[-2])
        
        # Find parallel opportunities
        # Tasks with no dependencies between them can run in parallel
        levels = self._topological_levels(graph)
        
        # Update task metadata with execution level
        for level, nodes in enumerate(levels):
            for node in nodes:
                for subtask in task.subtasks:
                    if subtask.task_id == node:
                        subtask.metadata["execution_level"] = level
                        subtask.metadata["can_parallel"] = True
        
        return task
    
    def _topological_levels(self, graph: nx.DiGraph) -> List[List[str]]:
        """Get topological levels for parallel execution"""
        levels = []
        remaining = set(graph.nodes())
        
        while remaining:
            # Find nodes with no dependencies in remaining set
            level = []
            for node in remaining:
                predecessors = set(graph.predecessors(node))
                if not predecessors.intersection(remaining):
                    level.append(node)
            
            if not level:
                # Cycle detected or error
                level = list(remaining)
                remaining = set()
            else:
                levels.append(level)
                remaining -= set(level)
        
        return levels
    
    def estimate_task_duration(self, task: CollaborativeTask, agent_count: int = 1) -> timedelta:
        """Estimate task duration based on complexity and parallelism"""
        base_duration = {
            TaskComplexity.TRIVIAL: timedelta(minutes=15),
            TaskComplexity.SIMPLE: timedelta(minutes=30),
            TaskComplexity.MODERATE: timedelta(hours=2),
            TaskComplexity.COMPLEX: timedelta(hours=8),
            TaskComplexity.VERY_COMPLEX: timedelta(days=2)
        }
        
        task_duration = base_duration.get(task.complexity, timedelta(hours=1))
        
        if task.subtasks:
            # Calculate based on critical path
            graph = nx.DiGraph()
            durations = {}
            
            for subtask in task.subtasks:
                graph.add_node(subtask.task_id)
                durations[subtask.task_id] = self.estimate_task_duration(subtask, 1)
                
                for dep_id in subtask.dependencies:
                    if any(st.task_id == dep_id for st in task.subtasks):
                        graph.add_edge(dep_id, subtask.task_id)
            
            if graph.nodes():
                # Find critical path
                if nx.is_directed_acyclic_graph(graph):
                    # Topological sort and calculate max path
                    topo_order = list(nx.topological_sort(graph))
                    max_duration = {}
                    
                    for node in topo_order:
                        predecessors = list(graph.predecessors(node))
                        if predecessors:
                            max_pred = max(max_duration[pred] for pred in predecessors)
                            max_duration[node] = max_pred + durations[node]
                        else:
                            max_duration[node] = durations[node]
                    
                    if max_duration:
                        task_duration = max(max_duration.values())
        
        # Adjust for parallel execution
        if agent_count > 1:
            # Amdahl's law approximation
            parallel_fraction = 0.7  # Assume 70% can be parallelized
            speedup = 1 / ((1 - parallel_fraction) + parallel_fraction / agent_count)
            task_duration = task_duration / speedup
        
        return task_duration


def create_sample_task() -> CollaborativeTask:
    """Create a sample task for testing"""
    return CollaborativeTask(
        description="Build a web application with user authentication, database backend, and RESTful API",
        complexity=TaskComplexity.COMPLEX,
        required_capabilities=["frontend", "backend", "database", "security"]
    )


if __name__ == "__main__":
    print("Task Decomposition System")
    print("=" * 60)
    
    # Initialize decomposer
    decomposer = TaskDecomposer()
    
    # Create sample task
    task = create_sample_task()
    print(f"\nOriginal Task: {task.description}")
    print(f"Complexity: {task.complexity.name}")
    
    # Analyze task
    analysis = decomposer.analyze_task(task.description)
    print(f"\nTask Analysis:")
    print(f"  Type: {analysis['task_type'].value}")
    print(f"  Detected Capabilities: {analysis['detected_capabilities']}")
    print(f"  Suggested Strategy: {analysis['suggested_strategy'].value}")
    print(f"  Matched Patterns: {analysis['matched_patterns']}")
    
    # Decompose task
    decomposed = decomposer.decompose_task(task, strategy=DecompositionStrategy.FUNCTIONAL)
    
    print(f"\nDecomposed into {len(decomposed.subtasks)} subtasks:")
    for i, subtask in enumerate(decomposed.subtasks, 1):
        deps = f" (depends on: {len(subtask.dependencies)} tasks)" if subtask.dependencies else ""
        print(f"  {i}. {subtask.description}{deps}")
    
    # Optimize dependencies
    optimized = decomposer.optimize_dependencies(decomposed)
    
    print("\nExecution Plan:")
    levels = defaultdict(list)
    for subtask in optimized.subtasks:
        level = subtask.metadata.get("execution_level", 0)
        levels[level].append(subtask.description[:50])
    
    for level in sorted(levels.keys()):
        print(f"  Level {level} (parallel):")
        for task_desc in levels[level]:
            print(f"    - {task_desc}")
    
    # Estimate duration
    duration = decomposer.estimate_task_duration(optimized, agent_count=3)
    print(f"\nEstimated Duration with 3 agents: {duration}")
    
    print("\n✅ Task Decomposition System ready!")