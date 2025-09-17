#!/usr/bin/env python3
"""
Agent Role Assignment System for Multi-Agent Collaboration
Dynamically assigns roles to agents based on capabilities and task requirements
"""

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_collaboration import (
    CollaborativeTask,
    TaskComplexity,
    AgentRole,
    CollaborativeAgent
)
from agent_communication_protocol import AgentMessage, Performative
from task_decomposition import TaskDecomposer, DecompositionStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoleRequirement(Enum):
    """Requirements for different roles"""
    MANDATORY = "mandatory"      # Must have this capability
    PREFERRED = "preferred"       # Should have this capability
    OPTIONAL = "optional"         # Nice to have


@dataclass
class RoleSpecification:
    """Specification for an agent role"""
    role: AgentRole
    required_capabilities: Dict[str, RoleRequirement]
    min_experience_level: int = 0  # 0-10
    max_concurrent_tasks: int = 3
    coordination_level: int = 1    # 1-5, higher means more coordination needed
    decision_authority: int = 1    # 1-5, higher means more authority
    
    def matches_agent(self, agent_capabilities: List[str], agent_experience: int = 0) -> float:
        """Calculate how well an agent matches this role (0.0 to 1.0)"""
        score = 0.0
        max_score = 0.0
        
        for capability, requirement in self.required_capabilities.items():
            if requirement == RoleRequirement.MANDATORY:
                max_score += 3.0
                if capability in agent_capabilities:
                    score += 3.0
                else:
                    return 0.0  # Mandatory capability missing
                    
            elif requirement == RoleRequirement.PREFERRED:
                max_score += 2.0
                if capability in agent_capabilities:
                    score += 2.0
                    
            elif requirement == RoleRequirement.OPTIONAL:
                max_score += 1.0
                if capability in agent_capabilities:
                    score += 1.0
        
        # Add experience factor
        if agent_experience >= self.min_experience_level:
            score += 1.0
        max_score += 1.0
        
        return score / max_score if max_score > 0 else 0.0


@dataclass
class AgentProfile:
    """Extended profile for an agent including history and performance"""
    agent_id: str
    capabilities: List[str]
    experience_level: int = 0
    current_workload: int = 0
    max_workload: int = 5
    completed_tasks: int = 0
    success_rate: float = 1.0
    average_task_time: timedelta = field(default_factory=lambda: timedelta(hours=1))
    specializations: List[str] = field(default_factory=list)
    availability: bool = True
    preferred_roles: List[AgentRole] = field(default_factory=list)
    role_history: Dict[str, int] = field(default_factory=dict)  # role -> count
    
    def update_from_task(self, task: CollaborativeTask, success: bool, duration: timedelta):
        """Update profile based on completed task"""
        self.completed_tasks += 1
        
        # Update success rate (weighted average)
        self.success_rate = ((self.success_rate * (self.completed_tasks - 1)) + 
                           (1.0 if success else 0.0)) / self.completed_tasks
        
        # Update average task time
        self.average_task_time = (
            (self.average_task_time * (self.completed_tasks - 1) + duration) / 
            self.completed_tasks
        )
        
        # Update experience
        if success:
            self.experience_level = min(10, self.experience_level + 0.1)
    
    def is_available_for_role(self) -> bool:
        """Check if agent is available for new role assignment"""
        return self.availability and self.current_workload < self.max_workload


class RoleAssignmentStrategy(Enum):
    """Different strategies for role assignment"""
    CAPABILITY_BASED = "capability_based"      # Match based on capabilities
    EXPERIENCE_BASED = "experience_based"      # Prioritize experience
    LOAD_BALANCED = "load_balanced"            # Balance workload
    SPECIALIZATION = "specialization"          # Use specialized agents
    COST_OPTIMIZED = "cost_optimized"         # Minimize cost/time
    HUNGARIAN = "hungarian"                    # Optimal assignment algorithm


class RoleAssigner:
    """System for assigning roles to agents"""
    
    def __init__(self):
        self.role_specifications = self._initialize_role_specs()
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.current_assignments: Dict[str, Tuple[AgentRole, str]] = {}  # agent_id -> (role, task_id)
        self.assignment_history: List[Dict] = []
        self.performance_metrics: Dict[str, Dict] = defaultdict(lambda: {
            'assignments': 0,
            'successes': 0,
            'average_match_score': 0.0
        })
    
    def _initialize_role_specs(self) -> Dict[AgentRole, RoleSpecification]:
        """Initialize role specifications mapped to actual Agent Lightning capabilities"""
        return {
            AgentRole.COORDINATOR: RoleSpecification(
                role=AgentRole.COORDINATOR,
                required_capabilities={
                    "can_design_architecture": RoleRequirement.MANDATORY,
                    "can_write_documentation": RoleRequirement.MANDATORY,
                    "can_analyze_data": RoleRequirement.PREFERRED,
                    "can_generate_reports": RoleRequirement.PREFERRED,
                    "can_review_code": RoleRequirement.OPTIONAL
                },
                min_experience_level=5,
                max_concurrent_tasks=1,
                coordination_level=5,
                decision_authority=5
            ),
            
            AgentRole.WORKER: RoleSpecification(
                role=AgentRole.WORKER,
                required_capabilities={
                    "can_write_code": RoleRequirement.MANDATORY,
                    "can_debug": RoleRequirement.PREFERRED,
                    "can_optimize": RoleRequirement.OPTIONAL
                },
                min_experience_level=0,
                max_concurrent_tasks=5,
                coordination_level=1,
                decision_authority=1
            ),
            
            AgentRole.SPECIALIST: RoleSpecification(
                role=AgentRole.SPECIALIST,
                required_capabilities={
                    "can_analyze_data": RoleRequirement.MANDATORY,
                    "can_design_architecture": RoleRequirement.PREFERRED,
                    "can_optimize": RoleRequirement.PREFERRED
                },
                min_experience_level=3,
                max_concurrent_tasks=2,
                coordination_level=2,
                decision_authority=3
            ),
            
            AgentRole.REVIEWER: RoleSpecification(
                role=AgentRole.REVIEWER,
                required_capabilities={
                    "can_review_code": RoleRequirement.MANDATORY,
                    "can_test": RoleRequirement.MANDATORY,
                    "can_write_documentation": RoleRequirement.PREFERRED
                },
                min_experience_level=2,
                max_concurrent_tasks=3,
                coordination_level=2,
                decision_authority=3
            ),
            
            AgentRole.AGGREGATOR: RoleSpecification(
                role=AgentRole.AGGREGATOR,
                required_capabilities={
                    "can_analyze_data": RoleRequirement.MANDATORY,
                    "can_generate_reports": RoleRequirement.MANDATORY,
                    "can_write_documentation": RoleRequirement.PREFERRED
                },
                min_experience_level=2,
                max_concurrent_tasks=2,
                coordination_level=3,
                decision_authority=2
            ),
            
            AgentRole.MONITOR: RoleSpecification(
                role=AgentRole.MONITOR,
                required_capabilities={
                    "can_analyze_data": RoleRequirement.MANDATORY,
                    "can_generate_reports": RoleRequirement.MANDATORY,
                    "can_optimize": RoleRequirement.OPTIONAL
                },
                min_experience_level=1,
                max_concurrent_tasks=10,
                coordination_level=1,
                decision_authority=2
            )
        }
    
    def register_agent(self, agent: CollaborativeAgent, profile: Optional[AgentProfile] = None):
        """Register an agent with the role assigner"""
        if profile:
            self.agent_profiles[agent.agent_id] = profile
        else:
            # Create default profile from agent
            self.agent_profiles[agent.agent_id] = AgentProfile(
                agent_id=agent.agent_id,
                capabilities=agent.capabilities,
                experience_level=0,
                specializations=[]
            )
        
        logger.info(f"Registered agent {agent.agent_id} with capabilities: {agent.capabilities}")
    
    def analyze_task_requirements(self, task: CollaborativeTask) -> Dict[AgentRole, int]:
        """Analyze task to determine required roles and counts"""
        role_requirements = {}
        
        # Determine task characteristics
        is_complex = task.complexity.value >= 3
        has_subtasks = len(task.subtasks) > 0
        needs_coordination = has_subtasks or task.complexity == TaskComplexity.VERY_COMPLEX
        needs_review = task.complexity.value >= 2
        needs_monitoring = task.deadline is not None
        
        # Assign roles based on task analysis
        if needs_coordination:
            role_requirements[AgentRole.COORDINATOR] = 1
        
        # Workers based on complexity and subtasks
        worker_count = max(1, min(task.complexity.value, len(task.subtasks) if has_subtasks else 1))
        role_requirements[AgentRole.WORKER] = worker_count
        
        # Specialists for complex tasks
        if is_complex:
            # Check for specific domain requirements
            specialist_count = 0
            if any(cap in ["security", "performance", "architecture"] for cap in task.required_capabilities):
                specialist_count += 1
            if any(cap in ["machine_learning", "data_science", "ai"] for cap in task.required_capabilities):
                specialist_count += 1
            if specialist_count > 0:
                role_requirements[AgentRole.SPECIALIST] = specialist_count
        
        # Reviewer for quality assurance
        if needs_review:
            role_requirements[AgentRole.REVIEWER] = 1
        
        # Aggregator for multi-part results
        if has_subtasks and len(task.subtasks) > 2:
            role_requirements[AgentRole.AGGREGATOR] = 1
        
        # Monitor for deadline-driven tasks
        if needs_monitoring:
            role_requirements[AgentRole.MONITOR] = 1
        
        return role_requirements
    
    def calculate_assignment_matrix(
        self,
        agents: List[str],
        roles: List[AgentRole],
        task: CollaborativeTask
    ) -> np.ndarray:
        """Calculate cost matrix for role assignment (lower is better)"""
        n_agents = len(agents)
        n_roles = len(roles)
        
        # Create cost matrix (we'll minimize, so invert scores)
        cost_matrix = np.full((n_agents, n_roles), 1000.0)  # High cost for impossible assignments
        
        for i, agent_id in enumerate(agents):
            if agent_id not in self.agent_profiles:
                continue
                
            profile = self.agent_profiles[agent_id]
            
            for j, role in enumerate(roles):
                if not profile.is_available_for_role():
                    continue
                
                spec = self.role_specifications[role]
                
                # Calculate match score
                match_score = spec.matches_agent(profile.capabilities, profile.experience_level)
                
                if match_score > 0:
                    # Convert to cost (invert and adjust)
                    base_cost = 1.0 - match_score
                    
                    # Adjust for workload
                    workload_penalty = profile.current_workload * 0.1
                    
                    # Adjust for role history (prefer variety)
                    history_bonus = 0
                    if role.value in profile.role_history:
                        history_bonus = min(0.2, profile.role_history[role.value] * 0.02)
                    
                    # Adjust for success rate
                    performance_bonus = (1.0 - profile.success_rate) * 0.3
                    
                    # Final cost
                    cost_matrix[i, j] = base_cost + workload_penalty + history_bonus + performance_bonus
        
        return cost_matrix
    
    def assign_roles(
        self,
        task: CollaborativeTask,
        available_agents: List[CollaborativeAgent],
        strategy: RoleAssignmentStrategy = RoleAssignmentStrategy.HUNGARIAN
    ) -> Dict[str, AgentRole]:
        """Assign roles to agents for a task"""
        
        # Analyze task requirements
        required_roles = self.analyze_task_requirements(task)
        
        logger.info(f"Task requires roles: {required_roles}")
        
        # Flatten role list (handle multiple instances of same role)
        role_list = []
        for role, count in required_roles.items():
            role_list.extend([role] * count)
        
        # Get available agent IDs
        agent_ids = [agent.agent_id for agent in available_agents 
                    if self.agent_profiles.get(agent.agent_id, AgentProfile(agent.agent_id, [])).is_available_for_role()]
        
        if not agent_ids:
            logger.error("No available agents for role assignment")
            return {}
        
        # Apply strategy
        if strategy == RoleAssignmentStrategy.HUNGARIAN:
            assignments = self._hungarian_assignment(agent_ids, role_list, task)
        elif strategy == RoleAssignmentStrategy.CAPABILITY_BASED:
            assignments = self._capability_based_assignment(agent_ids, role_list, task)
        elif strategy == RoleAssignmentStrategy.EXPERIENCE_BASED:
            assignments = self._experience_based_assignment(agent_ids, role_list, task)
        elif strategy == RoleAssignmentStrategy.LOAD_BALANCED:
            assignments = self._load_balanced_assignment(agent_ids, role_list, task)
        elif strategy == RoleAssignmentStrategy.SPECIALIZATION:
            assignments = self._specialization_assignment(agent_ids, role_list, task)
        else:
            assignments = self._cost_optimized_assignment(agent_ids, role_list, task)
        
        # Update current assignments and profiles
        for agent_id, role in assignments.items():
            self.current_assignments[agent_id] = (role, task.task_id)
            if agent_id in self.agent_profiles:
                self.agent_profiles[agent_id].current_workload += 1
                # Update role history
                if role.value not in self.agent_profiles[agent_id].role_history:
                    self.agent_profiles[agent_id].role_history[role.value] = 0
                self.agent_profiles[agent_id].role_history[role.value] += 1
        
        # Record assignment
        self.assignment_history.append({
            'timestamp': datetime.now(),
            'task_id': task.task_id,
            'assignments': assignments,
            'strategy': strategy.value
        })
        
        # Update metrics
        self._update_metrics(assignments, agent_ids, role_list)
        
        logger.info(f"Role assignments: {assignments}")
        
        return assignments
    
    def _hungarian_assignment(
        self,
        agent_ids: List[str],
        roles: List[AgentRole],
        task: CollaborativeTask
    ) -> Dict[str, AgentRole]:
        """Use Hungarian algorithm for optimal assignment"""
        
        # Handle case where we have more roles than agents or vice versa
        n_agents = len(agent_ids)
        n_roles = len(roles)
        
        if n_agents == 0 or n_roles == 0:
            return {}
        
        # Calculate cost matrix
        cost_matrix = self.calculate_assignment_matrix(agent_ids, roles, task)
        
        # If we have more roles than agents, some roles won't be filled
        # If we have more agents than roles, some agents won't get roles
        if n_agents < n_roles:
            # Pad with dummy agents (high cost)
            padding = np.full((n_roles - n_agents, n_roles), 1000.0)
            cost_matrix = np.vstack([cost_matrix, padding])
            agent_ids = agent_ids + [f"dummy_{i}" for i in range(n_roles - n_agents)]
        elif n_agents > n_roles:
            # Pad with dummy roles (high cost)
            padding = np.full((n_agents, n_agents - n_roles), 1000.0)
            cost_matrix = np.hstack([cost_matrix, padding])
            roles = roles + [None] * (n_agents - n_roles)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create assignment dictionary
        assignments = {}
        for i, j in zip(row_ind, col_ind):
            agent_id = agent_ids[i]
            role = roles[j] if j < len(roles) and roles[j] is not None else None
            
            # Skip dummy assignments
            if not agent_id.startswith("dummy_") and role is not None:
                # Check if assignment is valid (cost < threshold)
                if cost_matrix[i, j] < 100:
                    assignments[agent_id] = role
        
        return assignments
    
    def _capability_based_assignment(
        self,
        agent_ids: List[str],
        roles: List[AgentRole],
        task: CollaborativeTask
    ) -> Dict[str, AgentRole]:
        """Assign based primarily on capability match"""
        assignments = {}
        assigned_agents = set()
        
        # Sort roles by criticality (coordinator first, monitor last)
        role_priority = [
            AgentRole.COORDINATOR,
            AgentRole.SPECIALIST,
            AgentRole.REVIEWER,
            AgentRole.WORKER,
            AgentRole.AGGREGATOR,
            AgentRole.MONITOR
        ]
        
        sorted_roles = sorted(roles, key=lambda r: role_priority.index(r) if r in role_priority else 999)
        
        for role in sorted_roles:
            best_agent = None
            best_score = 0.0
            
            for agent_id in agent_ids:
                if agent_id in assigned_agents:
                    continue
                
                profile = self.agent_profiles.get(agent_id)
                if not profile:
                    continue
                
                spec = self.role_specifications[role]
                score = spec.matches_agent(profile.capabilities, profile.experience_level)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
            
            if best_agent and best_score > 0.3:  # Minimum threshold
                assignments[best_agent] = role
                assigned_agents.add(best_agent)
        
        return assignments
    
    def _experience_based_assignment(
        self,
        agent_ids: List[str],
        roles: List[AgentRole],
        task: CollaborativeTask
    ) -> Dict[str, AgentRole]:
        """Assign most experienced agents to critical roles"""
        assignments = {}
        
        # Sort agents by experience
        sorted_agents = sorted(
            agent_ids,
            key=lambda a: self.agent_profiles.get(a, AgentProfile(a, [])).experience_level,
            reverse=True
        )
        
        # Sort roles by required experience
        sorted_roles = sorted(
            roles,
            key=lambda r: self.role_specifications[r].min_experience_level,
            reverse=True
        )
        
        # Assign in order
        for i, role in enumerate(sorted_roles):
            if i < len(sorted_agents):
                agent_id = sorted_agents[i]
                profile = self.agent_profiles.get(agent_id)
                spec = self.role_specifications[role]
                
                # Check if agent can handle role
                if profile and spec.matches_agent(profile.capabilities, profile.experience_level) > 0:
                    assignments[agent_id] = role
        
        return assignments
    
    def _load_balanced_assignment(
        self,
        agent_ids: List[str],
        roles: List[AgentRole],
        task: CollaborativeTask
    ) -> Dict[str, AgentRole]:
        """Assign to balance workload across agents"""
        assignments = {}
        
        # Sort agents by current workload (ascending)
        sorted_agents = sorted(
            agent_ids,
            key=lambda a: self.agent_profiles.get(a, AgentProfile(a, [])).current_workload
        )
        
        # Assign roles round-robin style to least loaded agents
        role_index = 0
        for agent_id in sorted_agents:
            if role_index >= len(roles):
                break
            
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                continue
            
            # Find suitable role for this agent
            for i in range(len(roles)):
                role = roles[(role_index + i) % len(roles)]
                spec = self.role_specifications[role]
                
                if spec.matches_agent(profile.capabilities, profile.experience_level) > 0:
                    assignments[agent_id] = role
                    role_index = (role_index + i + 1) % len(roles)
                    break
        
        return assignments
    
    def _specialization_assignment(
        self,
        agent_ids: List[str],
        roles: List[AgentRole],
        task: CollaborativeTask
    ) -> Dict[str, AgentRole]:
        """Assign based on agent specializations"""
        assignments = {}
        assigned_agents = set()
        
        for role in roles:
            best_agent = None
            best_specialization_match = 0
            
            for agent_id in agent_ids:
                if agent_id in assigned_agents:
                    continue
                
                profile = self.agent_profiles.get(agent_id)
                if not profile:
                    continue
                
                # Check specialization match
                spec = self.role_specifications[role]
                base_match = spec.matches_agent(profile.capabilities, profile.experience_level)
                
                if base_match > 0:
                    # Bonus for specialization
                    specialization_bonus = 0
                    for specialization in profile.specializations:
                        if specialization in task.required_capabilities:
                            specialization_bonus += 0.2
                    
                    total_match = base_match + specialization_bonus
                    
                    if total_match > best_specialization_match:
                        best_specialization_match = total_match
                        best_agent = agent_id
            
            if best_agent:
                assignments[best_agent] = role
                assigned_agents.add(best_agent)
        
        return assignments
    
    def _cost_optimized_assignment(
        self,
        agent_ids: List[str],
        roles: List[AgentRole],
        task: CollaborativeTask
    ) -> Dict[str, AgentRole]:
        """Assign to minimize expected time/cost"""
        assignments = {}
        assigned_agents = set()
        
        for role in roles:
            best_agent = None
            best_cost = float('inf')
            
            for agent_id in agent_ids:
                if agent_id in assigned_agents:
                    continue
                
                profile = self.agent_profiles.get(agent_id)
                if not profile:
                    continue
                
                spec = self.role_specifications[role]
                match_score = spec.matches_agent(profile.capabilities, profile.experience_level)
                
                if match_score > 0:
                    # Estimate cost (time)
                    base_time = profile.average_task_time.total_seconds()
                    
                    # Adjust for match quality (better match = faster)
                    time_factor = 2.0 - match_score  # 1.0 to 2.0
                    
                    # Adjust for workload (loaded agents are slower)
                    workload_factor = 1.0 + (profile.current_workload * 0.1)
                    
                    estimated_cost = base_time * time_factor * workload_factor
                    
                    if estimated_cost < best_cost:
                        best_cost = estimated_cost
                        best_agent = agent_id
            
            if best_agent:
                assignments[best_agent] = role
                assigned_agents.add(best_agent)
        
        return assignments
    
    def release_assignment(self, agent_id: str):
        """Release an agent from current assignment"""
        if agent_id in self.current_assignments:
            del self.current_assignments[agent_id]
            
            if agent_id in self.agent_profiles:
                self.agent_profiles[agent_id].current_workload = max(
                    0, self.agent_profiles[agent_id].current_workload - 1
                )
    
    def get_agent_for_role(self, role: AgentRole, exclude: List[str] = None) -> Optional[str]:
        """Get best available agent for a specific role"""
        exclude = exclude or []
        best_agent = None
        best_score = 0.0
        
        spec = self.role_specifications[role]
        
        for agent_id, profile in self.agent_profiles.items():
            if agent_id in exclude or not profile.is_available_for_role():
                continue
            
            score = spec.matches_agent(profile.capabilities, profile.experience_level)
            
            # Adjust for current performance
            score *= profile.success_rate
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def suggest_role_for_agent(self, agent: CollaborativeAgent) -> Optional[AgentRole]:
        """Suggest best role for an agent based on capabilities"""
        best_role = None
        best_score = 0.0
        
        profile = self.agent_profiles.get(
            agent.agent_id,
            AgentProfile(agent.agent_id, agent.capabilities)
        )
        
        for role, spec in self.role_specifications.items():
            score = spec.matches_agent(profile.capabilities, profile.experience_level)
            
            if score > best_score:
                best_score = score
                best_role = role
        
        return best_role
    
    def _update_metrics(self, assignments: Dict[str, AgentRole], agents: List[str], roles: List[AgentRole]):
        """Update performance metrics"""
        for agent_id, role in assignments.items():
            if agent_id in self.agent_profiles:
                profile = self.agent_profiles[agent_id]
                spec = self.role_specifications[role]
                
                match_score = spec.matches_agent(profile.capabilities, profile.experience_level)
                
                metrics = self.performance_metrics[agent_id]
                metrics['assignments'] += 1
                
                # Update average match score
                prev_avg = metrics['average_match_score']
                metrics['average_match_score'] = (
                    (prev_avg * (metrics['assignments'] - 1) + match_score) / 
                    metrics['assignments']
                )
    
    def get_assignment_report(self) -> Dict[str, Any]:
        """Get report on current assignments and performance"""
        return {
            'current_assignments': {
                agent_id: {'role': role.value, 'task': task_id}
                for agent_id, (role, task_id) in self.current_assignments.items()
            },
            'agent_utilization': {
                agent_id: {
                    'current_workload': profile.current_workload,
                    'max_workload': profile.max_workload,
                    'utilization': profile.current_workload / profile.max_workload
                }
                for agent_id, profile in self.agent_profiles.items()
            },
            'performance_metrics': dict(self.performance_metrics),
            'recent_assignments': self.assignment_history[-10:] if self.assignment_history else []
        }


# Integration with existing Agent Lightning system
async def test_role_assignment():
    """Test the role assignment system with real agents"""
    print("\n" + "="*60)
    print("Testing Role Assignment System")
    print("="*60)
    
    # Import necessary components
    from agent_collaboration import SpecializedCollaborativeAgent
    from agent_config import AgentConfigManager
    from enhanced_production_api import EnhancedAgentService
    
    # Create role assigner
    assigner = RoleAssigner()
    
    # Initialize real agent service and config manager
    config_manager = AgentConfigManager()
    agent_service = EnhancedAgentService()
    
    # Get available agents from the system
    agent_names = config_manager.list_agents()
    agents = []
    
    # Create collaborative agent wrappers for existing agents
    for agent_name in agent_names[:6]:  # Use first 6 agents for testing
        config = config_manager.get_agent(agent_name)
        if config:
            agent = SpecializedCollaborativeAgent(
                agent_id=agent_name,
                config=config,
                agent_service=agent_service
            )
            agents.append(agent)
            
            # Create and register agent profile based on actual capabilities
            capabilities = [k for k, v in config.capabilities.__dict__.items() if v]
            
            # Set experience level based on agent type
            experience_map = {
                'security_expert': 7,
                'system_architect': 8,
                'database_specialist': 6,
                'full_stack_developer': 5,
                'devops_engineer': 6,
                'blockchain_developer': 4,
                'data_scientist': 5,
                'ui_ux_designer': 4,
                'mobile_developer': 4,
                'information_analyst': 3
            }
            
            profile = AgentProfile(
                agent_id=agent_name,
                capabilities=capabilities,
                experience_level=experience_map.get(agent_name, 3),
                success_rate=0.85 + np.random.random() * 0.15,
                specializations=capabilities[:3] if len(capabilities) > 2 else capabilities
            )
            
            assigner.register_agent(agent, profile)
    
    print(f"\nRegistered {len(agents)} agents")
    
    # Create test tasks with different complexities
    tasks = [
        CollaborativeTask(
            description="Simple data processing task",
            complexity=TaskComplexity.SIMPLE,
            required_capabilities=["data_processing", "python"]
        ),
        CollaborativeTask(
            description="Complex web application development",
            complexity=TaskComplexity.VERY_COMPLEX,
            required_capabilities=["frontend", "backend", "database", "security"],
            deadline=datetime.now() + timedelta(hours=48)
        ),
        CollaborativeTask(
            description="Security audit and optimization",
            complexity=TaskComplexity.COMPLEX,
            required_capabilities=["security", "testing", "analysis"]
        )
    ]
    
    # Add subtasks to complex task
    tasks[1].add_subtask(CollaborativeTask(description="Design database schema", complexity=TaskComplexity.MODERATE))
    tasks[1].add_subtask(CollaborativeTask(description="Implement REST API", complexity=TaskComplexity.MODERATE))
    tasks[1].add_subtask(CollaborativeTask(description="Create React frontend", complexity=TaskComplexity.MODERATE))
    
    # Test different assignment strategies
    strategies = [
        RoleAssignmentStrategy.HUNGARIAN,
        RoleAssignmentStrategy.CAPABILITY_BASED,
        RoleAssignmentStrategy.LOAD_BALANCED
    ]
    
    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Task: {task.description}")
        print(f"Complexity: {task.complexity.name}")
        
        # Analyze requirements
        requirements = assigner.analyze_task_requirements(task)
        print(f"Required roles: {requirements}")
        
        for strategy in strategies:
            print(f"\nStrategy: {strategy.value}")
            
            # Assign roles
            assignments = assigner.assign_roles(task, agents, strategy)
            
            if assignments:
                print("Assignments:")
                for agent_id, role in assignments.items():
                    profile = assigner.agent_profiles[agent_id]
                    print(f"  {agent_id} -> {role.value} (exp: {profile.experience_level}, success: {profile.success_rate:.2f})")
            else:
                print("  No assignments made")
            
            # Release assignments for next test
            for agent_id in assignments.keys():
                assigner.release_assignment(agent_id)
    
    # Get final report
    report = assigner.get_assignment_report()
    print(f"\n{'='*40}")
    print("Assignment Report:")
    print(f"Total assignments in history: {len(assigner.assignment_history)}")
    print(f"Performance metrics tracked for {len(report['performance_metrics'])} agents")
    
    return assigner


if __name__ == "__main__":
    print("Agent Role Assignment System")
    print("="*60)
    
    # Run test
    assigner = asyncio.run(test_role_assignment())
    
    print("\n✅ Agent Role Assignment System ready!")