
"""
Orchestration Workflows for Agent Lightning
Implements various agent coordination patterns following the framework design
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from collections import deque
import json
import uuid
from mdp_agents import MDPAgent, AgentState, AgentAction, MDPTransition
from memory_manager import MemoryManager
from observability_setup import AgentLightningObservability, AgentSpan


class WorkflowType(Enum):
    """Types of orchestration workflows"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    DYNAMIC = "dynamic"
    PIPELINE = "pipeline"
    GRAPH = "graph"
    CONSENSUS = "consensus"
    MAPREDUCE = "map_reduce"


class AgentRole(Enum):
    """Agent roles in workflows"""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"
    ROUTER = "router"
    SPECIALIST = "specialist"


@dataclass
class WorkflowTask:
    """Represents a task in the workflow"""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    required_agents: List[str]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    workflow_id: str
    status: str
    results: Dict[str, Any]
    transitions: List[MDPTransition]
    execution_time: float
    metadata: Dict[str, Any]


class OrchestrationWorkflow:
    """
    Base class for agent orchestration workflows
    Implements coordination patterns from Agent Lightning
    """
    
    def __init__(self,
                 workflow_type: WorkflowType,
                 agents: Dict[str, MDPAgent],
                 memory_manager: Optional[MemoryManager] = None,
                 observability: Optional[AgentLightningObservability] = None):
        """
        Initialize orchestration workflow
        
        Args:
            workflow_type: Type of workflow
            agents: Dictionary of agents
            memory_manager: Shared memory manager
            observability: Observability instance
        """
        self.workflow_type = workflow_type
        self.agents = agents
        self.memory_manager = memory_manager or MemoryManager()
        self.observability = observability
        
        # Workflow state
        self.workflow_id = str(uuid.uuid4())
        self.tasks_queue = deque()
        self.completed_tasks = []
        self.failed_tasks = []
        self.transitions = []
        
        # Execution tracking
        self.start_time = None
        self.end_time = None
        
        print(f"🔄 Initialized {workflow_type.value} workflow with {len(agents)} agents")
    
    async def execute(self, task: WorkflowTask) -> WorkflowResult:
        """
        Execute workflow for a task
        
        Args:
            task: Workflow task to execute
            
        Returns:
            WorkflowResult
        """
        self.start_time = time.time()
        
        # Start observability span if available
        if self.observability:
            with self.observability.trace_agent_execution(
                agent_id=f"workflow_{self.workflow_id}",
                task_type=self.workflow_type.value
            ) as span:
                span.set_attribute("task_id", task.task_id)
                result = await self._execute_workflow(task)
        else:
            result = await self._execute_workflow(task)
        
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        return WorkflowResult(
            workflow_id=self.workflow_id,
            status="completed" if not self.failed_tasks else "partial",
            results=result,
            transitions=self.transitions,
            execution_time=execution_time,
            metadata={
                "workflow_type": self.workflow_type.value,
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks)
            }
        )
    
    async def _execute_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Override in subclasses for specific workflow logic"""
        raise NotImplementedError


class SequentialWorkflow(OrchestrationWorkflow):
    """
    Sequential workflow - agents execute one after another
    Output of one agent becomes input to the next
    """
    
    def __init__(self, agents: List[MDPAgent], **kwargs):
        # Convert list to dict with sequential naming
        agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        super().__init__(WorkflowType.SEQUENTIAL, agents_dict, **kwargs)
        self.agent_sequence = agents
    
    async def _execute_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute agents sequentially"""
        current_input = task.input_data
        results = {}
        
        for i, agent in enumerate(self.agent_sequence):
            agent_name = f"agent_{i}"
            
            # Create state from current input
            state = agent.observe({
                "input": current_input,
                "context": {"previous_results": results},
                "semantic_variables": task.metadata
            })
            
            # Agent acts
            action, transition = agent.act(state)
            
            # Store transition
            self.transitions.append(transition)
            
            # Store result
            results[agent_name] = {
                "action": action.to_dict(),
                "confidence": action.confidence
            }
            
            # Update input for next agent
            current_input = action.content
            
            # Store in memory if available
            if self.memory_manager:
                self.memory_manager.store_episodic(
                    content={
                        "agent": agent_name,
                        "input": state.to_dict(),
                        "output": action.to_dict()
                    },
                    importance=action.confidence
                )
        
        return results


class ParallelWorkflow(OrchestrationWorkflow):
    """
    Parallel workflow - agents execute simultaneously
    Results are collected and optionally aggregated
    """
    
    def __init__(self, agents: Dict[str, MDPAgent], max_workers: int = 4, **kwargs):
        super().__init__(WorkflowType.PARALLEL, agents, **kwargs)
        self.max_workers = max_workers
    
    async def _execute_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute agents in parallel"""
        results = {}
        
        # Create tasks for all agents
        async_tasks = []
        for agent_name, agent in self.agents.items():
            async_task = self._execute_agent_async(agent_name, agent, task)
            async_tasks.append(async_task)
        
        # Execute all tasks concurrently
        completed_results = await asyncio.gather(*async_tasks)
        
        # Collect results
        for agent_name, result in completed_results:
            results[agent_name] = result
        
        return results
    
    async def _execute_agent_async(self, agent_name: str, agent: MDPAgent, 
                                   task: WorkflowTask) -> Tuple[str, Dict]:
        """Execute single agent asynchronously"""
        # Create state
        state = agent.observe({
            "input": task.input_data,
            "context": task.metadata,
            "semantic_variables": {}
        })
        
        # Agent acts
        action, transition = agent.act(state)
        
        # Store transition
        self.transitions.append(transition)
        
        return agent_name, {
            "action": action.to_dict(),
            "confidence": action.confidence
        }


class HierarchicalWorkflow(OrchestrationWorkflow):
    """
    Hierarchical workflow - high-level agents coordinate low-level agents
    Implements the hierarchical RL approach from Agent Lightning
    """
    
    def __init__(self, 
                 high_level_agents: Dict[str, MDPAgent],
                 low_level_agents: Dict[str, MDPAgent],
                 **kwargs):
        all_agents = {**high_level_agents, **low_level_agents}
        super().__init__(WorkflowType.HIERARCHICAL, all_agents, **kwargs)
        self.high_level_agents = high_level_agents
        self.low_level_agents = low_level_agents
    
    async def _execute_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute hierarchical coordination"""
        results = {
            "high_level": {},
            "low_level": {}
        }
        
        # Phase 1: High-level planning
        high_level_plan = await self._execute_high_level(task)
        results["high_level"] = high_level_plan
        
        # Phase 2: Low-level execution based on plan
        for subtask in high_level_plan.get("subtasks", []):
            subtask_results = await self._execute_low_level(subtask)
            results["low_level"][subtask["id"]] = subtask_results
        
        # Phase 3: High-level validation/aggregation
        if "validator" in self.high_level_agents:
            validation = await self._validate_results(results)
            results["validation"] = validation
        
        return results
    
    async def _execute_high_level(self, task: WorkflowTask) -> Dict:
        """Execute high-level planning agents"""
        plan = {"subtasks": []}
        
        for agent_name, agent in self.high_level_agents.items():
            state = agent.observe({
                "input": task.input_data,
                "context": {"task_type": task.task_type},
                "semantic_variables": {"hierarchy_level": "high"}
            })
            
            action, transition = agent.act(state)
            self.transitions.append(transition)
            
            # Parse plan from action (simplified)
            if "plan" in action.content.lower():
                # Extract subtasks from plan
                subtasks = self._parse_plan(action.content)
                plan["subtasks"].extend(subtasks)
            
            plan[agent_name] = action.to_dict()
        
        return plan
    
    async def _execute_low_level(self, subtask: Dict) -> Dict:
        """Execute low-level agents for subtask"""
        results = {}
        
        # Select appropriate low-level agent
        agent_name = subtask.get("assigned_agent", list(self.low_level_agents.keys())[0])
        if agent_name in self.low_level_agents:
            agent = self.low_level_agents[agent_name]
            
            state = agent.observe({
                "input": subtask.get("description", ""),
                "context": subtask,
                "semantic_variables": {"hierarchy_level": "low"}
            })
            
            action, transition = agent.act(state)
            self.transitions.append(transition)
            
            results = {
                "agent": agent_name,
                "action": action.to_dict(),
                "completed": True
            }
        
        return results
    
    async def _validate_results(self, results: Dict) -> Dict:
        """Validate results using high-level validator"""
        validator = self.high_level_agents.get("validator")
        if not validator:
            return {"validated": True}
        
        state = validator.observe({
            "input": json.dumps(results),
            "context": {"task": "validation"},
            "semantic_variables": {"hierarchy_level": "high"}
        })
        
        action, transition = validator.act(state)
        self.transitions.append(transition)
        
        return {
            "validated": "approved" in action.content.lower(),
            "feedback": action.content
        }
    
    def _parse_plan(self, plan_text: str) -> List[Dict]:
        """Parse plan text into subtasks (simplified)"""
        subtasks = []
        lines = plan_text.split('\n')
        
        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in ["step", "task", "1.", "2.", "3."]):
                subtasks.append({
                    "id": f"subtask_{i}",
                    "description": line.strip(),
                    "assigned_agent": None  # Will be assigned dynamically
                })
        
        return subtasks


class DynamicWorkflow(OrchestrationWorkflow):
    """
    Dynamic workflow - agents decide next steps based on results
    Implements adaptive orchestration
    """
    
    def __init__(self, agents: Dict[str, MDPAgent], 
                 router_agent: MDPAgent,
                 **kwargs):
        super().__init__(WorkflowType.DYNAMIC, agents, **kwargs)
        self.router_agent = router_agent
        self.execution_graph = nx.DiGraph()
    
    async def _execute_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute dynamic workflow with routing decisions"""
        results = {}
        current_task = task
        visited_agents = set()
        max_iterations = 10  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            # Router decides next agent
            next_agent_name = await self._route_task(current_task, visited_agents)
            
            if not next_agent_name or next_agent_name == "complete":
                break
            
            if next_agent_name in self.agents:
                agent = self.agents[next_agent_name]
                
                # Execute agent
                state = agent.observe({
                    "input": current_task.input_data,
                    "context": {"iteration": iteration, "previous_agents": list(visited_agents)},
                    "semantic_variables": current_task.metadata
                })
                
                action, transition = agent.act(state)
                self.transitions.append(transition)
                
                # Store result
                results[f"{next_agent_name}_{iteration}"] = {
                    "action": action.to_dict(),
                    "confidence": action.confidence
                }
                
                # Update task for next iteration
                current_task.input_data = {"previous_output": action.content}
                visited_agents.add(next_agent_name)
                
                # Add to execution graph
                if iteration > 0:
                    prev_agent = list(visited_agents)[-2] if len(visited_agents) > 1 else "start"
                    self.execution_graph.add_edge(prev_agent, next_agent_name)
        
        results["execution_path"] = list(visited_agents)
        return results
    
    async def _route_task(self, task: WorkflowTask, visited: set) -> str:
        """Router agent decides next agent"""
        state = self.router_agent.observe({
            "input": f"Route task: {task.task_type}",
            "context": {
                "available_agents": [a for a in self.agents.keys() if a not in visited],
                "visited_agents": list(visited),
                "task_data": task.input_data
            },
            "semantic_variables": {"role": "router"}
        })
        
        action, transition = self.router_agent.act(state)
        self.transitions.append(transition)
        
        # Parse routing decision (simplified)
        for agent_name in self.agents.keys():
            if agent_name in action.content.lower():
                return agent_name
        
        if "complete" in action.content.lower() or "done" in action.content.lower():
            return "complete"
        
        # Default: pick first unvisited agent
        unvisited = [a for a in self.agents.keys() if a not in visited]
        return unvisited[0] if unvisited else "complete"


class ConsensusWorkflow(OrchestrationWorkflow):
    """
    Consensus workflow - multiple agents vote on decisions
    Implements democratic coordination
    """
    
    def __init__(self, agents: Dict[str, MDPAgent], 
                 consensus_threshold: float = 0.5,
                 **kwargs):
        super().__init__(WorkflowType.CONSENSUS, agents, **kwargs)
        self.consensus_threshold = consensus_threshold
    
    async def _execute_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute consensus-based decision making"""
        votes = {}
        proposals = {}
        
        # Phase 1: Collect proposals from all agents
        for agent_name, agent in self.agents.items():
            state = agent.observe({
                "input": task.input_data,
                "context": {"role": "voter"},
                "semantic_variables": task.metadata
            })
            
            action, transition = agent.act(state)
            self.transitions.append(transition)
            
            proposals[agent_name] = {
                "proposal": action.content,
                "confidence": action.confidence
            }
        
        # Phase 2: Agents vote on proposals
        for voter_name, voter in self.agents.items():
            agent_votes = {}
            
            for proposer_name, proposal in proposals.items():
                if proposer_name != voter_name:  # Don't vote on own proposal
                    vote_state = voter.observe({
                        "input": f"Vote on proposal: {proposal['proposal'][:200]}",
                        "context": {"proposer": proposer_name},
                        "semantic_variables": {"task": "voting"}
                    })
                    
                    vote_action, vote_transition = voter.act(vote_state)
                    self.transitions.append(vote_transition)
                    
                    # Parse vote (simplified)
                    vote_value = 1.0 if "approve" in vote_action.content.lower() else 0.0
                    agent_votes[proposer_name] = vote_value
            
            votes[voter_name] = agent_votes
        
        # Phase 3: Tally votes and determine consensus
        proposal_scores = {}
        for proposer in proposals.keys():
            scores = [votes[voter].get(proposer, 0) for voter in votes.keys()]
            proposal_scores[proposer] = sum(scores) / len(scores) if scores else 0
        
        # Find winning proposal
        winner = max(proposal_scores.items(), key=lambda x: x[1])
        consensus_reached = winner[1] >= self.consensus_threshold
        
        return {
            "proposals": proposals,
            "votes": votes,
            "scores": proposal_scores,
            "winner": winner[0],
            "winning_proposal": proposals[winner[0]],
            "consensus_reached": consensus_reached,
            "consensus_score": winner[1]
        }


class MapReduceWorkflow(OrchestrationWorkflow):
    """
    MapReduce workflow - distribute work and aggregate results
    Suitable for data processing and analysis tasks
    """
    
    def __init__(self, 
                 mapper_agents: Dict[str, MDPAgent],
                 reducer_agent: MDPAgent,
                 **kwargs):
        all_agents = {**mapper_agents, "reducer": reducer_agent}
        super().__init__(WorkflowType.MAPREDUCE, all_agents, **kwargs)
        self.mapper_agents = mapper_agents
        self.reducer_agent = reducer_agent
    
    async def _execute_workflow(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute map-reduce pattern"""
        # Phase 1: Map - distribute work to mappers
        map_results = await self._map_phase(task)
        
        # Phase 2: Reduce - aggregate results
        reduce_result = await self._reduce_phase(map_results)
        
        return {
            "map_results": map_results,
            "reduce_result": reduce_result
        }
    
    async def _map_phase(self, task: WorkflowTask) -> Dict[str, Any]:
        """Execute mapping phase in parallel"""
        # Split task data for mappers
        data_chunks = self._split_data(task.input_data)
        
        # Execute mappers in parallel
        map_tasks = []
        for i, (mapper_name, mapper) in enumerate(self.mapper_agents.items()):
            chunk = data_chunks[i % len(data_chunks)] if data_chunks else task.input_data
            map_task = self._execute_mapper(mapper_name, mapper, chunk)
            map_tasks.append(map_task)
        
        # Collect results
        map_results = await asyncio.gather(*map_tasks)
        
        return dict(map_results)
    
    async def _execute_mapper(self, mapper_name: str, mapper: MDPAgent, 
                             data_chunk: Any) -> Tuple[str, Dict]:
        """Execute single mapper"""
        state = mapper.observe({
            "input": data_chunk,
            "context": {"role": "mapper"},
            "semantic_variables": {"operation": "map"}
        })
        
        action, transition = mapper.act(state)
        self.transitions.append(transition)
        
        return mapper_name, {
            "processed": action.content,
            "confidence": action.confidence
        }
    
    async def _reduce_phase(self, map_results: Dict[str, Any]) -> Dict:
        """Execute reduction phase"""
        state = self.reducer_agent.observe({
            "input": json.dumps(map_results),
            "context": {"role": "reducer"},
            "semantic_variables": {"operation": "reduce"}
        })
        
        action, transition = self.reducer_agent.act(state)
        self.transitions.append(transition)
        
        return {
            "aggregated": action.content,
            "confidence": action.confidence
        }
    
    def _split_data(self, data: Any) -> List[Any]:
        """Split data for map phase (simplified)"""
        if isinstance(data, list):
            # Split list into chunks
            chunk_size = max(1, len(data) // len(self.mapper_agents))
            return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        elif isinstance(data, dict):
            # Split dict by keys
            items = list(data.items())
            chunk_size = max(1, len(items) // len(self.mapper_agents))
            return [dict(items[i:i+chunk_size]) for i in range(0, len(items), chunk_size)]
        else:
            # Can't split, return as is
            return [data]


# Factory function for creating workflows
def create_workflow(workflow_type: WorkflowType,
                   agents: Dict[str, MDPAgent],
                   **kwargs) -> OrchestrationWorkflow:
    """
    Factory function to create appropriate workflow
    
    Args:
        workflow_type: Type of workflow to create
        agents: Dictionary of agents
        **kwargs: Additional workflow-specific parameters
        
    Returns:
        OrchestrationWorkflow instance
    """
    if workflow_type == WorkflowType.SEQUENTIAL:
        agent_list = list(agents.values())
        return SequentialWorkflow(agent_list, **kwargs)
    
    elif workflow_type == WorkflowType.PARALLEL:
        return ParallelWorkflow(agents, **kwargs)
    
    elif workflow_type == WorkflowType.HIERARCHICAL:
        # Split agents into high and low level (simplified)
        high_level = {k: v for k, v in agents.items() if "high" in k.lower() or "plan" in k.lower()}
        low_level = {k: v for k, v in agents.items() if k not in high_level}
        return HierarchicalWorkflow(high_level, low_level, **kwargs)
    
    elif workflow_type == WorkflowType.DYNAMIC:
        # Need a router agent
        router = kwargs.pop("router_agent", list(agents.values())[0])
        return DynamicWorkflow(agents, router, **kwargs)
    
    elif workflow_type == WorkflowType.CONSENSUS:
        return ConsensusWorkflow(agents, **kwargs)
    
    elif workflow_type == WorkflowType.MAPREDUCE:
        # Need mapper and reducer agents
        reducer = kwargs.pop("reducer_agent", list(agents.values())[-1])
        mappers = {k: v for k, v in agents.items() if v != reducer}
        return MapReduceWorkflow(mappers, reducer, **kwargs)
    
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    print("🔄 Testing Orchestration Workflows")
    print("=" * 60)
    
    # Create test agents
    test_agents = {
        "researcher": MDPAgent(role="Researcher", hierarchy_level="high"),
        "writer": MDPAgent(role="Writer", hierarchy_level="low"),
        "reviewer": MDPAgent(role="Reviewer", hierarchy_level="high")
    }
    
    # Test task
    test_task = WorkflowTask(
        task_id="test_001",
        task_type="content_creation",
        input_data={"topic": "AI orchestration patterns"},
        required_agents=["researcher", "writer", "reviewer"],
        metadata={"priority": "high"}
    )
    
    async def test_workflows():
        # Test Sequential Workflow
        print("\n📝 Testing Sequential Workflow...")
        seq_workflow = create_workflow(WorkflowType.SEQUENTIAL, test_agents)
        seq_result = await seq_workflow.execute(test_task)
        print(f"   Completed in {seq_result.execution_time:.2f}s")
        print(f"   Transitions collected: {len(seq_result.transitions)}")
        
        # Test Parallel Workflow
        print("\n⚡ Testing Parallel Workflow...")
        par_workflow = create_workflow(WorkflowType.PARALLEL, test_agents)
        par_result = await par_workflow.execute(test_task)
        print(f"   Completed in {par_result.execution_time:.2f}s")
        print(f"   Agents executed: {len(par_result.results)}")
        
        # Test Hierarchical Workflow
        print("\n🏗️ Testing Hierarchical Workflow...")
        hier_workflow = create_workflow(WorkflowType.HIERARCHICAL, test_agents)
        hier_result = await hier_workflow.execute(test_task)
        print(f"   Completed in {hier_result.execution_time:.2f}s")
        print(f"   High-level results: {len(hier_result.results.get('high_level', {}))}")
        print(f"   Low-level results: {len(hier_result.results.get('low_level', {}))}")
        
        # Test Consensus Workflow
        print("\n🗳️ Testing Consensus Workflow...")
        consensus_workflow = create_workflow(WorkflowType.CONSENSUS, test_agents)
        consensus_result = await consensus_workflow.execute(test_task)
        print(f"   Completed in {consensus_result.execution_time:.2f}s")
        print(f"   Consensus reached: {consensus_result.results.get('consensus_reached')}")
        print(f"   Winner: {consensus_result.results.get('winner')}")
    
    # Run tests
    asyncio.run(test_workflows())
    
    print("\n✅ Orchestration workflows test complete!")