#!/usr/bin/env python3
"""
LangGraph Integration Service
Provides stateful workflow management with graph-based agent orchestration
"""

import os
import sys
import json
import asyncio
import uuid
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared components
from shared.data_access import DataAccessLayer
from shared.cache import get_cache

# LangGraph imports with fallback
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
    from langgraph.checkpoint import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Mock classes for when LangGraph isn't installed
    class StateGraph:
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = {}
            self.entry_point = None
        
        def add_node(self, name, func):
            self.nodes[name] = func
        
        def add_edge(self, from_node, to_node):
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)
        
        def add_conditional_edges(self, from_node, condition_func, edge_map):
            self.edges[from_node] = {"conditional": condition_func, "map": edge_map}
        
        def set_entry_point(self, node):
            self.entry_point = node
        
        def compile(self, checkpointer=None):
            return self
    
    END = "END"
    
    class MemorySaver:
        def __init__(self):
            self.memory = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# State schema for workflows
class WorkflowState(TypedDict):
    """State schema for LangGraph workflows"""
    messages: Annotated[Sequence[dict], "Conversation messages"]
    current_step: str
    context: Dict[str, Any]
    results: List[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]


class NodeType(str, Enum):
    """Types of workflow nodes"""
    START = "start"
    AGENT = "agent"
    TOOL = "tool"
    DECISION = "decision"
    PARALLEL = "parallel"
    END = "end"


class WorkflowNode(BaseModel):
    """Workflow node definition"""
    id: str = Field(description="Node ID")
    type: NodeType = Field(description="Node type")
    name: str = Field(description="Node name")
    description: str = Field(description="Node description")
    agent_id: Optional[str] = Field(default=None, description="Agent ID for agent nodes")
    tool_name: Optional[str] = Field(default=None, description="Tool name for tool nodes")
    decision_criteria: Optional[Dict[str, Any]] = Field(default=None, description="Decision criteria")
    next_nodes: List[str] = Field(default_factory=list, description="Next node IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")


class WorkflowDefinition(BaseModel):
    """Workflow definition"""
    id: str = Field(description="Workflow ID")
    name: str = Field(description="Workflow name")
    description: str = Field(description="Workflow description")
    nodes: List[WorkflowNode] = Field(description="Workflow nodes")
    edges: List[Dict[str, str]] = Field(description="Workflow edges")
    entry_node: str = Field(description="Entry node ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")


class WorkflowExecution(BaseModel):
    """Workflow execution request"""
    workflow_id: str = Field(description="Workflow ID to execute")
    input_data: Dict[str, Any] = Field(description="Input data for workflow")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    checkpoint_id: Optional[str] = Field(default=None, description="Checkpoint to resume from")


class LangGraphService:
    """LangGraph Integration Service"""
    
    def __init__(self):
        self.app = FastAPI(title="LangGraph Integration Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("langgraph")
        self.cache = get_cache()
        
        # Workflow storage
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.graph_instances: Dict[str, StateGraph] = {}
        self.checkpointers: Dict[str, MemorySaver] = {}
        self.active_executions: Dict[str, Any] = {}
        
        # Initialize sample workflows
        self._initialize_sample_workflows()
        
        logger.info(f"✅ LangGraph Integration initialized (Library Available: {LANGGRAPH_AVAILABLE})")
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _initialize_sample_workflows(self):
        """Initialize sample workflows"""
        # Research workflow
        research_workflow = WorkflowDefinition(
            id="research-workflow",
            name="Research Assistant Workflow",
            description="Multi-step research with web search and summarization",
            nodes=[
                WorkflowNode(
                    id="start",
                    type=NodeType.START,
                    name="Start Research",
                    description="Initialize research task",
                    next_nodes=["search"]
                ),
                WorkflowNode(
                    id="search",
                    type=NodeType.AGENT,
                    name="Search Agent",
                    description="Search for information",
                    agent_id="search-agent",
                    next_nodes=["analyze"]
                ),
                WorkflowNode(
                    id="analyze",
                    type=NodeType.AGENT,
                    name="Analysis Agent",
                    description="Analyze search results",
                    agent_id="analysis-agent",
                    next_nodes=["decision"]
                ),
                WorkflowNode(
                    id="decision",
                    type=NodeType.DECISION,
                    name="Quality Check",
                    description="Check if results are sufficient",
                    decision_criteria={"min_sources": 3},
                    next_nodes=["search", "summarize"]
                ),
                WorkflowNode(
                    id="summarize",
                    type=NodeType.AGENT,
                    name="Summary Agent",
                    description="Create final summary",
                    agent_id="summary-agent",
                    next_nodes=["end"]
                ),
                WorkflowNode(
                    id="end",
                    type=NodeType.END,
                    name="Complete",
                    description="Research complete",
                    next_nodes=[]
                )
            ],
            edges=[
                {"from": "start", "to": "search"},
                {"from": "search", "to": "analyze"},
                {"from": "analyze", "to": "decision"},
                {"from": "decision", "to": "search"},  # Loop back if needed
                {"from": "decision", "to": "summarize"},
                {"from": "summarize", "to": "end"}
            ],
            entry_node="start"
        )
        self.workflows[research_workflow.id] = research_workflow
        
        # Code review workflow
        code_review_workflow = WorkflowDefinition(
            id="code-review-workflow",
            name="Code Review Workflow",
            description="Automated code review with multiple checks",
            nodes=[
                WorkflowNode(
                    id="start",
                    type=NodeType.START,
                    name="Start Review",
                    description="Initialize code review",
                    next_nodes=["parallel-checks"]
                ),
                WorkflowNode(
                    id="parallel-checks",
                    type=NodeType.PARALLEL,
                    name="Parallel Checks",
                    description="Run multiple checks in parallel",
                    next_nodes=["syntax", "security", "performance", "style"]
                ),
                WorkflowNode(
                    id="syntax",
                    type=NodeType.TOOL,
                    name="Syntax Check",
                    description="Check code syntax",
                    tool_name="syntax-checker",
                    next_nodes=["aggregate"]
                ),
                WorkflowNode(
                    id="security",
                    type=NodeType.TOOL,
                    name="Security Scan",
                    description="Scan for security issues",
                    tool_name="security-scanner",
                    next_nodes=["aggregate"]
                ),
                WorkflowNode(
                    id="performance",
                    type=NodeType.AGENT,
                    name="Performance Analysis",
                    description="Analyze performance",
                    agent_id="performance-agent",
                    next_nodes=["aggregate"]
                ),
                WorkflowNode(
                    id="style",
                    type=NodeType.TOOL,
                    name="Style Check",
                    description="Check code style",
                    tool_name="style-checker",
                    next_nodes=["aggregate"]
                ),
                WorkflowNode(
                    id="aggregate",
                    type=NodeType.AGENT,
                    name="Aggregate Results",
                    description="Combine all review results",
                    agent_id="aggregator-agent",
                    next_nodes=["end"]
                ),
                WorkflowNode(
                    id="end",
                    type=NodeType.END,
                    name="Review Complete",
                    description="Code review complete",
                    next_nodes=[]
                )
            ],
            edges=[
                {"from": "start", "to": "parallel-checks"},
                {"from": "parallel-checks", "to": "syntax"},
                {"from": "parallel-checks", "to": "security"},
                {"from": "parallel-checks", "to": "performance"},
                {"from": "parallel-checks", "to": "style"},
                {"from": "syntax", "to": "aggregate"},
                {"from": "security", "to": "aggregate"},
                {"from": "performance", "to": "aggregate"},
                {"from": "style", "to": "aggregate"},
                {"from": "aggregate", "to": "end"}
            ],
            entry_node="start"
        )
        self.workflows[code_review_workflow.id] = code_review_workflow
    
    def _build_graph(self, workflow: WorkflowDefinition) -> StateGraph:
        """Build LangGraph from workflow definition"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, using mock implementation")
            return StateGraph(WorkflowState)
        
        # Create graph
        graph = StateGraph(WorkflowState)
        
        # Add nodes
        for node in workflow.nodes:
            if node.type == NodeType.AGENT:
                graph.add_node(node.id, self._create_agent_node(node))
            elif node.type == NodeType.TOOL:
                graph.add_node(node.id, self._create_tool_node(node))
            elif node.type == NodeType.DECISION:
                graph.add_node(node.id, self._create_decision_node(node))
            elif node.type == NodeType.PARALLEL:
                graph.add_node(node.id, self._create_parallel_node(node))
            elif node.type != NodeType.START and node.type != NodeType.END:
                graph.add_node(node.id, self._create_generic_node(node))
        
        # Add edges
        for edge in workflow.edges:
            from_node = edge["from"]
            to_node = edge["to"]
            
            if to_node == "end":
                graph.add_edge(from_node, END)
            else:
                # Check if it's a decision node
                from_node_obj = next((n for n in workflow.nodes if n.id == from_node), None)
                if from_node_obj and from_node_obj.type == NodeType.DECISION:
                    # Add conditional edges
                    graph.add_conditional_edges(
                        from_node,
                        self._create_condition_function(from_node_obj),
                        {node_id: node_id for node_id in from_node_obj.next_nodes}
                    )
                else:
                    graph.add_edge(from_node, to_node)
        
        # Set entry point
        if workflow.entry_node != "start":
            graph.set_entry_point(workflow.entry_node)
        
        return graph
    
    def _create_agent_node(self, node: WorkflowNode):
        """Create an agent node function"""
        async def agent_node(state: WorkflowState) -> WorkflowState:
            logger.info(f"Executing agent node: {node.name}")
            
            # Simulate agent execution
            state["current_step"] = node.id
            state["messages"].append({
                "role": "assistant",
                "content": f"Agent {node.agent_id} processing: {node.description}"
            })
            
            # Add result
            state["results"].append({
                "node_id": node.id,
                "agent_id": node.agent_id,
                "status": "completed",
                "output": f"Result from {node.name}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
        
        return agent_node
    
    def _create_tool_node(self, node: WorkflowNode):
        """Create a tool node function"""
        async def tool_node(state: WorkflowState) -> WorkflowState:
            logger.info(f"Executing tool node: {node.name}")
            
            state["current_step"] = node.id
            state["messages"].append({
                "role": "tool",
                "content": f"Tool {node.tool_name} executed: {node.description}"
            })
            
            # Simulate tool execution
            state["results"].append({
                "node_id": node.id,
                "tool_name": node.tool_name,
                "status": "completed",
                "output": f"Tool result from {node.name}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
        
        return tool_node
    
    def _create_decision_node(self, node: WorkflowNode):
        """Create a decision node function"""
        async def decision_node(state: WorkflowState) -> WorkflowState:
            logger.info(f"Executing decision node: {node.name}")
            
            state["current_step"] = node.id
            
            # Make decision based on criteria
            decision_made = False
            if node.decision_criteria:
                # Example: Check if we have enough sources
                if "min_sources" in node.decision_criteria:
                    sources_count = len([r for r in state["results"] if "source" in r.get("output", "")])
                    decision_made = sources_count >= node.decision_criteria["min_sources"]
            
            state["context"]["decision_result"] = decision_made
            state["messages"].append({
                "role": "system",
                "content": f"Decision at {node.name}: {'Proceed' if decision_made else 'Loop back'}"
            })
            
            return state
        
        return decision_node
    
    def _create_parallel_node(self, node: WorkflowNode):
        """Create a parallel execution node"""
        async def parallel_node(state: WorkflowState) -> WorkflowState:
            logger.info(f"Executing parallel node: {node.name}")
            
            state["current_step"] = node.id
            state["context"]["parallel_execution"] = True
            state["messages"].append({
                "role": "system",
                "content": f"Starting parallel execution: {node.description}"
            })
            
            return state
        
        return parallel_node
    
    def _create_generic_node(self, node: WorkflowNode):
        """Create a generic node function"""
        async def generic_node(state: WorkflowState) -> WorkflowState:
            logger.info(f"Executing node: {node.name}")
            
            state["current_step"] = node.id
            state["messages"].append({
                "role": "system",
                "content": f"Processing {node.name}: {node.description}"
            })
            
            return state
        
        return generic_node
    
    def _create_condition_function(self, node: WorkflowNode):
        """Create a condition function for decision nodes"""
        def condition(state: WorkflowState) -> str:
            # Check decision result
            if state["context"].get("decision_result", False):
                # Proceed to next step (usually summarize)
                return node.next_nodes[1] if len(node.next_nodes) > 1 else node.next_nodes[0]
            else:
                # Loop back (usually to search)
                return node.next_nodes[0]
        
        return condition
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "langgraph_integration",
                "status": "healthy",
                "langgraph_available": LANGGRAPH_AVAILABLE,
                "workflows_loaded": len(self.workflows),
                "active_executions": len(self.active_executions),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/workflows")
        async def list_workflows():
            """List available workflows"""
            return {
                "workflows": [
                    {
                        "id": wf.id,
                        "name": wf.name,
                        "description": wf.description,
                        "nodes_count": len(wf.nodes),
                        "entry_node": wf.entry_node
                    }
                    for wf in self.workflows.values()
                ],
                "count": len(self.workflows)
            }
        
        @self.app.get("/workflows/{workflow_id}")
        async def get_workflow(workflow_id: str):
            """Get workflow details"""
            if workflow_id not in self.workflows:
                raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            return workflow.dict()
        
        @self.app.post("/workflows")
        async def create_workflow(workflow: WorkflowDefinition):
            """Create a new workflow"""
            if workflow.id in self.workflows:
                raise HTTPException(status_code=400, detail=f"Workflow {workflow.id} already exists")
            
            self.workflows[workflow.id] = workflow
            
            # Build and compile graph
            graph = self._build_graph(workflow)
            checkpointer = MemorySaver() if LANGGRAPH_AVAILABLE else None
            compiled_graph = graph.compile(checkpointer=checkpointer)
            
            self.graph_instances[workflow.id] = compiled_graph
            self.checkpointers[workflow.id] = checkpointer
            
            logger.info(f"Created workflow: {workflow.id}")
            
            return {"status": "created", "workflow_id": workflow.id}
        
        @self.app.post("/execute")
        async def execute_workflow(execution: WorkflowExecution, background_tasks: BackgroundTasks):
            """Execute a workflow"""
            if execution.workflow_id not in self.workflows:
                raise HTTPException(status_code=404, detail=f"Workflow {execution.workflow_id} not found")
            
            execution_id = str(uuid.uuid4())
            
            # Initialize state
            initial_state = WorkflowState(
                messages=[{"role": "user", "content": json.dumps(execution.input_data)}],
                current_step="start",
                context=execution.context,
                results=[],
                error=None,
                metadata={"execution_id": execution_id, "started_at": datetime.utcnow().isoformat()}
            )
            
            # Store execution
            self.active_executions[execution_id] = {
                "workflow_id": execution.workflow_id,
                "state": initial_state,
                "status": "running",
                "started_at": datetime.utcnow()
            }
            
            # Execute in background
            background_tasks.add_task(
                self._execute_workflow_async,
                execution_id,
                execution.workflow_id,
                initial_state,
                execution.checkpoint_id
            )
            
            return {
                "execution_id": execution_id,
                "workflow_id": execution.workflow_id,
                "status": "started"
            }
        
        @self.app.get("/executions/{execution_id}")
        async def get_execution_status(execution_id: str):
            """Get execution status"""
            if execution_id not in self.active_executions:
                raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
            
            execution = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "workflow_id": execution["workflow_id"],
                "status": execution["status"],
                "current_step": execution["state"]["current_step"],
                "results_count": len(execution["state"]["results"]),
                "messages_count": len(execution["state"]["messages"]),
                "started_at": execution["started_at"].isoformat()
            }
        
        @self.app.get("/executions/{execution_id}/results")
        async def get_execution_results(execution_id: str):
            """Get execution results"""
            if execution_id not in self.active_executions:
                raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
            
            execution = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "status": execution["status"],
                "results": execution["state"]["results"],
                "messages": execution["state"]["messages"],
                "error": execution["state"]["error"]
            }
        
        @self.app.post("/executions/{execution_id}/checkpoint")
        async def create_checkpoint(execution_id: str):
            """Create checkpoint for execution"""
            if execution_id not in self.active_executions:
                raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
            
            checkpoint_id = str(uuid.uuid4())
            
            # Store checkpoint
            self.cache.set(
                f"checkpoint:{checkpoint_id}",
                json.dumps({
                    "execution_id": execution_id,
                    "state": self.active_executions[execution_id]["state"],
                    "created_at": datetime.utcnow().isoformat()
                }),
                ttl=3600
            )
            
            return {"checkpoint_id": checkpoint_id}
    
    async def _execute_workflow_async(self, execution_id: str, workflow_id: str, 
                                     initial_state: WorkflowState, checkpoint_id: Optional[str]):
        """Execute workflow asynchronously"""
        try:
            logger.info(f"Starting workflow execution: {execution_id}")
            
            # Get compiled graph
            if workflow_id not in self.graph_instances:
                # Build graph if not cached
                workflow = self.workflows[workflow_id]
                graph = self._build_graph(workflow)
                checkpointer = MemorySaver() if LANGGRAPH_AVAILABLE else None
                compiled_graph = graph.compile(checkpointer=checkpointer)
                self.graph_instances[workflow_id] = compiled_graph
                self.checkpointers[workflow_id] = checkpointer
            
            graph = self.graph_instances[workflow_id]
            
            # Restore from checkpoint if provided
            if checkpoint_id:
                checkpoint_data = self.cache.get(f"checkpoint:{checkpoint_id}")
                if checkpoint_data:
                    checkpoint = json.loads(checkpoint_data)
                    initial_state = checkpoint["state"]
                    logger.info(f"Restored from checkpoint: {checkpoint_id}")
            
            # Simulate execution (in real implementation, would invoke graph)
            if LANGGRAPH_AVAILABLE:
                # Execute with real LangGraph
                config = {"configurable": {"thread_id": execution_id}}
                result = await graph.ainvoke(initial_state, config)
            else:
                # Mock execution
                result = initial_state
                result["current_step"] = "end"
                result["results"].append({
                    "node_id": "final",
                    "status": "completed",
                    "output": "Workflow completed successfully",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Update execution status
            self.active_executions[execution_id]["state"] = result
            self.active_executions[execution_id]["status"] = "completed"
            
            logger.info(f"Workflow execution completed: {execution_id}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.active_executions[execution_id]["status"] = "failed"
            self.active_executions[execution_id]["state"]["error"] = str(e)
    
    async def startup(self):
        """Startup tasks"""
        logger.info("LangGraph Integration Service starting up...")
        
        # Build graphs for pre-loaded workflows
        for workflow_id, workflow in self.workflows.items():
            graph = self._build_graph(workflow)
            checkpointer = MemorySaver() if LANGGRAPH_AVAILABLE else None
            compiled_graph = graph.compile(checkpointer=checkpointer)
            self.graph_instances[workflow_id] = compiled_graph
            self.checkpointers[workflow_id] = checkpointer
        
        logger.info(f"LangGraph ready with {len(self.workflows)} workflows")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("LangGraph Integration Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = LangGraphService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("LANGGRAPH_PORT", 8016))
    logger.info(f"Starting LangGraph Integration Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()