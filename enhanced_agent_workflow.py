"""
Enhanced Agent Workflow System
Integrates LangChain agents with tools for improved multi-agent coordination
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from langchain_agent_wrapper import LangChainAgentManager
from services.rl_orchestrator_improved import ImprovedRLOrchestrator

logger = logging.getLogger(__name__)


class WorkflowTask:
    """Represents a task in the workflow"""
    
    def __init__(self, task_id: str, description: str, dependencies: List[str] = None):
        self.task_id = task_id
        self.description = description
        self.dependencies = dependencies or []
        self.status = "pending"
        self.assigned_agent = None
        self.result = None
        self.started_at = None
        self.completed_at = None


class EnhancedAgentWorkflow:
    """
    Enhanced workflow system that coordinates multiple LangChain agents with tools
    """
    
    def __init__(self):
        self.agent_manager = LangChainAgentManager()
        self.orchestrator = ImprovedRLOrchestrator()
        self.workflows: Dict[str, List[WorkflowTask]] = {}
        self.active_sessions: Dict[str, str] = {}  # workflow_id -> session_id
        
        logger.info(f"Enhanced workflow initialized with {len(self.agent_manager.agents)} agents")
    
    async def create_workflow(self, workflow_id: str, tasks: List[Dict[str, Any]]) -> str:
        """
        Create a new multi-agent workflow
        
        Args:
            workflow_id: Unique workflow identifier
            tasks: List of task definitions with descriptions and dependencies
            
        Returns:
            Workflow session ID
        """
        workflow_tasks = []
        
        for task_def in tasks:
            task = WorkflowTask(
                task_id=task_def["task_id"],
                description=task_def["description"],
                dependencies=task_def.get("dependencies", [])
            )
            workflow_tasks.append(task)
        
        self.workflows[workflow_id] = workflow_tasks
        session_id = f"workflow_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_sessions[workflow_id] = session_id
        
        logger.info(f"Created workflow {workflow_id} with {len(workflow_tasks)} tasks")
        return session_id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow with dependency resolution and agent coordination
        
        Args:
            workflow_id: Workflow to execute
            
        Returns:
            Workflow execution results
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        tasks = self.workflows[workflow_id]
        session_id = self.active_sessions[workflow_id]
        results = {}
        
        logger.info(f"Executing workflow {workflow_id} with session {session_id}")
        
        # Execute tasks in dependency order
        while any(task.status == "pending" for task in tasks):
            # Find ready tasks (no pending dependencies)
            ready_tasks = [
                task for task in tasks 
                if task.status == "pending" and 
                all(dep_task.status == "completed" for dep_task in tasks if dep_task.task_id in task.dependencies)
            ]
            
            if not ready_tasks:
                # Check for circular dependencies or stuck tasks
                pending_tasks = [task for task in tasks if task.status == "pending"]
                if pending_tasks:
                    logger.error(f"Workflow {workflow_id} stuck - possible circular dependencies")
                    break
            
            # Execute ready tasks in parallel
            execution_tasks = []
            for task in ready_tasks:
                execution_tasks.append(self._execute_single_task(task, session_id, results))
            
            if execution_tasks:
                await asyncio.gather(*execution_tasks)
        
        # Compile final results
        workflow_result = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "status": "completed" if all(task.status == "completed" for task in tasks) else "failed",
            "tasks": {
                task.task_id: {
                    "status": task.status,
                    "assigned_agent": task.assigned_agent,
                    "result": task.result,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at
                }
                for task in tasks
            },
            "execution_summary": results
        }
        
        logger.info(f"Workflow {workflow_id} completed with status: {workflow_result['status']}")
        return workflow_result
    
    async def _execute_single_task(self, task: WorkflowTask, session_id: str, shared_results: Dict[str, Any]):
        """Execute a single task with the best available agent"""
        try:
            task.status = "running"
            task.started_at = datetime.utcnow().isoformat()
            
            # Select best agent for this task
            agent_id, confidence, reason, is_valid = await self.orchestrator.select_agent_with_capability_check(
                task.description
            )
            
            task.assigned_agent = agent_id
            logger.info(f"Task {task.task_id} assigned to {agent_id} (confidence: {confidence:.2f})")
            
            # Get the LangChain agent
            agent = self.agent_manager.get_agent(agent_id)
            if not agent:
                raise Exception(f"Agent {agent_id} not available")
            
            # Prepare context from previous task results
            context = self._build_task_context(task, shared_results)
            
            # Execute task with agent
            full_prompt = f"{task.description}\n\nContext from previous tasks:\n{context}"
            result = agent.invoke(full_prompt, session_id)
            
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.utcnow().isoformat()
            
            # Store result for dependent tasks
            shared_results[task.task_id] = {
                "description": task.description,
                "result": result,
                "agent": agent_id,
                "completed_at": task.completed_at
            }
            
            # Provide feedback to orchestrator for learning
            await self._provide_task_feedback(task, True, 8.0)  # Assume success for now
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = "failed"
            task.result = f"Error: {str(e)}"
            task.completed_at = datetime.utcnow().isoformat()
            
            # Provide negative feedback
            if task.assigned_agent:
                await self._provide_task_feedback(task, False, 2.0)
            
            logger.error(f"Task {task.task_id} failed: {e}")
    
    def _build_task_context(self, current_task: WorkflowTask, shared_results: Dict[str, Any]) -> str:
        """Build context from completed dependency tasks"""
        context_parts = []
        
        for dep_id in current_task.dependencies:
            if dep_id in shared_results:
                dep_result = shared_results[dep_id]
                context_parts.append(f"Task '{dep_id}': {dep_result['result'][:200]}...")
        
        return "\n".join(context_parts) if context_parts else "No previous context available."
    
    async def _provide_task_feedback(self, task: WorkflowTask, success: bool, quality_score: float):
        """Provide feedback to the orchestrator for learning"""
        try:
            feedback = {
                "task_id": task.task_id,
                "agent_id": task.assigned_agent,
                "success": success,
                "quality_score": quality_score
            }
            
            # This would normally call the orchestrator's feedback endpoint
            # For now, we'll just log it
            logger.info(f"Feedback: {feedback}")
            
        except Exception as e:
            logger.error(f"Failed to provide feedback: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        tasks = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "total_tasks": len(tasks),
            "completed_tasks": sum(1 for task in tasks if task.status == "completed"),
            "failed_tasks": sum(1 for task in tasks if task.status == "failed"),
            "running_tasks": sum(1 for task in tasks if task.status == "running"),
            "pending_tasks": sum(1 for task in tasks if task.status == "pending"),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "assigned_agent": task.assigned_agent,
                    "description": task.description[:100] + "..." if len(task.description) > 100 else task.description
                }
                for task in tasks
            ]
        }
    
    def list_available_agents(self) -> Dict[str, Any]:
        """List all available agents and their capabilities"""
        agents_info = {}
        
        for agent_name, agent_wrapper in self.agent_manager.agents.items():
            agents_info[agent_name] = {
                "name": agent_name,
                "description": agent_wrapper.agent_config.description,
                "tools": [tool.name for tool in agent_wrapper.tools],
                "capabilities": {
                    "can_write_code": agent_wrapper.agent_config.capabilities.can_write_code,
                    "can_debug": agent_wrapper.agent_config.capabilities.can_debug,
                    "can_review_code": agent_wrapper.agent_config.capabilities.can_review_code,
                    "can_test": agent_wrapper.agent_config.capabilities.can_test,
                    "can_write_documentation": agent_wrapper.agent_config.capabilities.can_write_documentation
                }
            }
        
        return agents_info


# Example workflow definitions
EXAMPLE_WORKFLOWS = {
    "web_development": [
        {
            "task_id": "requirements",
            "description": "Analyze requirements for a simple web application with user authentication",
            "dependencies": []
        },
        {
            "task_id": "architecture",
            "description": "Design system architecture based on the requirements",
            "dependencies": ["requirements"]
        },
        {
            "task_id": "backend",
            "description": "Implement backend API with authentication endpoints",
            "dependencies": ["architecture"]
        },
        {
            "task_id": "frontend",
            "description": "Create frontend interface that connects to the backend API",
            "dependencies": ["backend"]
        },
        {
            "task_id": "testing",
            "description": "Write and execute tests for both frontend and backend",
            "dependencies": ["frontend"]
        },
        {
            "task_id": "deployment",
            "description": "Create deployment configuration and deploy the application",
            "dependencies": ["testing"]
        }
    ],
    
    "data_analysis": [
        {
            "task_id": "data_collection",
            "description": "Collect and validate sample dataset for analysis",
            "dependencies": []
        },
        {
            "task_id": "data_cleaning",
            "description": "Clean and preprocess the collected data",
            "dependencies": ["data_collection"]
        },
        {
            "task_id": "analysis",
            "description": "Perform statistical analysis and generate insights",
            "dependencies": ["data_cleaning"]
        },
        {
            "task_id": "visualization",
            "description": "Create charts and visualizations of the analysis results",
            "dependencies": ["analysis"]
        },
        {
            "task_id": "report",
            "description": "Write comprehensive report with findings and recommendations",
            "dependencies": ["visualization"]
        }
    ]
}


async def test_enhanced_workflow():
    """Test the enhanced workflow system"""
    workflow = EnhancedAgentWorkflow()
    
    # Test simple workflow
    test_tasks = [
        {
            "task_id": "task1",
            "description": "Calculate the factorial of 5 using Python",
            "dependencies": []
        },
        {
            "task_id": "task2", 
            "description": "Write a brief explanation of how factorial calculation works",
            "dependencies": ["task1"]
        }
    ]
    
    # Create and execute workflow
    session_id = await workflow.create_workflow("test_workflow", test_tasks)
    print(f"Created workflow with session: {session_id}")
    
    result = await workflow.execute_workflow("test_workflow")
    print(f"Workflow result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_workflow())