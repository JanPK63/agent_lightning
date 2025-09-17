#!/usr/bin/env python3
"""
Spec-Driven Development Service for Agent Lightning

This service provides REST endpoints for managing workflow specifications,
translating them to executable plans, and executing them via LangGraph or
the Agent Lightning server.
"""

import time
import uuid
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from agentlightning.types import (
    Spec,
    SpecPlan,
    SpecExecution,
    Task,
    GenericResponse,
)


class SpecService:
    """
    Service for managing workflow specifications and their execution.
    """

    def __init__(self):
        self.specs: Dict[str, Spec] = {}  # In-memory storage for MVP
        self.executions: Dict[str, SpecExecution] = {}
        self.specs_dir = Path("./specs")
        self.specs_dir.mkdir(exist_ok=True)

    def validate_spec(self, spec: Spec) -> bool:
        """
        Validate a specification for correctness.
        """
        if not spec.id or not spec.name:
            return False

        # Check workflow step dependencies
        step_ids = {step.id for step in spec.workflow}
        for step in spec.workflow:
            for dep in step.dependencies:
                if dep not in step_ids:
                    return False

        return True

    def save_spec(self, spec: Spec) -> None:
        """
        Save a specification to memory and disk.
        """
        spec.updated_at = time.time()
        if not spec.created_at:
            spec.created_at = spec.updated_at

        self.specs[spec.id] = spec

        # Save to file
        spec_file = self.specs_dir / f"{spec.id}.json"
        with open(spec_file, 'w') as f:
            f.write(spec.model_dump_json(indent=2))

    def get_spec(self, spec_id: str) -> Optional[Spec]:
        """
        Retrieve a specification by ID.
        """
        return self.specs.get(spec_id)

    def generate_plan(self, spec: Spec) -> SpecPlan:
        """
        Generate an execution plan from a specification.
        """
        tasks = []

        # Convert workflow steps to tasks
        for step in spec.workflow:
            task = Task(
                rollout_id=f"{spec.id}_{step.id}_{uuid.uuid4().hex[:8]}",
                input={
                    "step_id": step.id,
                    "step_name": step.name,
                    "description": step.description,
                    "inputs": step.inputs,
                    "agent_type": step.agent_type,
                },
                metadata={
                    "spec_id": spec.id,
                    "step_dependencies": step.dependencies,
                }
            )
            tasks.append(task)

        # Create workflow definition for LangGraph
        workflow_def = {
            "nodes": [
                {
                    "id": step.id,
                    "name": step.name,
                    "type": step.agent_type or "generic",
                    "dependencies": step.dependencies,
                }
                for step in spec.workflow
            ],
            "edges": [
                {"from": dep, "to": step.id}
                for step in spec.workflow
                for dep in step.dependencies
            ]
        }

        return SpecPlan(
            spec_id=spec.id,
            tasks=tasks,
            workflow_definition=workflow_def,
            resource_requirements=spec.resources,
            metadata={"generated_at": time.time()}
        )

    def execute_plan(self, plan: SpecPlan) -> str:
        """
        Execute a plan and return execution ID.
        """
        execution_id = f"exec_{plan.spec_id}_{uuid.uuid4().hex[:8]}"

        execution = SpecExecution(
            spec_id=plan.spec_id,
            execution_id=execution_id,
            status="running",
            started_at=time.time(),
            metadata={"plan_tasks": len(plan.tasks)}
        )

        self.executions[execution_id] = execution

        # For MVP, execute using DevTaskLoader and AgentRunner
        try:
            # This would be done asynchronously in production
            self._execute_plan_sync(plan, execution)
        except Exception as e:
            execution.status = "failed"
            execution.errors.append(str(e))
            execution.completed_at = time.time()

        return execution_id

    def _execute_plan_sync(self, plan: SpecPlan,
                           execution: SpecExecution) -> None:
        """
        Synchronously execute a plan (for MVP).
        """
        # Create a simple agent for demo purposes
        from examples.noop_demo_agent import SimpleNoOpLitAgent

        agent = SimpleNoOpLitAgent()
        rollouts = []

        for task in plan.tasks:
            # Simulate task execution
            try:
                rollout_result = agent.training_rollout(
                    task=task.input,
                    rollout_id=task.rollout_id,
                    resources={}  # Empty for MVP
                )
                rollouts.append(rollout_result)
            except Exception as e:
                error_msg = f"Task {task.rollout_id} failed: {str(e)}"
                execution.errors.append(error_msg)

        execution.results = {"rollouts": [r.model_dump() for r in rollouts]}
        execution.rollouts = rollouts
        execution.status = "completed"
        execution.completed_at = time.time()

    def get_execution(self, execution_id: str) -> Optional[SpecExecution]:
        """
        Get execution status by ID.
        """
        return self.executions.get(execution_id)


# FastAPI Application
app = FastAPI(
    title="Agent Lightning Spec Service",
    description="REST API for spec-driven development",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
spec_service = SpecService()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "spec-service"}


@app.post("/specs", response_model=GenericResponse)
async def create_spec(spec: Spec):
    """
    Create and validate a new specification.
    """
    if not spec_service.validate_spec(spec):
        raise HTTPException(status_code=400, detail="Invalid specification")

    spec_service.save_spec(spec)

    return GenericResponse(
        status="success",
        message=f"Specification '{spec.name}' created successfully",
        data={"spec_id": spec.id}
    )


@app.get("/specs/{spec_id}")
async def get_spec(spec_id: str):
    """
    Retrieve a specification by ID.
    """
    spec = spec_service.get_spec(spec_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Specification not found")

    return spec


@app.post("/specs/{spec_id}/plan", response_model=SpecPlan)
async def generate_plan(spec_id: str):
    """
    Generate an execution plan from a specification.
    """
    spec = spec_service.get_spec(spec_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Specification not found")

    plan = spec_service.generate_plan(spec)
    return plan


@app.post("/specs/{spec_id}/execute", response_model=GenericResponse)
async def execute_spec(spec_id: str, background_tasks: BackgroundTasks):
    """
    Execute a specification.
    """
    spec = spec_service.get_spec(spec_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Specification not found")

    # Generate plan first
    plan = spec_service.generate_plan(spec)

    # Execute plan
    execution_id = spec_service.execute_plan(plan)

    return GenericResponse(
        status="success",
        message=f"Execution started for spec '{spec.name}'",
        data={"execution_id": execution_id}
    )


@app.get("/executions/{execution_id}")
async def get_execution(execution_id: str):
    """
    Get execution status by ID.
    """
    execution = spec_service.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    return execution


@app.get("/specs")
async def list_specs():
    """
    List all specifications.
    """
    return {
        "specs": [
            {
                "id": spec.id,
                "name": spec.name,
                "description": spec.description,
                "version": spec.version,
                "created_at": spec.created_at,
                "updated_at": spec.updated_at
            }
            for spec in spec_service.specs.values()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8029)