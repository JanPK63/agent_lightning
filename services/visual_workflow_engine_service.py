#!/usr/bin/env python3
"""
Visual Workflow Engine Microservice
Handles visual project management and workflow execution
Based on Agent Lightning microservices architecture
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import networkx as nx
import httpx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared components
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectStatus(str, Enum):
    """Visual project status"""
    DRAFT = "draft"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class ProjectCreate(BaseModel):
    """Create new visual project"""
    name: str = Field(description="Project name")
    description: str = Field(description="Project description")
    agent_id: Optional[str] = Field(default=None, description="Associated agent ID")
    template_id: Optional[str] = Field(default=None, description="Template to use")


class ComponentAdd(BaseModel):
    """Add component to project"""
    project_id: str = Field(description="Project ID")
    component_type: str = Field(description="Component type")
    component_id: str = Field(description="Component ID from library")
    position: Dict[str, float] = Field(description="Position on canvas")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Component configuration")


class ConnectionCreate(BaseModel):
    """Create connection between components"""
    project_id: str = Field(description="Project ID")
    source_id: str = Field(description="Source component ID")
    source_port: str = Field(description="Source port name")
    target_id: str = Field(description="Target component ID")
    target_port: str = Field(description="Target port name")


class WorkflowExecuteRequest(BaseModel):
    """Execute workflow request"""
    project_id: str = Field(description="Project ID")
    inputs: Optional[Dict[str, Any]] = Field(default=None, description="Workflow inputs")
    breakpoints: Optional[List[str]] = Field(default=None, description="Component IDs for breakpoints")


class VisualCodeBuilder:
    """Enterprise Visual Code Builder Component"""

    def __init__(self):
        self.blocks = {}
        self.connections = []

    def create_block(self, block_type: str, config: dict) -> str:
        """Create a new code block"""
        block_id = str(uuid.uuid4())
        self.blocks[block_id] = {
            "type": block_type,
            "config": config,
            "created_at": datetime.utcnow().isoformat()
        }
        return block_id

    def connect_blocks(self, source_id: str, target_id: str) -> bool:
        """Connect two blocks"""
        if source_id in self.blocks and target_id in self.blocks:
            self.connections.append({
                "source": source_id,
                "target": target_id
            })
            return True
        return False


class VisualProject:
    """Represents a visual programming project"""

    def __init__(self, project_id: str, name: str, description: str):
        self.id = project_id
        self.name = name
        self.description = description
        self.graph = nx.DiGraph()
        self.components = {}
        self.connections = []
        self.metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "status": ProjectStatus.DRAFT.value,
            "version": "1.0.0"
        }

    def add_component(self, component_id: str, component_data: dict):
        """Add component to project"""
        self.graph.add_node(component_id, **component_data)
        self.components[component_id] = component_data

    def add_connection(self, source: str, target: str, connection_data: dict):
        """Add connection between components"""
        self.graph.add_edge(source, target, **connection_data)
        self.connections.append({
            "source": source,
            "target": target,
            **connection_data
        })

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate project structure"""
        errors = []

        # Only check connectivity if there are components
        if len(self.graph.nodes) > 0:
            # Check for cycles in logic flow
            if not nx.is_directed_acyclic_graph(self.graph):
                errors.append("Project contains circular dependencies")

            # Check for disconnected components only if there are multiple components
            if len(self.graph.nodes) > 1 and not nx.is_weakly_connected(self.graph):
                errors.append("Project has disconnected components")

        # Validate component configurations
        for comp_id, comp_data in self.components.items():
            if "type" not in comp_data:
                errors.append(f"Component {comp_id} missing type")

        return len(errors) == 0, errors

    def execute_workflow(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the workflow"""
        result = {"status": "executed", "outputs": {}}

        # Simple execution logic - in production this would be more sophisticated
        # Process components in topological order
        try:
            if nx.is_directed_acyclic_graph(self.graph):
                execution_order = list(nx.topological_sort(self.graph))

                component_outputs = {}
                if inputs:
                    component_outputs.update(inputs)

                for comp_id in execution_order:
                    comp_data = self.components[comp_id]
                    comp_type = comp_data.get("type")

                    # Mock execution based on component type
                    if comp_type == "data":
                        component_outputs[comp_id] = {"data": comp_data.get("config", {})}
                    elif comp_type == "logic":
                        component_outputs[comp_id] = {"result": "processed"}
                    elif comp_type == "ai":
                        component_outputs[comp_id] = {"prediction": "mock_result"}
                    else:
                        component_outputs[comp_id] = {"output": "executed"}

                result["outputs"] = component_outputs
            else:
                result["status"] = "error"
                result["error"] = "Circular dependencies detected"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "components": self.components,
            "connections": self.connections,
            "metadata": self.metadata
        }


class VisualWorkflowEngineService:
    """Visual Workflow Engine Microservice"""

    def __init__(self):
        self.app = FastAPI(title="Visual Workflow Engine Service", version="1.0.0")

        # Initialize components
        self.dal = DataAccessLayer("visual_workflow_engine")
        self.cache = get_cache()
        self.code_builder = VisualCodeBuilder()

        # Active projects
        self.active_projects: Dict[str, VisualProject] = {}

        # WebSocket connections for real-time collaboration
        self.websocket_connections: Dict[str, List[WebSocket]] = {}

        # HTTP client for inter-service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Service URLs
        self.component_registry_url = os.getenv("VISUAL_COMPONENT_REGISTRY_URL", "http://localhost:8008")

        logger.info("✅ Connected to shared database and initialized workflow engine")

        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()

    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            health_status = self.dal.health_check()
            return {
                "service": "visual_workflow_engine",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "active_projects": len(self.active_projects),
                "websocket_connections": sum(len(conns) for conns in self.websocket_connections.values()),
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.post("/projects")
        async def create_project(project: ProjectCreate):
            """Create new visual project"""
            try:
                project_id = str(uuid.uuid4())

                # Create project instance
                visual_project = VisualProject(
                    project_id=project_id,
                    name=project.name,
                    description=project.description
                )

                # Load template if specified
                if project.template_id:
                    async with self.http_client as client:
                        response = await client.get(f"{self.component_registry_url}/templates/{project.template_id}")
                        if response.status_code == 200:
                            template_data = response.json().get("template", {})
                            # Apply template components
                            for comp in template_data.get("components", []):
                                visual_project.add_component(comp["id"], comp)
                            for conn in template_data.get("connections", []):
                                visual_project.add_connection(
                                    conn["source"],
                                    conn["target"],
                                    conn
                                )

                # Store in cache and memory
                self.active_projects[project_id] = visual_project
                self.cache.set(f"visual_project:{project_id}", visual_project.to_dict(), ttl=3600)

                # If associated with agent, emit event
                if project.agent_id:
                    self.dal.event_bus.emit(EventChannel.AGENT_UPDATED, {
                        "agent_id": project.agent_id,
                        "visual_project_id": project_id
                    })

                logger.info(f"Created visual project {project_id}: {project.name}")
                return visual_project.to_dict()

            except Exception as e:
                logger.error(f"Failed to create project: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/projects")
        async def list_projects():
            """List all visual projects"""
            try:
                projects = []

                # Get from cache
                for key in self.cache.redis_client.keys("visual_project:*"):
                    project = self.cache.get(key)
                    if project:
                        projects.append(project)

                return {
                    "projects": projects,
                    "count": len(projects),
                    "active": list(self.active_projects.keys())
                }

            except Exception as e:
                logger.error(f"Failed to list projects: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/projects/{project_id}")
        async def get_project(project_id: str):
            """Get project details"""
            try:
                # Check active projects first
                if project_id in self.active_projects:
                    return self.active_projects[project_id].to_dict()

                # Check cache
                project = self.cache.get(f"visual_project:{project_id}")
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")

                return project

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get project: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/components/add")
        async def add_component(request: ComponentAdd):
            """Add component to project"""
            try:
                project = self._get_active_project(request.project_id)

                # Validate component exists in registry
                async with self.http_client as client:
                    response = await client.get(f"{self.component_registry_url}/components/{request.component_type}/{request.component_id}")
                    if response.status_code != 200:
                        raise HTTPException(status_code=404, detail="Component not found in registry")

                    component_info = response.json().get("component", {})

                # Add to project
                comp_instance_id = f"{request.component_id}_{uuid.uuid4().hex[:8]}"
                component_data = {
                    "type": request.component_type,
                    "component_id": request.component_id,
                    "position": request.position,
                    "config": request.config or component_info.get("default_config", {}),
                    "metadata": component_info.get("metadata", {})
                }

                project.add_component(comp_instance_id, component_data)

                # Update cache
                self.cache.set(
                    f"visual_project:{request.project_id}",
                    project.to_dict(),
                    ttl=3600
                )

                # Notify WebSocket clients
                await self._broadcast_update(request.project_id, {
                    "type": "component_added",
                    "component_id": comp_instance_id,
                    "data": component_data
                })

                # Emit event
                self.dal.event_bus.emit(EventChannel.COMPONENT_USED, {
                    "component_id": request.component_id,
                    "project_id": request.project_id
                })

                return {
                    "component_id": comp_instance_id,
                    "status": "added"
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to add component: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/connections/create")
        async def create_connection(request: ConnectionCreate):
            """Create connection between components"""
            try:
                project = self._get_active_project(request.project_id)

                # Validate components exist
                if request.source_id not in project.components:
                    raise HTTPException(status_code=404, detail=f"Source component {request.source_id} not found")
                if request.target_id not in project.components:
                    raise HTTPException(status_code=404, detail=f"Target component {request.target_id} not found")

                # Create connection
                connection_data = {
                    "source_port": request.source_port,
                    "target_port": request.target_port,
                    "created_at": datetime.utcnow().isoformat()
                }

                project.add_connection(request.source_id, request.target_id, connection_data)

                # Validate project after connection
                valid, errors = project.validate()
                if not valid:
                    # Rollback connection
                    project.graph.remove_edge(request.source_id, request.target_id)
                    project.connections.pop()
                    raise HTTPException(status_code=400, detail=f"Invalid connection: {', '.join(errors)}")

                # Update cache
                self.cache.set(
                    f"visual_project:{request.project_id}",
                    project.to_dict(),
                    ttl=3600
                )

                # Notify WebSocket clients
                await self._broadcast_update(request.project_id, {
                    "type": "connection_created",
                    "connection": {
                        "source": request.source_id,
                        "target": request.target_id,
                        **connection_data
                    }
                })

                return {"status": "connected"}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/workflows/execute")
        async def execute_workflow(request: WorkflowExecuteRequest):
            """Execute a visual workflow"""
            try:
                project = self._get_active_project(request.project_id)

                # Validate project first
                valid, errors = project.validate()
                if not valid:
                    raise HTTPException(status_code=400, detail=f"Project validation failed: {', '.join(errors)}")

                # Execute workflow
                result = project.execute_workflow(request.inputs)

                # Emit event
                self.dal.event_bus.emit(EventChannel.WORKFLOW_EXECUTED, {
                    "project_id": request.project_id,
                    "status": result.get("status"),
                    "execution_time": datetime.utcnow().isoformat()
                })

                return result

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to execute workflow: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/projects/{project_id}")
        async def delete_project(project_id: str):
            """Delete a project"""
            try:
                # Remove from active projects
                if project_id in self.active_projects:
                    del self.active_projects[project_id]

                # Remove from cache
                self.cache.delete(f"visual_project:{project_id}")

                # Emit event
                self.dal.event_bus.emit(EventChannel.PROJECT_DELETED, {
                    "project_id": project_id
                })

                return {"status": "deleted", "project_id": project_id}

            except Exception as e:
                logger.error(f"Failed to delete project: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws/{project_id}")
        async def websocket_endpoint(websocket: WebSocket, project_id: str):
            """WebSocket for real-time collaboration"""
            await websocket.accept()

            # Add to connections
            if project_id not in self.websocket_connections:
                self.websocket_connections[project_id] = []
            self.websocket_connections[project_id].append(websocket)

            try:
                while True:
                    # Receive updates from client
                    data = await websocket.receive_json()

                    # Process update
                    update_type = data.get("type")

                    if update_type == "component_move":
                        # Update component position
                        await self._handle_component_move(project_id, data)
                    elif update_type == "selection_change":
                        # Broadcast selection to other clients
                        await self._broadcast_update(project_id, data, exclude=websocket)

            except WebSocketDisconnect:
                # Remove from connections
                self.websocket_connections[project_id].remove(websocket)
                if not self.websocket_connections[project_id]:
                    del self.websocket_connections[project_id]

    def _get_active_project(self, project_id: str) -> VisualProject:
        """Get active project or load from cache"""
        if project_id not in self.active_projects:
            # Try to load from cache
            cached = self.cache.get(f"visual_project:{project_id}")
            if not cached:
                raise HTTPException(status_code=404, detail="Project not found")

            # Reconstruct project
            project = VisualProject(
                project_id=cached["id"],
                name=cached["name"],
                description=cached["description"]
            )
            project.components = cached["components"]
            project.connections = cached["connections"]
            project.metadata = cached["metadata"]

            # Rebuild graph
            for comp_id, comp_data in project.components.items():
                project.graph.add_node(comp_id, **comp_data)
            for conn in project.connections:
                project.graph.add_edge(conn["source"], conn["target"], **conn)

            self.active_projects[project_id] = project

        return self.active_projects[project_id]

    async def _broadcast_update(self, project_id: str, update: dict, exclude: Optional[WebSocket] = None):
        """Broadcast update to all connected clients"""
        if project_id in self.websocket_connections:
            for ws in self.websocket_connections[project_id]:
                if ws != exclude:
                    try:
                        await ws.send_json(update)
                    except:
                        # Connection might be closed
                        pass

    async def _handle_component_move(self, project_id: str, data: dict):
        """Handle component position update"""
        project = self._get_active_project(project_id)
        component_id = data.get("component_id")
        position = data.get("position")

        if component_id in project.components:
            project.components[component_id]["position"] = position

            # Update cache
            self.cache.set(
                f"visual_project:{project_id}",
                project.to_dict(),
                ttl=3600
            )

            # Broadcast to other clients
            await self._broadcast_update(project_id, data)

    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""

        def on_agent_created(event):
            """Handle agent creation - could auto-create visual project"""
            agent_id = event.data.get('agent_id')
            logger.info(f"Agent {agent_id} created - workflow engine notified")

        def on_task_completed(event):
            """Handle task completion - update visual feedback"""
            task_id = event.data.get('task_id')
            project_id = event.data.get('visual_project_id')

            if project_id:
                # Update visual feedback for completed task
                asyncio.create_task(self._broadcast_update(project_id, {
                    "type": "task_completed",
                    "task_id": task_id,
                    "timestamp": datetime.utcnow().isoformat()
                }))

        # Register handlers
        self.dal.event_bus.on(EventChannel.AGENT_CREATED, on_agent_created)
        self.dal.event_bus.on(EventChannel.TASK_COMPLETED, on_task_completed)

        logger.info("Event handlers registered for workflow engine service")

    async def startup(self):
        """Startup tasks"""
        logger.info("Visual Workflow Engine Service starting up...")

        # Verify connections
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")

        logger.info("Visual Workflow Engine Service ready")

    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual Workflow Engine Service shutting down...")

        # Close HTTP client
        await self.http_client.aclose()

        # Close all WebSocket connections
        for project_id, connections in self.websocket_connections.items():
            for ws in connections:
                await ws.close()

        # Save active projects to cache
        for project_id, project in self.active_projects.items():
            self.cache.set(
                f"visual_project:{project_id}",
                project.to_dict(),
                ttl=3600
            )

        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn

    service = VisualWorkflowEngineService()

    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)

    # Run service
    port = int(os.getenv("VISUAL_WORKFLOW_ENGINE_PORT", 8007))
    logger.info(f"Starting Visual Workflow Engine Service on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()