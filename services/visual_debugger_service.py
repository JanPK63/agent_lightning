#!/usr/bin/env python3
"""
Visual Debugger Microservice
Handles debugging sessions for visual projects
Based on Agent Lightning microservices architecture
"""

import os
import sys
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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


class DebugSessionCreate(BaseModel):
    """Create debug session request"""
    project_id: str = Field(description="Project ID to debug")
    breakpoints: Optional[List[str]] = Field(default=None, description="Component IDs for breakpoints")
    watch_variables: Optional[List[str]] = Field(default=None, description="Variables to watch")
    inputs: Optional[Dict[str, Any]] = Field(default=None, description="Initial inputs for debugging")


class DebugStep(BaseModel):
    """Debug step request"""
    session_id: str = Field(description="Debug session ID")
    action: str = Field(description="Debug action (step_over, step_into, continue, stop)")


class VariableWatch(BaseModel):
    """Variable watch update"""
    session_id: str = Field(description="Debug session ID")
    variable_name: str = Field(description="Variable name")
    value: Any = Field(description="Variable value")


class VisualDebugger:
    """Visual debugging component"""

    def __init__(self):
        self.sessions = {}

    def create_session(self, project: dict, breakpoints: list = None, watch_variables: list = None, inputs: dict = None) -> dict:
        """Create debug session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "project": project,
            "breakpoints": breakpoints or [],
            "watch_variables": watch_variables or [],
            "inputs": inputs or {},
            "state": "initialized",
            "current_component": None,
            "execution_stack": [],
            "variable_values": {},
            "breakpoints_hit": [],
            "created_at": datetime.utcnow().isoformat()
        }
        return self.sessions[session_id]

    def step_execution(self, session_id: str, action: str) -> dict:
        """Step through execution"""
        if session_id not in self.sessions:
            raise ValueError("Session not found")

        session = self.sessions[session_id]

        if action == "start":
            session["state"] = "running"
            session["current_component"] = self._get_next_component(session)
            return self._get_execution_state(session)

        elif action == "step_over":
            if session["current_component"]:
                # Execute current component and move to next
                self._execute_component(session, session["current_component"])
                session["current_component"] = self._get_next_component(session)

                # Check for breakpoints
                if session["current_component"] in session["breakpoints"]:
                    session["state"] = "paused"
                    session["breakpoints_hit"].append(session["current_component"])

            return self._get_execution_state(session)

        elif action == "continue":
            session["state"] = "running"
            # Continue execution until breakpoint or end
            while session["current_component"] and session["current_component"] not in session["breakpoints"]:
                self._execute_component(session, session["current_component"])
                session["current_component"] = self._get_next_component(session)

            if session["current_component"] in session["breakpoints"]:
                session["state"] = "paused"
                session["breakpoints_hit"].append(session["current_component"])
            elif not session["current_component"]:
                session["state"] = "completed"

            return self._get_execution_state(session)

        elif action == "stop":
            session["state"] = "stopped"
            return self._get_execution_state(session)

        return self._get_execution_state(session)

    def _get_next_component(self, session: dict) -> Optional[str]:
        """Get next component in execution order"""
        project = session["project"]
        components = project.get("components", {})
        connections = project.get("connections", [])

        # Simple topological execution order
        # In production, this would use proper graph traversal
        executed = set()
        for comp_id in session.get("execution_stack", []):
            executed.add(comp_id)

        for comp_id in components.keys():
            if comp_id not in executed:
                # Check if all dependencies are executed
                dependencies = [conn["source"] for conn in connections if conn["target"] == comp_id]
                if all(dep in executed for dep in dependencies):
                    return comp_id

        return None

    def _execute_component(self, session: dict, component_id: str):
        """Execute a single component"""
        project = session["project"]
        components = project.get("components", {})

        if component_id in components:
            comp_data = components[component_id]
            comp_type = comp_data.get("type")

            # Mock execution - update variable values
            if comp_type == "data":
                session["variable_values"][component_id] = {"data": comp_data.get("config", {})}
            elif comp_type == "logic":
                session["variable_values"][component_id] = {"result": "processed"}
            elif comp_type == "ai":
                session["variable_values"][component_id] = {"prediction": "debug_result"}

            session["execution_stack"].append(component_id)

    def _get_execution_state(self, session: dict) -> dict:
        """Get current execution state"""
        return {
            "session_id": list(self.sessions.keys())[list(self.sessions.values()).index(session)],
            "state": session["state"],
            "current_component": session["current_component"],
            "execution_stack": session["execution_stack"],
            "variable_values": session["variable_values"],
            "breakpoints_hit": session["breakpoints_hit"],
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get debug session"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete debug session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


class VisualDebuggerService:
    """Visual Debugger Microservice"""

    def __init__(self):
        self.app = FastAPI(title="Visual Debugger Service", version="1.0.0")

        # Initialize components
        self.dal = DataAccessLayer("visual_debugger")
        self.cache = get_cache()
        self.debugger = VisualDebugger()

        # WebSocket connections for real-time debugging
        self.websocket_connections: Dict[str, List[WebSocket]] = {}

        # HTTP client for inter-service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Service URLs
        self.workflow_engine_url = os.getenv("VISUAL_WORKFLOW_ENGINE_URL", "http://localhost:8007")

        logger.info("✅ Connected to shared database and initialized debugger")

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
                "service": "visual_debugger",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "active_sessions": len(self.debugger.sessions),
                "websocket_connections": sum(len(conns) for conns in self.websocket_connections.values()),
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.post("/sessions")
        async def create_debug_session(request: DebugSessionCreate):
            """Create debug session"""
            try:
                # Get project data from workflow engine
                async with self.http_client as client:
                    response = await client.get(f"{self.workflow_engine_url}/projects/{request.project_id}")
                    if response.status_code != 200:
                        raise HTTPException(status_code=404, detail="Project not found in workflow engine")
                    project_data = response.json()

                # Create debug session
                debug_session = self.debugger.create_session(
                    project_data,
                    breakpoints=request.breakpoints,
                    watch_variables=request.watch_variables,
                    inputs=request.inputs
                )

                # Store session
                session_id = str(uuid.uuid4())
                self.cache.set(f"debug_session:{session_id}", {
                    "session_id": session_id,
                    "project_id": request.project_id,
                    "session": debug_session,
                    "started_at": datetime.utcnow().isoformat()
                }, ttl=1800)

                # Emit event
                self.dal.event_bus.emit(EventChannel.DEBUG_SESSION_STARTED, {
                    "session_id": session_id,
                    "project_id": request.project_id
                })

                return {
                    "session_id": session_id,
                    "status": "created",
                    "state": debug_session["state"]
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create debug session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/sessions/step")
        async def step_debug_session(request: DebugStep):
            """Step through debug session"""
            try:
                # Get session from cache
                session_data = self.cache.get(f"debug_session:{request.session_id}")
                if not session_data:
                    raise HTTPException(status_code=404, detail="Debug session not found")

                # Step execution
                result = self.debugger.step_execution(request.session_id, request.action)

                # Update cache
                session_data["session"] = self.debugger.get_session(request.session_id)
                self.cache.set(f"debug_session:{request.session_id}", session_data, ttl=1800)

                # Broadcast to WebSocket clients
                await self._broadcast_debug_update(request.session_id, result)

                # Emit event
                self.dal.event_bus.emit(EventChannel.DEBUG_STEP_EXECUTED, {
                    "session_id": request.session_id,
                    "action": request.action,
                    "state": result["state"]
                })

                return result

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to step debug session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/sessions/{session_id}")
        async def get_debug_session(session_id: str):
            """Get debug session details"""
            try:
                session_data = self.cache.get(f"debug_session:{session_id}")
                if not session_data:
                    raise HTTPException(status_code=404, detail="Debug session not found")

                session = self.debugger.get_session(session_id)
                if session:
                    return self.debugger._get_execution_state(session)
                else:
                    return session_data["session"]

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get debug session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/sessions/{session_id}")
        async def delete_debug_session(session_id: str):
            """Delete debug session"""
            try:
                success = self.debugger.delete_session(session_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Debug session not found")

                # Remove from cache
                self.cache.delete(f"debug_session:{session_id}")

                # Emit event
                self.dal.event_bus.emit(EventChannel.DEBUG_SESSION_ENDED, {
                    "session_id": session_id
                })

                return {"status": "deleted", "session_id": session_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete debug session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/sessions")
        async def list_debug_sessions():
            """List active debug sessions"""
            try:
                sessions = []
                for key in self.cache.redis_client.keys("debug_session:*"):
                    session = self.cache.get(key)
                    if session:
                        sessions.append({
                            "session_id": session["session_id"],
                            "project_id": session.get("project_id"),
                            "state": session["session"]["state"],
                            "started_at": session["started_at"]
                        })

                return {
                    "sessions": sessions,
                    "count": len(sessions)
                }

            except Exception as e:
                logger.error(f"Failed to list debug sessions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket for real-time debugging"""
            await websocket.accept()

            # Add to connections
            if session_id not in self.websocket_connections:
                self.websocket_connections[session_id] = []
            self.websocket_connections[session_id].append(websocket)

            try:
                while True:
                    # Receive debug commands from client
                    data = await websocket.receive_json()

                    # Process debug command
                    command = data.get("command")
                    if command == "step":
                        action = data.get("action", "step_over")
                        result = self.debugger.step_execution(session_id, action)

                        # Update cache
                        session_data = self.cache.get(f"debug_session:{session_id}")
                        if session_data:
                            session_data["session"] = self.debugger.get_session(session_id)
                            self.cache.set(f"debug_session:{session_id}", session_data, ttl=1800)

                        # Send result back
                        await websocket.send_json(result)

                    elif command == "watch_variable":
                        # Update variable watch
                        var_name = data.get("variable_name")
                        var_value = data.get("value")
                        # In production, this would update the session's watch variables
                        await websocket.send_json({"status": "watched", "variable": var_name})

            except WebSocketDisconnect:
                # Remove from connections
                self.websocket_connections[session_id].remove(websocket)
                if not self.websocket_connections[session_id]:
                    del self.websocket_connections[session_id]

    async def _broadcast_debug_update(self, session_id: str, update: dict):
        """Broadcast debug update to all connected clients"""
        if session_id in self.websocket_connections:
            for ws in self.websocket_connections[session_id]:
                try:
                    await ws.send_json(update)
                except:
                    # Connection might be closed
                    pass

    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""

        def on_workflow_executed(event):
            """Handle workflow execution for debugging"""
            project_id = event.data.get('project_id')
            logger.info(f"Workflow {project_id} executed - debugger notified")

        # Register handlers
        self.dal.event_bus.on(EventChannel.WORKFLOW_EXECUTED, on_workflow_executed)

        logger.info("Event handlers registered for debugger service")

    async def startup(self):
        """Startup tasks"""
        logger.info("Visual Debugger Service starting up...")

        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")

        logger.info("Visual Debugger Service ready")

    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual Debugger Service shutting down...")

        # Close HTTP client
        await self.http_client.aclose()

        # Close all WebSocket connections
        for session_id, connections in self.websocket_connections.items():
            for ws in connections:
                await ws.close()

        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn

    service = VisualDebuggerService()

    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)

    # Run service
    port = int(os.getenv("VISUAL_DEBUGGER_PORT", 8010))
    logger.info(f"Starting Visual Debugger Service on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()