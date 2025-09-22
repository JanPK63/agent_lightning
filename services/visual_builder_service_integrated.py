#!/usr/bin/env python3
"""
Visual Code Builder API Gateway
Provides unified API interface for visual programming services
Routes requests to appropriate microservices
Based on Agent Lightning microservices architecture
"""

import os
import sys
import json
import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import logging

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
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


class VisualBuilderAPIGateway:
    """API Gateway for Visual Builder Microservices"""

    def __init__(self):
        self.app = FastAPI(title="Visual Code Builder API Gateway", version="3.0.0")

        # Initialize shared components
        self.dal = DataAccessLayer("visual_builder_gateway")
        self.cache = get_cache()

        # HTTP client for inter-service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Service URLs (would come from service discovery in production)
        self.workflow_engine_url = os.getenv("VISUAL_WORKFLOW_ENGINE_URL", "http://localhost:8007")
        self.component_registry_url = os.getenv("VISUAL_COMPONENT_REGISTRY_URL", "http://localhost:8008")
        self.code_generator_url = os.getenv("VISUAL_CODE_GENERATOR_URL", "http://localhost:8009")
        self.debugger_url = os.getenv("VISUAL_DEBUGGER_URL", "http://localhost:8010")
        self.deployment_url = os.getenv("VISUAL_DEPLOYMENT_URL", "http://localhost:8011")
        self.ai_assistant_url = os.getenv("VISUAL_AI_ASSISTANT_URL", "http://localhost:8012")

        # WebSocket connections for real-time collaboration
        self.websocket_connections: Dict[str, List[WebSocket]] = {}

        logger.info("✅ Initialized Visual Builder API Gateway")

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
        """Setup API routes - forward to appropriate microservices"""

        @self.app.get("/health")
        async def health():
            """Health check endpoint - aggregate health from all services"""
            try:
                health_status = await self._aggregate_health_checks()
                return {
                    "service": "visual_builder_gateway",
                    "status": health_status["overall"],
                    "services": health_status["services"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "service": "visual_builder_gateway",
                    "status": "degraded",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }

        @self.app.get("/")
        async def root():
            """Serve visual builder UI"""
            return HTMLResponse(content="""
            <html>
                <head>
                    <title>Visual Code Builder</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        h1 { color: #333; }
                        .status { color: green; }
                    </style>
                </head>
                <body>
                    <h1>Visual Code Builder API Gateway</h1>
                    <p class="status">Gateway is running on port 8006</p>
                    <p>WebSocket endpoint: ws://localhost:8006/ws/{project_id}</p>
                    <p>API documentation: <a href="/docs">/docs</a></p>
                    <p>Services:</p>
                    <ul>
                        <li>Workflow Engine: 8007</li>
                        <li>Component Registry: 8008</li>
                        <li>Code Generator: 8009</li>
                        <li>Debugger: 8010</li>
                        <li>Deployment: 8011</li>
                        <li>AI Assistant: 8012</li>
                    </ul>
                </body>
            </html>
            """)

        # Project management routes - forward to workflow engine
        @self.app.post("/projects")
        async def create_project(request: Request):
            """Create new visual project"""
            return await self._forward_request("POST", f"{self.workflow_engine_url}/projects", request)

        @self.app.get("/projects")
        async def list_projects(request: Request):
            """List all visual projects"""
            return await self._forward_request("GET", f"{self.workflow_engine_url}/projects", request)

        @self.app.get("/projects/{project_id}")
        async def get_project(project_id: str, request: Request):
            """Get project details"""
            return await self._forward_request("GET", f"{self.workflow_engine_url}/projects/{project_id}", request)

        @self.app.delete("/projects/{project_id}")
        async def delete_project(project_id: str, request: Request):
            """Delete a project"""
            return await self._forward_request("DELETE", f"{self.workflow_engine_url}/projects/{project_id}", request)

        # Component management routes - forward to workflow engine and component registry
        @self.app.post("/components/add")
        async def add_component(request: Request):
            """Add component to project"""
            return await self._forward_request("POST", f"{self.workflow_engine_url}/components/add", request)

        @self.app.post("/connections/create")
        async def create_connection(request: Request):
            """Create connection between components"""
            return await self._forward_request("POST", f"{self.workflow_engine_url}/connections/create", request)

        @self.app.get("/components/library")
        async def get_component_library(request: Request):
            """Get available components"""
            return await self._forward_request("GET", f"{self.component_registry_url}/components", request)

        @self.app.get("/templates")
        async def get_templates(request: Request):
            """Get available project templates"""
            return await self._forward_request("GET", f"{self.component_registry_url}/templates", request)

        # Code generation routes - forward to code generator
        @self.app.post("/generate/code")
        async def generate_code(request: Request):
            """Generate code from visual project"""
            return await self._forward_request("POST", f"{self.code_generator_url}/generate", request)

        @self.app.get("/download/{code_id}")
        async def download_code(code_id: str, request: Request):
            """Download generated code as a file"""
            return await self._forward_request("GET", f"{self.code_generator_url}/code/{code_id}", request)

        # Workflow execution routes - forward to workflow engine
        @self.app.post("/workflows/execute")
        async def execute_workflow(request: Request):
            """Execute a visual workflow"""
            return await self._forward_request("POST", f"{self.workflow_engine_url}/workflows/execute", request)

        # Debugging routes - forward to debugger
        @self.app.post("/debug")
        async def debug_project(request: Request):
            """Start debugging session"""
            return await self._forward_request("POST", f"{self.debugger_url}/sessions", request)

        @self.app.post("/debug/step")
        async def step_debug_session(request: Request):
            """Step through debug session"""
            return await self._forward_request("POST", f"{self.debugger_url}/sessions/step", request)

        @self.app.get("/debug/{session_id}")
        async def get_debug_session(session_id: str, request: Request):
            """Get debug session details"""
            return await self._forward_request("GET", f"{self.debugger_url}/sessions/{session_id}", request)

        # Deployment routes - forward to deployment service
        @self.app.post("/deploy")
        async def deploy_project(request: Request):
            """Deploy visual project"""
            return await self._forward_request("POST", f"{self.deployment_url}/deploy", request)

        @self.app.get("/deployments/{deployment_id}")
        async def get_deployment_status(deployment_id: str, request: Request):
            """Get deployment status"""
            return await self._forward_request("GET", f"{self.deployment_url}/deployments/{deployment_id}", request)

        # AI assistance routes - forward to AI assistant
        @self.app.post("/ai/suggest")
        async def get_ai_suggestion(request: Request):
            """Get AI-powered suggestion"""
            return await self._forward_request("POST", f"{self.ai_assistant_url}/suggest", request)

        @self.app.post("/ai/optimize")
        async def optimize_code(request: Request):
            """Optimize code using AI"""
            return await self._forward_request("POST", f"{self.ai_assistant_url}/optimize", request)

        # WebSocket for real-time collaboration
        @self.app.websocket("/ws/{project_id}")
        async def websocket_endpoint(websocket: WebSocket, project_id: str):
            """WebSocket for real-time collaboration - proxy to workflow engine"""
            await websocket.accept()

            # Add to connections
            if project_id not in self.websocket_connections:
                self.websocket_connections[project_id] = []
            self.websocket_connections[project_id].append(websocket)

            try:
                # Connect to workflow engine WebSocket
                async with self.http_client.stream("GET", f"{self.workflow_engine_url}/ws/{project_id}") as response:
                    if response.status_code != 200:
                        await websocket.send_json({"error": "Failed to connect to workflow engine"})
                        return

                    # Proxy messages between client and workflow engine
                    while True:
                        # Receive from client
                        client_data = await websocket.receive_json()

                        # Forward to workflow engine (simplified - in production use proper WS proxy)
                        # For now, just broadcast to other connected clients
                        await self._broadcast_update(project_id, client_data, exclude=websocket)

            except WebSocketDisconnect:
                # Remove from connections
                self.websocket_connections[project_id].remove(websocket)
                if not self.websocket_connections[project_id]:
                    del self.websocket_connections[project_id]

    async def _forward_request(self, method: str, url: str, request: Request) -> dict:
        """Forward HTTP request to microservice"""
        try:
            # Get request body
            body = await request.body()

            # Forward request
            async with self.http_client as client:
                response = await client.request(
                    method=method,
                    url=url,
                    content=body,
                    headers=dict(request.headers)
                )

                if response.status_code >= 400:
                    # Return error response
                    return {
                        "error": f"Service error: {response.status_code}",
                        "details": response.text
                    }

                return response.json()

        except httpx.RequestError as e:
            logger.error(f"Request forwarding failed: {e}")
            return {"error": "Service unavailable", "details": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in request forwarding: {e}")
            return {"error": "Internal gateway error", "details": str(e)}

    async def _aggregate_health_checks(self) -> dict:
        """Aggregate health checks from all microservices"""
        services = {
            "workflow_engine": self.workflow_engine_url,
            "component_registry": self.component_registry_url,
            "code_generator": self.code_generator_url,
            "debugger": self.debugger_url,
            "deployment": self.deployment_url,
            "ai_assistant": self.ai_assistant_url
        }

        health_results = {}
        overall_status = "healthy"

        async with self.http_client as client:
            for service_name, service_url in services.items():
                try:
                    response = await client.get(f"{service_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        health_results[service_name] = response.json()
                        if health_results[service_name].get("status") != "healthy":
                            overall_status = "degraded"
                    else:
                        health_results[service_name] = {"status": "unhealthy", "error": response.status_code}
                        overall_status = "unhealthy"
                except Exception as e:
                    health_results[service_name] = {"status": "unreachable", "error": str(e)}
                    overall_status = "unhealthy"

        return {
            "overall": overall_status,
            "services": health_results
        }

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

    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""

        def on_service_event(event):
            """Handle events from microservices"""
            logger.info(f"Received event: {event.type} from {event.data.get('service', 'unknown')}")

        # Register handlers for various events
        self.dal.event_bus.on(EventChannel.SYSTEM_METRICS, on_service_event)
        self.dal.event_bus.on(EventChannel.WORKFLOW_EXECUTED, on_service_event)
        self.dal.event_bus.on(EventChannel.CODE_GENERATED, on_service_event)

        logger.info("Event handlers registered for API gateway")

    async def startup(self):
        """Startup tasks"""
        logger.info("Visual Builder API Gateway starting up...")

        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")

        # Check microservice availability
        try:
            health_status = await self._aggregate_health_checks()
            if health_status["overall"] != "healthy":
                logger.warning(f"Some microservices are not healthy: {health_status}")
        except Exception as e:
            logger.warning(f"Could not check microservice health: {e}")

        logger.info("Visual Builder API Gateway ready")

    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual Builder API Gateway shutting down...")

        # Close HTTP client
        await self.http_client.aclose()

        # Close all WebSocket connections
        for project_id, connections in self.websocket_connections.items():
            for ws in connections:
                await ws.close()

        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn

    service = VisualBuilderAPIGateway()

    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)

    # Run service
    port = int(os.getenv("VISUAL_BUILDER_PORT", 8006))
    logger.info(f"Starting Visual Code Builder API Gateway on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()