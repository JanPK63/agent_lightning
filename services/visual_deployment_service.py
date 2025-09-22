#!/usr/bin/env python3
"""
Visual Deployment Microservice
Handles deployment configuration and execution for visual projects
Based on Agent Lightning microservices architecture
"""

import os
import sys
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
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


class DeployRequest(BaseModel):
    """Deploy visual project"""
    project_id: str = Field(description="Project ID")
    environment: str = Field(default="development", description="Target environment")
    auto_scale: bool = Field(default=False, description="Enable auto-scaling")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Deployment configuration overrides")


class DeploymentStatus(BaseModel):
    """Deployment status"""
    deployment_id: str = Field(description="Deployment ID")
    project_id: str = Field(description="Project ID")
    status: str = Field(description="Deployment status")
    environment: str = Field(description="Target environment")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Last update timestamp")
    logs: Optional[List[str]] = Field(default=None, description="Deployment logs")


class DeploymentGenerator:
    """Enterprise Deployment Configuration Generator"""

    def __init__(self):
        self.environments = ["development", "staging", "production"]
        self.providers = ["kubernetes", "docker", "aws", "azure", "gcp"]

    def generate_deployment_config(self, project: dict, environment: str, auto_scale: bool, overrides: Optional[Dict[str, Any]] = None) -> dict:
        """Generate deployment configuration"""
        base_config = {
            "environment": environment,
            "auto_scale": auto_scale,
            "resources": {
                "cpu": "2 cores" if environment == "production" else "1 core",
                "memory": "4Gi" if environment == "production" else "2Gi",
                "replicas": 3 if environment == "production" and auto_scale else 1
            },
            "deployment_type": "kubernetes" if environment == "production" else "docker",
            "health_checks": {
                "liveness": "/health",
                "readiness": "/ready"
            },
            "networking": {
                "ports": [8080],
                "service_type": "LoadBalancer" if environment == "production" else "ClusterIP"
            }
        }

        # Apply overrides
        if overrides:
            self._deep_update(base_config, overrides)

        return base_config

    def _deep_update(self, base: dict, updates: dict):
        """Deep update dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


class DeploymentBlockFactory:
    """Factory for creating deployment blocks"""

    def create_block(self, block_type: str, config: dict = None) -> dict:
        """Create deployment block"""
        config = config or {}

        blocks = {
            "container": {
                "type": "container",
                "image": config.get("image", "agent-lightning:latest"),
                "ports": config.get("ports", [8080]),
                "env": config.get("env", {}),
                "resources": config.get("resources", {})
            },
            "service": {
                "type": "service",
                "selector": config.get("selector", "app=agent"),
                "ports": config.get("service_ports", [{"port": 80, "targetPort": 8080}]),
                "type": config.get("service_type", "ClusterIP")
            },
            "ingress": {
                "type": "ingress",
                "rules": config.get("rules", []),
                "tls": config.get("tls", True),
                "annotations": config.get("annotations", {})
            },
            "configmap": {
                "type": "configmap",
                "data": config.get("data", {})
            },
            "secret": {
                "type": "secret",
                "data": config.get("secret_data", {})
            }
        }
        return blocks.get(block_type, {})


class VisualDeploymentService:
    """Visual Deployment Microservice"""

    def __init__(self):
        self.app = FastAPI(title="Visual Deployment Service", version="1.0.0")

        # Initialize components
        self.dal = DataAccessLayer("visual_deployment")
        self.cache = get_cache()
        self.deployment_gen = DeploymentGenerator()
        self.deployment_factory = DeploymentBlockFactory()

        # HTTP client for inter-service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Service URLs
        self.workflow_engine_url = os.getenv("VISUAL_WORKFLOW_ENGINE_URL", "http://localhost:8007")
        self.code_generator_url = os.getenv("VISUAL_CODE_GENERATOR_URL", "http://localhost:8009")

        logger.info("✅ Connected to shared database and initialized deployment service")

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
                "service": "visual_deployment",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "supported_environments": self.deployment_gen.environments,
                "supported_providers": self.deployment_gen.providers,
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.post("/deploy")
        async def deploy_project(request: DeployRequest):
            """Deploy visual project"""
            try:
                # Get project data from workflow engine
                async with self.http_client as client:
                    response = await client.get(f"{self.workflow_engine_url}/projects/{request.project_id}")
                    if response.status_code != 200:
                        raise HTTPException(status_code=404, detail="Project not found in workflow engine")
                    project_data = response.json()

                # Generate deployment configuration
                deployment_config = self.deployment_gen.generate_deployment_config(
                    project_data,
                    environment=request.environment,
                    auto_scale=request.auto_scale,
                    overrides=request.config_overrides
                )

                # Create deployment task
                deployment_task = {
                    "deployment_id": str(uuid.uuid4()),
                    "project_id": request.project_id,
                    "environment": request.environment,
                    "config": deployment_config,
                    "status": "pending",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "logs": []
                }

                # Store deployment
                self.cache.set(f"deployment:{deployment_task['deployment_id']}", deployment_task, ttl=3600)

                # Start deployment process (async)
                # In production, this would trigger actual deployment
                deployment_task["status"] = "in_progress"
                deployment_task["logs"].append(f"Starting deployment to {request.environment}")
                self.cache.set(f"deployment:{deployment_task['deployment_id']}", deployment_task, ttl=3600)

                # Emit event
                self.dal.event_bus.emit(EventChannel.WORKFLOW_STARTED, {
                    "type": "project_deployment",
                    "deployment_id": deployment_task["deployment_id"],
                    "project_id": request.project_id,
                    "environment": request.environment
                })

                # Simulate deployment completion (in production, this would be async)
                await self._simulate_deployment(deployment_task["deployment_id"])

                return {
                    "deployment_id": deployment_task["deployment_id"],
                    "status": "initiated",
                    "environment": request.environment,
                    "config": deployment_config
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to deploy project: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/deployments/{deployment_id}")
        async def get_deployment_status(deployment_id: str):
            """Get deployment status"""
            try:
                deployment = self.cache.get(f"deployment:{deployment_id}")
                if not deployment:
                    raise HTTPException(status_code=404, detail="Deployment not found")

                return DeploymentStatus(**deployment)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get deployment status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/deployments")
        async def list_deployments(project_id: Optional[str] = None):
            """List deployments"""
            try:
                deployments = []
                pattern = f"deployment:*{project_id}*" if project_id else "deployment:*"

                for key in self.cache.redis_client.keys(pattern):
                    deployment = self.cache.get(key)
                    if deployment:
                        deployments.append({
                            "deployment_id": deployment["deployment_id"],
                            "project_id": deployment["project_id"],
                            "status": deployment["status"],
                            "environment": deployment["environment"],
                            "created_at": deployment["created_at"]
                        })

                return {
                    "deployments": deployments,
                    "count": len(deployments)
                }

            except Exception as e:
                logger.error(f"Failed to list deployments: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/deployments/{deployment_id}")
        async def cancel_deployment(deployment_id: str):
            """Cancel deployment"""
            try:
                deployment = self.cache.get(f"deployment:{deployment_id}")
                if not deployment:
                    raise HTTPException(status_code=404, detail="Deployment not found")

                if deployment["status"] in ["pending", "in_progress"]:
                    deployment["status"] = "cancelled"
                    deployment["updated_at"] = datetime.utcnow().isoformat()
                    deployment["logs"].append("Deployment cancelled by user")
                    self.cache.set(f"deployment:{deployment_id}", deployment, ttl=3600)

                    return {"status": "cancelled", "deployment_id": deployment_id}
                else:
                    raise HTTPException(status_code=400, detail="Cannot cancel completed deployment")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to cancel deployment: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/config/templates")
        async def get_deployment_templates():
            """Get deployment configuration templates"""
            try:
                templates = {
                    "development": {
                        "environment": "development",
                        "auto_scale": False,
                        "resources": {"cpu": "1 core", "memory": "2Gi", "replicas": 1},
                        "deployment_type": "docker"
                    },
                    "staging": {
                        "environment": "staging",
                        "auto_scale": False,
                        "resources": {"cpu": "1 core", "memory": "2Gi", "replicas": 2},
                        "deployment_type": "kubernetes"
                    },
                    "production": {
                        "environment": "production",
                        "auto_scale": True,
                        "resources": {"cpu": "2 cores", "memory": "4Gi", "replicas": 3},
                        "deployment_type": "kubernetes"
                    }
                }

                return {"templates": templates}

            except Exception as e:
                logger.error(f"Failed to get deployment templates: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _simulate_deployment(self, deployment_id: str):
        """Simulate deployment process"""
        # In production, this would trigger actual cloud deployment
        deployment = self.cache.get(f"deployment:{deployment_id}")
        if deployment:
            import asyncio
            await asyncio.sleep(2)  # Simulate deployment time

            deployment["status"] = "completed"
            deployment["updated_at"] = datetime.utcnow().isoformat()
            deployment["logs"].extend([
                "Building container image...",
                "Pushing to registry...",
                "Creating Kubernetes resources...",
                "Deployment completed successfully"
            ])
            self.cache.set(f"deployment:{deployment_id}", deployment, ttl=3600)

            # Emit completion event
            self.dal.event_bus.emit(EventChannel.WORKFLOW_COMPLETED, {
                "type": "project_deployment",
                "deployment_id": deployment_id,
                "project_id": deployment["project_id"]
            })

    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""

        def on_code_generated(event):
            """Handle code generation - could trigger deployment"""
            project_id = event.data.get('project_id')
            logger.info(f"Code generated for project {project_id} - deployment service notified")

        # Register handlers
        self.dal.event_bus.on(EventChannel.CODE_GENERATED, on_code_generated)

        logger.info("Event handlers registered for deployment service")

    async def startup(self):
        """Startup tasks"""
        logger.info("Visual Deployment Service starting up...")

        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")

        logger.info("Visual Deployment Service ready")

    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual Deployment Service shutting down...")

        # Close HTTP client
        await self.http_client.aclose()

        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn

    service = VisualDeploymentService()

    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)

    # Run service
    port = int(os.getenv("VISUAL_DEPLOYMENT_PORT", 8011))
    logger.info(f"Starting Visual Deployment Service on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()