#!/usr/bin/env python3
"""
Visual Component Registry Microservice
Manages visual components and project templates
Based on Agent Lightning microservices architecture
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared components
from shared.data_access import DataAccessLayer
from shared.events import EventChannel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentDefinition(BaseModel):
    """Component definition"""
    id: str = Field(description="Component ID")
    name: str = Field(description="Component name")
    category: str = Field(description="Component category")
    ports: Dict[str, Any] = Field(description="Input/output ports")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    default_config: Optional[Dict[str, Any]] = Field(default=None, description="Default configuration")


class TemplateDefinition(BaseModel):
    """Template definition"""
    id: str = Field(description="Template ID")
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    categories: List[str] = Field(description="Template categories")
    components: List[Dict[str, Any]] = Field(description="Template components")
    connections: List[Dict[str, Any]] = Field(description="Template connections")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ComponentLibrary:
    """Component library for visual builder"""

    def __init__(self):
        self.components = {
            "logic": {
                "condition": {
                    "name": "Condition",
                    "ports": {"in": 1, "out": 2},
                    "metadata": {"description": "Conditional logic component"}
                },
                "loop": {
                    "name": "Loop",
                    "ports": {"in": 1, "out": 1},
                    "metadata": {"description": "Loop iteration component"}
                },
                "function": {
                    "name": "Function",
                    "ports": {"in": 1, "out": 1},
                    "metadata": {"description": "Custom function component"}
                }
            },
            "data": {
                "input": {
                    "name": "Input",
                    "ports": {"in": 0, "out": 1},
                    "metadata": {"description": "Data input component"}
                },
                "output": {
                    "name": "Output",
                    "ports": {"in": 1, "out": 0},
                    "metadata": {"description": "Data output component"}
                },
                "transform": {
                    "name": "Transform",
                    "ports": {"in": 1, "out": 1},
                    "metadata": {"description": "Data transformation component"}
                }
            },
            "ai": {
                "llm": {
                    "name": "LLM",
                    "ports": {"in": 1, "out": 1},
                    "metadata": {"description": "Large Language Model component"}
                },
                "classifier": {
                    "name": "Classifier",
                    "ports": {"in": 1, "out": 1},
                    "metadata": {"description": "Classification component"}
                },
                "embedder": {
                    "name": "Embedder",
                    "ports": {"in": 1, "out": 1},
                    "metadata": {"description": "Text embedding component"}
                }
            }
        }

    def get_component(self, category: str, component_id: str) -> dict:
        """Get component by category and ID"""
        return self.components.get(category, {}).get(component_id, {})

    def get_category(self, category: str) -> dict:
        """Get all components in category"""
        return self.components.get(category, {})

    def get_component_count(self) -> int:
        """Get total component count"""
        return sum(len(cat) for cat in self.components.values())

    def add_component(self, category: str, component_id: str, component_data: dict) -> bool:
        """Add a new component"""
        if category not in self.components:
            self.components[category] = {}
        self.components[category][component_id] = component_data
        return True

    def update_component(self, category: str, component_id: str, component_data: dict) -> bool:
        """Update an existing component"""
        if category in self.components and component_id in self.components[category]:
            self.components[category][component_id].update(component_data)
            return True
        return False

    def delete_component(self, category: str, component_id: str) -> bool:
        """Delete a component"""
        if category in self.components and component_id in self.components[category]:
            del self.components[category][component_id]
            return True
        return False

    def initialize_default_components(self):
        """Initialize default components"""
        logger.info(f"Initialized {self.get_component_count()} default components")


class TemplateLibrary:
    """Project template library"""

    def __init__(self):
        self.templates = {
            "basic_agent": {
                "name": "Basic Agent",
                "description": "Simple agent with input, processing, and output",
                "categories": ["basic", "agent"],
                "components": [
                    {"id": "input", "type": "data", "config": {"source": "user"}},
                    {"id": "processor", "type": "logic", "config": {"algorithm": "basic"}},
                    {"id": "output", "type": "data", "config": {"destination": "response"}}
                ],
                "connections": [
                    {"source": "input", "target": "processor"},
                    {"source": "processor", "target": "output"}
                ],
                "metadata": {"difficulty": "beginner", "use_case": "simple_automation"}
            },
            "ml_pipeline": {
                "name": "ML Pipeline",
                "description": "Machine learning pipeline with data processing and model training",
                "categories": ["ml", "data", "ai"],
                "components": [
                    {"id": "data_loader", "type": "data", "config": {"source": "dataset"}},
                    {"id": "preprocessor", "type": "logic", "config": {"steps": ["normalize", "encode"]}},
                    {"id": "model", "type": "ai", "config": {"type": "neural_network"}},
                    {"id": "evaluator", "type": "logic", "config": {"metrics": ["accuracy", "f1"]}}
                ],
                "connections": [
                    {"source": "data_loader", "target": "preprocessor"},
                    {"source": "preprocessor", "target": "model"},
                    {"source": "model", "target": "evaluator"}
                ],
                "metadata": {"difficulty": "intermediate", "use_case": "machine_learning"}
            }
        }

    def get_template(self, template_id: str) -> dict:
        """Get template by ID"""
        return self.templates.get(template_id, {})

    def list_templates(self) -> list:
        """List all templates"""
        return list(self.templates.values())

    def get_categories(self) -> list:
        """Get template categories"""
        return ["basic", "ml", "api", "workflow", "integration"]

    def add_template(self, template_id: str, template_data: dict) -> bool:
        """Add a new template"""
        self.templates[template_id] = template_data
        return True

    def update_template(self, template_id: str, template_data: dict) -> bool:
        """Update an existing template"""
        if template_id in self.templates:
            self.templates[template_id].update(template_data)
            return True
        return False

    def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        if template_id in self.templates:
            del self.templates[template_id]
            return True
        return False

    def load_default_templates(self):
        """Load default templates"""
        logger.info(f"Loaded {len(self.templates)} default templates")


class VisualComponentRegistryService:
    """Visual Component Registry Microservice"""

    def __init__(self):
        self.app = FastAPI(title="Visual Component Registry Service", version="1.0.0")

        # Initialize components
        self.dal = DataAccessLayer("visual_component_registry")
        self.components = ComponentLibrary()
        self.templates = TemplateLibrary()

        logger.info("✅ Connected to shared database and initialized component registry")

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
                "service": "visual_component_registry",
                "status": "healthy" if health_status['database'] else "degraded",
                "database": health_status['database'],
                "components_count": self.components.get_component_count(),
                "templates_count": len(self.templates.templates),
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.get("/components")
        async def get_component_library():
            """Get available components"""
            try:
                return {
                    "categories": {
                        "logic": self.components.get_category("logic"),
                        "data": self.components.get_category("data"),
                        "ui": {},  # UI components would be added here
                        "ai": self.components.get_category("ai"),
                        "integration": {},  # Integration components
                        "workflow": {}  # Workflow components
                    },
                    "total": self.components.get_component_count()
                }

            except Exception as e:
                logger.error(f"Failed to get component library: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/components/{category}")
        async def get_components_by_category(category: str):
            """Get components in a specific category"""
            try:
                components = self.components.get_category(category)
                if not components:
                    raise HTTPException(status_code=404, detail=f"Category '{category}' not found")

                return {
                    "category": category,
                    "components": components,
                    "count": len(components)
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get components for category {category}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/components/{category}/{component_id}")
        async def get_component(category: str, component_id: str):
            """Get specific component details"""
            try:
                component = self.components.get_component(category, component_id)
                if not component:
                    raise HTTPException(status_code=404, detail=f"Component '{component_id}' not found in category '{category}'")

                return {
                    "category": category,
                    "component_id": component_id,
                    "component": component
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get component {category}/{component_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/components/{category}")
        async def add_component(category: str, component: ComponentDefinition):
            """Add a new component (admin)"""
            try:
                # Validate component data
                if component.category != category:
                    raise HTTPException(status_code=400, detail="Component category mismatch")

                success = self.components.add_component(category, component.id, component.dict())
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to add component")

                # Emit event
                self.dal.event_bus.emit(EventChannel.SYSTEM_METRICS, {
                    "service": "visual_component_registry",
                    "metric": "component_added",
                    "category": category,
                    "component_id": component.id
                })

                return {"status": "added", "component_id": component.id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to add component: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/components/{category}/{component_id}")
        async def update_component(category: str, component_id: str, component: ComponentDefinition):
            """Update an existing component (admin)"""
            try:
                if component.id != component_id or component.category != category:
                    raise HTTPException(status_code=400, detail="Component ID or category mismatch")

                success = self.components.update_component(category, component_id, component.dict())
                if not success:
                    raise HTTPException(status_code=404, detail="Component not found")

                return {"status": "updated", "component_id": component_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update component: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/components/{category}/{component_id}")
        async def delete_component(category: str, component_id: str):
            """Delete a component (admin)"""
            try:
                success = self.components.delete_component(category, component_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Component not found")

                return {"status": "deleted", "component_id": component_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete component: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/templates")
        async def get_templates():
            """Get available project templates"""
            try:
                return {
                    "templates": self.templates.list_templates(),
                    "categories": self.templates.get_categories(),
                    "count": len(self.templates.templates)
                }

            except Exception as e:
                logger.error(f"Failed to get templates: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/templates/{template_id}")
        async def get_template(template_id: str):
            """Get specific template details"""
            try:
                template = self.templates.get_template(template_id)
                if not template:
                    raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

                return {
                    "template_id": template_id,
                    "template": template
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get template {template_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/templates")
        async def add_template(template: TemplateDefinition):
            """Add a new template (admin)"""
            try:
                success = self.templates.add_template(template.id, template.dict())
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to add template")

                # Emit event
                self.dal.event_bus.emit(EventChannel.SYSTEM_METRICS, {
                    "service": "visual_component_registry",
                    "metric": "template_added",
                    "template_id": template.id
                })

                return {"status": "added", "template_id": template.id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to add template: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/templates/{template_id}")
        async def update_template(template_id: str, template: TemplateDefinition):
            """Update an existing template (admin)"""
            try:
                if template.id != template_id:
                    raise HTTPException(status_code=400, detail="Template ID mismatch")

                success = self.templates.update_template(template_id, template.dict())
                if not success:
                    raise HTTPException(status_code=404, detail="Template not found")

                return {"status": "updated", "template_id": template_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update template: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/templates/{template_id}")
        async def delete_template(template_id: str):
            """Delete a template (admin)"""
            try:
                success = self.templates.delete_template(template_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Template not found")

                return {"status": "deleted", "template_id": template_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to delete template: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""

        def on_component_usage(event):
            """Track component usage for analytics"""
            component_id = event.data.get('component_id')
            logger.info(f"Component {component_id} used")

        def on_template_usage(event):
            """Track template usage for analytics"""
            template_id = event.data.get('template_id')
            logger.info(f"Template {template_id} used")

        # Register handlers
        self.dal.event_bus.on(EventChannel.COMPONENT_USED, on_component_usage)
        self.dal.event_bus.on(EventChannel.TEMPLATE_USED, on_template_usage)

        logger.info("Event handlers registered for component registry service")

    async def startup(self):
        """Startup tasks"""
        logger.info("Visual Component Registry Service starting up...")

        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")

        # Initialize component libraries
        self.components.initialize_default_components()
        self.templates.load_default_templates()

        logger.info("Visual Component Registry Service ready")

    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual Component Registry Service shutting down...")

        # Cleanup database connections
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn

    service = VisualComponentRegistryService()

    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)

    # Run service
    port = int(os.getenv("VISUAL_COMPONENT_REGISTRY_PORT", 8008))
    logger.info(f"Starting Visual Component Registry Service on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()