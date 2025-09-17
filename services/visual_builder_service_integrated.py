#!/usr/bin/env python3
"""
Visual Code Builder Microservice - Integrated with Shared Database
Provides visual programming interface for agent development
Integrates all visual components into a unified service
Based on SA-008: Complete System Integration
"""

import os
import sys
import json
import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import networkx as nx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared data access layer
from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Visual components will be integrated through internal classes
# This provides better encapsulation and enterprise-grade structure

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


class ComponentType(str, Enum):
    """Types of visual components"""
    LOGIC = "logic"
    DATA = "data"
    UI = "ui"
    AI = "ai"
    INTEGRATION = "integration"
    WORKFLOW = "workflow"


# Pydantic models
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


class CodeGenerateRequest(BaseModel):
    """Request code generation"""
    project_id: str = Field(description="Project ID")
    language: str = Field(default="python", description="Target language")
    optimize: bool = Field(default=True, description="Optimize generated code")


class DeployRequest(BaseModel):
    """Deploy visual project"""
    project_id: str = Field(description="Project ID")
    environment: str = Field(default="development", description="Target environment")
    auto_scale: bool = Field(default=False, description="Enable auto-scaling")


class DebugRequest(BaseModel):
    """Debug visual project"""
    project_id: str = Field(description="Project ID")
    breakpoints: Optional[List[str]] = Field(default=None, description="Component IDs for breakpoints")
    watch_variables: Optional[List[str]] = Field(default=None, description="Variables to watch")


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


class DeploymentGenerator:
    """Enterprise Deployment Configuration Generator"""
    
    def __init__(self):
        self.environments = ["development", "staging", "production"]
        self.providers = ["kubernetes", "docker", "aws", "azure", "gcp"]
        
    def generate_deployment_config(self, project: dict, environment: str, auto_scale: bool) -> dict:
        """Generate deployment configuration"""
        return {
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
            }
        }


class DeploymentBlockFactory:
    """Factory for creating deployment blocks"""
    
    def create_block(self, block_type: str) -> dict:
        """Create deployment block"""
        blocks = {
            "container": {
                "type": "container",
                "image": "agent-lightning:latest",
                "ports": [8080]
            },
            "service": {
                "type": "service",
                "selector": "app=agent",
                "ports": [{"port": 80, "targetPort": 8080}]
            },
            "ingress": {
                "type": "ingress",
                "rules": [],
                "tls": True
            }
        }
        return blocks.get(block_type, {})


class AIAssistant:
    """AI-powered coding assistant"""
    
    def __init__(self):
        self.suggestions_cache = {}
        
    def get_suggestion(self, project: dict, query: str) -> dict:
        """Get AI suggestion for project"""
        # In production, this would call Claude or another AI service
        return {
            "suggestion": f"Based on your query '{query}', I recommend adding error handling and validation.",
            "code_snippets": [
                "try:\n    # Your code here\nexcept Exception as e:\n    logger.error(f'Error: {e}')"
            ],
            "best_practices": [
                "Always validate inputs",
                "Use proper error handling",
                "Log important events"
            ]
        }


class VisualDebugger:
    """Visual debugging component"""
    
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, project: dict, breakpoints: list = None, watch_variables: list = None) -> dict:
        """Create debug session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "project": project,
            "breakpoints": breakpoints or [],
            "watch_variables": watch_variables or [],
            "state": "initialized"
        }
        return self.sessions[session_id]
        
    def step_through(self, session_id: str) -> dict:
        """Step through code"""
        if session_id in self.sessions:
            return {
                "current_line": 1,
                "variables": {},
                "call_stack": []
            }
        return {}


class TemplateLibrary:
    """Project template library"""
    
    def __init__(self):
        self.templates = {
            "basic_agent": {
                "name": "Basic Agent",
                "components": [
                    {"id": "input", "type": "data", "config": {"source": "user"}},
                    {"id": "processor", "type": "logic", "config": {"algorithm": "basic"}},
                    {"id": "output", "type": "data", "config": {"destination": "response"}}
                ],
                "connections": [
                    {"source": "input", "target": "processor"},
                    {"source": "processor", "target": "output"}
                ]
            },
            "ml_pipeline": {
                "name": "ML Pipeline",
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
                ]
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
    
    def load_default_templates(self):
        """Load default templates"""
        logger.info(f"Loaded {len(self.templates)} default templates")


class ComponentLibrary:
    """Component library for visual builder"""
    
    def __init__(self):
        self.components = {
            "logic": {
                "condition": {"name": "Condition", "ports": {"in": 1, "out": 2}},
                "loop": {"name": "Loop", "ports": {"in": 1, "out": 1}},
                "function": {"name": "Function", "ports": {"in": 1, "out": 1}}
            },
            "data": {
                "input": {"name": "Input", "ports": {"in": 0, "out": 1}},
                "output": {"name": "Output", "ports": {"in": 1, "out": 0}},
                "transform": {"name": "Transform", "ports": {"in": 1, "out": 1}}
            },
            "ai": {
                "llm": {"name": "LLM", "ports": {"in": 1, "out": 1}},
                "classifier": {"name": "Classifier", "ports": {"in": 1, "out": 1}},
                "embedder": {"name": "Embedder", "ports": {"in": 1, "out": 1}}
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
        
    def initialize_default_components(self):
        """Initialize default components"""
        logger.info(f"Initialized {self.get_component_count()} default components")


class CodeTranslator:
    """Translate visual projects to code"""
    
    def __init__(self):
        self.supported_languages = ["python", "javascript", "java", "go"]
        
    def translate_project(self, project: dict, language: str = "python", optimize: bool = True) -> str:
        """Translate visual project to code"""
        if language not in self.supported_languages:
            language = "python"

        if language == "python":
            return self._generate_fastapi_scaffold(project, optimize)
        else:
            return self._generate_basic_python(project, optimize)

    def _generate_fastapi_scaffold(self, project: dict, optimize: bool = True) -> str:
        """Generate a FastAPI scaffold from the visual project"""
        project_name = project.get('name', 'GeneratedAPI').replace(' ', '')
        components = project.get("components", {})

        code_lines = [
            "#!/usr/bin/env python3",
            f"\"\"\"{project_name} - Generated FastAPI Application from Visual Code Builder\"\"\"",
            "",
            "from fastapi import FastAPI, HTTPException",
            "from pydantic import BaseModel",
            "from typing import Any, Dict, List, Optional",
            "import uvicorn",
            "import logging",
            "",
            "logger = logging.getLogger(__name__)",
            "",
            "# Pydantic models for API",
        ]

        # Generate Pydantic models based on components
        for comp_id, comp_data in components.items():
            comp_type = comp_data.get('type', 'generic')
            if comp_type == 'data':
                # Clean component ID for model name (replace hyphens/underscores with spaces, then title case, then remove spaces)
                clean_comp_id = comp_id.replace('_', ' ').replace('-', ' ').title().replace(' ', '')
                model_name = f"{clean_comp_id}Model"
                code_lines.extend([
                    f"class {model_name}(BaseModel):",
                    f"    \"\"\"Model for {comp_id} component\"\"\"",
                    "    id: Optional[str] = None",
                    "    name: str",
                    "    data: Dict[str, Any] = {}",
                    ""
                ])

        # Create FastAPI app
        code_lines.extend([
            "app = FastAPI(",
            f"    title=\"{project_name} API\",",
            "    description=\"Generated from Visual Code Builder\",",
            "    version=\"1.0.0\"",
            ")",
            "",
            "# API Routes",
            "@app.get(\"/\")",
            "async def root():",
            f"    return {{\"message\": \"{project_name} API is running\", \"components\": {len(components)}}}",
            "",
            "@app.get(\"/health\")",
            "async def health():",
            "    return {\"status\": \"healthy\", \"service\": \"generated_api\"}",
            ""
        ])

        # Generate routes based on components
        for comp_id, comp_data in components.items():
            comp_type = comp_data.get('type', 'generic')
            route_name = comp_id.replace('_', '-')

            if comp_type == 'data':
                # Use the same cleaned model name as in model generation
                clean_comp_id = comp_id.replace('_', ' ').replace('-', ' ').title().replace(' ', '')
                model_name = f"{clean_comp_id}Model"
                code_lines.extend([
                    f"@app.get(\"/{route_name}\")",
                    "async def get_data():",
                    f"    \"\"\"Get data from {comp_id} component\"\"\"",
                    f"    return {{\"component\": \"{comp_id}\", \"type\": \"{comp_type}\", \"data\": {{}}}}",
                    "",
                    f"@app.post(\"/{route_name}\")",
                    f"async def create_data(item: {model_name}):",
                    f"    \"\"\"Create data for {comp_id} component\"\"\"",
                    f"    return {{\"status\": \"created\", \"component\": \"{comp_id}\", \"item\": item.dict()}}",
                    ""
                ])
            elif comp_type == 'logic':
                code_lines.extend([
                    f"@app.post(\"/{route_name}/execute\")",
                    "async def execute_logic(data: Dict[str, Any]):",
                    f"    \"\"\"Execute logic from {comp_id} component\"\"\"",
                    f"    return {{\"status\": \"executed\", \"component\": \"{comp_id}\", \"result\": data}}",
                    ""
                ])
            elif comp_type == 'ai':
                code_lines.extend([
                    f"@app.post(\"/{route_name}/predict\")",
                    "async def ai_predict(input_data: Dict[str, Any]):",
                    f"    \"\"\"AI prediction from {comp_id} component\"\"\"",
                    f"    return {{\"status\": \"predicted\", \"component\": \"{comp_id}\", \"prediction\": \"mock_result\"}}",
                    ""
                ])

        # Add main execution block
        code_lines.extend([
            "if __name__ == \"__main__\":",
            "    uvicorn.run(",
            "        \"main:app\",",
            "        host=\"0.0.0.0\",",
            "        port=8000,",
            "        reload=True",
            "    )"
        ])

        return "\n".join(code_lines)

    def _generate_basic_python(self, project: dict, optimize: bool = True) -> str:
        """Generate basic Python code (fallback)"""
        code_lines = [
            "#!/usr/bin/env python3",
            "\"\"\"Generated from Visual Code Builder\"\"\"",
            "",
            "import asyncio",
            "import logging",
            "from typing import Any, Dict, List",
            "",
            "logger = logging.getLogger(__name__)",
            "",
            ""
        ]

        project_name = project.get('name', 'GeneratedAgent').replace(' ', '')

        # Generate main class
        code_lines.extend([
            f"class {project_name}:",
            f"    \"\"\"{'Generated agent from visual builder'}\"\"\"",
            "    ",
            "    def __init__(self):",
            "        self.components = {}",
            "        self.connections = []",
            "        self.setup_components()",
            "    ",
            "    def setup_components(self):",
            "        \"\"\"Initialize components\"\"\"",
        ])

        # Add components
        for comp_id, comp_data in project.get("components", {}).items():
            code_lines.append(f"        # Component: {comp_id}")
            code_lines.append(f"        self.components['{comp_id}'] = {comp_data}")

        code_lines.extend([
            "",
            "    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:",
            "        \"\"\"Execute the workflow\"\"\"",
            "        result = {}",
            "        # Process through components based on connections",
            "        return result",
            "",
            "",
            "if __name__ == '__main__':",
            f"    agent = {project_name}()",
            "    asyncio.run(agent.execute({}))"
        ])

        return "\n".join(code_lines)


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


class VisualBuilderService:
    """Main Visual Code Builder Service - Integrated with shared database"""
    
    def __init__(self):
        self.app = FastAPI(title="Visual Code Builder Service (Integrated)", version="2.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("visual_builder")
        self.cache = get_cache()
        
        # Initialize visual components
        self.code_builder = VisualCodeBuilder()
        self.deployment_gen = DeploymentGenerator()
        self.deployment_factory = DeploymentBlockFactory()
        self.ai_assistant = AIAssistant()
        self.debugger = VisualDebugger()
        self.templates = TemplateLibrary()
        self.components = ComponentLibrary()
        self.translator = CodeTranslator()
        
        # Active projects
        self.active_projects: Dict[str, VisualProject] = {}
        
        # WebSocket connections for real-time collaboration
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        
        logger.info("✅ Connected to shared database and initialized visual components")
        
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
                "service": "visual_builder",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "active_projects": len(self.active_projects),
                "websocket_connections": sum(len(conns) for conns in self.websocket_connections.values()),
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
                    <h1>Visual Code Builder Service</h1>
                    <p class="status">Service is running on port 8006</p>
                    <p>WebSocket endpoint: ws://localhost:8006/ws/{project_id}</p>
                    <p>API documentation: <a href="/docs">/docs</a></p>
                </body>
            </html>
            """)
            
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
                    template = self.templates.get_template(project.template_id)
                    if template:
                        # Apply template components
                        for comp in template.get("components", []):
                            visual_project.add_component(comp["id"], comp)
                        for conn in template.get("connections", []):
                            visual_project.add_connection(
                                conn["source"], 
                                conn["target"],
                                conn
                            )
                            
                # Store in cache and memory
                self.active_projects[project_id] = visual_project
                self.cache.set(f"visual_project:{project_id}", visual_project.to_dict(), ttl=3600)
                
                # If associated with agent, update agent
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
                
                # Get component from library
                component = self.components.get_component(
                    request.component_type,
                    request.component_id
                )
                
                if not component:
                    raise HTTPException(status_code=404, detail="Component not found in library")
                    
                # Add to project
                comp_instance_id = f"{request.component_id}_{uuid.uuid4().hex[:8]}"
                component_data = {
                    "type": request.component_type,
                    "component_id": request.component_id,
                    "position": request.position,
                    "config": request.config or component.get("default_config", {}),
                    "metadata": component.get("metadata", {})
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
                
        @self.app.post("/generate/code")
        async def generate_code(request: CodeGenerateRequest):
            """Generate code from visual project"""
            try:
                project = self._get_active_project(request.project_id)
                
                # Validate project first
                valid, errors = project.validate()
                if not valid:
                    raise HTTPException(status_code=400, detail=f"Project validation failed: {', '.join(errors)}")
                    
                # Generate code using translator
                generated_code = self.translator.translate_project(
                    project.to_dict(),
                    language=request.language,
                    optimize=request.optimize
                )
                
                # Store generated code
                code_id = str(uuid.uuid4())
                self.cache.set(f"generated_code:{code_id}", {
                    "project_id": request.project_id,
                    "language": request.language,
                    "code": generated_code,
                    "timestamp": datetime.utcnow().isoformat()
                }, ttl=3600)
                
                # Emit event
                self.dal.event_bus.emit(EventChannel.SYSTEM_METRICS, {
                    "service": "visual_builder",
                    "metric": "code_generated",
                    "project_id": request.project_id,
                    "code_id": code_id
                })
                
                return {
                    "code_id": code_id,
                    "language": request.language,
                    "code": generated_code,
                    "lines": len(generated_code.split('\n'))
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to generate code: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/download/{code_id}")
        async def download_code(code_id: str):
            """Download generated code as a file"""
            try:
                # Get code from cache
                code_data = self.cache.get(f"generated_code:{code_id}")
                if not code_data:
                    raise HTTPException(status_code=404, detail="Code not found")

                # Create filename
                filename = f"generated_{code_data['language']}_{code_id[:8]}.py"

                # Return as file response
                from fastapi.responses import Response
                return Response(
                    content=code_data["code"],
                    media_type="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to download code: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to generate code: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/deploy")
        async def deploy_project(request: DeployRequest):
            """Deploy visual project"""
            try:
                project = self._get_active_project(request.project_id)
                
                # Generate deployment configuration
                deployment_config = self.deployment_gen.generate_deployment_config(
                    project.to_dict(),
                    environment=request.environment,
                    auto_scale=request.auto_scale
                )
                
                # Create deployment task
                deployment_task = {
                    "project_id": request.project_id,
                    "environment": request.environment,
                    "config": deployment_config,
                    "status": "pending"
                }
                
                # Store deployment
                deployment_id = str(uuid.uuid4())
                self.cache.set(f"deployment:{deployment_id}", deployment_task, ttl=3600)
                
                # Trigger deployment workflow
                self.dal.event_bus.emit(EventChannel.WORKFLOW_STARTED, {
                    "type": "project_deployment",
                    "deployment_id": deployment_id,
                    "project_id": request.project_id
                })
                
                # Update project status
                project.metadata["status"] = ProjectStatus.DEPLOYED.value
                self.cache.set(
                    f"visual_project:{request.project_id}",
                    project.to_dict(),
                    ttl=3600
                )
                
                return {
                    "deployment_id": deployment_id,
                    "status": "initiated",
                    "environment": request.environment
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to deploy project: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/debug")
        async def debug_project(request: DebugRequest):
            """Start debugging session"""
            try:
                project = self._get_active_project(request.project_id)
                
                # Initialize debug session
                debug_session = self.debugger.create_session(
                    project.to_dict(),
                    breakpoints=request.breakpoints,
                    watch_variables=request.watch_variables
                )
                
                # Store session
                session_id = str(uuid.uuid4())
                self.cache.set(f"debug_session:{session_id}", {
                    "project_id": request.project_id,
                    "session": debug_session,
                    "started_at": datetime.utcnow().isoformat()
                }, ttl=1800)
                
                return {
                    "session_id": session_id,
                    "status": "debugging",
                    "breakpoints": request.breakpoints or []
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to start debug session: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/components/library")
        async def get_component_library():
            """Get available components"""
            try:
                return {
                    "categories": {
                        "logic": self.components.get_category("logic"),
                        "data": self.components.get_category("data"),
                        "ui": self.components.get_category("ui"),
                        "ai": self.components.get_category("ai"),
                        "integration": self.components.get_category("integration"),
                        "workflow": self.components.get_category("workflow")
                    },
                    "total": self.components.get_component_count()
                }
                
            except Exception as e:
                logger.error(f"Failed to get component library: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/templates")
        async def get_templates():
            """Get available project templates"""
            try:
                return {
                    "templates": self.templates.list_templates(),
                    "categories": self.templates.get_categories()
                }
                
            except Exception as e:
                logger.error(f"Failed to get templates: {e}")
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
                    elif update_type == "ai_assist":
                        # Handle AI assistance request
                        response = await self._handle_ai_assist(project_id, data)
                        await websocket.send_json(response)
                        
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
            
    async def _handle_ai_assist(self, project_id: str, data: dict) -> dict:
        """Handle AI assistance request"""
        project = self._get_active_project(project_id)
        query = data.get("query", "")
        
        # Get AI suggestion
        suggestion = self.ai_assistant.get_suggestion(
            project.to_dict(),
            query
        )
        
        return {
            "type": "ai_response",
            "suggestion": suggestion,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""
        
        def on_agent_created(event):
            """Handle agent creation - create associated visual project"""
            agent_id = event.data.get('agent_id')
            agent_name = event.data.get('name', 'Unnamed Agent')
            
            # Auto-create visual project for new agent
            project = VisualProject(
                project_id=str(uuid.uuid4()),
                name=f"Visual Project for {agent_name}",
                description=f"Auto-generated visual project for agent {agent_id}"
            )
            
            # Add default components based on agent type
            # This would be more sophisticated in production
            
            self.cache.set(
                f"visual_project:{project.id}",
                project.to_dict(),
                ttl=3600
            )
            
            logger.info(f"Created visual project for agent {agent_id}")
            
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
        
        logger.info("Event handlers registered for visual builder service")
        
    async def startup(self):
        """Startup tasks"""
        logger.info("Visual Code Builder Service (Integrated) starting up...")
        
        # Verify connections
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")
            
        # Initialize component libraries
        self.components.initialize_default_components()
        self.templates.load_default_templates()
        
        logger.info("Visual Code Builder ready")
        
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual Code Builder Service shutting down...")
        
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
    
    service = VisualBuilderService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("VISUAL_BUILDER_PORT", 8006))
    logger.info(f"Starting Visual Code Builder Service (Integrated) on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()