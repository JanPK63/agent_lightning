#!/usr/bin/env python3
"""
Visual Code Generator Microservice
Handles code generation from visual projects
Based on Agent Lightning microservices architecture
"""

import os
import sys
import json
import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared components
from shared.data_access import DataAccessLayer
from shared.events import EventChannel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeGenerateRequest(BaseModel):
    """Request code generation"""
    project_id: str = Field(description="Project ID")
    language: str = Field(default="python", description="Target language")
    optimize: bool = Field(default=True, description="Optimize generated code")
    project_data: Optional[Dict[str, Any]] = Field(default=None, description="Project data (if not fetching from workflow engine)")


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
                # Clean component ID for model name
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


class VisualCodeGeneratorService:
    """Visual Code Generator Microservice"""

    def __init__(self):
        self.app = FastAPI(title="Visual Code Generator Service", version="1.0.0")

        # Initialize components
        self.dal = DataAccessLayer("visual_code_generator")
        self.translator = CodeTranslator()

        # HTTP client for inter-service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Service URLs (would come from service discovery in production)
        self.workflow_engine_url = os.getenv("VISUAL_WORKFLOW_ENGINE_URL", "http://localhost:8007")
        self.component_registry_url = os.getenv("VISUAL_COMPONENT_REGISTRY_URL", "http://localhost:8008")

        logger.info("✅ Connected to shared database and initialized code generator")

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
                "service": "visual_code_generator",
                "status": "healthy" if health_status['database'] else "degraded",
                "database": health_status['database'],
                "supported_languages": self.translator.supported_languages,
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.post("/generate")
        async def generate_code(request: CodeGenerateRequest):
            """Generate code from visual project"""
            try:
                # Get project data from workflow engine if not provided
                project_data = request.project_data
                if not project_data:
                    async with self.http_client as client:
                        response = await client.get(f"{self.workflow_engine_url}/projects/{request.project_id}")
                        if response.status_code != 200:
                            raise HTTPException(status_code=404, detail="Project not found in workflow engine")
                        project_data = response.json()

                # Validate project has required data
                if not project_data or "components" not in project_data:
                    raise HTTPException(status_code=400, detail="Invalid project data")

                # Generate code using translator
                generated_code = self.translator.translate_project(
                    project_data,
                    language=request.language,
                    optimize=request.optimize
                )

                # Store generated code
                code_id = str(uuid.uuid4())
                code_record = {
                    "code_id": code_id,
                    "project_id": request.project_id,
                    "language": request.language,
                    "code": generated_code,
                    "timestamp": datetime.utcnow().isoformat(),
                    "optimize": request.optimize
                }

                # Store in database via DAL
                self.dal.store_generated_code(code_record)

                # Emit event
                self.dal.event_bus.emit(EventChannel.SYSTEM_METRICS, {
                    "service": "visual_code_generator",
                    "metric": "code_generated",
                    "project_id": request.project_id,
                    "code_id": code_id,
                    "language": request.language
                })

                return {
                    "code_id": code_id,
                    "language": request.language,
                    "code": generated_code,
                    "lines": len(generated_code.split('\n')),
                    "project_id": request.project_id
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to generate code: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/code/{code_id}")
        async def get_generated_code(code_id: str):
            """Retrieve generated code by ID"""
            try:
                code_record = self.dal.get_generated_code(code_id)
                if not code_record:
                    raise HTTPException(status_code=404, detail="Generated code not found")

                return code_record

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to retrieve code: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/languages")
        async def get_supported_languages():
            """Get list of supported programming languages"""
            return {
                "languages": self.translator.supported_languages,
                "default": "python"
            }

    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""

        def on_project_updated(event):
            """Handle project updates - could invalidate cached code"""
            project_id = event.data.get('project_id')
            logger.info(f"Project {project_id} updated, code may need regeneration")

        # Register handlers
        self.dal.event_bus.on(EventChannel.PROJECT_UPDATED, on_project_updated)

        logger.info("Event handlers registered for code generator service")

    async def startup(self):
        """Startup tasks"""
        logger.info("Visual Code Generator Service starting up...")

        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")

        logger.info("Visual Code Generator Service ready")

    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual Code Generator Service shutting down...")

        # Close HTTP client
        await self.http_client.aclose()

        # Cleanup database connections
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn

    service = VisualCodeGeneratorService()

    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)

    # Run service
    port = int(os.getenv("VISUAL_CODE_GENERATOR_PORT", 8009))
    logger.info(f"Starting Visual Code Generator Service on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()