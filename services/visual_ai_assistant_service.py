#!/usr/bin/env python3
"""
Visual AI Assistant Microservice
Provides AI-powered coding assistance and suggestions for visual projects
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


class AISuggestionRequest(BaseModel):
    """AI suggestion request"""
    project_id: str = Field(description="Project ID")
    query: str = Field(description="User query or context")
    component_id: Optional[str] = Field(default=None, description="Specific component ID")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class CodeOptimizationRequest(BaseModel):
    """Code optimization request"""
    project_id: str = Field(description="Project ID")
    code: str = Field(description="Code to optimize")
    language: str = Field(default="python", description="Programming language")


class AIAssistant:
    """AI-powered coding assistant"""

    def __init__(self):
        self.suggestions_cache = {}
        self.supported_languages = ["python", "javascript", "java", "go", "typescript"]

    def get_suggestion(self, project: dict, query: str, component_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> dict:
        """Get AI suggestion for project"""
        cache_key = f"{project['id']}_{query}_{component_id}"
        if cache_key in self.suggestions_cache:
            return self.suggestions_cache[cache_key]

        # Analyze project structure
        components = project.get("components", {})
        connections = project.get("connections", [])

        # Generate context-aware suggestions
        suggestion = self._generate_smart_suggestion(project, query, component_id, context)

        # Cache suggestion
        self.suggestions_cache[cache_key] = suggestion

        return suggestion

    def _generate_smart_suggestion(self, project: dict, query: str, component_id: Optional[str], context: Optional[Dict[str, Any]]) -> dict:
        """Generate smart suggestion based on project analysis"""
        components = project.get("components", {})
        connections = project.get("connections", [])

        # Analyze component types used
        component_types = set()
        for comp in components.values():
            component_types.add(comp.get("type", "unknown"))

        # Generate suggestions based on query patterns
        if "error" in query.lower() or "exception" in query.lower():
            suggestion = self._generate_error_handling_suggestion(component_types)
        elif "performance" in query.lower() or "speed" in query.lower():
            suggestion = self._generate_performance_suggestion(component_types)
        elif "test" in query.lower():
            suggestion = self._generate_testing_suggestion(component_types)
        elif "security" in query.lower():
            suggestion = self._generate_security_suggestion(component_types)
        else:
            suggestion = self._generate_general_suggestion(query, component_types)

        # Add project-specific insights
        suggestion["project_insights"] = {
            "total_components": len(components),
            "component_types": list(component_types),
            "connections_count": len(connections),
            "has_cycles": self._detect_cycles(connections)
        }

        return suggestion

    def _generate_error_handling_suggestion(self, component_types: set) -> dict:
        """Generate error handling suggestions"""
        return {
            "suggestion": "Add comprehensive error handling and validation",
            "code_snippets": [
                "try:\n    # Your code here\n    result = process_data(data)\nexcept ValueError as e:\n    logger.error(f'Validation error: {e}')\n    raise\nexcept Exception as e:\n    logger.error(f'Unexpected error: {e}')\n    raise",
                "from pydantic import ValidationError\n\ntry:\n    validated_data = MyModel(**input_data)\nexcept ValidationError as e:\n    return {\"error\": e.errors()}"
            ],
            "best_practices": [
                "Use specific exception types",
                "Log errors with context",
                "Validate inputs early",
                "Use circuit breakers for external calls",
                "Implement proper error propagation"
            ],
            "priority": "high"
        }

    def _generate_performance_suggestion(self, component_types: set) -> dict:
        """Generate performance optimization suggestions"""
        suggestions = []

        if "ai" in component_types:
            suggestions.extend([
                "Consider using async/await for AI API calls",
                "Implement caching for expensive AI computations",
                "Use batch processing for multiple AI requests"
            ])

        if "data" in component_types:
            suggestions.extend([
                "Use streaming for large data processing",
                "Implement pagination for data retrieval",
                "Consider using async database drivers"
            ])

        return {
            "suggestion": "Optimize performance with async processing and caching",
            "code_snippets": [
                "import asyncio\n\nasync def process_batch(items: List[dict]) -> List[dict]:\n    tasks = [process_item(item) for item in items]\n    return await asyncio.gather(*tasks)",
                "@lru_cache(maxsize=128)\ndef expensive_computation(param: str) -> dict:\n    # Expensive operation here\n    return result"
            ],
            "best_practices": suggestions,
            "priority": "medium"
        }

    def _generate_testing_suggestion(self, component_types: set) -> dict:
        """Generate testing suggestions"""
        return {
            "suggestion": "Implement comprehensive unit and integration tests",
            "code_snippets": [
                "import pytest\n\nclass TestMyComponent:\n    def test_success_case(self):\n        # Arrange\n        input_data = {'key': 'value'}\n        \n        # Act\n        result = my_function(input_data)\n        \n        # Assert\n        assert result['status'] == 'success'\n        assert 'data' in result",
                "def test_error_handling():\n    with pytest.raises(ValueError):\n        my_function(invalid_input)"
            ],
            "best_practices": [
                "Write tests before implementation (TDD)",
                "Test both success and error cases",
                "Use fixtures for test data setup",
                "Mock external dependencies",
                "Aim for >80% code coverage"
            ],
            "priority": "high"
        }

    def _generate_security_suggestion(self, component_types: set) -> dict:
        """Generate security suggestions"""
        return {
            "suggestion": "Implement security best practices and input validation",
            "code_snippets": [
                "from pydantic import BaseModel, validator\n\nclass SecureInput(BaseModel):\n    user_id: int\n    \n    @validator('user_id')\n    def validate_user_id(cls, v):\n        if v <= 0:\n            raise ValueError('User ID must be positive')\n        return v",
                "import bleach\n\n# Sanitize user input\nclean_input = bleach.clean(user_input, tags=[], strip=True)"
            ],
            "best_practices": [
                "Validate and sanitize all inputs",
                "Use parameterized queries for database access",
                "Implement proper authentication and authorization",
                "Log security events",
                "Regular security audits and updates"
            ],
            "priority": "critical"
        }

    def _generate_general_suggestion(self, query: str, component_types: set) -> dict:
        """Generate general improvement suggestions"""
        return {
            "suggestion": f"Based on your query '{query}', consider these improvements",
            "code_snippets": [
                "# Add logging for better observability\nlogger.info(f'Processing {len(data)} items')\n\n# Use type hints for better code quality\ndef process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:\n    pass"
            ],
            "best_practices": [
                "Add comprehensive logging",
                "Use type hints and docstrings",
                "Follow SOLID principles",
                "Implement proper error handling",
                "Write unit tests"
            ],
            "priority": "low"
        }

    def _detect_cycles(self, connections: List[Dict[str, Any]]) -> bool:
        """Detect cycles in component connections"""
        # Simple cycle detection - in production use NetworkX
        try:
            import networkx as nx
            G = nx.DiGraph()
            for conn in connections:
                G.add_edge(conn["source"], conn["target"])
            return not nx.is_directed_acyclic_graph(G)
        except ImportError:
            # Fallback simple cycle detection
            return False

    def optimize_code(self, code: str, language: str) -> dict:
        """Optimize code using AI suggestions"""
        optimizations = []

        if language == "python":
            optimizations = self._optimize_python_code(code)
        elif language == "javascript":
            optimizations = self._optimize_javascript_code(code)
        else:
            optimizations = ["General optimizations apply"]

        return {
            "original_code": code,
            "optimizations": optimizations,
            "optimized_code": self._apply_optimizations(code, optimizations, language),
            "language": language
        }

    def _optimize_python_code(self, code: str) -> List[str]:
        """Python-specific optimizations"""
        optimizations = []

        if "for " in code and "range(len(" in code:
            optimizations.append("Use enumerate instead of range(len())")

        if "print(" in code:
            optimizations.append("Replace print statements with proper logging")

        if "except:" in code:
            optimizations.append("Use specific exception types instead of bare except")

        return optimizations

    def _optimize_javascript_code(self, code: str) -> List[str]:
        """JavaScript-specific optimizations"""
        optimizations = []

        if "var " in code:
            optimizations.append("Use let/const instead of var")

        if "function(" in code and "=>" not in code:
            optimizations.append("Consider using arrow functions where appropriate")

        return optimizations

    def _apply_optimizations(self, code: str, optimizations: List[str], language: str) -> str:
        """Apply optimizations to code (simplified)"""
        # In production, this would use AST parsing and transformation
        optimized = code

        if language == "python":
            optimized = optimized.replace("print(", "logger.info(")
            optimized = optimized.replace("except:", "except Exception as e:")

        return optimized


class VisualAIAssistantService:
    """Visual AI Assistant Microservice"""

    def __init__(self):
        self.app = FastAPI(title="Visual AI Assistant Service", version="1.0.0")

        # Initialize components
        self.dal = DataAccessLayer("visual_ai_assistant")
        self.cache = get_cache()
        self.ai_assistant = AIAssistant()

        # HTTP client for inter-service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Service URLs
        self.workflow_engine_url = os.getenv("VISUAL_WORKFLOW_ENGINE_URL", "http://localhost:8007")
        self.component_registry_url = os.getenv("VISUAL_COMPONENT_REGISTRY_URL", "http://localhost:8008")

        logger.info("✅ Connected to shared database and initialized AI assistant")

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
                "service": "visual_ai_assistant",
                "status": "healthy" if health_status['database'] and health_status['cache'] else "degraded",
                "database": health_status['database'],
                "cache": health_status['cache'],
                "supported_languages": self.ai_assistant.supported_languages,
                "cached_suggestions": len(self.ai_assistant.suggestions_cache),
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.post("/suggest")
        async def get_ai_suggestion(request: AISuggestionRequest):
            """Get AI-powered suggestion for project"""
            try:
                # Get project data from workflow engine
                async with self.http_client as client:
                    response = await client.get(f"{self.workflow_engine_url}/projects/{request.project_id}")
                    if response.status_code != 200:
                        raise HTTPException(status_code=404, detail="Project not found in workflow engine")
                    project_data = response.json()

                # Get AI suggestion
                suggestion = self.ai_assistant.get_suggestion(
                    project_data,
                    request.query,
                    request.component_id,
                    request.context
                )

                # Cache suggestion
                cache_key = f"suggestion:{request.project_id}:{uuid.uuid4()}"
                self.cache.set(cache_key, suggestion, ttl=3600)

                # Emit event
                self.dal.event_bus.emit(EventChannel.AI_SUGGESTION_GENERATED, {
                    "project_id": request.project_id,
                    "query": request.query,
                    "suggestion_type": suggestion.get("priority", "general")
                })

                return {
                    "suggestion": suggestion,
                    "project_id": request.project_id,
                    "query": request.query,
                    "timestamp": datetime.utcnow().isoformat()
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to generate AI suggestion: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/optimize")
        async def optimize_code(request: CodeOptimizationRequest):
            """Optimize code using AI"""
            try:
                # Optimize code
                optimization_result = self.ai_assistant.optimize_code(
                    request.code,
                    request.language
                )

                # Cache result
                cache_key = f"optimization:{request.project_id}:{uuid.uuid4()}"
                self.cache.set(cache_key, optimization_result, ttl=3600)

                # Emit event
                self.dal.event_bus.emit(EventChannel.CODE_OPTIMIZED, {
                    "project_id": request.project_id,
                    "language": request.language,
                    "optimizations_count": len(optimization_result.get("optimizations", []))
                })

                return optimization_result

            except Exception as e:
                logger.error(f"Failed to optimize code: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/suggestions/{project_id}")
        async def get_project_suggestions(project_id: str):
            """Get cached suggestions for a project"""
            try:
                suggestions = []
                pattern = f"suggestion:{project_id}:*"

                for key in self.cache.redis_client.keys(pattern):
                    suggestion = self.cache.get(key)
                    if suggestion:
                        suggestions.append(suggestion)

                return {
                    "project_id": project_id,
                    "suggestions": suggestions,
                    "count": len(suggestions)
                }

            except Exception as e:
                logger.error(f"Failed to get project suggestions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/capabilities")
        async def get_ai_capabilities():
            """Get AI assistant capabilities"""
            try:
                return {
                    "supported_languages": self.ai_assistant.supported_languages,
                    "suggestion_types": [
                        "error_handling",
                        "performance",
                        "testing",
                        "security",
                        "general_improvements"
                    ],
                    "optimization_features": [
                        "code_analysis",
                        "performance_suggestions",
                        "security_improvements",
                        "best_practices"
                    ],
                    "caching_enabled": True
                }

            except Exception as e:
                logger.error(f"Failed to get AI capabilities: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_event_handlers(self):
        """Setup event handlers for cross-service communication"""

        def on_project_updated(event):
            """Handle project updates - could invalidate cached suggestions"""
            project_id = event.data.get('project_id')
            # Clear cached suggestions for updated project
            pattern = f"suggestion:{project_id}:*"
            for key in self.cache.redis_client.keys(pattern):
                self.cache.delete(key)
            logger.info(f"Cleared cached suggestions for project {project_id}")

        def on_component_added(event):
            """Handle component additions - could trigger suggestions"""
            project_id = event.data.get('project_id')
            component_id = event.data.get('component_id')
            logger.info(f"Component {component_id} added to project {project_id} - AI assistant notified")

        # Register handlers
        self.dal.event_bus.on(EventChannel.PROJECT_UPDATED, on_project_updated)
        self.dal.event_bus.on(EventChannel.COMPONENT_ADDED, on_component_added)

        logger.info("Event handlers registered for AI assistant service")

    async def startup(self):
        """Startup tasks"""
        logger.info("Visual AI Assistant Service starting up...")

        # Verify database connection
        health = self.dal.health_check()
        if not health['database']:
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        if not health['cache']:
            logger.warning("Cache not available, performance may be degraded")

        logger.info("Visual AI Assistant Service ready")

    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Visual AI Assistant Service shutting down...")

        # Close HTTP client
        await self.http_client.aclose()

        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn

    service = VisualAIAssistantService()

    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)

    # Run service
    port = int(os.getenv("VISUAL_AI_ASSISTANT_PORT", 8012))
    logger.info(f"Starting Visual AI Assistant Service on port {port}")

    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()