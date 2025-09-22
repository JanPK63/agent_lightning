#!/usr/bin/env python3
"""
Integration tests for Visual Builder Microservices
Tests the microservices architecture and API gateway functionality
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import httpx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import service classes for testing
from services.visual_builder_service_integrated import VisualBuilderAPIGateway
from services.visual_workflow_engine_service import VisualWorkflowEngineService
from services.visual_component_registry_service import VisualComponentRegistryService
from services.visual_code_generator_service import VisualCodeGeneratorService
from services.visual_debugger_service import VisualDebuggerService
from services.visual_deployment_service import VisualDeploymentService
from services.visual_ai_assistant_service import VisualAIAssistantService


class TestMicroservicesIntegration:
    """Test microservices integration"""

    def test_services_can_be_imported(self):
        """Test that all microservice classes can be imported"""
        assert VisualBuilderAPIGateway
        assert VisualWorkflowEngineService
        assert VisualComponentRegistryService
        assert VisualCodeGeneratorService
        assert VisualDebuggerService
        assert VisualDeploymentService
        assert VisualAIAssistantService

    def test_services_have_required_attributes(self):
        """Test that services have required attributes"""
        # Test API Gateway
        gateway = VisualBuilderAPIGateway()
        assert hasattr(gateway, 'app')
        assert hasattr(gateway, '_forward_request')
        assert hasattr(gateway, '_aggregate_health_checks')

        # Test Workflow Engine
        workflow = VisualWorkflowEngineService()
        assert hasattr(workflow, 'app')
        assert hasattr(workflow, 'code_builder')
        assert hasattr(workflow, 'active_projects')

        # Test Component Registry
        registry = VisualComponentRegistryService()
        assert hasattr(registry, 'app')
        assert hasattr(registry, 'components')
        assert hasattr(registry, 'templates')

        # Test Code Generator
        generator = VisualCodeGeneratorService()
        assert hasattr(generator, 'app')
        assert hasattr(generator, 'translator')

        # Test Debugger
        debugger = VisualDebuggerService()
        assert hasattr(debugger, 'app')
        assert hasattr(debugger, 'debugger')

        # Test Deployment
        deployment = VisualDeploymentService()
        assert hasattr(deployment, 'app')
        assert hasattr(deployment, 'deployment_gen')

        # Test AI Assistant
        assistant = VisualAIAssistantService()
        assert hasattr(assistant, 'app')
        assert hasattr(assistant, 'ai_assistant')

    @pytest.mark.asyncio
    async def test_api_gateway_health_aggregation(self):
        """Test API gateway health check aggregation"""
        gateway = VisualBuilderAPIGateway()

        # Mock httpx client
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            health_status = await gateway._aggregate_health_checks()

            assert "overall" in health_status
            assert "services" in health_status
            assert len(health_status["services"]) == 6  # All 6 microservices

    def test_service_initialization(self):
        """Test that services can be initialized without errors"""
        try:
            # These should not raise exceptions during initialization
            VisualBuilderAPIGateway()
            VisualWorkflowEngineService()
            VisualComponentRegistryService()
            VisualCodeGeneratorService()
            VisualDebuggerService()
            VisualDeploymentService()
            VisualAIAssistantService()
            assert True
        except Exception as e:
            pytest.fail(f"Service initialization failed: {e}")

    def test_component_registry_has_default_components(self):
        """Test that component registry has default components loaded"""
        registry = VisualComponentRegistryService()

        # Check that default components are loaded
        assert len(registry.components.components) > 0
        assert "logic" in registry.components.components
        assert "data" in registry.components.components
        assert "ai" in registry.components.components

    def test_template_library_has_default_templates(self):
        """Test that template library has default templates loaded"""
        registry = VisualComponentRegistryService()

        # Check that default templates are loaded
        assert len(registry.templates.templates) > 0
        assert "basic_agent" in registry.templates.templates
        assert "ml_pipeline" in registry.templates.templates

    def test_code_translator_supports_languages(self):
        """Test that code translator supports expected languages"""
        generator = VisualCodeGeneratorService()

        expected_languages = ["python", "javascript", "java", "go"]
        assert generator.translator.supported_languages == expected_languages

    def test_ai_assistant_has_suggestions(self):
        """Test that AI assistant can generate suggestions"""
        assistant = VisualAIAssistantService()

        # Test suggestion generation
        project = {"id": "test_project", "name": "test", "components": {}}
        suggestion = assistant.ai_assistant.get_suggestion(project, "add error handling")

        assert "suggestion" in suggestion
        assert "code_snippets" in suggestion
        assert "best_practices" in suggestion

    def test_workflow_validation(self):
        """Test workflow validation logic"""
        from services.visual_workflow_engine_service import VisualProject

        # Create a simple project
        project = VisualProject("test", "Test Project", "A test project")

        # Test validation with no components
        valid, errors = project.validate()
        assert valid  # Should be valid with no components

        # Add components
        project.add_component("comp1", {"type": "data", "config": {}})
        project.add_component("comp2", {"type": "logic", "config": {}})

        # Test validation with disconnected components
        valid, errors = project.validate()
        assert not valid  # Should be invalid with disconnected components
        assert "disconnected components" in " ".join(errors)

    def test_deployment_config_generation(self):
        """Test deployment configuration generation"""
        deployment = VisualDeploymentService()

        project = {"name": "test"}
        config = deployment.deployment_gen.generate_deployment_config(
            project, "development", False
        )

        assert "environment" in config
        assert "resources" in config
        assert "deployment_type" in config
        assert config["environment"] == "development"
        assert config["auto_scale"] == False

    @pytest.mark.asyncio
    async def test_api_gateway_request_forwarding(self):
        """Test API gateway request forwarding"""
        gateway = VisualBuilderAPIGateway()

        # Mock request
        mock_request = Mock()
        mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
        mock_request.headers = {}

        # Mock httpx response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        with patch.object(gateway.http_client, 'request', new_callable=AsyncMock) as mock_request_func:
            mock_request_func.return_value = mock_response

            result = await gateway._forward_request("GET", "http://test-service:8000/test", mock_request)

            assert result == {"result": "success"}

    def test_service_ports_configuration(self):
        """Test that services use correct default ports"""
        # Check environment variable defaults
        assert os.getenv("VISUAL_BUILDER_PORT", "8006") == "8006"
        assert os.getenv("VISUAL_WORKFLOW_ENGINE_PORT", "8007") == "8007"
        assert os.getenv("VISUAL_COMPONENT_REGISTRY_PORT", "8008") == "8008"
        assert os.getenv("VISUAL_CODE_GENERATOR_PORT", "8009") == "8009"
        assert os.getenv("VISUAL_DEBUGGER_PORT", "8010") == "8010"
        assert os.getenv("VISUAL_DEPLOYMENT_PORT", "8011") == "8011"
        assert os.getenv("VISUAL_AI_ASSISTANT_PORT", "8012") == "8012"

    def test_docker_compose_services_defined(self):
        """Test that docker-compose.yml includes all microservices"""
        import yaml

        with open("docker-compose.yml", "r") as f:
            compose_data = yaml.safe_load(f)

        services = compose_data["services"]

        # Check that all microservices are defined
        expected_services = [
            "visual-builder-gateway",
            "visual-workflow-engine",
            "visual-component-registry",
            "visual-code-generator",
            "visual-debugger",
            "visual-deployment",
            "visual-ai-assistant"
        ]

        for service in expected_services:
            assert service in services, f"Service {service} not found in docker-compose.yml"

    def test_service_dependencies(self):
        """Test that services have correct dependencies in docker-compose"""
        import yaml

        with open("docker-compose.yml", "r") as f:
            compose_data = yaml.safe_load(f)

        services = compose_data["services"]

        # Check gateway dependencies
        gateway_deps = services["visual-builder-gateway"].get("depends_on", [])
        expected_deps = [
            "visual-workflow-engine",
            "visual-component-registry",
            "visual-code-generator",
            "visual-debugger",
            "visual-deployment",
            "visual-ai-assistant"
        ]

        for dep in expected_deps:
            assert dep in gateway_deps, f"Gateway missing dependency: {dep}"


if __name__ == "__main__":
    pytest.main([__file__])