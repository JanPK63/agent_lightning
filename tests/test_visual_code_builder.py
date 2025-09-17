import pytest
import requests


class TestVisualCodeBuilderIntegration:
    """Integration tests for Visual Code Builder Service"""

    def setup_method(self):
        """Setup test fixtures"""
        self.base_url = "http://localhost:8006"
        self.session = requests.Session()

    def test_health_check(self):
        """Test service health check"""
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "visual_builder"
        assert "status" in data
        assert "database" in data
        assert "cache" in data

    def test_component_library(self):
        """Test component library retrieval"""
        response = self.session.get(f"{self.base_url}/components/library")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "total" in data
        assert data["total"] > 0

    def test_create_project(self):
        """Test project creation"""
        project_data = {
            "name": "Test Integration Project",
            "description": "Created by integration test"
        }
        response = self.session.post(
            f"{self.base_url}/projects",
            json=project_data
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == project_data["name"]
        assert data["description"] == project_data["description"]

        # Store project ID for other tests
        self.test_project_id = data["id"]

    def test_add_component(self):
        """Test adding component to project"""
        if not hasattr(self, 'test_project_id'):
            self.test_create_project()

        component_data = {
            "project_id": self.test_project_id,
            "component_type": "logic",
            "component_id": "condition",
            "position": {"x": 50, "y": 50}
        }
        response = self.session.post(
            f"{self.base_url}/components/add",
            json=component_data
        )
        assert response.status_code == 200
        data = response.json()
        assert "component_id" in data
        assert "status" in data

    def test_generate_code(self):
        """Test code generation"""
        # Ensure we have a project with components
        self.test_create_project()
        self.test_add_component()

        code_request = {
            "project_id": self.test_project_id,
            "language": "python"
        }
        response = self.session.post(
            f"{self.base_url}/generate/code",
            json=code_request
        )
        assert response.status_code == 200
        data = response.json()
        assert "code_id" in data
        assert "language" in data
        assert "code" in data
        assert "lines" in data

        # Verify code contains expected elements
        assert "from fastapi import FastAPI" in data["code"]
        assert "uvicorn.run" in data["code"]

        # Store code ID for download test
        self.test_code_id = data["code_id"]

    def test_download_code(self):
        """Test code download"""
        if not hasattr(self, 'test_code_id'):
            self.test_generate_code()

        response = self.session.get(
            f"{self.base_url}/download/{self.test_code_id}"
        )
        assert response.status_code == 200

        # Check content type
        assert "text/plain" in response.headers.get("content-type", "")

        # Verify code content
        code_content = response.text
        assert "from fastapi import FastAPI" in code_content
        assert "uvicorn.run" in code_content

    def test_list_projects(self):
        """Test project listing"""
        response = self.session.get(f"{self.base_url}/projects")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert "count" in data
        assert isinstance(data["projects"], list)

    def test_get_project(self):
        """Test project retrieval"""
        if not hasattr(self, 'test_project_id'):
            self.test_create_project()

        response = self.session.get(
            f"{self.base_url}/projects/{self.test_project_id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == self.test_project_id
        assert "components" in data
        assert "connections" in data
        assert "metadata" in data


class TestCodeTranslator:
    """Unit tests for CodeTranslator"""

    def setup_method(self):
        """Setup test fixtures"""
        # Import here to avoid issues if service isn't running
        try:
            from services.visual_builder_service_integrated \
                import CodeTranslator
            self.translator = CodeTranslator()
        except ImportError:
            pytest.skip("Visual Code Builder service not available")

    def test_supported_languages(self):
        """Test supported languages"""
        assert "python" in self.translator.supported_languages
        assert "javascript" in self.translator.supported_languages

    def test_generate_fastapi_scaffold_empty_project(self):
        """Test FastAPI scaffold generation for empty project"""
        project = {
            "name": "EmptyProject",
            "components": {}
        }
        code = self.translator._generate_fastapi_scaffold(project)

        assert "from fastapi import FastAPI" in code
        assert "uvicorn.run" in code
        assert "EmptyProject API" in code
        assert "@app.get(\"/\")" in code
        assert "@app.get(\"/health\")" in code

    def test_generate_fastapi_scaffold_with_components(self):
        """Test FastAPI scaffold generation with components"""
        project = {
            "name": "TestAPI",
            "components": {
                "user_data": {
                    "type": "data",
                    "component_id": "input",
                    "position": {"x": 100, "y": 100}
                },
                "logic_processor": {
                    "type": "logic",
                    "component_id": "function",
                    "position": {"x": 200, "y": 200}
                },
                "ai_predictor": {
                    "type": "ai",
                    "component_id": "classifier",
                    "position": {"x": 300, "y": 300}
                }
            }
        }
        code = self.translator._generate_fastapi_scaffold(project)

        # Check FastAPI imports
        assert "from fastapi import FastAPI" in code
        assert "from pydantic import BaseModel" in code

        # Check generated routes
        assert "@app.get(\"/user-data\")" in code
        assert "@app.post(\"/user-data\")" in code
        assert "@app.post(\"/logic-processor/execute\")" in code
        assert "@app.post(\"/ai-predictor/predict\")" in code

        # Check Pydantic models
        assert "class UserDataModel(BaseModel):" in code

    def test_generate_basic_python_fallback(self):
        """Test basic Python code generation fallback"""
        project = {
            "name": "FallbackProject",
            "components": {
                "test_comp": {
                    "type": "logic",
                    "component_id": "function"
                }
            }
        }
        code = self.translator._generate_basic_python(project)

        assert "class FallbackProject:" in code
        assert "async def execute(" in code
        assert "asyncio.run" in code
        assert "test_comp" in code