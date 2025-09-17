#!/usr/bin/env python3
"""
CI/CD Matrix Service
Provides multi-language CI/CD pipeline generation and management
Supports GitHub Actions, GitLab CI, Jenkins, CircleCI, and more
"""

import os
import sys
import json
import yaml
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jinja2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.cache import get_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CICDPlatform(str, Enum):
    """Supported CI/CD platforms"""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"
    AZURE_DEVOPS = "azure_devops"
    BITBUCKET = "bitbucket"
    TRAVIS_CI = "travis_ci"


class Language(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"


class PipelineRequest(BaseModel):
    """Pipeline generation request"""
    platform: CICDPlatform = Field(description="CI/CD platform")
    languages: List[Language] = Field(description="Programming languages")
    project_name: str = Field(description="Project name")
    stages: Optional[List[str]] = Field(
        default=["lint", "test", "build", "security", "deploy"],
        description="Pipeline stages"
    )
    environments: Optional[List[str]] = Field(
        default=["dev", "staging", "production"],
        description="Deployment environments"
    )
    features: Optional[Dict[str, bool]] = Field(
        default_factory=dict,
        description="Feature flags"
    )


class MatrixConfig(BaseModel):
    """CI/CD matrix configuration"""
    os: List[str] = Field(default=["ubuntu-latest"], description="Operating systems")
    language_versions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Language versions"
    )
    include: Optional[List[Dict]] = Field(default=None, description="Matrix includes")
    exclude: Optional[List[Dict]] = Field(default=None, description="Matrix excludes")


class PipelineStatus(BaseModel):
    """Pipeline execution status"""
    pipeline_id: str = Field(description="Pipeline ID")
    status: str = Field(description="Status: running, success, failed, cancelled")
    platform: str = Field(description="CI/CD platform")
    branch: str = Field(description="Branch name")
    commit: str = Field(description="Commit SHA")
    started_at: str = Field(description="Start time")
    finished_at: Optional[str] = Field(default=None, description="Finish time")
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Stage statuses")


class CICDMatrixService:
    """CI/CD Matrix Service for multi-language pipeline support"""
    
    def __init__(self):
        self.app = FastAPI(title="CI/CD Matrix Service", version="1.0.0")
        
        # Initialize components
        self.dal = DataAccessLayer("cicd_matrix")
        self.cache = get_cache()
        
        # Template engine for pipeline generation
        self.template_env = jinja2.Environment(
            loader=jinja2.DictLoader(self._load_templates())
        )
        
        # Language configurations
        self.language_configs = self._load_language_configs()
        
        # Platform-specific configurations
        self.platform_configs = self._load_platform_configs()
        
        logger.info("✅ CI/CD Matrix Service initialized")
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _load_templates(self) -> Dict[str, str]:
        """Load CI/CD pipeline templates"""
        return {
            "github_actions": """name: {{ project_name }} CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
{% for language in languages %}
  {{ language }}-pipeline:
    runs-on: {% raw %}${{ matrix.os }}{% endraw %}
    strategy:
      matrix:
        os: {{ os }}
        {{ language }}-version: {{ language_versions[language] }}
    
    steps:
    - uses: actions/checkout@v3
    
    {% if language == 'python' %}
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: {% raw %}${{ matrix.python-version }}{% endraw %}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8 black mypy
    
    - name: Lint
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check .
        mypy .
    
    - name: Test
      run: |
        pytest --cov=./ --cov-report=xml
    
    {% elif language == 'javascript' or language == 'typescript' %}
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: {% raw %}${{ matrix.node-version }}{% endraw %}
    
    - name: Install dependencies
      run: npm ci
    
    - name: Lint
      run: npm run lint
    
    - name: Test
      run: npm test -- --coverage
    
    - name: Build
      run: npm run build
    
    {% elif language == 'go' %}
    - name: Setup Go
      uses: actions/setup-go@v4
      with:
        go-version: {% raw %}${{ matrix.go-version }}{% endraw %}
    
    - name: Install dependencies
      run: go mod download
    
    - name: Lint
      run: |
        go fmt ./...
        go vet ./...
        golangci-lint run
    
    - name: Test
      run: go test -v -race -coverprofile=coverage.out ./...
    
    - name: Build
      run: go build -v ./...
    
    {% elif language == 'java' %}
    - name: Setup Java
      uses: actions/setup-java@v3
      with:
        java-version: {% raw %}${{ matrix.java-version }}{% endraw %}
        distribution: 'temurin'
    
    - name: Build with Maven
      run: mvn clean compile
    
    - name: Test
      run: mvn test
    
    - name: Package
      run: mvn package
    {% endif %}
    
    {% if 'security' in stages %}
    - name: Security Scan
      run: |
        # Run security scanning
        echo "Running security scan for {{ language }}"
    {% endif %}
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: {{ language }}
{% endfor %}

{% if 'deploy' in stages %}
  deploy:
    needs: [{% for lang in languages %}{{ lang }}-pipeline{{ ", " if not loop.last }}{% endfor %}]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    {% for env in environments %}
    - name: Deploy to {{ env }}
      if: {% if env == 'production' %}github.ref == 'refs/heads/main'{% else %}true{% endif %}
      run: |
        echo "Deploying to {{ env }}"
        # Add deployment commands here
    {% endfor %}
{% endif %}
""",
            
            "gitlab_ci": """.gitlab-ci.yml
stages:
{% for stage in stages %}
  - {{ stage }}
{% endfor %}

variables:
  DOCKER_DRIVER: overlay2

{% for language in languages %}
{{ language }}-lint:
  stage: lint
  {% if language == 'python' %}
  image: python:{{ language_versions[language][0] }}
  script:
    - pip install flake8 black mypy
    - flake8 .
    - black --check .
    - mypy .
  {% elif language == 'javascript' or language == 'typescript' %}
  image: node:{{ language_versions[language][0] }}
  script:
    - npm ci
    - npm run lint
  {% elif language == 'go' %}
  image: golang:{{ language_versions[language][0] }}
  script:
    - go fmt ./...
    - go vet ./...
  {% endif %}

{{ language }}-test:
  stage: test
  {% if language == 'python' %}
  image: python:{{ language_versions[language][0] }}
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest --cov=./
  {% elif language == 'javascript' or language == 'typescript' %}
  image: node:{{ language_versions[language][0] }}
  script:
    - npm ci
    - npm test
  {% elif language == 'go' %}
  image: golang:{{ language_versions[language][0] }}
  script:
    - go test -v ./...
  {% endif %}
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
{% endfor %}

{% if 'deploy' in stages %}
{% for env in environments %}
deploy-{{ env }}:
  stage: deploy
  script:
    - echo "Deploying to {{ env }}"
  environment:
    name: {{ env }}
  {% if env == 'production' %}
  only:
    - main
  when: manual
  {% endif %}
{% endfor %}
{% endif %}
""",
            
            "jenkins": """pipeline {
    agent any
    
    environment {
        PROJECT_NAME = '{{ project_name }}'
    }
    
    stages {
        {% for stage in stages %}
        stage('{{ stage|capitalize }}') {
            {% if stage == 'test' %}
            matrix {
                axes {
                    axis {
                        name 'LANGUAGE'
                        values {{ languages|join(', ') }}
                    }
                    axis {
                        name 'OS'
                        values {{ os|join(', ') }}
                    }
                }
                stages {
                    stage('Test') {
                        steps {
                            script {
                                if (env.LANGUAGE == 'python') {
                                    sh 'python -m pytest'
                                } else if (env.LANGUAGE == 'javascript') {
                                    sh 'npm test'
                                } else if (env.LANGUAGE == 'go') {
                                    sh 'go test ./...'
                                }
                            }
                        }
                    }
                }
            }
            {% else %}
            steps {
                echo 'Running {{ stage }}'
            }
            {% endif %}
        }
        {% endfor %}
    }
    
    post {
        always {
            junit '**/test-reports/*.xml'
            publishHTML([
                reportDir: 'coverage',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }
    }
}
"""
        }
    
    def _load_language_configs(self) -> Dict[str, Dict]:
        """Load language-specific configurations"""
        return {
            Language.PYTHON: {
                "versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "test_framework": "pytest",
                "linters": ["flake8", "black", "mypy", "pylint"],
                "build_tool": "setuptools",
                "package_manager": "pip"
            },
            Language.JAVASCRIPT: {
                "versions": ["16", "18", "20"],
                "test_framework": "jest",
                "linters": ["eslint", "prettier"],
                "build_tool": "webpack",
                "package_manager": "npm"
            },
            Language.TYPESCRIPT: {
                "versions": ["16", "18", "20"],
                "test_framework": "jest",
                "linters": ["eslint", "prettier", "tslint"],
                "build_tool": "webpack",
                "package_manager": "npm"
            },
            Language.GO: {
                "versions": ["1.19", "1.20", "1.21"],
                "test_framework": "go test",
                "linters": ["golangci-lint", "go fmt", "go vet"],
                "build_tool": "go build",
                "package_manager": "go mod"
            },
            Language.JAVA: {
                "versions": ["8", "11", "17", "21"],
                "test_framework": "junit",
                "linters": ["checkstyle", "spotbugs"],
                "build_tool": "maven",
                "package_manager": "maven"
            },
            Language.RUST: {
                "versions": ["stable", "beta", "nightly"],
                "test_framework": "cargo test",
                "linters": ["clippy", "rustfmt"],
                "build_tool": "cargo",
                "package_manager": "cargo"
            },
            Language.RUBY: {
                "versions": ["2.7", "3.0", "3.1", "3.2"],
                "test_framework": "rspec",
                "linters": ["rubocop"],
                "build_tool": "rake",
                "package_manager": "bundler"
            },
            Language.PHP: {
                "versions": ["7.4", "8.0", "8.1", "8.2"],
                "test_framework": "phpunit",
                "linters": ["phpcs", "phpmd"],
                "build_tool": "composer",
                "package_manager": "composer"
            }
        }
    
    def _load_platform_configs(self) -> Dict[str, Dict]:
        """Load platform-specific configurations"""
        return {
            CICDPlatform.GITHUB_ACTIONS: {
                "config_file": ".github/workflows/ci.yml",
                "runners": ["ubuntu-latest", "windows-latest", "macos-latest"],
                "features": ["matrix", "artifacts", "caching", "secrets", "environments"]
            },
            CICDPlatform.GITLAB_CI: {
                "config_file": ".gitlab-ci.yml",
                "runners": ["docker", "shell", "kubernetes"],
                "features": ["stages", "artifacts", "cache", "environments", "manual_approval"]
            },
            CICDPlatform.JENKINS: {
                "config_file": "Jenkinsfile",
                "runners": ["any", "docker", "kubernetes"],
                "features": ["matrix", "parallel", "stages", "post_actions"]
            },
            CICDPlatform.CIRCLECI: {
                "config_file": ".circleci/config.yml",
                "runners": ["docker", "machine", "macos"],
                "features": ["workflows", "orbs", "caching", "artifacts"]
            },
            CICDPlatform.AZURE_DEVOPS: {
                "config_file": "azure-pipelines.yml",
                "runners": ["ubuntu", "windows", "macos"],
                "features": ["stages", "jobs", "templates", "artifacts"]
            }
        }
    
    async def generate_pipeline(self, request: PipelineRequest) -> Dict[str, Any]:
        """Generate CI/CD pipeline configuration"""
        try:
            # Get platform configuration
            platform_config = self.platform_configs.get(request.platform)
            if not platform_config:
                raise ValueError(f"Unsupported platform: {request.platform}")
            
            # Prepare template context
            context = {
                "project_name": request.project_name,
                "languages": request.languages,
                "stages": request.stages,
                "environments": request.environments,
                "os": platform_config["runners"][:1],  # Default to first runner
                "language_versions": {}
            }
            
            # Add language versions
            for language in request.languages:
                lang_config = self.language_configs.get(language)
                if lang_config:
                    context["language_versions"][language] = lang_config["versions"][:2]
            
            # Add feature flags
            if request.features:
                context.update(request.features)
            
            # Generate pipeline configuration
            template = self.template_env.get_template(request.platform.value)
            pipeline_config = template.render(context)
            
            # Store in cache
            pipeline_id = f"{request.project_name}_{request.platform.value}_{datetime.utcnow().timestamp()}"
            cache_key = f"pipeline:{pipeline_id}"
            
            pipeline_data = {
                "id": pipeline_id,
                "platform": request.platform.value,
                "config": pipeline_config,
                "config_file": platform_config["config_file"],
                "languages": request.languages,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.cache.set(cache_key, pipeline_data, ttl=3600)
            
            logger.info(f"Generated {request.platform.value} pipeline for {request.project_name}")
            
            return pipeline_data
            
        except Exception as e:
            logger.error(f"Failed to generate pipeline: {e}")
            raise
    
    async def create_matrix(self, languages: List[Language], 
                          platform: CICDPlatform,
                          matrix_config: Optional[MatrixConfig] = None) -> Dict[str, Any]:
        """Create CI/CD matrix for multi-language testing"""
        if not matrix_config:
            matrix_config = MatrixConfig()
        
        matrix = {
            "platform": platform.value,
            "strategy": {
                "matrix": {
                    "os": matrix_config.os,
                    "language": []
                }
            }
        }
        
        # Build language matrix
        for language in languages:
            lang_config = self.language_configs.get(language)
            if lang_config:
                lang_matrix = {
                    "name": language.value,
                    "versions": matrix_config.language_versions.get(
                        language.value, 
                        lang_config["versions"][:3]
                    ),
                    "test_framework": lang_config["test_framework"],
                    "linters": lang_config["linters"]
                }
                matrix["strategy"]["matrix"]["language"].append(lang_matrix)
        
        # Add includes/excludes if specified
        if matrix_config.include:
            matrix["strategy"]["matrix"]["include"] = matrix_config.include
        if matrix_config.exclude:
            matrix["strategy"]["matrix"]["exclude"] = matrix_config.exclude
        
        # Platform-specific adjustments
        if platform == CICDPlatform.GITHUB_ACTIONS:
            matrix["strategy"]["fail-fast"] = False
            matrix["strategy"]["max-parallel"] = 4
        elif platform == CICDPlatform.GITLAB_CI:
            matrix["parallel"] = {
                "matrix": matrix["strategy"]["matrix"]
            }
            del matrix["strategy"]
        
        return matrix
    
    async def get_pipeline_status(self, pipeline_id: str) -> PipelineStatus:
        """Get pipeline execution status"""
        # In a real implementation, this would integrate with actual CI/CD platforms
        # For now, return mock status from cache
        
        cache_key = f"pipeline_status:{pipeline_id}"
        status = self.cache.get(cache_key)
        
        if not status:
            # Create mock status
            status = PipelineStatus(
                pipeline_id=pipeline_id,
                status="running",
                platform="github_actions",
                branch="main",
                commit="abc123def456",
                started_at=datetime.utcnow().isoformat(),
                stages=[
                    {"name": "lint", "status": "success", "duration": 45},
                    {"name": "test", "status": "running", "duration": None},
                    {"name": "build", "status": "pending", "duration": None}
                ]
            )
            self.cache.set(cache_key, status.dict(), ttl=300)
        
        return PipelineStatus(**status) if isinstance(status, dict) else status
    
    async def generate_docker_compose(self, languages: List[Language]) -> str:
        """Generate Docker Compose configuration for multi-language setup"""
        services = {}
        
        for language in languages:
            lang_config = self.language_configs.get(language)
            if not lang_config:
                continue
            
            service_name = f"{language.value}_service"
            
            if language == Language.PYTHON:
                services[service_name] = {
                    "image": f"python:{lang_config['versions'][0]}",
                    "working_dir": "/app",
                    "volumes": [".:/app"],
                    "command": "python -m pytest"
                }
            elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                services[service_name] = {
                    "image": f"node:{lang_config['versions'][0]}",
                    "working_dir": "/app",
                    "volumes": [".:/app"],
                    "command": "npm test"
                }
            elif language == Language.GO:
                services[service_name] = {
                    "image": f"golang:{lang_config['versions'][0]}",
                    "working_dir": "/app",
                    "volumes": [".:/app"],
                    "command": "go test ./..."
                }
            elif language == Language.JAVA:
                services[service_name] = {
                    "image": f"maven:3-openjdk-{lang_config['versions'][0]}",
                    "working_dir": "/app",
                    "volumes": [".:/app"],
                    "command": "mvn test"
                }
        
        compose_config = {
            "version": "3.8",
            "services": services
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    async def validate_pipeline(self, platform: CICDPlatform, config: str) -> Dict[str, Any]:
        """Validate pipeline configuration"""
        try:
            # Parse based on platform
            if platform in [CICDPlatform.GITHUB_ACTIONS, CICDPlatform.AZURE_DEVOPS]:
                parsed = yaml.safe_load(config)
            elif platform == CICDPlatform.GITLAB_CI:
                parsed = yaml.safe_load(config)
            elif platform == CICDPlatform.JENKINS:
                # Basic Jenkinsfile validation
                if "pipeline" not in config or "stages" not in config:
                    raise ValueError("Invalid Jenkinsfile structure")
                parsed = {"valid": True}
            else:
                parsed = yaml.safe_load(config)
            
            return {
                "valid": True,
                "platform": platform.value,
                "parsed": parsed,
                "warnings": [],
                "errors": []
            }
            
        except Exception as e:
            return {
                "valid": False,
                "platform": platform.value,
                "errors": [str(e)],
                "warnings": []
            }
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "cicd_matrix",
                "status": "healthy",
                "platforms": len(self.platform_configs),
                "languages": len(self.language_configs),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/pipeline/generate")
        async def generate_pipeline(request: PipelineRequest):
            """Generate CI/CD pipeline"""
            return await self.generate_pipeline(request)
        
        @self.app.post("/matrix/create")
        async def create_matrix(
            languages: List[Language],
            platform: CICDPlatform,
            config: Optional[MatrixConfig] = None
        ):
            """Create CI/CD matrix"""
            return await self.create_matrix(languages, platform, config)
        
        @self.app.get("/pipeline/{pipeline_id}/status")
        async def get_status(pipeline_id: str):
            """Get pipeline status"""
            return await self.get_pipeline_status(pipeline_id)
        
        @self.app.post("/docker-compose/generate")
        async def generate_docker_compose(languages: List[Language]):
            """Generate Docker Compose configuration"""
            config = await self.generate_docker_compose(languages)
            return {
                "config": config,
                "filename": "docker-compose.yml"
            }
        
        @self.app.post("/pipeline/validate")
        async def validate_pipeline(platform: CICDPlatform, config: str):
            """Validate pipeline configuration"""
            return await self.validate_pipeline(platform, config)
        
        @self.app.get("/languages")
        async def list_languages():
            """List supported languages"""
            return {
                "languages": [lang.value for lang in Language],
                "configs": self.language_configs
            }
        
        @self.app.get("/platforms")
        async def list_platforms():
            """List supported CI/CD platforms"""
            return {
                "platforms": [platform.value for platform in CICDPlatform],
                "configs": self.platform_configs
            }
        
        @self.app.get("/templates/{platform}")
        async def get_template(platform: CICDPlatform):
            """Get pipeline template"""
            template = self.template_env.get_template(platform.value)
            return {
                "platform": platform.value,
                "template": template.source,
                "config_file": self.platform_configs[platform]["config_file"]
            }
    
    async def startup(self):
        """Startup tasks"""
        logger.info("CI/CD Matrix Service starting up...")
        logger.info(f"Loaded {len(self.language_configs)} language configurations")
        logger.info(f"Loaded {len(self.platform_configs)} platform configurations")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("CI/CD Matrix Service shutting down...")
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = CICDMatrixService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("CICD_MATRIX_PORT", 8021))
    logger.info(f"Starting CI/CD Matrix Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()