#!/usr/bin/env python3
"""
Project Configuration Management System
Manages project configurations for Agent Lightning including deployment targets,
directory structures, and documentation locations
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


@dataclass
class DeploymentTarget:
    """Configuration for a deployment target"""
    name: str
    type: str  # local, ubuntu_server, aws_ec2
    
    # Local config
    local_path: Optional[str] = None
    
    # Ubuntu/SSH config
    server_ip: Optional[str] = None
    username: Optional[str] = "ubuntu"
    ssh_key_path: Optional[str] = None
    server_directory: Optional[str] = None
    
    # AWS config
    aws_region: Optional[str] = None
    instance_type: Optional[str] = None
    aws_key_name: Optional[str] = None
    
    # Common
    is_default: bool = False
    description: Optional[str] = None


@dataclass
class DirectoryStructure:
    """Project directory structure mapping"""
    root_path: str
    frontend_path: Optional[str] = None
    backend_path: Optional[str] = None
    database_path: Optional[str] = None
    docs_path: Optional[str] = None
    tests_path: Optional[str] = None
    config_path: Optional[str] = None
    blockchain_path: Optional[str] = None
    
    # File patterns
    source_extensions: List[str] = field(default_factory=lambda: ['.py', '.js', '.ts', '.java', '.go'])
    ignore_patterns: List[str] = field(default_factory=lambda: ['node_modules', '__pycache__', '.git'])


@dataclass
class Documentation:
    """Documentation locations and resources"""
    readme_path: Optional[str] = None
    api_docs_path: Optional[str] = None
    architecture_docs_path: Optional[str] = None
    requirements_path: Optional[str] = None
    
    # External resources
    wiki_url: Optional[str] = None
    confluence_url: Optional[str] = None
    github_repo: Optional[str] = None
    
    # Custom docs
    custom_docs: Dict[str, str] = field(default_factory=dict)


@dataclass
class TechStack:
    """Project technology stack information"""
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    cloud_services: List[str] = field(default_factory=list)
    
    # Package managers
    package_manager: Optional[str] = None  # npm, pip, maven, etc.
    
    # Versions
    runtime_versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    """Complete project configuration"""
    project_name: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Sub-configurations
    deployment_targets: List[DeploymentTarget] = field(default_factory=list)
    directory_structure: Optional[DirectoryStructure] = None
    documentation: Optional[Documentation] = None
    tech_stack: Optional[TechStack] = None
    
    # Project metadata
    team_members: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Agent preferences
    preferred_agents: List[str] = field(default_factory=list)
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class ProjectConfigManager:
    """Manages project configurations"""
    
    def __init__(self, config_dir: str = None):
        """Initialize the config manager"""
        if config_dir is None:
            config_dir = os.path.expanduser("~/.agent-lightning/projects")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Current active project
        self.active_project: Optional[str] = None
        self._load_active_project()
    
    def _get_config_path(self, project_name: str) -> Path:
        """Get the configuration file path for a project"""
        # Clean project name for filesystem
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        return self.config_dir / f"{safe_name}.json"
    
    def _load_active_project(self):
        """Load the active project name"""
        active_file = self.config_dir / ".active_project"
        if active_file.exists():
            self.active_project = active_file.read_text().strip()
    
    def _save_active_project(self):
        """Save the active project name"""
        active_file = self.config_dir / ".active_project"
        if self.active_project:
            active_file.write_text(self.active_project)
        elif active_file.exists():
            active_file.unlink()
    
    def create_project(self, config: ProjectConfig) -> bool:
        """Create a new project configuration"""
        try:
            config_path = self._get_config_path(config.project_name)
            
            # Check if project already exists
            if config_path.exists():
                return False
            
            # Save configuration
            config.created_at = datetime.now().isoformat()
            config.updated_at = config.created_at
            
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            # Set as active if no active project
            if not self.active_project:
                self.active_project = config.project_name
                self._save_active_project()
            
            return True
            
        except Exception as e:
            print(f"Error creating project: {e}")
            return False
    
    def update_project(self, project_name: str, config: ProjectConfig) -> bool:
        """Update an existing project configuration"""
        try:
            config_path = self._get_config_path(project_name)
            
            if not config_path.exists():
                return False
            
            # Update timestamp
            config.updated_at = datetime.now().isoformat()
            
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error updating project: {e}")
            return False
    
    def get_project(self, project_name: str = None) -> Optional[ProjectConfig]:
        """Get a project configuration"""
        try:
            # Use active project if no name specified
            if project_name is None:
                project_name = self.active_project
            
            if not project_name:
                return None
            
            config_path = self._get_config_path(project_name)
            
            if not config_path.exists():
                return None
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct the configuration
            config = ProjectConfig(**data)
            
            # Reconstruct nested dataclasses
            if config.deployment_targets:
                config.deployment_targets = [DeploymentTarget(**dt) for dt in config.deployment_targets]
            
            if config.directory_structure:
                config.directory_structure = DirectoryStructure(**config.directory_structure)
            
            if config.documentation:
                config.documentation = Documentation(**config.documentation)
            
            if config.tech_stack:
                config.tech_stack = TechStack(**config.tech_stack)
            
            return config
            
        except Exception as e:
            print(f"Error loading project: {e}")
            return None
    
    def delete_project(self, project_name: str) -> bool:
        """Delete a project configuration"""
        try:
            config_path = self._get_config_path(project_name)
            
            if not config_path.exists():
                return False
            
            config_path.unlink()
            
            # Clear active project if it was deleted
            if self.active_project == project_name:
                self.active_project = None
                self._save_active_project()
            
            return True
            
        except Exception as e:
            print(f"Error deleting project: {e}")
            return False
    
    def list_projects(self) -> List[str]:
        """List all available projects"""
        projects = []
        for config_file in self.config_dir.glob("*.json"):
            if config_file.name != ".active_project":
                projects.append(config_file.stem.replace('_', ' '))
        return sorted(projects)
    
    def set_active_project(self, project_name: str) -> bool:
        """Set the active project"""
        if self.get_project(project_name):
            self.active_project = project_name
            self._save_active_project()
            return True
        return False
    
    def get_active_project(self) -> Optional[ProjectConfig]:
        """Get the active project configuration"""
        return self.get_project(self.active_project) if self.active_project else None
    
    def get_default_deployment(self, project_name: str = None) -> Optional[DeploymentTarget]:
        """Get the default deployment target for a project"""
        config = self.get_project(project_name)
        if config and config.deployment_targets:
            # Look for default
            for target in config.deployment_targets:
                if target.is_default:
                    return target
            # Return first if no default
            return config.deployment_targets[0]
        return None
    
    def add_deployment_target(self, project_name: str, target: DeploymentTarget) -> bool:
        """Add a deployment target to a project"""
        config = self.get_project(project_name)
        if config:
            # Check if name already exists
            for existing in config.deployment_targets:
                if existing.name == target.name:
                    return False
            
            config.deployment_targets.append(target)
            return self.update_project(project_name, config)
        return False
    
    def remove_deployment_target(self, project_name: str, target_name: str) -> bool:
        """Remove a deployment target from a project"""
        config = self.get_project(project_name)
        if config:
            config.deployment_targets = [
                t for t in config.deployment_targets if t.name != target_name
            ]
            return self.update_project(project_name, config)
        return False


# Example usage and templates
def create_example_project():
    """Create an example project configuration"""
    
    # Create project config
    project = ProjectConfig(
        project_name="Blockchain Platform",
        description="Multi-chain blockchain platform with frontend and backend",
        
        deployment_targets=[
            DeploymentTarget(
                name="Local Development",
                type="local",
                local_path="/Users/jankootstra/blockchain-project",
                is_default=True,
                description="Local development environment"
            ),
            DeploymentTarget(
                name="AWS Production",
                type="ubuntu_server",
                server_ip="13.38.102.28",
                username="ubuntu",
                ssh_key_path="~/blockchain.pem",
                server_directory="/home/ubuntu/blockchain",
                description="Production Ubuntu server on AWS"
            )
        ],
        
        directory_structure=DirectoryStructure(
            root_path="/home/ubuntu/blockchain",
            frontend_path="frontend/",
            backend_path="backend/",
            blockchain_path="blockchain/",
            database_path="database/",
            docs_path="docs/",
            tests_path="tests/"
        ),
        
        documentation=Documentation(
            readme_path="README.md",
            api_docs_path="docs/api.md",
            architecture_docs_path="docs/architecture.md",
            github_repo="https://github.com/username/blockchain-project"
        ),
        
        tech_stack=TechStack(
            languages=["Python", "JavaScript", "Solidity"],
            frameworks=["React", "FastAPI", "Hyperledger Fabric"],
            databases=["PostgreSQL", "Redis"],
            tools=["Docker", "Kubernetes"],
            cloud_services=["AWS EC2", "AWS S3"],
            package_manager="npm",
            runtime_versions={
                "python": "3.11",
                "node": "18.0",
                "go": "1.20"
            }
        ),
        
        preferred_agents=["system_architect", "blockchain_developer", "full_stack_developer"],
        
        tags=["blockchain", "production", "multi-chain"]
    )
    
    return project


if __name__ == "__main__":
    # Create config manager
    manager = ProjectConfigManager()
    
    # Create example project
    example = create_example_project()
    
    if manager.create_project(example):
        print(f"✅ Created project: {example.project_name}")
    
    # List projects
    print("\n📁 Available projects:")
    for project_name in manager.list_projects():
        print(f"  - {project_name}")
    
    # Get active project
    active = manager.get_active_project()
    if active:
        print(f"\n⭐ Active project: {active.project_name}")
        print(f"  Deployment targets: {len(active.deployment_targets)}")
        
        # Show default deployment
        default_target = manager.get_default_deployment()
        if default_target:
            print(f"  Default deployment: {default_target.name} ({default_target.type})")