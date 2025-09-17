# Minimal project config for containerized dashboard
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import os

class DeploymentType(Enum):
    LOCAL = "local"
    UBUNTU_SERVER = "ubuntu_server"
    AWS_EC2 = "aws_ec2"

@dataclass
class DeploymentTarget:
    name: str
    type: str
    is_default: bool = False
    local_path: Optional[str] = None
    server_ip: Optional[str] = None
    username: Optional[str] = None
    ssh_key_path: Optional[str] = None
    server_directory: Optional[str] = None
    aws_region: Optional[str] = None
    instance_type: Optional[str] = None
    aws_key_name: Optional[str] = None

@dataclass
class DirectoryStructure:
    root_path: str
    frontend_path: Optional[str] = None
    backend_path: Optional[str] = None
    database_path: Optional[str] = None
    blockchain_path: Optional[str] = None
    docs_path: Optional[str] = None
    tests_path: Optional[str] = None

@dataclass
class TechStack:
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)

@dataclass
class Documentation:
    readme_path: Optional[str] = None
    api_docs_path: Optional[str] = None

@dataclass
class ProjectConfig:
    project_name: str
    description: str
    deployment_targets: List[DeploymentTarget] = field(default_factory=list)
    directory_structure: Optional[DirectoryStructure] = None
    tech_stack: Optional[TechStack] = None
    documentation: Optional[Documentation] = None

class ProjectConfigManager:
    def __init__(self):
        self.config_dir = "/tmp/agent_projects"
        self.active_project = None
        os.makedirs(self.config_dir, exist_ok=True)
    
    def create_project(self, config: ProjectConfig) -> bool:
        """Create a new project"""
        return True
    
    def list_projects(self) -> List[str]:
        """List all projects"""
        return []
    
    def get_project(self, name: str) -> Optional[ProjectConfig]:
        """Get project by name"""
        return None
    
    def update_project(self, name: str, config: ProjectConfig) -> bool:
        """Update project"""
        return True
    
    def delete_project(self, name: str) -> bool:
        """Delete project"""
        return True
    
    def set_active_project(self, name: str):
        """Set active project"""
        self.active_project = name
    
    def get_active_project(self) -> Optional[ProjectConfig]:
        """Get active project config"""
        return None
    
    def get_default_deployment(self, project_name: str = None) -> Optional[DeploymentTarget]:
        """Get default deployment target"""
        return None