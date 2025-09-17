#!/usr/bin/env python3
"""
Test script to create a project configuration with proper SSH settings
"""

from project_config import (
    ProjectConfigManager, ProjectConfig, DeploymentTarget,
    DirectoryStructure, Documentation, TechStack
)

# Create project manager
manager = ProjectConfigManager()

# Create a test project with correct SSH key path
test_project = ProjectConfig(
    project_name="Ubuntu Server Project",
    description="Test project for Ubuntu server deployment",
    
    deployment_targets=[
        DeploymentTarget(
            name="Local Development",
            type="local",
            local_path="/Users/jankootstra/agent-lightning-main",
            is_default=False,
            description="Local development environment"
        ),
        DeploymentTarget(
            name="Ubuntu Production Server",
            type="ubuntu_server",
            server_ip="13.38.102.28",
            username="ubuntu",
            ssh_key_path="blockchain.pem",  # This will be found in current directory
            server_directory="/home/ubuntu/agent-app",
            is_default=True,
            description="Production Ubuntu server on AWS"
        )
    ],
    
    directory_structure=DirectoryStructure(
        root_path="/home/ubuntu/agent-app",
        frontend_path="frontend/",
        backend_path="backend/",
        database_path="database/",
        docs_path="docs/",
        tests_path="tests/"
    ),
    
    tech_stack=TechStack(
        languages=["Python", "JavaScript", "TypeScript"],
        frameworks=["FastAPI", "React", "Node.js"],
        databases=["PostgreSQL", "Redis"],
        tools=["Docker", "Kubernetes"],
        cloud_services=["AWS EC2", "AWS S3"],
        package_manager="npm",
        runtime_versions={
            "python": "3.11",
            "node": "18.0"
        }
    ),
    
    preferred_agents=["full_stack_developer", "devops_engineer", "system_architect"],
    
    tags=["production", "ubuntu", "aws"]
)

# Try to create the project
if manager.create_project(test_project):
    print(f"✅ Created project: {test_project.project_name}")
    manager.set_active_project(test_project.project_name)
    print(f"⭐ Set as active project")
else:
    # Project might already exist, try updating
    if manager.update_project(test_project.project_name, test_project):
        print(f"✅ Updated existing project: {test_project.project_name}")
        manager.set_active_project(test_project.project_name)
        print(f"⭐ Set as active project")
    else:
        print(f"❌ Failed to create or update project")

# List all projects
print("\n📁 Available projects:")
for project_name in manager.list_projects():
    print(f"  - {project_name}")

# Show default deployment
default_target = manager.get_default_deployment()
if default_target:
    print(f"\n🎯 Default deployment: {default_target.name}")
    print(f"  Type: {default_target.type}")
    if default_target.type == "ubuntu_server":
        print(f"  Server: {default_target.server_ip}")
        print(f"  SSH Key: {default_target.ssh_key_path}")
        print(f"  Directory: {default_target.server_directory}")