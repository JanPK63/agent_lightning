#!/usr/bin/env python3
"""
Quick script to update the local development path for the project
"""

from project_config import ProjectConfigManager

# Create project manager
manager = ProjectConfigManager()

# Get the current project
project = manager.get_project("Ubuntu Server Project")

if project:
    print(f"Current project: {project.project_name}")
    
    # Find and update the local development target
    for target in project.deployment_targets:
        if target.type == "local" and target.name == "Local Development":
            print(f"Current local path: {target.local_path}")
            
            # Update the path
            target.local_path = "/Users/jankootstra/agent-lightning-main"
            
            print(f"New local path: {target.local_path}")
            
            # Save the changes
            if manager.update_project(project.project_name, project):
                print("✅ Successfully updated local development path!")
            else:
                print("❌ Failed to update project")
            break
    else:
        print("Local Development target not found")
else:
    print("Project not found")