#!/usr/bin/env python3
"""
Fix SSH key path in project configuration
"""

from project_config import ProjectConfigManager

# Create project manager
manager = ProjectConfigManager()

# Get the current project
project = manager.get_project("Ubuntu Server Project")

if project:
    print(f"Updating SSH key paths for: {project.project_name}")
    
    # Update all Ubuntu server targets with correct SSH key path
    for target in project.deployment_targets:
        if target.type == "ubuntu_server":
            old_path = target.ssh_key_path
            # Update to full path
            target.ssh_key_path = "/Users/jankootstra/blockchain.pem"
            print(f"  Updated {target.name}:")
            print(f"    Old path: {old_path}")
            print(f"    New path: {target.ssh_key_path}")
    
    # Save the changes
    if manager.update_project(project.project_name, project):
        print("✅ Successfully updated SSH key paths!")
    else:
        print("❌ Failed to update project")
else:
    print("Project not found")