"""
Project configuration management interface for Agent Lightning Monitoring Dashboard
"""

import streamlit as st
from typing import Dict, List, Any, Optional

from .models import DashboardConfig


class ProjectConfigInterface:
    """Handles project configuration management"""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def render_project_config(self):
        """Render project configuration management interface"""
        st.header("⚙️ Project Configuration Management")

        from project_config import (
            ProjectConfigManager, ProjectConfig, DeploymentTarget,
            DirectoryStructure, Documentation, TechStack
        )

        # Initialize project manager
        manager = ProjectConfigManager()

        # Create two columns for layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📁 Projects")

            # List existing projects
            projects = manager.list_projects()

            # Active project indicator
            if manager.active_project:
                st.info(f"⭐ Active: {manager.active_project}")

            # Project selector
            if projects:
                selected_project = st.selectbox(
                    "Select Project",
                    options=["<Create New>"] + projects,
                    index=0 if not manager.active_project else projects.index(manager.active_project) + 1 if manager.active_project in projects else 0
                )
            else:
                selected_project = "<Create New>"
                st.info("No projects configured yet")

            # Set active project button
            if selected_project != "<Create New>" and selected_project != manager.active_project:
                if st.button("Set as Active", type="primary"):
                    manager.set_active_project(selected_project)
                    st.success(f"✅ Set {selected_project} as active project")
                    st.rerun()

            # Delete project button
            if selected_project != "<Create New>":
                if st.button("🗑️ Delete Project", type="secondary"):
                    if manager.delete_project(selected_project):
                        st.success(f"Deleted project: {selected_project}")
                        st.rerun()

        with col2:
            if selected_project == "<Create New>":
                self._render_create_project(manager)
            else:
                self._render_edit_project(manager, selected_project)

    def _render_create_project(self, manager):
        """Render create new project form"""
        st.subheader("Create New Project")

        with st.form("new_project_form"):
            project_name = st.text_input("Project Name", placeholder="My Blockchain Project")
            description = st.text_area("Description", placeholder="Multi-chain blockchain platform...")

            st.markdown("### 🎯 Deployment Targets")

            # Default local deployment
            local_path = st.text_input(
                "Local Project Path",
                placeholder="/Users/yourname/project",
                help="Path to your local project directory"
            )

            # Optional remote deployment
            st.markdown("**Remote Server (Optional)**")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                server_ip = st.text_input("Server IP", placeholder="13.38.102.28")
                ssh_user = st.text_input("SSH Username", value="ubuntu")
            with col_r2:
                ssh_key = st.text_input("SSH Key Path", placeholder="~/blockchain.pem")
                server_dir = st.text_input("Server Directory", placeholder="/home/ubuntu/project")

            st.markdown("### 📂 Directory Structure")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                frontend_path = st.text_input("Frontend", placeholder="frontend/")
                backend_path = st.text_input("Backend", placeholder="backend/")
                database_path = st.text_input("Database", placeholder="database/")
            with col_d2:
                blockchain_path = st.text_input("Blockchain", placeholder="blockchain/")
                docs_path = st.text_input("Documentation", placeholder="docs/")
                tests_path = st.text_input("Tests", placeholder="tests/")

            st.markdown("### 🛠️ Tech Stack")
            languages = st.text_input("Languages (comma-separated)", placeholder="Python, JavaScript, Solidity")
            frameworks = st.text_input("Frameworks (comma-separated)", placeholder="React, FastAPI, Hyperledger")

            # Submit button
            if st.form_submit_button("Create Project", type="primary"):
                if project_name and description:
                    # Create deployment targets
                    targets = []
                    if local_path:
                        targets.append(DeploymentTarget(
                            name="Local Development",
                            type="local",
                            local_path=local_path,
                            is_default=True
                        ))

                    if server_ip:
                        targets.append(DeploymentTarget(
                            name="Remote Server",
                            type="ubuntu_server",
                            server_ip=server_ip,
                            username=ssh_user,
                            ssh_key_path=ssh_key,
                            server_directory=server_dir
                        ))

                    # Create directory structure
                    dir_struct = DirectoryStructure(
                        root_path=local_path or "/",
                        frontend_path=frontend_path or None,
                        backend_path=backend_path or None,
                        database_path=database_path or None,
                        blockchain_path=blockchain_path or None,
                        docs_path=docs_path or None,
                        tests_path=tests_path or None
                    )

                    # Create tech stack
                    tech = TechStack(
                        languages=[l.strip() for l in languages.split(",")] if languages else [],
                        frameworks=[f.strip() for f in frameworks.split(",")] if frameworks else []
                    )

                    # Create project config
                    config = ProjectConfig(
                        project_name=project_name,
                        description=description,
                        deployment_targets=targets,
                        directory_structure=dir_struct,
                        tech_stack=tech
                    )

                    if manager.create_project(config):
                        st.success(f"✅ Created project: {project_name}")
                        st.rerun()
                    else:
                        st.error("Project with this name already exists")
                else:
                    st.error("Please fill in project name and description")

    def _render_edit_project(self, manager, selected_project):
        """Render edit existing project interface"""
        project = manager.get_project(selected_project)
        if not project:
            st.error(f"Project {selected_project} not found")
            return

        st.subheader(f"📋 {project.project_name}")
        st.text(project.description)

        # Show deployment targets
        st.markdown("### 🎯 Deployment Targets")

        # Add new deployment target button
        if st.button("➕ Add New Deployment Target", key=f"add_target_{selected_project}"):
            if f"adding_target_{selected_project}" not in st.session_state:
                st.session_state[f"adding_target_{selected_project}"] = True
            else:
                st.session_state[f"adding_target_{selected_project}"] = True

        # Add new target form
        if st.session_state.get(f"adding_target_{selected_project}", False):
            self._render_add_deployment_target(manager, project, selected_project)

        # Display existing targets
        if project.deployment_targets:
            for i, target in enumerate(project.deployment_targets):
                self._render_deployment_target(manager, project, target, i, selected_project)

        # Show directory structure
        if project.directory_structure:
            st.markdown("### 📂 Directory Structure")
            dirs = project.directory_structure
            if dirs.frontend_path:
                st.text(f"Frontend: {dirs.frontend_path}")
            if dirs.backend_path:
                st.text(f"Backend: {dirs.backend_path}")
            if dirs.blockchain_path:
                st.text(f"Blockchain: {dirs.blockchain_path}")

        # Show tech stack
        if project.tech_stack:
            st.markdown("### 🛠️ Tech Stack")
            if project.tech_stack.languages:
                st.text(f"Languages: {', '.join(project.tech_stack.languages)}")
            if project.tech_stack.frameworks:
                st.text(f"Frameworks: {', '.join(project.tech_stack.frameworks)}")

        # Database Configuration Section
        st.markdown("### 🗄️ Database Configuration")
        self._render_database_config(manager, project, selected_project)

        # Quick deployment section
        st.markdown("### 🚀 Quick Deploy")
        default_target = manager.get_default_deployment(selected_project)
        if default_target:
            st.success(f"Ready to deploy to: {default_target.name}")
            if st.button("Use in Task Assignment"):
                # Store in session state for task assignment
                st.session_state.selected_project = selected_project
                if default_target.type == "ubuntu_server":
                    st.session_state.deployment_config = {
                        "type": default_target.type,
                        "server_ip": default_target.server_ip,
                        "username": default_target.username or "ubuntu",
                        "key_path": default_target.ssh_key_path,
                        "working_directory": default_target.server_directory or "/home/ubuntu"
                    }
                elif default_target.type == "local":
                    st.session_state.deployment_config = {
                        "type": default_target.type,
                        "path": default_target.local_path
                    }
                elif default_target.type == "aws_ec2":
                    st.session_state.deployment_config = {
                        "type": default_target.type,
                        "region": default_target.aws_region,
                        "instance_type": default_target.instance_type,
                        "key_name": default_target.aws_key_name
                    }
                else:
                    st.session_state.deployment_config = {
                        "type": default_target.type
                    }
                st.info("✅ Project config loaded for Task Assignment")
        else:
            st.warning("No deployment targets configured")

    def _render_add_deployment_target(self, manager, project, selected_project):
        """Render add new deployment target form"""
        with st.form(f"new_target_form_{selected_project}"):
            st.subheader("Add New Deployment Target")
            target_name = st.text_input("Target Name", placeholder="e.g., Production Server")
            target_type = st.selectbox("Type", ["local", "ubuntu_server", "aws_ec2"])

            if target_type == "local":
                local_path = st.text_input("Local Path", placeholder="/Users/yourname/project")
            elif target_type == "ubuntu_server":
                server_ip = st.text_input("Server IP", placeholder="13.38.102.28")
                username = st.text_input("Username", value="ubuntu")
                ssh_key = st.text_input("SSH Key Path", placeholder="~/keys/server.pem")
                server_dir = st.text_input("Server Directory", value="/home/ubuntu")

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("➕ Add Target"):
                    if target_name:
                        new_target = DeploymentTarget(
                            name=target_name,
                            type=target_type,
                            is_default=len(project.deployment_targets) == 0
                        )

                        if target_type == "local":
                            new_target.local_path = local_path
                        elif target_type == "ubuntu_server":
                            new_target.server_ip = server_ip
                            new_target.username = username
                            new_target.ssh_key_path = ssh_key
                            new_target.server_directory = server_dir

                        project.deployment_targets.append(new_target)
                        manager.update_project(selected_project, project)
                        st.session_state[f"adding_target_{selected_project}"] = False
                        st.success(f"✅ Added deployment target: {target_name}")
                        st.rerun()
                    else:
                        st.error("Please provide a target name")
            with col2:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state[f"adding_target_{selected_project}"] = False
                    st.rerun()

    def _render_deployment_target(self, manager, project, target, index, selected_project):
        """Render individual deployment target"""
        with st.expander(f"{target.name} ({target.type})" + (" ⭐" if target.is_default else "")):
            # Create edit form for this target
            edit_key = f"edit_{selected_project}_{index}"

            if f"editing_{edit_key}" not in st.session_state:
                st.session_state[f"editing_{edit_key}"] = False

            if st.session_state[f"editing_{edit_key}"]:
                # Edit mode
                with st.form(f"edit_form_{edit_key}"):
                    new_name = st.text_input("Name", value=target.name)

                    if target.type == "local":
                        new_path = st.text_input("Local Path", value=target.local_path or "")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("💾 Save"):
                                target.name = new_name
                                target.local_path = new_path
                                manager.update_project(selected_project, project)
                                st.session_state[f"editing_{edit_key}"] = False
                                st.success("✅ Updated deployment target")
                                st.rerun()
                        with col2:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state[f"editing_{edit_key}"] = False
                                st.rerun()

                    elif target.type == "ubuntu_server":
                        new_server_ip = st.text_input("Server IP", value=target.server_ip or "")
                        new_username = st.text_input("Username", value=target.username or "ubuntu")
                        new_ssh_key = st.text_input("SSH Key Path", value=target.ssh_key_path or "")
                        new_server_dir = st.text_input("Server Directory", value=target.server_directory or "/home/ubuntu")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("💾 Save"):
                                target.name = new_name
                                target.server_ip = new_server_ip
                                target.username = new_username
                                target.ssh_key_path = new_ssh_key
                                target.server_directory = new_server_dir
                                manager.update_project(selected_project, project)
                                st.session_state[f"editing_{edit_key}"] = False
                                st.success("✅ Updated deployment target")
                                st.rerun()
                        with col2:
                            if st.form_submit_button("❌ Cancel"):
                                st.session_state[f"editing_{edit_key}"] = False
                                st.rerun()
            else:
                # View mode
                if target.type == "local":
                    st.text(f"Path: {target.local_path}")
                elif target.type == "ubuntu_server":
                    st.text(f"Server: {target.username}@{target.server_ip}")
                    st.text(f"Key: {target.ssh_key_path}")
                    st.text(f"Directory: {target.server_directory}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✏️ Edit", key=f"edit_btn_{edit_key}"):
                        st.session_state[f"editing_{edit_key}"] = True
                        st.rerun()
                with col2:
                    if not target.is_default:
                        if st.button("⭐ Set Default", key=f"default_{edit_key}"):
                            # Set as default
                            for t in project.deployment_targets:
                                t.is_default = (t.name == target.name)
                            manager.update_project(selected_project, project)
                            st.success(f"Set {target.name} as default")
                            st.rerun()
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{edit_key}"):
                        if len(project.deployment_targets) > 1:
                            project.deployment_targets.remove(target)
                            # If deleted was default, set first as default
                            if target.is_default and project.deployment_targets:
                                project.deployment_targets[0].is_default = True
                            manager.update_project(selected_project, project)
                            st.success(f"Deleted {target.name}")
                            st.rerun()
                        else:
                            st.error("Cannot delete last deployment target")

    def _render_database_config(self, manager, project, selected_project):
        """Render database configuration section"""
        # Add database configuration button
        if st.button("⚙️ Configure Database", key=f"config_db_{selected_project}"):
            if f"configuring_db_{selected_project}" not in st.session_state:
                st.session_state[f"configuring_db_{selected_project}"] = True
            else:
                st.session_state[f"configuring_db_{selected_project}"] = True

        # Database configuration form
        if st.session_state.get(f"configuring_db_{selected_project}", False):
            with st.form(f"db_config_form_{selected_project}"):
                st.subheader("Database Settings")

                # Database type selection
                db_type = st.selectbox(
                    "Database Type",
                    options=["sqlite", "postgresql", "mongodb"],
                    index=0,
                    help="Choose the database backend for this project"
                )

                if db_type == "sqlite":
                    db_path = st.text_input(
                        "Database File Path",
                        value="./data/agent_lightning.db",
                        help="Path to SQLite database file"
                    )
                    connection_string = f"sqlite:///{db_path}"

                elif db_type == "postgresql":
                    col_pg1, col_pg2 = st.columns(2)
                    with col_pg1:
                        pg_host = st.text_input("Host", value="localhost")
                        pg_port = st.text_input("Port", value="5432")
                        pg_database = st.text_input("Database", value="agent_lightning")
                    with col_pg2:
                        pg_user = st.text_input("Username", value="postgres")
                        pg_password = st.text_input("Password", type="password")
                        pg_ssl = st.checkbox("SSL Connection", value=False)

                    connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
                    if pg_ssl:
                        connection_string += "?sslmode=require"

                elif db_type == "mongodb":
                    col_mongo1, col_mongo2 = st.columns(2)
                    with col_mongo1:
                        mongo_host = st.text_input("Host", value="localhost")
                        mongo_port = st.text_input("Port", value="27017")
                        mongo_database = st.text_input("Database", value="agent_lightning")
                    with col_mongo2:
                        mongo_user = st.text_input("Username (optional)")
                        mongo_password = st.text_input("Password (optional)", type="password")
                        mongo_auth_db = st.text_input("Auth Database", value="admin", help="Database for authentication")

                    # Build MongoDB connection string
                    if mongo_user and mongo_password:
                        connection_string = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_database}?authSource={mongo_auth_db}"
                    else:
                        connection_string = f"mongodb://{mongo_host}:{mongo_port}/{mongo_database}"

                    st.info("💡 MongoDB supports document storage for flexible data models and is ideal for agent memory and knowledge bases.")

                # Connection string preview
                st.markdown("**Connection String Preview:**")
                st.code(connection_string, language="text")

                # Test connection button
                test_connection = st.checkbox("Test connection on save", value=True)

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.form_submit_button("💾 Save Database Config"):
                        # Store database configuration in project
                        if not hasattr(project, 'database_config'):
                            project.database_config = {}

                        project.database_config = {
                            "type": db_type,
                            "connection_string": connection_string,
                            "host": locals().get(f"{db_type}_host", mongo_host if db_type == "mongodb" else ""),
                            "port": locals().get(f"{db_type}_port", mongo_port if db_type == "mongodb" else ""),
                            "database": locals().get(f"{db_type}_database", mongo_database if db_type == "mongodb" else ""),
                            "username": locals().get(f"{db_type}_user", mongo_user if db_type == "mongodb" else ""),
                            "password": locals().get(f"{db_type}_password", mongo_password if db_type == "mongodb" else ""),
                            "ssl": locals().get(f"{db_type}_ssl", False),
                            "auth_database": locals().get(f"{db_type}_auth_db", mongo_auth_db if db_type == "mongodb" else "")
                        }

                        # Test connection if requested
                        if test_connection:
                            try:
                                if db_type == "mongodb":
                                    import pymongo
                                    client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                                    client.admin.command('ping')
                                    st.success("✅ MongoDB connection successful!")
                                elif db_type == "postgresql":
                                    import psycopg2
                                    conn = psycopg2.connect(connection_string)
                                    conn.close()
                                    st.success("✅ PostgreSQL connection successful!")
                                else:
                                    # SQLite doesn't need connection test
                                    st.success("✅ SQLite configuration saved!")
                            except Exception as e:
                                st.error(f"❌ Connection test failed: {str(e)}")
                                st.warning("Configuration saved but connection failed. Please check your settings.")

                        manager.update_project(selected_project, project)
                        st.session_state[f"configuring_db_{selected_project}"] = False
                        st.success("✅ Database configuration saved!")
                        st.rerun()

                with col_cancel:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state[f"configuring_db_{selected_project}"] = False
                        st.rerun()

        # Display current database configuration
        if hasattr(project, 'database_config') and project.database_config:
            db_config = project.database_config
            st.markdown(f"**Current Database:** {db_config.get('type', 'Not configured').title()}")

            if db_config.get('type') == 'mongodb':
                st.info("🗄️ MongoDB is configured for document storage and flexible data models.")
                with st.expander("Database Details"):
                    st.text(f"Host: {db_config.get('host', 'N/A')}")
                    st.text(f"Port: {db_config.get('port', 'N/A')}")
                    st.text(f"Database: {db_config.get('database', 'N/A')}")
                    if db_config.get('username'):
                        st.text(f"Username: {db_config['username']}")
                    st.text("Connection String: [hidden for security]")
            elif db_config.get('type') == 'postgresql':
                st.info("🐘 PostgreSQL is configured for relational data storage.")
            elif db_config.get('type') == 'sqlite':
                st.info("📁 SQLite is configured for local file-based storage.")
        else:
            st.info("No database configured yet. Click 'Configure Database' to set up.")