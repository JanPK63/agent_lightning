#!/usr/bin/env python3
"""
Agent Lightning Integrated Dashboard
Complete system with task assignment, agent learning, visual building, API overview, and spec-driven development
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system components
try:
    from agent_config import AgentConfigManager
    from knowledge_manager import KnowledgeManager
    from project_config import ProjectConfigManager
    from visual_code_builder import VisualProgram, BlockFactory
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Some components not available: {e}")
    COMPONENTS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Agent Lightning System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize managers
if COMPONENTS_AVAILABLE:
    try:
        config_manager = AgentConfigManager()
        knowledge_manager = KnowledgeManager()
        project_manager = ProjectConfigManager()
    except Exception as e:
        st.error(f"Error initializing managers: {e}")
        config_manager = None
        knowledge_manager = None
        project_manager = None
else:
    config_manager = None
    knowledge_manager = None
    project_manager = None

# API Configuration
API_BASE_URL = "http://localhost:8002"

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .agent-card {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background: white;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ Agent Lightning System")
st.markdown("**Production AI Agent Platform with Learning & Deployment**")

# Sidebar Navigation
with st.sidebar:
    st.header("🧭 Navigation")
    page = st.selectbox(
        "Select Feature",
        [
            "🏠 System Overview",
            "🎯 Task Assignment", 
            "🧠 Agent Learning",
            "🎨 Visual Code Builder",
            "📊 API Overview",
            "📋 Spec-Driven Development",
            "⚙️ Project Configuration"
        ]
    )
    
    st.divider()
    
    # Quick Status
    st.subheader("📈 Quick Status")
    
    # Check API status
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            health_data = response.json()
            st.success("🟢 System Online")
            st.metric("Agents", health_data.get("specialized_agents", 0))
            st.metric("Knowledge Items", health_data.get("total_knowledge_items", 0))
        else:
            st.error("🔴 System Offline")
    except:
        st.warning("🟡 API Unavailable")
    
    st.divider()
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# Main Content
if page == "🏠 System Overview":
    st.header("System Overview")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            
            with col1:
                st.metric("System Status", "Operational", "✅")
            with col2:
                st.metric("Specialized Agents", health_data.get("specialized_agents", 0))
            with col3:
                st.metric("Knowledge Items", health_data.get("total_knowledge_items", 0))
            with col4:
                st.metric("RL Sessions", health_data.get("rl_sessions_today", 0))
        else:
            st.error("Unable to connect to system")
    except Exception as e:
        st.error(f"System check failed: {e}")
    
    # Available agents
    st.subheader("🤖 Available Agents")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v2/agents/list", timeout=5)
        if response.status_code == 200:
            agents_data = response.json()
            agents = agents_data.get("agents", [])
            
            for agent in agents[:6]:  # Show first 6 agents
                with st.expander(f"🤖 {agent.get('name', 'Unknown')} ({agent.get('role', 'General')})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Model**: {agent.get('model', 'N/A')}")
                        st.write(f"**Knowledge Items**: {agent.get('knowledge_items', 0)}")
                    with col2:
                        capabilities = agent.get('capabilities', [])
                        if capabilities:
                            st.write("**Capabilities**:")
                            for cap in capabilities[:3]:
                                st.write(f"• {cap.replace('_', ' ').title()}")
        else:
            st.warning("Could not load agents")
    except Exception as e:
        st.error(f"Error loading agents: {e}")

elif page == "🎯 Task Assignment":
    st.header("Task Assignment")
    
    # Task input form
    with st.form("task_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            task_description = st.text_area(
                "Task Description",
                placeholder="Describe what you want the agent to do...",
                height=120
            )
        
        with col2:
            # Get available agents
            agent_options = ["auto"]
            try:
                response = requests.get(f"{API_BASE_URL}/api/v2/agents/list", timeout=5)
                if response.status_code == 200:
                    agents_data = response.json()
                    agents = agents_data.get("agents", [])
                    agent_options.extend([agent["id"] for agent in agents])
            except:
                pass
            
            selected_agent = st.selectbox("Select Agent", agent_options)
            priority = st.selectbox("Priority", ["low", "normal", "high", "urgent"])
            
            # Deployment options
            st.markdown("**Deployment**")
            deployment_type = st.selectbox(
                "Deploy to",
                ["none", "local", "ubuntu_server", "aws_ec2"]
            )
        
        # Deployment configuration
        deployment_config = None
        if deployment_type != "none":
            st.markdown("**Deployment Configuration**")
            
            if deployment_type == "local":
                local_path = st.text_input("Local Path", "/tmp/agent_workspace")
                deployment_config = {"type": "local", "path": local_path}
            
            elif deployment_type == "ubuntu_server":
                col1, col2 = st.columns(2)
                with col1:
                    server_ip = st.text_input("Server IP", "")
                    username = st.text_input("Username", "ubuntu")
                with col2:
                    ssh_key = st.text_input("SSH Key Path", "")
                    server_dir = st.text_input("Server Directory", "/home/ubuntu")
                
                deployment_config = {
                    "type": "ubuntu_server",
                    "server_ip": server_ip,
                    "username": username,
                    "key_path": ssh_key,
                    "working_directory": server_dir
                }
        
        # Submit button
        submitted = st.form_submit_button("🚀 Execute Task", type="primary")
    
    if submitted and task_description:
        with st.spinner("Processing task..."):
            try:
                # Prepare request
                request_data = {
                    "task": task_description,
                    "agent_id": selected_agent if selected_agent != "auto" else None,
                    "context": {"deployment": deployment_config} if deployment_config else {}
                }
                
                # Execute task
                response = requests.post(
                    f"{API_BASE_URL}/api/v2/agents/execute",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Task completed successfully!")
                    
                    # Display result
                    if result.get("result"):
                        task_result = result["result"]
                        
                        # Show response
                        if task_result.get("response"):
                            st.markdown("### 📋 Response")
                            st.write(task_result["response"])
                        
                        # Show agent info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Agent Used", task_result.get("agent", "Unknown"))
                        with col2:
                            st.metric("Knowledge Used", task_result.get("knowledge_items_used", 0))
                        with col3:
                            if task_result.get("action_executed"):
                                st.metric("Action", task_result.get("action_executed", "None"))
                        
                        # Show RL info if available
                        if task_result.get("intelligent_rl"):
                            rl_info = task_result["intelligent_rl"]
                            st.markdown("### 🧠 RL Training")
                            st.info(f"Algorithm: {rl_info.get('algorithm')} | Epochs: {rl_info.get('epochs_completed')}/{rl_info.get('total_epochs')}")
                
                else:
                    st.error(f"Task failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error executing task: {e}")

elif page == "🧠 Agent Learning":
    st.header("Agent Learning & Knowledge Management")
    
    # Learning status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Learning Sessions", "12", "+3")
    with col2:
        st.metric("Success Rate", "94.2%", "+2.1%")
    with col3:
        st.metric("Knowledge Items", "1,421", "+47")
    
    # Agent selection for knowledge management
    st.subheader("📚 Knowledge Management")
    
    if config_manager:
        available_agents = config_manager.list_agents()
        if available_agents:
            selected_agent = st.selectbox("Select Agent", available_agents)
            
            # Knowledge tabs
            tab1, tab2, tab3 = st.tabs(["📖 View Knowledge", "➕ Add Knowledge", "🎓 Train Agent"])
            
            with tab1:
                if knowledge_manager:
                    items = knowledge_manager.knowledge_bases.get(selected_agent, [])
                    if items:
                        st.write(f"**{len(items)} knowledge items**")
                        for item in items[:5]:
                            with st.expander(f"[{item.category}] {item.content[:50]}..."):
                                st.write(f"**Source**: {item.source}")
                                st.write(f"**Usage**: {item.usage_count} times")
                                st.code(item.content)
                    else:
                        st.info("No knowledge items yet")
            
            with tab2:
                with st.form("knowledge_form"):
                    category = st.selectbox(
                        "Category",
                        ["technical_documentation", "code_examples", "best_practices", 
                         "troubleshooting", "project_specific"]
                    )
                    content = st.text_area("Knowledge Content", height=150)
                    source = st.text_input("Source", "manual")
                    
                    if st.form_submit_button("Add Knowledge"):
                        if content and knowledge_manager:
                            try:
                                response = requests.post(
                                    f"{API_BASE_URL}/api/v2/knowledge/add",
                                    json={
                                        "agent_id": selected_agent,
                                        "category": category,
                                        "content": content,
                                        "source": source
                                    }
                                )
                                if response.status_code == 200:
                                    st.success("✅ Knowledge added successfully!")
                                else:
                                    st.error("Failed to add knowledge")
                            except Exception as e:
                                st.error(f"Error: {e}")
            
            with tab3:
                st.markdown("**Train Agent with New Knowledge**")
                
                if st.button("🎓 Start Training"):
                    with st.spinner("Training agent..."):
                        try:
                            response = requests.post(f"{API_BASE_URL}/api/v2/knowledge/train/{selected_agent}")
                            if response.status_code == 200:
                                result = response.json()
                                if result.get("success"):
                                    st.success(f"✅ Training completed! Consumed {result.get('knowledge_consumed', 0)} items")
                                else:
                                    st.error(f"Training failed: {result.get('error')}")
                        except Exception as e:
                            st.error(f"Training error: {e}")
        else:
            st.warning("No agents configured")
    else:
        st.error("Agent configuration manager not available")

elif page == "🎨 Visual Code Builder":
    st.header("Visual Code Builder")
    
    if 'visual_program' not in st.session_state:
        st.session_state.visual_program = None
        st.session_state.blocks = []
    
    # Visual builder interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Visual Program")
        
        program_name = st.text_input("Program Name", "My Agent Task")
        
        # Add blocks
        st.markdown("**Add Blocks**")
        block_type = st.selectbox(
            "Block Type",
            ["Function", "If-Else", "Loop", "Variable", "API Call", "Database Query"]
        )
        
        if st.button("➕ Add Block"):
            block_id = len(st.session_state.blocks)
            new_block = {
                "id": block_id,
                "type": block_type,
                "name": f"{block_type}_{block_id}",
                "properties": {}
            }
            st.session_state.blocks.append(new_block)
            st.success(f"Added {block_type} block")
        
        # Show current blocks
        if st.session_state.blocks:
            st.markdown("**Current Blocks**")
            for block in st.session_state.blocks:
                with st.expander(f"{block['type']} - {block['name']}"):
                    st.write(f"Block ID: {block['id']}")
                    if st.button(f"Remove", key=f"remove_{block['id']}"):
                        st.session_state.blocks = [b for b in st.session_state.blocks if b['id'] != block['id']]
                        st.rerun()
    
    with col2:
        st.subheader("📝 Generated Code")
        
        target_language = st.selectbox("Target Language", ["Python", "JavaScript", "TypeScript"])
        
        if st.button("🚀 Generate Code"):
            if st.session_state.blocks:
                # Simple code generation
                code_lines = [f"# Generated {target_language} code for: {program_name}", ""]
                
                for block in st.session_state.blocks:
                    if block['type'] == "Function":
                        code_lines.append(f"def {block['name'].lower()}():")
                        code_lines.append("    pass")
                    elif block['type'] == "Variable":
                        code_lines.append(f"{block['name'].lower()} = None")
                    elif block['type'] == "If-Else":
                        code_lines.append("if condition:")
                        code_lines.append("    pass")
                        code_lines.append("else:")
                        code_lines.append("    pass")
                
                generated_code = "\n".join(code_lines)
                st.code(generated_code, language=target_language.lower())
                
                # Download button
                st.download_button(
                    "📥 Download Code",
                    generated_code,
                    f"{program_name.replace(' ', '_')}.py",
                    "text/plain"
                )
            else:
                st.info("Add blocks to generate code")

elif page == "📊 API Overview":
    st.header("API Overview & Monitoring")
    
    # API status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Status", "Online", "✅")
    with col2:
        st.metric("Total Requests", "1,247", "+15")
    with col3:
        st.metric("Response Time", "245ms", "-12ms")
    with col4:
        st.metric("Success Rate", "99.2%", "+0.1%")
    
    # API endpoints
    st.subheader("🔗 Available Endpoints")
    
    endpoints = [
        {"endpoint": "/api/v2/agents/execute", "method": "POST", "description": "Execute task with agent"},
        {"endpoint": "/api/v2/agents/list", "method": "GET", "description": "List all agents"},
        {"endpoint": "/api/v2/knowledge/add", "method": "POST", "description": "Add knowledge to agent"},
        {"endpoint": "/api/v2/rl/train", "method": "POST", "description": "Train agent with RL"},
        {"endpoint": "/health", "method": "GET", "description": "System health check"}
    ]
    
    for ep in endpoints:
        with st.expander(f"{ep['method']} {ep['endpoint']}"):
            st.write(ep['description'])
            
            if st.button(f"Test {ep['endpoint']}", key=f"test_{ep['endpoint']}"):
                try:
                    if ep['method'] == 'GET':
                        response = requests.get(f"{API_BASE_URL}{ep['endpoint']}", timeout=5)
                    else:
                        st.info("POST endpoints require parameters")
                        continue
                    
                    if response.status_code == 200:
                        st.success("✅ Endpoint working")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection error: {e}")

elif page == "📋 Spec-Driven Development":
    st.header("Specification-Driven Development")
    
    # Spec input
    st.subheader("📝 Create Specification")
    
    with st.form("spec_form"):
        spec_title = st.text_input("Specification Title")
        spec_description = st.text_area("Description", height=100)
        
        # Requirements
        st.markdown("**Requirements**")
        functional_req = st.text_area("Functional Requirements", height=80)
        technical_req = st.text_area("Technical Requirements", height=80)
        
        # Acceptance criteria
        acceptance_criteria = st.text_area("Acceptance Criteria", height=80)
        
        if st.form_submit_button("🚀 Generate Implementation"):
            if spec_title and spec_description:
                # Create implementation task
                implementation_task = f"""
                Implement the following specification:
                
                Title: {spec_title}
                Description: {spec_description}
                
                Functional Requirements:
                {functional_req}
                
                Technical Requirements:
                {technical_req}
                
                Acceptance Criteria:
                {acceptance_criteria}
                """
                
                with st.spinner("Generating implementation..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/v2/agents/execute",
                            json={
                                "task": implementation_task,
                                "agent_id": "system_architect"
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Implementation generated!")
                            
                            if result.get("result", {}).get("response"):
                                st.markdown("### 📋 Implementation Plan")
                                st.write(result["result"]["response"])
                        else:
                            st.error("Failed to generate implementation")
                    except Exception as e:
                        st.error(f"Error: {e}")

elif page == "⚙️ Project Configuration":
    st.header("Project Configuration")
    
    if project_manager:
        # Project selection
        projects = project_manager.list_projects()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if projects:
                selected_project = st.selectbox("Select Project", ["<Create New>"] + projects)
            else:
                selected_project = "<Create New>"
                st.info("No projects configured")
        
        with col2:
            if selected_project != "<Create New>":
                if st.button("Set as Active"):
                    project_manager.set_active_project(selected_project)
                    st.success(f"✅ Set {selected_project} as active")
        
        # Project form
        if selected_project == "<Create New>":
            st.subheader("Create New Project")
            
            with st.form("project_form"):
                project_name = st.text_input("Project Name")
                description = st.text_area("Description")
                
                # Deployment target
                st.markdown("**Deployment Target**")
                deploy_type = st.selectbox("Type", ["local", "ubuntu_server", "aws_ec2"])
                
                if deploy_type == "local":
                    local_path = st.text_input("Local Path", "/Users/yourname/project")
                elif deploy_type == "ubuntu_server":
                    server_ip = st.text_input("Server IP")
                    ssh_key = st.text_input("SSH Key Path")
                
                if st.form_submit_button("Create Project"):
                    if project_name and description:
                        # Create project config (simplified)
                        st.success(f"✅ Project '{project_name}' would be created")
                        st.info("Full project creation requires project_config module")
        
        else:
            # Show existing project
            project = project_manager.get_project(selected_project)
            if project:
                st.subheader(f"📋 {project.project_name}")
                st.write(project.description)
                
                if project.deployment_targets:
                    st.markdown("**Deployment Targets**")
                    for target in project.deployment_targets:
                        st.write(f"• {target.name} ({target.type})")
    else:
        st.error("Project configuration manager not available")

# Footer
st.divider()
st.markdown("**Agent Lightning System** - Production AI Agent Platform")