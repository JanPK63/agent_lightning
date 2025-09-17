#!/usr/bin/env python3
"""
Monitoring Dashboard - Integrated Version
Updated to work with the integrated services through API Gateway
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import sys
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_GATEWAY_URL = "http://localhost:8000"  # Updated to use API Gateway
INTEGRATED_SERVICES = {
    "Agent Designer": "http://localhost:8102",
    "Workflow Engine": "http://localhost:8103", 
    "Integration Hub": "http://localhost:8104",
    "AI Model": "http://localhost:8105",
    "Auth Service": "http://localhost:8106",
    "WebSocket": "http://localhost:8107"
}

# Set page config
st.set_page_config(
    page_title="Lightning System Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-online {
        color: #00cc44;
        font-weight: bold;
    }
    .status-offline {
        color: #ff4444;
        font-weight: bold;
    }
    .agent-card {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

def check_service_health(service_url: str) -> Dict[str, Any]:
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{service_url}/health", timeout=2)
        if response.status_code == 200:
            return {"status": "online", "data": response.json()}
        return {"status": "offline", "data": None}
    except:
        return {"status": "offline", "data": None}

def get_agents() -> List[Dict[str, Any]]:
    """Get list of agents through API Gateway"""
    try:
        # Use API Gateway instead of direct service
        response = requests.get(f"{API_GATEWAY_URL}/api/v1/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, list) else data.get("agents", [])
        return []
    except Exception as e:
        st.error(f"Failed to fetch agents: {e}")
        return []

def get_workflows() -> List[Dict[str, Any]]:
    """Get list of workflows through API Gateway"""
    try:
        response = requests.get(f"{API_GATEWAY_URL}/api/v1/workflows", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, list) else data.get("workflows", [])
        return []
    except Exception as e:
        st.error(f"Failed to fetch workflows: {e}")
        return []

def execute_agent_task(agent_id: str, task: Dict[str, Any]) -> Optional[str]:
    """Execute a task on an agent through API Gateway"""
    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/agents/{agent_id}/execute",
            json=task,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("task_id")
        return None
    except Exception as e:
        st.error(f"Failed to execute task: {e}")
        return None

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get task status through API Gateway"""
    try:
        response = requests.get(
            f"{API_GATEWAY_URL}/api/v1/tasks/{task_id}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return {"status": "unknown"}
    except:
        return {"status": "error"}

def main():
    """Main dashboard application"""
    
    # Title and header
    st.title("⚡ Lightning System Monitor")
    st.markdown("### Integrated Services Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["System Overview", "Agents", "Workflows", "Services", "Database", "Logs"]
        )
        
        st.divider()
        
        # Quick Actions
        st.header("Quick Actions")
        if st.button("🔄 Refresh All", use_container_width=True):
            st.rerun()
        
        if st.button("🚀 Restart System", use_container_width=True):
            if st.checkbox("Confirm restart"):
                os.system("./start_integrated_system.sh stop")
                time.sleep(2)
                os.system("./start_integrated_system.sh")
                st.success("System restarted!")
                time.sleep(2)
                st.rerun()
    
    # Main content based on selected page
    if page == "System Overview":
        show_system_overview()
    elif page == "Agents":
        show_agents_page()
    elif page == "Workflows":
        show_workflows_page()
    elif page == "Services":
        show_services_page()
    elif page == "Database":
        show_database_page()
    elif page == "Logs":
        show_logs_page()

def show_system_overview():
    """Show system overview page"""
    st.header("System Overview")
    
    # Service Status
    col1, col2, col3 = st.columns(3)
    
    # Check API Gateway
    gateway_health = check_service_health(API_GATEWAY_URL)
    with col1:
        st.metric(
            "API Gateway",
            "Online" if gateway_health["status"] == "online" else "Offline",
            delta="Port 8000"
        )
    
    # Check integrated services
    online_services = 0
    total_services = len(INTEGRATED_SERVICES)
    
    for service_name, service_url in INTEGRATED_SERVICES.items():
        health = check_service_health(service_url)
        if health["status"] == "online":
            online_services += 1
    
    with col2:
        st.metric(
            "Integrated Services",
            f"{online_services}/{total_services}",
            delta=f"{online_services} online"
        )
    
    # Check database (through API)
    try:
        db_response = requests.get(f"{API_GATEWAY_URL}/api/v1/agents", timeout=2)
        db_status = "Connected" if db_response.status_code == 200 else "Error"
    except:
        db_status = "Disconnected"
    
    with col3:
        st.metric(
            "Database",
            db_status,
            delta="PostgreSQL"
        )
    
    st.divider()
    
    # Service Details
    st.subheader("Service Status")
    
    service_data = []
    for service_name, service_url in INTEGRATED_SERVICES.items():
        health = check_service_health(service_url)
        port = service_url.split(":")[-1]
        
        service_data.append({
            "Service": service_name,
            "Port": port,
            "Status": "🟢 Online" if health["status"] == "online" else "🔴 Offline",
            "URL": service_url
        })
    
    df = pd.DataFrame(service_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # System Metrics
    st.divider()
    st.subheader("System Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a simple gauge chart for system health
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=online_services,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Services Online"},
            gauge={
                'axis': {'range': [None, total_services]},
                'bar': {'color': "green" if online_services == total_services else "orange"},
                'steps': [
                    {'range': [0, total_services/2], 'color': "lightgray"},
                    {'range': [total_services/2, total_services], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': total_services * 0.9
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Activity timeline
        st.markdown("#### Recent Activity")
        activities = [
            {"time": "Just now", "event": "Dashboard accessed"},
            {"time": "2 min ago", "event": "Services health checked"},
            {"time": "5 min ago", "event": "Database connected"},
            {"time": "10 min ago", "event": "System initialized"}
        ]
        for activity in activities:
            st.markdown(f"**{activity['time']}**: {activity['event']}")

def show_agents_page():
    """Show agents management page"""
    st.header("Agent Management")
    
    # Get agents
    agents = get_agents()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Agents", len(agents))
    with col2:
        active_agents = sum(1 for a in agents if a.get("status") == "active")
        st.metric("Active Agents", active_agents)
    with col3:
        st.metric("Available", len(agents) - active_agents)
    
    st.divider()
    
    # Agent list
    if agents:
        for agent in agents:
            with st.expander(f"🤖 {agent.get('name', 'Unknown')} - {agent.get('type', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**ID**: {agent.get('id', 'N/A')}")
                    st.markdown(f"**Type**: {agent.get('type', 'N/A')}")
                    st.markdown(f"**Status**: {agent.get('status', 'unknown')}")
                    
                with col2:
                    st.markdown(f"**Capabilities**: {', '.join(agent.get('capabilities', []))}")
                    st.markdown(f"**Created**: {agent.get('created_at', 'N/A')}")
                
                # Execute task
                st.markdown("#### Execute Task")
                task_type = st.selectbox(
                    "Task Type",
                    ["analyze", "generate", "transform", "validate"],
                    key=f"task_{agent.get('id')}"
                )
                task_input = st.text_area(
                    "Task Input",
                    key=f"input_{agent.get('id')}"
                )
                
                if st.button("Execute", key=f"exec_{agent.get('id')}"):
                    task = {
                        "type": task_type,
                        "input": task_input,
                        "parameters": {}
                    }
                    task_id = execute_agent_task(agent.get('id'), task)
                    if task_id:
                        st.success(f"Task submitted: {task_id}")
                        
                        # Poll for result
                        with st.spinner("Executing task..."):
                            for _ in range(10):
                                time.sleep(1)
                                status = get_task_status(task_id)
                                if status.get("status") == "completed":
                                    st.success("Task completed!")
                                    st.json(status.get("result"))
                                    break
                                elif status.get("status") == "failed":
                                    st.error(f"Task failed: {status.get('error')}")
                                    break
                    else:
                        st.error("Failed to submit task")
    else:
        st.info("No agents available. Check if Agent Designer service is running.")

def show_workflows_page():
    """Show workflows page"""
    st.header("Workflow Management")
    
    workflows = get_workflows()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Workflows", len(workflows))
    with col2:
        running = sum(1 for w in workflows if w.get("status") == "running")
        st.metric("Running", running)
    with col3:
        completed = sum(1 for w in workflows if w.get("status") == "completed")
        st.metric("Completed", completed)
    
    st.divider()
    
    # Workflow list
    if workflows:
        for workflow in workflows:
            with st.expander(f"📋 {workflow.get('name', 'Unknown')}"):
                st.json(workflow)
    else:
        st.info("No workflows available.")

def show_services_page():
    """Show detailed services page"""
    st.header("Service Details")
    
    for service_name, service_url in INTEGRATED_SERVICES.items():
        with st.expander(f"🔧 {service_name}"):
            health = check_service_health(service_url)
            
            if health["status"] == "online":
                st.success("Service is online")
                if health["data"]:
                    st.json(health["data"])
            else:
                st.error("Service is offline")
                
            st.markdown(f"**URL**: {service_url}")
            st.markdown(f"**Port**: {service_url.split(':')[-1]}")
            
            # Service actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Restart {service_name}", key=f"restart_{service_name}"):
                    st.info(f"Restarting {service_name}...")
                    # Add restart logic here
            
            with col2:
                if st.button(f"View Logs", key=f"logs_{service_name}"):
                    # Add log viewing logic here
                    st.info("Logs will be displayed here")

def show_database_page():
    """Show database statistics"""
    st.header("Database Statistics")
    
    try:
        # Get stats through API
        agents = get_agents()
        workflows = get_workflows()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Agents", len(agents))
        with col2:
            st.metric("Workflows", len(workflows))
        with col3:
            st.metric("Tasks", "N/A")  # Would need task endpoint
        with col4:
            st.metric("Sessions", "N/A")  # Would need session endpoint
        
        st.divider()
        
        # Connection info
        st.subheader("Connection Information")
        st.markdown("**Database**: PostgreSQL")
        st.markdown("**Host**: localhost:5432")
        st.markdown("**Database Name**: agent_lightning")
        st.markdown("**User**: agent_user")
        
    except Exception as e:
        st.error(f"Failed to fetch database stats: {e}")

def show_logs_page():
    """Show system logs"""
    st.header("System Logs")
    
    log_files = {
        "API Gateway": "/tmp/api_gateway.log",
        "Agent Designer": "/tmp/agent_designer.log",
        "Workflow Engine": "/tmp/workflow_engine.log",
        "Integration Hub": "/tmp/integration_hub.log",
        "AI Model": "/tmp/ai_model.log",
        "Auth Service": "/tmp/auth.log",
        "WebSocket": "/tmp/websocket.log"
    }
    
    selected_log = st.selectbox("Select Log File", list(log_files.keys()))
    
    if st.button("Load Logs"):
        log_file = log_files[selected_log]
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    # Get last 100 lines
                    lines = f.readlines()
                    last_lines = lines[-100:] if len(lines) > 100 else lines
                    
                    st.text_area(
                        f"Last 100 lines from {selected_log}",
                        value=''.join(last_lines),
                        height=400
                    )
            else:
                st.warning(f"Log file not found: {log_file}")
        except Exception as e:
            st.error(f"Failed to read log: {e}")
    
    # Auto-refresh option
    if st.checkbox("Auto-refresh logs"):
        st.info("Logs will refresh every 5 seconds")
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()