"""
Monitoring Dashboard for Agent Lightning
Real-time visualization of training metrics and agent performance
Implements web-based dashboard with live updates
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import asyncio
import requests
from pathlib import Path

@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time"""
    timestamp: datetime
    agent_id: str
    metric_name: str
    value: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard"""
    refresh_interval: int = 1  # seconds
    max_data_points: int = 1000
    metrics_retention: int = 3600  # seconds
    alert_thresholds: Dict = field(default_factory=dict)
    dashboard_port: int = 8501
    websocket_port: int = 8765
    enhanced_api_url: str = "http://localhost:8002"  # Enhanced API endpoint

class MetricsCollector:
    """Collects and stores metrics for dashboard"""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=self.config.max_data_points))
        self.alerts = []
        self.agent_states = {}
        
        # Performance metrics
        self.training_metrics = {
            "loss": deque(maxlen=self.config.max_data_points),
            "reward": deque(maxlen=self.config.max_data_points),
            "accuracy": deque(maxlen=self.config.max_data_points),
            "learning_rate": deque(maxlen=self.config.max_data_points)
        }
        
        # Agent metrics
        self.agent_metrics = defaultdict(lambda: {
            "task_completion": deque(maxlen=100),
            "response_time": deque(maxlen=100),
            "error_rate": deque(maxlen=100),
            "confidence": deque(maxlen=100)
        })
        
        # System metrics
        self.system_metrics = {
            "cpu_usage": deque(maxlen=self.config.max_data_points),
            "memory_usage": deque(maxlen=self.config.max_data_points),
            "gpu_usage": deque(maxlen=self.config.max_data_points),
            "network_io": deque(maxlen=self.config.max_data_points)
        }

class MonitoringDashboard:
    """
    Main monitoring dashboard for Agent Lightning
    Provides real-time visualization of training and performance metrics
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.collector = MetricsCollector(config)
        self.observability = None
        
        # Set alert thresholds
        self.config.alert_thresholds = {
            "loss": {"max": 2.0},
            "error_rate": {"max": 0.1},
            "response_time": {"max": 5.0},
            "memory_usage": {"max": 0.9}
        }
        
        print(f"📊 Monitoring Dashboard initialized")
        print(f"   Dashboard port: {config.dashboard_port}")
        print(f"   Refresh interval: {config.refresh_interval}s")
    
    def create_dashboard(self):
        """Create Streamlit dashboard"""
        st.set_page_config(
            page_title="Agent Lightning Monitor",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better layout
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .alert-box {
            background-color: #ff4b4b;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .task-result {
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            margin: 25px 0;
            border-left: 5px solid #28a745;
            width: 100%;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
        }
        .task-result h3 {
            color: #28a745;
            margin-top: 0;
            margin-bottom: 15px;
        }
        .stMarkdown {
            max-width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("⚡ Agent Lightning Monitoring Dashboard")
        st.markdown("Real-time monitoring of agent training and performance")
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Refresh settings
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=self.config.refresh_interval
            )
            
            # Time window
            time_window = st.selectbox(
                "Time Window",
                options=[60, 300, 900, 1800, 3600],
                format_func=lambda x: f"{x//60} min" if x >= 60 else f"{x} sec"
            )
            
            # Agent filter
            st.header("Filters")
            available_agents = self._get_available_agents()
            selected_agents = st.multiselect(
                "Select Agents",
                options=available_agents,
                default=available_agents  # Show all agents by default
            )
            
            # Metric selection
            selected_metrics = st.multiselect(
                "Metrics to Display",
                options=["loss", "reward", "accuracy", "response_time", "error_rate"],
                default=["loss", "reward", "accuracy"]
            )
        
        # Main content area
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Training Metrics",
            "🤖 Agent Performance", 
            "💻 System Resources",
            "🔔 Alerts",
            "📊 Analytics",
            "🎯 Task Assignment"
        ])
        
        with tab1:
            self._render_training_metrics(selected_agents, selected_metrics, time_window)
        
        with tab2:
            self._render_agent_performance(selected_agents, time_window)
        
        with tab3:
            self._render_system_resources()
        
        with tab4:
            self._render_alerts()
        
        with tab5:
            self._render_analytics(selected_agents, time_window)
        
        with tab6:
            self._render_task_assignment()
        
        # Auto-refresh logic at the end after content is rendered
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def _render_training_metrics(self, agents: List[str], metrics: List[str], window: int):
        """Render training metrics tab"""
        st.header("Training Metrics")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_loss = self._get_latest_metric("loss")
            st.metric(
                label="Current Loss",
                value=f"{latest_loss:.4f}",
                delta=self._calculate_delta("loss")
            )
        
        with col2:
            latest_reward = self._get_latest_metric("reward")
            st.metric(
                label="Average Reward",
                value=f"{latest_reward:.3f}",
                delta=self._calculate_delta("reward")
            )
        
        with col3:
            latest_accuracy = self._get_latest_metric("accuracy")
            st.metric(
                label="Accuracy",
                value=f"{latest_accuracy:.1%}",
                delta=self._calculate_delta("accuracy", percentage=True)
            )
        
        with col4:
            learning_rate = self._get_latest_metric("learning_rate")
            st.metric(
                label="Learning Rate",
                value=f"{learning_rate:.6f}",
                delta=None
            )
        
        # Training curves
        st.subheader("Training Progress")
        
        # Loss curve
        if "loss" in metrics:
            fig_loss = self._create_line_chart(
                "loss", agents, window,
                title="Loss Over Time",
                y_label="Loss"
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # Reward curve
        if "reward" in metrics:
            fig_reward = self._create_line_chart(
                "reward", agents, window,
                title="Reward Over Time",
                y_label="Reward"
            )
            st.plotly_chart(fig_reward, use_container_width=True)
        
        # Accuracy curve
        if "accuracy" in metrics:
            fig_accuracy = self._create_line_chart(
                "accuracy", agents, window,
                title="Accuracy Over Time",
                y_label="Accuracy (%)"
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)
    
    def _render_agent_performance(self, agents: List[str], window: int):
        """Render agent performance tab"""
        st.header("Agent Performance")
        
        # Agent status grid
        st.subheader("Agent Status")
        
        cols = st.columns(min(len(agents), 4))
        for i, agent in enumerate(agents[:4]):
            with cols[i]:
                status = self._get_agent_status(agent)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{agent}</h4>
                    <p>Status: {status['state']}</p>
                    <p>Tasks: {status['tasks_completed']}</p>
                    <p>Success Rate: {status['success_rate']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance comparison
        st.subheader("Performance Comparison")
        
        # Create comparison chart
        fig_comparison = self._create_agent_comparison_chart(agents)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Response time distribution
        st.subheader("Response Time Distribution")
        
        fig_response = self._create_response_time_histogram(agents, window)
        st.plotly_chart(fig_response, use_container_width=True)
        
        # Task completion heatmap
        st.subheader("Task Completion Heatmap")
        
        fig_heatmap = self._create_completion_heatmap(agents)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def _render_system_resources(self):
        """Render system resources tab"""
        st.header("System Resources")
        
        # Resource gauges
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_usage = self._get_latest_metric("cpu_usage")
            fig_cpu = self._create_gauge(
                cpu_usage * 100,
                "CPU Usage",
                max_value=100,
                unit="%"
            )
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            memory_usage = self._get_latest_metric("memory_usage")
            fig_memory = self._create_gauge(
                memory_usage * 100,
                "Memory Usage",
                max_value=100,
                unit="%"
            )
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col3:
            gpu_usage = self._get_latest_metric("gpu_usage")
            fig_gpu = self._create_gauge(
                gpu_usage * 100,
                "GPU Usage",
                max_value=100,
                unit="%"
            )
            st.plotly_chart(fig_gpu, use_container_width=True)
    
    def _render_alerts(self):
        """Render alerts tab"""
        st.header("🔔 Alerts & Notifications")
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            critical_alerts = len([a for a in self.collector.alerts 
                                 if a.get("type") == "critical"])
            st.metric("Critical Alerts", critical_alerts)
        
        with col2:
            warning_alerts = len([a for a in self.collector.alerts 
                                if a.get("type") == "warning"])
            st.metric("Warnings", warning_alerts)
        
        with col3:
            info_alerts = len([a for a in self.collector.alerts 
                             if a.get("type") == "info"])
            st.metric("Info", info_alerts)
        
        # Recent alerts
        st.subheader("Recent Alerts")
        
        if self.collector.alerts:
            for alert in self.collector.alerts[-10:]:
                alert_type = alert.get("type", "info")
                icon = "🔴" if alert_type == "critical" else "🟡" if alert_type == "warning" else "🔵"
                
                st.markdown(f"""
                <div class="alert-box">
                    {icon} <strong>{alert['timestamp'].strftime('%H:%M:%S')}</strong> - 
                    Agent: {alert['agent_id']} | 
                    Metric: {alert['metric']} | 
                    Value: {alert['value']:.3f} | 
                    Threshold: {alert.get('threshold', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alerts at this time")
    
    def _render_analytics(self, agents: List[str], window: int):
        """Render analytics tab"""
        st.header("📊 Analytics & Insights")
        
        # Performance trends
        st.subheader("Performance Trends")
        
        trend_data = self._calculate_trends(agents, window)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Improvement rate
            fig_improvement = go.Figure(data=[
                go.Bar(
                    x=list(trend_data.keys()),
                    y=[v["improvement_rate"] for v in trend_data.values()],
                    text=[f"{v['improvement_rate']:.1%}" for v in trend_data.values()],
                    textposition="auto"
                )
            ])
            fig_improvement.update_layout(
                title="Improvement Rate by Agent",
                yaxis_title="Improvement %"
            )
            st.plotly_chart(fig_improvement, use_container_width=True)
        
        with col2:
            # Efficiency score
            fig_efficiency = go.Figure(data=[
                go.Bar(
                    x=list(trend_data.keys()),
                    y=[v["efficiency_score"] for v in trend_data.values()],
                    text=[f"{v['efficiency_score']:.2f}" for v in trend_data.values()],
                    textposition="auto"
                )
            ])
            fig_efficiency.update_layout(
                title="Efficiency Score by Agent",
                yaxis_title="Score"
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
    def _render_task_assignment(self):
        """Render task assignment interface"""
        st.header("🎯 Task Assignment & Agent Interaction")
        
        # Note about auto-refresh
        st.info("💡 Tip: Turn off 'Auto Refresh' in the sidebar while using this tab to keep results visible.")
        
        # Initialize session state for API connection
        if 'api_connected' not in st.session_state:
            st.session_state.api_connected = False
        if 'api_token' not in st.session_state:
            st.session_state.api_token = None
        if 'task_history' not in st.session_state:
            st.session_state.task_history = []
        if 'available_agents' not in st.session_state:
            st.session_state.available_agents = []
        
        # API Connection Status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Connect to the Production API to assign tasks to agents")
        with col2:
            if st.button("🔌 Connect to API"):
                try:
                    # Authenticate with Enhanced API
                    response = requests.post(
                        "http://localhost:8002/api/v1/auth/token",
                        params={"username": "admin", "password": "admin"},
                        timeout=5
                    )
                    if response.status_code == 200:
                        st.session_state.api_token = response.json()["access_token"]
                        st.session_state.api_connected = True
                        
                        # Fetch available agents from enhanced API
                        try:
                            agents_response = requests.get("http://localhost:8002/api/v2/agents/list")
                            if agents_response.status_code == 200:
                                agents_data = agents_response.json()
                                st.session_state.available_agents = agents_data.get("agents", [])
                        except:
                            pass
                        
                        st.success("✅ Connected to API!")
                    else:
                        st.error("❌ Failed to connect to API")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: Make sure the API is running on port 8002")
        
        if st.session_state.api_connected:
            st.success("✅ API Connected - Ready to assign tasks")
            
            # Task Assignment Section
            st.subheader("📝 Create New Task")
            
            col1, col2 = st.columns(2)
            
            with col1:
                task_description = st.text_area(
                    "Task Description",
                    placeholder="Enter your task here... e.g., 'Research the latest AI trends' or 'Write a blog post about quantum computing'",
                    height=100
                )
                
                # Agent selection - use dynamically fetched agents
                if st.session_state.available_agents:
                    agent_options = ["auto"] + [agent["id"] for agent in st.session_state.available_agents]
                    agent_labels = ["Auto-select"] + [f"{agent['name']} ({agent['role']})" for agent in st.session_state.available_agents]
                    agent_type = st.selectbox(
                        "Select Agent",
                        options=agent_options,
                        format_func=lambda x: agent_labels[agent_options.index(x)],
                        help="Choose 'auto' to let the system select the best agent"
                    )
                else:
                    # Fallback to default agents
                    agent_type = st.selectbox(
                        "Select Agent",
                        options=["auto", "researcher", "writer", "reviewer", "optimizer"],
                        help="Choose 'auto' to let the system select the best agent"
                    )
                
                priority = st.select_slider(
                    "Priority",
                    options=["low", "normal", "high", "urgent"],
                    value="normal"
                )
            
            with col2:
                st.markdown("**Available Agents:**")
                if st.session_state.available_agents:
                    for agent in st.session_state.available_agents[:7]:  # Show first 7 agents
                        icon = {
                            "full_stack_developer": "💻",
                            "mobile_developer": "📱", 
                            "security_expert": "🔒",
                            "devops_engineer": "🚀",
                            "data_scientist": "📊",
                            "ui_ux_designer": "🎨",
                            "blockchain_developer": "⛓️",
                            "researcher": "🔍",
                            "writer": "✍️",
                            "reviewer": "👁️",
                            "optimizer": "⚡"
                        }.get(agent["id"], "🤖")
                        st.markdown(f"- {icon} **{agent['name']}**: {agent['role']}")
                else:
                    st.markdown("""
                    - 🔍 **Researcher**: Research and information gathering
                    - ✍️ **Writer**: Content creation and documentation
                    - 👁️ **Reviewer**: Code and content review
                    - ⚡ **Optimizer**: Performance optimization
                    """)
            
            # Submit button
            if st.button("🚀 Submit Task", type="primary", disabled=not task_description):
                with st.spinner("Processing task..."):
                    try:
                        headers = {
                            "Authorization": f"Bearer {st.session_state.api_token}",
                            "Content-Type": "application/json"
                        }
                        
                        # Execute task with agent
                        task_response = requests.post(
                            "http://localhost:8002/api/v2/agents/execute",
                            json={
                                "task": task_description,
                                "agent_id": agent_type if agent_type != "auto" else None,
                                "timeout": 60
                            },
                            headers=headers,
                            timeout=30
                        )
                        
                        if task_response.status_code == 200:
                            task_data = task_response.json()
                            task_id = task_data.get("task_id")
                            
                            st.success(f"✅ Task submitted! ID: {task_id}")
                            st.info("⏳ Task is being processed. Results will appear below when ready.")
                            
                            # Wait a short moment then check once
                            time.sleep(3)
                            
                            # Check task status once
                            try:
                                status_response = requests.get(
                                    f"http://localhost:8002/api/v1/tasks/{task_id}",
                                    headers=headers,
                                    timeout=5
                                )
                                
                                if status_response.status_code == 200:
                                    task_status = status_response.json()
                                    
                                    if task_status.get("status") in ["completed", "failed"]:
                                        result = task_status.get("result", {})
                                        
                                        # Extract AI response from the result
                                        ai_response = None
                                        
                                        if isinstance(result, dict):
                                            # Try direct response field first
                                            ai_response = result.get("response")
                                        elif isinstance(result, str):
                                            ai_response = result
                                        
                                        if ai_response:
                                            # Display result in center with full width
                                            st.success("✅ Task completed!")
                                            
                                            # Create full-width centered container for result
                                            st.markdown(f"""
                                            <div class="task-result">
                                                <h3>📋 Task Result</h3>
                                                <div style="white-space: pre-wrap; word-wrap: break-word;">{ai_response}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Show execution time if available
                                            if isinstance(result, dict) and "execution_time" in result:
                                                st.caption(f"⏱️ Execution time: {result['execution_time']:.2f} seconds")
                                        else:
                                            st.info("Task is processing. Check 'Recent Tasks' below or refresh the page.")
                                    else:
                                        st.info("Task is still processing. Check 'Recent Tasks' below or refresh the page.")
                            except:
                                st.info("Task submitted. Check 'Recent Tasks' below for status.")
                        else:
                            st.error(f"Failed to create task: {task_response.text}")
                    
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. The task might still be processing.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Error: {str(e)}")
        
        else:
            st.warning("⚠️ Not connected to API. Click 'Connect to API' to start assigning tasks.")
            st.markdown("""
            **Requirements:**
            1. Make sure the Production API is running on port 8002
            2. Run: `python enhanced_production_api.py`
            """)
    
    # Helper methods
    def _get_available_agents(self) -> List[str]:
        """Get list of available agents from the API"""
        try:
            # Try to get agents from the enhanced API  
            api_url = "http://localhost:8002/api/v2/agents/list"
            response = requests.get(api_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Extract agent IDs from the response
                agents = [agent['id'] for agent in data.get('agents', [])]
                if agents:
                    return agents
        except Exception as e:
            pass  # Silently fall back to defaults
        
        # Fallback to all known agents if API fails
        return [
            "full_stack_developer",
            "mobile_developer", 
            "security_expert",
            "devops_engineer",
            "data_scientist",
            "ui_ux_designer",
            "blockchain_developer"
        ]
    
    def _get_latest_metric(self, metric_name: str) -> float:
        """Get latest value for a metric"""
        # Simulated for demo
        if metric_name == "loss":
            return 0.234 + np.random.random() * 0.1
        elif metric_name == "reward":
            return 0.85 + np.random.random() * 0.1
        elif metric_name == "accuracy":
            return 0.92 + np.random.random() * 0.05
        elif metric_name == "learning_rate":
            return 0.0001
        elif metric_name == "cpu_usage":
            return 0.45 + np.random.random() * 0.2
        elif metric_name == "memory_usage":
            return 0.60 + np.random.random() * 0.2
        elif metric_name == "gpu_usage":
            return 0.75 + np.random.random() * 0.15
        else:
            return np.random.random()
    
    def _calculate_delta(self, metric_name: str, percentage: bool = False) -> str:
        """Calculate metric delta"""
        # Simulated for demo
        delta = np.random.random() * 0.1 - 0.05
        if percentage:
            return f"{delta:.1%}"
        else:
            return f"{delta:.3f}"
    
    def _get_agent_status(self, agent_id: str) -> Dict:
        """Get real agent status from stored task results"""
        # Initialize session state for tracking real stats if not exists
        if 'agent_stats' not in st.session_state:
            st.session_state.agent_stats = {}
        
        # Get or initialize stats for this agent
        if agent_id not in st.session_state.agent_stats:
            st.session_state.agent_stats[agent_id] = {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "last_active": None
            }
        
        stats = st.session_state.agent_stats[agent_id]
        total_tasks = stats["tasks_completed"] + stats["tasks_failed"]
        
        # Determine state based on last activity
        state = "Idle"
        if stats["last_active"]:
            time_since_active = (datetime.now() - stats["last_active"]).seconds
            if time_since_active < 60:
                state = "Active"
            elif time_since_active < 300:
                state = "Ready"
        
        return {
            "state": state,
            "tasks_completed": total_tasks,
            "success_rate": stats["tasks_completed"] / total_tasks if total_tasks > 0 else 0.0
        }
    
    def _create_line_chart(self, metric: str, agents: List[str], 
                          window: int, title: str, y_label: str) -> go.Figure:
        """Create line chart for metric"""
        fig = go.Figure()
        
        for agent in agents:
            # Generate sample data
            timestamps = pd.date_range(end=datetime.now(), periods=20, freq='1min')
            values = np.random.random(20) * 0.5 + 0.5
            if metric == "loss":
                values = values[::-1] * 0.5  # Decreasing trend
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode="lines",
                name=agent
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_label,
            hovermode="x unified"
        )
        
        return fig
    
    def _create_gauge(self, value: float, title: str, 
                     max_value: float = 100, unit: str = "") -> go.Figure:
        """Create gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            number={"suffix": unit},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, max_value * 0.5], "color": "lightgray"},
                    {"range": [max_value * 0.5, max_value * 0.8], "color": "gray"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=250)
        return fig
    
    def _create_agent_comparison_chart(self, agents: List[str]) -> go.Figure:
        """Create agent comparison chart"""
        metrics = ["Task Success", "Avg Response Time", "Error Rate", "Confidence"]
        
        fig = go.Figure()
        
        for agent in agents:
            values = [
                np.random.random() * 100,  # Task success
                np.random.random() * 5,     # Response time
                np.random.random() * 10,    # Error rate
                np.random.random() * 100    # Confidence
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill="toself",
                name=agent
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Agent Performance Comparison"
        )
        
        return fig
    
    def _create_response_time_histogram(self, agents: List[str], window: int) -> go.Figure:
        """Create response time histogram"""
        fig = go.Figure()
        
        for agent in agents:
            response_times = np.random.exponential(2, 100)
            fig.add_trace(go.Histogram(
                x=response_times,
                name=agent,
                opacity=0.7,
                nbinsx=20
            ))
        
        fig.update_layout(
            title="Response Time Distribution",
            xaxis_title="Response Time (seconds)",
            yaxis_title="Frequency",
            barmode="overlay"
        )
        
        return fig
    
    def _create_completion_heatmap(self, agents: List[str]) -> go.Figure:
        """Create task completion heatmap"""
        # Simulated data
        hours = list(range(24))
        
        data = []
        for agent in agents:
            completions = np.random.randint(0, 20, 24)
            data.append(completions)
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=hours,
            y=agents,
            colorscale="Viridis"
        ))
        
        fig.update_layout(
            title="Task Completions by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Agent"
        )
        
        return fig
    
    def _calculate_trends(self, agents: List[str], window: int) -> Dict:
        """Calculate performance trends"""
        trends = {}
        
        for agent in agents:
            # Simulated trend calculation
            trends[agent] = {
                "improvement_rate": np.random.random() * 0.2 - 0.05,
                "efficiency_score": np.random.random() * 100,
                "stability": np.random.random()
            }
        
        return trends


def run_dashboard():
    """Run the monitoring dashboard"""
    config = DashboardConfig(
        refresh_interval=2,
        max_data_points=1000,
        dashboard_port=8501
    )
    
    dashboard = MonitoringDashboard(config)
    dashboard.create_dashboard()


# Streamlit app initialization
# This runs when streamlit run monitoring_dashboard.py is executed
config = DashboardConfig()
dashboard = MonitoringDashboard(config)
dashboard.create_dashboard()