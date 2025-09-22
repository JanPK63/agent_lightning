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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import asyncio
import requests
# import websocket  # Not needed for dashboard functionality
from pathlib import Path

# Import Agent Lightning components
from observability_setup import AgentLightningObservability, MetricsAggregator

# Import Visual Code Builder components
try:
    from visual_code_builder import VisualProgram, BlockFactory, BlockType
    from visual_component_library import ComponentLibrary, ComponentCategory
    from visual_to_code_translator import VisualToCodeTranslator, TargetLanguage
    from code_preview_panel import CodePreviewPanel, PreviewSettings, PreviewTheme
    from visual_code_blocks import InteractiveBlock, VisualCanvas, BlockStyle
    from visual_debugger import VisualDebugger, DebugState, BreakpointType
    VISUAL_CODE_BUILDER_AVAILABLE = True
except ImportError as e:
    print(f"Visual Code Builder import error: {e}")
    VISUAL_CODE_BUILDER_AVAILABLE = False


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
    
    def add_metric(self, snapshot: MetricSnapshot):
        """Add a metric snapshot"""
        key = f"{snapshot.agent_id}_{snapshot.metric_name}"
        self.metrics_buffer[key].append(snapshot)
        
        # Check for alerts
        self._check_alerts(snapshot)
        
        # Update specific metric categories
        if snapshot.metric_name in self.training_metrics:
            self.training_metrics[snapshot.metric_name].append({
                "timestamp": snapshot.timestamp,
                "value": snapshot.value
            })
        
        if snapshot.agent_id in self.agent_metrics:
            if snapshot.metric_name in self.agent_metrics[snapshot.agent_id]:
                self.agent_metrics[snapshot.agent_id][snapshot.metric_name].append({
                    "timestamp": snapshot.timestamp,
                    "value": snapshot.value
                })
    
    def _check_alerts(self, snapshot: MetricSnapshot):
        """Check if metric triggers any alerts"""
        if snapshot.metric_name in self.config.alert_thresholds:
            threshold = self.config.alert_thresholds[snapshot.metric_name]
            
            if isinstance(threshold, dict):
                if "min" in threshold and snapshot.value < threshold["min"]:
                    self.alerts.append({
                        "timestamp": snapshot.timestamp,
                        "agent_id": snapshot.agent_id,
                        "metric": snapshot.metric_name,
                        "value": snapshot.value,
                        "threshold": threshold["min"],
                        "type": "below_minimum"
                    })
                
                if "max" in threshold and snapshot.value > threshold["max"]:
                    self.alerts.append({
                        "timestamp": snapshot.timestamp,
                        "agent_id": snapshot.agent_id,
                        "metric": snapshot.metric_name,
                        "value": snapshot.value,
                        "threshold": threshold["max"],
                        "type": "above_maximum"
                    })
    
    def get_recent_metrics(self, metric_name: str, agent_id: str = None, 
                          window_seconds: int = 300) -> pd.DataFrame:
        """Get recent metrics as DataFrame"""
        data = []
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        
        for key, snapshots in self.metrics_buffer.items():
            if metric_name in key:
                if agent_id is None or agent_id in key:
                    for snapshot in snapshots:
                        if snapshot.timestamp > cutoff_time:
                            data.append({
                                "timestamp": snapshot.timestamp,
                                "agent_id": snapshot.agent_id,
                                "metric": snapshot.metric_name,
                                "value": snapshot.value
                            })
        
        return pd.DataFrame(data)


class MonitoringDashboard:
    """
    Main monitoring dashboard for Agent Lightning
    Provides real-time visualization of training and performance metrics
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.collector = MetricsCollector(config)
        # Disable observability due to threading issues
        self.observability = None
        # self.observability = AgentLightningObservability(
        #     prometheus_port=8003,
        #     enable_console_export=False
        # )
        
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
        
        # Custom CSS
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📈 Training Metrics",
            "🤖 Agent Performance", 
            "💻 System Resources",
            "🔔 Alerts",
            "📊 Analytics",
            "🎯 Task Assignment",
            "🧠 Agent Knowledge",
            "⚙️ Project Config",
            "🎨 Visual Code Builder"
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
        
        with tab7:
            self._render_agent_knowledge()
        
        with tab8:
            self._render_project_config()
        
        with tab9:
            self._render_visual_code_builder()
        
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
        
        # Resource timeline
        st.subheader("Resource Usage Over Time")
        
        fig_resources = make_subplots(
            rows=3, cols=1,
            subplot_titles=("CPU Usage", "Memory Usage", "GPU Usage"),
            shared_xaxes=True
        )
        
        # Add traces for each resource
        for i, (metric, title) in enumerate([
            ("cpu_usage", "CPU"),
            ("memory_usage", "Memory"),
            ("gpu_usage", "GPU")
        ], 1):
            data = self._get_metric_history(metric, window_seconds=1800)
            if not data.empty:
                fig_resources.add_trace(
                    go.Scatter(
                        x=data["timestamp"],
                        y=data["value"] * 100,
                        name=title,
                        mode="lines"
                    ),
                    row=i, col=1
                )
        
        fig_resources.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_resources, use_container_width=True)
    
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
        
        # Correlation matrix
        st.subheader("Metric Correlations")
        
        correlation_matrix = self._calculate_correlations()
        fig_corr = px.imshow(
            correlation_matrix,
            labels=dict(x="Metric", y="Metric", color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Predictions
        st.subheader("Performance Predictions")
        
        predictions = self._generate_predictions(agents)
        
        fig_pred = go.Figure()
        for agent in agents:
            if agent in predictions:
                fig_pred.add_trace(go.Scatter(
                    x=predictions[agent]["timestamps"],
                    y=predictions[agent]["predicted_performance"],
                    mode="lines",
                    name=f"{agent} (predicted)",
                    line=dict(dash="dash")
                ))
        
        fig_pred.update_layout(
            title="24-Hour Performance Forecast",
            xaxis_title="Time",
            yaxis_title="Predicted Performance"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
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
    
    def _get_metric_history(self, metric_name: str, window_seconds: int = 300) -> pd.DataFrame:
        """Get metric history"""
        # Simulated for demo
        # Ensure we have at least 1 second frequency to avoid division by zero
        freq_seconds = max(1, window_seconds // 100)
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=100,
            freq=f"{freq_seconds}S"
        )
        
        values = np.random.random(100) * 0.5 + 0.5
        if metric_name == "loss":
            values = values[::-1] * 0.5  # Decreasing trend
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "value": values
        })
    
    def _create_line_chart(self, metric: str, agents: List[str], 
                          window: int, title: str, y_label: str) -> go.Figure:
        """Create line chart for metric"""
        fig = go.Figure()
        
        for agent in agents:
            data = self._get_metric_history(metric, window)
            if not data.empty:
                fig.add_trace(go.Scatter(
                    x=data["timestamp"],
                    y=data["value"],
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
    
    def _calculate_correlations(self) -> pd.DataFrame:
        """Calculate metric correlations"""
        # Simulated correlation matrix
        metrics = ["Loss", "Reward", "Accuracy", "Response Time", "Error Rate"]
        
        n = len(metrics)
        corr_matrix = np.random.random((n, n))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)  # Diagonal = 1
        
        return pd.DataFrame(corr_matrix, index=metrics, columns=metrics)
    
    def _generate_predictions(self, agents: List[str]) -> Dict:
        """Generate performance predictions"""
        predictions = {}
        
        future_timestamps = pd.date_range(
            start=datetime.now(),
            periods=24,
            freq="H"
        )
        
        for agent in agents:
            # Simple trend prediction
            base_performance = 0.8 + np.random.random() * 0.15
            trend = np.random.random() * 0.01 - 0.005
            
            predicted = []
            for i in range(24):
                performance = base_performance + trend * i + np.random.random() * 0.05
                predicted.append(min(1.0, max(0, performance)))
            
            predictions[agent] = {
                "timestamps": future_timestamps,
                "predicted_performance": predicted
            }
        
        return predictions


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
                
                # Deployment Options Section
                st.markdown("---")
                st.markdown("**🚀 Deployment Options** (for code generation tasks)")
                
                # Check if there's a project loaded from Project Config tab
                from project_config import ProjectConfigManager
                manager = ProjectConfigManager()
                
                # Option to use saved project config
                use_saved_config = False
                if manager.active_project:
                    col_cfg1, col_cfg2 = st.columns([2, 1])
                    with col_cfg1:
                        st.info(f"📋 Active Project: {manager.active_project}")
                    with col_cfg2:
                        if st.button("Use Project Config"):
                            project = manager.get_active_project()
                            if project:
                                default_target = manager.get_default_deployment()
                                if default_target:
                                    if default_target.type == "ubuntu_server":
                                        st.session_state.deployment_from_project = {
                                            "type": default_target.type,
                                            "server_ip": default_target.server_ip,
                                            "username": default_target.username or "ubuntu",
                                            "key_path": default_target.ssh_key_path,
                                            "working_directory": default_target.server_directory or "/home/ubuntu"
                                        }
                                    elif default_target.type == "local":
                                        st.session_state.deployment_from_project = {
                                            "type": default_target.type,
                                            "path": default_target.local_path
                                        }
                                    elif default_target.type == "aws_ec2":
                                        st.session_state.deployment_from_project = {
                                            "type": default_target.type,
                                            "region": default_target.aws_region,
                                            "instance_type": default_target.instance_type,
                                            "key_name": default_target.aws_key_name
                                        }
                                    else:
                                        st.session_state.deployment_from_project = {
                                            "type": default_target.type
                                        }
                                    st.success(f"✅ Loaded config for: {default_target.name}")
                                    use_saved_config = True
                
                # Use saved config if available, otherwise show manual options
                if 'deployment_from_project' in st.session_state:
                    deployment_config = st.session_state.deployment_from_project
                    deployment_type = deployment_config.get('type', 'none')  # Set deployment_type from saved config
                    st.success(f"Using saved deployment: {deployment_type}")
                    if st.button("Clear and use manual config"):
                        del st.session_state.deployment_from_project
                        st.rerun()
                else:
                    deployment_type = st.radio(
                        "Where should the code be deployed?",
                        options=["none", "local", "aws_ec2", "ubuntu_server"],
                        format_func=lambda x: {
                            "none": "📄 Just generate code (no deployment)",
                            "local": "💻 Local (MacBook)",
                            "aws_ec2": "☁️ AWS EC2 Instance",
                            "ubuntu_server": "🖥️ Ubuntu Server"
                        }[x],
                        index=0
                    )
                    
                    deployment_config = {}
                    
                    if deployment_type == "local":
                        local_path = st.text_input(
                            "Local Directory Path",
                            placeholder="/Users/yourname/projects/my-app",
                            help="Where to create/deploy the code on your Mac"
                        )
                        deployment_config = {
                            "type": "local",
                            "path": local_path
                        }
                    
                    elif deployment_type == "aws_ec2":
                        col_aws1, col_aws2 = st.columns(2)
                        with col_aws1:
                            aws_region = st.selectbox(
                                "AWS Region",
                                options=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
                                index=0
                            )
                            instance_type = st.selectbox(
                                "Instance Type",
                                options=["t2.micro", "t2.small", "t3.micro", "t3.small"],
                                index=0
                            )
                        with col_aws2:
                            aws_key = st.text_input(
                                "Key Pair Name (optional)",
                                placeholder="my-key-pair",
                                help="Leave empty to auto-create"
                            )
                            working_dir = st.text_input(
                                "Working Directory",
                                value="/home/ec2-user/agent-app",
                                help="Where to deploy on EC2"
                            )
                        deployment_config = {
                            "type": "aws_ec2",
                            "region": aws_region,
                            "instance_type": instance_type,
                            "key_name": aws_key if aws_key else None,
                            "working_directory": working_dir
                        }
                    
                    elif deployment_type == "ubuntu_server":
                        col_ub1, col_ub2 = st.columns(2)
                        with col_ub1:
                            server_ip = st.text_input(
                                "Server IP Address",
                                placeholder="192.168.1.100 or domain.com",
                                help="Your Ubuntu server IP or domain"
                            )
                            ssh_user = st.text_input(
                                "SSH Username",
                                value="ubuntu",
                                help="SSH username for the server"
                            )
                        with col_ub2:
                            ssh_key_path = st.text_input(
                                "SSH Key Path",
                                placeholder="/Users/yourname/.ssh/id_rsa",
                                help="Path to your SSH private key"
                            )
                            server_path = st.text_input(
                                "Server Directory",
                                value="/home/ubuntu/agent-app",
                                help="Where to deploy on server"
                            )
                        deployment_config = {
                            "type": "ubuntu_server",
                            "server_ip": server_ip,
                            "username": ssh_user,
                            "key_path": ssh_key_path,
                            "working_directory": server_path
                        }
            
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
                        
                        # Execute task with agent including deployment config
                        task_response = requests.post(
                            "http://localhost:8002/api/v2/agents/execute",
                            json={
                                "task": task_description,
                                "agent_id": agent_type if agent_type != "auto" else None,
                                "context": {
                                    "deployment": deployment_config if deployment_type != "none" else None
                                },
                                "timeout": 60
                            },
                            headers=headers,
                            timeout=30
                        )
                        
                        if task_response.status_code == 200:
                            task_data = task_response.json()
                            task_id = task_data.get("task_id")
                            
                            # Store task ID for checking
                            if 'pending_tasks' not in st.session_state:
                                st.session_state.pending_tasks = []
                            
                            st.session_state.pending_tasks.append({
                                "task_id": task_id,
                                "description": task_description,
                                "agent": agent_type,
                                "deployment": deployment_config if deployment_type != "none" else None,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # Update agent stats for real tracking
                            if 'agent_stats' not in st.session_state:
                                st.session_state.agent_stats = {}
                            
                            # Determine which agent was used
                            actual_agent = task_data.get("metadata", {}).get("agent_id", agent_type)
                            if actual_agent and actual_agent != "auto":
                                if actual_agent not in st.session_state.agent_stats:
                                    st.session_state.agent_stats[actual_agent] = {
                                        "tasks_completed": 0,
                                        "tasks_failed": 0,
                                        "last_active": None
                                    }
                                
                                # Task was just submitted, will check result later
                                st.session_state.agent_stats[actual_agent]["last_active"] = datetime.now()
                            
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
                                        # Check if there's a result even if status is "failed" (Redis error)
                                        result = task_status.get("result", {})
                                        
                                        # Extract the actual AI response from the nested structure
                                        ai_response = None
                                        if isinstance(result, dict) and "output" in result:
                                            output = result["output"]
                                            if isinstance(output, dict):
                                                for agent_key, agent_data in output.items():
                                                    if isinstance(agent_data, dict) and "action" in agent_data:
                                                        action = agent_data["action"]
                                                        if isinstance(action, dict) and "content" in action:
                                                            ai_response = action["content"]
                                                            break
                                        
                                        if ai_response:
                                            # Add to history
                                            st.session_state.task_history.append({
                                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                                "task": task_description[:50] + "..." if len(task_description) > 50 else task_description,
                                                "agent": agent_type,
                                                "result": ai_response
                                            })
                                            
                                            # Update agent stats - task completed successfully
                                            actual_agent = task_status.get("metadata", {}).get("agent_id", agent_type)
                                            if actual_agent and actual_agent != "auto":
                                                if actual_agent in st.session_state.agent_stats:
                                                    st.session_state.agent_stats[actual_agent]["tasks_completed"] += 1
                                            
                                            # Display result
                                            st.success("✅ Task completed!")
                                            st.markdown("### 📋 Result:")
                                            st.write(ai_response)
                                            
                                            # Show execution time if available
                                            if isinstance(result, dict) and "execution_time" in result:
                                                st.caption(f"⏱️ Execution time: {result['execution_time']:.2f} seconds")
                                            
                                            # Remove from pending
                                            st.session_state.pending_tasks = [t for t in st.session_state.pending_tasks if t["task_id"] != task_id]
                                        else:
                                            # Store task for later checking
                                            st.info("Task is processing. Check 'Pending Tasks' below or refresh the page.")
                                    else:
                                        st.info("Task is still processing. Check 'Pending Tasks' below or refresh the page.")
                            except:
                                st.info("Task submitted. Check 'Pending Tasks' below for status.")
                        else:
                            st.error(f"Failed to create task: {task_response.text}")
                            # Update agent stats - task failed
                            # Note: actual_agent would only be defined if task succeeded
                            # For failed tasks, we use the requested agent
                            failed_agent = agent_type if agent_type != "auto" else None
                            if failed_agent:
                                if failed_agent not in st.session_state.agent_stats:
                                    st.session_state.agent_stats[failed_agent] = {
                                        "tasks_completed": 0,
                                        "tasks_failed": 0,
                                        "last_active": None
                                    }
                                st.session_state.agent_stats[failed_agent]["tasks_failed"] += 1
                    
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. The task might still be processing.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Pending Tasks Section
            if 'pending_tasks' in st.session_state and st.session_state.pending_tasks:
                st.subheader("⏳ Pending Tasks")
                
                for pending_task in st.session_state.pending_tasks:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"{pending_task['timestamp']} - {pending_task['description'][:50]}...")
                    with col2:
                        if st.button(f"Check", key=f"check_{pending_task['task_id']}"):
                            try:
                                headers = {
                                    "Authorization": f"Bearer {st.session_state.api_token}",
                                    "Content-Type": "application/json"
                                }
                                status_response = requests.get(
                                    f"http://localhost:8002/api/v1/tasks/{pending_task['task_id']}",
                                    headers=headers,
                                    timeout=5
                                )
                                
                                if status_response.status_code == 200:
                                    task_status = status_response.json()
                                    
                                    if task_status.get("status") in ["completed", "failed"]:
                                        result = task_status.get("result", {})
                                        
                                        # Extract AI response
                                        ai_response = None
                                        if isinstance(result, dict) and "output" in result:
                                            output = result["output"]
                                            if isinstance(output, dict):
                                                for agent_key, agent_data in output.items():
                                                    if isinstance(agent_data, dict) and "action" in agent_data:
                                                        action = agent_data["action"]
                                                        if isinstance(action, dict) and "content" in action:
                                                            ai_response = action["content"]
                                                            break
                                        
                                        if ai_response:
                                            # Add to history
                                            st.session_state.task_history.append({
                                                "timestamp": pending_task['timestamp'],
                                                "task": pending_task['description'][:50] + "..." if len(pending_task['description']) > 50 else pending_task['description'],
                                                "agent": pending_task['agent'],
                                                "result": ai_response
                                            })
                                            
                                            # Remove from pending
                                            st.session_state.pending_tasks = [t for t in st.session_state.pending_tasks if t["task_id"] != pending_task["task_id"]]
                                            
                                            st.success("Task completed! Check Recent Tasks below.")
                                            st.rerun()
                                        else:
                                            st.json(task_status)
                                    else:
                                        st.info("Still processing...")
                            except Exception as e:
                                st.error(f"Error checking task: {str(e)}")
            
            # Chat Interface
            st.subheader("💬 Quick Chat with Agent")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                chat_message = st.text_input(
                    "Message",
                    placeholder="Ask a quick question..."
                )
            with col2:
                chat_agent = st.selectbox(
                    "Agent",
                    options=["researcher", "writer", "reviewer", "optimizer"],
                    key="chat_agent"
                )
            
            if st.button("💬 Send", disabled=not chat_message):
                with st.spinner("Agent thinking..."):
                    try:
                        headers = {
                            "Authorization": f"Bearer {st.session_state.api_token}",
                            "Content-Type": "application/json"
                        }
                        
                        response = requests.post(
                            f"http://localhost:8002/agents/{chat_agent}/chat",
                            json={
                                "message": chat_message,
                                "agent_id": chat_agent,
                                "context": {}
                            },
                            headers=headers,
                            timeout=15
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.markdown(f"**{chat_agent.title()}:** {result.get('response', 'No response')}")
                        else:
                            st.error(f"Chat failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Task History
            if st.session_state.task_history:
                st.subheader("📜 Recent Tasks")
                
                # Create a dataframe for better display
                history_df = pd.DataFrame(st.session_state.task_history)
                
                # Display in a nice table
                for idx, row in history_df.iterrows():
                    with st.expander(f"{row['timestamp']} - {row['task']} (Agent: {row['agent']})"):
                        if isinstance(row['result'], dict):
                            st.json(row['result'])
                        else:
                            st.write(row['result'])
            
            # Example Tasks
            st.subheader("💡 Example Tasks")
            
            examples = [
                ("Research the latest developments in quantum computing", "researcher"),
                ("Write a blog post about AI safety", "writer"),
                ("Review this code: def add(a,b): return a+b", "reviewer"),
                ("Optimize database query performance", "optimizer")
            ]
            
            cols = st.columns(2)
            for idx, (example_task, example_agent) in enumerate(examples):
                with cols[idx % 2]:
                    if st.button(f"Try: {example_task[:30]}...", key=f"example_{idx}"):
                        # Populate the task field
                        st.info(f"Task example loaded! Click 'Submit Task' to execute.")
                        st.session_state.example_task = example_task
                        st.session_state.example_agent = example_agent
        
        else:
            st.warning("⚠️ Not connected to API. Click 'Connect to API' to start assigning tasks.")
            st.markdown("""
            **Requirements:**
            1. Make sure the Production API is running on port 8001
            2. Run: `python -m uvicorn production_api:app --port 8001`
            """)


    def _render_agent_knowledge(self):
        """Render agent knowledge management interface"""
        st.header("🧠 Agent Knowledge Management")
        
        # Import knowledge and agent managers
        try:
            from agent_config import AgentConfigManager
            from knowledge_manager import KnowledgeManager
            
            config_manager = AgentConfigManager()
            knowledge_manager = KnowledgeManager()
        except ImportError:
            st.error("Knowledge management system not available. Run setup_agents.py first.")
            return
        
        # Sidebar for agent selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Select Agent")
            
            # Get list of configured agents
            agent_list = config_manager.list_agents()
            if not agent_list:
                st.warning("No agents configured yet")
                if st.button("Setup Default Agents"):
                    import subprocess
                    subprocess.run(["/Users/jankootstra/miniforge3/bin/python", "setup_agents.py"])
                    st.success("Agents configured! Refresh the page.")
            else:
                selected_agent = st.selectbox(
                    "Agent",
                    options=agent_list,
                    help="Select an agent to manage its knowledge"
                )
                
                # Show agent details
                if selected_agent:
                    agent = config_manager.get_agent(selected_agent)
                    if agent:
                        st.markdown(f"**Role:** {agent.role.value}")
                        st.markdown(f"**Model:** {agent.model}")
                        
                        # Show capabilities
                        capabilities = [k.replace("can_", "").replace("_", " ").title() 
                                      for k, v in agent.capabilities.__dict__.items() if v]
                        st.markdown(f"**Capabilities:** {', '.join(capabilities)}")
        
        with col2:
            if agent_list and selected_agent:
                # Knowledge management tabs
                kb_tab1, kb_tab2, kb_tab3, kb_tab4, kb_tab5 = st.tabs([
                    "📚 View Knowledge",
                    "➕ Add Knowledge",
                    "🎓 Train Agent",
                    "🔍 Search Knowledge",
                    "📊 Statistics"
                ])
                
                with kb_tab1:
                    st.subheader("📚 Knowledge Base")
                    
                    # Get knowledge items
                    items = knowledge_manager.knowledge_bases.get(selected_agent, [])
                    
                    if items:
                        # Group by category
                        categories = {}
                        for item in items:
                            if item.category not in categories:
                                categories[item.category] = []
                            categories[item.category].append(item)
                        
                        # Display by category
                        for category, cat_items in categories.items():
                            with st.expander(f"{category} ({len(cat_items)} items)"):
                                for item in cat_items[:5]:  # Show first 5
                                    st.markdown(f"**Source:** {item.source}")
                                    st.code(item.content[:200] + "..." if len(item.content) > 200 else item.content)
                                    st.caption(f"Usage: {item.usage_count} times | Relevance: {item.relevance_score:.2f}")
                    else:
                        st.info("No knowledge items yet. Add some in the 'Add Knowledge' tab!")
                
                with kb_tab2:
                    st.subheader("➕ Add New Knowledge")
                    
                    # Add knowledge form
                    category = st.selectbox(
                        "Category",
                        options=[
                            "technical_documentation",
                            "code_examples",
                            "best_practices",
                            "troubleshooting",
                            "architecture_patterns",
                            "api_references",
                            "tutorials",
                            "project_specific",
                            "domain_knowledge"
                        ]
                    )
                    
                    content = st.text_area(
                        "Knowledge Content",
                        placeholder="Enter the knowledge item here...",
                        height=200
                    )
                    
                    source = st.text_input(
                        "Source",
                        placeholder="e.g., official_docs, experience, stackoverflow"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Add Knowledge", type="primary", disabled=not content):
                            item = knowledge_manager.add_knowledge(
                                selected_agent,
                                category,
                                content,
                                source or "manual"
                            )
                            st.success(f"✅ Added knowledge item: {item.id}")
                            st.rerun()
                    
                    with col2:
                        # Upload file option
                        uploaded_file = st.file_uploader(
                            "Or upload a file",
                            type=["txt", "md", "json", "py", "js", "java", "cpp"]
                        )
                        
                        if uploaded_file is not None:
                            content = uploaded_file.read().decode("utf-8")
                            if st.button("Import from file"):
                                item = knowledge_manager.add_knowledge(
                                    selected_agent,
                                    "project_specific",
                                    content,
                                    f"uploaded:{uploaded_file.name}"
                                )
                                st.success(f"✅ Imported knowledge from {uploaded_file.name}")
                                st.rerun()
                
                with kb_tab3:
                    st.subheader("🎓 Train Agent with Knowledge")
                    
                    # Import knowledge trainer
                    try:
                        from knowledge_trainer import KnowledgeTrainer
                        trainer = KnowledgeTrainer()
                    except ImportError:
                        st.error("Knowledge trainer not available. Please ensure knowledge_trainer.py is installed.")
                        trainer = None
                    
                    if trainer:
                        # Get consumption stats
                        stats = trainer.get_consumption_stats(selected_agent)
                        
                        # Show current status
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Knowledge", stats["total_knowledge_items"])
                        with col2:
                            st.metric("New Items", stats["new_knowledge_available"])
                        with col3:
                            last_consumption = stats.get("last_consumption")
                            if last_consumption:
                                from datetime import datetime
                                time_ago = (datetime.now() - last_consumption).total_seconds() / 3600
                                st.metric("Last Trained", f"{time_ago:.1f}h ago")
                            else:
                                st.metric("Last Trained", "Never")
                        
                        st.markdown("---")
                        
                        # Training options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🔄 Consume New Knowledge")
                            st.markdown("Make the agent consume and integrate new knowledge items into its configuration.")
                            
                            force_all = st.checkbox("Process all knowledge (not just new)", value=False)
                            
                            if st.button("🧠 Consume Knowledge", type="primary"):
                                with st.spinner("Processing knowledge..."):
                                    result = trainer.consume_knowledge(selected_agent, force_all)
                                    
                                    if result.knowledge_integrated > 0:
                                        st.success(f"✅ Successfully consumed {result.knowledge_consumed} items, integrated {result.knowledge_integrated}")
                                        if result.improvements:
                                            st.info("Improvements: " + ", ".join(result.improvements))
                                    elif result.knowledge_consumed == 0:
                                        st.info("No new knowledge to consume")
                                    else:
                                        st.warning("Knowledge consumed but not integrated")
                                    
                                    if result.errors:
                                        st.error("Errors: " + ", ".join(result.errors))
                        
                        with col2:
                            st.markdown("### 🎯 Test Training")
                            st.markdown("Test the agent with sample queries to verify knowledge integration.")
                            
                            test_queries = st.text_area(
                                "Test Queries (one per line)",
                                placeholder="How do I optimize a database query?\nWhat's the best caching strategy?\nHow to handle authentication?",
                                height=100
                            )
                            
                            if st.button("🧪 Run Training Test"):
                                if test_queries:
                                    queries = [q.strip() for q in test_queries.split('\n') if q.strip()]
                                    with st.spinner(f"Testing {len(queries)} queries..."):
                                        import asyncio
                                        try:
                                            results = asyncio.run(trainer.active_training_session(selected_agent, queries))
                                            
                                            st.success(f"Tested {results['queries_tested']} queries")
                                            st.info(f"Knowledge applied in {results['knowledge_applied']} responses")
                                            
                                            # Show responses
                                            for i, response in enumerate(results.get('responses', [])):
                                                with st.expander(f"Query {i+1}: {response.get('query', '')}"):
                                                    if 'error' in response:
                                                        st.error(response['error'])
                                                    else:
                                                        st.markdown(f"**Knowledge Used:** {response.get('knowledge_used', 0)} items")
                                                        # Show full response or at least 2000 chars
                                                        response_text = response.get('response', '')
                                                        if len(response_text) > 2000:
                                                            st.markdown(response_text[:2000] + "...")
                                                            with st.expander("Show full response"):
                                                                st.markdown(response_text)
                                                        else:
                                                            st.markdown(response_text)
                                        except Exception as e:
                                            st.error(f"Training test failed: {str(e)}")
                                else:
                                    st.warning("Please enter test queries")
                        
                        # Knowledge by category chart
                        if stats.get("knowledge_by_category"):
                            st.markdown("### 📊 Knowledge Distribution")
                            import pandas as pd
                            df = pd.DataFrame(
                                list(stats["knowledge_by_category"].items()),
                                columns=["Category", "Count"]
                            )
                            st.bar_chart(df.set_index("Category"))
                        
                        # Auto-train all agents button
                        st.markdown("---")
                        st.markdown("### 🚀 Bulk Operations")
                        if st.button("🔄 Auto-Train All Agents"):
                            with st.spinner("Training all agents..."):
                                results = trainer.auto_consume_all_agents()
                                
                                success_count = sum(1 for r in results.values() if r.knowledge_integrated > 0)
                                st.success(f"✅ Trained {success_count} agents successfully")
                                
                                for agent_name, result in results.items():
                                    if result.knowledge_integrated > 0:
                                        st.info(f"{agent_name}: Consumed {result.knowledge_consumed}, integrated {result.knowledge_integrated}")
                                    elif result.errors:
                                        st.warning(f"{agent_name}: {', '.join(result.errors)}")
                
                with kb_tab4:
                    st.subheader("🔍 Search Knowledge")
                    
                    query = st.text_input(
                        "Search Query",
                        placeholder="Enter keywords to search..."
                    )
                    
                    search_category = st.selectbox(
                        "Filter by Category (optional)",
                        options=["All"] + knowledge_manager.categories
                    )
                    
                    if st.button("Search", disabled=not query):
                        results = knowledge_manager.search_knowledge(
                            selected_agent,
                            query,
                            category=search_category if search_category != "All" else None,
                            limit=10
                        )
                        
                        if results:
                            st.success(f"Found {len(results)} matching items:")
                            for item in results:
                                with st.expander(f"[{item.category}] {item.source}"):
                                    st.code(item.content)
                                    st.caption(f"Relevance: {item.relevance_score:.2f} | Used: {item.usage_count} times")
                        else:
                            st.info("No matching knowledge items found")
                
                with kb_tab5:
                    st.subheader("📊 Knowledge Statistics")
                    
                    stats = knowledge_manager.get_statistics(selected_agent)
                    
                    if stats.get("total_items", 0) > 0:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Items", stats["total_items"])
                        with col2:
                            st.metric("Total Usage", stats["total_usage"])
                        with col3:
                            st.metric("Avg Usage", f"{stats['average_usage']:.1f}")
                        
                        # Category distribution
                        if stats.get("categories"):
                            st.subheader("Category Distribution")
                            category_df = pd.DataFrame(
                                list(stats["categories"].items()),
                                columns=["Category", "Count"]
                            )
                            fig = px.pie(category_df, values="Count", names="Category", 
                                       title="Knowledge by Category")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Most used items
                        if stats.get("most_used"):
                            st.subheader("Most Used Knowledge")
                            for item in stats["most_used"]:
                                st.markdown(f"- [{item.category}] {item.content[:100]}... (Used: {item.usage_count} times)")
                        
                        # Export/Import options
                        st.subheader("Data Management")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Export Knowledge Base"):
                                export_file = f"{selected_agent}_knowledge.json"
                                knowledge_manager.export_knowledge_base(selected_agent, export_file)
                                st.success(f"✅ Exported to {export_file}")
                        
                        with col2:
                            st.info("Use the 'Add Knowledge' tab to import files")
                    else:
                        st.info("No knowledge items yet for this agent")
    
    def _render_project_config(self):
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
                # Create new project form
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
            
            else:
                # Display and edit existing project
                project = manager.get_project(selected_project)
                if project:
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
                    if project.deployment_targets:
                        for i, target in enumerate(project.deployment_targets):
                            with st.expander(f"{target.name} ({target.type})" + (" ⭐" if target.is_default else "")):
                                # Create edit form for this target
                                edit_key = f"edit_{selected_project}_{i}"
                                
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
    
    def _render_visual_code_builder(self):
        """Render Visual Code Builder interface"""
        st.header("🎨 Visual Code Builder")
        
        if not VISUAL_CODE_BUILDER_AVAILABLE:
            st.error("Visual Code Builder components not available. Please ensure all components are installed.")
            return
        
        # Initialize components in session state
        if 'visual_program' not in st.session_state:
            st.session_state.visual_program = VisualProgram(name="Agent Task")
            st.session_state.component_library = ComponentLibrary()
            st.session_state.translator = VisualToCodeTranslator()
            st.session_state.preview_panel = CodePreviewPanel(
                PreviewSettings(theme=PreviewTheme.MONOKAI)
            )
            st.session_state.block_factory = BlockFactory()
        
        # Sidebar for component library
        with st.sidebar:
            st.subheader("📦 Component Library")
            
            # Component categories
            category = st.selectbox(
                "Category",
                [cat.value for cat in ComponentCategory],
                key="component_category"
            )
            
            # Show components in selected category
            selected_category = ComponentCategory(category)
            templates = st.session_state.component_library.get_templates_by_category(selected_category)
            
            st.write(f"**{len(templates)} components available**")
            
            # Component search
            search_term = st.text_input("🔍 Search components", key="component_search")
            if search_term:
                templates = st.session_state.component_library.search_templates(search_term)
            
            # Display component templates
            for template in templates[:10]:  # Limit to 10 for performance
                with st.expander(f"{template.icon} {template.name}"):
                    st.write(template.description)
                    if st.button(f"Add {template.name}", key=f"add_{template.template_id}"):
                        block = template.create_instance()
                        st.session_state.visual_program.add_block(block)
                        st.success(f"Added {template.name} block")
        
        # Main interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Visual Program")
            
            # Program info
            program_name = st.text_input(
                "Program Name",
                value=st.session_state.visual_program.name,
                key="program_name"
            )
            st.session_state.visual_program.name = program_name
            
            # Quick add common blocks
            st.write("**Quick Add Blocks:**")
            button_col1, button_col2, button_col3, button_col4 = st.columns(4)
            
            with button_col1:
                if st.button("➕ Function"):
                    block = st.session_state.block_factory.create_function_block()
                    st.session_state.visual_program.add_block(block)
                    st.success("Added Function block")
            
            with button_col2:
                if st.button("❓ If-Else"):
                    block = st.session_state.block_factory.create_if_block()
                    st.session_state.visual_program.add_block(block)
                    st.success("Added If-Else block")
            
            with button_col3:
                if st.button("🔁 For Loop"):
                    block = st.session_state.block_factory.create_for_loop_block()
                    st.session_state.visual_program.add_block(block)
                    st.success("Added For Loop block")
            
            with button_col4:
                if st.button("📦 Variable"):
                    block = st.session_state.block_factory.create_variable_block()
                    st.session_state.visual_program.add_block(block)
                    st.success("Added Variable block")
            
            # Display current blocks
            st.write(f"**Current Blocks ({len(st.session_state.visual_program.blocks)}):**")
            
            for i, block in enumerate(st.session_state.visual_program.blocks):
                with st.expander(f"{block.icon} {block.title} ({block.block_type.value})"):
                    # Block properties
                    for prop_name, prop_value in block.properties.items():
                        new_value = st.text_input(
                            prop_name.replace('_', ' ').title(),
                            value=str(prop_value),
                            key=f"prop_{block.block_id}_{prop_name}"
                        )
                        block.properties[prop_name] = new_value
                    
                    # Delete block button
                    if st.button(f"🗑️ Delete", key=f"delete_{block.block_id}"):
                        st.session_state.visual_program.blocks.remove(block)
                        st.rerun()
            
            # Connections (simplified for Streamlit)
            if len(st.session_state.visual_program.blocks) >= 2:
                st.write("**Connect Blocks:**")
                block_names = [f"{b.title} ({i})" for i, b in enumerate(st.session_state.visual_program.blocks)]
                
                col_from, col_to = st.columns(2)
                with col_from:
                    from_block = st.selectbox("From Block", block_names, key="conn_from")
                with col_to:
                    to_block = st.selectbox("To Block", block_names, key="conn_to")
                
                if st.button("🔗 Connect"):
                    from_idx = block_names.index(from_block)
                    to_idx = block_names.index(to_block)
                    from_block_obj = st.session_state.visual_program.blocks[from_idx]
                    to_block_obj = st.session_state.visual_program.blocks[to_idx]
                    
                    # Simplified connection - connect first output to first input
                    if from_block_obj.output_ports and to_block_obj.input_ports:
                        st.session_state.visual_program.connect_blocks(
                            from_block_obj.block_id,
                            from_block_obj.output_ports[0].name,
                            to_block_obj.block_id,
                            to_block_obj.input_ports[0].name
                        )
                        st.success(f"Connected {from_block} to {to_block}")
        
        with col2:
            st.subheader("📝 Generated Code")
            
            # Target language selection
            language = st.selectbox(
                "Target Language",
                ["python", "javascript", "typescript"],
                key="target_language"
            )
            
            target_lang = TargetLanguage(language)
            
            # Generate code button
            if st.button("🚀 Generate Code"):
                try:
                    # Generate code
                    code = st.session_state.translator.translate_program(
                        st.session_state.visual_program,
                        target_lang
                    )
                    
                    # Validate
                    valid, errors = st.session_state.translator.validate_translation(code, target_lang)
                    
                    if valid:
                        st.success("✅ Code generated successfully!")
                    else:
                        st.warning("⚠️ Generated code has syntax issues:")
                        for error in errors:
                            st.error(error)
                    
                    # Display code
                    st.code(code, language=language)
                    
                    # Download button
                    file_extension = "py" if language == "python" else "js" if language == "javascript" else "ts"
                    st.download_button(
                        label=f"📥 Download {program_name}.{file_extension}",
                        data=code,
                        file_name=f"{program_name}.{file_extension}",
                        mime="text/plain"
                    )
                    
                    # Statistics
                    lines = code.split('\n')
                    st.info(f"📊 Generated {len(lines)} lines of {language} code")
                    
                except Exception as e:
                    st.error(f"Error generating code: {str(e)}")
            
            # Code preview placeholder
            if not st.session_state.visual_program.blocks:
                st.info("Add blocks to generate code")
            else:
                st.write("**Program Structure:**")
                execution_order = st.session_state.visual_program.get_execution_order()
                for i, block in enumerate(execution_order, 1):
                    st.write(f"{i}. {block.icon} {block.title}")
        
        # Templates section
        st.divider()
        st.subheader("📚 Code Templates")
        
        template_col1, template_col2, template_col3 = st.columns(3)
        
        with template_col1:
            if st.button("🔄 Data Processing Pipeline"):
                # Clear current program
                st.session_state.visual_program = VisualProgram(name="Data Pipeline")
                
                # Add function block
                func = st.session_state.block_factory.create_function_block()
                func.properties["function_name"] = "process_data"
                func.properties["parameters"] = ["data"]
                st.session_state.visual_program.add_block(func)
                
                # Add for loop
                loop = st.session_state.block_factory.create_for_loop_block()
                loop.properties["variable_name"] = "item"
                st.session_state.visual_program.add_block(loop)
                
                # Add output
                output = st.session_state.block_factory.create_output_block()
                st.session_state.visual_program.add_block(output)
                
                st.success("Created Data Processing Pipeline template")
                st.rerun()
        
        with template_col2:
            if st.button("🌐 API Handler"):
                st.session_state.visual_program = VisualProgram(name="API Handler")
                
                # Add API call block
                api_block = st.session_state.block_factory.create_api_call_block()
                api_block.properties["url"] = "https://api.example.com/data"
                api_block.properties["method"] = "GET"
                st.session_state.visual_program.add_block(api_block)
                
                # Add if block for error checking
                if_block = st.session_state.block_factory.create_if_block()
                if_block.properties["condition_expression"] = "response.status == 200"
                st.session_state.visual_program.add_block(if_block)
                
                st.success("Created API Handler template")
                st.rerun()
        
        with template_col3:
            if st.button("🤖 Agent Task"):
                st.session_state.visual_program = VisualProgram(name="Agent Task")
                
                # Add function for agent task
                func = st.session_state.block_factory.create_function_block()
                func.properties["function_name"] = "execute_agent_task"
                func.properties["parameters"] = ["agent_id", "task"]
                st.session_state.visual_program.add_block(func)
                
                # Add variable for result
                var = st.session_state.block_factory.create_variable_block()
                var.properties["variable_name"] = "result"
                st.session_state.visual_program.add_block(var)
                
                st.success("Created Agent Task template")
                st.rerun()
        
        # Visual Debugger Section
        st.divider()
        st.subheader("🐛 Visual Debugger")
        
        # Initialize debugger if needed
        if 'visual_debugger' not in st.session_state:
            st.session_state.visual_debugger = VisualDebugger(st.session_state.visual_program)
        
        # Update debugger with current program
        st.session_state.visual_debugger.set_program(st.session_state.visual_program)
        
        debug_col1, debug_col2, debug_col3 = st.columns([2, 3, 2])
        
        with debug_col1:
            st.write("**🎮 Debug Controls**")
            
            # Debug state display
            state_colors = {
                DebugState.IDLE: "🔵",
                DebugState.RUNNING: "🟢",
                DebugState.PAUSED: "🟡",
                DebugState.ERROR: "🔴",
                DebugState.STEPPING: "🟠",
                DebugState.STOPPED: "⚫"
            }
            current_state = st.session_state.visual_debugger.state
            st.info(f"Status: {state_colors.get(current_state, '⚪')} **{current_state.value.upper()}**")
            
            # Control buttons
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("▶️ Start", disabled=current_state != DebugState.IDLE, key="debug_start"):
                    import asyncio
                    try:
                        asyncio.run(st.session_state.visual_debugger.start_debugging())
                    except:
                        pass
                    st.rerun()
                
                if st.button("⏸️ Pause", disabled=current_state != DebugState.RUNNING, key="debug_pause"):
                    st.session_state.visual_debugger.pause_debugging()
                    st.rerun()
            
            with btn_col2:
                if st.button("⏹️ Stop", disabled=current_state == DebugState.IDLE, key="debug_stop"):
                    st.session_state.visual_debugger.stop_debugging()
                    st.rerun()
                
                if st.button("⏭️ Step", disabled=current_state != DebugState.PAUSED, key="debug_step"):
                    st.session_state.visual_debugger.step_over()
                    st.rerun()
            
            # Breakpoints section
            st.write("**🔴 Breakpoints**")
            if st.session_state.visual_program.blocks:
                block_names = [f"{b.title}" for b in st.session_state.visual_program.blocks]
                selected_idx = st.selectbox(
                    "Add breakpoint to:",
                    range(len(block_names)),
                    format_func=lambda x: block_names[x],
                    key="bp_select"
                )
                
                if st.button("➕ Add Breakpoint", key="add_bp"):
                    block = st.session_state.visual_program.blocks[selected_idx]
                    bp = st.session_state.visual_debugger.add_breakpoint(
                        block_id=block.block_id,
                        breakpoint_type=BreakpointType.LINE
                    )
                    st.success(f"Added breakpoint")
                    st.rerun()
            
            # List current breakpoints
            if st.session_state.visual_debugger.breakpoints:
                st.write("Active breakpoints:")
                for i, bp in enumerate(st.session_state.visual_debugger.breakpoints):
                    if bp.block_id and bp.enabled:
                        block = next((b for b in st.session_state.visual_program.blocks 
                                    if b.block_id == bp.block_id), None)
                        if block:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"• {block.title}")
                            with col2:
                                if st.button("❌", key=f"rm_{i}"):
                                    st.session_state.visual_debugger.remove_breakpoint(bp.breakpoint_id)
                                    st.rerun()
        
        with debug_col2:
            st.write("**📊 Debug Information**")
            
            # Variables display
            variables = st.session_state.visual_debugger.get_variables()
            if variables:
                with st.expander("🔍 Variables", expanded=True):
                    for name, value in list(variables.items())[:10]:
                        st.code(f"{name} = {str(value)[:100]}", language="python")
            
            # Call stack
            call_stack = st.session_state.visual_debugger.get_call_stack()
            if call_stack:
                with st.expander("📚 Call Stack", expanded=True):
                    for frame in call_stack[:5]:
                        st.write(f"→ {frame['function']}")
            
            # Watch expressions
            with st.expander("👁️ Watch Expressions"):
                expr = st.text_input("Add expression:", key="watch_input")
                if st.button("Add Watch", key="add_watch"):
                    st.session_state.visual_debugger.watcher.add_expression(expr)
                    st.success(f"Watching: {expr}")
                
                # Show watched expressions
                if st.session_state.visual_debugger.watcher.watch_expressions:
                    st.write("Watched:")
                    for expr in st.session_state.visual_debugger.watcher.watch_expressions:
                        result = st.session_state.visual_debugger.evaluate_expression(expr)
                        st.code(f"{expr} = {result}", language="python")
        
        with debug_col3:
            st.write("**📜 Execution Timeline**")
            
            timeline = st.session_state.visual_debugger.get_execution_timeline()
            if timeline:
                with st.expander("Recent Execution", expanded=True):
                    for trace in timeline[-5:]:
                        if trace.get('block_id'):
                            block = next((b for b in st.session_state.visual_program.blocks 
                                        if b.block_id == trace['block_id']), None)
                            if block:
                                st.write(f"• {block.title}")
                                if trace.get('output'):
                                    st.code(trace['output'], language="text")
            else:
                st.info("No execution history")
            
            # Performance metrics
            if current_state != DebugState.IDLE:
                perf = st.session_state.visual_debugger.get_performance_metrics()
                col1, col2 = st.columns(2)
                with col1:
                    if perf.get('total_duration'):
                        st.metric("Duration", f"{perf['total_duration']:.2f}s")
                with col2:
                    if perf.get('block_statistics'):
                        st.metric("Blocks", len(perf['block_statistics']))
            
            # Error display
            if st.session_state.visual_debugger.last_error:
                st.error(f"Error: {st.session_state.visual_debugger.last_error.get('message', 'Unknown error')}")


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