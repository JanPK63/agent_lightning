"""
Metrics collection and visualization for Agent Lightning Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .models import MetricSnapshot, DashboardConfig


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

    def get_latest_metric(self, metric_name: str) -> float:
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

    def calculate_delta(self, metric_name: str, percentage: bool = False) -> str:
        """Calculate metric delta"""
        # Simulated for demo
        delta = np.random.random() * 0.1 - 0.05
        if percentage:
            return f"{delta:.1%}"
        else:
            return f"{delta:.3f}"

    def get_agent_status(self, agent_id: str) -> Dict:
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

    def get_metric_history(self, metric_name: str, window_seconds: int = 300) -> pd.DataFrame:
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

    def create_line_chart(self, metric: str, agents: List[str],
                         window: int, title: str, y_label: str) -> go.Figure:
        """Create line chart for metric"""
        fig = go.Figure()

        for agent in agents:
            data = self.get_metric_history(metric, window)
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

    def create_gauge(self, value: float, title: str,
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

    def create_agent_comparison_chart(self, agents: List[str]) -> go.Figure:
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

    def create_response_time_histogram(self, agents: List[str], window: int) -> go.Figure:
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

    def create_completion_heatmap(self, agents: List[str]) -> go.Figure:
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

    def calculate_trends(self, agents: List[str], window: int) -> Dict:
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

    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate metric correlations"""
        # Simulated correlation matrix
        metrics = ["Loss", "Reward", "Accuracy", "Response Time", "Error Rate"]

        n = len(metrics)
        corr_matrix = np.random.random((n, n))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)  # Diagonal = 1

        return pd.DataFrame(corr_matrix, index=metrics, columns=metrics)

    def generate_predictions(self, agents: List[str]) -> Dict:
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