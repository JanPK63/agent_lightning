"""
Main Dashboard Orchestrator for Agent Lightning
Brings together all dashboard modules in a clean, modular architecture
"""

import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from .models import DashboardConfig
from .metrics import MetricsCollector
from .task_assignment import TaskAssignmentInterface
from .agent_knowledge import AgentKnowledgeInterface
from .project_config import ProjectConfigInterface
from .visual_code_builder import VisualCodeBuilderInterface
from .spec_driven_dev import SpecDrivenDevInterface


class MonitoringDashboard:
    """
    Main monitoring dashboard orchestrator
    Coordinates all dashboard modules and provides unified interface
    """

    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.collector = MetricsCollector(config)

        # Initialize observability (disabled due to threading issues)
        self.observability = None

        # Set alert thresholds
        self.config.alert_thresholds = {
            "loss": {"max": 2.0},
            "error_rate": {"max": 0.1},
            "response_time": {"max": 5.0},
            "memory_usage": {"max": 0.9}
        }

        # Initialize memory manager
        try:
            from postgres_memory_manager import PostgreSQLMemoryManager
            self.memory_manager = PostgreSQLMemoryManager()
        except:
            self.memory_manager = None

        # Initialize agent service orchestrator
        self.orchestrator = None
        try:
            from agent_service_orchestrator import get_orchestrator, initialize_orchestrator
            import asyncio

            # Initialize orchestrator in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(initialize_orchestrator())
            self.orchestrator = get_orchestrator()
            print("✅ Agent Service Orchestrator initialized")
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            self.orchestrator = None

        # Initialize interface modules
        self.task_interface = TaskAssignmentInterface(self.config)
        self.knowledge_interface = AgentKnowledgeInterface(self.config)
        self.project_interface = ProjectConfigInterface(self.config)
        self.visual_interface = VisualCodeBuilderInterface(self.config)
        self.spec_interface = SpecDrivenDevInterface(self.config)

        print(f"📊 Monitoring Dashboard initialized")
        print(f"   Dashboard port: {config.dashboard_port}")
        print(f"   Refresh interval: {config.refresh_interval}s")
        print(f"   Orchestrator: {'✅ Available' if self.orchestrator else '❌ Not available'}")

    def create_dashboard(self):
        """Create the main Streamlit dashboard"""
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

        # Sidebar configuration
        with st.sidebar:
            self._render_sidebar_config()

        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "📈 Training Metrics",
            "🤖 Agent Performance",
            "💻 System Resources",
            "🔔 Alerts",
            "📊 Analytics",
            "🎯 Task Assignment",
            "🧠 Agent Knowledge",
            "⚙️ Project Config",
            "🎨 Visual Code Builder",
            "📋 Spec Driven Development"
        ])

        with tab1:
            self._render_training_metrics()
        with tab2:
            self._render_agent_performance()
        with tab3:
            self._render_system_resources()
        with tab4:
            self._render_alerts()
        with tab5:
            self._render_analytics()
        with tab6:
            self.task_interface._render_task_assignment()
        with tab7:
            self.knowledge_interface.render_knowledge_management()
        with tab8:
            self.project_interface.render_project_config()
        with tab9:
            self.visual_interface.render_visual_code_builder()
        with tab10:
            self.spec_interface.render_spec_driven_development()

        # Auto-refresh logic
        if st.session_state.get('auto_refresh', True):
            time.sleep(self.config.refresh_interval)
            st.rerun()

    def _render_sidebar_config(self):
        """Render sidebar configuration"""
        st.header("Configuration")

        # Refresh settings
        auto_refresh = st.checkbox("Auto Refresh", value=True, key="auto_refresh")
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
        available_agents = self.task_interface.get_available_agents()
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

        # Store in session state for use by other components
        st.session_state.selected_agents = selected_agents
        st.session_state.selected_metrics = selected_metrics
        st.session_state.time_window = time_window

    def _render_training_metrics(self):
        """Render training metrics tab"""
        st.header("Training Metrics")

        selected_agents = st.session_state.get('selected_agents', [])
        selected_metrics = st.session_state.get('selected_metrics', [])
        time_window = st.session_state.get('time_window', 300)

        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            latest_loss = self.collector.get_latest_metric("loss")
            st.metric(
                label="Current Loss",
                value=f"{latest_loss:.4f}",
                delta=self.collector.calculate_delta("loss")
            )

        with col2:
            latest_reward = self.collector.get_latest_metric("reward")
            st.metric(
                label="Average Reward",
                value=f"{latest_reward:.3f}",
                delta=self.collector.calculate_delta("reward")
            )

        with col3:
            latest_accuracy = self.collector.get_latest_metric("accuracy")
            st.metric(
                label="Accuracy",
                value=f"{latest_accuracy:.1%}",
                delta=self.collector.calculate_delta("accuracy", percentage=True)
            )

        with col4:
            learning_rate = self.collector.get_latest_metric("learning_rate")
            st.metric(
                label="Learning Rate",
                value=f"{learning_rate:.6f}",
                delta=None
            )

        # Training curves
        st.subheader("Training Progress")

        # Loss curve
        if "loss" in selected_metrics:
            fig_loss = self.collector.create_line_chart(
                "loss", selected_agents, time_window,
                title="Loss Over Time",
                y_label="Loss"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        # Reward curve
        if "reward" in selected_metrics:
            fig_reward = self.collector.create_line_chart(
                "reward", selected_agents, time_window,
                title="Reward Over Time",
                y_label="Reward"
            )
            st.plotly_chart(fig_reward, use_container_width=True)

        # Accuracy curve
        if "accuracy" in selected_metrics:
            fig_accuracy = self.collector.create_line_chart(
                "accuracy", selected_agents, time_window,
                title="Accuracy Over Time",
                y_label="Accuracy (%)"
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)

    def _render_agent_performance(self):
        """Render agent performance tab"""
        st.header("Agent Performance")

        selected_agents = st.session_state.get('selected_agents', [])
        time_window = st.session_state.get('time_window', 300)

        # Agent status grid
        st.subheader("Agent Status")

        cols = st.columns(min(len(selected_agents), 4))
        for i, agent in enumerate(selected_agents[:4]):
            with cols[i]:
                status = self.collector.get_agent_status(agent)
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

        fig_comparison = self.collector.create_agent_comparison_chart(selected_agents)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Response time distribution
        st.subheader("Response Time Distribution")

        fig_response = self.collector.create_response_time_histogram(selected_agents, time_window)
        st.plotly_chart(fig_response, use_container_width=True)

        # Task completion heatmap
        st.subheader("Task Completion Heatmap")

        fig_heatmap = self.collector.create_completion_heatmap(selected_agents)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    def _render_system_resources(self):
        """Render system resources tab"""
        st.header("System Resources")

        # Resource gauges
        col1, col2, col3 = st.columns(3)

        with col1:
            cpu_usage = self.collector.get_latest_metric("cpu_usage")
            fig_cpu = self.collector.create_gauge(
                cpu_usage * 100,
                "CPU Usage",
                max_value=100,
                unit="%"
            )
            st.plotly_chart(fig_cpu, use_container_width=True)

        with col2:
            memory_usage = self.collector.get_latest_metric("memory_usage")
            fig_memory = self.collector.create_gauge(
                memory_usage * 100,
                "Memory Usage",
                max_value=100,
                unit="%"
            )
            st.plotly_chart(fig_memory, use_container_width=True)

        with col3:
            gpu_usage = self.collector.get_latest_metric("gpu_usage")
            fig_gpu = self.collector.create_gauge(
                gpu_usage * 100,
                "GPU Usage",
                max_value=100,
                unit="%"
            )
            st.plotly_chart(fig_gpu, use_container_width=True)

        # Resource timeline
        st.subheader("Resource Usage Over Time")

        from plotly.subplots import make_subplots
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
            data = self.collector.get_metric_history(metric, window_seconds=1800)
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

        # Database Connection Pool Metrics
        st.subheader("🗄️ Database Connection Pool")

        try:
            from shared.database import db_manager

            # Get pool statistics
            pool_stats = db_manager.get_pool_stats()

            if pool_stats.get("error"):
                st.warning(f"Pool stats unavailable: {pool_stats['error']}")
            elif pool_stats.get("type") == "mongodb":
                st.info("🗄️ MongoDB uses native connection pooling")
                st.metric("Pool Type", "MongoDB Native")
            else:
                # Display pool metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Active Connections",
                        pool_stats.get("checkedout", 0),
                        help="Currently active database connections"
                    )

                with col2:
                    st.metric(
                        "Available Connections",
                        pool_stats.get("connections_in_pool", 0),
                        help="Connections available in pool"
                    )

                with col3:
                    st.metric(
                        "Pool Size",
                        pool_stats.get("pool_size", 0),
                        help="Total configured pool size"
                    )

                with col4:
                    st.metric(
                        "Overflow",
                        pool_stats.get("connections_overflow", 0),
                        help="Connections created beyond pool size"
                    )

                # Pool utilization gauge
                total_connections = pool_stats.get("checkedout", 0) + pool_stats.get("connections_in_pool", 0)
                pool_size = pool_stats.get("pool_size", 1)
                utilization = (total_connections / pool_size) * 100 if pool_size > 0 else 0

                fig_pool = self.collector.create_gauge(
                    utilization,
                    "Pool Utilization",
                    max_value=100,
                    unit="%",
                    color="normal" if utilization < 80 else "warning" if utilization < 95 else "danger"
                )
                st.plotly_chart(fig_pool, use_container_width=True)

                # Connection info
                conn_info = db_manager.get_connection_info()
                with st.expander("Connection Details"):
                    st.write(f"**Database Type:** {conn_info.get('type', 'Unknown').title()}")
                    st.write(f"**Pool Class:** {conn_info.get('pool_class', 'N/A')}")
                    st.write(f"**Pool Size:** {conn_info.get('pool_size', 'N/A')}")
                    st.write(f"**Max Overflow:** {conn_info.get('max_overflow', 'N/A')}")
                    st.write(f"**Timeout:** {conn_info.get('pool_timeout', 'N/A')}s")
                    st.write(f"**Recycle:** {conn_info.get('pool_recycle', 'N/A')}s")
                    st.write(f"**Pre-ping:** {conn_info.get('pool_pre_ping', 'N/A')}")

        except Exception as e:
            st.error(f"Error loading database metrics: {str(e)}")
            st.info("Database connection pooling metrics require SQLAlchemy engine to be initialized")

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
                timestamp = alert.get("timestamp", datetime.now())

                st.markdown(f"""
                <div class="alert-box">
                    {icon} <strong>{timestamp.strftime('%H:%M:%S')}</strong> -
                    Agent: {alert['agent_id']} |
                    Metric: {alert['metric']} |
                    Value: {alert['value']:.3f} |
                    Threshold: {alert.get('threshold', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No alerts at this time")

    def _render_analytics(self):
        """Render analytics tab"""
        st.header("📊 Analytics & Insights")

        selected_agents = st.session_state.get('selected_agents', [])
        time_window = st.session_state.get('time_window', 300)

        # Performance trends
        st.subheader("Performance Trends")

        trend_data = self.collector.calculate_trends(selected_agents, time_window)

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

        correlation_matrix = self.collector.calculate_correlations()
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

        predictions = self.collector.generate_predictions(selected_agents)

        fig_pred = go.Figure()
        for agent in selected_agents:
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
if __name__ == "__main__":
    config = DashboardConfig()
    dashboard = MonitoringDashboard(config)
    dashboard.create_dashboard()
else:
    # For imports
    config = DashboardConfig()
    dashboard = MonitoringDashboard(config)
    dashboard.create_dashboard()