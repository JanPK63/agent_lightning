#!/usr/bin/env python3
"""
Grafana Dashboard Creator
Creates comprehensive dashboards for the Agent System metrics visualization
"""

import os
import json
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class PanelConfig:
    """Configuration for a Grafana panel"""
    title: str
    panel_type: str  # graph, gauge, stat, table, heatmap, etc.
    datasource: str
    query: str
    grid_pos: Dict[str, int]  # x, y, w, h
    unit: Optional[str] = None
    thresholds: Optional[Dict] = None
    legend: bool = True
    transparent: bool = False
    
@dataclass
class DashboardConfig:
    """Configuration for a complete dashboard"""
    title: str
    uid: str
    description: str
    tags: List[str]
    panels: List[PanelConfig]
    refresh_interval: str = "5s"
    time_range: str = "now-30m"

class GrafanaDashboardCreator:
    """Creates and manages Grafana dashboards for the Agent System"""
    
    def __init__(self):
        self.grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        self.username = os.getenv("GRAFANA_USER", "admin")
        self.password = os.getenv("GRAFANA_PASSWORD", "admin123")
        self.api_key = os.getenv("GRAFANA_API_KEY", None)
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        
    def create_performance_dashboard(self) -> DashboardConfig:
        """Create the main performance monitoring dashboard"""
        panels = [
            # CPU Usage Panel
            PanelConfig(
                title="CPU Usage",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "performance_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
  |> filter(fn: (r) => r["metric_type"] == "cpu")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
                ''',
                grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
                unit="percent",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 60},
                        {"color": "red", "value": 80}
                    ]
                }
            ),
            
            # Memory Usage Panel
            PanelConfig(
                title="Memory Usage",
                panel_type="gauge",
                datasource="InfluxDB",
                query='''
from(bucket: "performance_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
  |> filter(fn: (r) => r["metric_type"] == "memory")
  |> last()
                ''',
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 8},
                unit="percent",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 70},
                        {"color": "red", "value": 90}
                    ]
                }
            ),
            
            # Disk I/O Panel
            PanelConfig(
                title="Disk I/O",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "performance_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
  |> filter(fn: (r) => r["metric_type"] == "disk_io")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
                ''',
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 8},
                unit="Bps"
            ),
            
            # Network Traffic Panel
            PanelConfig(
                title="Network Traffic",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "performance_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
  |> filter(fn: (r) => r["metric_type"] == "network")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
                ''',
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
                unit="Bps",
                legend=True
            ),
            
            # Process Count Panel
            PanelConfig(
                title="Process Count",
                panel_type="stat",
                datasource="InfluxDB",
                query='''
from(bucket: "performance_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
  |> filter(fn: (r) => r["metric_type"] == "process_count")
  |> last()
                ''',
                grid_pos={"x": 12, "y": 8, "w": 6, "h": 4},
                unit="short"
            ),
            
            # System Uptime Panel
            PanelConfig(
                title="System Uptime",
                panel_type="stat",
                datasource="InfluxDB",
                query='''
from(bucket: "performance_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
  |> filter(fn: (r) => r["metric_type"] == "uptime")
  |> last()
                ''',
                grid_pos={"x": 18, "y": 8, "w": 6, "h": 4},
                unit="s"
            ),
            
            # Top Processes by CPU
            PanelConfig(
                title="Top Processes by CPU",
                panel_type="table",
                datasource="InfluxDB",
                query='''
from(bucket: "performance_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "processes")
  |> filter(fn: (r) => r["_field"] == "cpu_percent")
  |> group(columns: ["process_name"])
  |> mean()
  |> sort(columns: ["_value"], desc: true)
  |> limit(n: 10)
                ''',
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
                legend=False
            )
        ]
        
        return DashboardConfig(
            title="System Performance Monitoring",
            uid="agent-performance",
            description="Real-time performance metrics for the Agent System",
            tags=["performance", "monitoring", "system"],
            panels=panels,
            refresh_interval="5s",
            time_range="now-30m"
        )
    
    def create_agent_metrics_dashboard(self) -> DashboardConfig:
        """Create dashboard for agent-specific metrics"""
        panels = [
            # Tasks Completed
            PanelConfig(
                title="Tasks Completed",
                panel_type="stat",
                datasource="InfluxDB",
                query='''
from(bucket: "agent_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "agent_metrics")
  |> filter(fn: (r) => r["_field"] == "tasks_completed")
  |> sum()
                ''',
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                unit="short"
            ),
            
            # Agent Status
            PanelConfig(
                title="Agent Status",
                panel_type="table",
                datasource="InfluxDB",
                query='''
from(bucket: "agent_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "agent_metrics")
  |> filter(fn: (r) => r["_field"] == "status")
  |> last()
  |> group(columns: ["agent"])
                ''',
                grid_pos={"x": 6, "y": 0, "w": 12, "h": 8},
                legend=False
            ),
            
            # Task Completion Rate
            PanelConfig(
                title="Task Completion Rate",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "agent_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "agent_metrics")
  |> filter(fn: (r) => r["_field"] == "tasks_completed")
  |> aggregateWindow(every: 1m, fn: sum, createEmpty: false)
  |> derivative(unit: 1m, nonNegative: true)
                ''',
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 8},
                unit="ops",
                legend=True
            ),
            
            # Agent Resource Usage
            PanelConfig(
                title="Agent Resource Usage",
                panel_type="heatmap",
                datasource="InfluxDB",
                query='''
from(bucket: "agent_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "agent_metrics")
  |> filter(fn: (r) => r["_field"] == "resource_usage")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
                ''',
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
                unit="percent"
            ),
            
            # Error Rate
            PanelConfig(
                title="Error Rate",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "agent_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "agent_metrics")
  |> filter(fn: (r) => r["_field"] == "errors")
  |> aggregateWindow(every: v.windowPeriod, fn: sum, createEmpty: false)
                ''',
                grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
                unit="short",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 1},
                        {"color": "red", "value": 5}
                    ]
                }
            )
        ]
        
        return DashboardConfig(
            title="Agent Metrics Dashboard",
            uid="agent-metrics",
            description="Agent-specific metrics and performance indicators",
            tags=["agents", "tasks", "monitoring"],
            panels=panels,
            refresh_interval="10s",
            time_range="now-1h"
        )
    
    def create_alerts_dashboard(self) -> DashboardConfig:
        """Create dashboard for system alerts"""
        panels = [
            # Alert Count by Severity
            PanelConfig(
                title="Alert Count by Severity",
                panel_type="piechart",
                datasource="InfluxDB",
                query='''
from(bucket: "alerts")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "alerts")
  |> group(columns: ["level"])
  |> count()
                ''',
                grid_pos={"x": 0, "y": 0, "w": 8, "h": 8}
            ),
            
            # Recent Alerts
            PanelConfig(
                title="Recent Alerts",
                panel_type="table",
                datasource="InfluxDB",
                query='''
from(bucket: "alerts")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "alerts")
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 20)
                ''',
                grid_pos={"x": 8, "y": 0, "w": 16, "h": 8},
                legend=False
            ),
            
            # Alert Timeline
            PanelConfig(
                title="Alert Timeline",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "alerts")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "alerts")
  |> group(columns: ["level"])
  |> aggregateWindow(every: v.windowPeriod, fn: count, createEmpty: false)
                ''',
                grid_pos={"x": 0, "y": 8, "w": 24, "h": 8},
                legend=True
            ),
            
            # Critical Alerts
            PanelConfig(
                title="Critical Alerts",
                panel_type="stat",
                datasource="InfluxDB",
                query='''
from(bucket: "alerts")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "alerts")
  |> filter(fn: (r) => r["level"] == "critical")
  |> count()
                ''',
                grid_pos={"x": 0, "y": 16, "w": 6, "h": 4},
                unit="short",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "red", "value": 1}
                    ]
                }
            )
        ]
        
        return DashboardConfig(
            title="System Alerts Dashboard",
            uid="agent-alerts",
            description="System alerts and notifications monitoring",
            tags=["alerts", "monitoring", "notifications"],
            panels=panels,
            refresh_interval="5s",
            time_range="now-24h"
        )
    
    def create_test_metrics_dashboard(self) -> DashboardConfig:
        """Create dashboard for test execution metrics"""
        panels = [
            # Test Coverage
            PanelConfig(
                title="Test Coverage",
                panel_type="gauge",
                datasource="InfluxDB",
                query='''
from(bucket: "test_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "test_metrics")
  |> filter(fn: (r) => r["_field"] == "coverage")
  |> last()
                ''',
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 8},
                unit="percent",
                thresholds={
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 60},
                        {"color": "green", "value": 80}
                    ]
                }
            ),
            
            # Test Execution Time
            PanelConfig(
                title="Test Execution Time",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "test_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "test_metrics")
  |> filter(fn: (r) => r["_field"] == "execution_time")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
                ''',
                grid_pos={"x": 6, "y": 0, "w": 18, "h": 8},
                unit="s"
            ),
            
            # Test Pass/Fail Ratio
            PanelConfig(
                title="Test Pass/Fail Ratio",
                panel_type="piechart",
                datasource="InfluxDB",
                query='''
from(bucket: "test_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "test_metrics")
  |> filter(fn: (r) => r["_field"] == "test_result")
  |> group(columns: ["result"])
  |> count()
                ''',
                grid_pos={"x": 0, "y": 8, "w": 8, "h": 8}
            ),
            
            # Test Runs Over Time
            PanelConfig(
                title="Test Runs Over Time",
                panel_type="graph",
                datasource="InfluxDB",
                query='''
from(bucket: "test_metrics")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "test_metrics")
  |> filter(fn: (r) => r["_field"] == "test_run")
  |> aggregateWindow(every: v.windowPeriod, fn: count, createEmpty: false)
                ''',
                grid_pos={"x": 8, "y": 8, "w": 16, "h": 8},
                unit="short"
            )
        ]
        
        return DashboardConfig(
            title="Test Metrics Dashboard",
            uid="test-metrics",
            description="Test execution and coverage metrics",
            tags=["testing", "coverage", "quality"],
            panels=panels,
            refresh_interval="30s",
            time_range="now-6h"
        )
    
    def generate_dashboard_json(self, config: DashboardConfig) -> Dict:
        """Generate Grafana dashboard JSON from config"""
        dashboard = {
            "dashboard": {
                "uid": config.uid,
                "title": config.title,
                "tags": config.tags,
                "timezone": "browser",
                "panels": [],
                "schemaVersion": 37,
                "version": 1,
                "refresh": config.refresh_interval,
                "time": {
                    "from": config.time_range,
                    "to": "now"
                },
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 0,
                "links": [],
                "liveNow": False,
                "weekStart": ""
            },
            "overwrite": True
        }
        
        # Add panels
        for i, panel_config in enumerate(config.panels):
            panel = {
                "id": i + 1,
                "title": panel_config.title,
                "type": panel_config.panel_type,
                "datasource": {
                    "type": "influxdb",
                    "uid": panel_config.datasource
                },
                "gridPos": panel_config.grid_pos,
                "targets": [
                    {
                        "query": panel_config.query.strip(),
                        "refId": "A"
                    }
                ],
                "transparent": panel_config.transparent,
                "options": {}
            }
            
            # Add unit if specified
            if panel_config.unit:
                panel["fieldConfig"] = {
                    "defaults": {
                        "unit": panel_config.unit
                    }
                }
            
            # Add thresholds if specified
            if panel_config.thresholds:
                if "fieldConfig" not in panel:
                    panel["fieldConfig"] = {"defaults": {}}
                panel["fieldConfig"]["defaults"]["thresholds"] = panel_config.thresholds
            
            # Add legend configuration
            if panel_config.panel_type == "graph":
                panel["options"]["legend"] = {
                    "displayMode": "list" if panel_config.legend else "hidden",
                    "placement": "bottom"
                }
            
            dashboard["dashboard"]["panels"].append(panel)
        
        return dashboard
    
    def save_dashboard_to_file(self, config: DashboardConfig, filename: str):
        """Save dashboard configuration to JSON file"""
        dashboard_json = self.generate_dashboard_json(config)
        
        output_dir = "grafana/dashboards"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(dashboard_json, f, indent=2)
        
        print(f"✅ Saved dashboard to {filepath}")
        return filepath
    
    def upload_dashboard_to_grafana(self, config: DashboardConfig):
        """Upload dashboard to Grafana via API"""
        dashboard_json = self.generate_dashboard_json(config)
        
        try:
            response = self.session.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard_json,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Dashboard '{config.title}' uploaded successfully!")
                print(f"   URL: {self.grafana_url}{result.get('url', '')}")
                return True
            else:
                print(f"❌ Failed to upload dashboard: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading dashboard: {e}")
            return False
    
    def create_all_dashboards(self):
        """Create and save all dashboards"""
        print("🎨 Creating Grafana Dashboards")
        print("=" * 60)
        
        dashboards = [
            (self.create_performance_dashboard(), "performance-dashboard.json"),
            (self.create_agent_metrics_dashboard(), "agent-metrics-dashboard.json"),
            (self.create_alerts_dashboard(), "alerts-dashboard.json"),
            (self.create_test_metrics_dashboard(), "test-metrics-dashboard.json")
        ]
        
        for config, filename in dashboards:
            print(f"\n📊 Creating {config.title}...")
            filepath = self.save_dashboard_to_file(config, filename)
            
            # Try to upload to Grafana
            if self.upload_dashboard_to_grafana(config):
                print(f"   ✅ Dashboard uploaded to Grafana")
            else:
                print(f"   ⚠️  Dashboard saved locally but not uploaded")
        
        print("\n" + "=" * 60)
        print("✅ All dashboards created successfully!")
        print("\n📁 Dashboard files saved in: grafana/dashboards/")
        print("📊 To import manually:")
        print("   1. Open Grafana: http://localhost:3000")
        print("   2. Go to Dashboards > Import")
        print("   3. Upload the JSON files from grafana/dashboards/")


def test_dashboard_creation():
    """Test dashboard creation"""
    creator = GrafanaDashboardCreator()
    creator.create_all_dashboards()


if __name__ == "__main__":
    print("Grafana Dashboard Creator for Agent System")
    print("=" * 60)
    
    # Load environment variables if available
    if os.path.exists(".env.influxdb"):
        from dotenv import load_dotenv
        load_dotenv(".env.influxdb")
    
    test_dashboard_creation()