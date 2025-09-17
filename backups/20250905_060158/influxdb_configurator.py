#!/usr/bin/env python3
"""
InfluxDB Configurator
Sets up buckets, retention policies, and initial configuration for the Agent System
"""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from influxdb_client import InfluxDBClient, BucketRetentionRules
    from influxdb_client.client.exceptions import InfluxDBError
    INFLUXDB_AVAILABLE = True
except ImportError as e:
    INFLUXDB_AVAILABLE = False
    print(f"Warning: InfluxDB client not installed. Error: {e}")
    print("Install with: pip install influxdb-client")


@dataclass
class BucketConfig:
    """Configuration for an InfluxDB bucket"""
    name: str
    description: str
    retention_seconds: int  # 0 for infinite
    shard_group_duration_seconds: Optional[int] = None
    
    @property
    def retention_days(self) -> int:
        """Get retention in days"""
        return self.retention_seconds // 86400 if self.retention_seconds > 0 else 0
    
    @property
    def retention_str(self) -> str:
        """Get human-readable retention string"""
        if self.retention_seconds == 0:
            return "infinite"
        days = self.retention_days
        if days >= 365:
            return f"{days // 365} year(s)"
        elif days >= 30:
            return f"{days // 30} month(s)"
        else:
            return f"{days} day(s)"


class InfluxDBConfigurator:
    """Configures InfluxDB for the Agent System"""
    
    # Bucket configurations
    BUCKETS = [
        BucketConfig(
            name="performance_metrics",
            description="High-resolution performance metrics (CPU, memory, disk, network)",
            retention_seconds=30 * 86400,  # 30 days
            shard_group_duration_seconds=86400  # 1 day shards
        ),
        BucketConfig(
            name="performance_metrics_long",
            description="Downsampled long-term performance metrics",
            retention_seconds=365 * 86400,  # 1 year
            shard_group_duration_seconds=7 * 86400  # 1 week shards
        ),
        BucketConfig(
            name="agent_metrics",
            description="Agent-specific metrics (task completion, resource usage)",
            retention_seconds=90 * 86400,  # 90 days
            shard_group_duration_seconds=86400
        ),
        BucketConfig(
            name="alerts",
            description="System alerts and notifications",
            retention_seconds=180 * 86400,  # 180 days
            shard_group_duration_seconds=7 * 86400
        ),
        BucketConfig(
            name="test_metrics",
            description="Test execution and coverage metrics",
            retention_seconds=60 * 86400,  # 60 days
            shard_group_duration_seconds=86400
        ),
        BucketConfig(
            name="deployment_metrics",
            description="Deployment and release metrics",
            retention_seconds=0,  # Infinite retention
            shard_group_duration_seconds=30 * 86400  # 30 day shards
        ),
        BucketConfig(
            name="debug_metrics",
            description="Short-term high-frequency debug metrics",
            retention_seconds=3 * 86400,  # 3 days only
            shard_group_duration_seconds=3600  # 1 hour shards
        )
    ]
    
    def __init__(self):
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
        self.org = os.getenv("INFLUXDB_ORG", "agent-system")
        self.client = None
        self.buckets_api = None
        self.orgs_api = None
        self.tasks_api = None
        
    def connect(self) -> bool:
        """Connect to InfluxDB"""
        if not INFLUXDB_AVAILABLE:
            print("❌ InfluxDB client not available")
            return False
        
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
            
            # Test connection
            health = self.client.health()
            if health.status != "pass":
                print(f"❌ InfluxDB health check failed: {health.status}")
                return False
            
            # Get APIs
            self.buckets_api = self.client.buckets_api()
            self.orgs_api = self.client.organizations_api()
            self.tasks_api = self.client.tasks_api()
            
            print(f"✅ Connected to InfluxDB at {self.url}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to InfluxDB: {e}")
            return False
    
    def get_organization(self) -> Optional[Any]:
        """Get the organization"""
        try:
            orgs = self.orgs_api.find_organizations(org=self.org)
            if orgs and len(orgs) > 0:
                return orgs[0]
            return None
        except Exception as e:
            print(f"Error getting organization: {e}")
            return None
    
    def create_buckets(self):
        """Create all configured buckets"""
        if not self.buckets_api:
            print("❌ Not connected to InfluxDB")
            return
        
        org = self.get_organization()
        if not org:
            print(f"❌ Organization '{self.org}' not found")
            return
        
        print(f"\n📦 Configuring buckets for organization: {self.org}")
        print("-" * 60)
        
        existing_buckets = []
        try:
            buckets = self.buckets_api.find_buckets(org=self.org)
            existing_buckets = [b.name for b in buckets.buckets]
        except:
            pass
        
        for bucket_config in self.BUCKETS:
            if bucket_config.name in existing_buckets:
                print(f"  ✓ Bucket '{bucket_config.name}' already exists")
                continue
            
            try:
                # Create retention rules
                retention_rules = []
                if bucket_config.retention_seconds > 0:
                    retention_rules.append(
                        BucketRetentionRules(
                            every_seconds=bucket_config.retention_seconds,
                            type="expire"
                        )
                    )
                
                # Create bucket
                bucket = self.buckets_api.create_bucket(
                    bucket_name=bucket_config.name,
                    description=bucket_config.description,
                    org_id=org.id,
                    retention_rules=retention_rules if retention_rules else None
                )
                
                print(f"  ✅ Created bucket: {bucket_config.name}")
                print(f"     Description: {bucket_config.description}")
                print(f"     Retention: {bucket_config.retention_str}")
                
            except InfluxDBError as e:
                if "already exists" in str(e):
                    print(f"  ✓ Bucket '{bucket_config.name}' already exists")
                else:
                    print(f"  ❌ Failed to create bucket '{bucket_config.name}': {e}")
            except Exception as e:
                print(f"  ❌ Error creating bucket '{bucket_config.name}': {e}")
    
    def create_downsampling_task(self):
        """Create a task to downsample high-resolution metrics"""
        if not self.tasks_api:
            print("❌ Not connected to InfluxDB")
            return
        
        print("\n⚙️  Creating downsampling tasks...")
        print("-" * 60)
        
        # Flux script for downsampling
        flux_script = '''
option task = {
    name: "Downsample Performance Metrics",
    every: 1h,
}

from(bucket: "performance_metrics")
    |> range(start: -task.every)
    |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
    |> aggregateWindow(
        every: 5m,
        fn: mean,
        createEmpty: false
    )
    |> to(bucket: "performance_metrics_long", org: "''' + self.org + '''")
'''
        
        try:
            org = self.get_organization()
            if not org:
                print("❌ Organization not found")
                return
            
            # Check if task already exists
            tasks = self.tasks_api.find_tasks(name="Downsample Performance Metrics")
            if tasks:
                print("  ✓ Downsampling task already exists")
                return
            
            # Create the task
            task = self.tasks_api.create_task(
                flux=flux_script,
                org_id=org.id
            )
            
            print(f"  ✅ Created downsampling task: {task.name}")
            
        except Exception as e:
            print(f"  ❌ Error creating downsampling task: {e}")
    
    def create_continuous_queries(self):
        """Create continuous queries for real-time aggregations"""
        print("\n📊 Creating continuous queries...")
        print("-" * 60)
        
        queries = [
            {
                "name": "1-minute averages",
                "description": "Calculate 1-minute averages for all metrics",
                "flux": '''
from(bucket: "performance_metrics")
    |> range(start: -1m)
    |> filter(fn: (r) => r["_measurement"] == "performance_metrics")
    |> mean()
    |> set(key: "_measurement", value: "performance_metrics_1m")
'''
            },
            {
                "name": "Alert summary",
                "description": "Summarize alerts by severity",
                "flux": '''
from(bucket: "alerts")
    |> range(start: -1h)
    |> filter(fn: (r) => r["_field"] == "level")
    |> group(columns: ["level"])
    |> count()
'''
            }
        ]
        
        for query in queries:
            print(f"  📝 {query['name']}: {query['description']}")
        
        print("\n  ℹ️  Note: Continuous queries can be created via the InfluxDB UI")
    
    def display_configuration_summary(self):
        """Display a summary of the configuration"""
        print("\n" + "="*60)
        print("📋 INFLUXDB CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"\n🔗 Connection:")
        print(f"   URL: {self.url}")
        print(f"   Organization: {self.org}")
        
        print(f"\n📦 Buckets ({len(self.BUCKETS)}):")
        total_retention_bytes = 0
        for bucket in self.BUCKETS:
            icon = "♾️" if bucket.retention_seconds == 0 else "📊"
            print(f"   {icon} {bucket.name}: {bucket.retention_str}")
            
        print(f"\n💾 Data Retention Strategy:")
        print(f"   • High-resolution: 30 days (performance_metrics)")
        print(f"   • Long-term: 1 year (performance_metrics_long)")
        print(f"   • Debug data: 3 days (debug_metrics)")
        print(f"   • Deployments: Forever (deployment_metrics)")
        
        print(f"\n🔄 Automation:")
        print(f"   • Downsampling: Every hour")
        print(f"   • Data aggregation: Real-time")
        print(f"   • Retention policies: Automatic")
        
        print("\n" + "="*60)
    
    def write_sample_data(self):
        """Write sample data to test the configuration"""
        print("\n📝 Writing sample data...")
        print("-" * 60)
        
        if not self.client:
            print("❌ Not connected to InfluxDB")
            return
        
        write_api = self.client.write_api()
        
        # Sample data points
        samples = [
            'performance_metrics,host=agent-system,metric_type=cpu value=45.2',
            'performance_metrics,host=agent-system,metric_type=memory value=78.5',
            'agent_metrics,agent=code_generator tasks_completed=15i',
            'alerts,level=warning,metric=cpu message="High CPU usage"',
            'test_metrics,project=agent-lightning coverage=76.8'
        ]
        
        for sample in samples:
            try:
                write_api.write(
                    bucket="performance_metrics",
                    org=self.org,
                    record=sample
                )
                print(f"  ✅ Written: {sample[:50]}...")
            except Exception as e:
                print(f"  ❌ Failed to write sample: {e}")
        
        write_api.close()
    
    def configure_all(self):
        """Run all configuration steps"""
        print("\n🚀 Starting InfluxDB Configuration")
        print("="*60)
        
        # Connect
        if not self.connect():
            print("\n❌ Configuration failed: Could not connect to InfluxDB")
            print("\nPlease ensure:")
            print("1. InfluxDB is running (./influxdb_manager.sh status)")
            print("2. Initial setup is complete (visit http://localhost:8086)")
            print("3. Environment variables are set correctly")
            return False
        
        # Configure
        self.create_buckets()
        self.create_downsampling_task()
        self.create_continuous_queries()
        self.write_sample_data()
        
        # Summary
        self.display_configuration_summary()
        
        # Close connection
        if self.client:
            self.client.close()
        
        print("\n✅ InfluxDB configuration complete!")
        return True


def test_configuration():
    """Test the InfluxDB configuration"""
    configurator = InfluxDBConfigurator()
    return configurator.configure_all()


if __name__ == "__main__":
    print("InfluxDB Configurator for Agent System")
    print("="*60)
    
    if not INFLUXDB_AVAILABLE:
        print("\n⚠️  InfluxDB client is required")
        print("   Install with: pip install influxdb-client")
        sys.exit(1)
    
    # Check environment
    if not os.getenv("INFLUXDB_TOKEN"):
        print("\n⚠️  Warning: INFLUXDB_TOKEN not set")
        print("   Using default token. Please update after initial setup.")
    
    # Run configuration
    success = test_configuration()
    
    if success:
        print("\n🎉 Your InfluxDB is now configured for the Agent System!")
        print("\nNext steps:")
        print("1. Start sending metrics from your agents")
        print("2. Create Grafana dashboards for visualization")
        print("3. Set up alerts for critical metrics")
    else:
        print("\n⚠️  Configuration incomplete. Please check the errors above.")