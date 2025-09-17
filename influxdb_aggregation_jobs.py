#!/usr/bin/env python3
"""
InfluxDB Data Aggregation Jobs
Creates and manages automated data aggregation tasks for long-term storage
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from influxdb_client import InfluxDBClient, TasksApi, Task
from influxdb_client.client.exceptions import InfluxDBError
from dotenv import load_dotenv

# Load environment
load_dotenv('.env.influxdb')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AggregationJob:
    """Configuration for a data aggregation job"""
    name: str
    description: str
    source_bucket: str
    destination_bucket: str
    interval: str  # e.g., "1h", "1d", "1w"
    aggregation_window: str  # e.g., "5m", "1h", "1d"
    measurement: str
    aggregation_functions: List[str]  # mean, max, min, sum, count
    flux_query: Optional[str] = None


class InfluxDBAggregationManager:
    """Manages data aggregation jobs in InfluxDB"""
    
    def __init__(self):
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN')
        self.org = os.getenv('INFLUXDB_ORG', 'agent-system')
        
        if not self.token:
            logger.warning("INFLUXDB_TOKEN not set, using default")
            self.token = "agent-system-token-supersecret-12345678"
        
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        self.tasks_api = self.client.tasks_api()
        
        # Define aggregation jobs
        self.jobs = self._define_aggregation_jobs()
    
    def _define_aggregation_jobs(self) -> List[AggregationJob]:
        """Define all aggregation jobs"""
        return [
            # 1. Performance metrics: 5-minute averages for long-term storage
            AggregationJob(
                name="performance_5min_avg",
                description="Aggregate performance metrics to 5-minute averages",
                source_bucket="performance_metrics",
                destination_bucket="performance_metrics_long",
                interval="5m",  # Run every 5 minutes
                aggregation_window="5m",
                measurement="system_metrics",
                aggregation_functions=["mean", "max", "min"]
            ),
            
            # 2. Hourly performance summary
            AggregationJob(
                name="performance_hourly",
                description="Create hourly performance summaries",
                source_bucket="performance_metrics",
                destination_bucket="performance_metrics_long",
                interval="1h",  # Run every hour
                aggregation_window="1h",
                measurement="system_metrics",
                aggregation_functions=["mean", "max", "min", "count"]
            ),
            
            # 3. Daily agent metrics summary
            AggregationJob(
                name="agent_daily_summary",
                description="Daily summary of agent activities",
                source_bucket="agent_metrics",
                destination_bucket="agent_metrics",
                interval="24h",  # Run daily
                aggregation_window="24h",
                measurement="agent_activity",
                aggregation_functions=["sum", "mean", "count"]
            ),
            
            # 4. Alert aggregation by severity
            AggregationJob(
                name="alerts_hourly_summary",
                description="Hourly alert summary by severity",
                source_bucket="alerts",
                destination_bucket="alerts",
                interval="1h",
                aggregation_window="1h",
                measurement="alert_events",
                aggregation_functions=["count", "last"]
            ),
            
            # 5. Test metrics daily rollup
            AggregationJob(
                name="test_metrics_daily",
                description="Daily test execution summary",
                source_bucket="test_metrics",
                destination_bucket="test_metrics",
                interval="24h",
                aggregation_window="24h",
                measurement="test_results",
                aggregation_functions=["mean", "sum", "count"]
            )
        ]
    
    def generate_flux_query(self, job: AggregationJob) -> str:
        """Generate Flux query for aggregation job"""
        
        # Build aggregation functions
        agg_functions = []
        for func in job.aggregation_functions:
            if func == "mean":
                agg_functions.append('mean: mean(column: "_value")')
            elif func == "max":
                agg_functions.append('max: max(column: "_value")')
            elif func == "min":
                agg_functions.append('min: min(column: "_value")')
            elif func == "sum":
                agg_functions.append('sum: sum(column: "_value")')
            elif func == "count":
                agg_functions.append('count: count(column: "_value")')
            elif func == "last":
                agg_functions.append('last: last(column: "_value")')
        
        agg_string = ",\n    ".join(agg_functions)
        
        # Generate the Flux query
        query = f'''
option task = {{
  name: "{job.name}",
  every: {job.interval},
  offset: 0s
}}

from(bucket: "{job.source_bucket}")
  |> range(start: -task.every)
  |> filter(fn: (r) => r["_measurement"] == "{job.measurement}")
  |> aggregateWindow(
      every: {job.aggregation_window},
      fn: (column, tables=<-) => tables |> mean(column: column),
      createEmpty: false
  )
  |> pivot(
      rowKey: ["_time"],
      columnKey: ["_field"],
      valueColumn: "_value"
  )
  |> map(fn: (r) => ({{
      r with
      _measurement: "{job.measurement}_aggregated",
      _time: r._time,
      aggregation_window: "{job.aggregation_window}"
  }}))
  |> to(
      bucket: "{job.destination_bucket}",
      org: "{self.org}"
  )
'''
        return query
    
    def create_task(self, job: AggregationJob) -> bool:
        """Create an InfluxDB task for the aggregation job"""
        try:
            # Generate Flux query if not provided
            if not job.flux_query:
                job.flux_query = self.generate_flux_query(job)
            
            # Check if task already exists
            existing_tasks = self.tasks_api.find_tasks(name=job.name)
            if existing_tasks and len(existing_tasks) > 0:
                logger.info(f"Task '{job.name}' already exists, skipping creation")
                return True
            
            # Create the task
            task = self.tasks_api.create_task_every(
                name=job.name,
                flux=job.flux_query,
                every=job.interval,
                organization=self.org
            )
            
            logger.info(f"✅ Created aggregation task: {job.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create task '{job.name}': {e}")
            return False
    
    def create_all_tasks(self):
        """Create all aggregation tasks"""
        logger.info("Creating data aggregation tasks...")
        logger.info("=" * 60)
        
        created = 0
        failed = 0
        
        for job in self.jobs:
            if self.create_task(job):
                created += 1
            else:
                failed += 1
        
        logger.info("=" * 60)
        logger.info(f"Summary: {created} tasks created/verified, {failed} failed")
    
    def list_tasks(self):
        """List all existing tasks"""
        try:
            tasks = self.tasks_api.find_tasks()
            
            if not tasks:
                logger.info("No tasks found")
                return
            
            logger.info("\nExisting Tasks:")
            logger.info("-" * 60)
            
            for task in tasks:
                status = "✅ Active" if task.status == "active" else "⏸️  Inactive"
                logger.info(f"  {task.name}: {status}")
                logger.info(f"    Description: {task.description or 'N/A'}")
                logger.info(f"    Schedule: Every {task.every}")
                logger.info(f"    Last Run: {task.latest_completed or 'Never'}")
                logger.info("")
                
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
    
    def enable_task(self, task_name: str):
        """Enable a specific task"""
        try:
            tasks = self.tasks_api.find_tasks(name=task_name)
            if tasks:
                task = tasks[0]
                task.status = "active"
                self.tasks_api.update_task(task.id, task)
                logger.info(f"✅ Enabled task: {task_name}")
            else:
                logger.error(f"Task '{task_name}' not found")
        except Exception as e:
            logger.error(f"Failed to enable task: {e}")
    
    def disable_task(self, task_name: str):
        """Disable a specific task"""
        try:
            tasks = self.tasks_api.find_tasks(name=task_name)
            if tasks:
                task = tasks[0]
                task.status = "inactive"
                self.tasks_api.update_task(task.id, task)
                logger.info(f"⏸️  Disabled task: {task_name}")
            else:
                logger.error(f"Task '{task_name}' not found")
        except Exception as e:
            logger.error(f"Failed to disable task: {e}")
    
    def delete_task(self, task_name: str):
        """Delete a specific task"""
        try:
            tasks = self.tasks_api.find_tasks(name=task_name)
            if tasks:
                self.tasks_api.delete_task(tasks[0].id)
                logger.info(f"🗑️  Deleted task: {task_name}")
            else:
                logger.error(f"Task '{task_name}' not found")
        except Exception as e:
            logger.error(f"Failed to delete task: {e}")
    
    def run_task_now(self, task_name: str):
        """Manually run a task immediately"""
        try:
            tasks = self.tasks_api.find_tasks(name=task_name)
            if tasks:
                run = self.tasks_api.run_manually(tasks[0].id)
                logger.info(f"🚀 Manually triggered task: {task_name}")
                logger.info(f"   Run ID: {run.id}")
            else:
                logger.error(f"Task '{task_name}' not found")
        except Exception as e:
            logger.error(f"Failed to run task: {e}")
    
    def create_custom_aggregation(self, 
                                   name: str,
                                   source_bucket: str,
                                   dest_bucket: str,
                                   measurement: str,
                                   interval: str = "1h",
                                   window: str = "10m"):
        """Create a custom aggregation job"""
        job = AggregationJob(
            name=name,
            description=f"Custom aggregation: {name}",
            source_bucket=source_bucket,
            destination_bucket=dest_bucket,
            interval=interval,
            aggregation_window=window,
            measurement=measurement,
            aggregation_functions=["mean", "max", "min"]
        )
        
        if self.create_task(job):
            logger.info(f"✅ Created custom aggregation: {name}")
        else:
            logger.error(f"❌ Failed to create custom aggregation: {name}")
    
    def get_task_runs(self, task_name: str, limit: int = 10):
        """Get recent runs of a task"""
        try:
            tasks = self.tasks_api.find_tasks(name=task_name)
            if not tasks:
                logger.error(f"Task '{task_name}' not found")
                return
            
            runs = self.tasks_api.get_runs(tasks[0].id, limit=limit)
            
            logger.info(f"\nRecent runs for '{task_name}':")
            logger.info("-" * 60)
            
            for run in runs:
                status_icon = "✅" if run.status == "success" else "❌"
                logger.info(f"  {status_icon} {run.started_at}: {run.status}")
                if run.log:
                    logger.info(f"    Log: {run.log[:100]}...")
                    
        except Exception as e:
            logger.error(f"Failed to get task runs: {e}")


def main():
    """Main function to manage aggregation jobs"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage InfluxDB data aggregation jobs")
    parser.add_argument("action", 
                        choices=["create", "list", "enable", "disable", 
                                 "delete", "run", "status"],
                        help="Action to perform")
    parser.add_argument("--task", help="Task name for specific operations")
    parser.add_argument("--all", action="store_true", 
                        help="Apply to all tasks")
    
    args = parser.parse_args()
    
    manager = InfluxDBAggregationManager()
    
    print("🔄 InfluxDB Data Aggregation Manager")
    print("=" * 60)
    
    if args.action == "create":
        manager.create_all_tasks()
        
    elif args.action == "list":
        manager.list_tasks()
        
    elif args.action == "enable":
        if args.task:
            manager.enable_task(args.task)
        elif args.all:
            for job in manager.jobs:
                manager.enable_task(job.name)
        else:
            print("Please specify --task or --all")
            
    elif args.action == "disable":
        if args.task:
            manager.disable_task(args.task)
        elif args.all:
            for job in manager.jobs:
                manager.disable_task(job.name)
        else:
            print("Please specify --task or --all")
            
    elif args.action == "delete":
        if args.task:
            manager.delete_task(args.task)
        elif args.all:
            response = input("⚠️  Delete ALL tasks? (yes/no): ")
            if response.lower() == "yes":
                for job in manager.jobs:
                    manager.delete_task(job.name)
        else:
            print("Please specify --task or --all")
            
    elif args.action == "run":
        if args.task:
            manager.run_task_now(args.task)
        else:
            print("Please specify --task to run")
            
    elif args.action == "status":
        if args.task:
            manager.get_task_runs(args.task)
        else:
            manager.list_tasks()
    
    print("\n✅ Operation completed")


if __name__ == "__main__":
    # If no arguments provided, create all tasks
    if len(sys.argv) == 1:
        manager = InfluxDBAggregationManager()
        print("🔄 InfluxDB Data Aggregation Setup")
        print("=" * 60)
        manager.create_all_tasks()
        print("\n📊 Current Tasks:")
        manager.list_tasks()
    else:
        main()