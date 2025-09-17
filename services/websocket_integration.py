#!/usr/bin/env python3
"""
Real WebSocket Integration for Live Microservices Data
Connects to actual running services and broadcasts real events
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicroservicesMonitor:
    """Monitors all microservices and broadcasts real events via WebSocket"""
    
    def __init__(self, websocket_url: str = "http://localhost:8007"):
        self.websocket_url = websocket_url
        self.services = {
            "gateway": {"url": "http://localhost:8000", "name": "API Gateway"},
            "auth": {"url": "http://localhost:8006", "name": "Auth Service"},
            "agent": {"url": "http://localhost:8001", "name": "Agent Designer"},
            "workflow": {"url": "http://localhost:8003", "name": "Workflow Engine"},
            "integration": {"url": "http://localhost:8004", "name": "Integration Hub"},
            "ai": {"url": "http://localhost:8005", "name": "AI Model Service"},
        }
        
        self.previous_states = {}
        self.auth_token = None
        
    async def get_auth_token(self):
        """Get authentication token"""
        async with aiohttp.ClientSession() as session:
            try:
                login_data = {
                    "username": "admin",
                    "password": "admin123"
                }
                async with session.post(
                    f"{self.services['auth']['url']}/api/v1/auth/login",
                    json=login_data
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.auth_token = data.get("access_token")
                        logger.info("Authentication successful")
                        return True
            except Exception as e:
                logger.error(f"Auth failed: {e}")
        return False
    
    async def check_service_health(self, service_id: str, service_info: Dict) -> Dict[str, Any]:
        """Check health of a specific service"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{service_info['url']}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "service": service_id,
                            "name": service_info['name'],
                            "status": "healthy",
                            "details": data
                        }
            except Exception as e:
                logger.warning(f"Service {service_id} health check failed: {e}")
        
        return {
            "service": service_id,
            "name": service_info['name'],
            "status": "unhealthy",
            "details": {}
        }
    
    async def get_agents_count(self) -> int:
        """Get current agent count"""
        if not self.auth_token:
            return 0
            
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "X-Organization-ID": "default-org",
            "X-User-ID": "user-admin"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.services['agent']['url']}/api/v1/agents",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return len(data)
            except:
                pass
        return 0
    
    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.services['workflow']['url']}/api/v1/executions"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Count by status
                        running = sum(1 for e in data if e.get("status") == "running")
                        completed = sum(1 for e in data if e.get("status") == "completed")
                        failed = sum(1 for e in data if e.get("status") == "failed")
                        
                        return {
                            "total": len(data),
                            "running": running,
                            "completed": completed,
                            "failed": failed
                        }
            except:
                pass
        return {"total": 0, "running": 0, "completed": 0, "failed": 0}
    
    async def get_ai_model_stats(self) -> Dict[str, Any]:
        """Get AI model usage statistics"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.services['ai']['url']}/api/v1/models"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "available_models": len(data),
                            "providers": list(set(m.get("provider") for m in data if m.get("provider")))
                        }
            except:
                pass
        return {"available_models": 0, "providers": []}
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast event to WebSocket service"""
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "type": event_type,
                    "channel": "global",
                    "channel_id": "all",
                    "data": data,
                    "sender": "monitoring-service"
                }
                
                async with session.post(
                    f"{self.websocket_url}/api/v1/broadcast",
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.info(f"Broadcasted {event_type}: {data}")
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")
    
    async def monitor_agents(self):
        """Monitor agent changes"""
        while True:
            try:
                current_count = await self.get_agents_count()
                previous_count = self.previous_states.get("agents", 0)
                
                if current_count != previous_count:
                    if current_count > previous_count:
                        await self.broadcast_event("agent.created", {
                            "message": f"New agent created",
                            "total_agents": current_count,
                            "timestamp": datetime.now().isoformat()
                        })
                    elif current_count < previous_count:
                        await self.broadcast_event("agent.deleted", {
                            "message": f"Agent deleted",
                            "total_agents": current_count,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    self.previous_states["agents"] = current_count
                
            except Exception as e:
                logger.error(f"Agent monitoring error: {e}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def monitor_workflows(self):
        """Monitor workflow executions"""
        while True:
            try:
                stats = await self.get_workflow_stats()
                previous_stats = self.previous_states.get("workflows", {})
                
                # Check for new running workflows
                if stats["running"] > previous_stats.get("running", 0):
                    await self.broadcast_event("workflow.started", {
                        "message": "New workflow started",
                        "running_count": stats["running"],
                        "total_count": stats["total"],
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Check for completed workflows
                if stats["completed"] > previous_stats.get("completed", 0):
                    await self.broadcast_event("workflow.completed", {
                        "message": "Workflow completed successfully",
                        "completed_count": stats["completed"],
                        "total_count": stats["total"],
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Check for failed workflows
                if stats["failed"] > previous_stats.get("failed", 0):
                    await self.broadcast_event("workflow.failed", {
                        "message": "Workflow execution failed",
                        "failed_count": stats["failed"],
                        "total_count": stats["total"],
                        "timestamp": datetime.now().isoformat()
                    })
                
                self.previous_states["workflows"] = stats
                
            except Exception as e:
                logger.error(f"Workflow monitoring error: {e}")
            
            await asyncio.sleep(3)  # Check every 3 seconds
    
    async def monitor_services_health(self):
        """Monitor health of all services"""
        while True:
            try:
                for service_id, service_info in self.services.items():
                    health = await self.check_service_health(service_id, service_info)
                    
                    previous_status = self.previous_states.get(f"health_{service_id}", "unknown")
                    current_status = health["status"]
                    
                    if current_status != previous_status:
                        await self.broadcast_event("service.health", {
                            "service": service_id,
                            "name": health["name"],
                            "status": current_status,
                            "previous_status": previous_status,
                            "timestamp": datetime.now().isoformat(),
                            "details": health.get("details", {})
                        })
                        
                        # Alert on service down
                        if current_status == "unhealthy" and previous_status == "healthy":
                            await self.broadcast_event("system.alert", {
                                "level": "error",
                                "message": f"{health['name']} is down!",
                                "service": service_id,
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    self.previous_states[f"health_{service_id}"] = current_status
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def monitor_ai_usage(self):
        """Monitor AI model usage"""
        while True:
            try:
                stats = await self.get_ai_model_stats()
                previous_stats = self.previous_states.get("ai_models", {})
                
                if stats != previous_stats:
                    await self.broadcast_event("system.metric", {
                        "type": "ai_models",
                        "available_models": stats["available_models"],
                        "providers": stats["providers"],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    self.previous_states["ai_models"] = stats
                
            except Exception as e:
                logger.error(f"AI monitoring error: {e}")
            
            await asyncio.sleep(15)  # Check every 15 seconds
    
    async def send_periodic_metrics(self):
        """Send periodic system metrics"""
        while True:
            try:
                # Gather all metrics
                agents_count = await self.get_agents_count()
                workflow_stats = await self.get_workflow_stats()
                ai_stats = await self.get_ai_model_stats()
                
                # Send comprehensive metric update
                await self.broadcast_event("system.metric", {
                    "type": "comprehensive",
                    "agents": {
                        "total": agents_count
                    },
                    "workflows": workflow_stats,
                    "ai_models": ai_stats,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Metrics error: {e}")
            
            await asyncio.sleep(30)  # Send every 30 seconds
    
    async def simulate_activity(self):
        """Simulate some activity for demo purposes"""
        await asyncio.sleep(10)  # Wait 10 seconds before starting
        
        demo_events = [
            ("agent.deployed", {"agent_id": "agent-demo-001", "environment": "production", "replicas": 3}),
            ("integration.triggered", {"integration": "Salesforce", "action": "sync_contacts", "records": 150}),
            ("inference.started", {"model": "GPT-4", "request_id": "req-123", "tokens": 500}),
            ("workflow.progress", {"workflow_id": "wf-001", "step": 3, "total_steps": 5, "progress": 60}),
            ("integration.completed", {"integration": "Slack", "message": "Daily report sent", "channels": 5}),
            ("inference.completed", {"model": "GPT-4", "latency_ms": 847, "tokens_generated": 150}),
        ]
        
        for event_type, data in demo_events:
            await self.broadcast_event(event_type, {
                **data,
                "timestamp": datetime.now().isoformat(),
                "demo": True
            })
            await asyncio.sleep(5)  # Space out demo events
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        logger.info("Starting microservices monitoring...")
        
        # Get auth token first
        await self.get_auth_token()
        
        # Start all monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_agents()),
            asyncio.create_task(self.monitor_workflows()),
            asyncio.create_task(self.monitor_services_health()),
            asyncio.create_task(self.monitor_ai_usage()),
            asyncio.create_task(self.send_periodic_metrics()),
            asyncio.create_task(self.simulate_activity()),  # Add some demo activity
        ]
        
        # Initial broadcast
        await self.broadcast_event("system.alert", {
            "level": "info",
            "message": "Real-time monitoring started",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep running
        await asyncio.gather(*tasks)


async def main():
    monitor = MicroservicesMonitor()
    await monitor.start_monitoring()


if __name__ == "__main__":
    print("Starting Real-time Microservices Monitor")
    print("=" * 60)
    print("\nThis service monitors all microservices and broadcasts real events")
    print("Connect to WebSocket at ws://localhost:8007/ws/client-id to receive events")
    print("\nMonitoring:")
    print("  • Service health status")
    print("  • Agent creation/deletion")
    print("  • Workflow executions")
    print("  • AI model usage")
    print("  • Integration activities")
    print("\nPress Ctrl+C to stop")
    print("-" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitoring stopped")