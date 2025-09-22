#!/usr/bin/env python3
"""
Agent Service Orchestrator

Manages the lifecycle of agent services, providing:
- Automatic service discovery and registration
- Health monitoring and failover
- Load balancing across agent instances
- Service mesh for inter-agent communication
- Integration with the task execution pipeline

This orchestrator bridges the gap between agent definitions in the capability matcher
and actual running agent services.
"""

import asyncio
import aiohttp
import json
import logging
import time
import subprocess
import signal
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
import psutil

from agent_capability_matcher import get_capability_matcher, AgentCapability

logger = logging.getLogger(__name__)


@dataclass
class AgentServiceInstance:
    """Represents a running agent service instance"""
    agent_id: str
    service_url: str
    port: int
    process: Optional[subprocess.Popen] = None
    status: str = "stopped"  # stopped, starting, running, unhealthy, failed
    last_health_check: Optional[datetime] = None
    health_check_interval: int = 30  # seconds
    consecutive_failures: int = 0
    max_failures: int = 3
    capabilities: List[str] = field(default_factory=list)
    specialization: str = ""
    load_factor: float = 0.0  # 0.0 to 1.0
    last_task_time: Optional[datetime] = None
    task_count: int = 0
    startup_time: Optional[datetime] = None


@dataclass
class OrchestratorConfig:
    """Configuration for the agent service orchestrator"""
    health_check_interval: int = 30
    service_timeout: int = 300  # seconds to wait for service startup
    max_concurrent_tasks_per_agent: int = 5
    load_balancing_enabled: bool = True
    auto_start_services: bool = True
    service_discovery_port_range: tuple = (9001, 9060)  # Range for agent services
    mock_fallback_enabled: bool = True


class AgentServiceOrchestrator:
    """
    Orchestrates agent services, managing their lifecycle and providing
    a unified interface for task execution.
    """

    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.capability_matcher = get_capability_matcher()
        self.services: Dict[str, List[AgentServiceInstance]] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        # Health monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Service mesh for inter-agent communication
        self.service_mesh = {}
        self.mesh_enabled = False

        logger.info("Agent Service Orchestrator initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()

    async def initialize(self):
        """Initialize the orchestrator"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

        # Start health monitoring
        self.start_health_monitoring()

        # Auto-discover existing services
        await self.discover_services()

        # Auto-start configured services if enabled
        if self.config.auto_start_services:
            await self.auto_start_services()

        logger.info("Agent Service Orchestrator ready")

    async def shutdown(self):
        """Shutdown the orchestrator"""
        self.stop_health_monitoring()

        if self.session:
            await self.session.close()

        # Stop all managed services
        await self.stop_all_services()

        logger.info("Agent Service Orchestrator shutdown complete")

    def start_health_monitoring(self):
        """Start the health monitoring thread"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop the health monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    def _health_monitor_loop(self):
        """Health monitoring loop (runs in separate thread)"""
        while self.monitoring_active:
            try:
                # Run health checks in event loop
                asyncio.run(self._perform_health_checks())
            except Exception as e:
                logger.error(f"Health check error: {e}")

            time.sleep(self.config.health_check_interval)

    async def _perform_health_checks(self):
        """Perform health checks on all services"""
        for agent_id, instances in self.services.items():
            for instance in instances:
                await self._check_service_health(instance)

    async def _check_service_health(self, instance: AgentServiceInstance) -> bool:
        """Check health of a specific service instance"""
        try:
            health_url = f"{instance.service_url}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    data = await response.json()
                    instance.status = "running"
                    instance.consecutive_failures = 0
                    instance.last_health_check = datetime.now()

                    # Update capabilities if available
                    if "capabilities" in data:
                        instance.capabilities = data["capabilities"]

                    return True
                else:
                    instance.consecutive_failures += 1

        except Exception as e:
            instance.consecutive_failures += 1
            logger.warning(f"Health check failed for {instance.agent_id}: {e}")

        # Handle failures
        instance.last_health_check = datetime.now()

        if instance.consecutive_failures >= instance.max_failures:
            instance.status = "failed"
            logger.error(f"Service {instance.agent_id} marked as failed after {instance.consecutive_failures} failures")

            # Attempt restart if auto-restart is enabled
            if self.config.auto_start_services:
                await self._restart_service(instance)
        else:
            instance.status = "unhealthy"

        return False

    async def _restart_service(self, instance: AgentServiceInstance):
        """Attempt to restart a failed service"""
        logger.info(f"Attempting to restart service {instance.agent_id}")

        # Stop existing process if running
        if instance.process and instance.process.poll() is None:
            instance.process.terminate()
            try:
                instance.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                instance.process.kill()

        # Start new instance
        success = await self.start_service(instance.agent_id)
        if success:
            logger.info(f"Successfully restarted service {instance.agent_id}")
        else:
            logger.error(f"Failed to restart service {instance.agent_id}")

    async def discover_services(self):
        """Discover existing agent services by scanning ports"""
        logger.info("Discovering existing agent services...")

        discovered_count = 0

        for port in range(self.config.service_discovery_port_range[0],
                         self.config.service_discovery_port_range[1] + 1):
            try:
                service_url = f"http://localhost:{port}"
                health_url = f"{service_url}/health"

                async with self.session.get(health_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        data = await response.json()
                        agent_id = data.get("agent_id")

                        if agent_id:
                            # Register discovered service
                            instance = AgentServiceInstance(
                                agent_id=agent_id,
                                service_url=service_url,
                                port=port,
                                status="running",
                                capabilities=data.get("capabilities", []),
                                specialization=data.get("specialization", "")
                            )

                            if agent_id not in self.services:
                                self.services[agent_id] = []
                            self.services[agent_id].append(instance)

                            discovered_count += 1
                            logger.info(f"Discovered service: {agent_id} on port {port}")

            except Exception:
                # Port not responding, continue
                continue

        logger.info(f"Service discovery complete. Found {discovered_count} running services")

    async def auto_start_services(self):
        """Auto-start services for agents that don't have running instances"""
        logger.info("Auto-starting missing agent services...")

        started_count = 0

        for agent_id, agent_capability in self.capability_matcher.agents.items():
            # Check if we have any running instances
            instances = self.services.get(agent_id, [])
            running_instances = [i for i in instances if i.status == "running"]

            if not running_instances:
                # No running instances, try to start one
                success = await self.start_service(agent_id)
                if success:
                    started_count += 1

        logger.info(f"Auto-started {started_count} agent services")

    async def start_service(self, agent_id: str, port: Optional[int] = None) -> bool:
        """Start an agent service"""
        try:
            agent_capability = self.capability_matcher.get_agent_info(agent_id)
            if not agent_capability:
                logger.error(f"No capability definition found for agent {agent_id}")
                return False

            # Find available port
            if port is None:
                port = await self._find_available_port()

            if port is None:
                logger.error(f"No available ports for agent {agent_id}")
                return False

            # Determine service script path
            service_script = self._get_service_script_path(agent_id)
            if not service_script.exists():
                logger.error(f"Service script not found: {service_script}")
                return False

            # Start the service process
            cmd = [
                "python", str(service_script),
                "--agent-id", agent_id,
                "--port", str(port)
            ]

            logger.info(f"Starting service {agent_id} on port {port}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )

            # Create service instance
            instance = AgentServiceInstance(
                agent_id=agent_id,
                service_url=f"http://localhost:{port}",
                port=port,
                process=process,
                status="starting",
                capabilities=agent_capability.capabilities,
                specialization=agent_capability.name,
                startup_time=datetime.now()
            )

            if agent_id not in self.services:
                self.services[agent_id] = []
            self.services[agent_id].append(instance)

            # Wait for service to become healthy
            await self._wait_for_service_startup(instance)

            return instance.status == "running"

        except Exception as e:
            logger.error(f"Failed to start service {agent_id}: {e}")
            return False

    async def _wait_for_service_startup(self, instance: AgentServiceInstance):
        """Wait for a service to start up and become healthy"""
        start_time = time.time()

        while time.time() - start_time < self.config.service_timeout:
            await self._check_service_health(instance)
            if instance.status == "running":
                logger.info(f"Service {instance.agent_id} started successfully on port {instance.port}")
                return

            await asyncio.sleep(2)

        logger.error(f"Service {instance.agent_id} failed to start within timeout")
        instance.status = "failed"

    async def _find_available_port(self) -> Optional[int]:
        """Find an available port in the configured range"""
        for port in range(self.config.service_discovery_port_range[0],
                         self.config.service_discovery_port_range[1] + 1):

            # Check if port is in use
            if self._is_port_in_use(port):
                continue

            # Check if we already have a service on this port
            port_in_use = any(
                instance.port == port
                for instances in self.services.values()
                for instance in instances
            )

            if not port_in_use:
                return port

        return None

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except OSError:
                return True

    def _get_service_script_path(self, agent_id: str) -> Path:
        """Get the service script path for an agent"""
        # Map agent IDs to service scripts
        service_scripts = {
            "web_developer": "services/web_developer_agent.py",
            "data_analyst": "services/data_analyst_agent.py",
            "security_expert": "services/security_expert_agent.py",
            "devops_engineer": "services/devops_engineer_agent.py",
            "qa_tester": "services/qa_tester_agent.py",
            "general_assistant": "services/general_assistant_agent.py"
        }

        script_name = service_scripts.get(agent_id, f"services/{agent_id}_agent.py")
        return Path(script_name)

    async def stop_service(self, agent_id: str, port: Optional[int] = None) -> bool:
        """Stop an agent service"""
        try:
            instances = self.services.get(agent_id, [])

            if port:
                instances = [i for i in instances if i.port == port]

            stopped_count = 0
            for instance in instances:
                if instance.process and instance.process.poll() is None:
                    instance.process.terminate()
                    try:
                        instance.process.wait(timeout=10)
                        instance.status = "stopped"
                        stopped_count += 1
                    except subprocess.TimeoutExpired:
                        instance.process.kill()
                        instance.status = "killed"
                        stopped_count += 1

            return stopped_count > 0

        except Exception as e:
            logger.error(f"Failed to stop service {agent_id}: {e}")
            return False

    async def stop_all_services(self):
        """Stop all managed services"""
        logger.info("Stopping all agent services...")

        stop_tasks = []
        for agent_id in self.services.keys():
            stop_tasks.append(self.stop_service(agent_id))

        await asyncio.gather(*stop_tasks, return_exceptions=True)
        logger.info("All services stopped")

    async def execute_task(self, task_description: str, agent_id: Optional[str] = None,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task using the appropriate agent service"""
        context = context or {}

        # Find best agent if not specified
        if not agent_id:
            agent_id, confidence, reason = self.capability_matcher.find_best_agent(task_description)
            logger.info(f"Auto-selected agent {agent_id} with confidence {confidence:.2f}")

        # Get available instances for the agent
        instances = self.services.get(agent_id, [])
        running_instances = [i for i in instances if i.status == "running"]

        if not running_instances:
            # No running instances, try to start one
            if self.config.auto_start_services:
                logger.info(f"No running instances for {agent_id}, attempting to start...")
                success = await self.start_service(agent_id)
                if success:
                    running_instances = [i for i in self.services.get(agent_id, []) if i.status == "running"]

            # If still no instances and mock fallback enabled, use mock
            if not running_instances and self.config.mock_fallback_enabled:
                return await self._execute_mock_task(task_description, agent_id, context)

            if not running_instances:
                raise RuntimeError(f"No available instances for agent {agent_id}")

        # Select instance (load balancing)
        instance = self._select_instance(running_instances)

        # Execute task
        return await self._execute_on_instance(instance, task_description, context)

    def _select_instance(self, instances: List[AgentServiceInstance]) -> AgentServiceInstance:
        """Select an instance for task execution (load balancing)"""
        if not self.config.load_balancing_enabled or len(instances) == 1:
            return instances[0]

        # Simple load balancing: select instance with lowest load factor
        return min(instances, key=lambda i: i.load_factor)

    async def _execute_on_instance(self, instance: AgentServiceInstance,
                                  task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task on a specific service instance"""
        try:
            execute_url = f"{instance.service_url}/execute"

            payload = {
                "task_id": f"task_{int(time.time())}_{instance.agent_id}",
                "task_description": task_description,
                "context": context
            }

            # Update load factor
            instance.load_factor = min(1.0, instance.load_factor + 0.2)
            instance.task_count += 1
            instance.last_task_time = datetime.now()

            async with self.session.post(execute_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Decrease load factor
                    instance.load_factor = max(0.0, instance.load_factor - 0.2)

                    return {
                        "status": "completed",
                        "agent_id": instance.agent_id,
                        "result": result,
                        "service_url": instance.service_url
                    }
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Service returned {response.status}: {error_text}")

        except Exception as e:
            # Decrease load factor on error
            instance.load_factor = max(0.0, instance.load_factor - 0.2)

            logger.error(f"Task execution failed on {instance.agent_id}: {e}")
            raise

    async def _execute_mock_task(self, task_description: str, agent_id: str,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mock task when no real service is available"""
        logger.warning(f"Using mock execution for agent {agent_id}")

        # Simple mock response based on agent type
        mock_responses = {
            "web_developer": "Mock web development response: Created HTML/CSS/JavaScript solution",
            "data_analyst": "Mock data analysis response: Generated insights and visualizations",
            "security_expert": "Mock security response: Performed security assessment",
            "devops_engineer": "Mock DevOps response: Configured deployment pipeline",
            "qa_tester": "Mock QA response: Executed test suite and found issues",
            "general_assistant": "Mock general response: Task completed successfully"
        }

        response = mock_responses.get(agent_id, f"Mock response from {agent_id}")

        return {
            "status": "completed",
            "agent_id": agent_id,
            "result": {
                "response": response,
                "mock": True,
                "task_description": task_description
            },
            "service_url": "mock://fallback"
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            "total_agents": len(self.capability_matcher.agents),
            "running_services": 0,
            "total_instances": 0,
            "services": {}
        }

        for agent_id, instances in self.services.items():
            service_info = {
                "instances": len(instances),
                "running": 0,
                "statuses": {}
            }

            for instance in instances:
                status["total_instances"] += 1
                service_info["statuses"][instance.port] = instance.status

                if instance.status == "running":
                    status["running_services"] += 1
                    service_info["running"] += 1

            status["services"][agent_id] = service_info

        return status

    def enable_service_mesh(self):
        """Enable service mesh for inter-agent communication"""
        self.mesh_enabled = True
        self.service_mesh = {}

        # Build service mesh routing table
        for agent_id, instances in self.services.items():
            running_instances = [i for i in instances if i.status == "running"]
            if running_instances:
                self.service_mesh[agent_id] = [i.service_url for i in running_instances]

        logger.info("Service mesh enabled")

    async def communicate_with_agent(self, from_agent: str, to_agent: str,
                                   message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message between agents via service mesh"""
        if not self.mesh_enabled:
            raise RuntimeError("Service mesh not enabled")

        if to_agent not in self.service_mesh:
            raise RuntimeError(f"Agent {to_agent} not available in service mesh")

        # Select target instance
        target_urls = self.service_mesh[to_agent]
        target_url = target_urls[0]  # Simple selection, could be load balanced

        # Send inter-agent message
        message_url = f"{target_url}/inter_agent_message"
        payload = {
            "from_agent": from_agent,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

        async with self.session.post(message_url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise RuntimeError(f"Inter-agent communication failed: {error_text}")


# Global orchestrator instance
_orchestrator: Optional[AgentServiceOrchestrator] = None


def get_orchestrator() -> AgentServiceOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentServiceOrchestrator()
    return _orchestrator


async def initialize_orchestrator():
    """Initialize the global orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentServiceOrchestrator()
        await _orchestrator.initialize()


async def shutdown_orchestrator():
    """Shutdown the global orchestrator"""
    global _orchestrator
    if _orchestrator:
        await _orchestrator.shutdown()
        _orchestrator = None


if __name__ == "__main__":
    # Example usage
    async def main():
        async with AgentServiceOrchestrator() as orchestrator:
            # Get status
            status = orchestrator.get_service_status()
            print(f"Service status: {status}")

            # Execute a task
            try:
                result = await orchestrator.execute_task(
                    "Create a simple HTML page with a button",
                    agent_id="web_developer"
                )
                print(f"Task result: {result}")
            except Exception as e:
                print(f"Task execution failed: {e}")

    asyncio.run(main())