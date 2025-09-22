#!/usr/bin/env python3
"""
Saga Transaction Client for Agent Lightning
Provides a client interface for interacting with the distributed transaction coordinator
"""

import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SagaTransaction:
    """Client-side transaction representation"""
    id: str
    name: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    steps: List[Dict[str, Any]] = None
    context: Dict[str, Any] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'SagaTransaction':
        """Create transaction from API response"""
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            steps=data.get("steps", []),
            context=data.get("context", {}),
            error=data.get("error")
        )


class SagaClient:
    """Client for interacting with the Saga Transaction Coordinator"""

    def __init__(self, base_url: str = "http://localhost:8008", timeout: int = 30):
        """Initialize the saga client

        Args:
            base_url: Base URL of the saga coordinator service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to saga coordinator"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

        url = f"{self.base_url}{endpoint}"

        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Request failed: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {e}")

    async def create_transaction(self, name: str, steps: List[Dict[str, Any]],
                               compensate_on_failure: bool = True,
                               timeout: int = 300) -> str:
        """Create a new saga transaction

        Args:
            name: Transaction name
            steps: List of transaction steps
            compensate_on_failure: Whether to compensate on failure
            timeout: Transaction timeout in seconds

        Returns:
            Transaction ID
        """
        payload = {
            "name": name,
            "steps": steps,
            "compensate_on_failure": compensate_on_failure,
            "timeout": timeout
        }

        response = await self._request("POST", "/api/v1/transactions", json=payload)
        return response["transaction_id"]

    async def execute_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Execute a saga transaction

        Args:
            transaction_id: ID of the transaction to execute

        Returns:
            Execution result
        """
        payload = {"transaction_id": transaction_id}
        return await self._request("POST", "/api/v1/transactions/execute", json=payload)

    async def get_transaction(self, transaction_id: str) -> SagaTransaction:
        """Get transaction status and details

        Args:
            transaction_id: Transaction ID

        Returns:
            Transaction object
        """
        response = await self._request("GET", f"/api/v1/transactions/{transaction_id}")
        return SagaTransaction.from_dict(response)

    async def list_transactions(self) -> List[Dict[str, Any]]:
        """List all transactions

        Returns:
            List of transaction summaries
        """
        return await self._request("GET", "/api/v1/transactions")

    async def get_transaction_history(self) -> List[Dict[str, Any]]:
        """Get transaction execution history

        Returns:
            List of historical transaction records
        """
        return await self._request("GET", "/api/v1/transactions/history")

    async def health_check(self) -> Dict[str, Any]:
        """Check service health

        Returns:
            Health status information
        """
        return await self._request("GET", "/health")

    # Pre-defined workflow methods
    async def create_agent_with_workflow(self, agent_config: Dict[str, Any],
                                       workflow_config: Dict[str, Any]) -> str:
        """Create an agent with associated workflow using pre-defined saga

        Args:
            agent_config: Agent configuration
            workflow_config: Workflow configuration

        Returns:
            Transaction ID
        """
        from services.distributed_transaction import SagaDefinitions

        steps = SagaDefinitions.create_agent_with_workflow()

        # Customize steps with provided config
        steps[0]["params"].update(agent_config)
        steps[1]["params"].update(workflow_config)

        return await self.create_transaction("Create Agent with Workflow", steps)

    async def create_visual_workflow(self, project_config: Dict[str, Any],
                                   component_configs: List[Dict[str, Any]]) -> str:
        """Create a visual workflow using pre-defined saga

        Args:
            project_config: Project configuration
            component_configs: List of component configurations

        Returns:
            Transaction ID
        """
        from services.distributed_transaction import SagaDefinitions

        steps = SagaDefinitions.create_visual_workflow()

        # Customize steps with provided config
        steps[0]["params"].update(project_config)
        steps[1]["params"]["components"] = component_configs

        return await self.create_transaction("Create Visual Workflow", steps)

    async def deploy_agent_system(self, agent_config: Dict[str, Any],
                                deployment_config: Dict[str, Any]) -> str:
        """Deploy a complete agent system using pre-defined saga

        Args:
            agent_config: Agent configuration
            deployment_config: Deployment configuration

        Returns:
            Transaction ID
        """
        from services.distributed_transaction import SagaDefinitions

        steps = SagaDefinitions.deploy_agent_system()

        # Customize steps with provided config
        steps[0]["params"].update(agent_config)
        steps[3]["params"].update(deployment_config)

        return await self.create_transaction("Deploy Agent System", steps)


# Synchronous wrapper for convenience
class SagaClientSync:
    """Synchronous wrapper for SagaClient"""

    def __init__(self, base_url: str = "http://localhost:8008", timeout: int = 30):
        """Initialize synchronous saga client"""
        self.client = SagaClient(base_url, timeout)

    def create_transaction(self, name: str, steps: List[Dict[str, Any]],
                         compensate_on_failure: bool = True,
                         timeout: int = 300) -> str:
        """Create a new saga transaction (sync)"""
        import asyncio
        return asyncio.run(self.client.create_transaction(name, steps, compensate_on_failure, timeout))

    def execute_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Execute a saga transaction (sync)"""
        import asyncio
        return asyncio.run(self.client.execute_transaction(transaction_id))

    def get_transaction(self, transaction_id: str) -> SagaTransaction:
        """Get transaction status (sync)"""
        import asyncio
        return asyncio.run(self.client.get_transaction(transaction_id))

    def list_transactions(self) -> List[Dict[str, Any]]:
        """List all transactions (sync)"""
        import asyncio
        return asyncio.run(self.client.list_transactions())

    def health_check(self) -> Dict[str, Any]:
        """Check service health (sync)"""
        import asyncio
        return asyncio.run(self.client.health_check())


# Convenience functions
async def create_saga_transaction(name: str, steps: List[Dict[str, Any]],
                                base_url: str = "http://localhost:8008") -> str:
    """Convenience function to create a saga transaction"""
    async with SagaClient(base_url) as client:
        return await client.create_transaction(name, steps)


async def execute_saga_transaction(transaction_id: str,
                                 base_url: str = "http://localhost:8008") -> Dict[str, Any]:
    """Convenience function to execute a saga transaction"""
    async with SagaClient(base_url) as client:
        return await client.execute_transaction(transaction_id)


def create_agent_workflow_saga(agent_config: Dict[str, Any],
                              workflow_config: Dict[str, Any],
                              base_url: str = "http://localhost:8008") -> str:
    """Convenience function to create agent with workflow"""
    client = SagaClientSync(base_url)
    return client.create_agent_with_workflow(agent_config, workflow_config)


def deploy_agent_system_saga(agent_config: Dict[str, Any],
                           deployment_config: Dict[str, Any],
                           base_url: str = "http://localhost:8008") -> str:
    """Convenience function to deploy agent system"""
    client = SagaClientSync(base_url)
    return client.deploy_agent_system(agent_config, deployment_config)


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def example():
        """Example usage of the saga client"""
        async with SagaClient() as client:
            # Check health
            health = await client.health_check()
            print(f"Service health: {health}")

            # Create a simple transaction
            steps = [
                {
                    "name": "Test Step",
                    "service": "test",
                    "action": "api/v1/test",
                    "params": {"message": "Hello World"}
                }
            ]

            try:
                transaction_id = await client.create_transaction("Example Transaction", steps)
                print(f"Created transaction: {transaction_id}")

                # Get transaction status
                transaction = await client.get_transaction(transaction_id)
                print(f"Transaction status: {transaction.status}")

            except Exception as e:
                print(f"Error: {e}")

    asyncio.run(example())