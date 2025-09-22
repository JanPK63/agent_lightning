#!/usr/bin/env python3
"""
Integration tests for distributed transaction coordinator
Tests full saga workflows with mocked services
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from services.distributed_transaction import TransactionService


class TestDistributedTransactionIntegration:
    """Integration tests for distributed transactions"""

    @pytest.fixture
    def client(self):
        """Create test client for the transaction service"""
        service = TransactionService()
        return TestClient(service.app)

    @pytest.fixture
    def sample_transaction_data(self):
        """Sample transaction data for testing"""
        return {
            "name": "Create Agent Workflow",
            "steps": [
                {
                    "name": "Create Agent",
                    "service": "agent",
                    "action": "api/v1/agents",
                    "params": {"name": "Test Agent", "type": "conversational"},
                    "compensate_action": "api/v1/agents/delete",
                    "compensate_params": {"agent_id": "{result.id}"}
                },
                {
                    "name": "Create Workflow",
                    "service": "workflow",
                    "action": "api/v1/workflows",
                    "params": {
                        "name": "Agent Workflow",
                        "agent_id": "{steps[0].result.id}"
                    },
                    "compensate_action": "api/v1/workflows/delete",
                    "compensate_params": {"workflow_id": "{result.id}"},
                    "depends_on": ["step-1"]
                }
            ]
        }

    def test_create_transaction_api(self, client, sample_transaction_data):
        """Test creating transaction via API"""
        response = client.post("/api/v1/transactions", json=sample_transaction_data)

        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert data["status"] == "pending"
        assert data["steps"] == 2

    def test_get_transaction_status_api(self, client, sample_transaction_data):
        """Test getting transaction status via API"""
        # Create transaction
        create_response = client.post("/api/v1/transactions", json=sample_transaction_data)
        transaction_id = create_response.json()["transaction_id"]

        # Get status
        status_response = client.get(f"/api/v1/transactions/{transaction_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["id"] == transaction_id
        assert status_data["name"] == "Create Agent Workflow"
        assert status_data["status"] == "pending"
        assert len(status_data["steps"]) == 2

    def test_get_nonexistent_transaction_api(self, client):
        """Test getting status of non-existent transaction"""
        response = client.get("/api/v1/transactions/non-existent-id")
        assert response.status_code == 404
        assert "Transaction not found" in response.json()["detail"]

    def test_list_transactions_api(self, client, sample_transaction_data):
        """Test listing all transactions via API"""
        # Create a transaction
        client.post("/api/v1/transactions", json=sample_transaction_data)

        # List transactions
        response = client.get("/api/v1/transactions")
        assert response.status_code == 200

        transactions = response.json()
        assert isinstance(transactions, list)
        assert len(transactions) >= 1

        # Check structure
        tx = transactions[0]
        assert "id" in tx
        assert "name" in tx
        assert "status" in tx
        assert "created_at" in tx
        assert "steps" in tx

    @patch('aiohttp.ClientSession')
    def test_execute_transaction_success(self, mock_session_class, client, sample_transaction_data):
        """Test successful transaction execution"""
        # Mock successful responses for both steps
        mock_responses = [
            # Agent creation response
            AsyncMock(status=200, json=AsyncMock(return_value={"id": "agent-123", "name": "Test Agent"})),
            # Workflow creation response
            AsyncMock(status=200, json=AsyncMock(return_value={"id": "workflow-456", "name": "Agent Workflow"}))
        ]

        mock_session = AsyncMock()
        mock_session.post.side_effect = mock_responses
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)

        # Create transaction
        create_response = client.post("/api/v1/transactions", json=sample_transaction_data)
        transaction_id = create_response.json()["transaction_id"]

        # Execute transaction
        execute_response = client.post("/api/v1/transactions/execute",
                                     json={"transaction_id": transaction_id})

        assert execute_response.status_code == 200
        result = execute_response.json()
        assert result["status"] == "committed"
        assert result["transaction_id"] == transaction_id
        assert "results" in result

    @patch('aiohttp.ClientSession')
    def test_execute_transaction_with_compensation(self, mock_session_class, client, sample_transaction_data):
        """Test transaction execution with failure and compensation"""
        # Mock responses: first step succeeds, second fails
        mock_responses = [
            # Agent creation succeeds
            AsyncMock(status=200, json=AsyncMock(return_value={"id": "agent-123", "name": "Test Agent"})),
            # Workflow creation fails
            AsyncMock(status=500, text=AsyncMock(return_value="Workflow creation failed")),
            # Compensation for agent deletion succeeds
            AsyncMock(status=200, json=AsyncMock(return_value={"deleted": True}))
        ]

        mock_session = AsyncMock()
        mock_session.post.side_effect = mock_responses
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)

        # Create transaction
        create_response = client.post("/api/v1/transactions", json=sample_transaction_data)
        transaction_id = create_response.json()["transaction_id"]

        # Execute transaction
        execute_response = client.post("/api/v1/transactions/execute",
                                     json={"transaction_id": transaction_id})

        assert execute_response.status_code == 200
        result = execute_response.json()
        assert result["status"] == "compensated"
        assert result["transaction_id"] == transaction_id
        assert "error" in result

    def test_health_check_api(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["service"] == "distributed-transaction"
        assert "transactions_active" in health_data

    def test_transaction_history_api(self, client):
        """Test transaction history endpoint"""
        response = client.get("/api/v1/transactions/history")
        assert response.status_code == 200

        history = response.json()
        assert isinstance(history, list)

    @patch('aiohttp.ClientSession')
    def test_transaction_timeout_handling(self, mock_session_class, client):
        """Test transaction timeout handling"""
        # Mock timeout on first step
        mock_session = AsyncMock()
        mock_session.post.side_effect = asyncio.TimeoutError()
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)

        transaction_data = {
            "name": "Timeout Test",
            "steps": [{
                "name": "Timeout Step",
                "service": "agent",
                "action": "api/v1/agents",
                "params": {"name": "Test"},
                "timeout": 1
            }]
        }

        # Create and execute transaction
        create_response = client.post("/api/v1/transactions", json=transaction_data)
        transaction_id = create_response.json()["transaction_id"]

        execute_response = client.post("/api/v1/transactions/execute",
                                     json={"transaction_id": transaction_id})

        assert execute_response.status_code == 200
        result = execute_response.json()
        assert result["status"] == "failed"
        assert "error" in result

    def test_transaction_context_passing(self, client):
        """Test that transaction context is properly passed between steps"""
        transaction_data = {
            "name": "Context Test",
            "steps": [
                {
                    "name": "Step 1",
                    "service": "service1",
                    "action": "action1",
                    "params": {"input": "test"}
                },
                {
                    "name": "Step 2",
                    "service": "service2",
                    "action": "action2",
                    "params": {"previous_output": "{steps[0].result.output}"},
                    "depends_on": ["step-1"]
                }
            ]
        }

        response = client.post("/api/v1/transactions", json=transaction_data)
        assert response.status_code == 200

        # Verify the transaction was created with proper dependencies
        transaction_id = response.json()["transaction_id"]
        status_response = client.get(f"/api/v1/transactions/{transaction_id}")
        status_data = status_response.json()

        assert len(status_data["steps"]) == 2
        assert status_data["steps"][1]["depends_on"] == ["step-1"]


if __name__ == "__main__":
    pytest.main([__file__])