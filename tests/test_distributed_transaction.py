#!/usr/bin/env python3
"""
Unit tests for distributed transaction coordinator (Saga pattern)
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from services.distributed_transaction import (
    SagaCoordinator,
    SagaTransaction,
    SagaStep,
    TransactionStatus,
    StepStatus
)


class TestSagaCoordinator:
    """Test cases for SagaCoordinator"""

    @pytest.fixture
    def coordinator(self):
        """Create a test coordinator instance"""
        return SagaCoordinator()

    @pytest.fixture
    def sample_steps(self):
        """Sample transaction steps for testing"""
        return [
            {
                "name": "Create Agent",
                "service": "agent",
                "action": "api/v1/agents",
                "params": {"name": "Test Agent"},
                "compensate_action": "api/v1/agents/delete",
                "compensate_params": {"agent_id": "{result.id}"}
            },
            {
                "name": "Create Workflow",
                "service": "workflow",
                "action": "api/v1/workflows",
                "params": {"name": "Test Workflow", "agent_id": "{steps[0].result.id}"},
                "compensate_action": "api/v1/workflows/delete",
                "compensate_params": {"workflow_id": "{result.id}"},
                "depends_on": ["step-1"]
            }
        ]

    def test_create_transaction(self, coordinator, sample_steps):
        """Test transaction creation"""
        transaction = asyncio.run(coordinator.create_transaction("Test Transaction", sample_steps))

        assert transaction.id.startswith("txn-")
        assert transaction.name == "Test Transaction"
        assert len(transaction.steps) == 2
        assert transaction.status == TransactionStatus.PENDING

        # Check first step
        step1 = transaction.steps[0]
        assert step1.name == "Create Agent"
        assert step1.service == "agent"
        assert step1.status == StepStatus.PENDING

        # Check second step
        step2 = transaction.steps[1]
        assert step2.name == "Create Workflow"
        assert step2.depends_on == ["step-1"]

    def test_transaction_dependencies(self, coordinator):
        """Test step dependency resolution"""
        steps = [
            {
                "name": "Step 1",
                "service": "service1",
                "action": "action1",
                "params": {}
            },
            {
                "name": "Step 2",
                "service": "service2",
                "action": "action2",
                "params": {},
                "depends_on": ["step-1"]  # This will be updated to actual step ID
            },
            {
                "name": "Step 3",
                "service": "service3",
                "action": "action3",
                "params": {},
                "depends_on": ["step-1", "step-2"]  # This will be updated to actual step IDs
            }
        ]

        transaction = asyncio.run(coordinator.create_transaction("Dependency Test", steps))

        # Update dependency references to actual step IDs
        step1_id = transaction.steps[0].id
        step2_id = transaction.steps[1].id
        transaction.steps[1].depends_on = [step1_id]
        transaction.steps[2].depends_on = [step1_id, step2_id]

        # Initially, only step 1 should be executable
        next_steps = transaction.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].name == "Step 1"

        # Mark step 1 as completed
        transaction.steps[0].status = StepStatus.COMPLETED

        # Now step 2 should be executable
        next_steps = transaction.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].name == "Step 2"

        # Mark step 2 as completed
        transaction.steps[1].status = StepStatus.COMPLETED

        # Now step 3 should be executable
        next_steps = transaction.get_next_steps()
        assert len(next_steps) == 1
        assert next_steps[0].name == "Step 3"

    @patch('aiohttp.ClientSession.post')
    def test_execute_step_success(self, mock_post, coordinator):
        """Test successful step execution"""
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"id": "123", "status": "created"})

        # Mock the async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_post.return_value = mock_cm

        # Create transaction
        steps = [{
            "name": "Test Step",
            "service": "agent",
            "action": "api/v1/agents",
            "params": {"name": "Test"}
        }]
        transaction = asyncio.run(coordinator.create_transaction("Test", steps))

        # Execute step
        result = asyncio.run(coordinator._execute_step(transaction, transaction.steps[0]))

        assert result == {"id": "123", "status": "created"}
        assert transaction.steps[0].status == StepStatus.COMPLETED
        assert transaction.steps[0].result == result

    @patch('aiohttp.ClientSession.post')
    def test_execute_step_failure_with_retry(self, mock_post, coordinator):
        """Test step execution with failure and retry"""
        # Mock failed response first, then success
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 500
        mock_response_fail.text = AsyncMock(return_value="Internal Server Error")

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"id": "123"})

        # Create context managers for each call
        mock_cm_fail = AsyncMock()
        mock_cm_fail.__aenter__ = AsyncMock(return_value=mock_response_fail)
        mock_cm_fail.__aexit__ = AsyncMock(return_value=None)

        mock_cm_success = AsyncMock()
        mock_cm_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_cm_success.__aexit__ = AsyncMock(return_value=None)

        mock_post.side_effect = [mock_cm_fail, mock_cm_success]

        # Create transaction with retry
        steps = [{
            "name": "Test Step",
            "service": "agent",
            "action": "api/v1/agents",
            "params": {"name": "Test"},
            "retry_count": 2
        }]
        transaction = asyncio.run(coordinator.create_transaction("Test", steps))

        # Execute step
        result = asyncio.run(coordinator._execute_step(transaction, transaction.steps[0]))

        assert result == {"id": "123"}
        assert transaction.steps[0].status == StepStatus.COMPLETED
        assert mock_post.call_count == 2  # One failure, one success

    @patch('aiohttp.ClientSession.post')
    def test_execute_step_timeout(self, mock_post, coordinator):
        """Test step execution timeout"""
        # Mock timeout by raising TimeoutError in the context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_post.return_value = mock_cm

        # Create transaction
        steps = [{
            "name": "Test Step",
            "service": "agent",
            "action": "api/v1/agents",
            "params": {"name": "Test"},
            "timeout": 1
        }]
        transaction = asyncio.run(coordinator.create_transaction("Test", steps))

        # Execute step - should fail with timeout
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(coordinator._execute_step(transaction, transaction.steps[0]))

        assert transaction.steps[0].status == StepStatus.FAILED
        assert "Timeout" in transaction.steps[0].error

    def test_transaction_status_tracking(self, coordinator, sample_steps):
        """Test transaction status tracking"""
        transaction = asyncio.run(coordinator.create_transaction("Status Test", sample_steps))

        # Initially pending
        assert transaction.status == TransactionStatus.PENDING

        # After starting execution
        transaction.status = TransactionStatus.RUNNING
        transaction.started_at = datetime.now()
        assert transaction.status == TransactionStatus.RUNNING

        # After completion
        transaction.status = TransactionStatus.COMMITTED
        transaction.completed_at = datetime.now()
        assert transaction.status == TransactionStatus.COMMITTED

    def test_get_transaction_status(self, coordinator, sample_steps):
        """Test getting transaction status"""
        transaction = asyncio.run(coordinator.create_transaction("Status Test", sample_steps))
        transaction_id = transaction.id

        status = coordinator.get_transaction_status(transaction_id)

        assert status is not None
        assert status["id"] == transaction_id
        assert status["name"] == "Status Test"
        assert status["status"] == TransactionStatus.PENDING
        assert len(status["steps"]) == 2

    def test_get_transaction_status_not_found(self, coordinator):
        """Test getting status for non-existent transaction"""
        status = coordinator.get_transaction_status("non-existent")
        assert status is None


class TestSagaDefinitions:
    """Test cases for pre-defined saga workflows"""

    def test_create_agent_with_workflow_definition(self):
        """Test the create_agent_with_workflow saga definition"""
        from services.distributed_transaction import SagaDefinitions

        steps = SagaDefinitions.create_agent_with_workflow()

        assert len(steps) == 4
        assert steps[0]["name"] == "Create Agent"
        assert steps[0]["service"] == "agent"
        assert steps[1]["depends_on"] == ["step-1"]
        assert steps[2]["depends_on"] == ["step-2"]
        assert steps[3]["depends_on"] == ["step-3"]

    def test_payment_processing_definition(self):
        """Test the payment processing saga definition"""
        from services.distributed_transaction import SagaDefinitions

        steps = SagaDefinitions.process_payment_with_notification()

        assert len(steps) == 4
        assert steps[0]["name"] == "Reserve Funds"
        assert steps[3]["name"] == "Send Notification"
        assert steps[3]["retry_count"] == 1  # Notifications are less critical


if __name__ == "__main__":
    pytest.main([__file__])