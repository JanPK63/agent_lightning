#!/usr/bin/env python3
"""
Integration tests for the Spec Service.
Note: FastAPI TestClient has version compatibility issues in this environment.
These tests demonstrate the service functionality through direct service calls.
"""

from agentlightning.types import Spec, WorkflowStep
from spec_service import SpecService


def test_spec_service_integration():
    """Test the SpecService functionality directly."""
    service = SpecService()

    # Create a test spec
    spec = Spec(
        id="integration_test_spec",
        name="Integration Test Spec",
        description="A spec for integration testing",
        workflow=[
            WorkflowStep(
                id="step1",
                name="First Step",
                description="The first step in integration test",
                dependencies=[]
            ),
            WorkflowStep(
                id="step2",
                name="Second Step",
                description="The second step in integration test",
                dependencies=["step1"]
            )
        ]
    )

    # Test validation
    assert service.validate_spec(spec) is True

    # Test saving and retrieving
    service.save_spec(spec)
    retrieved = service.get_spec("integration_test_spec")
    assert retrieved is not None
    assert retrieved.id == "integration_test_spec"
    assert retrieved.name == "Integration Test Spec"

    # Test plan generation
    plan = service.generate_plan(spec)
    assert plan.spec_id == "integration_test_spec"
    assert len(plan.tasks) == 2
    assert plan.workflow_definition is not None

    # Test execution
    execution_id = service.execute_plan(plan)
    assert execution_id.startswith("exec_integration_test_spec_")

    # Test execution status
    execution = service.get_execution(execution_id)
    assert execution is not None
    assert execution.status == "completed"
    assert len(execution.rollouts) == 2

    print("✅ All integration tests passed!")


if __name__ == "__main__":
    test_spec_service_integration()