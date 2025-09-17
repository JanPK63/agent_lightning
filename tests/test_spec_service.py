#!/usr/bin/env python3
"""
Unit tests for the Spec Service.
"""

import pytest
from agentlightning.types import Spec, WorkflowStep
from spec_service import SpecService


class TestSpecService:
    """Test cases for SpecService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = SpecService()

    def test_validate_spec_valid(self):
        """Test validation of a valid spec."""
        spec = Spec(
            id="test_spec",
            name="Test Spec",
            description="A test specification",
            workflow=[
                WorkflowStep(
                    id="step1",
                    name="First Step",
                    description="The first step",
                    dependencies=[]
                ),
                WorkflowStep(
                    id="step2",
                    name="Second Step",
                    description="The second step",
                    dependencies=["step1"]
                )
            ]
        )

        assert self.service.validate_spec(spec) is True

    def test_validate_spec_invalid_dependencies(self):
        """Test validation of spec with invalid dependencies."""
        spec = Spec(
            id="test_spec",
            name="Test Spec",
            description="A test specification",
            workflow=[
                WorkflowStep(
                    id="step1",
                    name="First Step",
                    description="The first step",
                    dependencies=["nonexistent"]
                )
            ]
        )

        assert self.service.validate_spec(spec) is False

    def test_validate_spec_missing_id(self):
        """Test validation of spec with missing ID."""
        spec = Spec(
            id="",
            name="Test Spec",
            description="A test specification",
            workflow=[]
        )

        assert self.service.validate_spec(spec) is False

    def test_save_and_get_spec(self):
        """Test saving and retrieving a spec."""
        spec = Spec(
            id="test_spec",
            name="Test Spec",
            description="A test specification",
            workflow=[]
        )

        self.service.save_spec(spec)
        retrieved = self.service.get_spec("test_spec")

        assert retrieved is not None
        assert retrieved.id == "test_spec"
        assert retrieved.name == "Test Spec"

    def test_get_nonexistent_spec(self):
        """Test retrieving a non-existent spec."""
        retrieved = self.service.get_spec("nonexistent")
        assert retrieved is None

    def test_generate_plan(self):
        """Test plan generation from spec."""
        spec = Spec(
            id="test_spec",
            name="Test Spec",
            description="A test specification",
            workflow=[
                WorkflowStep(
                    id="step1",
                    name="First Step",
                    description="The first step",
                    dependencies=[]
                )
            ]
        )

        plan = self.service.generate_plan(spec)

        assert plan.spec_id == "test_spec"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].input["step_id"] == "step1"
        assert plan.workflow_definition is not None

    def test_execute_plan(self):
        """Test plan execution."""
        spec = Spec(
            id="test_spec",
            name="Test Spec",
            description="A test specification",
            workflow=[
                WorkflowStep(
                    id="step1",
                    name="First Step",
                    description="The first step",
                    dependencies=[]
                )
            ]
        )

        plan = self.service.generate_plan(spec)
        execution_id = self.service.execute_plan(plan)

        assert execution_id.startswith("exec_test_spec_")
        execution = self.service.get_execution(execution_id)
        assert execution is not None
        assert execution.status == "completed"
        assert len(execution.rollouts) == 1

    def test_get_execution_nonexistent(self):
        """Test getting a non-existent execution."""
        execution = self.service.get_execution("nonexistent")
        assert execution is None


if __name__ == "__main__":
    pytest.main([__file__])