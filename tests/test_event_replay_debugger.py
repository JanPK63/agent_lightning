#!/usr/bin/env python3
"""
Tests for Event Replay Debugger functionality
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from shared.event_replay_debugger import (
    EventReplayDebugger,
    EventBreakpoint,
    DebugState
)
from shared.event_store import EventStore, create_event
from agentlightning.types import Event


class TestEventReplayDebugger:
    """Test cases for EventReplayDebugger"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_event_store = MagicMock(spec=EventStore)
        self.debugger = EventReplayDebugger(self.mock_event_store)

    def test_initial_state(self):
        """Test initial debugger state"""
        assert self.debugger.state == DebugState.IDLE
        assert self.debugger.current_event is None
        assert len(self.debugger.breakpoints) == 0
        assert len(self.debugger.execution_timeline) == 0

    def test_add_breakpoint(self):
        """Test adding breakpoints"""
        # Add breakpoint by event type
        bp_id = self.debugger.add_breakpoint(event_type="agent.created")
        assert bp_id == "bp_0"
        assert len(self.debugger.breakpoints) == 1
        assert self.debugger.breakpoints[0].event_type == "agent.created"

        # Add breakpoint with condition
        bp_id2 = self.debugger.add_breakpoint(
            event_type="task.completed",
            condition="event_data.get('reward', 0) > 0.8"
        )
        assert bp_id2 == "bp_1"
        assert len(self.debugger.breakpoints) == 2

    def test_remove_breakpoint(self):
        """Test removing breakpoints"""
        bp_id = self.debugger.add_breakpoint(event_type="test.event")

        # Remove existing breakpoint
        success = self.debugger.remove_breakpoint(bp_id)
        assert success is True
        assert len(self.debugger.breakpoints) == 0

        # Try to remove non-existent breakpoint
        success = self.debugger.remove_breakpoint("nonexistent")
        assert success is False

    def test_breakpoint_should_break(self):
        """Test breakpoint condition evaluation"""
        bp = EventBreakpoint("test", event_type="agent.created")

        # Create test event
        event = Event(
            event_id="test-event",
            aggregate_id="agent-123",
            aggregate_type="agent",
            event_type="agent.created",
            event_data={"name": "Test Agent"},
            timestamp=datetime.utcnow(),
            version=1
        )

        # Should break on matching event type
        assert bp.should_break(event)

        # Should not break on different event type
        event.event_type = "agent.updated"
        assert not bp.should_break(event)

    def test_breakpoint_with_condition(self):
        """Test breakpoint with condition"""
        bp = EventBreakpoint(
            "test",
            condition="event_data.get('reward', 0) > 0.8"
        )

        # Create test event with high reward
        event = Event(
            event_id="test-event",
            aggregate_id="task-123",
            aggregate_type="task",
            event_type="task.completed",
            event_data={"reward": 0.95},
            timestamp=datetime.utcnow(),
            version=1
        )

        # Should break when condition is met
        assert bp.should_break(event)

        # Should not break when condition is not met
        event.event_data["reward"] = 0.5
        assert not bp.should_break(event)

    def test_add_watch(self):
        """Test adding watch expressions"""
        # Add watch
        self.debugger.add_watch("event_data.get('reward', 0)")
        assert len(self.debugger.variable_watches) == 1

        # Add duplicate watch (should not duplicate)
        self.debugger.add_watch("event_data.get('reward', 0)")
        assert len(self.debugger.variable_watches) == 1

    def test_remove_watch(self):
        """Test removing watch expressions"""
        self.debugger.add_watch("test_expression")

        # Remove existing watch
        self.debugger.remove_watch("test_expression")
        assert len(self.debugger.variable_watches) == 0

        # Remove non-existent watch (should not error)
        self.debugger.remove_watch("nonexistent")
        assert len(self.debugger.variable_watches) == 0

    def test_get_debug_info(self):
        """Test getting debug information"""
        # Add some test data
        self.debugger.add_breakpoint(event_type="test.event")
        self.debugger.add_watch("event_type")

        info = self.debugger.get_debug_info()

        assert info["state"] == "idle"
        assert info["aggregate_id"] is None
        assert len(info["breakpoints"]) == 1
        assert len(info["watched_values"]) == 0  # No current event

    def test_export_debug_session(self):
        """Test exporting debug session"""
        # Add some test data
        self.debugger.add_breakpoint(event_type="test.event")
        self.debugger.add_watch("event_type")

        export_data = self.debugger.export_debug_session()

        assert "aggregate_id" in export_data
        assert "breakpoints" in export_data
        assert "execution_timeline" in export_data
        assert "variable_watches" in export_data
        assert "final_state" in export_data
        assert "exported_at" in export_data

    @pytest.mark.asyncio
    async def test_start_replay(self):
        """Test starting event replay"""
        # Mock the replay events generator
        mock_events = [
            Event(
                event_id="event-1",
                aggregate_id="agent-123",
                aggregate_type="agent",
                event_type="agent.created",
                event_data={"name": "Test Agent"},
                timestamp=datetime.utcnow(),
                version=1
            )
        ]

        async def mock_replay_generator():
            for event in mock_events:
                yield event

        self.mock_event_store.replay_events = MagicMock(
            return_value=mock_replay_generator()
        )

        # Start replay
        await self.debugger.start_replay("agent-123", 1)

        assert self.debugger.state == DebugState.RUNNING
        assert self.debugger.aggregate_id == "agent-123"
        assert self.debugger.from_version == 1

    @pytest.mark.asyncio
    async def test_pause_resume_replay(self):
        """Test pausing and resuming replay"""
        # Set debugger to running state
        self.debugger.state = DebugState.RUNNING

        # Pause
        await self.debugger.pause_replay()
        assert self.debugger.state == DebugState.PAUSED

        # Resume
        await self.debugger.resume_replay()
        assert self.debugger.state == DebugState.RUNNING

    @pytest.mark.asyncio
    async def test_stop_replay(self):
        """Test stopping replay"""
        # Set debugger to running state
        self.debugger.state = DebugState.RUNNING
        self.debugger.replay_generator = AsyncMock()

        # Stop
        await self.debugger.stop_replay()
        assert self.debugger.state == DebugState.STOPPED
        assert self.debugger.replay_generator is None
        assert self.debugger.current_event is None

    def test_get_watched_values_with_event(self):
        """Test getting watched values when there's a current event"""
        # Set current event
        self.debugger.current_event = Event(
            event_id="test-event",
            aggregate_id="task-123",
            aggregate_type="task",
            event_type="task.completed",
            event_data={"reward": 0.95, "duration": 45.2},
            timestamp=datetime.utcnow(),
            version=1
        )

        # Add watches
        self.debugger.add_watch("event_data.get('reward', 0)")
        self.debugger.add_watch("event_type")
        self.debugger.add_watch("version")

        watched_values = self.debugger.get_watched_values()

        assert "event_data.get('reward', 0)" in watched_values
        assert watched_values["event_data.get('reward', 0)"] == 0.95
        assert watched_values["event_type"] == "task.completed"
        assert watched_values["version"] == 1

    def test_get_watched_values_with_invalid_expression(self):
        """Test getting watched values with invalid expression"""
        # Set current event
        self.debugger.current_event = Event(
            event_id="test-event",
            aggregate_id="task-123",
            aggregate_type="task",
            event_type="task.completed",
            event_data={"reward": 0.95},
            timestamp=datetime.utcnow(),
            version=1
        )

        # Add invalid watch expression
        self.debugger.add_watch("invalid_syntax")

        watched_values = self.debugger.get_watched_values()

        assert "invalid_syntax" in watched_values
        assert "Error:" in str(watched_values["invalid_syntax"])


class TestEventBreakpoint:
    """Test cases for EventBreakpoint"""

    def test_breakpoint_creation(self):
        """Test creating breakpoints"""
        bp = EventBreakpoint("test_id", "agent.created", "version > 1")
        assert bp.breakpoint_id == "test_id"
        assert bp.event_type == "agent.created"
        assert bp.condition == "version > 1"
        assert bp.enabled is True

    def test_breakpoint_should_break_no_condition(self):
        """Test breakpoint without condition"""
        bp = EventBreakpoint("test", event_type="agent.created")

        event = Event(
            event_id="test",
            aggregate_id="agent-123",
            aggregate_type="agent",
            event_type="agent.created",
            event_data={},
            timestamp=datetime.utcnow(),
            version=1
        )

        assert bp.should_break(event)

        # Different event type
        event.event_type = "agent.updated"
        assert not bp.should_break(event)

    def test_breakpoint_should_break_with_condition(self):
        """Test breakpoint with condition"""
        bp = EventBreakpoint("test", condition="version > 1")

        event = Event(
            event_id="test",
            aggregate_id="agent-123",
            aggregate_type="agent",
            event_type="agent.created",
            event_data={},
            timestamp=datetime.utcnow(),
            version=2
        )

        assert bp.should_break(event)

        # Condition not met
        event.version = 1
        assert not bp.should_break(event)

    def test_breakpoint_disabled(self):
        """Test disabled breakpoint"""
        bp = EventBreakpoint("test", event_type="agent.created")
        bp.enabled = False

        event = Event(
            event_id="test",
            aggregate_id="agent-123",
            aggregate_type="agent",
            event_type="agent.created",
            event_data={},
            timestamp=datetime.utcnow(),
            version=1
        )

        assert not bp.should_break(event)


if __name__ == "__main__":
    pytest.main([__file__])