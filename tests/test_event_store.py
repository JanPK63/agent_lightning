#!/usr/bin/env python3
"""
Unit tests for the Event Store functionality
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from shared.event_store import EventStore, create_event, event_store
from agentlightning.types import Event, EventFilter


class TestEventStore:
    """Test cases for EventStore class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.event_store = EventStore()

    def teardown_method(self):
        """Clean up after tests"""
        # Close database connection
        if hasattr(self.event_store, 'db_session'):
            self.event_store.close()

    def test_save_event_success(self):
        """Test saving an event successfully"""
        # Create a test event
        event = create_event(
            aggregate_id="test-agent-123",
            aggregate_type="agent",
            event_type="created",
            event_data={"name": "Test Agent", "model": "gpt-4"},
            user_id="user-123",
            service_name="test-service"
        )

        # Save the event
        event_id = self.event_store.save_event(event)

        # Verify the event was saved
        assert event_id == event.event_id
        assert event_id is not None

        # Retrieve and verify
        retrieved_event = self.event_store.get_event(event_id)
        assert retrieved_event is not None
        assert retrieved_event.event_id == event_id
        assert retrieved_event.aggregate_id == "test-agent-123"
        assert retrieved_event.event_type == "created"

    def test_save_event_validation_error(self):
        """Test saving an event with validation errors"""
        # Create an invalid event (missing aggregate_id)
        event = Event(
            event_id=str(uuid.uuid4()),
            aggregate_id="",  # Invalid: empty aggregate_id
            aggregate_type="agent",
            event_type="created",
            event_data={"name": "Test"},
            timestamp=datetime.utcnow()
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Event must have an aggregate_id"):
            self.event_store.save_event(event)

    def test_get_event_not_found(self):
        """Test retrieving a non-existent event"""
        event = self.event_store.get_event("non-existent-id")
        assert event is None

    def test_get_events_by_aggregate(self):
        """Test retrieving events for a specific aggregate"""
        aggregate_id = "test-workflow-456"

        # Create and save multiple events for the same aggregate
        events = []
        for i in range(3):
            event = create_event(
                aggregate_id=aggregate_id,
                aggregate_type="workflow",
                event_type=f"step_{i}",
                event_data={"step": i, "status": "completed"},
                version=i + 1
            )
            self.event_store.save_event(event)
            events.append(event)

        # Retrieve events
        retrieved_events = self.event_store.get_events_by_aggregate(aggregate_id)

        # Verify
        assert len(retrieved_events) == 3
        assert all(e.aggregate_id == aggregate_id for e in retrieved_events)
        assert all(e.aggregate_type == "workflow" for e in retrieved_events)

        # Check ordering (should be by version ascending)
        versions = [e.version for e in retrieved_events]
        assert versions == [1, 2, 3]

    def test_get_events_by_aggregate_from_version(self):
        """Test retrieving events from a specific version"""
        aggregate_id = "test-task-789"

        # Create events with versions 1, 2, 3
        for version in range(1, 4):
            event = create_event(
                aggregate_id=aggregate_id,
                aggregate_type="task",
                event_type=f"update_{version}",
                event_data={"version": version},
                version=version
            )
            self.event_store.save_event(event)

        # Retrieve from version 2 onwards
        retrieved_events = self.event_store.get_events_by_aggregate(aggregate_id, from_version=2)

        # Should get versions 2 and 3
        assert len(retrieved_events) == 2
        versions = [e.version for e in retrieved_events]
        assert versions == [2, 3]

    def test_get_aggregate_version(self):
        """Test getting the current version of an aggregate"""
        aggregate_id = "test-resource-101"

        # Initially should be 0
        version = self.event_store.get_aggregate_version(aggregate_id)
        assert version == 0

        # Add events and check version
        for i in range(1, 4):
            event = create_event(
                aggregate_id=aggregate_id,
                aggregate_type="resource",
                event_type="updated",
                event_data={"update": i},
                version=i
            )
            self.event_store.save_event(event)

            current_version = self.event_store.get_aggregate_version(aggregate_id)
            assert current_version == i

    def test_query_events_by_type(self):
        """Test querying events by event type"""
        # Create events of different types
        event_types = ["created", "updated", "deleted", "processed"]

        for event_type in event_types:
            event = create_event(
                aggregate_id=f"test-{event_type}",
                aggregate_type="test",
                event_type=event_type,
                event_data={"action": event_type}
            )
            self.event_store.save_event(event)

        # Query for "updated" events
        event_filter = EventFilter(event_type="updated")
        results = self.event_store.query_events(event_filter)

        # Should find exactly one "updated" event
        assert len(results) == 1
        assert results[0].event_type == "updated"

    def test_query_events_by_aggregate_type(self):
        """Test querying events by aggregate type"""
        # Create events for different aggregate types
        aggregate_types = ["agent", "task", "workflow", "resource"]

        for agg_type in aggregate_types:
            event = create_event(
                aggregate_id=f"test-{agg_type}",
                aggregate_type=agg_type,
                event_type="created",
                event_data={"type": agg_type}
            )
            self.event_store.save_event(event)

        # Query for "task" aggregate type
        event_filter = EventFilter(aggregate_type="task")
        results = self.event_store.query_events(event_filter)

        # Should find exactly one task event
        assert len(results) == 1
        assert results[0].aggregate_type == "task"

    def test_query_events_by_time_range(self):
        """Test querying events by time range"""
        base_time = datetime.utcnow()

        # Create events at different times
        times = [
            base_time - timedelta(hours=2),
            base_time - timedelta(hours=1),
            base_time,
            base_time + timedelta(hours=1)
        ]

        for i, event_time in enumerate(times):
            event = create_event(
                aggregate_id=f"time-test-{i}",
                aggregate_type="test",
                event_type="timed",
                event_data={"index": i}
            )
            # Manually set timestamp
            event.timestamp = event_time
            self.event_store.save_event(event)

        # Query for events in the last hour
        from_time = base_time - timedelta(hours=1)
        to_time = base_time + timedelta(hours=2)

        event_filter = EventFilter(
            from_timestamp=from_time,
            to_timestamp=to_time
        )
        results = self.event_store.query_events(event_filter)

        # Should find events at index 1, 2, 3 (within time range)
        assert len(results) == 3
        indices = sorted([r.event_data["index"] for r in results])
        assert indices == [1, 2, 3]

    def test_query_events_pagination(self):
        """Test pagination in event queries"""
        # Create multiple events
        for i in range(10):
            event = create_event(
                aggregate_id=f"pagination-test-{i}",
                aggregate_type="test",
                event_type="paginated",
                event_data={"index": i}
            )
            self.event_store.save_event(event)

        # Query with pagination
        event_filter = EventFilter(event_type="paginated")

        # First page: limit 3, offset 0
        results_page1 = self.event_store.query_events(event_filter, limit=3, offset=0)
        assert len(results_page1) == 3

        # Second page: limit 3, offset 3
        results_page2 = self.event_store.query_events(event_filter, limit=3, offset=3)
        assert len(results_page2) == 3

        # Verify different results
        page1_indices = {r.event_data["index"] for r in results_page1}
        page2_indices = {r.event_data["index"] for r in results_page2}
        assert page1_indices.isdisjoint(page2_indices)

    def test_save_events_batch(self):
        """Test saving multiple events in batch"""
        events = []
        for i in range(5):
            event = create_event(
                aggregate_id=f"batch-test-{i}",
                aggregate_type="batch",
                event_type="batched",
                event_data={"batch_index": i}
            )
            events.append(event)

        # Save batch
        event_ids = self.event_store.save_events(events)

        # Verify all events were saved
        assert len(event_ids) == 5
        for event_id in event_ids:
            retrieved = self.event_store.get_event(event_id)
            assert retrieved is not None

    def test_save_events_batch_partial_failure(self):
        """Test batch save with some validation failures"""
        # Create mix of valid and invalid events
        events = []

        # Valid event
        valid_event = create_event(
            aggregate_id="valid-batch",
            aggregate_type="test",
            event_type="valid",
            event_data={"status": "ok"}
        )
        events.append(valid_event)

        # Invalid event (empty aggregate_id)
        invalid_event = Event(
            event_id=str(uuid.uuid4()),
            aggregate_id="",  # Invalid
            aggregate_type="test",
            event_type="invalid",
            event_data={"status": "bad"},
            timestamp=datetime.utcnow()
        )
        events.append(invalid_event)

        # Should raise exception and not save any events
        with pytest.raises(Exception):
            self.event_store.save_events(events)

        # Verify no events were saved
        for event in events:
            retrieved = self.event_store.get_event(event.event_id)
            assert retrieved is None


class TestCreateEventHelper:
    """Test cases for the create_event helper function"""

    def test_create_event_basic(self):
        """Test basic event creation"""
        event = create_event(
            aggregate_id="test-123",
            aggregate_type="agent",
            event_type="created",
            event_data={"name": "Test Agent"}
        )

        assert event.aggregate_id == "test-123"
        assert event.aggregate_type == "agent"
        assert event.event_type == "created"
        assert event.event_data == {"name": "Test Agent"}
        assert event.version == 1  # Should default to 1
        assert event.event_id is not None
        assert isinstance(event.timestamp, datetime)

    def test_create_event_with_metadata(self):
        """Test event creation with optional metadata"""
        event = create_event(
            aggregate_id="test-456",
            aggregate_type="task",
            event_type="started",
            event_data={"task": "test"},
            user_id="user-123",
            correlation_id="corr-456",
            causation_id="cause-789",
            service_name="test-service",
            metadata={"priority": "high"}
        )

        assert event.user_id == "user-123"
        assert event.correlation_id == "corr-456"
        assert event.causation_id == "cause-789"
        assert event.service_name == "test-service"
        assert event.metadata == {"priority": "high"}

    def test_create_event_auto_version(self):
        """Test that create_event automatically determines version"""
        # This test assumes the event store is empty for this aggregate
        event = create_event(
            aggregate_id="version-test",
            aggregate_type="test",
            event_type="test",
            event_data={"test": True}
        )

        # Since no events exist for this aggregate, version should be 1
        assert event.version == 1


class TestEventStoreIntegration:
    """Integration tests for EventStore with database"""

    def test_database_persistence(self):
        """Test that events persist across EventStore instances"""
        aggregate_id = "persistence-test"

        # Create and save event with first instance
        event_store1 = EventStore()
        event = create_event(
            aggregate_id=aggregate_id,
            aggregate_type="persistence",
            event_type="created",
            event_data={"persistent": True}
        )
        event_id = event_store1.save_event(event)
        event_store1.close()

        # Retrieve with second instance
        event_store2 = EventStore()
        retrieved_event = event_store2.get_event(event_id)
        event_store2.close()

        # Verify persistence
        assert retrieved_event is not None
        assert retrieved_event.event_id == event_id
        assert retrieved_event.aggregate_id == aggregate_id
        assert retrieved_event.event_data["persistent"] is True

    @patch('shared.event_store.get_db_session')
    def test_database_connection_error(self, mock_get_db):
        """Test handling of database connection errors"""
        # Mock database session to raise exception
        mock_get_db.side_effect = Exception("Database connection failed")

        event_store = EventStore()

        event = create_event(
            aggregate_id="error-test",
            aggregate_type="test",
            event_type="error",
            event_data={"error": True}
        )

        # Should handle database error gracefully
        with pytest.raises(Exception, match="Database connection failed"):
            event_store.save_event(event)


if __name__ == "__main__":
    pytest.main([__file__])