#!/usr/bin/env python3
"""
Event Store for Agent Lightning - Event Sourcing Implementation
Provides persistent storage and retrieval of events for audit and replay capabilities
"""

import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import Session

from .database import get_db_session
from .models import Event as EventModel, EventSnapshot as EventSnapshotModel
from agentlightning.types import Event, EventStream, EventFilter


class EventStore:
    """Event store for persisting and retrieving events using event sourcing patterns"""

    def __init__(self):
        """Initialize the event store"""
        self.db_session = get_db_session()

    def save_event(self, event: Event) -> str:
        """
        Save an event to the persistent store

        Args:
            event: The event to save

        Returns:
            str: The event ID

        Raises:
            ValueError: If event validation fails
        """
        # Validate event
        self._validate_event(event)

        # Convert to database model
        db_event = EventModel(
            event_id=event.event_id,
            aggregate_id=event.aggregate_id,
            aggregate_type=event.aggregate_type,
            event_type=event.event_type,
            event_data=event.event_data,
            timestamp=event.timestamp,
            version=event.version,
            correlation_id=event.correlation_id,
            causation_id=event.causation_id,
            user_id=event.user_id,
            service_name=event.service_name,
            event_metadata=event.metadata
        )

        try:
            self.db_session.add(db_event)
            self.db_session.commit()
            return event.event_id
        except Exception as e:
            self.db_session.rollback()
            raise Exception(f"Failed to save event {event.event_id}: {e}")

    def save_events(self, events: List[Event]) -> List[str]:
        """
        Save multiple events atomically

        Args:
            events: List of events to save

        Returns:
            List[str]: List of saved event IDs
        """
        if not events:
            return []

        # Validate all events
        for event in events:
            self._validate_event(event)

        # Convert to database models
        db_events = []
        for event in events:
            db_event = EventModel(
                event_id=event.event_id,
                aggregate_id=event.aggregate_id,
                aggregate_type=event.aggregate_type,
                event_type=event.event_type,
                event_data=event.event_data,
                timestamp=event.timestamp,
                version=event.version,
                correlation_id=event.correlation_id,
                causation_id=event.causation_id,
                user_id=event.user_id,
                service_name=event.service_name,
                event_metadata=event.metadata
            )
            db_events.append(db_event)

        try:
            self.db_session.add_all(db_events)
            self.db_session.commit()
            return [event.event_id for event in events]
        except Exception as e:
            self.db_session.rollback()
            raise Exception(f"Failed to save events: {e}")

    def get_event(self, event_id: str) -> Optional[Event]:
        """
        Retrieve a single event by ID

        Args:
            event_id: The event ID to retrieve

        Returns:
            Optional[Event]: The event if found, None otherwise
        """
        try:
            db_event = self.db_session.query(EventModel).filter_by(event_id=event_id).first()
            if db_event:
                return self._db_event_to_event(db_event)
            return None
        except Exception as e:
            raise Exception(f"Failed to retrieve event {event_id}: {e}")

    def get_events_by_aggregate(self, aggregate_id: str, from_version: int = 1) -> List[Event]:
        """
        Get all events for a specific aggregate from a given version

        Args:
            aggregate_id: The aggregate ID
            from_version: Starting version (default: 1)

        Returns:
            List[Event]: List of events for the aggregate
        """
        try:
            db_events = (
                self.db_session.query(EventModel)
                .filter(
                    and_(
                        EventModel.aggregate_id == aggregate_id,
                        EventModel.version >= from_version
                    )
                )
                .order_by(EventModel.version)
                .all()
            )

            return [self._db_event_to_event(db_event) for db_event in db_events]
        except Exception as e:
            raise Exception(f"Failed to retrieve events for aggregate {aggregate_id}: {e}")

    def get_event_stream(self, aggregate_id: str) -> EventStream:
        """
        Get the complete event stream for an aggregate

        Args:
            aggregate_id: The aggregate ID

        Returns:
            EventStream: The event stream
        """
        try:
            # Get aggregate type from first event
            first_event = (
                self.db_session.query(EventModel)
                .filter_by(aggregate_id=aggregate_id)
                .order_by(EventModel.version)
                .first()
            )

            if not first_event:
                raise ValueError(f"No events found for aggregate {aggregate_id}")

            aggregate_type = first_event.aggregate_type

            # Get all events
            events = self.get_events_by_aggregate(aggregate_id)

            # Get latest snapshot if available
            snapshot = self._get_latest_snapshot(aggregate_id)

            stream = EventStream(
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                events=events,
                version=len(events),
                snapshot=snapshot
            )

            return stream
        except Exception as e:
            raise Exception(f"Failed to get event stream for {aggregate_id}: {e}")

    def query_events(self, event_filter: EventFilter, limit: int = 100, offset: int = 0) -> List[Event]:
        """
        Query events based on filter criteria

        Args:
            event_filter: Filter criteria
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List[Event]: Filtered list of events
        """
        try:
            query = self.db_session.query(EventModel)

            # Apply filters
            if event_filter.aggregate_id:
                query = query.filter(EventModel.aggregate_id == event_filter.aggregate_id)

            if event_filter.aggregate_type:
                query = query.filter(EventModel.aggregate_type == event_filter.aggregate_type)

            if event_filter.event_type:
                query = query.filter(EventModel.event_type == event_filter.event_type)

            if event_filter.correlation_id:
                query = query.filter(EventModel.correlation_id == event_filter.correlation_id)

            if event_filter.user_id:
                query = query.filter(EventModel.user_id == event_filter.user_id)

            if event_filter.service_name:
                query = query.filter(EventModel.service_name == event_filter.service_name)

            if event_filter.from_timestamp:
                query = query.filter(EventModel.timestamp >= event_filter.from_timestamp)

            if event_filter.to_timestamp:
                query = query.filter(EventModel.timestamp <= event_filter.to_timestamp)

            # Apply metadata filters
            for key, value in event_filter.metadata_filters.items():
                # Note: This is a simple implementation. In production, you might want
                # to use JSON operators specific to your database
                query = query.filter(EventModel.event_metadata.contains({key: value}))

            # Order by timestamp descending (most recent first)
            query = query.order_by(desc(EventModel.timestamp))

            # Apply pagination
            query = query.limit(limit).offset(offset)

            db_events = query.all()
            return [self._db_event_to_event(db_event) for db_event in db_events]

        except Exception as e:
            raise Exception(f"Failed to query events: {e}")

    def get_aggregate_version(self, aggregate_id: str) -> int:
        """
        Get the current version of an aggregate

        Args:
            aggregate_id: The aggregate ID

        Returns:
            int: Current version (0 if no events exist)
        """
        try:
            result = (
                self.db_session.query(func.max(EventModel.version))
                .filter(EventModel.aggregate_id == aggregate_id)
                .scalar()
            )
            return result or 0
        except Exception as e:
            raise Exception(f"Failed to get version for aggregate {aggregate_id}: {e}")

    def save_snapshot(self, aggregate_id: str, snapshot_data: Dict[str, Any], version: int) -> None:
        """
        Save a snapshot of aggregate state

        Args:
            aggregate_id: The aggregate ID
            snapshot_data: The snapshot data
            version: The version at which the snapshot was taken
        """
        try:
            # Get the last event ID for this version
            last_event = (
                self.db_session.query(EventModel)
                .filter(
                    and_(
                        EventModel.aggregate_id == aggregate_id,
                        EventModel.version == version
                    )
                )
                .first()
            )

            if not last_event:
                raise ValueError(f"No event found for aggregate {aggregate_id} at version {version}")

            # Create snapshot
            snapshot = EventSnapshotModel(
                aggregate_id=aggregate_id,
                aggregate_type=last_event.aggregate_type,
                snapshot_data=snapshot_data,
                version=version,
                last_event_id=last_event.event_id
            )

            self.db_session.add(snapshot)
            self.db_session.commit()

        except Exception as e:
            self.db_session.rollback()
            raise Exception(f"Failed to save snapshot for {aggregate_id}: {e}")

    def replay_events(self, aggregate_id: str, from_version: int = 1) -> AsyncGenerator[Event, None]:
        """
        Replay events for an aggregate as an async generator

        Args:
            aggregate_id: The aggregate ID
            from_version: Starting version

        Yields:
            Event: Events in chronological order
        """
        try:
            db_events = (
                self.db_session.query(EventModel)
                .filter(
                    and_(
                        EventModel.aggregate_id == aggregate_id,
                        EventModel.version >= from_version
                    )
                )
                .order_by(EventModel.version)
                .yield_per(100)  # Process in batches
            )

            for db_event in db_events:
                yield self._db_event_to_event(db_event)

        except Exception as e:
            raise Exception(f"Failed to replay events for {aggregate_id}: {e}")

    def _validate_event(self, event: Event) -> None:
        """Validate an event before saving"""
        if not event.event_id:
            raise ValueError("Event must have an event_id")
        if not event.aggregate_id:
            raise ValueError("Event must have an aggregate_id")
        if not event.aggregate_type:
            raise ValueError("Event must have an aggregate_type")
        if not event.event_type:
            raise ValueError("Event must have an event_type")
        if event.version < 1:
            raise ValueError("Event version must be >= 1")

    def _db_event_to_event(self, db_event: EventModel) -> Event:
        """Convert database event model to domain event"""
        return Event(
            event_id=db_event.event_id,
            aggregate_id=db_event.aggregate_id,
            aggregate_type=db_event.aggregate_type,
            event_type=db_event.event_type,
            event_data=db_event.event_data,
            timestamp=db_event.timestamp,
            version=db_event.version,
            correlation_id=db_event.correlation_id,
            causation_id=db_event.causation_id,
            user_id=db_event.user_id,
            service_name=db_event.service_name,
            metadata=db_event.event_metadata
        )

    def _get_latest_snapshot(self, aggregate_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest snapshot for an aggregate"""
        try:
            snapshot = (
                self.db_session.query(EventSnapshotModel)
                .filter_by(aggregate_id=aggregate_id)
                .order_by(desc(EventSnapshotModel.version))
                .first()
            )
            return snapshot.snapshot_data if snapshot else None
        except Exception:
            return None

    def close(self):
        """Close the database session"""
        if self.db_session:
            self.db_session.close()


# Global event store instance
event_store = EventStore()


def create_event(
    aggregate_id: str,
    aggregate_type: str,
    event_type: str,
    event_data: Dict[str, Any],
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    service_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Helper function to create a new event

    Args:
        aggregate_id: The aggregate ID
        aggregate_type: The aggregate type
        event_type: The event type
        event_data: The event data
        user_id: Optional user ID
        correlation_id: Optional correlation ID
        causation_id: Optional causation ID
        service_name: Optional service name
        metadata: Optional metadata

    Returns:
        Event: The created event
    """
    # Get next version for the aggregate
    current_version = event_store.get_aggregate_version(aggregate_id)
    next_version = current_version + 1

    return Event(
        event_id=str(uuid.uuid4()),
        aggregate_id=aggregate_id,
        aggregate_type=aggregate_type,
        event_type=event_type,
        event_data=event_data,
        timestamp=datetime.utcnow(),
        version=next_version,
        correlation_id=correlation_id,
        causation_id=causation_id,
        user_id=user_id,
        service_name=service_name,
        metadata=metadata or {}
    )