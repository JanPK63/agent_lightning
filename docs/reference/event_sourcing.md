# Event Sourcing in Agent Lightning

## Overview

Event Sourcing is a powerful architectural pattern that stores the state of an application as a sequence of events. Instead of storing the current state of entities, we store the events that led to the current state. This approach provides several benefits:

- **Complete Audit Trail**: Every change to the system is recorded as an immutable event
- **Temporal Queries**: Ability to query the state of the system at any point in time
- **Event Replay**: Reconstruct application state by replaying events
- **Debugging**: Rich debugging capabilities with full event history
- **Analytics**: Powerful analytics based on event streams

## Architecture

### Core Components

#### Event Model
Events in Agent Lightning follow a structured format:

```python
class Event(BaseModel):
    event_id: str                    # Unique event identifier
    aggregate_id: str               # Entity ID (agent_id, task_id, etc.)
    aggregate_type: str             # Entity type ('agent', 'task', 'workflow', etc.)
    event_type: str                 # Event type ('created', 'updated', 'started', etc.)
    event_data: Dict[str, Any]      # Event payload with details
    timestamp: datetime             # When the event occurred
    version: int = 1                # Aggregate version for optimistic concurrency
    correlation_id: Optional[str]   # For tracking related events
    causation_id: Optional[str]     # ID of event that caused this event
    user_id: Optional[str]          # User who triggered the event
    service_name: Optional[str]     # Service that generated the event
    metadata: Dict[str, Any]        # Additional event metadata
```

#### Event Store
The `EventStore` class provides persistent storage and retrieval of events:

```python
from shared.event_store import EventStore, create_event

# Initialize event store
event_store = EventStore()

# Create and save an event
event = create_event(
    aggregate_id="agent-123",
    aggregate_type="agent",
    event_type="created",
    event_data={"name": "My Agent", "model": "gpt-4"},
    user_id="user-456"
)

event_id = event_store.save_event(event)
```

#### Event Stream
Events for a specific aggregate are organized into streams:

```python
# Get all events for an aggregate
events = event_store.get_events_by_aggregate("agent-123")

# Get event stream with current state
stream = event_store.get_event_stream("agent-123")
print(f"Current version: {stream.version}")
print(f"Total events: {len(stream.events)}")
```

## Event Types

### Agent Events
- `agent.created` - Agent was created
- `agent.updated` - Agent configuration was updated
- `agent.deployed` - Agent was deployed
- `agent.status` - Agent status changed

### Task Events
- `task.received` - Task was received by runner
- `task.started` - Task execution started
- `task.completed` - Task completed successfully
- `task.failed` - Task execution failed

### Workflow Events
- `workflow.created` - Workflow was created
- `workflow.started` - Workflow execution started
- `workflow.completed` - Workflow completed successfully
- `workflow.failed` - Workflow execution failed
- `workflow.step_completed` - Individual workflow step completed

### Rollout Events
- `rollout.started` - Rollout execution began
- `rollout.completed` - Rollout completed successfully
- `rollout.failed` - Rollout execution failed

### Specification Events
- `spec.created` - Specification was created
- `spec.updated` - Specification was updated
- `execution.started` - Spec execution started
- `execution.completed` - Spec execution completed
- `execution.failed` - Spec execution failed

## Usage Examples

### Publishing Events

```python
from shared.event_store import event_store, create_event

# Agent creation event
agent_created_event = create_event(
    aggregate_id="agent-123",
    aggregate_type="agent",
    event_type="created",
    event_data={
        "name": "Customer Support Agent",
        "model": "gpt-4",
        "capabilities": ["chat", "analysis"]
    },
    user_id="user-456",
    service_name="agent_service"
)

event_store.save_event(agent_created_event)

# Task completion event
task_completed_event = create_event(
    aggregate_id="task-789",
    aggregate_type="task",
    event_type="completed",
    event_data={
        "result": "Analysis complete",
        "execution_time": 45.2,
        "reward": 0.95
    },
    correlation_id="workflow-101",
    service_name="task_runner"
)

event_store.save_event(task_completed_event)
```

### Querying Events

```python
from agentlightning.types import EventFilter

# Find all events for a specific agent
agent_events = event_store.get_events_by_aggregate("agent-123")

# Query events by type
failed_tasks = event_store.query_events(
    EventFilter(event_type="failed", aggregate_type="task")
)

# Query events by time range
from datetime import datetime, timedelta

recent_events = event_store.query_events(
    EventFilter(
        from_timestamp=datetime.utcnow() - timedelta(hours=1),
        to_timestamp=datetime.utcnow()
    )
)

# Query with pagination
page_1 = event_store.query_events(
    EventFilter(aggregate_type="workflow"),
    limit=50,
    offset=0
)
```

### Event Replay

```python
# Basic event replay
async for event in event_store.replay_events("agent-123", from_version=5):
    print(f"Replaying: {event.event_type} at {event.timestamp}")
    # Apply event to reconstruct state
    apply_event_to_state(current_state, event)
```

### Event Replay Debugging

For advanced debugging of event replay, use the `EventReplayDebugger`:

```python
from shared.event_replay_debugger import event_replay_debugger

# Start debugging session
await event_replay_debugger.start_replay("agent-123", from_version=1)

# Add breakpoints
event_replay_debugger.add_breakpoint(event_type="agent.updated")
event_replay_debugger.add_breakpoint(
    condition="event_data.get('reward', 0) > 0.8"
)

# Add watch expressions
event_replay_debugger.add_watch("event_data.get('reward', 0)")
event_replay_debugger.add_watch("event_type")

# Control replay
await event_replay_debugger.pause_replay()
await event_replay_debugger.step_replay()
await event_replay_debugger.resume_replay()
await event_replay_debugger.stop_replay()

# Get debug information
debug_info = event_replay_debugger.get_debug_info()
timeline = event_replay_debugger.get_execution_timeline()
watched_values = event_replay_debugger.get_watched_values()

# Export debug session
session_data = event_replay_debugger.export_debug_session()
```

### Event Replay Debugger Service

The Event Replay Debugger Service provides REST API endpoints for debugging:

```bash
# Start the service
python -m services.event_replay_debugger_service

# API Endpoints:
# POST /sessions/start - Start replay session
# POST /sessions/{id}/pause - Pause replay
# POST /sessions/{id}/resume - Resume replay
# POST /sessions/{id}/step - Step to next event
# POST /sessions/{id}/stop - Stop replay
# POST /sessions/{id}/breakpoints - Add breakpoint
# DELETE /sessions/{id}/breakpoints/{bp_id} - Remove breakpoint
# POST /sessions/{id}/watches - Add watch expression
# GET /sessions/{id} - Get debug session info
# GET /sessions/{id}/timeline - Get execution timeline
# GET /sessions/{id}/export - Export debug session
```

Example API usage:

```python
import requests

# Start replay session
response = requests.post("http://localhost:8012/sessions/start", json={
    "aggregate_id": "agent-123",
    "from_version": 1
})
session_id = response.json()["session_id"]

# Add breakpoint
requests.post(f"http://localhost:8012/sessions/{session_id}/breakpoints", json={
    "event_type": "agent.updated"
})

# Get debug info
debug_info = requests.get(f"http://localhost:8012/sessions/{session_id}").json()
```

## Integration Points

### Agent Runner
The `AgentRunner` automatically publishes events for:
- Task reception (`task.received`)
- Rollout start (`rollout.started`)
- Rollout completion (`rollout.completed`)
- Rollout failure (`rollout.failed`)

### Specification Service
The `SpecService` publishes events for:
- Specification creation/updates (`spec.created`, `spec.updated`)
- Execution lifecycle (`execution.started`, `execution.completed`, `execution.failed`)

### Custom Event Publishing
You can publish custom events from your own services:

```python
# In your service
from shared.event_store import event_store, create_event

def publish_custom_event():
    event = create_event(
        aggregate_id="custom-entity-123",
        aggregate_type="custom",
        event_type="processed",
        event_data={"result": "success", "metadata": {...}},
        service_name="my_custom_service"
    )
    event_store.save_event(event)
```

## Database Schema

Events are stored in the `events` table with the following structure:

```sql
CREATE TABLE events (
    id VARCHAR(36) PRIMARY KEY,
    event_id VARCHAR(36) UNIQUE NOT NULL,
    aggregate_id VARCHAR(36) NOT NULL,
    aggregate_type VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data JSON NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    correlation_id VARCHAR(36),
    causation_id VARCHAR(36),
    user_id VARCHAR(36),
    service_name VARCHAR(50),
    event_metadata JSON DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX idx_events_aggregate_id ON events(aggregate_id);
CREATE INDEX idx_events_aggregate_type ON events(aggregate_type);
CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_correlation_id ON events(correlation_id);
```

Snapshots are stored in the `event_snapshots` table for performance:

```sql
CREATE TABLE event_snapshots (
    id VARCHAR(36) PRIMARY KEY,
    aggregate_id VARCHAR(36) NOT NULL,
    aggregate_type VARCHAR(50) NOT NULL,
    snapshot_data JSON NOT NULL,
    version INTEGER NOT NULL,
    last_event_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE
);
```

## Best Practices

### Event Design
1. **Immutable Events**: Never modify existing events
2. **Descriptive Event Types**: Use clear, descriptive event type names
3. **Rich Event Data**: Include all relevant information in event_data
4. **Correlation IDs**: Use correlation_id to track related events across services
5. **Versioning**: Use version field for optimistic concurrency

### Performance Considerations
1. **Snapshotting**: Use snapshots for aggregates with many events
2. **Indexing**: Ensure proper database indexes for query patterns
3. **Pagination**: Always use pagination for large result sets
4. **Batch Operations**: Use batch save for multiple related events

### Error Handling
1. **Graceful Degradation**: Continue operation if event publishing fails
2. **Retry Logic**: Implement retry logic for transient failures
3. **Logging**: Log event publishing failures for monitoring
4. **Validation**: Validate events before publishing

## Monitoring and Observability

### Metrics
- Total events published per minute/hour
- Events by aggregate type
- Event publishing success/failure rates
- Event store query performance

### Alerts
- Event publishing failures
- Event store database issues
- High event volume warnings
- Snapshot creation failures

## Migration and Versioning

When evolving event schemas:

1. **Backward Compatibility**: Ensure old events can still be processed
2. **Event Versioning**: Include version information in event metadata
3. **Migration Scripts**: Provide scripts to migrate existing events
4. **Documentation**: Document event schema changes

## Troubleshooting

### Common Issues

1. **Event Not Found**: Check event_id and database connectivity
2. **Version Conflicts**: Handle optimistic concurrency conflicts
3. **Performance Issues**: Check database indexes and consider snapshots
4. **Missing Events**: Verify event publishing code is being executed

### Debugging Tools

```python
# Get recent events for debugging
recent_events = event_store.query_events(
    EventFilter(),
    limit=10
)

# Check aggregate version
current_version = event_store.get_aggregate_version("aggregate-id")

# Validate event stream
stream = event_store.get_event_stream("aggregate-id")
for event in stream.events:
    print(f"Event {event.event_id}: {event.event_type} v{event.version}")
```

### Event Replay Debugger Troubleshooting

```python
# Check debugger state
debug_info = event_replay_debugger.get_debug_info()
print(f"State: {debug_info['state']}")
print(f"Current Event: {debug_info['current_event']}")

# Inspect breakpoints
breakpoints = event_replay_debugger.get_breakpoints()
for bp in breakpoints:
    print(f"Breakpoint {bp['id']}: {bp['event_type']} - {bp['condition']}")

# Check watched values
watched = event_replay_debugger.get_watched_values()
for expr, value in watched.items():
    print(f"{expr} = {value}")

# Export session for analysis
session_export = event_replay_debugger.export_debug_session()
```

## Future Enhancements

- **Event-Driven Architecture**: Expand to full event-driven communication
- **Event Streaming**: Integration with Kafka/Redis streams
- **Advanced Analytics**: Real-time event processing and analytics
- **Event Schema Registry**: Centralized event schema management
- **Cross-Service Events**: Enhanced correlation across microservices
- **Visual Event Replay Debugger**: Web-based UI for event replay debugging
- **Distributed Event Replay**: Replay events across multiple services
- **Event Replay Performance**: Optimize replay for large event streams