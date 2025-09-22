#!/usr/bin/env python3
"""
Event Replay Debugger for Agent Lightning
Provides debugging capabilities for event sourcing replay functionality
"""

import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from enum import Enum

from .event_store import EventStore, Event


class DebugState(Enum):
    """States for event replay debugger"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class EventBreakpoint:
    """Represents a debugging breakpoint for events"""
    def __init__(self, breakpoint_id: str, event_type: Optional[str] = None,
                 condition: Optional[str] = None, enabled: bool = True):
        self.breakpoint_id = breakpoint_id
        self.event_type = event_type
        self.condition = condition
        self.enabled = enabled

    def should_break(self, event: Event) -> bool:
        """Check if this breakpoint should trigger on the given event"""
        if not self.enabled:
            return False

        # Check event type
        if self.event_type and event.event_type != self.event_type:
            return False

        # Check condition (simple evaluation)
        if self.condition:
            try:
                # Create a safe evaluation context
                context = {
                    'event': event,
                    'event_data': event.event_data,
                    'event_type': event.event_type,
                    'aggregate_id': event.aggregate_id,
                    'version': event.version,
                    'timestamp': event.timestamp
                }
                return eval(self.condition, {"__builtins__": {}}, context)
            except Exception:
                return False

        return True


class EventReplayDebugger:
    """Debugger for event replay with breakpoints, stepping, and inspection"""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.breakpoints: List[EventBreakpoint] = []
        self.current_event: Optional[Event] = None
        self.state = DebugState.IDLE
        self.execution_timeline: List[Dict[str, Any]] = []
        self.variable_watches: List[str] = []
        self.aggregate_id: Optional[str] = None
        self.from_version: int = 1
        self.replay_generator: Optional[AsyncGenerator[Event, None]] = None
        self.last_error: Optional[Dict[str, Any]] = None

    def add_breakpoint(self, event_type: Optional[str] = None,
                       condition: Optional[str] = None) -> str:
        """Add a breakpoint for event replay"""
        breakpoint_id = f"bp_{len(self.breakpoints)}"
        breakpoint_obj = EventBreakpoint(breakpoint_id, event_type, condition)
        self.breakpoints.append(breakpoint_obj)
        return breakpoint_id

    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint"""
        for i, bp in enumerate(self.breakpoints):
            if bp.breakpoint_id == breakpoint_id:
                self.breakpoints.pop(i)
                return True
        return False

    def get_breakpoints(self) -> List[Dict[str, Any]]:
        """Get all breakpoints"""
        return [
            {
                'id': bp.breakpoint_id,
                'event_type': bp.event_type,
                'condition': bp.condition,
                'enabled': bp.enabled
            }
            for bp in self.breakpoints
        ]

    def add_watch(self, expression: str):
        """Add a variable watch expression"""
        if expression not in self.variable_watches:
            self.variable_watches.append(expression)

    def remove_watch(self, expression: str):
        """Remove a variable watch expression"""
        if expression in self.variable_watches:
            self.variable_watches.remove(expression)

    async def start_replay(self, aggregate_id: str, from_version: int = 1):
        """Start event replay with debugging"""
        if self.state != DebugState.IDLE:
            raise ValueError("Debugger is not in idle state")

        self.aggregate_id = aggregate_id
        self.from_version = from_version
        self.state = DebugState.RUNNING
        self.execution_timeline = []
        self.current_event = None
        self.last_error = None

        try:
            self.replay_generator = self.event_store.replay_events(
                aggregate_id, from_version)
            await self._step_to_next_event()
        except Exception as e:
            self.state = DebugState.ERROR
            self.last_error = {'message': str(e), 'type': type(e).__name__}

    async def pause_replay(self):
        """Pause event replay"""
        if self.state == DebugState.RUNNING:
            self.state = DebugState.PAUSED

    async def resume_replay(self):
        """Resume event replay"""
        if self.state == DebugState.PAUSED:
            self.state = DebugState.RUNNING
            await self._step_to_next_event()

    async def stop_replay(self):
        """Stop event replay"""
        self.state = DebugState.STOPPED
        self.replay_generator = None
        self.current_event = None

    async def step_replay(self):
        """Step to next event in replay"""
        if self.state in [DebugState.PAUSED, DebugState.IDLE]:
            self.state = DebugState.RUNNING
            await self._step_to_next_event()

    async def _step_to_next_event(self):
        """Internal method to step to next event"""
        if not self.replay_generator:
            return

        try:
            event = await self.replay_generator.__anext__()

            # Check breakpoints
            should_break = False
            for bp in self.breakpoints:
                if bp.should_break(event):
                    should_break = True
                    break

            # Record in timeline
            timeline_entry = {
                'event': event,
                'timestamp': datetime.utcnow(),
                'breakpoint_hit': should_break
            }
            self.execution_timeline.append(timeline_entry)

            self.current_event = event

            if should_break:
                self.state = DebugState.PAUSED
            elif self.state == DebugState.RUNNING:
                # Auto-continue if no breakpoint
                asyncio.create_task(self._step_to_next_event())

        except StopAsyncIteration:
            # End of replay
            self.state = DebugState.STOPPED
            self.replay_generator = None
        except Exception as e:
            self.state = DebugState.ERROR
            self.last_error = {'message': str(e), 'type': type(e).__name__}
            self.replay_generator = None

    def get_current_event(self) -> Optional[Event]:
        """Get the current event being replayed"""
        return self.current_event

    def get_execution_timeline(self) -> List[Dict[str, Any]]:
        """Get the execution timeline"""
        return self.execution_timeline

    def get_watched_values(self) -> Dict[str, Any]:
        """Get values for watched expressions"""
        if not self.current_event:
            return {}

        results = {}
        context = {
            'event': self.current_event,
            'event_data': self.current_event.event_data,
            'event_type': self.current_event.event_type,
            'aggregate_id': self.current_event.aggregate_id,
            'version': self.current_event.version,
            'timestamp': self.current_event.timestamp,
            'metadata': self.current_event.metadata
        }

        for expr in self.variable_watches:
            try:
                result = eval(expr, {"__builtins__": {}}, context)
                results[expr] = result
            except Exception as e:
                results[expr] = f"Error: {e}"

        return results

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        return {
            'state': self.state.value,
            'aggregate_id': self.aggregate_id,
            'from_version': self.from_version,
            'current_event': (self.current_event.dict()
                              if self.current_event else None),
            'breakpoints': self.get_breakpoints(),
            'timeline_length': len(self.execution_timeline),
            'watched_values': self.get_watched_values(),
            'last_error': self.last_error
        }

    def export_debug_session(self) -> Dict[str, Any]:
        """Export complete debug session data"""
        return {
            'aggregate_id': self.aggregate_id,
            'from_version': self.from_version,
            'breakpoints': self.get_breakpoints(),
            'execution_timeline': [
                {
                    'event_id': entry['event'].event_id,
                    'event_type': entry['event'].event_type,
                    'version': entry['event'].version,
                    'timestamp': entry['event'].timestamp.isoformat(),
                    'breakpoint_hit': entry['breakpoint_hit']
                }
                for entry in self.execution_timeline
            ],
            'variable_watches': self.variable_watches,
            'final_state': self.state.value,
            'last_error': self.last_error,
            'exported_at': datetime.utcnow().isoformat()
        }


# Global debugger instance
event_replay_debugger = EventReplayDebugger(EventStore())