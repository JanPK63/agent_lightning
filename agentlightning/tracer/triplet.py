#!/usr/bin/env python3
"""
Minimal fallback Triplet exporter stubs for tests when OpenTelemetry
is unavailable.
"""

from typing import List, Optional

try:
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import ReadableSpan
except ImportError:
    class ReadableSpan:
        def __init__(self, *args, **kwargs):
            pass

        def to_json(self):
            return {}

    class _TraceAPIShim:
        @staticmethod
        def format_span_id(x):
            return str(x)

    trace_api = _TraceAPIShim()

    class SpanContext:
        def __init__(self, trace_id, span_id, is_remote):
            self.trace_id = trace_id
            self.span_id = span_id
            self.is_remote = is_remote

    class SpanKind:
        INTERNAL = 2

from agentlightning.types import Triplet


class TripletExporter:
    def __init__(self,
                 repair_hierarchy: bool = True,
                 llm_call_match: str = r"openai\.chat\.completion",
                 agent_match: Optional[str] = None,
                 exclude_llm_call_in_reward: bool = True,
                 reward_match: object = None):
        self.repair_hierarchy = repair_hierarchy
        self.llm_call_match = llm_call_match
        self.agent_match = agent_match
        self.exclude_llm_call_in_reward = exclude_llm_call_in_reward
        self.reward_match = reward_match

    def export(self, spans: List[ReadableSpan]) -> List[Triplet]:
        # Minimal stub: return an empty trajectory to keep tests lightweight
        return []


class TraceTree:
    """Minimal placeholder for TraceTree used by TripletExporter tests."""

    @classmethod
    def from_spans(cls, spans: List[ReadableSpan]):
        return cls()

    def repair_hierarchy(self):
        pass

    def to_trajectory(self, **kwargs) -> List[Triplet]:
        return []
