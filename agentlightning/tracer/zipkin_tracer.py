"""
Zipkin Tracer for Agent Lightning

A Zipkin-specific tracer that extends the base tracer with Zipkin defaults.
"""

from .base import BaseTracer


class ZipkinTracer(BaseTracer):
    """
    Zipkin tracer using OpenTelemetry with Zipkin exporter.

    This tracer configures OpenTelemetry to export traces to Zipkin.
    """

    def __init__(self, **kwargs):
        # Initialize with base tracer - Zipkin functionality disabled due to OpenTelemetry issues
        super().__init__()
        # TODO: Re-enable Zipkin tracing when OpenTelemetry issues are resolved
        # For now, this is a placeholder that inherits from BaseTracer