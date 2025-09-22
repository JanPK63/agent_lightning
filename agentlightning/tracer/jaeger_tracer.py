"""
Jaeger Tracer for Agent Lightning

A Jaeger-specific tracer that extends the base tracer with Jaeger defaults.
"""

from .base import BaseTracer


class JaegerTracer(BaseTracer):
    """
    Jaeger tracer using OpenTelemetry with Jaeger exporter.

    This tracer configures OpenTelemetry to export traces to Jaeger.
    """

    def __init__(self, **kwargs):
        # Initialize with base tracer - Jaeger functionality disabled due to OpenTelemetry issues
        super().__init__()
        # TODO: Re-enable Jaeger tracing when OpenTelemetry issues are resolved
        # For now, this is a placeholder that inherits from BaseTracer