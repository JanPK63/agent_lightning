"""
Tracer Factory for Agent Lightning

Provides a unified interface for creating and configuring different tracing backends
including OpenTelemetry, Jaeger, Zipkin, and AgentOps.
"""

import logging
import os
from typing import Dict, Any, Optional, Type
from enum import Enum

from .base import BaseTracer
from .agentops import AgentOpsTracer
from .noop_tracer import NoOpTracer
from .jaeger_tracer import JaegerTracer
from .zipkin_tracer import ZipkinTracer

logger = logging.getLogger(__name__)


class TracerType(Enum):
    """Supported tracer types"""
    AGENTOPS = "agentops"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    NOOP = "noop"


class TracerConfig:
    """Configuration for tracer initialization"""

    def __init__(
        self,
        tracer_type: TracerType = TracerType.AGENTOPS,
        service_name: str = "agent-lightning",
        service_version: str = "1.0.0",
        environment: str = "development",
        **kwargs
    ):
        self.tracer_type = tracer_type
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.kwargs = kwargs

    @classmethod
    def from_env(cls) -> 'TracerConfig':
        """Create configuration from environment variables"""
        tracer_type_str = os.getenv("TRACER_TYPE", "agentops").lower()
        try:
            tracer_type = TracerType(tracer_type_str)
        except ValueError:
            logger.warning(f"Unknown tracer type '{tracer_type_str}', defaulting to agentops")
            tracer_type = TracerType.AGENTOPS

        return cls(
            tracer_type=tracer_type,
            service_name=os.getenv("TRACER_SERVICE_NAME", "agent-lightning"),
            service_version=os.getenv("TRACER_SERVICE_VERSION", "1.0.0"),
            environment=os.getenv("TRACER_ENVIRONMENT", "development"),
            # Tracer-specific configuration
            agentops_managed=os.getenv("AGENTOPS_MANAGED", "true").lower() == "true",
            instrument_managed=os.getenv("TRACER_INSTRUMENT_MANAGED", "true").lower() == "true",
            daemon=os.getenv("AGENTOPS_DAEMON", "true").lower() == "true",
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
            zipkin_endpoint=os.getenv("ZIPKIN_ENDPOINT", "http://localhost:9411/api/v2/spans"),
            otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
            sampling_rate=float(os.getenv("TRACER_SAMPLING_RATE", "1.0")),
        )


class TracerFactory:
    """Factory for creating tracer instances"""

    _tracer_classes: Dict[TracerType, Type[BaseTracer]] = {
        TracerType.AGENTOPS: AgentOpsTracer,
        TracerType.JAEGER: JaegerTracer,
        TracerType.ZIPKIN: ZipkinTracer,
        TracerType.NOOP: NoOpTracer,
    }

    @classmethod
    def register_tracer(cls, tracer_type: TracerType, tracer_class: Type[BaseTracer]):
        """Register a new tracer implementation"""
        cls._tracer_classes[tracer_type] = tracer_class
        logger.info(f"Registered tracer: {tracer_type.value}")

    @classmethod
    def create_tracer(cls, config: Optional[TracerConfig] = None) -> BaseTracer:
        """Create a tracer instance based on configuration"""
        if config is None:
            config = TracerConfig.from_env()

        tracer_class = cls._tracer_classes.get(config.tracer_type)

        if tracer_class is None:
            logger.warning(f"Tracer type {config.tracer_type.value} not implemented, using NoOpTracer")
            tracer_class = NoOpTracer

        try:
            # Create tracer with configuration-specific parameters
            if config.tracer_type == TracerType.AGENTOPS:
                tracer = tracer_class(
                    agentops_managed=config.kwargs.get('agentops_managed', True),
                    instrument_managed=config.kwargs.get('instrument_managed', True),
                    daemon=config.kwargs.get('daemon', True)
                )
            
            elif config.tracer_type == TracerType.JAEGER:
                tracer = tracer_class(
                    service_name=config.service_name,
                    service_version=config.service_version,
                    environment=config.environment,
                    endpoint=config.kwargs.get('jaeger_endpoint', 'http://localhost:14268/api/traces'),
                    sampling_rate=config.kwargs.get('sampling_rate', 1.0),
                    instrument_managed=config.kwargs.get('instrument_managed', True)
                )
            elif config.tracer_type == TracerType.ZIPKIN:
                tracer = tracer_class(
                    service_name=config.service_name,
                    service_version=config.service_version,
                    environment=config.environment,
                    endpoint=config.kwargs.get('zipkin_endpoint', 'http://localhost:9411/api/v2/spans'),
                    sampling_rate=config.kwargs.get('sampling_rate', 1.0),
                    instrument_managed=config.kwargs.get('instrument_managed', True)
                )
            elif config.tracer_type == TracerType.NOOP:
                tracer = tracer_class()
            else:
                # For other tracers, pass all kwargs
                tracer = tracer_class(**config.kwargs)

            logger.info(f"Created tracer: {config.tracer_type.value}")
            return tracer

        except Exception as e:
            logger.error(f"Failed to create tracer {config.tracer_type.value}: {e}")
            logger.info("Falling back to NoOpTracer")
            return NoOpTracer()

    @classmethod
    def get_available_tracers(cls) -> Dict[str, str]:
        """Get information about available tracers"""
        return {
            tracer_type.value: tracer_class.__name__
            for tracer_type, tracer_class in cls._tracer_classes.items()
        }


# Convenience functions
def create_tracer(config: Optional[TracerConfig] = None) -> BaseTracer:
    """Create a tracer instance (convenience function)"""
    return TracerFactory.create_tracer(config)


def get_tracer_from_env() -> BaseTracer:
    """Create a tracer instance from environment configuration"""
    config = TracerConfig.from_env()
    return create_tracer(config)