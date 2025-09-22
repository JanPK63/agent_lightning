"""
OpenTelemetry Observability Setup for Agent Lightning
Implements comprehensive monitoring, tracing, and metrics collection
Following Agent Lightning's approach to agent observability
"""

from opentelemetry import trace, baggage
from opentelemetry.metrics import set_meter_provider, get_meter_provider, get_meter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.metrics import MeterProvider, Counter, Histogram, UpDownCounter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode
from prometheus_client import start_http_server
import logging
import time
import errno
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from functools import wraps
import json
from dataclasses import dataclass, asdict


@dataclass
class AgentSpan:
    """Represents a span in agent execution trace"""
    name: str
    agent_id: str
    operation: str
    attributes: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    error: Optional[str] = None


class AgentLightningObservability:
    """
    Comprehensive observability for Agent Lightning
    Provides tracing, metrics, and logging for agent execution
    """
    
    def __init__(self,
                 service_name: str = "agent-lightning",
                 service_version: str = "1.0.0",
                 otlp_endpoint: str = "http://localhost:4317",
                 prometheus_port: int = 8000,
                 enable_console_export: bool = True):
        """
        Initialize observability
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            otlp_endpoint: OTLP collector endpoint
            prometheus_port: Port for Prometheus metrics
            enable_console_export: Enable console span export for debugging
        """
        self.service_name = service_name
        self.service_version = service_version
        
        # Create resource with error handling for threading issues
        try:
            self.resource = Resource.create({
                SERVICE_NAME: service_name,
                SERVICE_VERSION: service_version,
                "deployment.environment": "development",
                "framework": "agent-lightning",
                "telemetry.sdk.language": "python"
            })
        except Exception as e:
            # Fallback to simple resource if threading issues occur (common in Streamlit)
            print(f"Warning: Using fallback resource due to: {e}")
            self.resource = Resource({
                SERVICE_NAME: service_name,
                SERVICE_VERSION: service_version,
                "deployment.environment": "development"
            })
        
        # Setup tracing
        self._setup_tracing(otlp_endpoint, enable_console_export)
        
        # Setup metrics
        self._setup_metrics(prometheus_port)
        
        # Setup logging
        self._setup_logging()
        
        # Instrument HTTP requests
        RequestsInstrumentor().instrument()
        
        # Set propagator for distributed tracing
        set_global_textmap(B3MultiFormat())
        
        # Agent-specific metrics
        self._init_agent_metrics()
        
        print(f"📊 Observability initialized for {service_name}")
        print(f"   OTLP endpoint: {otlp_endpoint}")
        print(f"   Prometheus port: {prometheus_port}")
    
    def _setup_tracing(self, otlp_endpoint: str, enable_console: bool):
        """Setup distributed tracing"""
        # Create tracer provider
        provider = TracerProvider(resource=self.resource)
        
        # Add OTLP exporter
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True
            )
            provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Add console exporter for debugging
        if enable_console:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(
            self.service_name,
            self.service_version
        )
    
    def _setup_metrics(self, prometheus_port: int):
        """Setup metrics collection"""
        # Create metric readers
        readers = []
        
        # Prometheus reader
        if prometheus_port:
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)
            # Start Prometheus HTTP server
            try:
                start_http_server(prometheus_port)
                print(f"✅ Prometheus metrics server started on port {prometheus_port}")
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    print(f"⚠️ Prometheus port {prometheus_port} already in use, skipping metrics server startup")
                else:
                    raise
        
        # Create meter provider
        provider = MeterProvider(
            resource=self.resource,
            metric_readers=readers
        )
        
        # Set global meter provider
        set_meter_provider(provider)

        # Get meter
        self.meter = get_meter(
            self.service_name,
            self.service_version
        )
    
    def _setup_logging(self):
        """Setup structured logging"""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Instrument logging
        LoggingInstrumentor().instrument(set_logging_format=True)
        
        self.logger = logging.getLogger(self.service_name)
    
    def _init_agent_metrics(self):
        """Initialize agent-specific metrics"""
        # Counters
        self.agent_requests_counter = self.meter.create_counter(
            name="agent_requests_total",
            description="Total number of agent requests",
            unit="1"
        )
        
        self.agent_errors_counter = self.meter.create_counter(
            name="agent_errors_total",
            description="Total number of agent errors",
            unit="1"
        )
        
        self.transitions_counter = self.meter.create_counter(
            name="transitions_total",
            description="Total number of MDP transitions",
            unit="1"
        )
        
        # Histograms
        self.agent_latency_histogram = self.meter.create_histogram(
            name="agent_latency_seconds",
            description="Agent execution latency",
            unit="s"
        )
        
        self.reward_histogram = self.meter.create_histogram(
            name="agent_reward",
            description="Agent reward distribution",
            unit="1"
        )
        
        self.memory_usage_histogram = self.meter.create_histogram(
            name="memory_usage_bytes",
            description="Memory usage in bytes",
            unit="By"
        )
        
        # UpDown Counters (gauges)
        self.active_agents_gauge = self.meter.create_up_down_counter(
            name="active_agents",
            description="Number of active agents",
            unit="1"
        )
        
        self.memory_entries_gauge = self.meter.create_up_down_counter(
            name="memory_entries",
            description="Number of memory entries",
            unit="1"
        )
    
    @contextmanager
    def trace_agent_execution(self, agent_id: str, task_type: str):
        """
        Context manager for tracing agent execution
        
        Args:
            agent_id: ID of the agent
            task_type: Type of task being executed
        """
        # Start span
        with self.tracer.start_as_current_span(
            f"agent_execution_{agent_id}",
            attributes={
                "agent.id": agent_id,
                "task.type": task_type,
                "framework": "agent-lightning"
            }
        ) as span:
            start_time = time.time()
            
            # Increment counter
            self.agent_requests_counter.add(
                1,
                {"agent_id": agent_id, "task_type": task_type}
            )
            
            # Increment active agents
            self.active_agents_gauge.add(1)
            
            try:
                yield span
                
                # Record success
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                # Record error
                span.set_status(
                    Status(StatusCode.ERROR, str(e))
                )
                span.record_exception(e)
                
                # Increment error counter
                self.agent_errors_counter.add(
                    1,
                    {"agent_id": agent_id, "error_type": type(e).__name__}
                )
                
                # Log error
                self.logger.error(
                    f"Agent {agent_id} execution failed: {e}",
                    extra={"agent_id": agent_id, "task_type": task_type}
                )
                
                raise
            
            finally:
                # Record latency
                latency = time.time() - start_time
                self.agent_latency_histogram.record(
                    latency,
                    {"agent_id": agent_id, "task_type": task_type}
                )
                
                # Decrement active agents
                self.active_agents_gauge.add(-1)
                
                # Log completion
                self.logger.info(
                    f"Agent {agent_id} completed in {latency:.2f}s",
                    extra={
                        "agent_id": agent_id,
                        "task_type": task_type,
                        "latency": latency
                    }
                )
    
    def trace_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int):
        """
        Trace LLM API call
        
        Args:
            model: LLM model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        with self.tracer.start_as_current_span(
            f"llm_call_{model}",
            attributes={
                "llm.model": model,
                "llm.prompt_tokens": prompt_tokens,
                "llm.completion_tokens": completion_tokens,
                "llm.total_tokens": prompt_tokens + completion_tokens
            }
        ) as span:
            return span
    
    def trace_memory_operation(self, operation: str, memory_type: str, size: int):
        """
        Trace memory operation
        
        Args:
            operation: Type of operation (store, retrieve, consolidate)
            memory_type: Type of memory (episodic, semantic, working)
            size: Size of data
        """
        with self.tracer.start_as_current_span(
            f"memory_{operation}",
            attributes={
                "memory.operation": operation,
                "memory.type": memory_type,
                "memory.size": size
            }
        ) as span:
            # Update memory metrics
            if operation == "store":
                self.memory_entries_gauge.add(1, {"type": memory_type})
            elif operation == "consolidate":
                self.memory_entries_gauge.add(-size, {"type": memory_type})
            
            return span
    
    def record_transition(self, transition: Dict[str, Any]):
        """
        Record MDP transition metrics
        
        Args:
            transition: Transition data
        """
        # Extract metrics
        reward = transition.get("reward", 0)
        agent_id = transition.get("info", {}).get("agent_id", "unknown")
        hierarchy_level = transition.get("info", {}).get("hierarchy_level", "low")
        
        # Record transition
        self.transitions_counter.add(
            1,
            {
                "agent_id": agent_id,
                "hierarchy_level": hierarchy_level
            }
        )
        
        # Record reward
        self.reward_histogram.record(
            reward,
            {
                "agent_id": agent_id,
                "hierarchy_level": hierarchy_level
            }
        )
        
        # Create span for transition
        with self.tracer.start_as_current_span(
            "mdp_transition",
            attributes={
                "transition.reward": reward,
                "transition.agent_id": agent_id,
                "transition.hierarchy_level": hierarchy_level,
                "transition.done": transition.get("done", False)
            }
        ) as span:
            span.add_event(
                "transition_recorded",
                attributes={"reward": reward}
            )
    
    def record_training_metrics(self, metrics: Dict[str, float]):
        """
        Record training metrics
        
        Args:
            metrics: Dictionary of training metrics
        """
        with self.tracer.start_as_current_span("training_update") as span:
            for metric_name, value in metrics.items():
                # Add as span attribute
                span.set_attribute(f"training.{metric_name}", value)
                
                # Log metric
                self.logger.info(
                    f"Training metric - {metric_name}: {value}",
                    extra={"metric": metric_name, "value": value}
                )
    
    def create_span_decorator(self, span_name: str):
        """
        Create a decorator for tracing functions
        
        Args:
            span_name: Name of the span
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    span_name,
                    attributes={
                        "function": func.__name__,
                        "module": func.__module__
                    }
                ):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class TraceCollector:
    """
    Collects and aggregates traces for Agent Lightning
    Used for creating training data from execution traces
    """
    
    def __init__(self):
        self.traces: List[AgentSpan] = []
        self.current_trace_id = None
        self.trace_metadata = {}
    
    def start_trace(self, trace_id: str, metadata: Dict[str, Any]):
        """Start a new trace collection"""
        self.current_trace_id = trace_id
        self.trace_metadata[trace_id] = {
            "start_time": time.time(),
            "metadata": metadata,
            "spans": []
        }
    
    def add_span(self, span: AgentSpan):
        """Add a span to current trace"""
        if self.current_trace_id:
            self.trace_metadata[self.current_trace_id]["spans"].append(span)
            self.traces.append(span)
    
    def end_trace(self, trace_id: str):
        """End trace collection"""
        if trace_id in self.trace_metadata:
            self.trace_metadata[trace_id]["end_time"] = time.time()
            duration = (
                self.trace_metadata[trace_id]["end_time"] - 
                self.trace_metadata[trace_id]["start_time"]
            )
            self.trace_metadata[trace_id]["duration"] = duration
    
    def get_trace_data(self, trace_id: str) -> Dict[str, Any]:
        """Get trace data for training"""
        if trace_id not in self.trace_metadata:
            return None
        
        trace = self.trace_metadata[trace_id]
        
        # Convert spans to training format
        transitions = []
        for span in trace["spans"]:
            if span.operation == "agent_action":
                transition = {
                    "state": span.attributes.get("state"),
                    "action": span.attributes.get("action"),
                    "reward": span.attributes.get("reward", 0),
                    "next_state": span.attributes.get("next_state"),
                    "agent_id": span.agent_id,
                    "timestamp": span.start_time
                }
                transitions.append(transition)
        
        return {
            "trace_id": trace_id,
            "duration": trace.get("duration", 0),
            "transitions": transitions,
            "metadata": trace["metadata"]
        }
    
    def export_traces(self, format: str = "jsonl") -> str:
        """Export collected traces"""
        if format == "jsonl":
            lines = []
            for trace_id in self.trace_metadata:
                trace_data = self.get_trace_data(trace_id)
                if trace_data:
                    lines.append(json.dumps(trace_data))
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


class MetricsAggregator:
    """
    Aggregates metrics for monitoring dashboard
    """
    
    def __init__(self):
        self.metrics_buffer = []
        self.aggregated_metrics = {}
        
    def add_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add a metric to buffer"""
        self.metrics_buffer.append({
            "name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": time.time()
        })
    
    def aggregate(self, window_seconds: int = 60) -> Dict[str, Any]:
        """Aggregate metrics over time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Filter recent metrics
        recent_metrics = [
            m for m in self.metrics_buffer 
            if m["timestamp"] > cutoff_time
        ]
        
        # Group by metric name
        grouped = {}
        for metric in recent_metrics:
            name = metric["name"]
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(metric["value"])
        
        # Calculate aggregates
        aggregated = {}
        for name, values in grouped.items():
            if values:
                aggregated[name] = {
                    "count": len(values),
                    "sum": sum(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # Clean old metrics
        self.metrics_buffer = recent_metrics
        
        return aggregated


# Integration with Agent Lightning components
class ObservableAgent:
    """
    Wrapper to make agents observable
    """
    
    def __init__(self, agent, observability: AgentLightningObservability):
        self.agent = agent
        self.observability = observability
    
    def act(self, state):
        """Observable agent action"""
        with self.observability.trace_agent_execution(
            agent_id=self.agent.role,
            task_type="action"
        ) as span:
            # Record state
            span.set_attribute("state", json.dumps(state.to_dict()))
            
            # Execute action
            action, transition = self.agent.act(state)
            
            # Record action and transition
            span.set_attribute("action", json.dumps(action.to_dict()))
            span.set_attribute("reward", transition.reward)
            
            # Record transition metrics
            self.observability.record_transition(transition.to_dict())
            
            return action, transition


# Example usage
if __name__ == "__main__":
    print("🔭 Testing OpenTelemetry Observability Setup")
    print("=" * 60)
    
    # Initialize observability
    observability = AgentLightningObservability(
        service_name="agent-lightning-test",
        service_version="1.0.0",
        prometheus_port=8001,
        enable_console_export=True
    )
    
    # Test agent execution tracing
    print("\n📊 Testing agent execution tracing...")
    with observability.trace_agent_execution("test_agent", "test_task") as span:
        span.add_event("task_started")
        time.sleep(0.1)  # Simulate work
        span.add_event("task_completed")
    
    # Test LLM call tracing
    print("\n🤖 Testing LLM call tracing...")
    with observability.trace_llm_call("gpt-4o", 100, 50):
        time.sleep(0.05)  # Simulate API call
    
    # Test memory operation tracing
    print("\n🧠 Testing memory operation tracing...")
    with observability.trace_memory_operation("store", "episodic", 1024):
        time.sleep(0.02)  # Simulate memory operation
    
    # Test transition recording
    print("\n🔄 Testing transition recording...")
    test_transition = {
        "reward": 0.85,
        "done": False,
        "info": {
            "agent_id": "test_agent",
            "hierarchy_level": "high"
        }
    }
    observability.record_transition(test_transition)
    
    # Test training metrics
    print("\n📈 Testing training metrics recording...")
    training_metrics = {
        "loss": 0.234,
        "accuracy": 0.89,
        "learning_rate": 0.001
    }
    observability.record_training_metrics(training_metrics)
    
    # Test trace collector
    print("\n📝 Testing trace collector...")
    collector = TraceCollector()
    collector.start_trace("test_trace_1", {"task": "test"})
    
    test_span = AgentSpan(
        name="test_action",
        agent_id="test_agent",
        operation="agent_action",
        attributes={
            "state": {"input": "test"},
            "action": {"output": "response"},
            "reward": 0.9
        },
        start_time=time.time()
    )
    collector.add_span(test_span)
    collector.end_trace("test_trace_1")
    
    trace_data = collector.get_trace_data("test_trace_1")
    print(f"   Collected trace with {len(trace_data['transitions'])} transitions")
    
    # Test metrics aggregator
    print("\n📊 Testing metrics aggregator...")
    aggregator = MetricsAggregator()
    for i in range(10):
        aggregator.add_metric("test_metric", i * 0.1)
    
    aggregated = aggregator.aggregate(window_seconds=60)
    print(f"   Aggregated metrics: {aggregated}")
    
    print("\n✅ OpenTelemetry observability setup complete!")
    print("   Prometheus metrics available at: http://localhost:8001/metrics")