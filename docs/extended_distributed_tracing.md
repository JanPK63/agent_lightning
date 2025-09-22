# Extended Distributed Tracing

This document describes the extended distributed tracing system for Agent Lightning, which provides support for multiple tracing backends beyond AgentOps.

## Overview

The extended tracing system provides a unified interface for distributed tracing across different backends:

- **AgentOps**: Original tracing system (default)
- **OpenTelemetry**: Generic OTLP exporter
- **Jaeger**: Jaeger tracing backend
- **Zipkin**: Zipkin tracing backend
- **NoOp**: No-operation tracer for testing/disabled tracing

## Architecture

### Tracer Factory Pattern

The system uses a factory pattern to create tracers based on configuration:

```python
from agentlightning.tracer import TracerFactory, TracerConfig, TracerType

# Create tracer by type
config = TracerConfig(tracer_type=TracerType.JAEGER)
tracer = TracerFactory.create_tracer(config)

# Or use convenience function
tracer = get_tracer_from_env()
```

### Unified Interface

All tracers implement the `BaseTracer` interface:

```python
class BaseTracer:
    def init(self, *args, **kwargs): ...
    def teardown(self): ...
    def instrument(self, worker_id: int): ...
    def uninstrument(self, worker_id: int): ...
    def init_worker(self, worker_id: int): ...
    def teardown_worker(self, worker_id: int): ...
    def trace_context(self, name: Optional[str] = None): ...
    def get_last_trace(self) -> List[ReadableSpan]: ...
    def get_langchain_callback_handler(self, tags: Optional[List[str]] = None): ...
```

## Configuration

### Environment Variables

Configure tracing via environment variables:

```bash
# Tracer type
TRACER_TYPE=jaeger  # agentops, opentelemetry, jaeger, zipkin, noop

# Service information
TRACER_SERVICE_NAME=my-service
TRACER_SERVICE_VERSION=1.0.0
TRACER_ENVIRONMENT=production

# Sampling
TRACER_SAMPLING_RATE=0.1  # 10% sampling

# Instrumentation
TRACER_INSTRUMENT_MANAGED=true

# Backend-specific endpoints
OTLP_ENDPOINT=http://otel-collector:4317
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
ZIPKIN_ENDPOINT=http://zipkin:9411/api/v2/spans

# AgentOps specific (when TRACER_TYPE=agentops)
AGENTOPS_MANAGED=true
AGENTOPS_DAEMON=true
```

### Programmatic Configuration

```python
from agentlightning.tracer import TracerConfig, TracerType, create_tracer

config = TracerConfig(
    tracer_type=TracerType.JAEGER,
    service_name="my-service",
    service_version="1.0.0",
    environment="production",
    sampling_rate=0.1,
    jaeger_endpoint="http://jaeger:14268/api/traces"
)

tracer = create_tracer(config)
```

## Supported Backends

### AgentOps (Default)

The original AgentOps tracing system:

```bash
TRACER_TYPE=agentops
AGENTOPS_MANAGED=true
```

**Features:**
- Local AgentOps server management
- Automatic instrumentation
- LangChain integration
- Triplet export

### OpenTelemetry

Generic OpenTelemetry tracer with configurable exporters:

```bash
TRACER_TYPE=opentelemetry
OTLP_ENDPOINT=http://otel-collector:4317
```

**Supported exporters:**
- OTLP (gRPC and HTTP)
- Jaeger
- Zipkin
- Console (for debugging)

### Jaeger

Jaeger-specific tracer:

```bash
TRACER_TYPE=jaeger
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

**Features:**
- Native Jaeger protocol
- Optimized for Jaeger UI
- High-performance UDP transport

### Zipkin

Zipkin-compatible tracer:

```bash
TRACER_TYPE=zipkin
ZIPKIN_ENDPOINT=http://zipkin:9411/api/v2/spans
```

**Features:**
- Zipkin V2 protocol
- Compatible with Zipkin server
- JSON over HTTP transport

### NoOp Tracer

Disabled tracing for testing or performance:

```bash
TRACER_TYPE=noop
```

**Features:**
- Zero overhead
- Implements full interface
- Useful for testing and CI/CD

## Usage Examples

### Basic Usage

```python
from agentlightning.tracer import get_tracer_from_env

# Get configured tracer
tracer = get_tracer_from_env()

# Initialize for worker
tracer.init_worker(0)

# Use in tracing context
with tracer.trace_context("my_operation") as span:
    # Your code here
    do_something()

# Get trace data
trace_data = tracer.get_last_trace()
```

### LangChain Integration

```python
from agentlightning.tracer import get_tracer_from_env

tracer = get_tracer_from_env()
callback_handler = tracer.get_langchain_callback_handler(tags=["agent", "llm"])

# Use with LangChain
chain = LLMChain(llm=llm, callbacks=[callback_handler])
result = chain.run("Hello world")
```

### Custom Tracing

```python
from agentlightning.tracer import get_tracer_from_env

tracer = get_tracer_from_env()

with tracer.trace_context("custom_operation") as span:
    # Add custom attributes
    if hasattr(tracer, 'add_span_attribute'):
        tracer.add_span_attribute("user_id", "12345")
        tracer.add_span_attribute("operation_type", "data_processing")

    # Add events
    if hasattr(tracer, 'add_span_event'):
        tracer.add_span_event("processing_started", {"items": 100})

    try:
        # Your operation
        process_data()
        tracer.add_span_event("processing_completed")
    except Exception as e:
        # Set error status
        if hasattr(tracer, 'set_span_status'):
            tracer.set_span_status("ERROR", str(e))
        raise
```

## Instrumentation

### Automatic Instrumentation

The system supports automatic instrumentation of popular libraries:

```python
tracer = get_tracer_from_env()
tracer.instrument(worker_id=0)  # Instrument current process
```

**Supported libraries:**
- HTTP clients (requests, aiohttp)
- Database clients (SQLAlchemy, asyncpg)
- Message queues (Redis, Kafka)
- Web frameworks (FastAPI, Flask)

### Manual Instrumentation

For custom instrumentation:

```python
from agentlightning.tracer import get_tracer_from_env

tracer = get_tracer_from_env()

with tracer.trace_context("manual_span") as span:
    # Manual span creation
    span.set_attribute("custom.attribute", "value")
    span.add_event("custom_event", {"key": "value"})

    # Your code
    do_work()

    span.set_status(Status(StatusCode.OK))
```

## Sampling

Control trace volume with sampling:

```bash
# Sample 10% of traces
TRACER_SAMPLING_RATE=0.1

# Sample 100% (all traces)
TRACER_SAMPLING_RATE=1.0
```

Different backends support different sampling strategies:
- **Head sampling**: Decide at trace start
- **Tail sampling**: Decide after trace completion
- **Rate limiting**: Sample based on throughput

## Monitoring and Observability

### Health Checks

Tracers integrate with the health check system:

```python
from shared.health_check import health_check_service

# Tracers can be checked for connectivity
# Implementation depends on backend
```

### Metrics

Trace metrics are available through the monitoring system:

- Trace count and rate
- Span duration percentiles
- Error rates by service/operation
- Sampling effectiveness

### Dashboards

Integration with monitoring dashboards:

- **Jaeger UI**: Trace visualization and analysis
- **Zipkin UI**: Dependency graphs and latency analysis
- **Grafana**: Custom dashboards with trace metrics
- **AgentOps Dashboard**: Agent-specific analytics

## Deployment Examples

### Kubernetes with Jaeger

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-lightning
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: TRACER_TYPE
          value: "jaeger"
        - name: JAEGER_ENDPOINT
          value: "http://jaeger-collector:14268/api/traces"
        - name: TRACER_SERVICE_NAME
          value: "agent-lightning"
```

### Docker Compose with Zipkin

```yaml
version: '3.8'
services:
  app:
    environment:
      - TRACER_TYPE=zipkin
      - ZIPKIN_ENDPOINT=http://zipkin:9411/api/v2/spans
    depends_on:
      - zipkin

  zipkin:
    image: openzipkin/zipkin
    ports:
      - "9411:9411"
```

### Cloud Deployment (AWS X-Ray)

```bash
TRACER_TYPE=opentelemetry
OTLP_ENDPOINT=https://xray-collector.amazonaws.com/v1/traces
AWS_REGION=us-east-1
```

## Best Practices

### 1. Configuration Management

- Use environment variables for different environments
- Validate configuration at startup
- Provide sensible defaults

### 2. Sampling Strategy

- Start with low sampling rates in production
- Increase for debugging/problem analysis
- Use dynamic sampling based on load

### 3. Resource Management

- Configure appropriate buffer sizes
- Set reasonable timeouts
- Monitor exporter performance

### 4. Security

- Use HTTPS for production endpoints
- Implement authentication if required
- Avoid exposing sensitive data in spans

### 5. Performance

- Use async exporters when possible
- Batch spans for efficiency
- Monitor exporter queue sizes

## Troubleshooting

### Common Issues

1. **Traces not appearing**
   - Check endpoint connectivity
   - Verify sampling rate
   - Check exporter logs

2. **High latency**
   - Review exporter configuration
   - Check network connectivity
   - Monitor queue sizes

3. **Missing spans**
   - Verify instrumentation is applied
   - Check sampling configuration
   - Review span context propagation

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('agentlightning.tracer').setLevel(logging.DEBUG)
```

Test tracer connectivity:

```python
tracer = get_tracer_from_env()
tracer.init_worker(0)

with tracer.trace_context("test") as span:
    print("Span created successfully")

trace_data = tracer.get_last_trace()
print(f"Trace data: {trace_data}")
```

## Future Enhancements

- **Custom exporters**: Support for additional tracing backends
- **Distributed context**: Enhanced context propagation across services
- **Performance profiling**: Integration with profiling tools
- **Trace analytics**: Advanced analytics and alerting
- **Service mesh integration**: Istio, Linkerd integration

## Migration Guide

### From AgentOps-only to Multi-backend

1. **Assess current usage**
   ```python
   # Check current AgentOps usage
   from agentlightning.tracer import AgentOpsTracer
   ```

2. **Choose target backend**
   ```bash
   # Set environment variable
   export TRACER_TYPE=jaeger
   ```

3. **Update configuration**
   ```bash
   # Add backend-specific settings
   export JAEGER_ENDPOINT=http://jaeger:14268/api/traces
   ```

4. **Test migration**
   ```python
   # Test with new backend
   from agentlightning.tracer import get_tracer_from_env
   tracer = get_tracer_from_env()
   ```

5. **Gradual rollout**
   - Deploy to staging first
   - Monitor for issues
   - Roll back if necessary

The extended tracing system maintains backward compatibility while providing flexibility to use the best tracing backend for your infrastructure.