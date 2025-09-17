from typing import List, Optional, Any, Callable
from contextlib import contextmanager
from agentlightning.tracer.base import BaseTracer


class NoOpTracer(BaseTracer):
    """
    A no-op tracer implementation used for MVP/testing.
    It adheres to the BaseTracer interface but does not emit any actual spans.
    This allows the rest of the system (Runner, LitAgent) to operate without
    requiring a full tracing backend.
    """

    def __init__(self) -> None:
        # Maintain a placeholder for last guardable trace if needed by callers.
        self._last_trace: List[Any] = []

    @contextmanager
    def trace_context(self, name: Optional[str] = None) -> Any:
        """
        Start a tracing context. In the NoOp implementation this yields control
        immediately without creating spans.
        """
        yield

    def get_last_trace(self) -> List[Any]:
        """
        Return the last captured trace. For NoOp, this is an empty list.
        """
        return self._last_trace

    def trace_run(self, func: Callable, *args, **kwargs) -> Any:
        """
        Convenience wrapper to trace a synchronous function. No-op tracing.
        """
        name_for_trace = getattr(func, "__name__", "anonymous_function")
        with self.trace_context(name=name_for_trace):
            return func(*args, **kwargs)

    async def trace_run_async(
        self, func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """
        Convenience wrapper to trace the execution of an asynchronous function.
        """
        name_for_trace = getattr(func, "__name__", "anonymous_function")
        with self.trace_context(name=name_for_trace):
            return await func(*args, **kwargs)