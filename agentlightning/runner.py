import json
import logging
import time
from typing import List, Optional, Dict, Any

from opentelemetry.sdk.trace import ReadableSpan
from .client import AgentLightningClient
from .litagent import LitAgent
from .types import Rollout, Task, Triplet, RolloutRawResult
from .types import ParallelWorkerBase
from .tracer.base import BaseTracer
from .tracer import TripletExporter
from .knowledge_client import get_knowledge_client
from shared.event_store import event_store, create_event

logger = logging.getLogger(__name__)


class AgentRunner(ParallelWorkerBase):
    """Manages the agent's execution loop and integrates with AgentOps.

    This class orchestrates the interaction between the agent (`LitAgent`) and
    the server (`AgentLightningClient`). It handles polling for tasks, executing
    the agent's logic, and reporting results back to the server. If enabled,
    it will also automatically trace each rollout using AgentOps.

    Attributes:
        agent: The `LitAgent` instance containing the agent's logic.
        client: The `AgentLightningClient` for server communication.
        tracer: The tracer instance for this runner/worker.
        worker_id: An optional identifier for the worker process.
        max_tasks: The maximum number of tasks to process before stopping.
    """

    def __init__(
        self,
        agent: LitAgent,
        client: AgentLightningClient,
        tracer: Optional[BaseTracer] = None,
        triplet_exporter: Optional[TripletExporter] = None,
        worker_id: Optional[int] = None,
        max_tasks: Optional[int] = None,
        knowledge_service_url: Optional[str] = None,
        enable_knowledge_context: bool = True,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
    ):
        super().__init__()
        self.agent = agent
        self.client = client
        if tracer is None:
            try:
                from .tracer.noop_tracer import NoOpTracer
                tracer = NoOpTracer()
            except Exception:
                tracer = None
        self.tracer = tracer
        self.triplet_exporter = triplet_exporter

        # Worker-specific attributes
        self.worker_id = worker_id
        self.max_tasks = max_tasks

        # Knowledge integration
        self.enable_knowledge_context = enable_knowledge_context
        self.knowledge_client = None
        if enable_knowledge_context:
            try:
                self.knowledge_client = get_knowledge_client(
                    knowledge_service_url or "http://localhost:8014"
                )
                logger.info("Knowledge client initialized for context retrieval")
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge client: {e}")
                self.knowledge_client = None

        # Retry and backoff configuration
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

    def _log_prefix(self, rollout_id: Optional[str] = None) -> str:
        """Generates a standardized log prefix for the current worker."""
        if self.worker_id is not None:
            if rollout_id:
                return f"[Worker {self.worker_id} | Rollout {rollout_id}]"
            else:
                return f"[Worker {self.worker_id}]"
        if rollout_id:
            return f"[Rollout {rollout_id}]"
        return "[Default Worker]"

    def _to_rollout_object(
        self,
        result: RolloutRawResult,
        rollout_id: str,
    ) -> Rollout:
        """Standardizes the agent's return value into a Rollout object.

        Args:
            result: The output from the agent's rollout method.
            rollout_id: The unique identifier for the current task.

        Returns:
            A standardized `Rollout` object for reporting to the server.
        """
        trace: Any = None
        final_reward: Optional[float] = None
        triplets: Optional[List[Triplet]] = None
        trace_spans: Optional[List[ReadableSpan]] = None

        # Handle different types of results from the agent
        # Case 1: result is a float (final reward)
        if isinstance(result, float):
            final_reward = result
        # Case 2: result is a list of Triplets
        if isinstance(result, list) and all(isinstance(t, Triplet) for t in result):
            triplets = result  # type: ignore
        # Case 3: result is a list of ReadableSpan (OpenTelemetry spans)
        if isinstance(result, list) and all(isinstance(t, ReadableSpan) for t in result):
            trace_spans = result  # type: ignore
            trace = [json.loads(readable_span.to_json()) for readable_span in trace_spans]  # type: ignore
        # Case 4: result is a list of dict (trace JSON)
        if isinstance(result, list) and all(isinstance(t, dict) for t in result):
            trace = result
        # Case 5: result is a Rollout object
        if isinstance(result, Rollout):
            final_reward = result.final_reward
            triplets = result.triplets
            trace = result.trace

        # If the agent has tracing enabled, use the tracer's last trace if not already set
        if self.tracer and (trace is None or trace_spans is None):
            spans = self.tracer.get_last_trace()
            if spans:
                trace = [json.loads(readable_span.to_json()) for readable_span in spans]
                trace_spans = spans

        # Always extract triplets from the trace using TripletExporter
        if trace_spans:
            triplets = self.triplet_exporter.export(trace_spans)

        # If the agent has triplets, use the last one for final reward if not set
        if triplets and triplets[-1].reward is not None and final_reward is None:
            final_reward = triplets[-1].reward

        # Create the Rollout object with standardized fields
        result_dict: Dict[str, Any] = {
            "rollout_id": rollout_id,
        }
        if final_reward is not None:
            result_dict["final_reward"] = final_reward
        if triplets is not None:
            result_dict["triplets"] = triplets
        if trace is not None:
            result_dict["trace"] = trace

        if isinstance(result, Rollout):
            return result.model_copy(update=result_dict)
        return Rollout(**result_dict)

    def get_task_context(self, task: Task) -> Optional[List[Dict]]:
        """Get relevant knowledge context for a task."""
        if not self.knowledge_client or not self.enable_knowledge_context:
            return None

        try:
            # Extract task description from input
            task_description = ""
            if isinstance(task.input, str):
                task_description = task.input
            elif isinstance(task.input, dict):
                # Try to extract meaningful description from dict
                task_description = task.input.get('description', '') or \
                                 task.input.get('task', '') or \
                                 str(task.input)
            else:
                task_description = str(task.input)

            if not task_description:
                logger.debug(f"{self._log_prefix(task.rollout_id)} No task description available for context")
                return None

            # Get agent ID (assuming it's stored in task metadata or we need to derive it)
            agent_id = getattr(self.agent, 'agent_id', None) or \
                      task.metadata.get('agent_id') if task.metadata else None

            if not agent_id:
                logger.debug(f"{self._log_prefix(task.rollout_id)} No agent ID available for context")
                return None

            # Fetch context from knowledge service
            context = self.knowledge_client.get_context_for_task(
                agent_id=agent_id,
                task_description=task_description,
                limit=5
            )

            if context:
                logger.info(f"{self._log_prefix(task.rollout_id)} Retrieved {len(context)} knowledge items for context")
                return context
            else:
                logger.debug(f"{self._log_prefix(task.rollout_id)} No relevant knowledge found for task")
                return None

        except Exception as e:
            logger.warning(f"{self._log_prefix(task.rollout_id)} Failed to get task context: {e}")
            return None

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic and exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"{self._log_prefix()} All {self.max_retries + 1} attempts failed. Final error: {e}")
                    raise e
                backoff_time = min(self.backoff_base * (2 ** attempt), self.backoff_max)
                logger.warning(f"{self._log_prefix()} Attempt {attempt + 1} failed: {e}. Retrying in {backoff_time:.2f}s")
                time.sleep(backoff_time)

    def run(self) -> bool:
        """Poll the task and rollout once synchronously."""
        self.agent.set_runner(self)  # Ensure the agent has a reference to this runner

        task = self.client.poll_next_task()
        if task is None:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return False
        rollout_id = task.rollout_id

        # Event: Task received
        try:
            task_received_event = create_event(
                aggregate_id=task.rollout_id,
                aggregate_type="task",
                event_type="received",
                event_data={
                    "task_input": task.input,
                    "mode": task.mode,
                    "resources_id": task.resources_id,
                    "worker_id": self.worker_id
                },
                user_id=getattr(self.agent, 'agent_id', None),
                service_name="agent_runner"
            )
            event_store.save_event(task_received_event)
        except Exception as e:
            logger.warning(f"Failed to publish task received event: {e}")

        resources_id = task.resources_id
        resources_update = None
        if resources_id:
            resources_update = self.client.get_resources_by_id(resources_id)
        else:
            logger.debug(f"{self._log_prefix(rollout_id)} No 'resources_id'. Fetching latest resources.")
            resources_update = self.client.get_latest_resources()
        if not resources_update:
            logger.error(f"{self._log_prefix(rollout_id)} Failed to fetch resources. Skipping.")
            return False

        rollout_obj = Rollout(rollout_id=task.rollout_id)  # Default empty rollout

        # Get knowledge context for the task
        task_context = self.get_task_context(task)

        try:
            try:
                self.agent.on_rollout_start(task, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_start hook.")

            # Event: Rollout started
            try:
                rollout_started_event = create_event(
                    aggregate_id=task.rollout_id,
                    aggregate_type="rollout",
                    event_type="started",
                    event_data={
                        "task_input": task.input,
                        "mode": task.mode,
                        "resources_id": task.resources_id,
                        "worker_id": self.worker_id,
                        "agent_id": getattr(self.agent, 'agent_id', None)
                    },
                    user_id=getattr(self.agent, 'agent_id', None),
                    service_name="agent_runner"
                )
                event_store.save_event(rollout_started_event)
            except Exception as e:
                logger.warning(f"Failed to publish rollout started event: {e}")

            with self.tracer.trace_context(name=f"rollout_{rollout_id}"):
                start_time = time.time()
                rollout_method = self.agent.training_rollout if task.mode == "train" else self.agent.validation_rollout

                # Prepare enhanced input with context if available
                enhanced_input = task.input
                if task_context:
                    if isinstance(task.input, dict):
                        enhanced_input = {**task.input, "knowledge_context": task_context}
                    else:
                        # For non-dict inputs, create a wrapper
                        enhanced_input = {
                            "original_input": task.input,
                            "knowledge_context": task_context
                        }

                # Pass the enhanced input with context
                result = self._execute_with_retry(rollout_method, enhanced_input, task.rollout_id, resources_update.resources)
                rollout_obj = self._to_rollout_object(result, task.rollout_id)
                end_time = time.time()
                logger.info(
                    f"{self._log_prefix(rollout_id)} Completed in "
                    f"{end_time - start_time:.2f}s. Triplet length: "
                    f"{len(rollout_obj.triplets) if rollout_obj.triplets is not None else 'N/A'}. "
                    f"Reward: {rollout_obj.final_reward}"
                )

                # Event: Rollout completed
                try:
                    rollout_completed_event = create_event(
                        aggregate_id=task.rollout_id,
                        aggregate_type="rollout",
                        event_type="completed",
                        event_data={
                            "final_reward": rollout_obj.final_reward,
                            "triplet_count": len(rollout_obj.triplets) if rollout_obj.triplets else 0,
                            "execution_time": end_time - start_time,
                            "mode": task.mode,
                            "worker_id": self.worker_id
                        },
                        user_id=getattr(self.agent, 'agent_id', None),
                        service_name="agent_runner"
                    )
                    event_store.save_event(rollout_completed_event)
                except Exception as e:
                    logger.warning(f"Failed to publish rollout completed event: {e}")

        except Exception as e:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")

            # Event: Rollout failed
            try:
                rollout_failed_event = create_event(
                    aggregate_id=task.rollout_id,
                    aggregate_type="rollout",
                    event_type="failed",
                    event_data={
                        "error": str(e),
                        "task_input": task.input,
                        "mode": task.mode,
                        "worker_id": self.worker_id
                    },
                    user_id=getattr(self.agent, 'agent_id', None),
                    service_name="agent_runner"
                )
                event_store.save_event(rollout_failed_event)
            except Exception as event_error:
                logger.warning(f"Failed to publish rollout failed event: {event_error}")
        finally:
            try:
                self.agent.on_rollout_end(task, rollout_obj, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")
            self.client.post_rollout(rollout_obj)

        return True

    def iter(self) -> int:
        """Executes the synchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started sync rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            if self.run():
                num_tasks_processed += 1

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")

        logger.info(f"{self._log_prefix()} Finished sync rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

    async def run_async(self) -> bool:
        """Poll the task and rollout once."""
        self.agent.set_runner(self)  # Ensure the agent has a reference to this runner

        task = await self.client.poll_next_task_async()
        if task is None:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return False
        rollout_id = task.rollout_id

        resources_id = task.resources_id
        resources_update = None
        if resources_id:
            resources_update = await self.client.get_resources_by_id_async(resources_id)
        else:
            logger.debug(f"{self._log_prefix(rollout_id)} No 'resources_id'. Fetching latest resources.")
            resources_update = await self.client.get_latest_resources_async()
        if not resources_update:
            logger.error(f"{self._log_prefix(rollout_id)} Failed to fetch resources. Skipping.")
            return False

        rollout_obj = Rollout(rollout_id=task.rollout_id)  # Default empty rollout

        # Get knowledge context for the task
        task_context = self.get_task_context(task)

        try:
            try:
                self.agent.on_rollout_start(task, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_start hook.")

            with self.tracer.trace_context(name=f"rollout_{rollout_id}"):
                start_time = time.time()
                rollout_method = (
                    self.agent.training_rollout_async if task.mode == "train" else self.agent.validation_rollout_async
                )

                # Prepare enhanced input with context if available
                enhanced_input = task.input
                if task_context:
                    if isinstance(task.input, dict):
                        enhanced_input = {**task.input, "knowledge_context": task_context}
                    else:
                        # For non-dict inputs, create a wrapper
                        enhanced_input = {
                            "original_input": task.input,
                            "knowledge_context": task_context
                        }

                # Pass the enhanced input with context
                result = await rollout_method(enhanced_input, task.rollout_id, resources_update.resources)
                rollout_obj = self._to_rollout_object(result, task.rollout_id)
                end_time = time.time()
                logger.info(
                    f"{self._log_prefix(rollout_id)} Completed in "
                    f"{end_time - start_time:.2f}s. Reward: {rollout_obj.final_reward}"
                )
        except Exception:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")
        finally:
            try:
                self.agent.on_rollout_end(task, rollout_obj, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")
            await self.client.post_rollout_async(rollout_obj)

        return True

    async def iter_async(self) -> int:
        """Executes the asynchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            if await self.run_async():
                num_tasks_processed += 1

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")
        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed
