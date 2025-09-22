from typing import Any, Dict, List, Optional, Union, Literal, Annotated
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, Discriminator

# Mock ReadableSpan for tracing support (OpenTelemetry optional)
class ReadableSpan:
    """Mock ReadableSpan for environments without OpenTelemetry"""
    def __init__(self, *args, **kwargs):
        pass
    def to_json(self):
        return "{}"

OPENTELEMETRY_AVAILABLE = False

__all__ = [
    "Triplet",
    "Rollout",
    "Task",
    "TaskInput",
    "TaskIfAny",
    "RolloutRawResult",
    "Resource",
    "LLM",
    "PromptTemplate",
    "ResourceUnion",
    "NamedResources",
    "ResourcesUpdate",
    "GenericResponse",
    "Spec",
    "WorkflowStep",
    "SpecPlan",
    "SpecExecution",
    "ParallelWorkerBase",
    "Event",
    "EventEnvelope",
    "EventStream",
    "EventFilter",
    "MultiModalContent",
    "TextContent",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    "ContentType",
    "AgentMessage",
    "AgentAddress",
    "CommunicationProtocol",
    "MessageType",
]


class ContentType(str, Enum):
    """Enumeration of supported content types for multi-modal inputs"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class MultiModalContent(BaseModel):
    """Base class for multi-modal content"""
    content_type: ContentType
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextContent(MultiModalContent):
    """Text content for multi-modal inputs"""
    content_type: Literal[ContentType.TEXT] = ContentType.TEXT
    text: str


class ImageContent(MultiModalContent):
    """Image content for multi-modal inputs"""
    content_type: Literal[ContentType.IMAGE] = ContentType.IMAGE
    # Can be base64 encoded string, URL, or file path
    image_data: Union[str, bytes]
    format: Optional[str] = None  # e.g., "png", "jpeg", "webp"


class AudioContent(MultiModalContent):
    """Audio content for multi-modal inputs"""
    content_type: Literal[ContentType.AUDIO] = ContentType.AUDIO
    # Can be base64 encoded string, URL, or file path
    audio_data: Union[str, bytes]
    format: Optional[str] = None  # e.g., "mp3", "wav", "flac"
    duration: Optional[float] = None  # in seconds


class VideoContent(MultiModalContent):
    """Video content for multi-modal inputs"""
    content_type: Literal[ContentType.VIDEO] = ContentType.VIDEO
    # Can be base64 encoded string, URL, or file path
    video_data: Union[str, bytes]
    format: Optional[str] = None  # e.g., "mp4", "avi", "mov"
    duration: Optional[float] = None  # in seconds
    # Optional thumbnail image
    thumbnail: Optional[ImageContent] = None


class MessageType(str, Enum):
    """Types of messages that can be exchanged between agents"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"


class CommunicationProtocol(str, Enum):
    """Supported communication protocols for agent-to-agent communication"""
    DIRECT = "direct"  # Direct method calls
    MESSAGE_QUEUE = "message_queue"  # Async message queues
    PUBSUB = "pubsub"  # Publish-subscribe pattern
    RPC = "rpc"  # Remote procedure calls
    EVENT_STREAM = "event_stream"  # Event streaming


class AgentAddress(BaseModel):
    """Address information for an agent in the communication system"""
    agent_id: str
    agent_type: Optional[str] = None
    namespace: Optional[str] = None  # For multi-tenant deployments
    endpoint: Optional[str] = None  # Network endpoint if applicable
    capabilities: List[str] = Field(default_factory=list)  # Agent capabilities
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Message structure for agent-to-agent communication"""
    message_id: str
    sender: AgentAddress
    recipient: AgentAddress
    message_type: MessageType
    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT
    subject: str  # Brief description of the message purpose
    content: Any  # The actual message payload
    correlation_id: Optional[str] = None  # For request-response correlation
    reply_to: Optional[AgentAddress] = None  # Where to send responses
    timestamp: float
    ttl: Optional[float] = None  # Time to live in seconds
    priority: int = 0  # Message priority (higher = more important)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Triplet(BaseModel):
    """A standard structure for a single turn in a trajectory."""

    prompt: Any
    response: Any
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Rollout(BaseModel):
    """The standard reporting object from client to server."""

    rollout_id: str

    # Primary, high-level feedback
    final_reward: Optional[float] = None

    # Structured, sequential feedback for RL-style optimization
    triplets: Optional[List[Triplet]] = None

    # Optional, rich-context data for deep analysis
    trace: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of spans that conform to the OpenTelemetry JSON format. "
        "Users of the opentelemetry-sdk can generate this by calling "
        "json.loads(readable_span.to_json()).",
    )
    logs: Optional[List[str]] = None

    # A bucket for any other relevant information
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Support both legacy Any inputs and new multi-modal inputs
TaskInput = Union[
    Any,  # Legacy support for any input type
    str,  # Simple text input
    List[MultiModalContent],  # Multi-modal content list
    MultiModalContent,  # Single multi-modal content
]


class Task(BaseModel):
    """A task (rollout request) to be processed by the client agent."""

    rollout_id: str
    input: TaskInput

    mode: Optional[Literal["train", "val", "test"]] = None
    resources_id: Optional[str] = None

    # Optional fields for tracking task lifecycle
    create_time: Optional[float] = None
    last_claim_time: Optional[float] = None
    num_claims: Optional[int] = None

    # Allow additional metadata fields
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskIfAny(BaseModel):
    is_available: bool
    task: Optional[Task] = None


RolloutRawResult = Union[None, float, List[Triplet], List[Dict[str, Any]], List[ReadableSpan], Rollout]


class Resource(BaseModel):
    """
    Base class for all tunable resources.
    """

    resource_type: Any


class LLM(Resource):
    """
    Provide an LLM endpoint and model name as a resource.

    Attributes:
        endpoint (str): The URL of the LLM API endpoint.
        model (str): The identifier for the model to be used (e.g., 'gpt-4o').
        sampling_parameters (SamplingParameters): A dictionary of hyperparameters
            for model inference, such as temperature, top_p, etc.
    """

    resource_type: Literal["llm"] = "llm"
    endpoint: str
    model: str
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)


class PromptTemplate(Resource):
    """
    A prompt template as a resource.

    Attributes:
        template (str): The template string. The format depends on the engine.
        engine (Literal['jinja', 'f-string', 'poml']): The templating engine
            to use for rendering the prompt. I imagine users can use their own
            customized engines, but algos can only well operate on a subset of them.
    """

    resource_type: Literal["prompt_template"] = "prompt_template"
    template: str
    engine: Literal["jinja", "f-string", "poml"]


# Use discriminated union for proper deserialization
ResourceUnion = Annotated[Union[LLM, PromptTemplate], Field(discriminator="resource_type")]
NamedResources = Dict[str, ResourceUnion]
"""
A dictionary-like class to hold named resources.

Example:
    resources: NamedResources = {
        'main_llm': LLM(
            endpoint="http://localhost:8080",
            model="llama3",
            sampling_parameters={'temperature': 0.7, 'max_tokens': 100}
        ),
        'system_prompt': PromptTemplate(
            template="You are a helpful assistant.",
            engine='f-string'
        )
    }
"""


class ResourcesUpdate(BaseModel):
    """
    A resource update message to be sent from the server to clients.

    This message contains a dictionary of resources that clients should use
    for subsequent tasks. It is used to update the resources available to
    clients dynamically.
    """

    resources_id: str
    resources: NamedResources


class GenericResponse(BaseModel):
    """
    A generic response message that can be used for various purposes.
    """

    status: str = "success"
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class WorkflowStep(BaseModel):
    """
    A single step in a workflow specification.
    """
    id: str
    name: str
    description: str
    agent_type: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)  # IDs of prerequisite steps
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Spec(BaseModel):
    """
    A specification for a workflow or task that can be executed by agents.
    """
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    workflow: List[WorkflowStep] = Field(default_factory=list)
    resources: NamedResources = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class SpecPlan(BaseModel):
    """
    A plan generated from a spec, containing tasks and workflow definition.
    """
    spec_id: str
    tasks: List[Task] = Field(default_factory=list)
    workflow_definition: Optional[Dict[str, Any]] = None  # LangGraph workflow definition
    estimated_duration: Optional[float] = None
    resource_requirements: NamedResources = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpecExecution(BaseModel):
    """
    Result of executing a spec plan.
    """
    spec_id: str
    execution_id: str
    status: Literal["running", "completed", "failed"] = "running"
    results: Dict[str, Any] = Field(default_factory=dict)
    rollouts: List[Rollout] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParallelWorkerBase:
    """Base class for objects that can be parallelized across multiple worker processes.

    This class defines the standard lifecycle for parallel processing:

    Main Process:
        1. init() - Initialize the object in the main process
        2. spawn workers and call init_worker() in each worker
        3. run() - Execute the main workload in parallel across workers
        4. teardown_worker() - Clean up resources in each worker
        5. teardown() - Final cleanup in the main process

    Subclasses should implement the run() method and optionally override
    the lifecycle methods for custom initialization and cleanup behavior.
    """

    def __init__(self) -> None:
        """Initialize the base class. This method can be overridden by subclasses."""
        self.worker_id: Optional[int] = None

    def init(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        self.worker_id = worker_id

    def run(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        pass

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        pass


class Event(BaseModel):
    """Event model for event sourcing - captures state changes in the system"""
    event_id: str
    aggregate_id: str  # Entity ID (agent_id, task_id, workflow_id, etc.)
    aggregate_type: str  # 'agent', 'task', 'workflow', 'resource', 'rollout'
    event_type: str  # 'created', 'updated', 'started', 'completed', 'failed', etc.
    event_data: Dict[str, Any]  # Event payload with details
    timestamp: datetime
    version: int = 1  # Aggregate version for optimistic concurrency
    correlation_id: Optional[str] = None  # For tracking related events
    causation_id: Optional[str] = None  # ID of event that caused this event
    user_id: Optional[str] = None  # User who triggered the event
    service_name: Optional[str] = None  # Service that generated the event
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional event metadata

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EventEnvelope(BaseModel):
    """Envelope for event transmission and storage"""
    event: Event
    partition_key: Optional[str] = None  # For distributed event stores
    headers: Dict[str, Any] = Field(default_factory=dict)  # Transport headers


class EventStream(BaseModel):
    """Represents a stream of events for an aggregate"""
    aggregate_id: str
    aggregate_type: str
    events: List[Event] = Field(default_factory=list)
    version: int = 0
    snapshot: Optional[Dict[str, Any]] = None  # Latest snapshot if available

    def append_event(self, event: Event) -> None:
        """Append an event to the stream"""
        if event.aggregate_id != self.aggregate_id:
            raise ValueError("Event aggregate_id does not match stream")
        if event.version != self.version + 1:
            raise ValueError(f"Event version {event.version} does not follow stream version {self.version}")

        self.events.append(event)
        self.version = event.version

    def get_events_from_version(self, from_version: int) -> List[Event]:
        """Get events from a specific version onwards"""
        return [e for e in self.events if e.version >= from_version]


class EventFilter(BaseModel):
    """Filter criteria for querying events"""
    aggregate_id: Optional[str] = None
    aggregate_type: Optional[str] = None
    event_type: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    service_name: Optional[str] = None
    from_timestamp: Optional[datetime] = None
    to_timestamp: Optional[datetime] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
