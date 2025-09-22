"""
Database models for Agent Lightning
Using SQLAlchemy ORM for PostgreSQL database
"""

from sqlalchemy import (
    Column, String, Integer, JSON, DateTime, Text, Float,
    ForeignKey, Boolean, LargeBinary
)
from .encrypted_fields import EncryptedString, EncryptedText, EncryptedJSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class Agent(Base):
    """Agent model - represents AI agents in the system"""
    __tablename__ = 'agents'

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    model = Column(String(50), nullable=False)
    specialization = Column(String(50))
    status = Column(String(20), default='idle')

    # Original fields
    config = Column(JSON, default={})
    capabilities = Column(JSON, default=[])

    # Encrypted fields (for sensitive agent configuration)
    config_encrypted = Column(EncryptedJSON('agent_config_key'))
    capabilities_encrypted = Column(EncryptedJSON('agent_capabilities_key'))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        """Convert agent to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'model': self.model,
            'specialization': self.specialization,
            'status': self.status,
            'config': self.config,
            'capabilities': self.capabilities,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Task(Base):
    """Task model - represents tasks assigned to agents"""
    __tablename__ = 'tasks'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String(50), ForeignKey('agents.id', ondelete='CASCADE'))
    description = Column(Text)
    status = Column(String(20), default='pending')
    priority = Column(String(20), default='normal')
    context = Column(JSON, default={})
    result = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    def to_dict(self):
        """Convert task to dictionary"""
        return {
            'id': str(self.id),
            'agent_id': self.agent_id,
            'description': self.description,
            'status': self.status,
            'priority': self.priority,
            'context': self.context,
            'result': self.result,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class Knowledge(Base):
    """Knowledge model - stores agent learning and knowledge base"""
    __tablename__ = 'knowledge'
    
    id = Column(String(255), primary_key=True)
    agent_id = Column(String(50), ForeignKey('agents.id', ondelete='CASCADE'))
    category = Column(String(50))
    content = Column(Text)
    source = Column(Text)
    knowledge_metadata = Column('metadata', JSON, default={})
    usage_count = Column(Integer, default=0)
    relevance_score = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True))
    
    # Note: Vector embedding field would be added with pgvector extension
    # embedding = Column(Vector(1536))  # For similarity search
    
    def to_dict(self):
        """Convert knowledge to dictionary"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'category': self.category,
            'content': self.content,
            'source': self.source,
            'metadata': self.knowledge_metadata,
            'usage_count': self.usage_count,
            'relevance_score': self.relevance_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
        }

class Workflow(Base):
    """Workflow model - complex task orchestration"""
    __tablename__ = 'workflows'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100))
    description = Column(Text)

    # Original fields
    steps = Column(JSON)
    status = Column(String(20), default='draft')
    created_by = Column(String(50))
    assigned_to = Column(String(50))
    context = Column(JSON, default={})

    # Encrypted fields (for sensitive workflow data)
    steps_encrypted = Column(EncryptedJSON('workflow_steps_key'))
    context_encrypted = Column(EncryptedJSON('workflow_context_key'))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        """Convert workflow to dictionary"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'steps': self.steps,
            'status': self.status,
            'created_by': self.created_by,
            'assigned_to': self.assigned_to,
            'context': self.context,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Session(Base):
    """Session model - user session management"""
    __tablename__ = 'sessions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(100))
    token = Column(String(500), unique=True, index=True)
    data = Column(JSON, default={})
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self):
        """Convert session to dictionary"""
        return {
            'id': str(self.id),
            'user_id': self.user_id,
            'token': self.token,
            'data': self.data,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def is_expired(self):
        """Check if session is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

class Metric(Base):
    """Metric model - performance and monitoring data"""
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_name = Column(String(50), index=True)
    metric_name = Column(String(100), index=True)
    value = Column(Float)
    tags = Column(JSON, default={})
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def to_dict(self):
        """Convert metric to dictionary"""
        return {
            'id': self.id,
            'service_name': self.service_name,
            'metric_name': self.metric_name,
            'value': self.value,
            'tags': self.tags,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class Conversation(Base):
    """Conversation model - stores all Q&A for knowledge gathering"""
    __tablename__ = 'conversations'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(255), index=True)
    agent_id = Column(String(50), ForeignKey('agents.id', ondelete='CASCADE'), index=True)

    # Original fields (for backward compatibility)
    user_query = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)

    # Encrypted fields (for sensitive conversation data)
    user_query_encrypted = Column(EncryptedText('conversation_query_key'))
    agent_response_encrypted = Column(EncryptedText('conversation_response_key'))

    context = Column(JSON, default={})
    knowledge_metadata = Column(JSON, default={})
    success = Column(Boolean, default=True)
    knowledge_used = Column(Integer, default=0)
    rl_enhanced = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def to_dict(self):
        """Convert conversation to dictionary"""
        return {
            'id': str(self.id),
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'user_query': self.user_query,
            'agent_response': self.agent_response,
            'context': self.context,
            'metadata': self.knowledge_metadata,
            'success': self.success,
            'knowledge_used': self.knowledge_used,
            'rl_enhanced': self.rl_enhanced,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class User(Base):
    """User model - system users"""
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)

    # Original fields (for backward compatibility during migration)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    # Encrypted fields (to be used after encryption rollout)
    email_encrypted = Column(EncryptedString('user_email_key', 100))
    password_hash_encrypted = Column(EncryptedString('user_password_key', 255))

    role = Column(String(20), default='user')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        """Convert user to dictionary (without password)"""
        return {
            'id': str(self.id),
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ServerTask(Base):
    """Server task model - tasks in the server queue"""
    __tablename__ = 'server_tasks'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    rollout_id = Column(String(36), unique=True, index=True)
    task_input = Column(JSON, nullable=False)
    mode = Column(String(20))  # train, val, test
    resources_id = Column(String(36), index=True)
    create_time = Column(Float, nullable=False)
    last_claim_time = Column(Float)
    num_claims = Column(Integer, default=0)
    status = Column(String(20), default='queued')  # queued, processing, completed, failed
    task_metadata = Column('metadata', JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        """Convert server task to dictionary"""
        return {
            'id': str(self.id),
            'rollout_id': self.rollout_id,
            'input': self.task_input,
            'mode': self.mode,
            'resources_id': self.resources_id,
            'create_time': self.create_time,
            'last_claim_time': self.last_claim_time,
            'num_claims': self.num_claims,
            'status': self.status,
            'metadata': self.task_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ServerResource(Base):
    """Server resource model - versioned resources"""
    __tablename__ = 'server_resources'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    resources_id = Column(String(36), unique=True, index=True)
    resources = Column(JSON, nullable=False)
    is_latest = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        """Convert server resource to dictionary"""
        return {
            'id': str(self.id),
            'resources_id': self.resources_id,
            'resources': self.resources,
            'is_latest': self.is_latest,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ServerRollout(Base):
    """Server rollout model - completed rollouts"""
    __tablename__ = 'server_rollouts'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    rollout_id = Column(String(36), unique=True, index=True)
    final_reward = Column(Float)
    triplets = Column(JSON)  # List of Triplet objects as JSON
    trace = Column(JSON)  # Trace data as JSON
    logs = Column(JSON)  # Log data as JSON
    rollout_metadata = Column('metadata', JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        """Convert server rollout to dictionary"""
        return {
            'id': str(self.id),
            'rollout_id': self.rollout_id,
            'final_reward': self.final_reward,
            'triplets': self.triplets,
            'trace': self.trace,
            'logs': self.logs,
            'metadata': self.rollout_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ApiKeyRotationPolicy(Base):
    """API key rotation policy model"""
    __tablename__ = 'api_key_rotation_policies'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)

    # Rotation settings
    auto_rotate_days = Column(Integer, nullable=False, default=90)
    notify_before_days = Column(Integer, default=7)
    grace_period_days = Column(Integer, default=30)

    # Security settings
    require_manual_acknowledgment = Column(Boolean, default=False)
    max_rotation_count = Column(Integer, default=100)

    # Status
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(String(255))

    def to_dict(self):
        """Convert rotation policy to dictionary"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'auto_rotate_days': self.auto_rotate_days,
            'notify_before_days': self.notify_before_days,
            'grace_period_days': self.grace_period_days,
            'require_manual_acknowledgment': self.require_manual_acknowledgment,
            'max_rotation_count': self.max_rotation_count,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by
        }


class ApiKeyRotationHistory(Base):
    """API key rotation history model"""
    __tablename__ = 'api_key_rotation_history'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key_id = Column(String(36), ForeignKey('api_keys.id', ondelete='CASCADE'), nullable=False, index=True)

    # Key information
    old_key_hash = Column(String(128))
    new_key_hash = Column(String(128))
    old_expires_at = Column(DateTime(timezone=True))
    new_expires_at = Column(DateTime(timezone=True))

    # Rotation details
    rotated_at = Column(DateTime(timezone=True), server_default=func.now())
    rotated_by = Column(String(255))
    rotation_reason = Column(String(20))  # Will be enum in PostgreSQL
    rotation_policy_id = Column(String(36), ForeignKey('api_key_rotation_policies.id'))

    # Notification tracking
    notification_sent_at = Column(DateTime(timezone=True))
    notification_status = Column(String(20), default='pending')
    user_acknowledged_at = Column(DateTime(timezone=True))

    # Additional metadata
    notes = Column(Text)
    rotation_metadata = Column('metadata', JSON, default={})

    def to_dict(self):
        """Convert rotation history to dictionary"""
        return {
            'id': str(self.id),
            'api_key_id': str(self.api_key_id),
            'old_key_hash': self.old_key_hash,
            'new_key_hash': self.new_key_hash,
            'old_expires_at': self.old_expires_at.isoformat() if self.old_expires_at else None,
            'new_expires_at': self.new_expires_at.isoformat() if self.new_expires_at else None,
            'rotated_at': self.rotated_at.isoformat() if self.rotated_at else None,
            'rotated_by': self.rotated_by,
            'rotation_reason': self.rotation_reason,
            'rotation_policy_id': str(self.rotation_policy_id) if self.rotation_policy_id else None,
            'notification_sent_at': self.notification_sent_at.isoformat() if self.notification_sent_at else None,
            'notification_status': self.notification_status,
            'user_acknowledged_at': self.user_acknowledged_at.isoformat() if self.user_acknowledged_at else None,
            'notes': self.notes,
            'metadata': self.rotation_metadata
        }


class ApiKey(Base):
    """API key model for server authentication"""
    __tablename__ = 'api_keys'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_hash = Column(String(128), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), index=True)
    permissions = Column(JSON, default=['read'])  # List of permissions
    is_active = Column(Boolean, default=True, index=True)
    expires_at = Column(DateTime(timezone=True))
    last_used_at = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    rate_limit_requests = Column(Integer, default=100)  # Requests per window
    rate_limit_window = Column(Integer, default=60)  # Window in seconds

    # Rotation fields
    rotation_policy_id = Column(String(36), ForeignKey('api_key_rotation_policies.id'))
    last_rotated_at = Column(DateTime(timezone=True))
    next_rotation_at = Column(DateTime(timezone=True))
    rotation_count = Column(Integer, default=0)
    is_rotation_enabled = Column(Boolean, default=True)
    rotation_locked = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        """Convert API key to dictionary (without sensitive data)"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'user_id': self.user_id,
            'permissions': self.permissions,
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'usage_count': self.usage_count,
            'rate_limit_requests': self.rate_limit_requests,
            'rate_limit_window': self.rate_limit_window,
            'rotation_policy_id': str(self.rotation_policy_id) if self.rotation_policy_id else None,
            'last_rotated_at': self.last_rotated_at.isoformat() if self.last_rotated_at else None,
            'next_rotation_at': self.next_rotation_at.isoformat() if self.next_rotation_at else None,
            'rotation_count': self.rotation_count,
            'is_rotation_enabled': self.is_rotation_enabled,
            'rotation_locked': self.rotation_locked,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def is_expired(self):
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def is_due_for_rotation(self):
        """Check if API key is due for rotation"""
        if not self.is_rotation_enabled or self.rotation_locked:
            return False
        if not self.next_rotation_at:
            return False
        return datetime.utcnow() >= self.next_rotation_at

    def can_make_request(self):
        """Check if rate limit allows another request"""
        # This is a simple check - in production you'd use Redis or similar
        return self.is_active and not self.is_expired()


class EncryptionKey(Base):
    """Encryption key model for data encryption"""
    __tablename__ = 'encryption_keys'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_id = Column(String(255), nullable=False, unique=True, index=True)
    key_type = Column(String(20), nullable=False, index=True)  # 'master', 'data', 'field', 'record'
    name = Column(String(255))
    description = Column(Text)

    # Key data (encrypted)
    encrypted_key = Column(LargeBinary, nullable=False)
    key_hash = Column(String(64), nullable=False)
    algorithm = Column(String(50), nullable=False, default='aes-256-gcm')

    # Key hierarchy
    parent_key_id = Column(String(36), ForeignKey('encryption_keys.id'))
    derived_from = Column(String(255))

    # Lifecycle
    status = Column(String(20), nullable=False, default='active', index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    activated_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True), index=True)
    destroyed_at = Column(DateTime(timezone=True))

    # Rotation
    rotation_count = Column(Integer, default=0)
    last_rotated_at = Column(DateTime(timezone=True))
    next_rotation_at = Column(DateTime(timezone=True), index=True)

    # Security
    compromise_detected_at = Column(DateTime(timezone=True))
    security_level = Column(String(20), default='standard')

    # Metadata
    tags = Column(JSON, default=list)
    key_metadata = Column('metadata', JSON, default=dict)
    created_by = Column(String(255))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        """Convert encryption key to dictionary"""
        return {
            'id': str(self.id),
            'key_id': self.key_id,
            'key_type': self.key_type,
            'name': self.name,
            'description': self.description,
            'algorithm': self.algorithm,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'destroyed_at': self.destroyed_at.isoformat() if self.destroyed_at else None,
            'rotation_count': self.rotation_count,
            'last_rotated_at': self.last_rotated_at.isoformat() if self.last_rotated_at else None,
            'next_rotation_at': self.next_rotation_at.isoformat() if self.next_rotation_at else None,
            'security_level': self.security_level,
            'tags': self.tags,
            'metadata': self.key_metadata,
            'created_by': self.created_by,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class KeyUsageLog(Base):
    """Audit log for encryption key usage"""
    __tablename__ = 'key_usage_log'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_id = Column(String(36), ForeignKey('encryption_keys.id', ondelete='CASCADE'), nullable=False, index=True)
    operation = Column(String(50), nullable=False, index=True)  # 'encrypt', 'decrypt', 'derive', 'rotate'
    field_name = Column(String(255))
    table_name = Column(String(255))
    record_id = Column(String(255))
    user_id = Column(String(255))
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text)
    performance_ms = Column(Integer)
    usage_metadata = Column('metadata', JSON, default=dict)

    def to_dict(self):
        """Convert usage log to dictionary"""
        return {
            'id': str(self.id),
            'key_id': str(self.key_id),
            'operation': self.operation,
            'field_name': self.field_name,
            'table_name': self.table_name,
            'record_id': self.record_id,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'success': self.success,
            'error_message': self.error_message,
            'performance_ms': self.performance_ms,
            'metadata': self.usage_metadata
        }


class KeyRotationHistory(Base):
    """History of key rotation events"""
    __tablename__ = 'key_rotation_history'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_id = Column(String(36), ForeignKey('encryption_keys.id', ondelete='CASCADE'), nullable=False, index=True)
    old_key_hash = Column(String(64))
    new_key_hash = Column(String(64))
    rotation_reason = Column(String(100))
    rotated_by = Column(String(255))
    rotated_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    rotation_time_ms = Column(Integer)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text)
    old_expires_at = Column(DateTime(timezone=True))
    new_expires_at = Column(DateTime(timezone=True))
    rotation_metadata = Column('metadata', JSON, default=dict)

    def to_dict(self):
        """Convert rotation history to dictionary"""
        return {
            'id': str(self.id),
            'key_id': str(self.key_id),
            'old_key_hash': self.old_key_hash,
            'new_key_hash': self.new_key_hash,
            'rotation_reason': self.rotation_reason,
            'rotated_by': self.rotated_by,
            'rotated_at': self.rotated_at.isoformat() if self.rotated_at else None,
            'rotation_time_ms': self.rotation_time_ms,
            'success': self.success,
            'error_message': self.error_message,
            'old_expires_at': self.old_expires_at.isoformat() if self.old_expires_at else None,
            'new_expires_at': self.new_expires_at.isoformat() if self.new_expires_at else None,
            'metadata': self.rotation_metadata
        }


class KeyAccessAudit(Base):
    """Audit trail for key access operations"""
    __tablename__ = 'key_access_audit'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key_id = Column(String(36), ForeignKey('encryption_keys.id', ondelete='CASCADE'), nullable=False, index=True)
    access_type = Column(String(50), nullable=False, index=True)  # 'retrieve', 'store', 'delete', 'rotate'
    accessor_id = Column(String(255))
    accessor_type = Column(String(50))  # 'service', 'user', 'system'
    ip_address = Column(String(45))
    user_agent = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text)
    access_metadata = Column('metadata', JSON, default=dict)

    def to_dict(self):
        """Convert access audit to dictionary"""
        return {
            'id': str(self.id),
            'key_id': str(self.key_id),
            'access_type': self.access_type,
            'accessor_id': self.accessor_id,
            'accessor_type': self.accessor_type,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.access_metadata
        }


class Event(Base):
    """Event model for event sourcing - captures all state changes"""
    __tablename__ = 'events'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(36), unique=True, index=True, nullable=False)
    aggregate_id = Column(String(36), nullable=False, index=True)  # Entity ID (agent_id, task_id, etc.)
    aggregate_type = Column(String(50), nullable=False, index=True)  # 'agent', 'task', 'workflow', 'resource', 'rollout'
    event_type = Column(String(50), nullable=False, index=True)  # 'created', 'updated', 'started', 'completed', etc.
    event_data = Column(JSON, nullable=False)  # Event payload
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)
    version = Column(Integer, nullable=False, default=1)  # Aggregate version for optimistic concurrency
    correlation_id = Column(String(36), index=True)  # For tracking related events
    causation_id = Column(String(36), index=True)  # ID of event that caused this event
    user_id = Column(String(36), index=True)  # User who triggered the event
    service_name = Column(String(50), index=True)  # Service that generated the event
    event_metadata = Column('metadata', JSON, default=dict)  # Additional event metadata

    def to_dict(self):
        """Convert event to dictionary"""
        return {
            'id': str(self.id),
            'event_id': self.event_id,
            'aggregate_id': self.aggregate_id,
            'aggregate_type': self.aggregate_type,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'version': self.version,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'user_id': self.user_id,
            'service_name': self.service_name,
            'metadata': self.event_metadata
        }


class EventSnapshot(Base):
    """Snapshot model for event sourcing - periodic state snapshots for performance"""
    __tablename__ = 'event_snapshots'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    aggregate_id = Column(String(36), nullable=False, index=True)
    aggregate_type = Column(String(50), nullable=False, index=True)
    snapshot_data = Column(JSON, nullable=False)  # Current state snapshot
    version = Column(Integer, nullable=False)  # Aggregate version at time of snapshot
    last_event_id = Column(String(36), nullable=False)  # ID of last event included in snapshot
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True))  # When this snapshot expires

    def to_dict(self):
        """Convert snapshot to dictionary"""
        return {
            'id': str(self.id),
            'aggregate_id': self.aggregate_id,
            'aggregate_type': self.aggregate_type,
            'snapshot_data': self.snapshot_data,
            'version': self.version,
            'last_event_id': self.last_event_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }