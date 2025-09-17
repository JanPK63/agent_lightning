"""
Unified Data Access Layer for Agent Lightning
Provides consistent data access across all microservices with caching and events
"""

import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from shared.database import db_manager
from shared.cache import get_cache
from shared.events import EventBus, EventChannel
from shared.cache_decorators import cached
from shared.models import Agent, Task, Knowledge, Workflow, Session, Metric, User

logger = logging.getLogger(__name__)

class DataAccessLayer:
    """Unified data access for all microservices"""
    
    def __init__(self, service_name: str):
        """Initialize data access layer
        
        Args:
            service_name: Name of the service using this DAL
        """
        self.service_name = service_name
        self.db = db_manager
        self.cache = get_cache()
        self.event_bus = EventBus(service_name)
        self.event_bus.start()
        logger.info(f"DataAccessLayer initialized for {service_name}")
    
    # ==================== Agent Operations ====================
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent with caching
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent dictionary or None if not found
        """
        # Try cache first
        cache_key = f"agent:{agent_id}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for agent {agent_id}")
            return cached
        
        # Load from database
        with self.db.get_db() as session:
            agent = session.query(Agent).filter(
                Agent.id == agent_id
            ).first()
            
            if agent:
                agent_dict = agent.to_dict()
                # Cache for 1 hour
                self.cache.set(cache_key, agent_dict, ttl=3600)
                logger.debug(f"Loaded agent {agent_id} from database")
                return agent_dict
        
        logger.warning(f"Agent {agent_id} not found")
        return None
    
    def list_agents(self) -> List[Dict]:
        """List all agents with caching
        
        Returns:
            List of agent dictionaries
        """
        cache_key = "agents:all"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for agent list ({len(cached)} agents)")
            return cached
        
        with self.db.get_db() as session:
            agents = session.query(Agent).all()
            agent_list = [a.to_dict() for a in agents]
            
            # Cache for 5 minutes
            self.cache.set(cache_key, agent_list, ttl=300)
            logger.info(f"Loaded {len(agent_list)} agents from database")
            return agent_list
    
    def create_agent(self, agent_data: Dict) -> Dict:
        """Create agent with cache invalidation
        
        Args:
            agent_data: Agent data dictionary
            
        Returns:
            Created agent dictionary
        """
        with self.db.get_db() as session:
            # Check if exists
            existing = session.query(Agent).filter(
                Agent.id == agent_data['id']
            ).first()
            
            if existing:
                raise ValueError(f"Agent {agent_data['id']} already exists")
            
            agent = Agent(**agent_data)
            session.add(agent)
            session.commit()
            
            agent_dict = agent.to_dict()
            
            # Update cache
            self.cache.set(f"agent:{agent.id}", agent_dict, ttl=3600)
            self.cache.delete("agents:all")  # Invalidate list
            
            # Emit event
            self.event_bus.emit(
                EventChannel.AGENT_CREATED,
                {"agent_id": agent.id, "agent": agent_dict}
            )
            
            logger.info(f"Created agent {agent.id}")
            return agent_dict
    
    def update_agent(self, agent_id: str, updates: Dict) -> Optional[Dict]:
        """Update agent with cache synchronization
        
        Args:
            agent_id: Agent identifier
            updates: Dictionary of updates
            
        Returns:
            Updated agent dictionary or None if not found
        """
        with self.db.get_db() as session:
            agent = session.query(Agent).filter(
                Agent.id == agent_id
            ).first()
            
            if not agent:
                logger.warning(f"Agent {agent_id} not found for update")
                return None
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            agent.updated_at = datetime.utcnow()
            session.commit()
            agent_dict = agent.to_dict()
            
            # Update cache
            self.cache.set(f"agent:{agent_id}", agent_dict, ttl=3600)
            self.cache.delete("agents:all")  # Invalidate list
            
            # Emit event
            self.event_bus.emit(
                EventChannel.AGENT_UPDATED,
                {"agent_id": agent_id, "updates": updates}
            )
            
            logger.info(f"Updated agent {agent_id}")
            return agent_dict
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent with cache invalidation
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self.db.get_db() as session:
            agent = session.query(Agent).filter(
                Agent.id == agent_id
            ).first()
            
            if not agent:
                logger.warning(f"Agent {agent_id} not found for deletion")
                return False
            
            session.delete(agent)
            session.commit()
            
            # Clear cache
            self.cache.delete(f"agent:{agent_id}")
            self.cache.delete("agents:all")
            
            # Emit event
            self.event_bus.emit(
                EventChannel.AGENT_DELETED,
                {"agent_id": agent_id}
            )
            
            logger.info(f"Deleted agent {agent_id}")
            return True
    
    def update_agent_status(self, agent_id: str, status: str) -> Optional[Dict]:
        """Update agent status
        
        Args:
            agent_id: Agent identifier
            status: New status
            
        Returns:
            Updated agent dictionary
        """
        agent = self.update_agent(agent_id, {"status": status})
        
        if agent:
            # Emit status change event
            self.event_bus.emit(
                EventChannel.AGENT_STATUS,
                {"agent_id": agent_id, "status": status}
            )
        
        return agent
    
    # ==================== Task Operations ====================
    
    def create_task(self, task_data: Dict) -> Dict:
        """Create task with event emission
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            Created task dictionary
        """
        with self.db.get_db() as session:
            # Generate UUID if not provided
            if 'id' not in task_data:
                task_data['id'] = str(uuid.uuid4())
            
            task = Task(**task_data)
            session.add(task)
            session.commit()
            
            task_dict = task.to_dict()
            
            # Cache task
            self.cache.set(f"task:{task.id}", task_dict, ttl=900)
            
            # Emit event
            self.event_bus.emit(
                EventChannel.TASK_CREATED,
                {"task_id": str(task.id), "task": task_dict}
            )
            
            logger.info(f"Created task {task.id}")
            return task_dict
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task with caching
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task dictionary or None if not found
        """
        # Try cache first
        cache_key = f"task:{task_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Load from database
        with self.db.get_db() as session:
            task = session.query(Task).filter(
                Task.id == task_id
            ).first()
            
            if task:
                task_dict = task.to_dict()
                # Cache for 15 minutes
                self.cache.set(cache_key, task_dict, ttl=900)
                return task_dict
        
        return None
    
    def list_tasks(self, agent_id: Optional[str] = None, 
                   status: Optional[str] = None) -> List[Dict]:
        """List tasks with filtering
        
        Args:
            agent_id: Filter by agent (optional)
            status: Filter by status (optional)
            
        Returns:
            List of task dictionaries
        """
        with self.db.get_db() as session:
            query = session.query(Task)
            
            if agent_id:
                query = query.filter(Task.agent_id == agent_id)
            if status:
                query = query.filter(Task.status == status)
            
            tasks = query.order_by(Task.created_at.desc()).all()
            return [t.to_dict() for t in tasks]
    
    def update_task_status(self, task_id: str, status: str, 
                          result: Optional[Dict] = None,
                          error_message: Optional[str] = None) -> Optional[Dict]:
        """Update task status with events
        
        Args:
            task_id: Task identifier
            status: New status
            result: Task result (optional)
            error_message: Error message if failed (optional)
            
        Returns:
            Updated task dictionary
        """
        with self.db.get_db() as session:
            task = session.query(Task).filter(
                Task.id == task_id
            ).first()
            
            if not task:
                logger.warning(f"Task {task_id} not found for status update")
                return None
            
            task.status = status
            if result:
                task.result = result
            if error_message:
                task.error_message = error_message
            
            # Update timestamps
            if status == "started":
                task.started_at = datetime.utcnow()
                event_channel = EventChannel.TASK_STARTED
            elif status == "completed":
                task.completed_at = datetime.utcnow()
                event_channel = EventChannel.TASK_COMPLETED
            elif status == "failed":
                task.completed_at = datetime.utcnow()
                event_channel = EventChannel.TASK_FAILED
            else:
                event_channel = EventChannel.TASK_PROGRESS
            
            session.commit()
            task_dict = task.to_dict()
            
            # Update cache
            self.cache.set(f"task:{task_id}", task_dict, ttl=900)
            
            # Store result in cache if completed
            if status == "completed" and result:
                self.cache.set(f"task:result:{task_id}", result, ttl=3600)
            
            # Emit event
            self.event_bus.emit(
                event_channel,
                {
                    "task_id": str(task_id),
                    "status": status,
                    "result": result,
                    "error": error_message
                }
            )
            
            logger.info(f"Updated task {task_id} status to {status}")
            return task_dict
    
    # ==================== Knowledge Operations ====================
    
    def add_knowledge(self, agent_id: str, knowledge_data: Dict) -> Dict:
        """Add knowledge item for agent
        
        Args:
            agent_id: Agent identifier
            knowledge_data: Knowledge data
            
        Returns:
            Created knowledge dictionary
        """
        with self.db.get_db() as session:
            # Generate ID if not provided
            if 'id' not in knowledge_data:
                knowledge_data['id'] = f"{agent_id}_{uuid.uuid4().hex[:8]}"
            
            knowledge_data['agent_id'] = agent_id
            knowledge = Knowledge(**knowledge_data)
            session.add(knowledge)
            session.commit()
            
            knowledge_dict = knowledge.to_dict()
            
            # Cache knowledge
            cache_key = f"knowledge:{agent_id}:{knowledge.id}"
            self.cache.set(cache_key, knowledge_dict, ttl=21600)  # 6 hours
            
            # Invalidate agent knowledge list
            self.cache.delete(f"knowledge:{agent_id}:all")
            
            logger.info(f"Added knowledge {knowledge.id} for agent {agent_id}")
            return knowledge_dict
    
    def get_agent_knowledge(self, agent_id: str, 
                           category: Optional[str] = None) -> List[Dict]:
        """Get knowledge for agent
        
        Args:
            agent_id: Agent identifier
            category: Filter by category (optional)
            
        Returns:
            List of knowledge dictionaries
        """
        cache_key = f"knowledge:{agent_id}:all"
        if not category:
            # Try cache for full list
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        with self.db.get_db() as session:
            query = session.query(Knowledge).filter(
                Knowledge.agent_id == agent_id
            )
            
            if category:
                query = query.filter(Knowledge.category == category)
            
            knowledge_items = query.order_by(
                Knowledge.relevance_score.desc()
            ).all()
            
            result = [k.to_dict() for k in knowledge_items]
            
            if not category:
                # Cache full list
                self.cache.set(cache_key, result, ttl=1800)  # 30 minutes
            
            return result
    
    def update_knowledge_usage(self, knowledge_id: str) -> Optional[Dict]:
        """Update knowledge usage count
        
        Args:
            knowledge_id: Knowledge identifier
            
        Returns:
            Updated knowledge dictionary
        """
        with self.db.get_db() as session:
            knowledge = session.query(Knowledge).filter(
                Knowledge.id == knowledge_id
            ).first()
            
            if knowledge:
                knowledge.usage_count += 1
                knowledge.last_used_at = datetime.utcnow()
                session.commit()
                
                knowledge_dict = knowledge.to_dict()
                
                # Update cache
                cache_key = f"knowledge:{knowledge.agent_id}:{knowledge.id}"
                self.cache.set(cache_key, knowledge_dict, ttl=21600)
                
                return knowledge_dict
        
        return None
    
    # ==================== Workflow Operations ====================
    
    def create_workflow(self, workflow_data: Dict) -> Dict:
        """Create workflow
        
        Args:
            workflow_data: Workflow data
            
        Returns:
            Created workflow dictionary
        """
        with self.db.get_db() as session:
            workflow = Workflow(**workflow_data)
            session.add(workflow)
            session.commit()
            
            workflow_dict = workflow.to_dict()
            
            # Cache workflow
            self.cache.set(f"workflow:{workflow.id}", workflow_dict, ttl=1800)
            
            # Emit event
            self.event_bus.emit(
                EventChannel.WORKFLOW_STARTED,
                {"workflow_id": str(workflow.id), "workflow": workflow_dict}
            )
            
            logger.info(f"Created workflow {workflow.id}")
            return workflow_dict
    
    def update_workflow_status(self, workflow_id: str, 
                              status: str, step: Optional[int] = None) -> Optional[Dict]:
        """Update workflow status
        
        Args:
            workflow_id: Workflow identifier
            status: New status
            step: Current step (optional)
            
        Returns:
            Updated workflow dictionary
        """
        with self.db.get_db() as session:
            workflow = session.query(Workflow).filter(
                Workflow.id == workflow_id
            ).first()
            
            if not workflow:
                return None
            
            workflow.status = status
            workflow.updated_at = datetime.utcnow()
            
            if step is not None:
                # Update step in context
                if not workflow.context:
                    workflow.context = {}
                workflow.context['current_step'] = step
            
            session.commit()
            workflow_dict = workflow.to_dict()
            
            # Update cache
            self.cache.set(f"workflow:{workflow_id}", workflow_dict, ttl=1800)
            
            # Emit appropriate event
            if status == "completed":
                event_channel = EventChannel.WORKFLOW_COMPLETED
            elif status == "failed":
                event_channel = EventChannel.WORKFLOW_FAILED
            else:
                event_channel = EventChannel.WORKFLOW_STEP
            
            self.event_bus.emit(
                event_channel,
                {
                    "workflow_id": str(workflow_id),
                    "status": status,
                    "step": step
                }
            )
            
            return workflow_dict
    
    # ==================== Session Operations ====================
    
    def create_session(self, user_id: str, token: str, data: Dict = None) -> Dict:
        """Create user session
        
        Args:
            user_id: User identifier
            token: Session token
            data: Session data
            
        Returns:
            Created session dictionary
        """
        from datetime import timedelta
        
        with self.db.get_db() as session_db:
            session = Session(
                user_id=user_id,
                token=token,
                data=data or {},
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            session_db.add(session)
            session_db.commit()
            
            session_dict = session.to_dict()
            
            # Cache session
            self.cache.set(f"session:{token}", session_dict, ttl=86400)  # 24 hours
            
            return session_dict
    
    def get_session(self, token: str) -> Optional[Dict]:
        """Get session by token
        
        Args:
            token: Session token
            
        Returns:
            Session dictionary or None if not found/expired
        """
        # Try cache first
        cache_key = f"session:{token}"
        cached = self.cache.get(cache_key)
        if cached:
            # Check expiration
            expires_at = datetime.fromisoformat(cached['expires_at'])
            if expires_at > datetime.utcnow():
                return cached
            else:
                # Session expired, delete from cache
                self.cache.delete(cache_key)
        
        # Load from database
        with self.db.get_db() as session_db:
            session = session_db.query(Session).filter(
                Session.token == token
            ).first()
            
            if session and not session.is_expired():
                session_dict = session.to_dict()
                # Re-cache
                self.cache.set(cache_key, session_dict, ttl=86400)
                return session_dict
        
        return None
    
    def delete_session(self, token: str) -> bool:
        """Delete session
        
        Args:
            token: Session token
            
        Returns:
            True if deleted
        """
        with self.db.get_db() as session_db:
            session = session_db.query(Session).filter(
                Session.token == token
            ).first()
            
            if session:
                session_db.delete(session)
                session_db.commit()
                
                # Clear cache
                self.cache.delete(f"session:{token}")
                
                return True
        
        return False
    
    # ==================== Metrics Operations ====================
    
    def record_metric(self, metric_name: str, value: float, tags: Dict = None):
        """Record performance metric
        
        Args:
            metric_name: Metric name
            value: Metric value
            tags: Optional tags
        """
        with self.db.get_db() as session:
            metric = Metric(
                service_name=self.service_name,
                metric_name=metric_name,
                value=value,
                tags=tags or {}
            )
            session.add(metric)
            session.commit()
            
            # Cache recent metric
            cache_key = f"metrics:{self.service_name}:{metric_name}"
            self.cache.set(cache_key, value, ttl=300)  # 5 minutes
            
            # Emit metric event
            self.event_bus.emit(
                EventChannel.SYSTEM_METRICS,
                {
                    "service": self.service_name,
                    "metric": metric_name,
                    "value": value,
                    "tags": tags
                }
            )
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   minutes: int = 60) -> List[Dict]:
        """Get recent metrics
        
        Args:
            metric_name: Filter by metric name (optional)
            minutes: Time window in minutes
            
        Returns:
            List of metric dictionaries
        """
        from datetime import timedelta
        
        with self.db.get_db() as session:
            since = datetime.utcnow() - timedelta(minutes=minutes)
            
            query = session.query(Metric).filter(
                Metric.service_name == self.service_name,
                Metric.timestamp >= since
            )
            
            if metric_name:
                query = query.filter(Metric.metric_name == metric_name)
            
            metrics = query.order_by(Metric.timestamp.desc()).all()
            return [m.to_dict() for m in metrics]
    
    # ==================== Transaction Management ====================
    
    @contextmanager
    def distributed_transaction(self, timeout: int = 30):
        """Manage distributed transactions
        
        Args:
            timeout: Transaction timeout in seconds
            
        Yields:
            Transaction ID
        """
        tx_id = str(uuid.uuid4())
        
        try:
            # Begin transaction
            logger.info(f"Starting distributed transaction {tx_id}")
            
            # Acquire distributed lock
            with self.cache.lock(f"tx:{tx_id}", timeout=timeout):
                yield tx_id
            
            # Commit successful
            logger.info(f"Transaction {tx_id} committed")
            
        except Exception as e:
            # Rollback on error
            logger.error(f"Transaction {tx_id} failed: {e}")
            
            # Emit rollback event
            self.event_bus.emit(EventChannel.SYSTEM_ALERT, {
                "type": "transaction_failed",
                "tx_id": tx_id,
                "error": str(e),
                "service": self.service_name
            })
            raise
    
    # ==================== Health & Cleanup ====================
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email address from database
        
        Args:
            email: User email address
            
        Returns:
            User dictionary or None if not found
        """
        with self.db.get_db() as session:
            user = session.query(User).filter(
                User.email == email
            ).first()
            
            if user:
                # Return full user data including password_hash for authentication
                return {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'password_hash': user.password_hash,
                    'role': user.role,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'updated_at': user.updated_at.isoformat() if user.updated_at else None
                }
            
            return None
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components
        
        Returns:
            Health status dictionary
        """
        return {
            "database": self.db.health_check(),
            "cache": self.cache.health_check(),
            "service": self.service_name,
            "status": "healthy"
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.event_bus.stop()
            logger.info(f"DataAccessLayer cleanup complete for {self.service_name}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")