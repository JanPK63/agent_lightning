"""
MongoDB Data Access Layer for Agent Lightning
Provides MongoDB-compatible implementation of DataAccessLayer interface
"""

import uuid
import logging
from typing import Optional, List, Dict
from datetime import datetime

from shared.mongodb import mongodb_manager, document_storage
from shared.cache import get_cache
from shared.events import EventBus, EventChannel

logger = logging.getLogger(__name__)


class MongoDBDataAccessLayer:
    """MongoDB implementation of Data Access Layer"""

    def __init__(self, service_name: str):
        """Initialize MongoDB data access layer

        Args:
            service_name: Name of the service using this DAL
        """
        self.service_name = service_name
        self.mongodb = mongodb_manager
        self.document_storage = document_storage
        self.cache = get_cache()
        self.event_bus = EventBus(service_name)
        self.event_bus.start()
        logger.info(f"MongoDBDataAccessLayer initialized for {service_name}")

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

        # Load from MongoDB
        agent = self.document_storage.find_document("agents", {"id": agent_id})
        if agent:
            # Cache for 1 hour
            self.cache.set(cache_key, agent, ttl=3600)
            logger.debug(f"Loaded agent {agent_id} from MongoDB")
            return agent

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

        agents = self.document_storage.find_documents("agents", {})
        # Cache for 5 minutes
        self.cache.set(cache_key, agents, ttl=300)
        logger.info(f"Loaded {len(agents)} agents from MongoDB")
        return agents

    def create_agent(self, agent_data: Dict) -> Dict:
        """Create agent with cache invalidation

        Args:
            agent_data: Agent data dictionary

        Returns:
            Created agent dictionary
        """
        # Check if exists
        existing = self.document_storage.find_document("agents", {"id": agent_data['id']})
        if existing:
            raise ValueError(f"Agent {agent_data['id']} already exists")

        # Add timestamps
        agent_data['created_at'] = datetime.utcnow().isoformat()
        agent_data['updated_at'] = datetime.utcnow().isoformat()

        # Insert into MongoDB
        doc_id = self.document_storage.insert_document("agents", agent_data)
        if not doc_id:
            raise RuntimeError("Failed to create agent in MongoDB")

        agent_dict = agent_data.copy()
        agent_dict['_id'] = doc_id

        # Update cache
        self.cache.set(f"agent:{agent_data['id']}", agent_dict, ttl=3600)
        self.cache.delete("agents:all")  # Invalidate list

        # Emit event
        self.event_bus.emit(
            EventChannel.AGENT_CREATED,
            {"agent_id": agent_data['id'], "agent": agent_dict}
        )

        logger.info(f"Created agent {agent_data['id']}")
        return agent_dict

    def update_agent(self, agent_id: str, updates: Dict) -> Optional[Dict]:
        """Update agent with cache synchronization

        Args:
            agent_id: Agent identifier
            updates: Dictionary of updates

        Returns:
            Updated agent dictionary or None if not found
        """
        # Get current agent
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found for update")
            return None

        # Apply updates
        agent.update(updates)
        agent['updated_at'] = datetime.utcnow().isoformat()

        # Update in MongoDB
        success = self.document_storage.update_document("agents", {"id": agent_id}, agent)
        if not success:
            logger.error(f"Failed to update agent {agent_id} in MongoDB")
            return None

        # Update cache
        self.cache.set(f"agent:{agent_id}", agent, ttl=3600)
        self.cache.delete("agents:all")  # Invalidate list

        # Emit event
        self.event_bus.emit(
            EventChannel.AGENT_UPDATED,
            {"agent_id": agent_id, "updates": updates}
        )

        logger.info(f"Updated agent {agent_id}")
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent with cache invalidation

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted, False if not found
        """
        # Check if exists
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found for deletion")
            return False

        # Delete from MongoDB
        success = self.document_storage.delete_document("agents", {"id": agent_id})
        if success:
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

        return False

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
        # Generate UUID if not provided
        if 'id' not in task_data:
            task_data['id'] = str(uuid.uuid4())

        # Add timestamps
        task_data['created_at'] = datetime.utcnow().isoformat()

        # Insert into MongoDB
        doc_id = self.document_storage.insert_document("tasks", task_data)
        if not doc_id:
            raise RuntimeError("Failed to create task in MongoDB")

        task_dict = task_data.copy()
        task_dict['_id'] = doc_id

        # Cache task
        self.cache.set(f"task:{task_data['id']}", task_dict, ttl=900)

        # Emit event
        self.event_bus.emit(
            EventChannel.TASK_CREATED,
            {"task_id": str(task_data['id']), "task": task_dict}
        )

        logger.info(f"Created task {task_data['id']}")
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

        # Load from MongoDB
        task = self.document_storage.find_document("tasks", {"id": task_id})
        if task:
            # Cache for 15 minutes
            self.cache.set(cache_key, task, ttl=900)
            return task

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
        query = {}
        if agent_id:
            query['agent_id'] = agent_id
        if status:
            query['status'] = status

        tasks = self.document_storage.find_documents("tasks", query)
        # Sort by created_at descending (most recent first)
        tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return tasks

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
        # Get current task
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found for status update")
            return None

        # Apply updates
        updates = {"status": status}
        if result:
            updates['result'] = result
        if error_message:
            updates['error_message'] = error_message

        # Update timestamps
        if status == "started":
            updates['started_at'] = datetime.utcnow().isoformat()
            event_channel = EventChannel.TASK_STARTED
        elif status in ["completed", "failed"]:
            updates['completed_at'] = datetime.utcnow().isoformat()
            event_channel = EventChannel.TASK_COMPLETED if status == "completed" else EventChannel.TASK_FAILED
        else:
            event_channel = EventChannel.TASK_PROGRESS

        # Update in MongoDB
        success = self.document_storage.update_document("tasks", {"id": task_id}, updates)
        if not success:
            logger.error(f"Failed to update task {task_id} in MongoDB")
            return None

        # Update task dict
        task.update(updates)

        # Update cache
        self.cache.set(f"task:{task_id}", task, ttl=900)

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
        return task

    # ==================== Knowledge Operations ====================

    def add_knowledge(self, agent_id: str, knowledge_data: Dict) -> Dict:
        """Add knowledge item for agent

        Args:
            agent_id: Agent identifier
            knowledge_data: Knowledge data

        Returns:
            Created knowledge dictionary
        """
        # Generate ID if not provided
        if 'id' not in knowledge_data:
            knowledge_data['id'] = f"{agent_id}_{uuid.uuid4().hex[:8]}"

        knowledge_data['agent_id'] = agent_id
        knowledge_data['created_at'] = datetime.utcnow().isoformat()

        # Insert into MongoDB
        doc_id = self.document_storage.insert_document("knowledge", knowledge_data)
        if not doc_id:
            raise RuntimeError("Failed to create knowledge in MongoDB")

        knowledge_dict = knowledge_data.copy()
        knowledge_dict['_id'] = doc_id

        # Cache knowledge
        cache_key = f"knowledge:{agent_id}:{knowledge_data['id']}"
        self.cache.set(cache_key, knowledge_dict, ttl=21600)  # 6 hours

        # Invalidate agent knowledge list
        self.cache.delete(f"knowledge:{agent_id}:all")

        logger.info(f"Added knowledge {knowledge_data['id']} for agent {agent_id}")
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

        query = {"agent_id": agent_id}
        if category:
            query['category'] = category

        knowledge_items = self.document_storage.find_documents("knowledge", query)

        # Sort by relevance score descending
        knowledge_items.sort(key=lambda x: x.get('relevance_score', 1.0), reverse=True)

        if not category:
            # Cache full list
            self.cache.set(cache_key, knowledge_items, ttl=1800)  # 30 minutes

        return knowledge_items

    def update_knowledge_usage(self, knowledge_id: str) -> Optional[Dict]:
        """Update knowledge usage count

        Args:
            knowledge_id: Knowledge identifier

        Returns:
            Updated knowledge dictionary
        """
        # Find knowledge item
        knowledge = self.document_storage.find_document("knowledge", {"id": knowledge_id})
        if not knowledge:
            return None

        # Update usage
        updates = {
            "usage_count": knowledge.get('usage_count', 0) + 1,
            "last_used_at": datetime.utcnow().isoformat()
        }

        success = self.document_storage.update_document("knowledge", {"id": knowledge_id}, updates)
        if success:
            knowledge.update(updates)

            # Update cache
            cache_key = f"knowledge:{knowledge['agent_id']}:{knowledge_id}"
            self.cache.set(cache_key, knowledge, ttl=21600)

            return knowledge

        return None

    # ==================== Health & Cleanup ====================

    def health_check(self) -> Dict[str, bool]:
        """Check health of all components

        Returns:
            Health status dictionary
        """
        return {
            "mongodb": self.mongodb.health_check(),
            "cache": self.cache.health_check(),
            "service": self.service_name,
            "status": "healthy"
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.event_bus.stop()
            logger.info(f"MongoDBDataAccessLayer cleanup complete for {self.service_name}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")