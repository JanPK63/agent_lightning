class WorkflowQueue:
    """Redis-based workflow execution queue"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.queue_key = "workflow:queue"
        self.processing_key = "workflow:processing"

    def enqueue(self, workflow_id: str, priority: int = 5):
        """Add workflow to execution queue"""
        score = time.time() - (priority * 1000)  # Higher priority = lower score
        self.redis.zadd(self.queue_key, {workflow_id: score})
        logger.info(f"Enqueued workflow {workflow_id} with priority {priority}")

    def dequeue(self) -> Optional[str]:
        """Get next workflow to execute"""
        # Atomic move from queue to processing
        result = self.redis.zpopmin(self.queue_key)
        if result:
            workflow_id = result[0][0]
            self.redis.hset(self.processing_key, workflow_id, time.time())
            logger.info(f"Dequeued workflow {workflow_id} for processing")
            return workflow_id
        return None

    def complete(self, workflow_id: str):
        """Mark workflow as completed"""
        self.redis.hdel(self.processing_key, workflow_id)
        logger.info(f"Workflow {workflow_id} removed from processing")

    def get_queue_size(self) -> int:
        """Get number of workflows in queue"""
        return self.redis.zcard(self.queue_key)

    def get_processing(self) -> List[str]:
        """Get workflows currently being processed"""
        return list(self.redis.hkeys(self.processing_key))


class WorkflowRecovery:
    """Handles workflow recovery after failures"""

    def __init__(self, dal: DataAccessLayer, queue: WorkflowQueue):
        self.dal = dal
        self.queue = queue

    async def recover_interrupted_workflows(self):
        """Recover workflows interrupted by service restart"""
        logger.info("Starting workflow recovery...")

        # Find all workflows that were running
        with self.dal.db.get_db() as session:
            from shared.models import Workflow
            interrupted = session.query(Workflow).filter(
                Workflow.status == WorkflowStatus.RUNNING.value
            ).all()

            recovered = 0
            for workflow in interrupted:
                workflow_id = str(workflow.id)
                logger.info(f"Recovering workflow {workflow_id}")

                # Check last checkpoint
                last_checkpoint = workflow.context.get('checkpoint') if workflow.context else None

                if last_checkpoint:
                    # Resume from checkpoint
                    await self.resume_from_checkpoint(workflow_id, last_checkpoint)
                else:
                    # Re-enqueue workflow
                    self.queue.enqueue(workflow_id, priority=10)  # High priority for recovery

                recovered += 1

            logger.info(f"Recovered {recovered} interrupted workflows")

    async def resume_from_checkpoint(self, workflow_id: str, checkpoint: Dict):
        """Resume workflow from checkpoint"""
        logger.info(f"Resuming workflow {workflow_id} from checkpoint: {checkpoint}")
        # Implementation would restore workflow state and continue execution
        self.queue.enqueue(workflow_id, priority=10)