#!/usr/bin/env python3
"""
Batch Accumulator Service - Optimizes long interactions through intelligent batching
Implements adaptive batching, queue management, and performance optimization
"""

import os
import sys
import json
import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import numpy as np
import logging
from asyncio import Queue, QueueFull
import heapq

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_access import DataAccessLayer
from shared.events import EventChannel
from shared.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchStrategy(str, Enum):
    """Batching strategies"""
    TIME_BASED = "time_based"          # Batch by time window
    SIZE_BASED = "size_based"          # Batch by number of items
    MEMORY_BASED = "memory_based"      # Batch by memory usage
    ADAPTIVE = "adaptive"              # Dynamically adjust strategy
    PRIORITY = "priority"              # Priority-based batching
    SEMANTIC = "semantic"              # Group similar requests


class BatchStatus(str, Enum):
    """Batch processing status"""
    ACCUMULATING = "accumulating"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchConfig:
    """Batch configuration"""
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    max_batch_size: int = 32
    max_wait_time_ms: int = 100
    max_memory_mb: int = 512
    priority_levels: int = 3
    enable_compression: bool = True
    enable_deduplication: bool = True
    enable_prefetching: bool = True


@dataclass
class BatchItem:
    """Individual item in a batch"""
    id: str
    agent_id: str
    request_type: str
    payload: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority > other.priority  # Higher priority first


@dataclass
class Batch:
    """Batch container"""
    id: str
    strategy: BatchStrategy
    status: BatchStatus
    items: List[BatchItem]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return len(self.items)
    
    @property
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        return sum(
            len(json.dumps(item.payload).encode()) 
            for item in self.items
        )


class AdaptiveBatchOptimizer:
    """Adaptive optimization for batch parameters"""
    
    def __init__(self, initial_config: BatchConfig):
        self.config = initial_config
        self.performance_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
        # Adaptive parameters
        self.current_batch_size = initial_config.max_batch_size
        self.current_wait_time = initial_config.max_wait_time_ms
        
        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.beta = 0.9   # Momentum
        self.gamma = 0.95 # Discount factor
        
    def record_performance(
        self, 
        batch_size: int,
        processing_time_ms: float,
        success_rate: float
    ):
        """Record batch performance metrics"""
        throughput = batch_size / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
        
        self.performance_history.append({
            "batch_size": batch_size,
            "processing_time": processing_time_ms,
            "success_rate": success_rate,
            "throughput": throughput,
            "timestamp": datetime.now()
        })
        
        self.latency_history.append(processing_time_ms / batch_size if batch_size > 0 else 0)
        self.throughput_history.append(throughput)
        
        # Adapt parameters
        self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adaptively adjust batch parameters based on performance"""
        if len(self.performance_history) < 10:
            return
        
        # Calculate performance trends
        recent_perf = list(self.performance_history)[-10:]
        avg_throughput = np.mean([p["throughput"] for p in recent_perf])
        avg_latency = np.mean([p["processing_time"] / p["batch_size"] for p in recent_perf if p["batch_size"] > 0])
        success_rate = np.mean([p["success_rate"] for p in recent_perf])
        
        # Adjust batch size
        if success_rate < 0.95:  # If failures are high, reduce batch size
            self.current_batch_size = max(1, int(self.current_batch_size * 0.9))
        elif avg_latency > 100:  # If latency is high, reduce batch size
            self.current_batch_size = max(1, int(self.current_batch_size * 0.95))
        else:  # Otherwise, try to increase for better throughput
            self.current_batch_size = min(
                self.config.max_batch_size * 2,
                int(self.current_batch_size * 1.05)
            )
        
        # Adjust wait time
        if avg_throughput < 10:  # Low throughput, reduce wait time
            self.current_wait_time = max(10, int(self.current_wait_time * 0.9))
        else:
            self.current_wait_time = min(1000, int(self.current_wait_time * 1.05))
        
        logger.info(
            f"Adapted parameters: batch_size={self.current_batch_size}, "
            f"wait_time={self.current_wait_time}ms"
        )
    
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.current_batch_size
    
    def get_optimal_wait_time(self) -> int:
        """Get current optimal wait time"""
        return self.current_wait_time


class BatchAccumulator:
    """Main batch accumulation system"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.optimizer = AdaptiveBatchOptimizer(config)
        
        # Batch queues by agent
        self.agent_queues: Dict[str, asyncio.Queue] = {}
        self.priority_queues: Dict[str, List[BatchItem]] = defaultdict(list)
        
        # Active batches
        self.active_batches: Dict[str, Batch] = {}
        
        # Deduplication cache
        self.dedup_cache = {}
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "total_batches": 0,
            "items_processed": 0,
            "items_failed": 0,
            "total_wait_time": 0,
            "total_processing_time": 0
        }
        
        # Background tasks
        self.running = False
        self.background_tasks = []
    
    async def start(self):
        """Start the batch accumulator"""
        self.running = True
        
        # Start background processors
        self.background_tasks = [
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._timeout_monitor()),
            asyncio.create_task(self._stats_reporter())
        ]
        
        logger.info("Batch accumulator started")
    
    async def stop(self):
        """Stop the batch accumulator"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Batch accumulator stopped")
    
    async def add_item(self, item: BatchItem) -> str:
        """Add an item to be batched"""
        # Deduplication
        if self.config.enable_deduplication:
            item_hash = self._hash_item(item)
            if item_hash in self.dedup_cache:
                logger.debug(f"Duplicate item detected: {item.id}")
                return self.dedup_cache[item_hash]
            self.dedup_cache[item_hash] = item.id
        
        # Get or create queue for agent
        if item.agent_id not in self.agent_queues:
            self.agent_queues[item.agent_id] = asyncio.Queue(maxsize=1000)
        
        # Add to appropriate queue based on strategy
        if self.config.strategy == BatchStrategy.PRIORITY:
            heapq.heappush(self.priority_queues[item.agent_id], item)
        else:
            try:
                self.agent_queues[item.agent_id].put_nowait(item)
            except QueueFull:
                raise ValueError(f"Queue full for agent {item.agent_id}")
        
        self.stats["total_items"] += 1
        
        # Trigger batch creation if needed
        await self._check_batch_ready(item.agent_id)
        
        return item.id
    
    async def _check_batch_ready(self, agent_id: str):
        """Check if a batch is ready to process"""
        if self.config.strategy == BatchStrategy.SIZE_BASED:
            await self._check_size_based_batch(agent_id)
        elif self.config.strategy == BatchStrategy.TIME_BASED:
            await self._check_time_based_batch(agent_id)
        elif self.config.strategy == BatchStrategy.MEMORY_BASED:
            await self._check_memory_based_batch(agent_id)
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            await self._check_adaptive_batch(agent_id)
        elif self.config.strategy == BatchStrategy.PRIORITY:
            await self._check_priority_batch(agent_id)
    
    async def _check_size_based_batch(self, agent_id: str):
        """Check if size threshold is met"""
        if agent_id not in self.agent_queues:
            return
        
        queue = self.agent_queues[agent_id]
        if queue.qsize() >= self.config.max_batch_size:
            await self._create_batch(agent_id)
    
    async def _check_adaptive_batch(self, agent_id: str):
        """Use adaptive optimization to determine when to batch"""
        if agent_id not in self.agent_queues:
            return
        
        queue = self.agent_queues[agent_id]
        optimal_size = self.optimizer.get_optimal_batch_size()
        
        if queue.qsize() >= optimal_size:
            await self._create_batch(agent_id)
    
    async def _check_priority_batch(self, agent_id: str):
        """Check priority queue for high-priority items"""
        if agent_id not in self.priority_queues:
            return
        
        pq = self.priority_queues[agent_id]
        
        # Process immediately if high-priority item exists
        if pq and pq[0].priority >= 3:
            await self._create_batch(agent_id)
        # Or if we have enough items
        elif len(pq) >= self.config.max_batch_size:
            await self._create_batch(agent_id)
    
    async def _create_batch(self, agent_id: str) -> Optional[Batch]:
        """Create a batch from queued items"""
        items = []
        
        if self.config.strategy == BatchStrategy.PRIORITY:
            # Get items from priority queue
            pq = self.priority_queues[agent_id]
            batch_size = min(len(pq), self.config.max_batch_size)
            
            for _ in range(batch_size):
                if pq:
                    items.append(heapq.heappop(pq))
        else:
            # Get items from regular queue
            queue = self.agent_queues.get(agent_id)
            if not queue:
                return None
            
            batch_size = min(queue.qsize(), self.config.max_batch_size)
            
            for _ in range(batch_size):
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.01)
                    items.append(item)
                except asyncio.TimeoutError:
                    break
        
        if not items:
            return None
        
        # Create batch
        batch = Batch(
            id=str(uuid.uuid4()),
            strategy=self.config.strategy,
            status=BatchStatus.READY,
            items=items,
            created_at=datetime.now(),
            metadata={
                "agent_id": agent_id,
                "batch_size": len(items),
                "optimizer_state": {
                    "optimal_size": self.optimizer.get_optimal_batch_size(),
                    "optimal_wait": self.optimizer.get_optimal_wait_time()
                }
            }
        )
        
        self.active_batches[batch.id] = batch
        self.stats["total_batches"] += 1
        
        logger.info(f"Created batch {batch.id} with {len(items)} items for agent {agent_id}")
        
        return batch
    
    async def _batch_processor(self):
        """Background task to process ready batches"""
        while self.running:
            try:
                # Find ready batches
                ready_batches = [
                    batch for batch in self.active_batches.values()
                    if batch.status == BatchStatus.READY
                ]
                
                for batch in ready_batches:
                    asyncio.create_task(self._process_batch(batch))
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: Batch):
        """Process a batch of items"""
        batch.status = BatchStatus.PROCESSING
        batch.started_at = datetime.now()
        start_time = time.time()
        
        try:
            # Group items by request type for efficient processing
            grouped = defaultdict(list)
            for item in batch.items:
                grouped[item.request_type].append(item)
            
            results = {}
            failures = 0
            
            # Process each group
            for request_type, items in grouped.items():
                try:
                    # Batch process items of the same type
                    group_results = await self._process_group(request_type, items)
                    results.update(group_results)
                except Exception as e:
                    logger.error(f"Failed to process group {request_type}: {e}")
                    failures += len(items)
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            success_rate = 1.0 - (failures / len(batch.items))
            
            # Record performance
            self.optimizer.record_performance(
                batch_size=len(batch.items),
                processing_time_ms=processing_time,
                success_rate=success_rate
            )
            
            # Update batch status
            batch.status = BatchStatus.COMPLETED if failures == 0 else BatchStatus.PARTIAL
            batch.completed_at = datetime.now()
            batch.metadata["results"] = results
            batch.metadata["processing_time_ms"] = processing_time
            batch.metadata["success_rate"] = success_rate
            
            # Update statistics
            self.stats["items_processed"] += len(batch.items) - failures
            self.stats["items_failed"] += failures
            self.stats["total_processing_time"] += processing_time
            
            logger.info(
                f"Processed batch {batch.id}: {len(batch.items)} items, "
                f"{processing_time:.2f}ms, success_rate={success_rate:.2%}"
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            batch.status = BatchStatus.FAILED
            batch.metadata["error"] = str(e)
    
    async def _process_group(
        self, 
        request_type: str, 
        items: List[BatchItem]
    ) -> Dict[str, Any]:
        """Process a group of items of the same type"""
        # This would integrate with your actual processing logic
        # For now, simulate processing
        await asyncio.sleep(0.01 * len(items))  # Simulate processing time
        
        results = {}
        for item in items:
            results[item.id] = {
                "status": "completed",
                "result": f"Processed {request_type} for {item.agent_id}"
            }
        
        return results
    
    async def _timeout_monitor(self):
        """Monitor for batch timeouts"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check each agent's queue
                for agent_id in list(self.agent_queues.keys()):
                    queue = self.agent_queues[agent_id]
                    
                    if queue.qsize() > 0:
                        # Check if we should create a batch due to timeout
                        # This is simplified - in practice, track oldest item time
                        await self._create_batch(agent_id)
                
                await asyncio.sleep(self.config.max_wait_time_ms / 1000)
                
            except Exception as e:
                logger.error(f"Timeout monitor error: {e}")
                await asyncio.sleep(1)
    
    async def _stats_reporter(self):
        """Periodically report statistics"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                if self.stats["total_items"] > 0:
                    avg_wait = self.stats["total_wait_time"] / self.stats["total_items"]
                    avg_processing = self.stats["total_processing_time"] / max(1, self.stats["items_processed"])
                    
                    logger.info(
                        f"Batch Stats: Items={self.stats['total_items']}, "
                        f"Batches={self.stats['total_batches']}, "
                        f"Processed={self.stats['items_processed']}, "
                        f"Failed={self.stats['items_failed']}, "
                        f"Avg Wait={avg_wait:.2f}ms, "
                        f"Avg Processing={avg_processing:.2f}ms"
                    )
                
            except Exception as e:
                logger.error(f"Stats reporter error: {e}")
    
    def _hash_item(self, item: BatchItem) -> str:
        """Generate hash for deduplication"""
        content = f"{item.agent_id}:{item.request_type}:{json.dumps(item.payload, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            "active_batches": len(self.active_batches),
            "queued_items": sum(q.qsize() for q in self.agent_queues.values()),
            "optimizer_state": {
                "current_batch_size": self.optimizer.get_optimal_batch_size(),
                "current_wait_time": self.optimizer.get_optimal_wait_time()
            }
        }


class BatchAccumulatorService:
    """FastAPI service for batch accumulation"""
    
    def __init__(self):
        self.app = FastAPI(title="Batch Accumulator Service", version="1.0.0")
        
        # Initialize Data Access Layer
        self.dal = DataAccessLayer("batch_accumulator")
        
        # Cache
        self.cache = get_cache()
        
        # Batch accumulator with default config
        self.accumulator = BatchAccumulator(BatchConfig())
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        logger.info("✅ Batch Accumulator Service initialized")
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "service": "batch_accumulator",
                "status": "healthy",
                "stats": self.accumulator.get_stats()
            }
        
        @self.app.post("/batch/add")
        async def add_to_batch(
            agent_id: str,
            request_type: str,
            payload: Dict[str, Any],
            priority: int = 1
        ):
            """Add item to batch queue"""
            try:
                item = BatchItem(
                    id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    request_type=request_type,
                    payload=payload,
                    priority=priority
                )
                
                item_id = await self.accumulator.add_item(item)
                
                return {
                    "item_id": item_id,
                    "status": "queued"
                }
                
            except Exception as e:
                logger.error(f"Failed to add item: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch/config")
        async def update_config(config: BatchConfig):
            """Update batch configuration"""
            try:
                self.accumulator.config = config
                self.accumulator.optimizer = AdaptiveBatchOptimizer(config)
                
                return {
                    "status": "updated",
                    "config": asdict(config)
                }
                
            except Exception as e:
                logger.error(f"Failed to update config: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/batch/stats")
        async def get_stats():
            """Get batch accumulator statistics"""
            return self.accumulator.get_stats()
        
        @self.app.get("/batch/active")
        async def get_active_batches():
            """Get active batches"""
            return {
                "batches": [
                    {
                        "id": batch.id,
                        "status": batch.status.value,
                        "size": batch.size,
                        "created_at": batch.created_at.isoformat(),
                        "agent_id": batch.metadata.get("agent_id")
                    }
                    for batch in self.accumulator.active_batches.values()
                ]
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic stats
                    stats = self.accumulator.get_stats()
                    await websocket.send_json(stats)
                    await asyncio.sleep(1)
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    async def startup(self):
        """Startup tasks"""
        logger.info("Batch Accumulator Service starting up...")
        await self.accumulator.start()
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Batch Accumulator Service shutting down...")
        await self.accumulator.stop()
        self.dal.cleanup()


def main():
    """Main entry point"""
    import uvicorn
    
    service = BatchAccumulatorService()
    
    # Add lifecycle events
    service.app.add_event_handler("startup", service.startup)
    service.app.add_event_handler("shutdown", service.shutdown)
    
    # Run service
    port = int(os.getenv("BATCH_SERVICE_PORT", 8014))
    logger.info(f"Starting Batch Accumulator Service on port {port}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()