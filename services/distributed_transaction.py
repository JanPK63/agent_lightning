#!/usr/bin/env python3
"""
Distributed Transaction Coordinator
Implements Saga pattern for managing distributed transactions across microservices
"""

import asyncio
import aiohttp
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import deque
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionStatus(str, Enum):
    """Transaction statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMMITTING = "committing"
    COMMITTED = "committed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    TIMEOUT = "timeout"


class StepStatus(str, Enum):
    """Step execution statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


@dataclass
class SagaStep:
    """Represents a single step in a saga transaction"""
    id: str
    name: str
    service: str
    action: str
    params: Dict[str, Any]
    compensate_action: Optional[str] = None
    compensate_params: Optional[Dict[str, Any]] = None
    timeout: int = 30  # seconds
    retry_count: int = 3
    retry_delay: int = 1  # seconds
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def can_execute(self, completed_steps: List[str]) -> bool:
        """Check if step can be executed based on dependencies"""
        return all(dep in completed_steps for dep in self.depends_on)


@dataclass
class SagaTransaction:
    """Represents a distributed transaction using Saga pattern"""
    id: str
    name: str
    steps: List[SagaStep]
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    compensate_on_failure: bool = True
    isolation_level: str = "READ_COMMITTED"
    timeout: int = 300  # seconds
    
    def get_completed_steps(self) -> List[str]:
        """Get list of completed step IDs"""
        return [step.id for step in self.steps if step.status == StepStatus.COMPLETED]
    
    def get_failed_steps(self) -> List[str]:
        """Get list of failed step IDs"""
        return [step.id for step in self.steps if step.status == StepStatus.FAILED]
    
    def get_next_steps(self) -> List[SagaStep]:
        """Get next executable steps"""
        completed = self.get_completed_steps()
        return [
            step for step in self.steps 
            if step.status == StepStatus.PENDING and step.can_execute(completed)
        ]


class SagaCoordinator:
    """Coordinates distributed transactions using Saga pattern"""
    
    def __init__(self):
        self.transactions: Dict[str, SagaTransaction] = {}
        self.transaction_log: deque = deque(maxlen=1000)
        self.compensate_handlers: Dict[str, Callable] = {}
        self.service_endpoints: Dict[str, str] = {
            "auth": "http://localhost:8006",
            "agent": "http://localhost:8001",
            "workflow": "http://localhost:8003",
            "integration": "http://localhost:8004",
            "ai": "http://localhost:8005",
            "gateway": "http://localhost:8000"
        }
        
        # Transaction persistence
        self.persist_path = "transactions.db"
        self._load_transactions()
    
    def _load_transactions(self):
        """Load persisted transactions"""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'rb') as f:
                    self.transactions = pickle.load(f)
                logger.info(f"Loaded {len(self.transactions)} transactions")
            except Exception as e:
                logger.error(f"Failed to load transactions: {e}")
    
    def _save_transactions(self):
        """Persist transactions to disk"""
        try:
            with open(self.persist_path, 'wb') as f:
                pickle.dump(self.transactions, f)
        except Exception as e:
            logger.error(f"Failed to save transactions: {e}")
    
    async def create_transaction(self, name: str, steps: List[Dict[str, Any]]) -> SagaTransaction:
        """Create a new saga transaction"""
        transaction_id = f"txn-{uuid.uuid4().hex[:12]}"
        
        # Convert step definitions to SagaStep objects
        saga_steps = []
        for step_def in steps:
            step_id = f"step-{uuid.uuid4().hex[:8]}"
            saga_step = SagaStep(
                id=step_id,
                name=step_def.get("name", step_id),
                service=step_def["service"],
                action=step_def["action"],
                params=step_def.get("params", {}),
                compensate_action=step_def.get("compensate_action"),
                compensate_params=step_def.get("compensate_params", {}),
                timeout=step_def.get("timeout", 30),
                retry_count=step_def.get("retry_count", 3),
                depends_on=step_def.get("depends_on", [])
            )
            saga_steps.append(saga_step)
        
        # Create transaction
        transaction = SagaTransaction(
            id=transaction_id,
            name=name,
            steps=saga_steps
        )
        
        self.transactions[transaction_id] = transaction
        self._save_transactions()
        
        logger.info(f"Created transaction: {transaction_id} with {len(saga_steps)} steps")
        return transaction
    
    async def execute_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Execute a saga transaction"""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return {"error": "Transaction not found"}
        
        transaction.status = TransactionStatus.RUNNING
        transaction.started_at = datetime.now()
        
        try:
            # Execute steps
            while True:
                next_steps = transaction.get_next_steps()
                if not next_steps:
                    # Check if all steps completed
                    if all(step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] 
                           for step in transaction.steps):
                        break
                    
                    # Check for failures
                    if any(step.status == StepStatus.FAILED for step in transaction.steps):
                        raise Exception("Transaction has failed steps")
                    
                    # Wait for running steps
                    await asyncio.sleep(1)
                    continue
                
                # Execute next steps in parallel
                tasks = [self._execute_step(transaction, step) for step in next_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures
                for step, result in zip(next_steps, results):
                    if isinstance(result, Exception):
                        step.status = StepStatus.FAILED
                        step.error = str(result)
                        
                        if transaction.compensate_on_failure:
                            await self._compensate_transaction(transaction)
                            return {
                                "status": "compensated",
                                "transaction_id": transaction_id,
                                "error": str(result)
                            }
                        else:
                            transaction.status = TransactionStatus.FAILED
                            transaction.error = str(result)
                            self._save_transactions()
                            return {
                                "status": "failed",
                                "transaction_id": transaction_id,
                                "error": str(result)
                            }
            
            # Commit transaction
            transaction.status = TransactionStatus.COMMITTED
            transaction.completed_at = datetime.now()
            self._save_transactions()
            
            # Log transaction
            self.transaction_log.append({
                "transaction_id": transaction_id,
                "status": "committed",
                "duration": (transaction.completed_at - transaction.started_at).total_seconds(),
                "steps_executed": len(transaction.steps)
            })
            
            return {
                "status": "committed",
                "transaction_id": transaction_id,
                "results": {step.id: step.result for step in transaction.steps if step.result}
            }
            
        except Exception as e:
            logger.error(f"Transaction {transaction_id} failed: {e}")
            transaction.status = TransactionStatus.FAILED
            transaction.error = str(e)
            
            if transaction.compensate_on_failure:
                await self._compensate_transaction(transaction)
            
            self._save_transactions()
            return {
                "status": "failed",
                "transaction_id": transaction_id,
                "error": str(e)
            }
    
    async def _execute_step(self, transaction: SagaTransaction, step: SagaStep) -> Dict[str, Any]:
        """Execute a single saga step"""
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        for attempt in range(step.retry_count):
            try:
                # Get service endpoint
                endpoint = self.service_endpoints.get(step.service)
                if not endpoint:
                    raise Exception(f"Unknown service: {step.service}")
                
                # Prepare request with transaction context
                url = f"{endpoint}/{step.action}"
                payload = {
                    **step.params,
                    "_transaction": {
                        "id": transaction.id,
                        "step": step.id,
                        "isolation": transaction.isolation_level
                    }
                }
                
                # Execute request
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=step.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            step.status = StepStatus.COMPLETED
                            step.result = result
                            step.completed_at = datetime.now()
                            
                            # Update transaction context
                            if "context_update" in result:
                                transaction.context.update(result["context_update"])
                            
                            logger.info(f"Step {step.name} completed successfully")
                            return result
                        
                        else:
                            error_text = await response.text()
                            raise Exception(f"Service returned {response.status}: {error_text}")
                
            except asyncio.TimeoutError:
                logger.warning(f"Step {step.name} timeout (attempt {attempt + 1}/{step.retry_count})")
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(step.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    step.status = StepStatus.FAILED
                    step.error = "Timeout"
                    raise
            
            except Exception as e:
                logger.error(f"Step {step.name} failed (attempt {attempt + 1}/{step.retry_count}): {e}")
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(step.retry_delay * (2 ** attempt))
                else:
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    raise
    
    async def _compensate_transaction(self, transaction: SagaTransaction):
        """Compensate a failed transaction"""
        transaction.status = TransactionStatus.COMPENSATING
        logger.info(f"Compensating transaction {transaction.id}")
        
        # Get completed steps that need compensation (in reverse order)
        completed_steps = [
            step for step in reversed(transaction.steps)
            if step.status == StepStatus.COMPLETED and step.compensate_action
        ]
        
        for step in completed_steps:
            try:
                await self._execute_compensation(transaction, step)
                step.status = StepStatus.COMPENSATED
            except Exception as e:
                logger.error(f"Failed to compensate step {step.name}: {e}")
        
        transaction.status = TransactionStatus.COMPENSATED
        transaction.completed_at = datetime.now()
        self._save_transactions()
    
    async def _execute_compensation(self, transaction: SagaTransaction, step: SagaStep):
        """Execute compensation for a step"""
        endpoint = self.service_endpoints.get(step.service)
        if not endpoint:
            raise Exception(f"Unknown service: {step.service}")
        
        url = f"{endpoint}/{step.compensate_action}"
        payload = {
            **step.compensate_params,
            "original_result": step.result,
            "_transaction": {
                "id": transaction.id,
                "step": step.id,
                "compensation": True
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=step.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Compensation failed: {error_text}")
                
                logger.info(f"Compensated step {step.name}")
    
    def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction status"""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return None
        
        return {
            "id": transaction.id,
            "name": transaction.name,
            "status": transaction.status,
            "created_at": transaction.created_at.isoformat(),
            "started_at": transaction.started_at.isoformat() if transaction.started_at else None,
            "completed_at": transaction.completed_at.isoformat() if transaction.completed_at else None,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "status": step.status,
                    "error": step.error
                }
                for step in transaction.steps
            ],
            "context": transaction.context,
            "error": transaction.error
        }
    
    def get_transaction_history(self) -> List[Dict[str, Any]]:
        """Get transaction history"""
        return list(self.transaction_log)


# Example Saga Definitions
class SagaDefinitions:
    """Pre-defined saga workflows"""
    
    @staticmethod
    def create_agent_with_workflow():
        """Saga for creating an agent with associated workflow"""
        return [
            {
                "name": "Create Agent",
                "service": "agent",
                "action": "api/v1/agents",
                "params": {
                    "name": "Automated Agent",
                    "type": "conversational"
                },
                "compensate_action": "api/v1/agents/delete",
                "compensate_params": {"agent_id": "{result.id}"}
            },
            {
                "name": "Create Workflow",
                "service": "workflow",
                "action": "api/v1/workflows",
                "params": {
                    "name": "Agent Workflow",
                    "agent_id": "{steps[0].result.id}"
                },
                "compensate_action": "api/v1/workflows/delete",
                "compensate_params": {"workflow_id": "{result.id}"},
                "depends_on": ["step-1"]
            },
            {
                "name": "Configure Integration",
                "service": "integration",
                "action": "api/v1/integrations/configure",
                "params": {
                    "workflow_id": "{steps[1].result.id}",
                    "type": "webhook"
                },
                "compensate_action": "api/v1/integrations/remove",
                "compensate_params": {"integration_id": "{result.id}"},
                "depends_on": ["step-2"]
            },
            {
                "name": "Deploy Agent",
                "service": "agent",
                "action": "api/v1/agents/deploy",
                "params": {
                    "agent_id": "{steps[0].result.id}",
                    "environment": "production"
                },
                "compensate_action": "api/v1/agents/undeploy",
                "compensate_params": {"agent_id": "{steps[0].result.id}"},
                "depends_on": ["step-3"]
            }
        ]
    
    @staticmethod
    def process_payment_with_notification():
        """Saga for payment processing with notifications"""
        return [
            {
                "name": "Reserve Funds",
                "service": "payment",
                "action": "api/v1/payments/reserve",
                "params": {
                    "amount": 100.00,
                    "currency": "USD"
                },
                "compensate_action": "api/v1/payments/release",
                "compensate_params": {"reservation_id": "{result.reservation_id}"}
            },
            {
                "name": "Process Order",
                "service": "order",
                "action": "api/v1/orders/process",
                "params": {
                    "payment_id": "{steps[0].result.payment_id}"
                },
                "compensate_action": "api/v1/orders/cancel",
                "compensate_params": {"order_id": "{result.order_id}"},
                "depends_on": ["step-1"]
            },
            {
                "name": "Capture Payment",
                "service": "payment",
                "action": "api/v1/payments/capture",
                "params": {
                    "reservation_id": "{steps[0].result.reservation_id}"
                },
                "depends_on": ["step-2"]
            },
            {
                "name": "Send Notification",
                "service": "notification",
                "action": "api/v1/notifications/send",
                "params": {
                    "type": "order_confirmed",
                    "order_id": "{steps[1].result.order_id}"
                },
                "depends_on": ["step-3"],
                "retry_count": 1  # Notifications are less critical
            }
        ]


# FastAPI Service
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List


class CreateTransactionRequest(BaseModel):
    name: str
    steps: List[Dict[str, Any]]
    compensate_on_failure: bool = True
    timeout: int = 300


class ExecuteTransactionRequest(BaseModel):
    transaction_id: str


class TransactionService:
    """Distributed transaction service"""
    
    def __init__(self):
        self.app = FastAPI(title="Distributed Transaction Service", version="1.0.0")
        self.coordinator = SagaCoordinator()
        
        # Setup middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "service": "distributed-transaction",
                "transactions_active": len([
                    t for t in self.coordinator.transactions.values()
                    if t.status == TransactionStatus.RUNNING
                ])
            }
        
        @self.app.post("/api/v1/transactions")
        async def create_transaction(request: CreateTransactionRequest):
            """Create a new distributed transaction"""
            transaction = await self.coordinator.create_transaction(
                request.name,
                request.steps
            )
            return {
                "transaction_id": transaction.id,
                "status": transaction.status,
                "steps": len(transaction.steps)
            }
        
        @self.app.post("/api/v1/transactions/execute")
        async def execute_transaction(request: ExecuteTransactionRequest):
            """Execute a distributed transaction"""
            result = await self.coordinator.execute_transaction(
                request.transaction_id
            )
            return result
        
        @self.app.get("/api/v1/transactions/{transaction_id}")
        async def get_transaction(transaction_id: str):
            """Get transaction status"""
            status = self.coordinator.get_transaction_status(transaction_id)
            if not status:
                raise HTTPException(status_code=404, detail="Transaction not found")
            return status
        
        @self.app.get("/api/v1/transactions")
        async def list_transactions():
            """List all transactions"""
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "status": t.status,
                    "created_at": t.created_at.isoformat(),
                    "steps": len(t.steps)
                }
                for t in self.coordinator.transactions.values()
            ]
        
        @self.app.get("/api/v1/transactions/history")
        async def get_history():
            """Get transaction history"""
            return self.coordinator.get_transaction_history()


if __name__ == "__main__":
    import uvicorn
    
    print("Distributed Transaction Coordinator")
    print("=" * 60)
    print("\nImplementing Saga Pattern for microservices transactions")
    print("\nFeatures:")
    print("  • Orchestrated saga transactions")
    print("  • Automatic compensation on failure")
    print("  • Step dependencies and parallelism")
    print("  • Retry with exponential backoff")
    print("  • Transaction persistence")
    print("  • Isolation levels")
    
    service = TransactionService()
    
    uvicorn.run(service.app, host="0.0.0.0", port=8008, reload=False)