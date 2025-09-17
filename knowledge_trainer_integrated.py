#!/usr/bin/env python3
"""
Integrated Knowledge Consumption and Training System for Agent Lightning
Connects Knowledge Manager -> Agent Designer Service -> RL Orchestrator
Provides complete end-to-end training pipeline
"""

import json
import os
import sys
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knowledge_manager import KnowledgeManager, KnowledgeItem


@dataclass
class KnowledgeConsumptionResult:
    """Result of knowledge consumption process"""
    agent_name: str
    knowledge_consumed: int = 0
    knowledge_integrated: int = 0
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


class IntegratedKnowledgeTrainer:
    """
    Manages knowledge consumption and training using microservices architecture
    Integrates: Knowledge Manager -> Agent Designer Service -> RL Orchestrator
    """
    
    def __init__(self):
        self.knowledge_manager = KnowledgeManager()
        
        # Microservice endpoints
        self.agent_designer_url = "http://localhost:8002"
        self.rl_orchestrator_url = "http://localhost:8011"
        
        # Track knowledge processing
        self.consumption_history: Dict[str, List[KnowledgeConsumptionResult]] = {}
        self.last_processed: Dict[str, datetime] = {}
        
    def get_agent_from_service(self, agent_name: str) -> Optional[Dict]:
        """Get agent details from Agent Designer Service"""
        try:
            response = requests.get(f"{self.agent_designer_url}/agents/{agent_name}")
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
        
    def get_new_knowledge(self, agent_name: str, since: Optional[datetime] = None) -> List[KnowledgeItem]:
        """
        Get new knowledge items for an agent since last processing
        
        Args:
            agent_name: Name of the agent
            since: Datetime to check from (defaults to last processed time)
        
        Returns:
            List of new knowledge items
        """
        if not since:
            since = self.last_processed.get(agent_name, datetime.now() - timedelta(days=30))
        
        knowledge_items = self.knowledge_manager.knowledge_bases.get(agent_name, [])
        new_items = [
            item for item in knowledge_items 
            if item.timestamp > since
        ]
        
        return new_items
    
    def consume_knowledge(self, agent_name: str, force_all: bool = False) -> KnowledgeConsumptionResult:
        """
        Make an agent consume and process new knowledge through RL Orchestrator
        
        Args:
            agent_name: Name of the agent
            force_all: Process all knowledge, not just new items
        
        Returns:
            Consumption result with metrics
        """
        start_time = datetime.now()
        result = KnowledgeConsumptionResult(agent_name=agent_name)
        
        # Verify agent exists in Agent Designer Service
        agent = self.get_agent_from_service(agent_name)
        if not agent:
            result.errors.append(f"Agent {agent_name} not found in Agent Designer Service")
            return result
        
        # Get knowledge to process
        if force_all:
            knowledge_items = self.knowledge_manager.knowledge_bases.get(agent_name, [])
        else:
            knowledge_items = self.get_new_knowledge(agent_name)
        
        result.knowledge_consumed = len(knowledge_items)
        
        if not knowledge_items:
            return result
        
        # Convert knowledge items to format expected by RL Orchestrator
        training_items = []
        for item in knowledge_items:
            training_items.append({
                "content": item.content,
                "category": item.category,
                "importance": item.relevance_score
            })
        
        # Send to RL Orchestrator for training
        try:
            response = requests.post(
                f"{self.rl_orchestrator_url}/agents/{agent_name}/train",
                json={
                    "agent_id": agent_name,
                    "knowledge_items": training_items,
                    "training_config": {
                        "algorithm": "DQN",
                        "num_episodes": 100,
                        "learning_rate": 0.001
                    }
                }
            )
            
            if response.status_code == 200:
                result.knowledge_integrated = len(training_items)
                result.improvements.append(f"Successfully trained with {len(training_items)} knowledge items")
                
                # Update last processed time
                self.last_processed[agent_name] = datetime.now()
                
                # Mark knowledge items as consumed
                for item in knowledge_items:
                    item.usage_count += 1
                
                # Save updated knowledge base
                self.knowledge_manager.save_knowledge_base(agent_name)
            else:
                result.errors.append(f"Training failed: {response.text}")
                
        except Exception as e:
            result.errors.append(f"Failed to connect to RL Orchestrator: {str(e)}")
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store in history
        if agent_name not in self.consumption_history:
            self.consumption_history[agent_name] = []
        self.consumption_history[agent_name].append(result)
        
        return result
    
    def get_consumption_stats(self, agent_name: str) -> Dict[str, Any]:
        """
        Get consumption statistics for an agent including RL training status
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            Dictionary with consumption stats
        """
        knowledge_items = self.knowledge_manager.knowledge_bases.get(agent_name, [])
        new_items = self.get_new_knowledge(agent_name)
        
        stats = {
            "total_knowledge_items": len(knowledge_items),
            "new_knowledge_available": len(new_items),
            "last_consumption": self.last_processed.get(agent_name)
        }
        
        # Get training status from RL Orchestrator
        try:
            response = requests.get(f"{self.rl_orchestrator_url}/agents/{agent_name}/training-status")
            if response.status_code == 200:
                training_data = response.json()
                if training_data.get("last_trained") and training_data["last_trained"] != "Never":
                    stats["last_consumption"] = datetime.fromisoformat(training_data["last_trained"])
        except:
            pass
        
        return stats
    
    def test_agent_knowledge(self, agent_name: str, test_queries: List[str]) -> List[Dict[str, str]]:
        """
        Test agent with queries to verify knowledge integration
        
        Args:
            agent_name: Name of the agent
            test_queries: List of test queries
        
        Returns:
            List of query-response pairs
        """
        results = []
        
        for query in test_queries:
            # Search knowledge base for relevant items
            relevant_items = self.knowledge_manager.search_knowledge(
                agent_name, query, limit=3
            )
            
            # Format response based on found knowledge
            if relevant_items:
                response = f"Based on my knowledge:\n"
                for item in relevant_items:
                    response += f"- {item.content[:200]}...\n"
            else:
                response = "No relevant knowledge found for this query."
            
            results.append({
                "query": query,
                "response": response
            })
        
        return results
    
    async def active_training_session(self, agent_name: str, training_queries: List[str]) -> Dict[str, Any]:
        """
        Conduct an active training session with an agent using its new knowledge
        
        Args:
            agent_name: Name of the agent
            training_queries: List of queries to test the agent with
        
        Returns:
            Training results
        """
        results = {
            "agent": agent_name,
            "queries_tested": len(training_queries),
            "knowledge_applied": 0,
            "responses": []
        }
        
        # First consume any new knowledge
        consumption_result = self.consume_knowledge(agent_name)
        results["knowledge_consumed"] = consumption_result.knowledge_consumed
        results["knowledge_integrated"] = consumption_result.knowledge_integrated
        
        # Test agent with training queries
        for query in training_queries:
            # Search for relevant knowledge
            relevant_items = self.knowledge_manager.search_knowledge(
                agent_name, query, limit=3
            )
            
            if relevant_items:
                results["knowledge_applied"] += 1
                response = f"Based on training knowledge:\n"
                for item in relevant_items[:2]:  # Use top 2 items
                    response += f"- {item.content[:300]}...\n"
            else:
                response = "No specific knowledge found for this query. Using general capabilities."
            
            results["responses"].append({
                "query": query,
                "response": response,
                "knowledge_used": len(relevant_items)
            })
        
        return results


# Singleton instance for dashboard compatibility
_trainer_instance = None

def KnowledgeTrainer():
    """Factory function to maintain compatibility with dashboard"""
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = IntegratedKnowledgeTrainer()
    return _trainer_instance


if __name__ == "__main__":
    # Test the integrated trainer
    trainer = KnowledgeTrainer()
    
    # Test with test_engineer agent
    print("Testing integrated knowledge trainer with test_engineer...")
    
    # Get consumption stats
    stats = trainer.get_consumption_stats("test_engineer")
    print(f"Stats: {stats}")
    
    # Try to consume knowledge
    result = trainer.consume_knowledge("test_engineer", force_all=True)
    print(f"Consumption result: {result}")