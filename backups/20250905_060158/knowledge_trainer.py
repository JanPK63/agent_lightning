#!/usr/bin/env python3
"""
Knowledge Consumption and Training System for Agent Lightning
Allows agents to actively consume, process, and train on new knowledge
"""

import json
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knowledge_manager import KnowledgeManager, KnowledgeItem
from agent_config import AgentConfigManager, AgentConfig
from enhanced_production_api import EnhancedAgentService


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


class KnowledgeTrainer:
    """
    Manages active knowledge consumption and training for agents
    """
    
    def __init__(self):
        self.knowledge_manager = KnowledgeManager()
        self.config_manager = AgentConfigManager()
        self.agent_service = None  # Will be initialized when needed
        
        # Track knowledge processing
        self.consumption_history: Dict[str, List[KnowledgeConsumptionResult]] = {}
        self.last_processed: Dict[str, datetime] = {}
        
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
        Make an agent consume and process new knowledge
        
        Args:
            agent_name: Name of the agent
            force_all: Process all knowledge, not just new items
        
        Returns:
            Consumption result with metrics
        """
        start_time = datetime.now()
        result = KnowledgeConsumptionResult(agent_name=agent_name)
        
        # Get agent configuration
        agent_config = self.config_manager.get_agent(agent_name)
        if not agent_config:
            result.errors.append(f"Agent {agent_name} not found")
            return result
        
        # Get knowledge to process
        if force_all:
            knowledge_items = self.knowledge_manager.knowledge_bases.get(agent_name, [])
        else:
            knowledge_items = self.get_new_knowledge(agent_name)
        
        result.knowledge_consumed = len(knowledge_items)
        
        if not knowledge_items:
            return result
        
        # Process knowledge by category
        categorized = {}
        for item in knowledge_items:
            if item.category not in categorized:
                categorized[item.category] = []
            categorized[item.category].append(item)
        
        # Integrate knowledge into agent's configuration
        for category, items in categorized.items():
            try:
                self._integrate_knowledge_category(agent_config, category, items)
                result.knowledge_integrated += len(items)
                result.improvements.append(f"Integrated {len(items)} {category} items")
            except Exception as e:
                result.errors.append(f"Error processing {category}: {str(e)}")
        
        # Update agent configuration if knowledge was integrated
        if result.knowledge_integrated > 0:
            # Enhance system prompt with new knowledge
            self._update_agent_prompt(agent_config, categorized)
            
            # Save updated configuration
            self.config_manager.save_agent(agent_config)
            
            # Update relevance scores based on patterns
            self._update_relevance_scores(agent_name, knowledge_items)
        
        # Record processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        self.last_processed[agent_name] = datetime.now()
        
        # Save consumption history
        if agent_name not in self.consumption_history:
            self.consumption_history[agent_name] = []
        self.consumption_history[agent_name].append(result)
        
        return result
    
    def _integrate_knowledge_category(self, agent_config: AgentConfig, category: str, items: List[KnowledgeItem]):
        """
        Integrate knowledge items of a specific category into agent configuration
        """
        # Map categories to configuration fields
        category_mapping = {
            "best_practices": "best_practices",
            "technical_documentation": "documentation_urls",
            "code_examples": "examples",
            "troubleshooting": "custom_instructions",
            "architecture_patterns": "frameworks",
            "api_references": "technologies",
            "domain_knowledge": "domains"
        }
        
        if category in category_mapping:
            config_field = category_mapping[category]
            
            # Handle different field types
            if config_field == "best_practices":
                for item in items:
                    if item.content not in agent_config.knowledge_base.best_practices:
                        agent_config.knowledge_base.best_practices.append(item.content[:200])
            
            elif config_field == "examples":
                for item in items:
                    # Parse examples from content
                    if "Input:" in item.content and "Output:" in item.content:
                        parts = item.content.split("Output:")
                        if len(parts) == 2:
                            input_text = parts[0].replace("Input:", "").strip()
                            output_text = parts[1].strip()
                            example = {"input": input_text, "output": output_text}
                            if example not in agent_config.examples:
                                agent_config.examples.append(example)
            
            elif config_field == "custom_instructions":
                # Append to custom instructions
                new_instructions = "\n".join([item.content for item in items[:3]])  # Limit to 3
                if new_instructions not in agent_config.knowledge_base.custom_instructions:
                    agent_config.knowledge_base.custom_instructions += f"\n\n{new_instructions}"
    
    def _update_agent_prompt(self, agent_config: AgentConfig, categorized: Dict[str, List[KnowledgeItem]]):
        """
        Update agent's system prompt with new knowledge
        """
        # Build knowledge summary for prompt
        knowledge_summary = "\n\n## Recently Acquired Knowledge:\n"
        
        for category, items in list(categorized.items())[:3]:  # Top 3 categories
            knowledge_summary += f"\n### {category.replace('_', ' ').title()}:\n"
            for item in items[:2]:  # Top 2 items per category
                knowledge_summary += f"- {item.content[:150]}...\n"
        
        # Check if knowledge section already exists
        if "## Recently Acquired Knowledge:" not in agent_config.system_prompt:
            agent_config.system_prompt += knowledge_summary
        else:
            # Replace existing knowledge section
            parts = agent_config.system_prompt.split("## Recently Acquired Knowledge:")
            agent_config.system_prompt = parts[0] + knowledge_summary
    
    def _update_relevance_scores(self, agent_name: str, items: List[KnowledgeItem]):
        """
        Update relevance scores based on usage patterns
        """
        for item in items:
            # Increase relevance for recently added items
            days_old = (datetime.now() - item.timestamp).days
            if days_old < 7:
                item.relevance_score = min(1.0, item.relevance_score + 0.1)
            elif days_old > 30:
                item.relevance_score = max(0.1, item.relevance_score - 0.05)
            
            # Boost relevance for frequently used items
            if item.usage_count > 5:
                item.relevance_score = min(1.0, item.relevance_score + 0.2)
        
        # Save updated knowledge base
        self.knowledge_manager.save_knowledge_base(agent_name)
    
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
        
        # Initialize agent service if needed
        if not self.agent_service:
            from enhanced_production_api import EnhancedAgentService
            self.agent_service = EnhancedAgentService()
        
        # Test agent with training queries
        for query in training_queries:
            try:
                # Create a mock request
                from production_api import AgentRequest
                request = AgentRequest(
                    agent_id=agent_name,
                    task=query,
                    context={}
                )
                
                # Process with knowledge
                response = await self.agent_service.process_task_with_knowledge(request)
                
                # Check if knowledge was used
                if response.result.get("knowledge_items_used", 0) > 0:
                    results["knowledge_applied"] += 1
                
                results["responses"].append({
                    "query": query,
                    "response": response.result.get("response", ""),  # Full response, no truncation
                    "knowledge_used": response.result.get("knowledge_items_used", 0)
                })
                
            except Exception as e:
                results["responses"].append({
                    "query": query,
                    "error": str(e)
                })
        
        return results
    
    def get_consumption_stats(self, agent_name: str) -> Dict[str, Any]:
        """
        Get knowledge consumption statistics for an agent
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            Consumption statistics
        """
        stats = {
            "agent": agent_name,
            "total_knowledge_items": len(self.knowledge_manager.knowledge_bases.get(agent_name, [])),
            "last_consumption": self.last_processed.get(agent_name),
            "consumption_history": []
        }
        
        # Get consumption history
        if agent_name in self.consumption_history:
            for result in self.consumption_history[agent_name][-10:]:  # Last 10
                stats["consumption_history"].append({
                    "timestamp": result.timestamp.isoformat(),
                    "consumed": result.knowledge_consumed,
                    "integrated": result.knowledge_integrated,
                    "time_taken": result.processing_time
                })
        
        # Get knowledge by category
        knowledge_items = self.knowledge_manager.knowledge_bases.get(agent_name, [])
        categories = {}
        for item in knowledge_items:
            categories[item.category] = categories.get(item.category, 0) + 1
        stats["knowledge_by_category"] = categories
        
        # Get new knowledge count
        new_items = self.get_new_knowledge(agent_name)
        stats["new_knowledge_available"] = len(new_items)
        
        return stats
    
    def auto_consume_all_agents(self) -> Dict[str, KnowledgeConsumptionResult]:
        """
        Automatically consume new knowledge for all agents
        
        Returns:
            Dictionary of consumption results by agent
        """
        results = {}
        
        for agent_name in self.config_manager.list_agents():
            result = self.consume_knowledge(agent_name)
            results[agent_name] = result
            
            if result.knowledge_integrated > 0:
                print(f"✅ {agent_name}: Consumed {result.knowledge_consumed} items, "
                      f"integrated {result.knowledge_integrated}")
            elif result.errors:
                print(f"❌ {agent_name}: Errors: {', '.join(result.errors)}")
        
        return results


class KnowledgeTrainingAPI:
    """API endpoints for knowledge training"""
    
    def __init__(self):
        self.trainer = KnowledgeTrainer()
    
    async def consume_knowledge(self, agent_name: str, force_all: bool = False) -> Dict[str, Any]:
        """
        API endpoint to make an agent consume knowledge
        """
        result = self.trainer.consume_knowledge(agent_name, force_all)
        
        return {
            "success": len(result.errors) == 0,
            "agent": result.agent_name,
            "knowledge_consumed": result.knowledge_consumed,
            "knowledge_integrated": result.knowledge_integrated,
            "processing_time": result.processing_time,
            "improvements": result.improvements,
            "errors": result.errors
        }
    
    async def train_with_knowledge(self, agent_name: str, queries: List[str]) -> Dict[str, Any]:
        """
        API endpoint to train agent with queries
        """
        return await self.trainer.active_training_session(agent_name, queries)
    
    async def get_stats(self, agent_name: str) -> Dict[str, Any]:
        """
        API endpoint to get consumption statistics
        """
        return self.trainer.get_consumption_stats(agent_name)
    
    async def auto_consume_all(self) -> Dict[str, Any]:
        """
        API endpoint to auto-consume for all agents
        """
        results = self.trainer.auto_consume_all_agents()
        
        return {
            "agents_processed": len(results),
            "total_consumed": sum(r.knowledge_consumed for r in results.values()),
            "total_integrated": sum(r.knowledge_integrated for r in results.values()),
            "details": {
                name: {
                    "consumed": r.knowledge_consumed,
                    "integrated": r.knowledge_integrated,
                    "errors": r.errors
                }
                for name, r in results.items()
            }
        }


if __name__ == "__main__":
    # Example usage
    trainer = KnowledgeTrainer()
    
    print("🧠 Knowledge Consumption and Training System")
    print("=" * 60)
    
    # Consume knowledge for database_specialist
    print("\n1. Consuming knowledge for database_specialist...")
    result = trainer.consume_knowledge("database_specialist")
    
    print(f"   ✅ Consumed: {result.knowledge_consumed} items")
    print(f"   ✅ Integrated: {result.knowledge_integrated} items")
    print(f"   ⏱️  Time: {result.processing_time:.2f}s")
    
    if result.improvements:
        print(f"   📈 Improvements: {', '.join(result.improvements)}")
    
    # Get statistics
    print("\n2. Knowledge Statistics:")
    stats = trainer.get_consumption_stats("database_specialist")
    
    print(f"   Total Knowledge Items: {stats['total_knowledge_items']}")
    print(f"   New Knowledge Available: {stats['new_knowledge_available']}")
    print(f"   Knowledge by Category:")
    for category, count in stats['knowledge_by_category'].items():
        print(f"      - {category}: {count}")
    
    # Test with queries
    print("\n3. Testing with sample queries...")
    test_queries = [
        "How do I optimize a PostgreSQL query?",
        "What's the best caching strategy with Redis?",
        "How to design MongoDB schema for e-commerce?"
    ]
    
    async def test_training():
        results = await trainer.active_training_session("database_specialist", test_queries)
        
        print(f"   Queries Tested: {results['queries_tested']}")
        print(f"   Knowledge Applied: {results['knowledge_applied']}")
        
        for response in results['responses']:
            print(f"\n   Q: {response['query']}")
            if 'error' in response:
                print(f"   Error: {response['error']}")
            else:
                print(f"   Knowledge Used: {response['knowledge_used']} items")
                print(f"   Response: {response['response']}...")
    
    # Run async test
    try:
        asyncio.run(test_training())
    except:
        print("   (Skipping live test - API not running)")
    
    print("\n✅ Knowledge consumption and training system ready!")
    print("\nThe system will:")
    print("1. Automatically detect new knowledge added to agents")
    print("2. Consume and integrate it into agent configurations")
    print("3. Update agent prompts with recent knowledge")
    print("4. Track consumption history and statistics")
    print("5. Allow active training sessions with test queries")