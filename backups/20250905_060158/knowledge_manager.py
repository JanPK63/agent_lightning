"""
Knowledge Management System for Agent Lightning
Manages agent knowledge bases, context, and learning
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
import pickle
from collections import defaultdict
import numpy as np


@dataclass
class KnowledgeItem:
    """Individual piece of knowledge"""
    id: str
    category: str
    content: str
    source: str
    timestamp: datetime
    embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    relevance_score: float = 1.0


@dataclass 
class KnowledgeContext:
    """Context for a specific task or conversation"""
    task_id: str
    agent_id: str
    relevant_knowledge: List[KnowledgeItem] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    active_documents: List[str] = field(default_factory=list)
    custom_instructions: str = ""


class KnowledgeManager:
    """Manages knowledge bases for all agents"""
    
    def __init__(self, storage_dir: str = ".agent-knowledge"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Knowledge stores
        self.knowledge_bases: Dict[str, List[KnowledgeItem]] = defaultdict(list)
        self.contexts: Dict[str, KnowledgeContext] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Knowledge categories
        self.categories = [
            "technical_documentation",
            "code_examples",
            "best_practices", 
            "troubleshooting",
            "architecture_patterns",
            "api_references",
            "tutorials",
            "project_specific",
            "domain_knowledge",
            "conversation_memory"
        ]
        
        self.load_knowledge_bases()
    
    def load_knowledge_bases(self):
        """Load all knowledge bases from disk"""
        for kb_file in self.storage_dir.glob("*.json"):
            agent_name = kb_file.stem
            try:
                with open(kb_file, 'r') as f:
                    data = json.load(f)
                    items = []
                    for item_data in data:
                        item = KnowledgeItem(
                            id=item_data["id"],
                            category=item_data["category"],
                            content=item_data["content"],
                            source=item_data["source"],
                            timestamp=datetime.fromisoformat(item_data["timestamp"]),
                            metadata=item_data.get("metadata", {}),
                            usage_count=item_data.get("usage_count", 0),
                            relevance_score=item_data.get("relevance_score", 1.0)
                        )
                        items.append(item)
                    self.knowledge_bases[agent_name] = items
            except Exception as e:
                print(f"Error loading knowledge base for {agent_name}: {e}")
    
    def save_knowledge_base(self, agent_name: str):
        """Save a knowledge base to disk"""
        kb_file = self.storage_dir / f"{agent_name}.json"
        items = self.knowledge_bases.get(agent_name, [])
        
        data = []
        for item in items:
            data.append({
                "id": item.id,
                "category": item.category,
                "content": item.content,
                "source": item.source,
                "timestamp": item.timestamp.isoformat(),
                "metadata": item.metadata,
                "usage_count": item.usage_count,
                "relevance_score": item.relevance_score
            })
        
        with open(kb_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_knowledge(self, agent_name: str, category: str, content: str, 
                     source: str = "manual", metadata: Dict = None) -> KnowledgeItem:
        """Add new knowledge to an agent's knowledge base"""
        # Generate unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now()
        item_id = f"{category}_{content_hash}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        
        item = KnowledgeItem(
            id=item_id,
            category=category,
            content=content,
            source=source,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        self.knowledge_bases[agent_name].append(item)
        self.save_knowledge_base(agent_name)
        
        return item
    
    def search_knowledge(self, agent_name: str, query: str, 
                        category: Optional[str] = None, 
                        limit: int = 10) -> List[KnowledgeItem]:
        """Search for relevant knowledge items"""
        items = self.knowledge_bases.get(agent_name, [])
        
        # Filter by category if specified
        if category:
            items = [item for item in items if item.category == category]
        
        # Simple keyword matching (can be enhanced with embeddings)
        query_lower = query.lower()
        scored_items = []
        
        for item in items:
            score = 0
            content_lower = item.content.lower()
            
            # Check for exact match
            if query_lower in content_lower:
                score += 10
            
            # Check for word matches
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 1
            
            # Boost by relevance and usage
            score *= item.relevance_score
            score += item.usage_count * 0.1
            
            if score > 0:
                scored_items.append((score, item))
        
        # Sort by score and return top items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Update usage count for returned items
        results = []
        for score, item in scored_items[:limit]:
            item.usage_count += 1
            results.append(item)
        
        return results
    
    def create_context(self, task_id: str, agent_id: str, 
                      initial_query: str = "") -> KnowledgeContext:
        """Create a new context for a task"""
        context = KnowledgeContext(
            task_id=task_id,
            agent_id=agent_id
        )
        
        # Load relevant knowledge based on initial query
        if initial_query:
            relevant_items = self.search_knowledge(agent_id, initial_query)
            context.relevant_knowledge = relevant_items
        
        self.contexts[task_id] = context
        return context
    
    def update_context(self, task_id: str, message: Dict[str, str]):
        """Update context with new conversation"""
        if task_id in self.contexts:
            context = self.contexts[task_id]
            context.conversation_history.append(message)
            
            # Optionally search for more relevant knowledge based on new message
            if message.get("role") == "user":
                new_items = self.search_knowledge(
                    context.agent_id, 
                    message.get("content", ""),
                    limit=5
                )
                # Add new items not already in context
                existing_ids = {item.id for item in context.relevant_knowledge}
                for item in new_items:
                    if item.id not in existing_ids:
                        context.relevant_knowledge.append(item)
    
    def get_context_prompt(self, task_id: str) -> str:
        """Generate a context prompt for the agent"""
        if task_id not in self.contexts:
            return ""
        
        context = self.contexts[task_id]
        prompt_parts = []
        
        # Add custom instructions
        if context.custom_instructions:
            prompt_parts.append(f"Instructions: {context.custom_instructions}\n")
        
        # Add relevant knowledge
        if context.relevant_knowledge:
            prompt_parts.append("Relevant Knowledge:")
            for item in context.relevant_knowledge[:5]:  # Limit to top 5
                prompt_parts.append(f"- [{item.category}] {item.content[:200]}...")
            prompt_parts.append("")
        
        # Add conversation history
        if context.conversation_history:
            prompt_parts.append("Conversation History:")
            for msg in context.conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:100]
                prompt_parts.append(f"{role}: {content}...")
            prompt_parts.append("")
        
        return "\n".join(prompt_parts)
    
    def learn_from_interaction(self, agent_name: str, task_id: str, 
                              interaction: Dict[str, Any]):
        """Learn from an interaction to improve knowledge base"""
        # Extract useful information from the interaction
        if interaction.get("successful"):
            # Add successful solution to knowledge base
            problem = interaction.get("problem", "")
            solution = interaction.get("solution", "")
            
            if problem and solution:
                self.add_knowledge(
                    agent_name=agent_name,
                    category="troubleshooting",
                    content=f"Problem: {problem}\nSolution: {solution}",
                    source="interaction",
                    metadata={
                        "task_id": task_id,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": interaction.get("confidence", 0.8)
                    }
                )
    
    def export_knowledge_base(self, agent_name: str, output_file: str):
        """Export a knowledge base to a file"""
        items = self.knowledge_bases.get(agent_name, [])
        
        export_data = {
            "agent": agent_name,
            "exported_at": datetime.now().isoformat(),
            "total_items": len(items),
            "categories": list(set(item.category for item in items)),
            "knowledge_items": []
        }
        
        for item in items:
            export_data["knowledge_items"].append({
                "category": item.category,
                "content": item.content,
                "source": item.source,
                "metadata": item.metadata,
                "usage_count": item.usage_count
            })
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_knowledge_base(self, agent_name: str, input_file: str):
        """Import knowledge from a file"""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        for item_data in data.get("knowledge_items", []):
            self.add_knowledge(
                agent_name=agent_name,
                category=item_data["category"],
                content=item_data["content"],
                source=item_data.get("source", "imported"),
                metadata=item_data.get("metadata", {})
            )
    
    def get_statistics(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics about an agent's knowledge base"""
        items = self.knowledge_bases.get(agent_name, [])
        
        if not items:
            return {"total_items": 0}
        
        categories = defaultdict(int)
        sources = defaultdict(int)
        total_usage = 0
        
        for item in items:
            categories[item.category] += 1
            sources[item.source] += 1
            total_usage += item.usage_count
        
        return {
            "total_items": len(items),
            "categories": dict(categories),
            "sources": dict(sources),
            "total_usage": total_usage,
            "average_usage": total_usage / len(items),
            "most_used": sorted(items, key=lambda x: x.usage_count, reverse=True)[:5]
        }


# Example usage
if __name__ == "__main__":
    km = KnowledgeManager()
    
    # Add knowledge for full-stack developer
    km.add_knowledge(
        "full_stack_developer",
        "best_practices",
        "Always use parameterized queries to prevent SQL injection attacks",
        "security_guide"
    )
    
    km.add_knowledge(
        "full_stack_developer",
        "code_examples",
        "React hook for API calls: const useApi = (url) => { const [data, setData] = useState(null); ... }",
        "react_patterns"
    )
    
    # Search for knowledge
    results = km.search_knowledge("full_stack_developer", "SQL injection")
    for item in results:
        print(f"Found: {item.category} - {item.content[:50]}...")
    
    # Get statistics
    stats = km.get_statistics("full_stack_developer")
    print(f"\nKnowledge base statistics: {stats}")