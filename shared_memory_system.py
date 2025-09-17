#!/usr/bin/env python3
"""
Shared Memory System for Agent Lightning
Provides all agents with a unified view of project status, development history, and conversations
"""

import json
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from collections import deque
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager, MemoryEntry
from knowledge_manager import KnowledgeManager
from project_config import ProjectConfigManager


@dataclass
class ConversationEntry:
    """Single conversation entry"""
    timestamp: datetime
    agent: str
    user_query: str
    agent_response: str
    task_id: Optional[str] = None
    knowledge_used: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectMemory:
    """Project-specific memory"""
    project_name: str
    current_phase: str = "Planning"
    completed_tasks: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    active_branches: List[str] = field(default_factory=list)
    recent_deployments: List[Dict] = field(default_factory=list)
    project_context: str = ""
    last_updated: datetime = field(default_factory=datetime.now)


class SharedMemorySystem:
    """
    Unified memory system that provides all agents with shared context
    """
    
    def __init__(self, storage_dir: str = ".agent-memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize subsystems
        self.memory_manager = MemoryManager(
            persistence_path=self.storage_dir / "memory_store"
        )
        self.knowledge_manager = KnowledgeManager()
        self.project_manager = ProjectConfigManager()
        
        # Shared memory stores
        self.conversation_history = deque(maxlen=100)  # Last 100 conversations
        self.project_memories: Dict[str, ProjectMemory] = {}
        self.global_context: Dict[str, Any] = {}
        self.agent_interactions: Dict[str, List[ConversationEntry]] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load existing memory
        self.load_memory()
    
    def load_memory(self):
        """Load memory from disk"""
        # Load conversation history
        conv_file = self.storage_dir / "conversation_history.json"
        if conv_file.exists():
            try:
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = ConversationEntry(
                            timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                            agent=entry_data["agent"],
                            user_query=entry_data["user_query"],
                            agent_response=entry_data["agent_response"],
                            task_id=entry_data.get("task_id"),
                            knowledge_used=entry_data.get("knowledge_used", 0),
                            success=entry_data.get("success", True),
                            metadata=entry_data.get("metadata", {})
                        )
                        self.conversation_history.append(entry)
            except Exception as e:
                print(f"Error loading conversation history: {e}")
        
        # Load project memories
        project_file = self.storage_dir / "project_memories.json"
        if project_file.exists():
            try:
                with open(project_file, 'r') as f:
                    data = json.load(f)
                    for name, proj_data in data.items():
                        self.project_memories[name] = ProjectMemory(
                            project_name=proj_data["project_name"],
                            current_phase=proj_data["current_phase"],
                            completed_tasks=proj_data.get("completed_tasks", []),
                            pending_tasks=proj_data.get("pending_tasks", []),
                            active_branches=proj_data.get("active_branches", []),
                            recent_deployments=proj_data.get("recent_deployments", []),
                            project_context=proj_data.get("project_context", ""),
                            last_updated=datetime.fromisoformat(proj_data["last_updated"])
                        )
            except Exception as e:
                print(f"Error loading project memories: {e}")
    
    def save_memory(self):
        """Save memory to disk"""
        # Save conversation history
        conv_file = self.storage_dir / "conversation_history.json"
        conv_data = []
        for entry in self.conversation_history:
            conv_data.append({
                "timestamp": entry.timestamp.isoformat(),
                "agent": entry.agent,
                "user_query": entry.user_query,
                "agent_response": entry.agent_response[:1000],  # Limit response size
                "task_id": entry.task_id,
                "knowledge_used": entry.knowledge_used,
                "success": entry.success,
                "metadata": entry.metadata
            })
        
        with open(conv_file, 'w') as f:
            json.dump(conv_data, f, indent=2)
        
        # Save project memories
        project_file = self.storage_dir / "project_memories.json"
        proj_data = {}
        for name, memory in self.project_memories.items():
            proj_data[name] = {
                "project_name": memory.project_name,
                "current_phase": memory.current_phase,
                "completed_tasks": memory.completed_tasks[-50:],  # Last 50 tasks
                "pending_tasks": memory.pending_tasks[:50],  # Next 50 tasks
                "active_branches": memory.active_branches,
                "recent_deployments": memory.recent_deployments[-10:],  # Last 10 deployments
                "project_context": memory.project_context,
                "last_updated": memory.last_updated.isoformat()
            }
        
        with open(project_file, 'w') as f:
            json.dump(proj_data, f, indent=2)
    
    def add_conversation(self, agent: str, user_query: str, agent_response: str, 
                        task_id: Optional[str] = None, knowledge_used: int = 0,
                        success: bool = True, metadata: Dict = None):
        """Add a conversation to shared memory"""
        with self.lock:
            entry = ConversationEntry(
                timestamp=datetime.now(),
                agent=agent,
                user_query=user_query,
                agent_response=agent_response,
                task_id=task_id,
                knowledge_used=knowledge_used,
                success=success,
                metadata=metadata or {}
            )
            
            # Add to conversation history
            self.conversation_history.append(entry)
            
            # Add to agent-specific history
            if agent not in self.agent_interactions:
                self.agent_interactions[agent] = []
            self.agent_interactions[agent].append(entry)
            
            # Add to memory manager for retrieval
            self.memory_manager.store_episodic({
                "type": "conversation",
                "agent": agent,
                "query": user_query,
                "response": agent_response[:500],  # Store summary
                "timestamp": entry.timestamp.isoformat(),
                "success": success
            })
            
            # Save to disk
            self.save_memory()
    
    def update_project_memory(self, project_name: str, **updates):
        """Update project-specific memory"""
        with self.lock:
            if project_name not in self.project_memories:
                self.project_memories[project_name] = ProjectMemory(project_name=project_name)
            
            memory = self.project_memories[project_name]
            
            for key, value in updates.items():
                if hasattr(memory, key):
                    if key in ["completed_tasks", "pending_tasks", "active_branches"]:
                        # Append to lists
                        current = getattr(memory, key)
                        if isinstance(value, list):
                            current.extend(value)
                        else:
                            current.append(value)
                    elif key == "recent_deployments":
                        memory.recent_deployments.append({
                            "timestamp": datetime.now().isoformat(),
                            "details": value
                        })
                    else:
                        setattr(memory, key, value)
            
            memory.last_updated = datetime.now()
            self.save_memory()
    
    def get_agent_context(self, agent_name: str) -> str:
        """
        Get comprehensive context for an agent including:
        - Recent conversations
        - Project status
        - Recent tasks
        - Relevant knowledge
        """
        context_parts = []
        
        # Add current project context
        current_project = None
        if hasattr(self.project_manager, 'get_active_project'):
            current_project = self.project_manager.get_active_project()
        elif hasattr(self.project_manager, 'projects') and self.project_manager.projects:
            # Get the first project if available
            current_project = next(iter(self.project_manager.projects.values()), None)
        
        if current_project:
            context_parts.append(f"## Current Project: {current_project.project_name}")
            context_parts.append(f"Description: {current_project.description}")
            
            # Add project memory if available
            if current_project.project_name in self.project_memories:
                memory = self.project_memories[current_project.project_name]
                context_parts.append(f"\nCurrent Phase: {memory.current_phase}")
                
                if memory.completed_tasks:
                    recent_completed = memory.completed_tasks[-5:]
                    context_parts.append(f"\nRecently Completed Tasks:")
                    for task in recent_completed:
                        context_parts.append(f"  - {task}")
                
                if memory.pending_tasks:
                    next_tasks = memory.pending_tasks[:5]
                    context_parts.append(f"\nUpcoming Tasks:")
                    for task in next_tasks:
                        context_parts.append(f"  - {task}")
                
                if memory.recent_deployments:
                    last_deployment = memory.recent_deployments[-1]
                    context_parts.append(f"\nLast Deployment: {last_deployment.get('timestamp', 'Unknown')}")
        
        # Add recent conversations context
        recent_convs = list(self.conversation_history)[-10:]  # Last 10 conversations
        if recent_convs:
            context_parts.append("\n## Recent Conversations:")
            for conv in recent_convs[-5:]:  # Show last 5
                context_parts.append(f"\n[{conv.timestamp.strftime('%H:%M')}] {conv.agent}:")
                context_parts.append(f"  Q: {conv.user_query[:100]}...")
                if conv.success:
                    context_parts.append(f"  ✓ Completed successfully")
        
        # Add agent-specific interaction history
        if agent_name in self.agent_interactions:
            agent_history = self.agent_interactions[agent_name][-5:]
            if agent_history:
                context_parts.append(f"\n## Your Recent Interactions:")
                for interaction in agent_history:
                    context_parts.append(f"  - {interaction.user_query[:100]}...")
        
        # Add relevant memories from memory manager
        try:
            recent_memories = self.memory_manager.retrieve_relevant(
                {"agent": agent_name, "limit": 5},
                k=5
            )
            if recent_memories:
                context_parts.append("\n## Relevant Memories:")
                # Handle both tuple (memory, score) and direct memory formats
                for item in recent_memories:
                    if isinstance(item, tuple):
                        memory = item[0]
                    else:
                        memory = item
                    
                    # Extract content based on memory type
                    if hasattr(memory, 'content'):
                        content = memory.content
                        if isinstance(content, dict):
                            if content.get("type") == "task_completion":
                                context_parts.append(f"  - Completed: {content.get('task', 'Unknown task')}")
                            elif content.get("type") == "error":
                                context_parts.append(f"  - Issue encountered: {content.get('error', 'Unknown error')}")
                            else:
                                context_parts.append(f"  - {content.get('type', 'Memory')}: {str(content)[:100]}...")
        except Exception as e:
            # Skip memory retrieval if there's an error
            pass
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of conversations in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_convs = [c for c in self.conversation_history 
                       if c.timestamp > cutoff_time]
        
        summary = {
            "total_conversations": len(recent_convs),
            "successful": sum(1 for c in recent_convs if c.success),
            "failed": sum(1 for c in recent_convs if not c.success),
            "agents_used": {},
            "common_queries": {},
            "knowledge_utilization": 0
        }
        
        # Count by agent
        for conv in recent_convs:
            if conv.agent not in summary["agents_used"]:
                summary["agents_used"][conv.agent] = 0
            summary["agents_used"][conv.agent] += 1
            
            # Track common query patterns
            query_words = conv.user_query.lower().split()[:3]
            query_start = " ".join(query_words)
            if query_start not in summary["common_queries"]:
                summary["common_queries"][query_start] = 0
            summary["common_queries"][query_start] += 1
            
            # Sum knowledge usage
            summary["knowledge_utilization"] += conv.knowledge_used
        
        # Get top 5 common queries
        if summary["common_queries"]:
            sorted_queries = sorted(summary["common_queries"].items(), 
                                  key=lambda x: x[1], reverse=True)
            summary["common_queries"] = dict(sorted_queries[:5])
        
        return summary
    
    def share_learning_across_agents(self):
        """
        Share successful patterns and solutions across all agents
        """
        # Get successful interactions
        successful_interactions = [
            c for c in self.conversation_history 
            if c.success and c.knowledge_used > 0
        ]
        
        # Extract patterns and add to all agents' knowledge
        for interaction in successful_interactions[-20:]:  # Last 20 successful
            # Create a learning item
            learning_item = {
                "query": interaction.user_query,
                "successful_approach": interaction.agent_response[:500],
                "agent_used": interaction.agent,
                "knowledge_items": interaction.knowledge_used
            }
            
            # Add to semantic memory for retrieval
            self.memory_manager.store_semantic({
                "type": "shared_learning",
                "pattern": learning_item,
                "timestamp": interaction.timestamp.isoformat()
            })
    
    def get_project_status_report(self) -> str:
        """Generate a comprehensive project status report"""
        report = []
        
        # Header
        report.append("# 📊 Project Status Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Active projects
        if self.project_memories:
            report.append("## 🗂️ Active Projects")
            for name, memory in self.project_memories.items():
                report.append(f"\n### {name}")
                report.append(f"- **Phase**: {memory.current_phase}")
                report.append(f"- **Completed Tasks**: {len(memory.completed_tasks)}")
                report.append(f"- **Pending Tasks**: {len(memory.pending_tasks)}")
                report.append(f"- **Active Branches**: {', '.join(memory.active_branches) if memory.active_branches else 'None'}")
                report.append(f"- **Last Updated**: {memory.last_updated.strftime('%Y-%m-%d %H:%M')}")
        
        # Conversation summary
        conv_summary = self.get_conversation_summary(24)
        report.append("\n## 💬 Last 24 Hours Activity")
        report.append(f"- **Total Conversations**: {conv_summary['total_conversations']}")
        report.append(f"- **Successful**: {conv_summary['successful']}")
        report.append(f"- **Failed**: {conv_summary['failed']}")
        report.append(f"- **Knowledge Items Used**: {conv_summary['knowledge_utilization']}")
        
        if conv_summary["agents_used"]:
            report.append("\n### Most Active Agents:")
            for agent, count in sorted(conv_summary["agents_used"].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"  - {agent}: {count} conversations")
        
        if conv_summary["common_queries"]:
            report.append("\n### Common Query Patterns:")
            for query, count in conv_summary["common_queries"].items():
                report.append(f"  - \"{query}...\": {count} times")
        
        # Memory statistics
        report.append("\n## 🧠 Memory Statistics")
        try:
            # Get memory counts directly
            episodic_count = len(self.memory_manager.episodic_memory) if hasattr(self.memory_manager, 'episodic_memory') else 0
            semantic_count = len(self.memory_manager.semantic_memory) if hasattr(self.memory_manager, 'semantic_memory') else 0
            working_count = len(self.memory_manager.working_memory) if hasattr(self.memory_manager, 'working_memory') else 0
            
            report.append(f"- **Episodic Memories**: {episodic_count}")
            report.append(f"- **Semantic Memories**: {semantic_count}")
            report.append(f"- **Working Memory Items**: {working_count}")
            report.append(f"- **Total Conversations**: {len(self.conversation_history)}")
        except Exception as e:
            report.append(f"- Memory statistics unavailable: {str(e)}")
        
        return "\n".join(report)


class SharedMemoryAPI:
    """API for shared memory system"""
    
    def __init__(self):
        self.memory_system = SharedMemorySystem()
    
    async def record_conversation(self, agent: str, query: str, response: str, 
                                 task_id: str = None, knowledge_used: int = 0,
                                 success: bool = True, metadata: Dict = None):
        """Record a conversation in shared memory"""
        self.memory_system.add_conversation(
            agent=agent,
            user_query=query,
            agent_response=response,
            task_id=task_id,
            knowledge_used=knowledge_used,
            success=success,
            metadata=metadata
        )
        return {"status": "recorded"}
    
    async def get_context(self, agent: str) -> Dict[str, str]:
        """Get context for an agent"""
        context = self.memory_system.get_agent_context(agent)
        return {"agent": agent, "context": context}
    
    async def update_project(self, project_name: str, updates: Dict) -> Dict:
        """Update project memory"""
        self.memory_system.update_project_memory(project_name, **updates)
        return {"status": "updated", "project": project_name}
    
    async def get_status_report(self) -> Dict[str, str]:
        """Get project status report"""
        report = self.memory_system.get_project_status_report()
        return {"report": report}
    
    async def share_learnings(self) -> Dict:
        """Share learnings across agents"""
        self.memory_system.share_learning_across_agents()
        return {"status": "learnings_shared"}


if __name__ == "__main__":
    # Example usage
    memory = SharedMemorySystem()
    
    print("🧠 Shared Memory System Initialized")
    print("=" * 60)
    
    # Example: Add a conversation
    memory.add_conversation(
        agent="database_specialist",
        user_query="How do I optimize a PostgreSQL query?",
        agent_response="To optimize PostgreSQL queries, use EXPLAIN ANALYZE...",
        knowledge_used=3,
        success=True
    )
    
    # Example: Update project memory
    memory.update_project_memory(
        "Agent Lightning",
        current_phase="Development",
        completed_tasks=["Setup database specialist", "Add training system"],
        pending_tasks=["Integrate shared memory", "Add visualization"]
    )
    
    # Get context for an agent
    context = memory.get_agent_context("database_specialist")
    print("\n📝 Context for database_specialist:")
    print(context)
    
    # Get project status report
    report = memory.get_project_status_report()
    print("\n" + report)
    
    print("\n✅ Shared Memory System Ready!")
    print("\nFeatures:")
    print("1. Records all agent conversations")
    print("2. Maintains project status and history")
    print("3. Shares context across all agents")
    print("4. Tracks task completion and deployments")
    print("5. Provides unified view of development")