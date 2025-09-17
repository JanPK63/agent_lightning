#!/usr/bin/env python3
"""
Initialize PostgreSQL database for Agent Lightning
Creates tables, indexes, and loads default data
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.database import db_manager
from shared.models import Agent, Task, Knowledge, Workflow, Session, Metric, User
import json
from datetime import datetime, timedelta
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_default_agents():
    """Create the 10 default agents in database"""
    
    default_agents = [
        {
            "id": "researcher",
            "name": "Research Specialist",
            "model": "claude-3-haiku",
            "specialization": "researcher",
            "capabilities": ["research", "analysis", "summarization", "fact-checking"],
            "config": {"max_tokens": 4096, "temperature": 0.7}
        },
        {
            "id": "developer",
            "name": "Code Developer",
            "model": "claude-3-sonnet",
            "specialization": "developer",
            "capabilities": ["coding", "debugging", "refactoring", "testing"],
            "config": {"max_tokens": 8192, "temperature": 0.5}
        },
        {
            "id": "reviewer",
            "name": "Code Reviewer",
            "model": "claude-3-haiku",
            "specialization": "reviewer",
            "capabilities": ["code-review", "best-practices", "security-audit"],
            "config": {"max_tokens": 4096, "temperature": 0.3}
        },
        {
            "id": "optimizer",
            "name": "Performance Optimizer",
            "model": "claude-3-haiku",
            "specialization": "optimizer",
            "capabilities": ["performance-tuning", "profiling", "optimization"],
            "config": {"max_tokens": 4096, "temperature": 0.4}
        },
        {
            "id": "writer",
            "name": "Documentation Writer",
            "model": "claude-3-haiku",
            "specialization": "writer",
            "capabilities": ["documentation", "technical-writing", "content-creation"],
            "config": {"max_tokens": 4096, "temperature": 0.7}
        },
        {
            "id": "security_expert",
            "name": "Security Expert",
            "model": "claude-3-opus",
            "specialization": "security_expert",
            "capabilities": ["security-analysis", "vulnerability-assessment", "compliance"],
            "config": {"max_tokens": 8192, "temperature": 0.2}
        },
        {
            "id": "data_scientist",
            "name": "Data Scientist",
            "model": "claude-3-sonnet",
            "specialization": "data_scientist",
            "capabilities": ["data-analysis", "machine-learning", "visualization", "statistics"],
            "config": {"max_tokens": 8192, "temperature": 0.5}
        },
        {
            "id": "devops_engineer",
            "name": "DevOps Engineer",
            "model": "claude-3-haiku",
            "specialization": "devops_engineer",
            "capabilities": ["infrastructure", "deployment", "monitoring", "automation"],
            "config": {"max_tokens": 4096, "temperature": 0.4}
        },
        {
            "id": "blockchain_developer",
            "name": "Blockchain Developer",
            "model": "claude-3-haiku",
            "specialization": "blockchain_developer",
            "capabilities": ["smart-contracts", "web3", "defi", "cryptography"],
            "config": {"max_tokens": 4096, "temperature": 0.3}
        },
        {
            "id": "system_architect",
            "name": "System Architect",
            "model": "claude-3-opus",
            "specialization": "system_architect",
            "capabilities": ["architecture-design", "system-design", "scalability", "integration"],
            "config": {"max_tokens": 8192, "temperature": 0.5}
        }
    ]
    
    created_count = 0
    with db_manager.get_db() as db:
        for agent_data in default_agents:
            # Check if agent exists
            existing = db.query(Agent).filter(Agent.id == agent_data['id']).first()
            
            if not existing:
                db_agent = Agent(**agent_data)
                db.add(db_agent)
                created_count += 1
                logger.info(f"Created agent: {agent_data['name']}")
            else:
                # Update existing agent
                for key, value in agent_data.items():
                    setattr(existing, key, value)
                logger.info(f"Updated agent: {agent_data['name']}")
    
    logger.info(f"✅ Created/updated {len(default_agents)} agents in database")
    return created_count

def migrate_knowledge_base():
    """Migrate existing knowledge base files to database"""
    
    knowledge_dir = Path(".agent-knowledge")
    if not knowledge_dir.exists():
        logger.info("No existing knowledge base to migrate")
        return 0
    
    migrated_count = 0
    
    with db_manager.get_db() as db:
        for json_file in knowledge_dir.glob("*.json"):
            agent_id = json_file.stem
            
            try:
                with open(json_file) as f:
                    knowledge_items = json.load(f)
                
                for item in knowledge_items:
                    # Check if knowledge item exists
                    existing = db.query(Knowledge).filter(
                        Knowledge.id == item.get('id', '')
                    ).first()
                    
                    if not existing:
                        db_knowledge = Knowledge(
                            id=item.get('id', f"{agent_id}_{datetime.now().timestamp()}"),
                            agent_id=agent_id,
                            category=item.get('category', 'general'),
                            content=item.get('content', ''),
                            source=item.get('source', 'imported'),
                            metadata=item.get('metadata', {}),
                            usage_count=item.get('usage_count', 0),
                            relevance_score=item.get('relevance_score', 1.0)
                        )
                        db.add(db_knowledge)
                        migrated_count += 1
                
                logger.info(f"Migrated knowledge for agent: {agent_id}")
                
            except Exception as e:
                logger.error(f"Failed to migrate knowledge for {agent_id}: {e}")
    
    logger.info(f"✅ Migrated {migrated_count} knowledge items to database")
    return migrated_count

def create_admin_user():
    """Create default admin user"""
    
    with db_manager.get_db() as db:
        # Check if admin exists
        existing = db.query(User).filter(User.username == "admin").first()
        
        if not existing:
            # Create password hash (in production, use bcrypt or argon2)
            password_hash = hashlib.sha256("admin123".encode()).hexdigest()
            
            admin_user = User(
                username="admin",
                email="admin@agentlightning.ai",
                password_hash=password_hash,
                role="admin",
                is_active=True
            )
            db.add(admin_user)
            logger.info("✅ Created admin user (username: admin, password: admin123)")
        else:
            logger.info("Admin user already exists")

def create_sample_data():
    """Create sample tasks and workflows for testing"""
    
    with db_manager.get_db() as db:
        # Create sample tasks
        sample_tasks = [
            {
                "agent_id": "researcher",
                "description": "Research latest AI safety developments",
                "status": "completed",
                "priority": "high",
                "context": {"source": "sample_data"},
                "result": {"summary": "Sample research completed"}
            },
            {
                "agent_id": "developer",
                "description": "Implement new feature for user dashboard",
                "status": "pending",
                "priority": "normal",
                "context": {"source": "sample_data"}
            }
        ]
        
        for task_data in sample_tasks:
            task = Task(**task_data)
            db.add(task)
        
        # Create sample workflow
        workflow = Workflow(
            name="Data Analysis Pipeline",
            description="Complete data analysis workflow",
            steps=[
                {"step": 1, "agent": "data_scientist", "action": "analyze_data"},
                {"step": 2, "agent": "writer", "action": "create_report"},
                {"step": 3, "agent": "reviewer", "action": "review_report"}
            ],
            status="draft",
            created_by="admin"
        )
        db.add(workflow)
        
        logger.info("✅ Created sample data")

def main():
    """Main initialization function"""
    
    logger.info("=" * 60)
    logger.info("🚀 Agent Lightning Database Initialization")
    logger.info("=" * 60)
    
    # Check database connection
    if not db_manager.health_check():
        logger.error("❌ Failed to connect to database")
        logger.error("Please ensure PostgreSQL is running and configured correctly")
        logger.error("Connection string: " + db_manager.database_url.replace(
            db_manager.database_url.split('@')[0].split('//')[1], "***:***"
        ))
        sys.exit(1)
    
    logger.info("✅ Database connection successful")
    
    # Initialize database tables
    logger.info("Creating database tables...")
    if db_manager.init_database():
        logger.info("✅ Database tables created")
    else:
        logger.error("❌ Failed to create database tables")
        sys.exit(1)
    
    # Create default agents
    logger.info("Creating default agents...")
    create_default_agents()
    
    # Migrate existing knowledge base
    logger.info("Migrating knowledge base...")
    migrate_knowledge_base()
    
    # Create admin user
    logger.info("Creating admin user...")
    create_admin_user()
    
    # Create sample data
    logger.info("Creating sample data...")
    create_sample_data()
    
    # Show connection pool status
    pool_status = db_manager.get_pool_status()
    logger.info(f"Connection pool status: {pool_status}")
    
    logger.info("=" * 60)
    logger.info("✨ Database initialization complete!")
    logger.info("=" * 60)
    
    # Close connections
    db_manager.close()

if __name__ == "__main__":
    main()