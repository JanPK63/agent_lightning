#!/usr/bin/env python3
"""
JSON to PostgreSQL Migration Script
Migrates Agent Lightning data from JSON files to PostgreSQL database
"""

import json
import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from shared.models import Base, Agent, Knowledge
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONToPostgresMigration:
    def __init__(self, db_url="postgresql://localhost/agent_lightning"):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")
    
    def migrate_agents(self):
        """Migrate agent data from JSON files"""
        session = self.Session()
        
        # Define all agents from your list
        agents_data = [
            {"id": "security_expert", "name": "Security Expert", "specialization": "security"},
            {"id": "researcher", "name": "Test Researcher", "specialization": "research"},
            {"id": "data_scientist", "name": "Data Scientist", "specialization": "data_analysis"},
            {"id": "tester", "name": "Tester Agent", "specialization": "testing"},
            {"id": "devops_engineer", "name": "DevOps Engineer", "specialization": "infrastructure"},
            {"id": "reviewer", "name": "Reviewer Agent", "specialization": "code_review"},
            {"id": "full_stack_developer", "name": "Full Stack Developer", "specialization": "web_development"},
            {"id": "integrator", "name": "Integrator Agent", "specialization": "integration"},
            {"id": "router", "name": "Router Agent", "specialization": "routing"},
            {"id": "planner", "name": "Planner Agent", "specialization": "planning"},
            {"id": "ui_ux_designer", "name": "UI/UX Designer", "specialization": "design"},
            {"id": "test_engineer", "name": "Test Engineer", "specialization": "testing"},
            {"id": "database_specialist", "name": "Database Specialist", "specialization": "database"},
            {"id": "retriever", "name": "Retriever Agent", "specialization": "information_retrieval"},
            {"id": "blockchain_developer", "name": "Blockchain Developer", "specialization": "blockchain"},
            {"id": "system_architect", "name": "System Architect", "specialization": "architecture"},
            {"id": "information_analyst", "name": "Information Analyst", "specialization": "analysis"},
            {"id": "executor", "name": "Executor", "specialization": "task_execution"},
            {"id": "coder", "name": "Coder Agent", "specialization": "code_generation"},
            {"id": "mobile_developer", "name": "Mobile Developer", "specialization": "mobile"},
            {"id": "critic", "name": "Critic", "specialization": "evaluation"}
        ]
        
        for agent_data in agents_data:
            agent = Agent(
                id=agent_data["id"],
                name=agent_data["name"],
                specialization=agent_data["specialization"],
                model="gpt-4o",
                status="active"
            )
            session.merge(agent)
        
        session.commit()
        logger.info(f"Migrated {len(agents_data)} agents")
        session.close()
    
    def migrate_knowledge(self):
        """Migrate knowledge from JSON files to database"""
        session = self.Session()
        knowledge_dir = ".agent-knowledge"
        
        if not os.path.exists(knowledge_dir):
            logger.warning(f"Knowledge directory {knowledge_dir} not found")
            return
        
        total_items = 0
        
        for filename in os.listdir(knowledge_dir):
            if not filename.endswith('.json'):
                continue
                
            agent_id = filename.replace('.json', '')
            filepath = os.path.join(knowledge_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict) and 'knowledge_items' in data:
                    items = data['knowledge_items']
                else:
                    # Treat entire content as single knowledge item
                    items = [{
                        'id': f"{agent_id}_main",
                        'category': 'general',
                        'content': json.dumps(data),
                        'source': 'json_migration'
                    }]
                
                for item in items:
                    knowledge = Knowledge(
                        id=item.get('id', f"{agent_id}_{total_items}"),
                        agent_id=agent_id,
                        category=item.get('category', 'general'),
                        content=item.get('content', str(item)),
                        source=item.get('source', 'json_migration'),
                        usage_count=item.get('usage_count', 0),
                        relevance_score=item.get('relevance_score', 1.0)
                    )
                    session.merge(knowledge)
                    total_items += 1
                
                logger.info(f"Processed {filename}: {len(items)} items")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
        
        session.commit()
        logger.info(f"Migrated {total_items} knowledge items")
        session.close()
    
    def run_migration(self):
        """Run complete migration"""
        logger.info("Starting JSON to PostgreSQL migration")
        
        try:
            self.create_tables()
            self.migrate_agents()
            self.migrate_knowledge()
            logger.info("Migration completed successfully")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

if __name__ == "__main__":
    # Default connection string - update as needed
    DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost/agent_lightning")
    
    migration = JSONToPostgresMigration(DB_URL)
    migration.run_migration()