#!/usr/bin/env python3
"""
PostgreSQL Memory System Setup for Agent Lightning
Creates database, tables, indexes, and sample data
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os
from datetime import datetime, timedelta

def check_postgres_connection():
    """Check if PostgreSQL is running and accessible"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            user="postgres",
            password="postgres",
            port=5432,
            database="postgres"
        )
        conn.close()
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("\n🔧 Setup Instructions:")
        print("1. Install PostgreSQL: brew install postgresql")
        print("2. Start PostgreSQL: brew services start postgresql")
        print("3. Create user: createuser -s postgres")
        print("4. Set password: psql -c \"ALTER USER postgres PASSWORD 'postgres';\"")
        return False

def setup_memory_database():
    """Setup complete memory database system"""
    
    if not check_postgres_connection():
        return False
    
    try:
        # Connect to postgres database to create our database
        conn = psycopg2.connect(
            host="localhost",
            user="postgres",
            password="postgres",
            port=5432,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Drop existing database if exists (for clean setup)
            cur.execute("DROP DATABASE IF EXISTS agent_lightning_memory")
            print("🗑️ Dropped existing database")
            
            # Create new database
            cur.execute("CREATE DATABASE agent_lightning_memory")
            print("✅ Created database: agent_lightning_memory")
        
        conn.close()
        
        # Connect to the new database and create tables
        from postgres_memory_manager import PostgreSQLMemoryManager
        
        print("🔧 Initializing memory manager...")
        memory_manager = PostgreSQLMemoryManager()
        
        # Create sample data for demonstration
        print("📝 Creating sample memory data...")
        
        # Sample agents
        agents = [
            "full_stack_developer", "ai_ml_engineer", "data_scientist", 
            "security_expert", "devops_engineer"
        ]
        
        # Sample episodic memories (conversations)
        for i, agent in enumerate(agents):
            memory_manager.store_episodic_memory(
                agent_id=agent,
                conversation_data={
                    "participants": ["user", agent],
                    "messages": [
                        {"role": "user", "content": f"Hello {agent}, can you help me?"},
                        {"role": "assistant", "content": f"Hello! I'm {agent.replace('_', ' ').title()}. I'd be happy to help you."}
                    ],
                    "duration": 120 + i * 30,
                    "outcome": "successful",
                    "tags": ["greeting", "introduction"]
                },
                context={"session_id": f"session_{i}", "platform": "dashboard"}
            )
        
        # Sample semantic knowledge
        knowledge_items = [
            {
                "agent": "full_stack_developer",
                "knowledge": {
                    "topic": "React Hooks",
                    "content": "useState and useEffect are fundamental React hooks for state management and side effects",
                    "examples": ["const [count, setCount] = useState(0)"],
                    "confidence": 0.9
                },
                "category": "frontend_development"
            },
            {
                "agent": "ai_ml_engineer", 
                "knowledge": {
                    "topic": "Neural Networks",
                    "content": "Deep learning models with multiple layers for complex pattern recognition",
                    "applications": ["image recognition", "NLP", "recommendation systems"],
                    "confidence": 0.95
                },
                "category": "machine_learning"
            },
            {
                "agent": "security_expert",
                "knowledge": {
                    "topic": "OAuth 2.0",
                    "content": "Authorization framework for secure API access using tokens",
                    "security_considerations": ["token expiration", "scope limitation", "HTTPS only"],
                    "confidence": 0.88
                },
                "category": "security"
            }
        ]
        
        for item in knowledge_items:
            memory_manager.store_semantic_knowledge(
                agent_id=item["agent"],
                knowledge_data=item["knowledge"],
                category=item["category"],
                source="training_data"
            )
        
        # Sample procedural skills
        skills = [
            {
                "agent": "devops_engineer",
                "skill": {
                    "name": "Docker Containerization",
                    "steps": ["Write Dockerfile", "Build image", "Run container", "Deploy to registry"],
                    "proficiency": 0.85,
                    "success_rate": 0.92,
                    "usage_patterns": ["CI/CD", "local development", "production deployment"]
                },
                "skill_type": "containerization"
            },
            {
                "agent": "data_scientist",
                "skill": {
                    "name": "Data Preprocessing",
                    "steps": ["Load data", "Clean missing values", "Feature engineering", "Normalization"],
                    "proficiency": 0.90,
                    "success_rate": 0.88,
                    "usage_patterns": ["ML pipeline", "data analysis", "model training"]
                },
                "skill_type": "data_processing"
            }
        ]
        
        for skill in skills:
            memory_manager.store_procedural_skill(
                agent_id=skill["agent"],
                skill_data=skill["skill"],
                skill_type=skill["skill_type"]
            )
        
        # Sample working memory (active contexts)
        for agent in agents[:3]:  # Only for first 3 agents
            memory_manager.store_working_memory(
                agent_id=agent,
                context_data={
                    "current_focus": f"Working on {agent.replace('_', ' ')} tasks",
                    "active_tasks": [f"task_1_{agent}", f"task_2_{agent}"],
                    "temporary_data": {"session_vars": {"user_id": "user123"}},
                    "priority": "high"
                },
                ttl_minutes=120  # 2 hours
            )
        
        # Sample task history
        sample_tasks = [
            ("Create a React component", "full_stack_developer", {"response": "Created a functional React component with hooks", "status": "completed"}, 45.2),
            ("Analyze customer data", "data_scientist", {"response": "Performed comprehensive data analysis with insights", "status": "completed"}, 120.5),
            ("Security audit", "security_expert", {"response": "Completed security audit with recommendations", "status": "completed"}, 180.3),
            ("Deploy application", "devops_engineer", {"response": "Successfully deployed to production", "status": "completed"}, 90.1),
            ("Train ML model", "ai_ml_engineer", {"response": "Trained and validated machine learning model", "status": "completed"}, 300.7)
        ]
        
        for task_desc, agent, result, exec_time in sample_tasks:
            memory_manager.store_task_result(
                agent_id=agent,
                task_description=task_desc,
                result=result,
                execution_time=exec_time,
                tokens_used=1500 + int(exec_time * 10),
                cost=0.002 * (1500 + int(exec_time * 10)) / 1000
            )
        
        # Sample learning sessions
        for agent in agents:
            memory_manager.record_learning_session(
                agent_id=agent,
                session_type="knowledge_integration",
                knowledge_processed=25,
                knowledge_integrated=20,
                performance_improvement=0.15
            )
        
        # Get and display statistics
        print("\n📊 Memory System Statistics:")
        stats = memory_manager.get_memory_statistics()
        
        print(f"Memory by Type:")
        for mem_type, data in stats['memory_by_type'].items():
            print(f"  {mem_type.title()}: {data['count']} items")
        
        print(f"\nTask Statistics:")
        task_stats = stats['task_statistics']
        print(f"  Total Tasks: {task_stats['total_tasks']}")
        print(f"  Avg Execution Time: {task_stats['avg_execution_time']:.2f}s")
        print(f"  Total Tokens: {task_stats['total_tokens']}")
        print(f"  Total Cost: ${task_stats['total_cost']:.4f}")
        
        print(f"\nLearning Statistics:")
        learning_stats = stats['learning_statistics']
        print(f"  Learning Sessions: {learning_stats['learning_sessions']}")
        print(f"  Knowledge Processed: {learning_stats['total_knowledge_processed']}")
        print(f"  Knowledge Integrated: {learning_stats['total_knowledge_integrated']}")
        print(f"  Avg Improvement: {learning_stats['avg_improvement']:.2%}")
        
        memory_manager.close()
        
        print("\n🎉 PostgreSQL Memory System setup complete!")
        print("\n🚀 Next Steps:")
        print("1. Run: streamlit run monitoring_dashboard_integrated.py")
        print("2. Go to 'Agent Knowledge' tab to see memory system")
        print("3. Use 'Task Assignment' tab to create new memories")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_system():
    """Test the memory system functionality"""
    try:
        from postgres_memory_manager import PostgreSQLMemoryManager
        
        print("🧪 Testing memory system...")
        memory_manager = PostgreSQLMemoryManager()
        
        # Test episodic memory
        episode_id = memory_manager.store_episodic_memory(
            agent_id="test_agent",
            conversation_data={
                "messages": [{"role": "user", "content": "Test message"}],
                "tags": ["test"]
            }
        )
        print(f"✅ Episodic memory stored: {episode_id}")
        
        # Test semantic knowledge
        knowledge_id = memory_manager.store_semantic_knowledge(
            agent_id="test_agent",
            knowledge_data={"topic": "Test Topic", "content": "Test content"},
            category="test_category"
        )
        print(f"✅ Semantic knowledge stored: {knowledge_id}")
        
        # Test search
        results = memory_manager.search_semantic_knowledge(
            agent_id="test_agent",
            query="Test"
        )
        print(f"✅ Search returned {len(results)} results")
        
        # Test task storage
        task_id = memory_manager.store_task_result(
            agent_id="test_agent",
            task_description="Test task",
            result={"response": "Test response"},
            execution_time=1.5
        )
        print(f"✅ Task result stored: {task_id}")
        
        # Test statistics
        stats = memory_manager.get_memory_statistics("test_agent")
        print(f"✅ Statistics retrieved for test_agent")
        
        memory_manager.close()
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Agent Lightning PostgreSQL Memory System Setup")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        success = test_memory_system()
    else:
        success = setup_memory_database()
    
    sys.exit(0 if success else 1)