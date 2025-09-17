#!/usr/bin/env python3
"""
PostgreSQL setup for Agent Lightning task history persistence
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    """Create database and tables for Agent Lightning"""
    
    # Connect to PostgreSQL server (default database)
    try:
        conn = psycopg2.connect(
            host="localhost",
            user="postgres", 
            password="postgres",
            port=5432
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Create database if not exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = 'agent_lightning'")
            if not cur.fetchone():
                cur.execute("CREATE DATABASE agent_lightning")
                print("✅ Created database: agent_lightning")
            else:
                print("✅ Database agent_lightning already exists")
        
        conn.close()
        
        # Connect to the agent_lightning database
        conn = psycopg2.connect(
            host="localhost",
            database="agent_lightning",
            user="postgres",
            password="postgres",
            port=5432
        )
        
        with conn.cursor() as cur:
            # Create task_history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    task_description TEXT NOT NULL,
                    agent_id VARCHAR(100),
                    result TEXT,
                    status VARCHAR(50) DEFAULT 'completed',
                    execution_time FLOAT
                )
            """)
            
            # Create index for faster queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_history_timestamp 
                ON task_history(timestamp DESC)
            """)
            
            conn.commit()
            print("✅ Created task_history table with index")
        
        conn.close()
        print("🎉 PostgreSQL setup complete!")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("Make sure PostgreSQL is running and accessible with user 'postgres'")

if __name__ == "__main__":
    setup_database()