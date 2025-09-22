#!/usr/bin/env python3
"""
Populate database with agents from agent_capability_matcher.py

This script reads the agents defined in agent_capability_matcher.py
and inserts them into the SQLite database.
"""

import sqlite3
import json
from datetime import datetime

# Agent definitions from agent_capability_matcher.py
capability_agents = [
    {
        "id": "web_developer",
        "name": "Web Developer",
        "model": "gpt-4o",
        "specialization": "web_development",
        "capabilities": ["web development", "frontend", "backend", "html", "css", "javascript"],
        "config": {"max_tokens": 4096, "temperature": 0.7},
        "status": "idle"
    },
    {
        "id": "security_expert",
        "name": "Security Expert",
        "model": "gpt-4o",
        "specialization": "security_expert",
        "capabilities": ["security", "penetration testing", "vulnerability assessment", "encryption"],
        "config": {"max_tokens": 4096, "temperature": 0.2},
        "status": "idle"
    },
    {
        "id": "data_analyst",
        "name": "Data Analyst",
        "model": "gpt-4o",
        "specialization": "data_analysis",
        "capabilities": ["data analysis", "statistics", "visualization", "machine learning"],
        "config": {"max_tokens": 4096, "temperature": 0.6},
        "status": "idle"
    },
    {
        "id": "devops_engineer",
        "name": "DevOps Engineer",
        "model": "gpt-4o",
        "specialization": "devops_engineer",
        "capabilities": ["deployment", "infrastructure", "ci/cd", "monitoring"],
        "config": {"max_tokens": 4096, "temperature": 0.4},
        "status": "idle"
    },
    {
        "id": "qa_tester",
        "name": "QA Tester",
        "model": "gpt-4o",
        "specialization": "qa_tester",
        "capabilities": ["testing", "quality assurance", "automation"],
        "config": {"max_tokens": 4096, "temperature": 0.5},
        "status": "idle"
    },
    {
        "id": "general_assistant",
        "name": "General Assistant",
        "model": "gpt-4o",
        "specialization": "general_assistant",
        "capabilities": ["general tasks", "coordination", "documentation"],
        "config": {"max_tokens": 4096, "temperature": 0.7},
        "status": "idle"
    }
]

def populate_agents():
    """Populate the agents table with capability matcher agents"""

    conn = sqlite3.connect('agentlightning.db')
    cursor = conn.cursor()

    # Check if agents table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agents';")
    if not cursor.fetchone():
        print("❌ Agents table does not exist. Run database initialization first.")
        conn.close()
        return False

    # Insert agents
    inserted_count = 0
    for agent in capability_agents:
        try:
            # Check if agent already exists
            cursor.execute("SELECT id FROM agents WHERE id = ?", (agent['id'],))
            if cursor.fetchone():
                print(f"⚠️  Agent {agent['id']} already exists, skipping")
                continue

            # Insert new agent
            cursor.execute("""
                INSERT INTO agents (id, name, model, specialization, status, config, capabilities, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent['id'],
                agent['name'],
                agent['model'],
                agent['specialization'],
                agent['status'],
                json.dumps(agent['config']),
                json.dumps(agent['capabilities']),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            inserted_count += 1
            print(f"✅ Added agent: {agent['name']} ({agent['id']})")

        except Exception as e:
            print(f"❌ Failed to add agent {agent['id']}: {e}")

    conn.commit()
    conn.close()

    print(f"\n✅ Successfully added {inserted_count} agents to database")
    return True

def verify_agents():
    """Verify agents were added correctly"""
    conn = sqlite3.connect('agentlightning.db')
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, specialization FROM agents ORDER BY id;")
    agents = cursor.fetchall()

    print("\n📋 Agents in database:")
    print("-" * 50)
    for agent in agents:
        print(f"ID: {agent[0]}, Name: {agent[1]}, Specialization: {agent[2]}")

    conn.close()

if __name__ == "__main__":
    print("🚀 Populating database with capability matcher agents...")
    print("=" * 60)

    success = populate_agents()
    if success:
        verify_agents()

    print("=" * 60)
    print("✨ Database population complete!")