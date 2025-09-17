-- Agent Lightning PostgreSQL Schema
-- Migration from JSON to relational database

CREATE DATABASE agent_lightning;

-- Agents table
CREATE TABLE agents (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    specialization VARCHAR(100),
    model VARCHAR(50) DEFAULT 'gpt-4o',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge base table
CREATE TABLE knowledge_items (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) REFERENCES agents(id),
    category VARCHAR(50),
    content TEXT,
    source VARCHAR(200),
    relevance_score FLOAT DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table
CREATE TABLE tasks (
    id VARCHAR(50) PRIMARY KEY,
    agent_id VARCHAR(50) REFERENCES agents(id),
    description TEXT,
    result TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    execution_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_knowledge_agent ON knowledge_items(agent_id);
CREATE INDEX idx_knowledge_category ON knowledge_items(category);
CREATE INDEX idx_tasks_agent ON tasks(agent_id);
CREATE INDEX idx_tasks_status ON tasks(status);