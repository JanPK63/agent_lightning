-- Agent Lightning PostgreSQL Database Schema
-- Version: 1.0.0
-- Date: 2025-09-05

-- Create database (run as superuser)
-- CREATE DATABASE agent_lightning;
-- CREATE USER agent_user WITH ENCRYPTED PASSWORD 'agent_pass';
-- GRANT ALL PRIVILEGES ON DATABASE agent_lightning TO agent_user;

-- Connect to agent_lightning database
-- \c agent_lightning;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
-- CREATE EXTENSION IF NOT EXISTS "vector"; -- For embeddings (requires pgvector)

-- Set timezone
SET timezone = 'UTC';

-- Agents table: Core agent definitions
CREATE TABLE IF NOT EXISTS agents (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model VARCHAR(50) NOT NULL,
    specialization VARCHAR(50),
    status VARCHAR(20) DEFAULT 'idle',
    config JSONB DEFAULT '{}',
    capabilities JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table: Agent task tracking
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(50) REFERENCES agents(id) ON DELETE CASCADE,
    description TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    priority VARCHAR(20) DEFAULT 'normal',
    context JSONB DEFAULT '{}',
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Knowledge base: Agent learning storage
CREATE TABLE IF NOT EXISTS knowledge (
    id VARCHAR(100) PRIMARY KEY,
    agent_id VARCHAR(50) REFERENCES agents(id) ON DELETE CASCADE,
    category VARCHAR(50),
    content TEXT,
    source VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    -- embedding VECTOR(1536), -- For similarity search (requires pgvector)
    usage_count INTEGER DEFAULT 0,
    relevance_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Workflows: Complex task orchestration
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100),
    description TEXT,
    steps JSONB,
    status VARCHAR(20) DEFAULT 'draft',
    created_by VARCHAR(50),
    assigned_to VARCHAR(50),
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sessions: User session management
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100),
    token VARCHAR(500) UNIQUE,
    data JSONB DEFAULT '{}',
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Metrics: Performance tracking
CREATE TABLE IF NOT EXISTS metrics (
    id BIGSERIAL PRIMARY KEY,
    service_name VARCHAR(50),
    metric_name VARCHAR(100),
    value FLOAT,
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users: System users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_agent_id ON knowledge(agent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(agent_id, category);
CREATE INDEX IF NOT EXISTS idx_knowledge_usage ON knowledge(usage_count DESC);

CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_by ON workflows(created_by);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_service ON metrics(service_name, metric_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (for future multi-tenancy)
-- ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE knowledge ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE workflows ENABLE ROW LEVEL SECURITY;

-- Create views for common queries
CREATE OR REPLACE VIEW agent_task_stats AS
SELECT 
    a.id as agent_id,
    a.name as agent_name,
    COUNT(t.id) as total_tasks,
    COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed_tasks,
    COUNT(CASE WHEN t.status = 'pending' THEN 1 END) as pending_tasks,
    COUNT(CASE WHEN t.status = 'failed' THEN 1 END) as failed_tasks,
    AVG(EXTRACT(EPOCH FROM (t.completed_at - t.started_at))) as avg_completion_time_seconds
FROM agents a
LEFT JOIN tasks t ON a.id = t.agent_id
GROUP BY a.id, a.name;

CREATE OR REPLACE VIEW recent_metrics AS
SELECT *
FROM metrics
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;

-- Grant permissions to agent_user
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO agent_user;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO agent_user;
-- GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO agent_user;

-- Sample data comment
-- Run scripts/init_database.py to populate with default agents and sample data