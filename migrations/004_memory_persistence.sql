-- Memory Persistence Schema for Agent Learning
-- This migration adds comprehensive memory storage for agent experiences

-- Create enum for memory types
CREATE TYPE memory_type AS ENUM (
    'episodic',      -- Specific experiences/events
    'semantic',      -- General knowledge/facts
    'procedural',    -- How to do things
    'working',       -- Short-term active memory
    'consolidated'   -- Long-term consolidated memories
);

-- Create enum for memory importance levels
CREATE TYPE memory_importance AS ENUM (
    'critical',      -- Must never forget
    'high',          -- Very important
    'medium',        -- Standard importance
    'low',           -- Can be forgotten if needed
    'temporary'      -- Will be deleted after consolidation
);

-- Main memory storage table
CREATE TABLE IF NOT EXISTS agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    memory_type memory_type NOT NULL DEFAULT 'episodic',
    importance memory_importance NOT NULL DEFAULT 'medium',
    
    -- Memory content
    content JSONB NOT NULL,  -- Flexible storage for different memory structures
    embedding VECTOR(1536),   -- Vector embedding for similarity search
    
    -- Context and metadata
    context JSONB,            -- Additional context (task, environment, etc.)
    source VARCHAR(255),      -- Where this memory came from
    tags TEXT[],              -- Tags for categorization
    
    -- Temporal information
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    
    -- Memory strength and decay
    strength FLOAT DEFAULT 1.0,        -- Memory strength (0-1)
    decay_rate FLOAT DEFAULT 0.01,     -- How fast memory decays
    reinforcement_count INTEGER DEFAULT 0,  -- Times memory was reinforced
    
    -- Relationships
    parent_memory_id UUID REFERENCES agent_memories(id),  -- For hierarchical memories
    related_memories UUID[],  -- Array of related memory IDs
    
    -- Lifecycle
    is_consolidated BOOLEAN DEFAULT FALSE,
    consolidation_date TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,  -- For temporary memories
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for efficient querying
CREATE INDEX idx_agent_memories_agent_id ON agent_memories(agent_id);
CREATE INDEX idx_agent_memories_type ON agent_memories(memory_type);
CREATE INDEX idx_agent_memories_importance ON agent_memories(importance);
CREATE INDEX idx_agent_memories_created_at ON agent_memories(created_at);
CREATE INDEX idx_agent_memories_strength ON agent_memories(strength);
CREATE INDEX idx_agent_memories_tags ON agent_memories USING GIN(tags);
CREATE INDEX idx_agent_memories_content ON agent_memories USING GIN(content);

-- Create index for vector similarity search
CREATE INDEX idx_agent_memories_embedding ON agent_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Memory associations table (for linking memories)
CREATE TABLE IF NOT EXISTS memory_associations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_a_id UUID NOT NULL REFERENCES agent_memories(id) ON DELETE CASCADE,
    memory_b_id UUID NOT NULL REFERENCES agent_memories(id) ON DELETE CASCADE,
    association_type VARCHAR(50),  -- 'causal', 'temporal', 'semantic', etc.
    strength FLOAT DEFAULT 0.5,     -- Association strength
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(memory_a_id, memory_b_id)
);

-- Memory consolidation history
CREATE TABLE IF NOT EXISTS memory_consolidations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    consolidation_type VARCHAR(50),  -- 'daily', 'weekly', 'triggered'
    memories_processed INTEGER,
    memories_consolidated INTEGER,
    memories_pruned INTEGER,
    
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    
    metadata JSONB
);

-- Experience replay buffer for training
CREATE TABLE IF NOT EXISTS experience_replay_buffer (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES agent_memories(id) ON DELETE SET NULL,
    
    -- MDP components
    state JSONB NOT NULL,
    action JSONB NOT NULL,
    reward FLOAT NOT NULL,
    next_state JSONB NOT NULL,
    done BOOLEAN DEFAULT FALSE,
    
    -- Priority for prioritized experience replay
    priority FLOAT DEFAULT 1.0,
    sampling_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_replay_buffer_agent_id ON experience_replay_buffer(agent_id);
CREATE INDEX idx_replay_buffer_priority ON experience_replay_buffer(priority DESC);
CREATE INDEX idx_replay_buffer_created_at ON experience_replay_buffer(created_at);

-- Function to update memory strength based on access
CREATE OR REPLACE FUNCTION update_memory_strength()
RETURNS TRIGGER AS $$
BEGIN
    -- Increase strength slightly on access
    NEW.strength := LEAST(1.0, OLD.strength + 0.1);
    NEW.access_count := OLD.access_count + 1;
    NEW.accessed_at := CURRENT_TIMESTAMP;
    NEW.last_used := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to decay memory strength over time
CREATE OR REPLACE FUNCTION decay_memory_strength(agent_id_param VARCHAR(255))
RETURNS void AS $$
BEGIN
    UPDATE agent_memories
    SET strength = GREATEST(0.0, strength - (decay_rate * EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_used)) / 86400))
    WHERE agent_id = agent_id_param
    AND is_active = TRUE
    AND memory_type != 'consolidated';
END;
$$ LANGUAGE plpgsql;

-- Function to consolidate memories
CREATE OR REPLACE FUNCTION consolidate_memories(agent_id_param VARCHAR(255), threshold FLOAT DEFAULT 0.7)
RETURNS TABLE(consolidated_count INTEGER, pruned_count INTEGER) AS $$
DECLARE
    consol_count INTEGER := 0;
    prune_count INTEGER := 0;
BEGIN
    -- Mark strong memories as consolidated
    UPDATE agent_memories
    SET is_consolidated = TRUE,
        consolidation_date = CURRENT_TIMESTAMP,
        memory_type = 'consolidated'
    WHERE agent_id = agent_id_param
    AND strength >= threshold
    AND is_consolidated = FALSE
    AND is_active = TRUE;
    
    GET DIAGNOSTICS consol_count = ROW_COUNT;
    
    -- Prune weak temporary memories
    UPDATE agent_memories
    SET is_active = FALSE
    WHERE agent_id = agent_id_param
    AND strength < 0.1
    AND memory_type = 'temporary'
    AND is_active = TRUE;
    
    GET DIAGNOSTICS prune_count = ROW_COUNT;
    
    RETURN QUERY SELECT consol_count, prune_count;
END;
$$ LANGUAGE plpgsql;

-- Add trigger for memory access
CREATE TRIGGER trigger_memory_access
BEFORE UPDATE ON agent_memories
FOR EACH ROW
WHEN (NEW.accessed_at > OLD.accessed_at)
EXECUTE FUNCTION update_memory_strength();

-- Grant permissions
GRANT ALL ON agent_memories TO agent_user;
GRANT ALL ON memory_associations TO agent_user;
GRANT ALL ON memory_consolidations TO agent_user;
GRANT ALL ON experience_replay_buffer TO agent_user;
GRANT USAGE ON TYPE memory_type TO agent_user;
GRANT USAGE ON TYPE memory_importance TO agent_user;