-- Checkpoint System for Model Recovery
-- Enables saving and restoring agent states, models, and training progress

-- Create enum for checkpoint types
CREATE TYPE checkpoint_type AS ENUM (
    'model_state',      -- Neural network weights and parameters
    'training_state',   -- Optimizer state, learning rate, epoch
    'agent_state',      -- Agent configuration and memory
    'workflow_state',   -- Workflow execution state
    'full_snapshot'     -- Complete system snapshot
);

-- Create enum for checkpoint status
CREATE TYPE checkpoint_status AS ENUM (
    'creating',         -- Checkpoint being created
    'ready',           -- Available for restoration
    'restoring',       -- Currently being restored
    'archived',        -- Moved to cold storage
    'failed',          -- Creation or restoration failed
    'deleted'          -- Soft deleted
);

-- Main checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE CASCADE,
    workflow_id VARCHAR(255),
    
    -- Checkpoint metadata
    checkpoint_type checkpoint_type NOT NULL,
    checkpoint_status checkpoint_status NOT NULL DEFAULT 'creating',
    version VARCHAR(50),
    
    -- Training information
    epoch INTEGER,
    step INTEGER,
    training_loss FLOAT,
    validation_loss FLOAT,
    metrics JSONB,
    
    -- Storage information
    storage_path TEXT NOT NULL,  -- S3, local path, etc.
    file_size_bytes BIGINT,
    checksum VARCHAR(64),        -- SHA256 for integrity
    compression_type VARCHAR(20), -- gzip, bzip2, none
    
    -- State data (for small states)
    state_data JSONB,            -- Small state data stored directly
    model_architecture JSONB,    -- Model architecture definition
    hyperparameters JSONB,       -- Training hyperparameters
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    restored_at TIMESTAMP WITH TIME ZONE,
    restored_count INTEGER DEFAULT 0,
    
    -- Lifecycle
    expires_at TIMESTAMP WITH TIME ZONE,
    is_auto_checkpoint BOOLEAN DEFAULT FALSE,
    is_best_model BOOLEAN DEFAULT FALSE,
    parent_checkpoint_id UUID REFERENCES checkpoints(id),
    
    -- Description
    name VARCHAR(255),
    description TEXT,
    tags TEXT[]
);

-- Create indexes for efficient querying
CREATE INDEX idx_checkpoints_agent_id ON checkpoints(agent_id);
CREATE INDEX idx_checkpoints_workflow_id ON checkpoints(workflow_id);
CREATE INDEX idx_checkpoints_type ON checkpoints(checkpoint_type);
CREATE INDEX idx_checkpoints_status ON checkpoints(checkpoint_status);
CREATE INDEX idx_checkpoints_created_at ON checkpoints(created_at DESC);
CREATE INDEX idx_checkpoints_is_best ON checkpoints(is_best_model) WHERE is_best_model = TRUE;
CREATE INDEX idx_checkpoints_tags ON checkpoints USING GIN(tags);

-- Checkpoint restoration history
CREATE TABLE IF NOT EXISTS checkpoint_restorations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_id UUID NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    agent_id VARCHAR(255) REFERENCES agents(id),
    
    -- Restoration details
    restored_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    restored_by VARCHAR(255),
    restoration_time_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    -- Pre-restoration state (for rollback)
    previous_checkpoint_id UUID REFERENCES checkpoints(id),
    rollback_performed BOOLEAN DEFAULT FALSE,
    
    metadata JSONB
);

-- Model registry for versioning
CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    agent_id VARCHAR(255) REFERENCES agents(id),
    
    -- Model information
    framework VARCHAR(50),        -- pytorch, tensorflow, etc.
    architecture_type VARCHAR(100), -- transformer, cnn, rnn, etc.
    parameter_count BIGINT,
    
    -- Performance metrics
    training_metrics JSONB,
    validation_metrics JSONB,
    test_metrics JSONB,
    
    -- Checkpoint reference
    checkpoint_id UUID REFERENCES checkpoints(id),
    
    -- Deployment information
    is_deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    deployment_endpoint TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    description TEXT,
    tags TEXT[],
    
    UNIQUE(model_name, model_version)
);

-- Incremental checkpoint deltas (for efficient storage)
CREATE TABLE IF NOT EXISTS checkpoint_deltas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    base_checkpoint_id UUID NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    delta_checkpoint_id UUID NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    
    -- Delta information
    delta_size_bytes BIGINT,
    compression_ratio FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Changed parameters
    changed_parameters TEXT[],
    parameter_changes JSONB,
    
    UNIQUE(base_checkpoint_id, delta_checkpoint_id)
);

-- Checkpoint scheduling rules
CREATE TABLE IF NOT EXISTS checkpoint_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE CASCADE,
    
    -- Schedule configuration
    schedule_type VARCHAR(50),    -- 'periodic', 'on_improvement', 'on_milestone'
    interval_minutes INTEGER,      -- For periodic checkpoints
    improvement_threshold FLOAT,   -- For improvement-based checkpoints
    
    -- Retention policy
    max_checkpoints INTEGER DEFAULT 10,
    retention_days INTEGER DEFAULT 30,
    keep_best_n INTEGER DEFAULT 3,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_checkpoint_at TIMESTAMP WITH TIME ZONE,
    next_checkpoint_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Function to clean up old checkpoints
CREATE OR REPLACE FUNCTION cleanup_old_checkpoints()
RETURNS void AS $$
BEGIN
    -- Archive old checkpoints
    UPDATE checkpoints
    SET checkpoint_status = 'archived'
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
      AND checkpoint_status = 'ready'
      AND is_best_model = FALSE
      AND restored_count = 0;
    
    -- Delete very old archived checkpoints
    UPDATE checkpoints
    SET checkpoint_status = 'deleted'
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days'
      AND checkpoint_status = 'archived';
END;
$$ LANGUAGE plpgsql;

-- Function to mark best checkpoint
CREATE OR REPLACE FUNCTION mark_best_checkpoint(
    agent_id_param VARCHAR(255),
    checkpoint_id_param UUID
)
RETURNS void AS $$
BEGIN
    -- Unmark current best
    UPDATE checkpoints
    SET is_best_model = FALSE
    WHERE agent_id = agent_id_param
      AND is_best_model = TRUE;
    
    -- Mark new best
    UPDATE checkpoints
    SET is_best_model = TRUE
    WHERE id = checkpoint_id_param;
END;
$$ LANGUAGE plpgsql;

-- Function to create incremental checkpoint
CREATE OR REPLACE FUNCTION create_incremental_checkpoint(
    base_checkpoint_id UUID,
    new_state JSONB,
    storage_path TEXT
)
RETURNS UUID AS $$
DECLARE
    new_checkpoint_id UUID;
    base_state JSONB;
    delta JSONB;
BEGIN
    -- Get base state
    SELECT state_data INTO base_state
    FROM checkpoints
    WHERE id = base_checkpoint_id;
    
    -- Calculate delta (simplified - in practice would be more sophisticated)
    delta := new_state;
    
    -- Create new checkpoint
    INSERT INTO checkpoints (
        parent_checkpoint_id,
        agent_id,
        checkpoint_type,
        checkpoint_status,
        storage_path,
        state_data
    )
    SELECT 
        base_checkpoint_id,
        agent_id,
        checkpoint_type,
        'ready',
        storage_path,
        delta
    FROM checkpoints
    WHERE id = base_checkpoint_id
    RETURNING id INTO new_checkpoint_id;
    
    -- Record delta relationship
    INSERT INTO checkpoint_deltas (
        base_checkpoint_id,
        delta_checkpoint_id
    )
    VALUES (base_checkpoint_id, new_checkpoint_id);
    
    RETURN new_checkpoint_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL ON checkpoints TO agent_user;
GRANT ALL ON checkpoint_restorations TO agent_user;
GRANT ALL ON model_registry TO agent_user;
GRANT ALL ON checkpoint_deltas TO agent_user;
GRANT ALL ON checkpoint_schedules TO agent_user;
GRANT USAGE ON TYPE checkpoint_type TO agent_user;
GRANT USAGE ON TYPE checkpoint_status TO agent_user;