-- Encryption Keys Management System
-- Database schema for managing encryption keys and key rotation

-- Create enum for key types
CREATE TYPE key_type AS ENUM (
    'master',      -- Master encryption key
    'data',        -- Data encryption key (per table)
    'field',       -- Field encryption key (per sensitive field)
    'record'       -- Record encryption key (per database record)
);

-- Create enum for key status
CREATE TYPE key_status AS ENUM (
    'active',      -- Currently in use
    'rotating',    -- In rotation process
    'deprecated',  -- Deprecated but still valid
    'compromised', -- Compromised, do not use
    'destroyed'    -- Securely destroyed
);

-- Create enum for encryption algorithms
CREATE TYPE encryption_algorithm AS ENUM (
    'aes-256-gcm',     -- AES-256 with GCM mode
    'chacha20-poly1305' -- ChaCha20-Poly1305
);

-- Main encryption keys table
CREATE TABLE IF NOT EXISTS encryption_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_type key_type NOT NULL,
    key_id VARCHAR(255) NOT NULL UNIQUE, -- Unique identifier for the key
    name VARCHAR(255), -- Human-readable name
    description TEXT,

    -- Key data (encrypted)
    encrypted_key BYTEA NOT NULL, -- The actual encrypted key
    key_hash VARCHAR(64) NOT NULL, -- SHA256 hash for integrity verification
    algorithm encryption_algorithm NOT NULL DEFAULT 'aes-256-gcm',

    -- Key hierarchy
    parent_key_id UUID REFERENCES encryption_keys(id), -- Parent key in hierarchy
    derived_from VARCHAR(255), -- How this key was derived

    -- Lifecycle
    status key_status NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    activated_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    destroyed_at TIMESTAMP WITH TIME ZONE,

    -- Rotation
    rotation_count INTEGER DEFAULT 0,
    last_rotated_at TIMESTAMP WITH TIME ZONE,
    next_rotation_at TIMESTAMP WITH TIME ZONE,

    -- Security
    compromise_detected_at TIMESTAMP WITH TIME ZONE,
    security_level VARCHAR(20) DEFAULT 'standard', -- 'standard', 'high', 'critical'

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Key usage tracking table
CREATE TABLE IF NOT EXISTS key_usage_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id UUID NOT NULL REFERENCES encryption_keys(id) ON DELETE CASCADE,
    operation VARCHAR(50) NOT NULL, -- 'encrypt', 'decrypt', 'derive', 'rotate'
    field_name VARCHAR(255), -- Which field was encrypted/decrypted
    table_name VARCHAR(255), -- Which table
    record_id VARCHAR(255), -- Which record (if applicable)
    user_id VARCHAR(255), -- Who performed the operation
    ip_address INET, -- IP address of the operation
    user_agent TEXT, -- User agent string
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    performance_ms INTEGER, -- Operation time in milliseconds
    metadata JSONB DEFAULT '{}'
);

-- Key rotation history table
CREATE TABLE IF NOT EXISTS key_rotation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id UUID NOT NULL REFERENCES encryption_keys(id) ON DELETE CASCADE,
    old_key_hash VARCHAR(64),
    new_key_hash VARCHAR(64),
    rotation_reason VARCHAR(100), -- 'scheduled', 'manual', 'compromised', 'emergency'
    rotated_by VARCHAR(255),
    rotated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rotation_time_ms INTEGER,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    old_expires_at TIMESTAMP WITH TIME ZONE,
    new_expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- Key access audit table
CREATE TABLE IF NOT EXISTS key_access_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id UUID NOT NULL REFERENCES encryption_keys(id) ON DELETE CASCADE,
    access_type VARCHAR(50) NOT NULL, -- 'retrieve', 'store', 'delete', 'rotate'
    accessor_id VARCHAR(255), -- Who accessed the key
    accessor_type VARCHAR(50), -- 'service', 'user', 'system'
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_encryption_keys_type ON encryption_keys(key_type);
CREATE INDEX IF NOT EXISTS idx_encryption_keys_status ON encryption_keys(status);
CREATE INDEX IF NOT EXISTS idx_encryption_keys_key_id ON encryption_keys(key_id);
CREATE INDEX IF NOT EXISTS idx_encryption_keys_expires_at ON encryption_keys(expires_at);
CREATE INDEX IF NOT EXISTS idx_encryption_keys_next_rotation ON encryption_keys(next_rotation_at);
CREATE INDEX IF NOT EXISTS idx_encryption_keys_parent ON encryption_keys(parent_key_id);

CREATE INDEX IF NOT EXISTS idx_key_usage_key_id ON key_usage_log(key_id);
CREATE INDEX IF NOT EXISTS idx_key_usage_timestamp ON key_usage_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_key_usage_operation ON key_usage_log(operation);

CREATE INDEX IF NOT EXISTS idx_key_rotation_key_id ON key_rotation_history(key_id);
CREATE INDEX IF NOT EXISTS idx_key_rotation_timestamp ON key_rotation_history(rotated_at DESC);

CREATE INDEX IF NOT EXISTS idx_key_access_key_id ON key_access_audit(key_id);
CREATE INDEX IF NOT EXISTS idx_key_access_timestamp ON key_access_audit(timestamp DESC);

-- Function to generate key ID
CREATE OR REPLACE FUNCTION generate_key_id(key_type key_type, identifier VARCHAR(255))
RETURNS VARCHAR(255) AS $$
BEGIN
    RETURN key_type || '_' || identifier || '_' || encode(gen_random_bytes(8), 'hex');
END;
$$ LANGUAGE plpgsql;

-- Function to get active keys by type
CREATE OR REPLACE FUNCTION get_active_keys(key_type_param key_type)
RETURNS TABLE (
    id UUID,
    key_id VARCHAR(255),
    name VARCHAR(255),
    encrypted_key BYTEA,
    algorithm encryption_algorithm,
    expires_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ek.id,
        ek.key_id,
        ek.name,
        ek.encrypted_key,
        ek.algorithm,
        ek.expires_at
    FROM encryption_keys ek
    WHERE ek.key_type = key_type_param
      AND ek.status = 'active'
      AND (ek.expires_at IS NULL OR ek.expires_at > CURRENT_TIMESTAMP)
    ORDER BY ek.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to mark key as compromised
CREATE OR REPLACE FUNCTION mark_key_compromised(key_id_param VARCHAR(255))
RETURNS BOOLEAN AS $$
DECLARE
    affected_count INTEGER;
BEGIN
    UPDATE encryption_keys
    SET
        status = 'compromised',
        compromise_detected_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE key_id = key_id_param;

    GET DIAGNOSTICS affected_count = ROW_COUNT;

    -- Log the compromise
    INSERT INTO key_access_audit (
        key_id,
        access_type,
        accessor_type,
        error_message
    )
    SELECT
        id,
        'compromise_detected',
        'system',
        'Key marked as compromised'
    FROM encryption_keys
    WHERE key_id = key_id_param;

    RETURN affected_count > 0;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup expired keys (move to destroyed status)
CREATE OR REPLACE FUNCTION cleanup_expired_keys()
RETURNS INTEGER AS $$
DECLARE
    expired_count INTEGER;
BEGIN
    -- Mark expired keys as destroyed
    UPDATE encryption_keys
    SET
        status = 'destroyed',
        destroyed_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP
    WHERE status = 'active'
      AND expires_at IS NOT NULL
      AND expires_at < CURRENT_TIMESTAMP - INTERVAL '30 days'; -- Keep for 30 days after expiry

    GET DIAGNOSTICS expired_count = ROW_COUNT;

    -- Log cleanup
    IF expired_count > 0 THEN
        INSERT INTO key_access_audit (
            key_id,
            access_type,
            accessor_type,
            error_message
        )
        SELECT
            id,
            'cleanup',
            'system',
            'Key destroyed after expiry grace period'
        FROM encryption_keys
        WHERE status = 'destroyed'
          AND destroyed_at = CURRENT_TIMESTAMP;
    END IF;

    RETURN expired_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get keys due for rotation
CREATE OR REPLACE FUNCTION get_keys_due_for_rotation()
RETURNS TABLE (
    id UUID,
    key_id VARCHAR(255),
    key_type key_type,
    name VARCHAR(255),
    next_rotation_at TIMESTAMP WITH TIME ZONE,
    days_until_rotation INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ek.id,
        ek.key_id,
        ek.key_type,
        ek.name,
        ek.next_rotation_at,
        EXTRACT(EPOCH FROM (ek.next_rotation_at - CURRENT_TIMESTAMP))::INTEGER / 86400
    FROM encryption_keys ek
    WHERE ek.status = 'active'
      AND ek.next_rotation_at IS NOT NULL
      AND ek.next_rotation_at <= CURRENT_TIMESTAMP
    ORDER BY ek.next_rotation_at ASC;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL ON encryption_keys TO agent_user;
GRANT ALL ON key_usage_log TO agent_user;
GRANT ALL ON key_rotation_history TO agent_user;
GRANT ALL ON key_access_audit TO agent_user;

GRANT USAGE ON TYPE key_type TO agent_user;
GRANT USAGE ON TYPE key_status TO agent_user;
GRANT USAGE ON TYPE encryption_algorithm TO agent_user;

-- Add comments for documentation
COMMENT ON TABLE encryption_keys IS 'Stores encryption keys with metadata and lifecycle management';
COMMENT ON TABLE key_usage_log IS 'Audit log of all encryption key usage operations';
COMMENT ON TABLE key_rotation_history IS 'History of key rotation events';
COMMENT ON TABLE key_access_audit IS 'Audit trail of key access operations';

COMMENT ON COLUMN encryption_keys.key_type IS 'Type of encryption key (master, data, field, record)';
COMMENT ON COLUMN encryption_keys.status IS 'Current status of the key (active, rotating, deprecated, etc.)';
COMMENT ON COLUMN encryption_keys.encrypted_key IS 'The actual encrypted key data';
COMMENT ON COLUMN encryption_keys.key_hash IS 'SHA256 hash for integrity verification';
COMMENT ON COLUMN encryption_keys.rotation_count IS 'Number of times this key has been rotated';

-- Insert default master key placeholder (will be populated by application)
-- This is just a placeholder record; the actual key will be set by the application
INSERT INTO encryption_keys (
    key_id,
    key_type,
    name,
    description,
    encrypted_key,
    key_hash,
    algorithm,
    status,
    security_level
) VALUES (
    'master_default',
    'master',
    'Default Master Key',
    'Placeholder for master encryption key - to be replaced by application',
    E'\\x00', -- Placeholder encrypted key
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', -- SHA256 of empty string
    'aes-256-gcm',
    'active',
    'critical'
) ON CONFLICT (key_id) DO NOTHING;