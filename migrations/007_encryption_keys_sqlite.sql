-- Encryption Keys Management System - SQLite Version
-- Database schema for managing encryption keys and key rotation

-- Main encryption keys table
CREATE TABLE IF NOT EXISTS encryption_keys (
    id TEXT PRIMARY KEY,  -- UUID stored as TEXT in SQLite
    key_type TEXT NOT NULL,  -- 'master', 'data', 'field', 'record'
    key_id TEXT NOT NULL UNIQUE, -- Unique identifier for the key
    name TEXT, -- Human-readable name
    description TEXT,

    -- Key data (encrypted)
    encrypted_key BLOB NOT NULL, -- The actual encrypted key (BLOB in SQLite)
    key_hash TEXT NOT NULL, -- SHA256 hash for integrity verification
    algorithm TEXT NOT NULL DEFAULT 'aes-256-gcm',

    -- Key hierarchy
    parent_key_id TEXT REFERENCES encryption_keys(id), -- Parent key in hierarchy
    derived_from TEXT, -- How this key was derived

    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'rotating', 'deprecated', 'compromised', 'destroyed'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,  -- ISO timestamp as TEXT
    activated_at TEXT,
    expires_at TEXT,
    destroyed_at TEXT,

    -- Rotation
    rotation_count INTEGER DEFAULT 0,
    last_rotated_at TEXT,
    next_rotation_at TEXT,

    -- Security
    compromise_detected_at TEXT,
    security_level TEXT DEFAULT 'standard', -- 'standard', 'high', 'critical'

    -- Metadata
    tags TEXT DEFAULT '[]',  -- JSON array as TEXT
    metadata TEXT DEFAULT '{}',  -- JSON as TEXT
    created_by TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Key usage tracking table
CREATE TABLE IF NOT EXISTS key_usage_log (
    id TEXT PRIMARY KEY,  -- UUID as TEXT
    key_id TEXT NOT NULL REFERENCES encryption_keys(id) ON DELETE CASCADE,
    operation TEXT NOT NULL, -- 'encrypt', 'decrypt', 'derive', 'rotate'
    field_name TEXT, -- Which field was encrypted/decrypted
    table_name TEXT, -- Which table
    record_id TEXT, -- Which record (if applicable)
    user_id TEXT, -- Who performed the operation
    ip_address TEXT, -- IP address as TEXT (SQLite doesn't have INET)
    user_agent TEXT, -- User agent string
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    success INTEGER NOT NULL DEFAULT 1,  -- BOOLEAN as INTEGER in SQLite
    error_message TEXT,
    performance_ms INTEGER, -- Operation time in milliseconds
    metadata TEXT DEFAULT '{}'  -- JSON as TEXT
);

-- Key rotation history table
CREATE TABLE IF NOT EXISTS key_rotation_history (
    id TEXT PRIMARY KEY,  -- UUID as TEXT
    key_id TEXT NOT NULL REFERENCES encryption_keys(id) ON DELETE CASCADE,
    old_key_hash TEXT,
    new_key_hash TEXT,
    rotation_reason TEXT, -- 'scheduled', 'manual', 'compromised', 'emergency'
    rotated_by TEXT,
    rotated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    rotation_time_ms INTEGER,
    success INTEGER NOT NULL DEFAULT 1,  -- BOOLEAN as INTEGER
    error_message TEXT,
    old_expires_at TEXT,
    new_expires_at TEXT,
    metadata TEXT DEFAULT '{}'  -- JSON as TEXT
);

-- Key access audit table
CREATE TABLE IF NOT EXISTS key_access_audit (
    id TEXT PRIMARY KEY,  -- UUID as TEXT
    key_id TEXT NOT NULL REFERENCES encryption_keys(id) ON DELETE CASCADE,
    access_type TEXT NOT NULL, -- 'retrieve', 'store', 'delete', 'rotate'
    accessor_id TEXT, -- Who accessed the key
    accessor_type TEXT, -- 'service', 'user', 'system'
    ip_address TEXT,
    user_agent TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    success INTEGER NOT NULL DEFAULT 1,  -- BOOLEAN as INTEGER
    error_message TEXT,
    metadata TEXT DEFAULT '{}'  -- JSON as TEXT
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

-- Insert default master key placeholder (will be populated by application)
-- This is just a placeholder record; the actual key will be set by the application
INSERT OR IGNORE INTO encryption_keys (
    id,
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
    '550e8400-e29b-41d4-a716-446655440000',  -- Fixed UUID for SQLite
    'master_default',
    'master',
    'Default Master Key',
    'Placeholder for master encryption key - to be replaced by application',
    X'00', -- Placeholder encrypted key as BLOB
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', -- SHA256 of empty string
    'aes-256-gcm',
    'active',
    'critical'
);