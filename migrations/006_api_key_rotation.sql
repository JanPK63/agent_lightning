-- API Key Rotation System
-- Enables automatic rotation, expiration, and history tracking for API keys

-- Create enum for rotation reasons
CREATE TYPE rotation_reason AS ENUM (
    'scheduled',        -- Regular scheduled rotation
    'manual',          -- Manual rotation by admin
    'compromised',     -- Key suspected to be compromised
    'expired',         -- Key reached expiration date
    'policy_change'    -- Rotation policy changed
);

-- Create enum for notification status
CREATE TYPE notification_status AS ENUM (
    'pending',         -- Notification queued
    'sent',           -- Notification sent successfully
    'failed',         -- Notification failed to send
    'acknowledged'    -- User acknowledged notification
);

-- Enhance existing api_keys table with rotation fields
ALTER TABLE api_keys
ADD COLUMN IF NOT EXISTS rotation_policy_id UUID,
ADD COLUMN IF NOT EXISTS last_rotated_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS next_rotation_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS rotation_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS is_rotation_enabled BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS rotation_locked BOOLEAN DEFAULT FALSE;

-- Create rotation policies table
CREATE TABLE IF NOT EXISTS api_key_rotation_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,

    -- Rotation settings
    auto_rotate_days INTEGER NOT NULL DEFAULT 90,  -- Rotate every 90 days
    notify_before_days INTEGER DEFAULT 7,          -- Notify 7 days before rotation
    grace_period_days INTEGER DEFAULT 30,          -- Old key valid for 30 days after rotation

    -- Security settings
    require_manual_acknowledgment BOOLEAN DEFAULT FALSE,
    max_rotation_count INTEGER DEFAULT 100,       -- Maximum rotations before requiring review

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

-- Create rotation history table
CREATE TABLE IF NOT EXISTS api_key_rotation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,

    -- Key information
    old_key_hash VARCHAR(128),  -- Hash of the old key
    new_key_hash VARCHAR(128),  -- Hash of the new key
    old_expires_at TIMESTAMP WITH TIME ZONE,
    new_expires_at TIMESTAMP WITH TIME ZONE,

    -- Rotation details
    rotated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rotated_by VARCHAR(255),    -- User/system that performed rotation
    rotation_reason rotation_reason NOT NULL,
    rotation_policy_id UUID REFERENCES api_key_rotation_policies(id),

    -- Notification tracking
    notification_sent_at TIMESTAMP WITH TIME ZONE,
    notification_status notification_status DEFAULT 'pending',
    user_acknowledged_at TIMESTAMP WITH TIME ZONE,

    -- Additional metadata
    notes TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_api_keys_rotation_policy ON api_keys(rotation_policy_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_next_rotation ON api_keys(next_rotation_at) WHERE next_rotation_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_api_keys_rotation_enabled ON api_keys(is_rotation_enabled) WHERE is_rotation_enabled = TRUE;

CREATE INDEX IF NOT EXISTS idx_rotation_policies_active ON api_key_rotation_policies(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_rotation_policies_default ON api_key_rotation_policies(is_default) WHERE is_default = TRUE;

CREATE INDEX IF NOT EXISTS idx_rotation_history_api_key ON api_key_rotation_history(api_key_id);
CREATE INDEX IF NOT EXISTS idx_rotation_history_rotated_at ON api_key_rotation_history(rotated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rotation_history_reason ON api_key_rotation_history(rotation_reason);

-- Create default rotation policies
INSERT INTO api_key_rotation_policies (
    name, description, auto_rotate_days, notify_before_days, grace_period_days,
    require_manual_acknowledgment, is_default
) VALUES
(
    'standard',
    'Standard rotation policy: 90 days with 7 day notification',
    90, 7, 30, FALSE, TRUE
),
(
    'high_security',
    'High security policy: 30 days with manual acknowledgment required',
    30, 3, 7, TRUE, FALSE
),
(
    'low_security',
    'Low security policy: 180 days with minimal notification',
    180, 14, 60, FALSE, FALSE
);

-- Function to get next rotation date
CREATE OR REPLACE FUNCTION calculate_next_rotation(
    last_rotated_at TIMESTAMP WITH TIME ZONE,
    policy_id UUID
)
RETURNS TIMESTAMP WITH TIME ZONE AS $$
DECLARE
    policy_record RECORD;
    next_rotation TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Get policy details
    SELECT auto_rotate_days INTO policy_record
    FROM api_key_rotation_policies
    WHERE id = policy_id AND is_active = TRUE;

    IF NOT FOUND THEN
        -- Use default policy
        SELECT auto_rotate_days INTO policy_record
        FROM api_key_rotation_policies
        WHERE is_default = TRUE AND is_active = TRUE
        LIMIT 1;

        IF NOT FOUND THEN
            policy_record.auto_rotate_days := 90; -- Fallback
        END IF;
    END IF;

    -- Calculate next rotation
    IF last_rotated_at IS NULL THEN
        next_rotation := CURRENT_TIMESTAMP + INTERVAL '1 day' * policy_record.auto_rotate_days;
    ELSE
        next_rotation := last_rotated_at + INTERVAL '1 day' * policy_record.auto_rotate_days;
    END IF;

    RETURN next_rotation;
END;
$$ LANGUAGE plpgsql;

-- Function to rotate API key
CREATE OR REPLACE FUNCTION rotate_api_key(
    key_id UUID,
    new_key_hash VARCHAR(128),
    new_expires_at TIMESTAMP WITH TIME ZONE,
    rotated_by_user VARCHAR(255) DEFAULT 'system',
    rotation_reason_param rotation_reason DEFAULT 'scheduled',
    notes_param TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    old_key_record RECORD;
    rotation_history_id UUID;
    policy_record RECORD;
BEGIN
    -- Get current key information
    SELECT * INTO old_key_record
    FROM api_keys
    WHERE id = key_id AND is_active = TRUE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'API key not found or not active';
    END IF;

    -- Get rotation policy
    IF old_key_record.rotation_policy_id IS NOT NULL THEN
        SELECT * INTO policy_record
        FROM api_key_rotation_policies
        WHERE id = old_key_record.rotation_policy_id AND is_active = TRUE;
    ELSE
        SELECT * INTO policy_record
        FROM api_key_rotation_policies
        WHERE is_default = TRUE AND is_active = TRUE
        LIMIT 1;
    END IF;

    -- Update the API key
    UPDATE api_keys
    SET
        key_hash = new_key_hash,
        expires_at = new_expires_at,
        last_rotated_at = CURRENT_TIMESTAMP,
        next_rotation_at = calculate_next_rotation(CURRENT_TIMESTAMP, COALESCE(old_key_record.rotation_policy_id, policy_record.id)),
        rotation_count = rotation_count + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = key_id;

    -- Record rotation history
    INSERT INTO api_key_rotation_history (
        api_key_id,
        old_key_hash,
        new_key_hash,
        old_expires_at,
        new_expires_at,
        rotated_by,
        rotation_reason,
        rotation_policy_id,
        notes
    )
    VALUES (
        key_id,
        old_key_record.key_hash,
        new_key_hash,
        old_key_record.expires_at,
        new_expires_at,
        rotated_by_user,
        rotation_reason_param,
        COALESCE(old_key_record.rotation_policy_id, policy_record.id),
        notes_param
    )
    RETURNING id INTO rotation_history_id;

    -- Log the rotation
    RAISE NOTICE 'API key % rotated by % for reason %', key_id, rotated_by_user, rotation_reason_param;

    RETURN rotation_history_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get keys due for rotation
CREATE OR REPLACE FUNCTION get_keys_due_for_rotation()
RETURNS TABLE (
    key_id UUID,
    key_name VARCHAR(100),
    user_id UUID,
    next_rotation_at TIMESTAMP WITH TIME ZONE,
    days_until_rotation INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ak.id,
        ak.name,
        ak.user_id::UUID,
        ak.next_rotation_at,
        EXTRACT(EPOCH FROM (ak.next_rotation_at - CURRENT_TIMESTAMP))::INTEGER / 86400
    FROM api_keys ak
    WHERE ak.is_active = TRUE
      AND ak.is_rotation_enabled = TRUE
      AND ak.rotation_locked = FALSE
      AND ak.next_rotation_at IS NOT NULL
      AND ak.next_rotation_at <= CURRENT_TIMESTAMP
    ORDER BY ak.next_rotation_at ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup expired rotation history (older than 1 year)
CREATE OR REPLACE FUNCTION cleanup_old_rotation_history()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM api_key_rotation_history
    WHERE rotated_at < CURRENT_TIMESTAMP - INTERVAL '1 year';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL ON api_key_rotation_policies TO agent_user;
GRANT ALL ON api_key_rotation_history TO agent_user;
GRANT USAGE ON TYPE rotation_reason TO agent_user;
GRANT USAGE ON TYPE notification_status TO agent_user;

-- Add comments for documentation
COMMENT ON TABLE api_key_rotation_policies IS 'Defines rotation policies for API keys including frequency and notification settings';
COMMENT ON TABLE api_key_rotation_history IS 'Tracks all API key rotations with old/new key information and metadata';
COMMENT ON COLUMN api_keys.rotation_policy_id IS 'Reference to the rotation policy for this key';
COMMENT ON COLUMN api_keys.last_rotated_at IS 'Timestamp of the last rotation';
COMMENT ON COLUMN api_keys.next_rotation_at IS 'Timestamp when this key should be rotated next';
COMMENT ON COLUMN api_keys.rotation_count IS 'Number of times this key has been rotated';
COMMENT ON COLUMN api_keys.is_rotation_enabled IS 'Whether automatic rotation is enabled for this key';
COMMENT ON COLUMN api_keys.rotation_locked IS 'Whether this key is locked from rotation (e.g., during critical operations)';