-- Migration to add encrypted columns to existing tables
-- This migration adds the encrypted columns required for data encryption
-- SQLite compatible version

-- Add encrypted columns to users table
ALTER TABLE users ADD COLUMN email_encrypted VARCHAR(100);
ALTER TABLE users ADD COLUMN password_hash_encrypted VARCHAR(255);

-- Add encrypted columns to conversations table
ALTER TABLE conversations ADD COLUMN user_query_encrypted TEXT;
ALTER TABLE conversations ADD COLUMN agent_response_encrypted TEXT;

-- Add encrypted columns to agents table
ALTER TABLE agents ADD COLUMN config_encrypted TEXT;  -- JSON stored as TEXT in SQLite
ALTER TABLE agents ADD COLUMN capabilities_encrypted TEXT;  -- JSON stored as TEXT in SQLite

-- Add encrypted columns to workflows table
ALTER TABLE workflows ADD COLUMN steps_encrypted TEXT;  -- JSON stored as TEXT in SQLite
ALTER TABLE workflows ADD COLUMN context_encrypted TEXT;  -- JSON stored as TEXT in SQLite