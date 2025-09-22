-- Migration to add event sourcing tables
-- This migration creates the events and event_snapshots tables for event sourcing functionality
-- SQLite compatible version

-- Create events table for storing all system events
CREATE TABLE events (
    id VARCHAR(36) PRIMARY KEY,
    event_id VARCHAR(36) UNIQUE NOT NULL,
    aggregate_id VARCHAR(36) NOT NULL,
    aggregate_type VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data TEXT NOT NULL,  -- JSON stored as TEXT in SQLite
    timestamp DATETIME NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    correlation_id VARCHAR(36),
    causation_id VARCHAR(36),
    user_id VARCHAR(36),
    service_name VARCHAR(50),
    metadata TEXT DEFAULT '{}'  -- JSON stored as TEXT in SQLite
);

-- Create indexes for events table
CREATE INDEX idx_events_event_id ON events(event_id);
CREATE INDEX idx_events_aggregate_id ON events(aggregate_id);
CREATE INDEX idx_events_aggregate_type ON events(aggregate_type);
CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_correlation_id ON events(correlation_id);
CREATE INDEX idx_events_causation_id ON events(causation_id);
CREATE INDEX idx_events_user_id ON events(user_id);
CREATE INDEX idx_events_service_name ON events(service_name);

-- Create event_snapshots table for performance optimization
CREATE TABLE event_snapshots (
    id VARCHAR(36) PRIMARY KEY,
    aggregate_id VARCHAR(36) NOT NULL,
    aggregate_type VARCHAR(50) NOT NULL,
    snapshot_data TEXT NOT NULL,  -- JSON stored as TEXT in SQLite
    version INTEGER NOT NULL,
    last_event_id VARCHAR(36) NOT NULL,
    created_at DATETIME NOT NULL,
    expires_at DATETIME
);

-- Create indexes for event_snapshots table
CREATE INDEX idx_event_snapshots_aggregate_id ON event_snapshots(aggregate_id);
CREATE INDEX idx_event_snapshots_aggregate_type ON event_snapshots(aggregate_type);
CREATE INDEX idx_event_snapshots_version ON event_snapshots(version);
CREATE INDEX idx_event_snapshots_created_at ON event_snapshots(created_at);
CREATE INDEX idx_event_snapshots_expires_at ON event_snapshots(expires_at);