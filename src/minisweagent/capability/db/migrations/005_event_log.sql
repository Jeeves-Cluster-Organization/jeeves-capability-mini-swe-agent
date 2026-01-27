-- Migration 005: Event Log (L2) and Checkpointing
-- Persistent event log and checkpoint storage

CREATE TABLE IF NOT EXISTS event_log (
    event_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW(),
    event_category VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    agent_name VARCHAR(100),
    payload JSONB,
    metadata JSONB
);

CREATE INDEX idx_event_log_session ON event_log(session_id);
CREATE INDEX idx_event_log_timestamp ON event_log(timestamp);
CREATE INDEX idx_event_log_category ON event_log(event_category);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    next_agent VARCHAR(100),
    envelope_state JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_checkpoint_session ON checkpoints(session_id);
CREATE INDEX idx_checkpoint_created ON checkpoints(created_at);
