-- Migration 001: Working Memory (L4)
-- Session state persistence

CREATE TABLE IF NOT EXISTS session_state (
    session_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    focus_state JSONB,
    findings JSONB,
    entity_refs JSONB,
    metadata JSONB,
    ttl_seconds INTEGER DEFAULT 86400
);

CREATE INDEX idx_session_updated ON session_state(updated_at);
CREATE INDEX idx_session_findings ON session_state USING gin(findings);

-- Automatic cleanup function
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM session_state
    WHERE updated_at < NOW() - INTERVAL '1 second' * ttl_seconds;
END;
$$ LANGUAGE plpgsql;
