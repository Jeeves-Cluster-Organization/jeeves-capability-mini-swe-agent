-- Migration 002: Tool Health (L7)
-- Tool health monitoring and metrics

CREATE TABLE IF NOT EXISTS tool_health (
    tool_name VARCHAR(255) PRIMARY KEY,
    invocation_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    total_latency_ms BIGINT DEFAULT 0,
    last_invocation TIMESTAMP,
    status VARCHAR(50) DEFAULT 'healthy',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tool_invocations (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    latency_ms INTEGER NOT NULL,
    error_message TEXT,
    invoked_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tool_invocations_tool ON tool_invocations(tool_name);
CREATE INDEX idx_tool_invocations_time ON tool_invocations(invoked_at);

-- Automatic status calculation trigger
CREATE OR REPLACE FUNCTION update_tool_status()
RETURNS TRIGGER AS $$
DECLARE
    error_rate FLOAT;
BEGIN
    IF NEW.invocation_count > 0 THEN
        error_rate := NEW.failure_count::FLOAT / NEW.invocation_count;

        IF error_rate >= 0.5 THEN
            NEW.status := 'quarantined';
        ELSIF error_rate >= 0.1 THEN
            NEW.status := 'degraded';
        ELSE
            NEW.status := 'healthy';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tool_health_status_trigger
    BEFORE UPDATE ON tool_health
    FOR EACH ROW
    EXECUTE FUNCTION update_tool_status();
