-- Migration 004: Graph Storage (L5)
-- Code entity relationship graph

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(50) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_nodes_type ON graph_nodes(node_type);

CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    target_id VARCHAR(255) NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX idx_edges_source ON graph_edges(source_id);
CREATE INDEX idx_edges_target ON graph_edges(target_id);
CREATE INDEX idx_edges_type ON graph_edges(edge_type);

CREATE MATERIALIZED VIEW IF NOT EXISTS dependency_graph AS
SELECT
    source_id,
    target_id,
    edge_type,
    n1.metadata->>'file' as source_file,
    n2.metadata->>'file' as target_file
FROM graph_edges e
JOIN graph_nodes n1 ON e.source_id = n1.node_id
JOIN graph_nodes n2 ON e.target_id = n2.node_id
WHERE edge_type IN ('imports', 'calls', 'inherits');

CREATE INDEX idx_dep_graph_source ON dependency_graph(source_id);
CREATE INDEX idx_dep_graph_target ON dependency_graph(target_id);
