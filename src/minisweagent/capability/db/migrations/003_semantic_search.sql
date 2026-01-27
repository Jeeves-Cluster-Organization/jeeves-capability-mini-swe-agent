-- Migration 003: Semantic Search (L3)
-- Code embeddings for natural language search

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    source_file VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chunks_embedding ON semantic_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_chunks_file ON semantic_chunks(source_file);
CREATE INDEX idx_chunks_content ON semantic_chunks USING gin(to_tsvector('english', content));
