"""Code Indexer Service (L3) - Semantic Code Search."""

import logging
import hashlib
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """Semantic code chunk with embedding."""

    chunk_id: str
    source_file: str
    content: str
    score: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CodeIndexerService:
    """Service for semantic code search using embeddings (L3)."""

    def __init__(self, db_client, embedding_model="all-MiniLM-L6-v2"):
        """Initialize code indexer service.

        Args:
            db_client: Database client with pgvector support
            embedding_model: Sentence transformer model name
        """
        self.db = db_client
        self.embedding_model = embedding_model
        self._model = None

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
                logger.info(f"Loaded embedding model: {self.embedding_model}")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise

        return self._model

    def _generate_chunk_id(self, source_file: str, content: str) -> str:
        """Generate unique chunk ID.

        Args:
            source_file: Source file path
            content: Chunk content

        Returns:
            Unique chunk ID
        """
        data = f"{source_file}:{content}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _chunk_content(self, content: str, chunk_size: int = 512) -> List[str]:
        """Split content into chunks.

        Args:
            content: Content to chunk
            chunk_size: Approximate tokens per chunk

        Returns:
            List of content chunks
        """
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line.split())
            if current_size + line_size > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    async def index_file(self, file_path: str, content: str, chunk_size: int = 512):
        """Index a file for semantic search.

        Args:
            file_path: Path to file
            content: File content
            chunk_size: Approximate tokens per chunk
        """
        model = self._get_embedding_model()
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            # Delete existing chunks for this file
            await conn.execute(
                "DELETE FROM semantic_chunks WHERE source_file = $1",
                file_path
            )

            # Chunk content
            chunks = self._chunk_content(content, chunk_size)
            logger.info(f"Indexing {file_path}: {len(chunks)} chunks")

            # Generate embeddings
            embeddings = model.encode(chunks)

            # Insert chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._generate_chunk_id(file_path, chunk)

                # Convert embedding to list
                embedding_list = embedding.tolist()

                # Determine line numbers
                line_start = content[:content.find(chunk)].count('\n') + 1 if chunk in content else 0
                line_end = line_start + chunk.count('\n')

                await conn.execute("""
                    INSERT INTO semantic_chunks (chunk_id, source_file, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """, chunk_id, file_path, chunk, embedding_list, {
                    'chunk_index': i,
                    'line_start': line_start,
                    'line_end': line_end,
                })

            logger.info(f"Indexed {file_path}: {len(chunks)} chunks")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> List[SemanticChunk]:
        """Search codebase semantically.

        Args:
            query: Natural language query
            limit: Maximum results
            min_score: Minimum similarity score (0-1)

        Returns:
            List of matching code chunks
        """
        model = self._get_embedding_model()
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            # Generate query embedding
            query_embedding = model.encode([query])[0].tolist()

            # Search using cosine similarity
            rows = await conn.fetch("""
                SELECT
                    chunk_id,
                    source_file,
                    content,
                    metadata,
                    1 - (embedding <=> $1) as similarity
                FROM semantic_chunks
                WHERE 1 - (embedding <=> $1) >= $2
                ORDER BY similarity DESC
                LIMIT $3
            """, query_embedding, min_score, limit)

            results = []
            for row in rows:
                results.append(SemanticChunk(
                    chunk_id=row['chunk_id'],
                    source_file=row['source_file'],
                    content=row['content'],
                    score=row['similarity'],
                    metadata=row['metadata'] or {},
                ))

            logger.info(f"Semantic search for '{query}': {len(results)} results")
            return results

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def delete_file(self, file_path: str):
        """Delete indexed file.

        Args:
            file_path: Path to file
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute(
                "DELETE FROM semantic_chunks WHERE source_file = $1",
                file_path
            )
            logger.info(f"Deleted index for: {file_path}")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def get_indexed_files(self) -> List[str]:
        """Get list of indexed files.

        Returns:
            List of file paths
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            rows = await conn.fetch("""
                SELECT DISTINCT source_file
                FROM semantic_chunks
                ORDER BY source_file
            """)

            return [row['source_file'] for row in rows]

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)
