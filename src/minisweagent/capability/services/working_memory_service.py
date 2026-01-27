"""Working Memory Service (L4) - Session State Persistence."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """A discovered fact about the codebase."""

    id: str
    content: str
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EntityRef:
    """Reference to a code entity."""

    id: str
    type: str  # file, class, function, variable
    name: str
    location: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FocusState:
    """Current attention focus of the agent."""

    current_file: Optional[str] = None
    current_function: Optional[str] = None
    current_task: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemory:
    """Session working memory state."""

    session_id: str
    focus_state: Optional[FocusState] = None
    findings: List[Finding] = field(default_factory=list)
    entity_refs: List[EntityRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class WorkingMemoryService:
    """Service for managing session working memory (L4)."""

    def __init__(self, db_client):
        """Initialize working memory service.

        Args:
            db_client: Database client (asyncpg connection or pool)
        """
        self.db = db_client

    async def load_session(self, session_id: str) -> Optional[WorkingMemory]:
        """Load working memory for a session.

        Args:
            session_id: Session identifier

        Returns:
            WorkingMemory if found, None otherwise
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            row = await conn.fetchrow(
                "SELECT * FROM session_state WHERE session_id = $1",
                session_id
            )

            if not row:
                logger.info(f"No existing session found: {session_id}")
                return None

            # Deserialize from database
            focus_state = None
            if row['focus_state']:
                focus_state = FocusState(**row['focus_state'])

            findings = [Finding(**f) for f in (row['findings'] or [])]
            entity_refs = [EntityRef(**e) for e in (row['entity_refs'] or [])]

            memory = WorkingMemory(
                session_id=session_id,
                focus_state=focus_state,
                findings=findings,
                entity_refs=entity_refs,
                metadata=row['metadata'] or {},
                created_at=row['created_at'],
                updated_at=row['updated_at'],
            )

            logger.info(f"Loaded session {session_id}: {len(findings)} findings, {len(entity_refs)} entities")
            return memory

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def save_session(self, memory: WorkingMemory, ttl_seconds: int = 86400):
        """Save working memory for a session.

        Args:
            memory: Working memory to save
            ttl_seconds: Time to live in seconds (default 24 hours)
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            # Serialize to JSON
            focus_state_json = None
            if memory.focus_state:
                focus_state_json = {
                    'current_file': memory.focus_state.current_file,
                    'current_function': memory.focus_state.current_function,
                    'current_task': memory.focus_state.current_task,
                    'context': memory.focus_state.context,
                }

            findings_json = [
                {
                    'id': f.id,
                    'content': f.content,
                    'source': f.source,
                    'confidence': f.confidence,
                    'metadata': f.metadata,
                    'created_at': f.created_at.isoformat(),
                }
                for f in memory.findings
            ]

            entity_refs_json = [
                {
                    'id': e.id,
                    'type': e.type,
                    'name': e.name,
                    'location': e.location,
                    'metadata': e.metadata,
                }
                for e in memory.entity_refs
            ]

            # Upsert session state
            await conn.execute("""
                INSERT INTO session_state (
                    session_id, focus_state, findings, entity_refs, metadata, ttl_seconds, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (session_id)
                DO UPDATE SET
                    focus_state = EXCLUDED.focus_state,
                    findings = EXCLUDED.findings,
                    entity_refs = EXCLUDED.entity_refs,
                    metadata = EXCLUDED.metadata,
                    ttl_seconds = EXCLUDED.ttl_seconds,
                    updated_at = NOW()
            """, memory.session_id, focus_state_json, findings_json, entity_refs_json,
                memory.metadata, ttl_seconds)

            logger.info(f"Saved session {memory.session_id}: {len(memory.findings)} findings")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List active sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            rows = await conn.fetch("""
                SELECT
                    session_id,
                    created_at,
                    updated_at,
                    jsonb_array_length(COALESCE(findings, '[]'::jsonb)) as finding_count,
                    jsonb_array_length(COALESCE(entity_refs, '[]'::jsonb)) as entity_count
                FROM session_state
                ORDER BY updated_at DESC
                LIMIT $1
            """, limit)

            return [
                {
                    'session_id': row['session_id'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'finding_count': row['finding_count'],
                    'entity_count': row['entity_count'],
                }
                for row in rows
            ]

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def delete_session(self, session_id: str):
        """Delete a session.

        Args:
            session_id: Session to delete
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute(
                "DELETE FROM session_state WHERE session_id = $1",
                session_id
            )
            logger.info(f"Deleted session: {session_id}")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def cleanup_expired(self):
        """Clean up expired sessions."""
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            result = await conn.execute("""
                DELETE FROM session_state
                WHERE updated_at < NOW() - INTERVAL '1 second' * ttl_seconds
            """)

            count = int(result.split()[-1]) if result else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)
