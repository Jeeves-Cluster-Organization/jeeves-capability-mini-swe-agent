"""Checkpoint Service - Pipeline State Persistence for Resume."""

import logging
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Pipeline checkpoint."""

    checkpoint_id: str
    session_id: str
    agent_name: str
    next_agent: Optional[str]
    envelope_state: Dict[str, Any]
    created_at: datetime


class CheckpointService:
    """Service for checkpointing pipeline state."""

    def __init__(self, db_client):
        """Initialize checkpoint service.

        Args:
            db_client: Database client (asyncpg connection or pool)
        """
        self.db = db_client

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        session_id: str,
        agent_name: str,
        envelope_state: Dict[str, Any],
        next_agent: Optional[str] = None
    ):
        """Save a checkpoint.

        Args:
            checkpoint_id: Unique checkpoint identifier
            session_id: Session ID
            agent_name: Current agent name
            envelope_state: Serialized envelope state
            next_agent: Next agent to execute (optional)
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute("""
                INSERT INTO checkpoints (checkpoint_id, session_id, agent_name, next_agent, envelope_state)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (checkpoint_id) DO UPDATE SET
                    agent_name = EXCLUDED.agent_name,
                    next_agent = EXCLUDED.next_agent,
                    envelope_state = EXCLUDED.envelope_state,
                    created_at = NOW()
            """, checkpoint_id, session_id, agent_name, next_agent, envelope_state)

            logger.info(f"Saved checkpoint: {checkpoint_id} at agent {agent_name}")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Checkpoint if found, None otherwise
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            row = await conn.fetchrow(
                "SELECT * FROM checkpoints WHERE checkpoint_id = $1",
                checkpoint_id
            )

            if not row:
                return None

            return Checkpoint(
                checkpoint_id=row['checkpoint_id'],
                session_id=row['session_id'],
                agent_name=row['agent_name'],
                next_agent=row['next_agent'],
                envelope_state=row['envelope_state'],
                created_at=row['created_at'],
            )

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def load_latest_for_session(self, session_id: str) -> Optional[Checkpoint]:
        """Load latest checkpoint for a session.

        Args:
            session_id: Session ID

        Returns:
            Latest checkpoint if found, None otherwise
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            row = await conn.fetchrow("""
                SELECT * FROM checkpoints
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT 1
            """, session_id)

            if not row:
                return None

            return Checkpoint(
                checkpoint_id=row['checkpoint_id'],
                session_id=row['session_id'],
                agent_name=row['agent_name'],
                next_agent=row['next_agent'],
                envelope_state=row['envelope_state'],
                created_at=row['created_at'],
            )

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = $1",
                checkpoint_id
            )

            logger.info(f"Deleted checkpoint: {checkpoint_id}")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def delete_session_checkpoints(self, session_id: str):
        """Delete all checkpoints for a session.

        Args:
            session_id: Session ID
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            result = await conn.execute(
                "DELETE FROM checkpoints WHERE session_id = $1",
                session_id
            )

            count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {count} checkpoints for session {session_id}")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)
