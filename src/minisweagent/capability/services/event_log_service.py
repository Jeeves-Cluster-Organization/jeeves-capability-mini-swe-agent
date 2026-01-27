"""Event Log Service (L2) - Persistent Audit Trail."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EventLogEntry:
    """Event log entry."""

    event_id: int
    session_id: Optional[str]
    timestamp: datetime
    event_category: str
    event_type: str
    agent_name: Optional[str]
    payload: Dict[str, Any]
    metadata: Dict[str, Any]


class EventLogService:
    """Service for persistent event logging (L2)."""

    def __init__(self, db_client):
        """Initialize event log service.

        Args:
            db_client: Database client (asyncpg connection or pool)
        """
        self.db = db_client

    async def log_event(
        self,
        event_category: str,
        event_type: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Log an event.

        Args:
            event_category: Event category (agent_started, tool_executed, etc.)
            event_type: Specific event type
            payload: Event payload
            session_id: Session ID (optional)
            agent_name: Agent name (optional)
            metadata: Additional metadata (optional)

        Returns:
            Event ID
        """
        if metadata is None:
            metadata = {}

        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            row = await conn.fetchrow("""
                INSERT INTO event_log (session_id, event_category, event_type, agent_name, payload, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING event_id
            """, session_id, event_category, event_type, agent_name, payload, metadata)

            return row['event_id']

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def get_events(
        self,
        session_id: Optional[str] = None,
        event_category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[EventLogEntry]:
        """Query events.

        Args:
            session_id: Filter by session ID (optional)
            event_category: Filter by category (optional)
            limit: Maximum results
            offset: Result offset

        Returns:
            List of event log entries
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            query = "SELECT * FROM event_log WHERE 1=1"
            params = []
            param_count = 0

            if session_id:
                param_count += 1
                query += f" AND session_id = ${param_count}"
                params.append(session_id)

            if event_category:
                param_count += 1
                query += f" AND event_category = ${param_count}"
                params.append(event_category)

            query += f" ORDER BY timestamp DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
            params.extend([limit, offset])

            rows = await conn.fetch(query, *params)

            return [
                EventLogEntry(
                    event_id=row['event_id'],
                    session_id=row['session_id'],
                    timestamp=row['timestamp'],
                    event_category=row['event_category'],
                    event_type=row['event_type'],
                    agent_name=row['agent_name'],
                    payload=row['payload'] or {},
                    metadata=row['metadata'] or {},
                )
                for row in rows
            ]

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def get_session_events(self, session_id: str) -> List[EventLogEntry]:
        """Get all events for a session.

        Args:
            session_id: Session ID

        Returns:
            List of event log entries
        """
        return await self.get_events(session_id=session_id, limit=10000)

    async def delete_old_events(self, days: int = 30):
        """Delete events older than specified days.

        Args:
            days: Number of days to keep
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            result = await conn.execute("""
                DELETE FROM event_log
                WHERE timestamp < NOW() - INTERVAL '%s days'
            """, days)

            count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {count} events older than {days} days")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)
