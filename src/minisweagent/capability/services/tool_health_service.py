"""Tool Health Service (L7) - Tool Monitoring and Governance."""

import logging
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ToolMetrics:
    """Tool health metrics."""

    tool_name: str
    invocation_count: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    last_invocation: Optional[datetime]
    status: str  # healthy, degraded, quarantined
    success_rate: float
    error_rate: float


class ToolHealthService:
    """Service for monitoring tool health and performance (L7)."""

    def __init__(self, db_client):
        """Initialize tool health service.

        Args:
            db_client: Database client (asyncpg connection or pool)
        """
        self.db = db_client

    async def record_invocation(
        self,
        tool_name: str,
        success: bool,
        latency_ms: int,
        error_message: Optional[str] = None
    ):
        """Record a tool invocation.

        Args:
            tool_name: Tool identifier
            success: Whether invocation succeeded
            latency_ms: Execution time in milliseconds
            error_message: Error message if failed
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            # Record invocation history
            await conn.execute("""
                INSERT INTO tool_invocations (tool_name, success, latency_ms, error_message)
                VALUES ($1, $2, $3, $4)
            """, tool_name, success, latency_ms, error_message)

            # Update aggregate metrics
            await conn.execute("""
                INSERT INTO tool_health (
                    tool_name,
                    invocation_count,
                    success_count,
                    failure_count,
                    total_latency_ms,
                    last_invocation
                )
                VALUES ($1, 1, $2, $3, $4, NOW())
                ON CONFLICT (tool_name)
                DO UPDATE SET
                    invocation_count = tool_health.invocation_count + 1,
                    success_count = tool_health.success_count + $2,
                    failure_count = tool_health.failure_count + $3,
                    total_latency_ms = tool_health.total_latency_ms + $4,
                    last_invocation = NOW(),
                    updated_at = NOW()
            """, tool_name, 1 if success else 0, 0 if success else 1, latency_ms)

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def get_tool_status(self, tool_name: str) -> str:
        """Get tool health status.

        Args:
            tool_name: Tool identifier

        Returns:
            Status: 'healthy', 'degraded', or 'quarantined'
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            row = await conn.fetchrow(
                "SELECT status FROM tool_health WHERE tool_name = $1",
                tool_name
            )

            if not row:
                return "healthy"  # No history = healthy

            return row['status']

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def get_metrics(self, tool_name: str) -> Optional[ToolMetrics]:
        """Get tool metrics.

        Args:
            tool_name: Tool identifier

        Returns:
            ToolMetrics if found, None otherwise
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            row = await conn.fetchrow(
                "SELECT * FROM tool_health WHERE tool_name = $1",
                tool_name
            )

            if not row:
                return None

            avg_latency = (
                row['total_latency_ms'] / row['invocation_count']
                if row['invocation_count'] > 0
                else 0
            )

            success_rate = (
                row['success_count'] / row['invocation_count']
                if row['invocation_count'] > 0
                else 1.0
            )

            error_rate = (
                row['failure_count'] / row['invocation_count']
                if row['invocation_count'] > 0
                else 0.0
            )

            return ToolMetrics(
                tool_name=row['tool_name'],
                invocation_count=row['invocation_count'],
                success_count=row['success_count'],
                failure_count=row['failure_count'],
                avg_latency_ms=avg_latency,
                last_invocation=row['last_invocation'],
                status=row['status'],
                success_rate=success_rate,
                error_rate=error_rate,
            )

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def get_all_metrics(self) -> List[ToolMetrics]:
        """Get metrics for all tools.

        Returns:
            List of tool metrics
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            rows = await conn.fetch("""
                SELECT * FROM tool_health
                ORDER BY tool_name
            """)

            metrics = []
            for row in rows:
                avg_latency = (
                    row['total_latency_ms'] / row['invocation_count']
                    if row['invocation_count'] > 0
                    else 0
                )

                success_rate = (
                    row['success_count'] / row['invocation_count']
                    if row['invocation_count'] > 0
                    else 1.0
                )

                error_rate = (
                    row['failure_count'] / row['invocation_count']
                    if row['invocation_count'] > 0
                    else 0.0
                )

                metrics.append(ToolMetrics(
                    tool_name=row['tool_name'],
                    invocation_count=row['invocation_count'],
                    success_count=row['success_count'],
                    failure_count=row['failure_count'],
                    avg_latency_ms=avg_latency,
                    last_invocation=row['last_invocation'],
                    status=row['status'],
                    success_rate=success_rate,
                    error_rate=error_rate,
                ))

            return metrics

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)

    async def reset_metrics(self, tool_name: str):
        """Reset tool metrics (unquarantine).

        Args:
            tool_name: Tool to reset
        """
        conn = await self.db.acquire() if hasattr(self.db, 'acquire') else self.db

        try:
            await conn.execute(
                "DELETE FROM tool_health WHERE tool_name = $1",
                tool_name
            )

            await conn.execute(
                "DELETE FROM tool_invocations WHERE tool_name = $1",
                tool_name
            )

            logger.info(f"Reset metrics for tool: {tool_name}")

        finally:
            if hasattr(self.db, 'release'):
                await self.db.release(conn)
