"""CLI interrupt service for interactive execution.

This module provides an in-memory implementation of InterruptServiceProtocol
for CLI use, replacing the original InteractiveAgent blocking prompts.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.interrupts import (
    FlowInterrupt,
    InterruptKind,
    InterruptResponse,
    InterruptServiceProtocol,
    InterruptStatus,
)


class CLIInterruptService(InterruptServiceProtocol):
    """In-memory interrupt service for CLI use.

    This service:
    - Stores pending interrupts in a dictionary
    - Provides async methods to create, respond to, and wait for interrupts
    - Uses asyncio events for blocking until response
    """

    def __init__(self):
        """Initialize the CLI interrupt service."""
        self._interrupts: Dict[str, FlowInterrupt] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def create_interrupt(
        self,
        kind: InterruptKind,
        request_id: str,
        user_id: str,
        session_id: str,
        envelope_id: Optional[str] = None,
        question: Optional[str] = None,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> FlowInterrupt:
        """Create a new interrupt.

        Args:
            kind: Type of interrupt
            request_id: Request ID
            user_id: User ID
            session_id: Session ID
            envelope_id: Optional envelope ID
            question: Question for clarification interrupts
            message: Message for confirmation interrupts
            data: Additional data
            ttl_seconds: Time-to-live (unused in CLI)

        Returns:
            Created FlowInterrupt
        """
        interrupt_id = str(uuid.uuid4())

        interrupt = FlowInterrupt(
            id=interrupt_id,
            kind=kind,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            envelope_id=envelope_id,
            question=question,
            message=message,
            data=data or {},
            status=InterruptStatus.PENDING,
        )

        async with self._lock:
            self._interrupts[interrupt_id] = interrupt
            self._events[interrupt_id] = asyncio.Event()

        return interrupt

    async def get_interrupt(self, interrupt_id: str) -> Optional[FlowInterrupt]:
        """Get an interrupt by ID.

        Args:
            interrupt_id: The interrupt ID

        Returns:
            FlowInterrupt if found, None otherwise
        """
        async with self._lock:
            return self._interrupts.get(interrupt_id)

    async def respond(
        self,
        interrupt_id: str,
        response: InterruptResponse,
    ) -> Optional[FlowInterrupt]:
        """Respond to an interrupt.

        Args:
            interrupt_id: The interrupt ID
            response: The response

        Returns:
            Updated FlowInterrupt if found
        """
        async with self._lock:
            interrupt = self._interrupts.get(interrupt_id)
            if interrupt is None:
                return None

            interrupt.response = response
            interrupt.status = InterruptStatus.RESOLVED
            interrupt.resolved_at = datetime.now(timezone.utc)

            # Signal that the interrupt has been resolved
            event = self._events.get(interrupt_id)
            if event:
                event.set()

            return interrupt

    async def list_pending(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> List[FlowInterrupt]:
        """List pending interrupts.

        Args:
            user_id: Filter by user ID
            session_id: Filter by session ID
            request_id: Filter by request ID

        Returns:
            List of pending FlowInterrupts
        """
        async with self._lock:
            result = []
            for interrupt in self._interrupts.values():
                if interrupt.status != InterruptStatus.PENDING:
                    continue
                if user_id and interrupt.user_id != user_id:
                    continue
                if session_id and interrupt.session_id != session_id:
                    continue
                if request_id and interrupt.request_id != request_id:
                    continue
                result.append(interrupt)
            return result

    async def cancel(self, interrupt_id: str) -> bool:
        """Cancel an interrupt.

        Args:
            interrupt_id: The interrupt ID

        Returns:
            True if cancelled, False if not found
        """
        async with self._lock:
            interrupt = self._interrupts.get(interrupt_id)
            if interrupt is None:
                return False

            interrupt.status = InterruptStatus.CANCELLED
            interrupt.resolved_at = datetime.now(timezone.utc)

            # Signal that the interrupt has been resolved
            event = self._events.get(interrupt_id)
            if event:
                event.set()

            return True

    async def wait_for_response(
        self,
        interrupt_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[InterruptResponse]:
        """Wait for an interrupt to be resolved.

        Args:
            interrupt_id: The interrupt ID
            timeout: Optional timeout in seconds

        Returns:
            InterruptResponse if resolved, None if timed out or not found
        """
        async with self._lock:
            event = self._events.get(interrupt_id)
            if event is None:
                return None

        try:
            if timeout:
                await asyncio.wait_for(event.wait(), timeout)
            else:
                await event.wait()
        except asyncio.TimeoutError:
            return None

        async with self._lock:
            interrupt = self._interrupts.get(interrupt_id)
            if interrupt is None:
                return None
            return interrupt.response

    async def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Clean up old resolved interrupts.

        Args:
            max_age_seconds: Maximum age for resolved interrupts

        Returns:
            Number of cleaned up interrupts
        """
        now = datetime.now(timezone.utc)
        cleaned = 0

        async with self._lock:
            to_remove = []
            for interrupt_id, interrupt in self._interrupts.items():
                if interrupt.status in (InterruptStatus.RESOLVED, InterruptStatus.CANCELLED):
                    if interrupt.resolved_at:
                        age = (now - interrupt.resolved_at).total_seconds()
                        if age > max_age_seconds:
                            to_remove.append(interrupt_id)

            for interrupt_id in to_remove:
                del self._interrupts[interrupt_id]
                if interrupt_id in self._events:
                    del self._events[interrupt_id]
                cleaned += 1

        return cleaned


def create_cli_interrupt_service() -> CLIInterruptService:
    """Factory function to create a CLI interrupt service.

    Returns:
        CLIInterruptService instance
    """
    return CLIInterruptService()


__all__ = [
    "CLIInterruptService",
    "create_cli_interrupt_service",
]
