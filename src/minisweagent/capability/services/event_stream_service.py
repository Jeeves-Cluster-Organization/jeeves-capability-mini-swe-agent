"""Event Streaming Service - Real-time pipeline events."""

import logging
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventCategory(str, Enum):
    """Event categories."""

    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    TOOL_STARTED = "tool_started"
    TOOL_EXECUTED = "tool_executed"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AgentEvent:
    """Agent execution event."""

    event_id: str
    category: EventCategory
    agent_name: Optional[str]
    message: str
    metadata: dict
    timestamp: datetime


class EventStreamService:
    """Service for streaming pipeline events in real-time."""

    def __init__(self):
        """Initialize event stream service."""
        self.event_queue = asyncio.Queue()
        self.subscribers = []

    async def emit(self, category: EventCategory, message: str,
                   agent_name: Optional[str] = None, **metadata):
        """Emit an event.

        Args:
            category: Event category
            message: Event message
            agent_name: Agent that generated event
            **metadata: Additional event metadata
        """
        event = AgentEvent(
            event_id=f"{datetime.now().timestamp()}",
            category=category,
            agent_name=agent_name,
            message=message,
            metadata=metadata,
            timestamp=datetime.now(),
        )

        # Add to queue
        await self.event_queue.put(event)

        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

    async def stream_events(self) -> AsyncIterator[AgentEvent]:
        """Stream events as they occur.

        Yields:
            AgentEvent instances
        """
        while True:
            event = await self.event_queue.get()
            yield event

    def subscribe(self, callback):
        """Subscribe to events.

        Args:
            callback: Async callback function(event)
        """
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """Unsubscribe from events.

        Args:
            callback: Previously subscribed callback
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
