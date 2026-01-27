"""Clarification Handler - Request user input for ambiguous tasks."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ClarificationRequest:
    """A request for user clarification."""

    request_id: str
    question: str
    options: List[str]
    context: Dict[str, Any]
    selected: Optional[str] = None


class ClarificationHandler:
    """Handle clarification requests from agents."""

    def __init__(self, cli_service=None):
        """Initialize clarification handler.

        Args:
            cli_service: CLI service for interactive prompts (optional)
        """
        self.cli_service = cli_service
        self.pending_requests = {}

    async def request_clarification(
        self,
        question: str,
        options: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Request clarification from user.

        Args:
            question: Question to ask
            options: List of possible answers
            context: Additional context

        Returns:
            Selected option
        """
        if context is None:
            context = {}

        request_id = f"clarify_{len(self.pending_requests)}"

        request = ClarificationRequest(
            request_id=request_id,
            question=question,
            options=options,
            context=context,
        )

        self.pending_requests[request_id] = request

        # If CLI service available, prompt immediately
        if self.cli_service:
            selected = await self.cli_service.prompt_choice(question, options)
            request.selected = selected
            return selected

        # Otherwise, wait for external response (webhook, etc.)
        logger.warning(f"Clarification request pending: {request_id}")
        raise NotImplementedError("Webhook-based clarification not yet implemented")

    async def respond_to_request(self, request_id: str, selected: str):
        """Respond to a clarification request.

        Args:
            request_id: Request identifier
            selected: Selected option
        """
        if request_id not in self.pending_requests:
            raise ValueError(f"Unknown request: {request_id}")

        request = self.pending_requests[request_id]
        if selected not in request.options:
            raise ValueError(f"Invalid option: {selected}")

        request.selected = selected
        logger.info(f"Clarification {request_id}: {selected}")

    def get_pending_requests(self) -> List[ClarificationRequest]:
        """Get list of pending clarification requests.

        Returns:
            List of pending requests
        """
        return [r for r in self.pending_requests.values() if r.selected is None]
