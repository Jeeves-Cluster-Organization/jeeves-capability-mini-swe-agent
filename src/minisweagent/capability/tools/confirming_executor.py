"""Confirming tool executor wrapper.

This module provides a ToolExecutor wrapper that adds confirmation
handling before executing HIGH-risk tools.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.agents import ToolExecutor
from protocols.interrupts import InterruptResponse, InterruptStatus

from minisweagent.capability.interrupts.confirmation_handler import ConfirmationHandler
from minisweagent.capability.interrupts.cli_service import CLIInterruptService
from minisweagent.capability.interrupts.mode_manager import (
    ModeManager,
    ResponseAction,
    ExecutionMode,
)


class ConfirmingToolExecutor:
    """Tool executor that requires confirmation for HIGH-risk tools.

    Wraps another tool executor and adds confirmation checks before
    executing HIGH-risk tools. Uses the interrupt system to pause
    execution and wait for user approval.
    """

    def __init__(
        self,
        inner_executor: ToolExecutor,
        confirmation_handler: ConfirmationHandler,
        interrupt_service: CLIInterruptService,
        mode_manager: ModeManager,
        request_id: str = "",
        user_id: str = "anonymous",
        session_id: str = "",
    ):
        """Initialize the confirming tool executor.

        Args:
            inner_executor: The underlying tool executor
            confirmation_handler: Handler for confirmation logic
            interrupt_service: Service for managing interrupts
            mode_manager: Manager for execution modes
            request_id: Current request ID
            user_id: Current user ID
            session_id: Current session ID
        """
        self._inner = inner_executor
        self._confirmation_handler = confirmation_handler
        self._interrupt_service = interrupt_service
        self._mode_manager = mode_manager
        self._request_id = request_id
        self._user_id = user_id
        self._session_id = session_id

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool with confirmation check.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters

        Returns:
            Tool execution result or rejection status
        """
        # Sync mode manager with confirmation handler
        self._confirmation_handler.mode = self._mode_manager.mode.value

        # Check if confirmation is needed
        if self._confirmation_handler.should_confirm(tool_name, params):
            # Create confirmation interrupt
            interrupt = self._confirmation_handler.create_confirmation_interrupt(
                tool_name=tool_name,
                params=params,
                request_id=self._request_id,
                user_id=self._user_id,
                session_id=self._session_id,
            )

            # Store interrupt and return pending status
            await self._interrupt_service.create_interrupt(
                kind=interrupt.kind,
                request_id=interrupt.request_id,
                user_id=interrupt.user_id,
                session_id=interrupt.session_id,
                envelope_id=interrupt.envelope_id,
                message=interrupt.message,
                data=interrupt.data,
            )

            return {
                "status": "confirmation_required",
                "interrupt_id": interrupt.id,
                "tool_name": tool_name,
                "params": params,
                "message": interrupt.message,
            }

        # No confirmation needed, execute directly
        return await self._inner.execute(tool_name, params)

    async def execute_with_confirmation(
        self,
        tool_name: str,
        params: Dict[str, Any],
        response: InterruptResponse,
    ) -> Dict[str, Any]:
        """Execute a tool after receiving confirmation response.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            response: User's confirmation response

        Returns:
            Tool execution result or rejection status
        """
        # Handle the response
        action, message = self._mode_manager.handle_response(response)

        if action == ResponseAction.APPROVED:
            # User approved, execute the tool
            return await self._inner.execute(tool_name, params)

        elif action == ResponseAction.REJECTED:
            # User rejected, return rejection status
            return {
                "status": "rejected",
                "tool_name": tool_name,
                "message": message or "User rejected the action.",
            }

        elif action == ResponseAction.MODE_SWITCH:
            # Mode switched, re-evaluate confirmation
            # If now in yolo mode, execute directly
            if self._mode_manager.is_yolo_mode():
                return await self._inner.execute(tool_name, params)
            # Otherwise, return status indicating mode switch
            return {
                "status": "mode_switched",
                "new_mode": self._mode_manager.mode.value,
                "message": message,
            }

        elif action == ResponseAction.HELP_SHOWN:
            # Help was shown, return status
            return {
                "status": "help_shown",
                "message": message,
            }

        # Fallback: reject
        return {
            "status": "rejected",
            "message": "Unknown response action.",
        }

    def set_request_context(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """Update the request context.

        Args:
            request_id: New request ID
            user_id: New user ID
            session_id: New session ID
        """
        self._request_id = request_id
        self._user_id = user_id
        self._session_id = session_id


def create_confirming_executor(
    inner_executor: ToolExecutor,
    mode: str = "confirm",
    whitelist_patterns: Optional[list] = None,
) -> ConfirmingToolExecutor:
    """Factory function to create a confirming tool executor.

    Args:
        inner_executor: The underlying tool executor
        mode: Initial execution mode (yolo, confirm, human)
        whitelist_patterns: Patterns for commands that skip confirmation

    Returns:
        ConfirmingToolExecutor instance
    """
    from minisweagent.capability.interrupts import (
        create_confirmation_handler,
        create_cli_interrupt_service,
    )
    from minisweagent.capability.interrupts.mode_manager import create_mode_manager

    confirmation_handler = create_confirmation_handler(
        mode=mode,
        whitelist_patterns=whitelist_patterns,
    )
    interrupt_service = create_cli_interrupt_service()
    mode_manager = create_mode_manager(initial_mode=mode)

    return ConfirmingToolExecutor(
        inner_executor=inner_executor,
        confirmation_handler=confirmation_handler,
        interrupt_service=interrupt_service,
        mode_manager=mode_manager,
    )


__all__ = [
    "ConfirmingToolExecutor",
    "create_confirming_executor",
]
