"""Confirmation handler for tool execution.

This module handles the confirmation flow for HIGH-risk tools,
replacing the original InteractiveAgent.ask_confirmation() behavior.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.interrupts import FlowInterrupt, InterruptKind, InterruptStatus


# Tools that require confirmation in "confirm" mode
HIGH_RISK_TOOLS: Set[str] = {
    "bash_execute",
    "write_file",
    "edit_file",
}

# Tools that are always safe (read-only)
SAFE_TOOLS: Set[str] = {
    "read_file",
    "find_files",
    "grep_search",
}

# Tools that are medium risk (logged but not blocked)
MEDIUM_RISK_TOOLS: Set[str] = {
    "run_tests",
}


class ConfirmationHandler:
    """Handles tool confirmation before execution.

    Maps to the original InteractiveAgent behavior:
    - mode="yolo": Never confirm, execute immediately
    - mode="confirm": Confirm HIGH-risk tools, skip whitelisted commands
    - mode="human": User provides commands directly (via CLARIFICATION)
    """

    def __init__(
        self,
        mode: str = "confirm",
        whitelist_patterns: Optional[List[str]] = None,
    ):
        """Initialize the confirmation handler.

        Args:
            mode: Execution mode (yolo, confirm, human)
            whitelist_patterns: Regex patterns for commands that skip confirmation
        """
        self.mode = mode
        self.whitelist_patterns = [
            re.compile(p) for p in (whitelist_patterns or [])
        ]

    def should_confirm(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if tool execution requires confirmation.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters

        Returns:
            True if confirmation is required
        """
        # Yolo mode: never confirm
        if self.mode == "yolo":
            return False

        # Human mode: no tool confirmation (user issues commands directly)
        if self.mode == "human":
            return False

        # Safe tools: never confirm
        if tool_name in SAFE_TOOLS:
            return False

        # Medium risk tools: log but don't block
        if tool_name in MEDIUM_RISK_TOOLS:
            return False

        # High risk tools: require confirmation
        if tool_name in HIGH_RISK_TOOLS:
            # Check whitelist patterns for bash commands
            if tool_name == "bash_execute":
                command = params.get("command", "")
                for pattern in self.whitelist_patterns:
                    if pattern.match(command):
                        return False
            return True

        # Unknown tools: confirm to be safe
        return True

    def create_confirmation_interrupt(
        self,
        tool_name: str,
        params: Dict[str, Any],
        request_id: str,
        user_id: str,
        session_id: str,
        envelope_id: Optional[str] = None,
    ) -> FlowInterrupt:
        """Create a confirmation interrupt for tool execution.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            request_id: Current request ID
            user_id: User ID
            session_id: Session ID
            envelope_id: Optional envelope ID

        Returns:
            FlowInterrupt for confirmation
        """
        # Format the confirmation message
        if tool_name == "bash_execute":
            command = params.get("command", "")
            message = f"Execute bash command?\n\n```bash\n{command}\n```"
        elif tool_name == "write_file":
            path = params.get("path", "unknown")
            message = f"Write file: {path}?"
        elif tool_name == "edit_file":
            path = params.get("path", "unknown")
            message = f"Edit file: {path}?"
        else:
            message = f"Execute {tool_name} with params: {params}?"

        return FlowInterrupt(
            id=str(uuid.uuid4()),
            kind=InterruptKind.CONFIRMATION,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            envelope_id=envelope_id,
            message=message,
            data={
                "tool_name": tool_name,
                "params": params,
            },
            status=InterruptStatus.PENDING,
        )

    def set_mode(self, mode: str) -> None:
        """Set the execution mode.

        Args:
            mode: New mode (yolo, confirm, human)
        """
        if mode not in ("yolo", "confirm", "human"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode


def create_confirmation_handler(
    mode: str = "confirm",
    whitelist_patterns: Optional[List[str]] = None,
) -> ConfirmationHandler:
    """Factory function to create a confirmation handler.

    Args:
        mode: Execution mode (yolo, confirm, human)
        whitelist_patterns: Regex patterns for whitelisted commands

    Returns:
        ConfirmationHandler instance
    """
    return ConfirmationHandler(
        mode=mode,
        whitelist_patterns=whitelist_patterns,
    )


__all__ = [
    "ConfirmationHandler",
    "create_confirmation_handler",
    "HIGH_RISK_TOOLS",
    "SAFE_TOOLS",
    "MEDIUM_RISK_TOOLS",
]
