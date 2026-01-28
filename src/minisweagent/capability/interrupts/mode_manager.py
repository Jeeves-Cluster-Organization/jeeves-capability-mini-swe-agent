"""Mode manager for interactive execution.

This module handles mode switching commands from interrupt responses,
replacing the original InteractiveAgent._prompt_and_handle_special() behavior.
"""

from enum import Enum
from typing import Optional, Tuple

# jeeves-core is now a proper package - no sys.path manipulation needed

from jeeves_infra.protocols import InterruptResponse


class ExecutionMode(str, Enum):
    """Execution modes for the SWE agent."""
    YOLO = "yolo"       # Execute LM commands without confirmation
    CONFIRM = "confirm"  # Ask for confirmation before executing LM commands
    HUMAN = "human"      # Execute commands issued by the user


class ResponseAction(str, Enum):
    """Actions resulting from handling an interrupt response."""
    APPROVED = "approved"           # User approved the action
    REJECTED = "rejected"           # User rejected the action
    MODE_SWITCH = "mode_switch"     # User switched mode
    HELP_SHOWN = "help_shown"       # Help was displayed


# Mode switching commands
MODE_COMMANDS = {
    "/y": ExecutionMode.YOLO,
    "/c": ExecutionMode.CONFIRM,
    "/u": ExecutionMode.HUMAN,
}

HELP_TEXT = """
Current mode: {mode}

Available commands:
  /y  - Switch to yolo mode (execute LM commands without confirmation)
  /c  - Switch to confirm mode (ask for confirmation before executing)
  /u  - Switch to human mode (you type commands directly)
  /h  - Show this help

In confirm mode:
  Press Enter to confirm execution
  Type a message to reject with feedback
"""


class ModeManager:
    """Manages execution modes and handles mode switching.

    Handles the original InteractiveAgent mode commands:
    - /y: Switch to yolo mode
    - /c: Switch to confirm mode
    - /u: Switch to human mode
    - /h: Show help
    """

    def __init__(self, initial_mode: ExecutionMode = ExecutionMode.CONFIRM):
        """Initialize the mode manager.

        Args:
            initial_mode: Initial execution mode
        """
        self.mode = initial_mode

    def handle_response(
        self,
        response: InterruptResponse,
    ) -> Tuple[ResponseAction, Optional[str]]:
        """Handle an interrupt response and determine action.

        Args:
            response: The user's response to an interrupt

        Returns:
            Tuple of (action, optional message/detail)
        """
        text = (response.text or "").strip()

        # Check for help command
        if text == "/h":
            return ResponseAction.HELP_SHOWN, self.get_help_text()

        # Check for mode switch commands
        if text in MODE_COMMANDS:
            new_mode = MODE_COMMANDS[text]
            if new_mode == self.mode:
                return ResponseAction.HELP_SHOWN, f"Already in {self.mode.value} mode."
            old_mode = self.mode
            self.mode = new_mode
            return ResponseAction.MODE_SWITCH, f"Switched from {old_mode.value} to {new_mode.value} mode."

        # Check for approval
        if response.approved is True or text == "" or text == "/y":
            return ResponseAction.APPROVED, None

        # Otherwise, it's a rejection with feedback
        return ResponseAction.REJECTED, text if text else "User rejected the action."

    def get_help_text(self) -> str:
        """Get the help text for current mode.

        Returns:
            Formatted help text
        """
        return HELP_TEXT.format(mode=self.mode.value)

    def set_mode(self, mode: ExecutionMode) -> None:
        """Set the execution mode.

        Args:
            mode: New execution mode
        """
        self.mode = mode

    def is_yolo_mode(self) -> bool:
        """Check if in yolo mode (no confirmations)."""
        return self.mode == ExecutionMode.YOLO

    def is_confirm_mode(self) -> bool:
        """Check if in confirm mode."""
        return self.mode == ExecutionMode.CONFIRM

    def is_human_mode(self) -> bool:
        """Check if in human mode (user commands)."""
        return self.mode == ExecutionMode.HUMAN


def create_mode_manager(
    initial_mode: str = "confirm",
) -> ModeManager:
    """Factory function to create a mode manager.

    Args:
        initial_mode: Initial mode as string (yolo, confirm, human)

    Returns:
        ModeManager instance
    """
    mode = ExecutionMode(initial_mode)
    return ModeManager(initial_mode=mode)


__all__ = [
    "ModeManager",
    "ExecutionMode",
    "ResponseAction",
    "MODE_COMMANDS",
    "HELP_TEXT",
    "create_mode_manager",
]
