"""Interactive runner with interrupt handling.

This module provides the main execution loop for CLI use with interactive
interrupt handling, replacing the original InteractiveAgent behavior.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.rule import Rule

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.interrupts import InterruptKind, InterruptResponse

from minisweagent.capability.orchestrator import SWEOrchestrator
from minisweagent.capability.interrupts import (
    CLIInterruptService,
    ModeManager,
    ExecutionMode,
)
from minisweagent.capability.interrupts.mode_manager import ResponseAction


class InteractiveRunner:
    """Runs orchestrator with interactive interrupt handling.

    This class provides the main execution loop that:
    - Runs the orchestrator
    - Handles confirmation interrupts via CLI prompts
    - Supports mode switching (/y, /c, /u, /h)
    - Handles KeyboardInterrupt gracefully
    - Supports exit confirmation
    """

    def __init__(
        self,
        orchestrator: SWEOrchestrator,
        interrupt_service: CLIInterruptService,
        mode_manager: ModeManager,
        confirm_exit: bool = True,
    ):
        """Initialize the interactive runner.

        Args:
            orchestrator: The SWE orchestrator
            interrupt_service: CLI interrupt service
            mode_manager: Mode manager for execution modes
            confirm_exit: Whether to confirm before exiting
        """
        self._orchestrator = orchestrator
        self._interrupt_service = interrupt_service
        self._mode_manager = mode_manager
        self._confirm_exit = confirm_exit
        self._console = Console(highlight=False)
        self._step_count = 0

    async def run(self, task: str) -> Dict[str, Any]:
        """Run task with interactive interrupt handling.

        Args:
            task: The task description

        Returns:
            Result dictionary
        """
        self._step_count = 0

        try:
            while True:
                self._step_count += 1
                self._console.print(Rule(f"Step {self._step_count}"))

                try:
                    result = await self._orchestrator.run(task)

                    # Check for pending interrupt
                    if result.get("status") == "confirmation_required":
                        action = await self._handle_confirmation(result)
                        if action == "continue":
                            continue
                        elif action == "abort":
                            return {"status": "aborted", "message": "User aborted"}

                    # Check for completion
                    if result.get("status") == "completed" or result.get("completed"):
                        if self._confirm_exit:
                            should_exit = await self._handle_exit_confirmation(result)
                            if not should_exit:
                                # User wants to continue with new task
                                continue
                        return result

                    # Check for error
                    if result.get("status") == "error":
                        return result

                    # Otherwise, continue
                    return result

                except KeyboardInterrupt:
                    action = await self._handle_keyboard_interrupt()
                    if action == "continue":
                        continue
                    elif action == "abort":
                        return {"status": "interrupted", "message": "User interrupted"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _handle_confirmation(self, result: Dict[str, Any]) -> str:
        """Handle a confirmation interrupt.

        Args:
            result: The result containing confirmation info

        Returns:
            Action: "continue" or "abort"
        """
        message = result.get("message", "Confirm action?")

        self._console.print(f"\n[bold yellow]{message}[/bold yellow]")
        self._console.print(
            "[green][bold]Enter[/bold] to confirm[/green], "
            "or [green]Type /h for help[/green]"
        )

        user_input = await self._prompt_user()

        response = InterruptResponse(
            text=user_input,
            approved=(user_input == "" or user_input == "/y"),
        )

        action, message = self._mode_manager.handle_response(response)

        if action == ResponseAction.APPROVED:
            return "continue"
        elif action == ResponseAction.REJECTED:
            self._console.print(f"[yellow]Rejected: {message}[/yellow]")
            return "continue"  # Continue but with rejection feedback
        elif action == ResponseAction.MODE_SWITCH:
            self._console.print(f"[green]{message}[/green]")
            return "continue"
        elif action == ResponseAction.HELP_SHOWN:
            self._console.print(message)
            return "continue"

        return "continue"

    async def _handle_exit_confirmation(self, result: Dict[str, Any]) -> bool:
        """Handle exit confirmation.

        Args:
            result: The completion result

        Returns:
            True if should exit, False if continue with new task
        """
        self._console.print(
            "\n[bold green]Agent wants to finish.[/bold green] "
            "[green]Press Enter to quit, or type a new task.[/green]"
        )

        user_input = await self._prompt_user()

        if user_input.strip():
            # User provided new task
            self._console.print(f"[yellow]New task: {user_input}[/yellow]")
            return False
        else:
            return True

    async def _handle_keyboard_interrupt(self) -> str:
        """Handle KeyboardInterrupt.

        Returns:
            Action: "continue" or "abort"
        """
        self._console.print(
            "\n\n[bold yellow]Interrupted.[/bold yellow] "
            "[green]Type a comment/command[/green] (/h for help)"
        )

        try:
            user_input = await self._prompt_user()

            if not user_input or user_input in ("/y", "/c", "/u"):
                user_input = "Temporary interruption caught."

            self._console.print(f"[yellow]Continuing with: {user_input}[/yellow]")
            return "continue"

        except KeyboardInterrupt:
            # Double interrupt = abort
            return "abort"

    async def _prompt_user(self) -> str:
        """Prompt user for input.

        Returns:
            User input string
        """
        from prompt_toolkit.shortcuts import PromptSession

        prompt_session = PromptSession()

        # Run in thread to allow async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: prompt_session.prompt("[bold yellow]>[/bold yellow] "),
        )


def create_interactive_runner(
    orchestrator: SWEOrchestrator,
    mode: str = "confirm",
    confirm_exit: bool = True,
) -> InteractiveRunner:
    """Factory function to create an interactive runner.

    Args:
        orchestrator: The SWE orchestrator
        mode: Initial execution mode (yolo, confirm, human)
        confirm_exit: Whether to confirm before exiting

    Returns:
        InteractiveRunner instance
    """
    from minisweagent.capability.interrupts import (
        create_cli_interrupt_service,
    )
    from minisweagent.capability.interrupts.mode_manager import create_mode_manager

    interrupt_service = create_cli_interrupt_service()
    mode_manager = create_mode_manager(initial_mode=mode)

    return InteractiveRunner(
        orchestrator=orchestrator,
        interrupt_service=interrupt_service,
        mode_manager=mode_manager,
        confirm_exit=confirm_exit,
    )


__all__ = [
    "InteractiveRunner",
    "create_interactive_runner",
]
