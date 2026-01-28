"""Confirming tool executor wrapper.

This module provides a ToolExecutor wrapper that adds confirmation
handling before executing HIGH-risk tools and tool health monitoring (L7).
"""

import time
from typing import Any, Dict, Optional

# jeeves-core is now a proper package - no sys.path manipulation needed

from jeeves_infra.runtime import ToolExecutor
from jeeves_infra.protocols import InterruptResponse, InterruptStatus

from minisweagent.capability.interrupts.confirmation_handler import ConfirmationHandler
from minisweagent.capability.interrupts.cli_service import CLIInterruptService
from minisweagent.capability.interrupts.mode_manager import (
    ModeManager,
    ResponseAction,
    ExecutionMode,
)


class ConfirmingToolExecutor:
    """Tool executor that requires confirmation for HIGH-risk tools and monitors tool health (L7).

    Wraps another tool executor and adds:
    1. Confirmation checks for HIGH-risk tools
    2. Tool health monitoring (success/failure tracking)
    3. Automatic quarantine of failing tools

    Uses the interrupt system to pause execution and wait for user approval.
    """

    def __init__(
        self,
        inner_executor: ToolExecutor,
        confirmation_handler: ConfirmationHandler,
        interrupt_service: CLIInterruptService,
        mode_manager: ModeManager,
        tool_health_service: Optional[Any] = None,
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
            tool_health_service: Optional tool health service for monitoring (L7)
            request_id: Current request ID
            user_id: Current user ID
            session_id: Current session ID
        """
        self._inner = inner_executor
        self._confirmation_handler = confirmation_handler
        self._interrupt_service = interrupt_service
        self._mode_manager = mode_manager
        self._tool_health_service = tool_health_service
        self._request_id = request_id
        self._user_id = user_id
        self._session_id = session_id

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool with confirmation check and health monitoring.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters

        Returns:
            Tool execution result or rejection status
        """
        # Check tool health status (L7)
        if self._tool_health_service:
            status = await self._tool_health_service.get_tool_status(tool_name)
            if status == "quarantined":
                return {
                    "status": "quarantined",
                    "error": f"Tool {tool_name} is quarantined due to high failure rate (>50%). Use tool-reset to unquarantine.",
                    "tool_name": tool_name,
                }
            elif status == "degraded":
                # Log warning but allow execution
                print(f"⚠️  Warning: Tool {tool_name} is degraded (10-50% error rate)")

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

        # Execute tool with health monitoring
        start_time = time.time()
        success = False
        error_message = None

        try:
            result = await self._inner.execute(tool_name, params)
            success = result.get("status") in ("success", "completed", None)
            error_message = result.get("error") if not success else None
            return result
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            # Record invocation (L7)
            if self._tool_health_service:
                latency_ms = int((time.time() - start_time) * 1000)
                try:
                    await self._tool_health_service.record_invocation(
                        tool_name=tool_name,
                        success=success,
                        latency_ms=latency_ms,
                        error_message=error_message,
                    )
                except Exception:
                    # Don't fail execution if health monitoring fails
                    pass

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
            # User approved, execute with health monitoring
            start_time = time.time()
            success = False
            error_message = None

            try:
                result = await self._inner.execute(tool_name, params)
                success = result.get("status") in ("success", "completed", None)
                error_message = result.get("error") if not success else None
                return result
            except Exception as e:
                error_message = str(e)
                raise
            finally:
                # Record invocation (L7)
                if self._tool_health_service:
                    latency_ms = int((time.time() - start_time) * 1000)
                    try:
                        await self._tool_health_service.record_invocation(
                            tool_name=tool_name,
                            success=success,
                            latency_ms=latency_ms,
                            error_message=error_message,
                        )
                    except Exception:
                        pass

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
                # Execute with health monitoring
                start_time = time.time()
                success = False
                error_message = None

                try:
                    result = await self._inner.execute(tool_name, params)
                    success = result.get("status") in ("success", "completed", None)
                    error_message = result.get("error") if not success else None
                    return result
                except Exception as e:
                    error_message = str(e)
                    raise
                finally:
                    if self._tool_health_service:
                        latency_ms = int((time.time() - start_time) * 1000)
                        try:
                            await self._tool_health_service.record_invocation(
                                tool_name=tool_name,
                                success=success,
                                latency_ms=latency_ms,
                                error_message=error_message,
                            )
                        except Exception:
                            pass
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
    tool_health_service: Optional[Any] = None,
) -> ConfirmingToolExecutor:
    """Factory function to create a confirming tool executor.

    Args:
        inner_executor: The underlying tool executor
        mode: Initial execution mode (yolo, confirm, human)
        whitelist_patterns: Patterns for commands that skip confirmation
        tool_health_service: Optional tool health service for monitoring (L7)

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
        tool_health_service=tool_health_service,
    )


__all__ = [
    "ConfirmingToolExecutor",
    "create_confirming_executor",
]
