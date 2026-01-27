"""Tests for ConfirmingToolExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add jeeves-core to path for protocol imports
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.interrupts import InterruptResponse, InterruptStatus

from minisweagent.capability.tools.confirming_executor import (
    ConfirmingToolExecutor,
    create_confirming_executor,
)
from minisweagent.capability.interrupts.mode_manager import (
    ModeManager,
    ExecutionMode,
    ResponseAction,
)
from minisweagent.capability.interrupts.confirmation_handler import ConfirmationHandler


class MockToolExecutor:
    """Mock tool executor for testing."""

    def __init__(self, return_value=None):
        self.return_value = return_value or {"status": "success", "output": "done"}
        self.executed_calls = []

    async def execute(self, tool_name: str, params: dict):
        self.executed_calls.append((tool_name, params))
        return self.return_value


class MockInterruptService:
    """Mock interrupt service for testing."""

    def __init__(self):
        self.interrupts = []

    async def create_interrupt(self, **kwargs):
        self.interrupts.append(kwargs)


class MockToolHealthService:
    """Mock tool health service for testing."""

    def __init__(self, status="healthy"):
        self.status = status
        self.recorded_invocations = []

    async def get_tool_status(self, tool_name: str):
        return self.status

    async def record_invocation(self, tool_name: str, success: bool, latency_ms: int, error_message: str = None):
        self.recorded_invocations.append({
            "tool_name": tool_name,
            "success": success,
            "latency_ms": latency_ms,
            "error_message": error_message,
        })


class TestConfirmingExecutorBasic:
    """Basic tests for ConfirmingToolExecutor."""

    @pytest.fixture
    def inner_executor(self):
        return MockToolExecutor()

    @pytest.fixture
    def confirmation_handler(self):
        handler = MagicMock(spec=ConfirmationHandler)
        handler.should_confirm.return_value = False
        handler.mode = "yolo"
        return handler

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.YOLO
        return manager

    @pytest.fixture
    def executor(self, inner_executor, confirmation_handler, interrupt_service, mode_manager):
        return ConfirmingToolExecutor(
            inner_executor=inner_executor,
            confirmation_handler=confirmation_handler,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
        )

    @pytest.mark.asyncio
    async def test_execute_in_yolo_mode_no_confirmation(self, executor, inner_executor):
        """Test that yolo mode executes without confirmation."""
        result = await executor.execute("bash_execute", {"command": "echo hello"})

        assert result["status"] == "success"
        assert len(inner_executor.executed_calls) == 1

    @pytest.mark.asyncio
    async def test_set_request_context(self, executor):
        """Test setting request context."""
        executor.set_request_context("req-123", "user-456", "sess-789")

        assert executor._request_id == "req-123"
        assert executor._user_id == "user-456"
        assert executor._session_id == "sess-789"


class TestConfirmationRequired:
    """Tests for confirmation requirement handling."""

    @pytest.fixture
    def inner_executor(self):
        return MockToolExecutor()

    @pytest.fixture
    def confirmation_handler(self):
        handler = MagicMock(spec=ConfirmationHandler)
        handler.should_confirm.return_value = True
        handler.mode = "confirm"
        handler.create_confirmation_interrupt.return_value = MagicMock(
            id="int-123",
            kind="confirmation",
            request_id="req-123",
            user_id="user-456",
            session_id="sess-789",
            envelope_id="env-123",
            message="Confirm execution of bash_execute?",
            data={"tool": "bash_execute"},
        )
        return handler

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.CONFIRM
        return manager

    @pytest.fixture
    def executor(self, inner_executor, confirmation_handler, interrupt_service, mode_manager):
        return ConfirmingToolExecutor(
            inner_executor=inner_executor,
            confirmation_handler=confirmation_handler,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
        )

    @pytest.mark.asyncio
    async def test_returns_confirmation_required(self, executor, inner_executor):
        """Test that confirmation required is returned."""
        result = await executor.execute("bash_execute", {"command": "rm -rf /"})

        assert result["status"] == "confirmation_required"
        assert result["tool_name"] == "bash_execute"
        assert "interrupt_id" in result
        # Inner executor should NOT have been called
        assert len(inner_executor.executed_calls) == 0

    @pytest.mark.asyncio
    async def test_creates_interrupt(self, executor, interrupt_service):
        """Test that interrupt is created."""
        await executor.execute("bash_execute", {"command": "rm -rf /"})

        assert len(interrupt_service.interrupts) == 1


class TestToolHealthMonitoring:
    """Tests for tool health monitoring (L7)."""

    @pytest.fixture
    def inner_executor(self):
        return MockToolExecutor()

    @pytest.fixture
    def confirmation_handler(self):
        handler = MagicMock(spec=ConfirmationHandler)
        handler.should_confirm.return_value = False
        handler.mode = "yolo"
        return handler

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.YOLO
        return manager

    @pytest.mark.asyncio
    async def test_quarantined_tool_blocked(self, inner_executor, confirmation_handler, interrupt_service, mode_manager):
        """Test that quarantined tools are blocked."""
        health_service = MockToolHealthService(status="quarantined")
        executor = ConfirmingToolExecutor(
            inner_executor=inner_executor,
            confirmation_handler=confirmation_handler,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            tool_health_service=health_service,
        )

        result = await executor.execute("risky_tool", {})

        assert result["status"] == "quarantined"
        assert "quarantined" in result["error"]
        # Inner executor should NOT have been called
        assert len(inner_executor.executed_calls) == 0

    @pytest.mark.asyncio
    async def test_degraded_tool_allowed_with_warning(self, inner_executor, confirmation_handler, interrupt_service, mode_manager, capsys):
        """Test that degraded tools are allowed with warning."""
        health_service = MockToolHealthService(status="degraded")
        executor = ConfirmingToolExecutor(
            inner_executor=inner_executor,
            confirmation_handler=confirmation_handler,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            tool_health_service=health_service,
        )

        result = await executor.execute("slow_tool", {})

        assert result["status"] == "success"
        assert len(inner_executor.executed_calls) == 1
        # Check warning was printed
        captured = capsys.readouterr()
        assert "degraded" in captured.out

    @pytest.mark.asyncio
    async def test_healthy_tool_records_invocation(self, inner_executor, confirmation_handler, interrupt_service, mode_manager):
        """Test that healthy tool invocations are recorded."""
        health_service = MockToolHealthService(status="healthy")
        executor = ConfirmingToolExecutor(
            inner_executor=inner_executor,
            confirmation_handler=confirmation_handler,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            tool_health_service=health_service,
        )

        await executor.execute("bash_execute", {"command": "echo hello"})

        assert len(health_service.recorded_invocations) == 1
        assert health_service.recorded_invocations[0]["tool_name"] == "bash_execute"
        assert health_service.recorded_invocations[0]["success"] is True

    @pytest.mark.asyncio
    async def test_failed_invocation_recorded(self, confirmation_handler, interrupt_service, mode_manager):
        """Test that failed invocations are recorded."""
        inner_executor = MockToolExecutor(return_value={"status": "error", "error": "Command failed"})
        health_service = MockToolHealthService(status="healthy")
        executor = ConfirmingToolExecutor(
            inner_executor=inner_executor,
            confirmation_handler=confirmation_handler,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            tool_health_service=health_service,
        )

        await executor.execute("bash_execute", {"command": "exit 1"})

        assert len(health_service.recorded_invocations) == 1
        assert health_service.recorded_invocations[0]["success"] is False


class TestExecuteWithConfirmation:
    """Tests for execute_with_confirmation method."""

    @pytest.fixture
    def inner_executor(self):
        return MockToolExecutor()

    @pytest.fixture
    def confirmation_handler(self):
        handler = MagicMock(spec=ConfirmationHandler)
        handler.mode = "confirm"
        return handler

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.CONFIRM
        return manager

    @pytest.fixture
    def executor(self, inner_executor, confirmation_handler, interrupt_service, mode_manager):
        return ConfirmingToolExecutor(
            inner_executor=inner_executor,
            confirmation_handler=confirmation_handler,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
        )

    @pytest.mark.asyncio
    async def test_approved_executes_tool(self, executor, inner_executor, mode_manager):
        """Test that approved confirmation executes tool."""
        mode_manager.handle_response.return_value = (ResponseAction.APPROVED, None)
        response = MagicMock(spec=InterruptResponse)

        result = await executor.execute_with_confirmation("bash_execute", {"command": "ls"}, response)

        assert result["status"] == "success"
        assert len(inner_executor.executed_calls) == 1

    @pytest.mark.asyncio
    async def test_rejected_returns_rejection(self, executor, inner_executor, mode_manager):
        """Test that rejected confirmation returns rejection."""
        mode_manager.handle_response.return_value = (ResponseAction.REJECTED, "User said no")
        response = MagicMock(spec=InterruptResponse)

        result = await executor.execute_with_confirmation("bash_execute", {"command": "rm -rf /"}, response)

        assert result["status"] == "rejected"
        assert "User said no" in result["message"]
        # Inner executor should NOT have been called
        assert len(inner_executor.executed_calls) == 0

    @pytest.mark.asyncio
    async def test_mode_switch_to_yolo_executes(self, executor, inner_executor, mode_manager):
        """Test that mode switch to yolo executes tool."""
        mode_manager.handle_response.return_value = (ResponseAction.MODE_SWITCH, "Switched to yolo")
        mode_manager.is_yolo_mode.return_value = True
        response = MagicMock(spec=InterruptResponse)

        result = await executor.execute_with_confirmation("bash_execute", {"command": "ls"}, response)

        assert result["status"] == "success"
        assert len(inner_executor.executed_calls) == 1

    @pytest.mark.asyncio
    async def test_mode_switch_not_yolo_returns_status(self, executor, inner_executor, mode_manager):
        """Test that mode switch to non-yolo returns status."""
        mode_manager.handle_response.return_value = (ResponseAction.MODE_SWITCH, "Switched to human")
        mode_manager.is_yolo_mode.return_value = False
        mode_manager.mode = ExecutionMode.HUMAN
        response = MagicMock(spec=InterruptResponse)

        result = await executor.execute_with_confirmation("bash_execute", {"command": "ls"}, response)

        assert result["status"] == "mode_switched"
        assert result["new_mode"] == "human"
        # Inner executor should NOT have been called
        assert len(inner_executor.executed_calls) == 0

    @pytest.mark.asyncio
    async def test_help_shown_returns_status(self, executor, inner_executor, mode_manager):
        """Test that help shown returns status."""
        mode_manager.handle_response.return_value = (ResponseAction.HELP_SHOWN, "Help message here")
        response = MagicMock(spec=InterruptResponse)

        result = await executor.execute_with_confirmation("bash_execute", {}, response)

        assert result["status"] == "help_shown"
        assert "Help message" in result["message"]


class TestCreateConfirmingExecutor:
    """Tests for factory function."""

    def test_create_confirming_executor_default(self):
        """Test factory creates executor with defaults."""
        inner = MockToolExecutor()
        executor = create_confirming_executor(inner)

        assert isinstance(executor, ConfirmingToolExecutor)
        assert executor._inner == inner

    def test_create_confirming_executor_yolo_mode(self):
        """Test factory creates executor in yolo mode."""
        inner = MockToolExecutor()
        executor = create_confirming_executor(inner, mode="yolo")

        assert isinstance(executor, ConfirmingToolExecutor)
        assert executor._mode_manager.mode == ExecutionMode.YOLO

    def test_create_confirming_executor_with_whitelist(self):
        """Test factory creates executor with whitelist."""
        inner = MockToolExecutor()
        executor = create_confirming_executor(
            inner,
            mode="confirm",
            whitelist_patterns=["echo *", "ls *"]
        )

        assert isinstance(executor, ConfirmingToolExecutor)

    def test_create_confirming_executor_with_health_service(self):
        """Test factory creates executor with health service."""
        inner = MockToolExecutor()
        health_service = MockToolHealthService()
        executor = create_confirming_executor(
            inner,
            tool_health_service=health_service
        )

        assert executor._tool_health_service == health_service
