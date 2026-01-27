"""Tests for InteractiveRunner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add jeeves-core to path for protocol imports
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from protocols.interrupts import InterruptResponse

from minisweagent.capability.cli.interactive_runner import (
    InteractiveRunner,
    create_interactive_runner,
)
from minisweagent.capability.interrupts.mode_manager import (
    ModeManager,
    ExecutionMode,
    ResponseAction,
)


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self, results=None):
        self.results = results or [{"status": "completed"}]
        self.run_count = 0

    async def run(self, task: str):
        result = self.results[min(self.run_count, len(self.results) - 1)]
        self.run_count += 1
        return result


class MockInterruptService:
    """Mock interrupt service for testing."""
    pass


class TestInteractiveRunnerBasic:
    """Basic tests for InteractiveRunner."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.CONFIRM
        manager.handle_response.return_value = (ResponseAction.APPROVED, None)
        return manager

    @pytest.fixture
    def runner(self, orchestrator, interrupt_service, mode_manager):
        runner = InteractiveRunner(
            orchestrator=orchestrator,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            confirm_exit=False,  # Disable exit confirmation for simpler tests
        )
        # Mock console to avoid terminal output
        runner._console = MagicMock()
        return runner

    @pytest.mark.asyncio
    async def test_run_completes_successfully(self, runner, orchestrator):
        """Test that run completes when orchestrator returns completed."""
        orchestrator.results = [{"status": "completed", "final_message": "Done"}]

        result = await runner.run("Fix the bug")

        assert result["status"] == "completed"
        assert orchestrator.run_count == 1

    @pytest.mark.asyncio
    async def test_run_returns_error(self, runner, orchestrator):
        """Test that run returns error status."""
        orchestrator.results = [{"status": "error", "error": "Something failed"}]

        result = await runner.run("Fix the bug")

        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_run_increments_step_count(self, runner, orchestrator):
        """Test that step count increments."""
        orchestrator.results = [{"status": "completed"}]

        await runner.run("Task")

        assert runner._step_count == 1
        runner._console.print.assert_called()  # Should print step rule


class TestConfirmationHandling:
    """Tests for confirmation interrupt handling."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.CONFIRM
        return manager

    @pytest.fixture
    def runner(self, orchestrator, interrupt_service, mode_manager):
        runner = InteractiveRunner(
            orchestrator=orchestrator,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            confirm_exit=False,
        )
        runner._console = MagicMock()
        return runner

    @pytest.mark.asyncio
    async def test_confirmation_approved_continues(self, runner, orchestrator, mode_manager):
        """Test that approved confirmation continues execution."""
        orchestrator.results = [
            {"status": "confirmation_required", "message": "Confirm rm?"},
            {"status": "completed"},
        ]
        mode_manager.handle_response.return_value = (ResponseAction.APPROVED, None)

        # Mock user input
        with patch.object(runner, '_prompt_user', new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = ""  # Empty = approve

            result = await runner.run("Delete files")

        assert result["status"] == "completed"
        assert orchestrator.run_count == 2

    @pytest.mark.asyncio
    async def test_confirmation_rejected_continues_with_feedback(self, runner, orchestrator, mode_manager):
        """Test that rejection continues with feedback."""
        orchestrator.results = [
            {"status": "confirmation_required", "message": "Confirm?"},
            {"status": "completed"},
        ]
        mode_manager.handle_response.return_value = (ResponseAction.REJECTED, "User said no")

        with patch.object(runner, '_prompt_user', new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = "no"

            result = await runner.run("Task")

        # Should continue after rejection feedback
        assert orchestrator.run_count == 2

    @pytest.mark.asyncio
    async def test_mode_switch_continues(self, runner, orchestrator, mode_manager):
        """Test that mode switch continues execution."""
        orchestrator.results = [
            {"status": "confirmation_required", "message": "Confirm?"},
            {"status": "completed"},
        ]
        mode_manager.handle_response.return_value = (ResponseAction.MODE_SWITCH, "Switched to yolo")

        with patch.object(runner, '_prompt_user', new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = "/y"

            result = await runner.run("Task")

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_help_shown_continues(self, runner, orchestrator, mode_manager):
        """Test that help shown continues execution."""
        orchestrator.results = [
            {"status": "confirmation_required", "message": "Confirm?"},
            {"status": "completed"},
        ]
        mode_manager.handle_response.return_value = (ResponseAction.HELP_SHOWN, "Help text here")

        with patch.object(runner, '_prompt_user', new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = "/h"

            result = await runner.run("Task")

        # Help should be printed
        runner._console.print.assert_any_call("Help text here")


class TestExitConfirmation:
    """Tests for exit confirmation handling."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.CONFIRM
        return manager

    @pytest.mark.asyncio
    async def test_exit_confirmed_with_empty_input(self, orchestrator, interrupt_service, mode_manager):
        """Test that empty input confirms exit."""
        orchestrator.results = [{"status": "completed"}]

        runner = InteractiveRunner(
            orchestrator=orchestrator,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            confirm_exit=True,
        )
        runner._console = MagicMock()

        with patch.object(runner, '_prompt_user', new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = ""  # Empty = confirm exit

            result = await runner.run("Task")

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_new_task_continues_execution(self, orchestrator, interrupt_service, mode_manager):
        """Test that providing new task continues execution."""
        orchestrator.results = [
            {"status": "completed"},  # First completion
            {"status": "completed"},  # Second completion
        ]

        runner = InteractiveRunner(
            orchestrator=orchestrator,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            confirm_exit=True,
        )
        runner._console = MagicMock()

        call_count = 0
        async def mock_prompt():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "New task"  # First call: provide new task
            return ""  # Second call: confirm exit

        with patch.object(runner, '_prompt_user', side_effect=mock_prompt):
            result = await runner.run("Initial task")

        assert orchestrator.run_count == 2


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    @pytest.fixture
    def orchestrator(self):
        return MockOrchestrator()

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.CONFIRM
        return manager

    @pytest.fixture
    def runner(self, orchestrator, interrupt_service, mode_manager):
        runner = InteractiveRunner(
            orchestrator=orchestrator,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            confirm_exit=False,
        )
        runner._console = MagicMock()
        return runner

    @pytest.mark.asyncio
    async def test_single_interrupt_continues(self, runner, orchestrator):
        """Test that single keyboard interrupt allows continue."""
        orchestrator.results = [{"status": "completed"}]

        call_count = 0
        async def mock_run(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt()
            return {"status": "completed"}

        orchestrator.run = mock_run

        with patch.object(runner, '_prompt_user', new_callable=AsyncMock) as mock_prompt:
            mock_prompt.return_value = "continue please"

            result = await runner.run("Task")

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_double_interrupt_aborts(self, runner, orchestrator):
        """Test that double keyboard interrupt aborts."""
        async def mock_run(task):
            raise KeyboardInterrupt()

        orchestrator.run = mock_run

        # First prompt raises KeyboardInterrupt (double interrupt)
        with patch.object(runner, '_prompt_user', new_callable=AsyncMock) as mock_prompt:
            mock_prompt.side_effect = KeyboardInterrupt()

            result = await runner.run("Task")

        assert result["status"] == "interrupted"


class TestCreateInteractiveRunner:
    """Tests for factory function."""

    def test_create_interactive_runner_default(self):
        """Test factory creates runner with defaults."""
        orchestrator = MockOrchestrator()

        runner = create_interactive_runner(orchestrator)

        assert isinstance(runner, InteractiveRunner)
        assert runner._orchestrator == orchestrator
        assert runner._confirm_exit is True

    def test_create_interactive_runner_yolo_mode(self):
        """Test factory creates runner in yolo mode."""
        orchestrator = MockOrchestrator()

        runner = create_interactive_runner(orchestrator, mode="yolo")

        assert runner._mode_manager.mode == ExecutionMode.YOLO

    def test_create_interactive_runner_no_confirm_exit(self):
        """Test factory creates runner without exit confirmation."""
        orchestrator = MockOrchestrator()

        runner = create_interactive_runner(orchestrator, confirm_exit=False)

        assert runner._confirm_exit is False


class TestExceptionHandling:
    """Tests for exception handling."""

    @pytest.fixture
    def interrupt_service(self):
        return MockInterruptService()

    @pytest.fixture
    def mode_manager(self):
        manager = MagicMock(spec=ModeManager)
        manager.mode = ExecutionMode.CONFIRM
        return manager

    @pytest.mark.asyncio
    async def test_orchestrator_exception_returns_error(self, interrupt_service, mode_manager):
        """Test that orchestrator exception returns error status."""
        orchestrator = MockOrchestrator()
        async def mock_run(task):
            raise ValueError("Orchestrator failed")
        orchestrator.run = mock_run

        runner = InteractiveRunner(
            orchestrator=orchestrator,
            interrupt_service=interrupt_service,
            mode_manager=mode_manager,
            confirm_exit=False,
        )
        runner._console = MagicMock()

        result = await runner.run("Task")

        assert result["status"] == "error"
        assert "Orchestrator failed" in result["error"]
