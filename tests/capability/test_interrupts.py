"""Tests for interrupt handling system."""

import pytest


class TestConfirmationHandler:
    """Tests for ConfirmationHandler."""

    def test_yolo_mode_never_confirms(self):
        """Test that yolo mode never requires confirmation."""
        from minisweagent.capability.interrupts import ConfirmationHandler

        handler = ConfirmationHandler(mode="yolo")
        assert not handler.should_confirm("bash_execute", {"command": "rm -rf /"})

    def test_confirm_mode_requires_high_risk(self):
        """Test that confirm mode requires confirmation for HIGH-risk tools."""
        from minisweagent.capability.interrupts import ConfirmationHandler

        handler = ConfirmationHandler(mode="confirm")
        assert handler.should_confirm("bash_execute", {"command": "echo hi"})
        assert handler.should_confirm("write_file", {"path": "/tmp/test"})
        assert handler.should_confirm("edit_file", {"path": "/tmp/test"})

    def test_confirm_mode_skips_safe_tools(self):
        """Test that confirm mode skips safe tools."""
        from minisweagent.capability.interrupts import ConfirmationHandler

        handler = ConfirmationHandler(mode="confirm")
        assert not handler.should_confirm("read_file", {"path": "/tmp/test"})
        assert not handler.should_confirm("find_files", {"pattern": "*.py"})
        assert not handler.should_confirm("grep_search", {"pattern": "test"})

    def test_whitelist_patterns_skip_confirmation(self):
        """Test that whitelisted patterns skip confirmation."""
        from minisweagent.capability.interrupts import ConfirmationHandler

        handler = ConfirmationHandler(
            mode="confirm",
            whitelist_patterns=[r"^echo\s+", r"^ls\s*$"],
        )
        assert not handler.should_confirm("bash_execute", {"command": "echo hello"})
        assert not handler.should_confirm("bash_execute", {"command": "ls"})
        assert handler.should_confirm("bash_execute", {"command": "rm -rf /"})


class TestModeManager:
    """Tests for ModeManager."""

    def test_initial_mode_is_confirm(self):
        """Test that initial mode is confirm."""
        from minisweagent.capability.interrupts import ModeManager, ExecutionMode

        manager = ModeManager()
        assert manager.mode == ExecutionMode.CONFIRM

    def test_mode_switch_commands(self):
        """Test mode switching via commands."""
        from minisweagent.capability.interrupts import ModeManager, ExecutionMode
        from minisweagent.capability.interrupts.mode_manager import ResponseAction
        from jeeves_infra.protocols import InterruptResponse

        manager = ModeManager(initial_mode=ExecutionMode.CONFIRM)

        # Switch to yolo
        action, _ = manager.handle_response(InterruptResponse(text="/y"))
        assert action == ResponseAction.MODE_SWITCH
        assert manager.mode == ExecutionMode.YOLO

        # Switch to human
        action, _ = manager.handle_response(InterruptResponse(text="/u"))
        assert action == ResponseAction.MODE_SWITCH
        assert manager.mode == ExecutionMode.HUMAN

        # Switch to confirm
        action, _ = manager.handle_response(InterruptResponse(text="/c"))
        assert action == ResponseAction.MODE_SWITCH
        assert manager.mode == ExecutionMode.CONFIRM

    def test_empty_response_approves(self):
        """Test that empty response approves action."""
        from minisweagent.capability.interrupts import ModeManager
        from minisweagent.capability.interrupts.mode_manager import ResponseAction
        from jeeves_infra.protocols import InterruptResponse

        manager = ModeManager()
        action, _ = manager.handle_response(InterruptResponse(text=""))
        assert action == ResponseAction.APPROVED

    def test_text_response_rejects(self):
        """Test that text response rejects action."""
        from minisweagent.capability.interrupts import ModeManager
        from minisweagent.capability.interrupts.mode_manager import ResponseAction
        from jeeves_infra.protocols import InterruptResponse

        manager = ModeManager()
        action, message = manager.handle_response(InterruptResponse(text="no thanks"))
        assert action == ResponseAction.REJECTED
        assert message == "no thanks"

    def test_help_command_shows_help(self):
        """Test that /h shows help."""
        from minisweagent.capability.interrupts import ModeManager
        from minisweagent.capability.interrupts.mode_manager import ResponseAction
        from jeeves_infra.protocols import InterruptResponse

        manager = ModeManager()
        action, message = manager.handle_response(InterruptResponse(text="/h"))
        assert action == ResponseAction.HELP_SHOWN
        assert "Available commands" in message


class TestCLIInterruptService:
    """Tests for CLIInterruptService."""

    @pytest.mark.asyncio
    async def test_create_interrupt(self):
        """Test creating an interrupt."""
        from minisweagent.capability.interrupts import CLIInterruptService
        from jeeves_infra.protocols import InterruptKind, InterruptStatus

        service = CLIInterruptService()
        interrupt = await service.create_interrupt(
            kind=InterruptKind.CONFIRMATION,
            request_id="test-request",
            user_id="test-user",
            session_id="test-session",
            message="Test confirmation",
        )

        assert interrupt.kind == InterruptKind.CONFIRMATION
        assert interrupt.status == InterruptStatus.PENDING
        assert interrupt.message == "Test confirmation"

    @pytest.mark.asyncio
    async def test_respond_to_interrupt(self):
        """Test responding to an interrupt."""
        from minisweagent.capability.interrupts import CLIInterruptService
        from jeeves_infra.protocols import InterruptKind, InterruptResponse, InterruptStatus

        service = CLIInterruptService()
        interrupt = await service.create_interrupt(
            kind=InterruptKind.CONFIRMATION,
            request_id="test-request",
            user_id="test-user",
            session_id="test-session",
        )

        response = InterruptResponse(approved=True)
        updated = await service.respond(interrupt.id, response)

        assert updated.status == InterruptStatus.RESOLVED
        assert updated.response.approved is True

    @pytest.mark.asyncio
    async def test_list_pending_interrupts(self):
        """Test listing pending interrupts."""
        from minisweagent.capability.interrupts import CLIInterruptService
        from jeeves_infra.protocols import InterruptKind

        service = CLIInterruptService()

        await service.create_interrupt(
            kind=InterruptKind.CONFIRMATION,
            request_id="req-1",
            user_id="user-1",
            session_id="session-1",
        )
        await service.create_interrupt(
            kind=InterruptKind.CONFIRMATION,
            request_id="req-2",
            user_id="user-1",
            session_id="session-1",
        )

        pending = await service.list_pending(user_id="user-1")
        assert len(pending) == 2
