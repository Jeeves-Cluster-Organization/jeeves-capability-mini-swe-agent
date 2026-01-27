"""Tests for ClarificationHandler."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from minisweagent.capability.interrupts.clarification_handler import (
    ClarificationHandler,
    ClarificationRequest,
)


class TestClarificationRequest:
    """Tests for ClarificationRequest dataclass."""

    def test_create_request(self):
        """Test creating a clarification request."""
        request = ClarificationRequest(
            request_id="req-123",
            question="Which file should I modify?",
            options=["file1.py", "file2.py", "file3.py"],
            context={"task": "fix bug"},
        )

        assert request.request_id == "req-123"
        assert request.question == "Which file should I modify?"
        assert len(request.options) == 3
        assert request.context["task"] == "fix bug"
        assert request.selected is None

    def test_create_request_with_selection(self):
        """Test creating a request with pre-selected option."""
        request = ClarificationRequest(
            request_id="req-456",
            question="Continue?",
            options=["yes", "no"],
            context={},
            selected="yes",
        )

        assert request.selected == "yes"


class TestClarificationHandlerInit:
    """Tests for ClarificationHandler initialization."""

    def test_init_without_cli_service(self):
        """Test initialization without CLI service."""
        handler = ClarificationHandler()

        assert handler.cli_service is None
        assert handler.pending_requests == {}

    def test_init_with_cli_service(self):
        """Test initialization with CLI service."""
        cli_service = MagicMock()
        handler = ClarificationHandler(cli_service=cli_service)

        assert handler.cli_service == cli_service


class TestRequestClarification:
    """Tests for request_clarification method."""

    @pytest.mark.asyncio
    async def test_request_with_cli_service(self):
        """Test requesting clarification with CLI service."""
        cli_service = MagicMock()
        cli_service.prompt_choice = AsyncMock(return_value="option1")

        handler = ClarificationHandler(cli_service=cli_service)

        result = await handler.request_clarification(
            question="Choose an option:",
            options=["option1", "option2"],
            context={"source": "test"},
        )

        assert result == "option1"
        cli_service.prompt_choice.assert_called_once_with(
            "Choose an option:",
            ["option1", "option2"]
        )

    @pytest.mark.asyncio
    async def test_request_stores_pending(self):
        """Test that request is stored as pending."""
        cli_service = MagicMock()
        cli_service.prompt_choice = AsyncMock(return_value="yes")

        handler = ClarificationHandler(cli_service=cli_service)

        await handler.request_clarification(
            question="Continue?",
            options=["yes", "no"],
        )

        assert len(handler.pending_requests) == 1
        request = list(handler.pending_requests.values())[0]
        assert request.question == "Continue?"
        assert request.selected == "yes"

    @pytest.mark.asyncio
    async def test_request_without_cli_service_raises(self):
        """Test that request without CLI service raises NotImplementedError."""
        handler = ClarificationHandler()

        with pytest.raises(NotImplementedError, match="Webhook-based clarification"):
            await handler.request_clarification(
                question="Choose:",
                options=["a", "b"],
            )

    @pytest.mark.asyncio
    async def test_request_with_default_context(self):
        """Test that context defaults to empty dict."""
        cli_service = MagicMock()
        cli_service.prompt_choice = AsyncMock(return_value="ok")

        handler = ClarificationHandler(cli_service=cli_service)

        await handler.request_clarification(
            question="OK?",
            options=["ok", "cancel"],
            # No context provided
        )

        request = list(handler.pending_requests.values())[0]
        assert request.context == {}

    @pytest.mark.asyncio
    async def test_multiple_requests_get_unique_ids(self):
        """Test that multiple requests get unique IDs."""
        cli_service = MagicMock()
        cli_service.prompt_choice = AsyncMock(side_effect=["a", "b", "c"])

        handler = ClarificationHandler(cli_service=cli_service)

        await handler.request_clarification("Q1?", ["a", "b"])
        await handler.request_clarification("Q2?", ["a", "b"])
        await handler.request_clarification("Q3?", ["a", "b"])

        request_ids = list(handler.pending_requests.keys())
        assert len(set(request_ids)) == 3  # All unique


class TestRespondToRequest:
    """Tests for respond_to_request method."""

    @pytest.mark.asyncio
    async def test_respond_to_valid_request(self):
        """Test responding to a valid request."""
        handler = ClarificationHandler()

        # Manually add a pending request
        request = ClarificationRequest(
            request_id="req-123",
            question="Choose?",
            options=["a", "b", "c"],
            context={},
        )
        handler.pending_requests["req-123"] = request

        await handler.respond_to_request("req-123", "b")

        assert request.selected == "b"

    @pytest.mark.asyncio
    async def test_respond_to_unknown_request_raises(self):
        """Test that responding to unknown request raises ValueError."""
        handler = ClarificationHandler()

        with pytest.raises(ValueError, match="Unknown request"):
            await handler.respond_to_request("nonexistent", "option")

    @pytest.mark.asyncio
    async def test_respond_with_invalid_option_raises(self):
        """Test that responding with invalid option raises ValueError."""
        handler = ClarificationHandler()

        request = ClarificationRequest(
            request_id="req-123",
            question="Choose?",
            options=["a", "b"],
            context={},
        )
        handler.pending_requests["req-123"] = request

        with pytest.raises(ValueError, match="Invalid option"):
            await handler.respond_to_request("req-123", "c")


class TestGetPendingRequests:
    """Tests for get_pending_requests method."""

    def test_get_pending_returns_unselected(self):
        """Test that only unselected requests are returned."""
        handler = ClarificationHandler()

        # Add some requests
        handler.pending_requests["req-1"] = ClarificationRequest(
            request_id="req-1",
            question="Q1?",
            options=["a", "b"],
            context={},
            selected=None,  # Pending
        )
        handler.pending_requests["req-2"] = ClarificationRequest(
            request_id="req-2",
            question="Q2?",
            options=["x", "y"],
            context={},
            selected="x",  # Already answered
        )
        handler.pending_requests["req-3"] = ClarificationRequest(
            request_id="req-3",
            question="Q3?",
            options=["1", "2"],
            context={},
            selected=None,  # Pending
        )

        pending = handler.get_pending_requests()

        assert len(pending) == 2
        request_ids = [r.request_id for r in pending]
        assert "req-1" in request_ids
        assert "req-3" in request_ids
        assert "req-2" not in request_ids

    def test_get_pending_returns_empty_when_none_pending(self):
        """Test returns empty list when no pending requests."""
        handler = ClarificationHandler()

        pending = handler.get_pending_requests()

        assert pending == []

    def test_get_pending_after_all_answered(self):
        """Test returns empty after all requests answered."""
        handler = ClarificationHandler()

        handler.pending_requests["req-1"] = ClarificationRequest(
            request_id="req-1",
            question="Q?",
            options=["yes", "no"],
            context={},
            selected="yes",
        )

        pending = handler.get_pending_requests()

        assert pending == []
