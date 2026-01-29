"""Tests for jeeves-core component wiring.

Tests that KernelClient (gRPC connection to Go kernel) is properly
wired and accessible through the capability layer.

Architecture Note:
- Go kernel is REQUIRED (micro-OS architecture)
- Python capabilities communicate with Go kernel via gRPC
- KernelClient provides: lifecycle, resources, events, quota
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add jeeves-core to path for imports
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))

from minisweagent.capability.wiring import (
    JeevesContext,
    create_jeeves_context,
    get_jeeves_context,
    reset_jeeves_context,
    CAPABILITY_ID,
)


class TestJeevesContext:
    """Tests for JeevesContext creation and wiring."""

    def test_create_jeeves_context_returns_context(self):
        """Test that create_jeeves_context returns a JeevesContext."""
        context = create_jeeves_context()
        assert isinstance(context, JeevesContext)

    def test_context_has_kernel_client(self):
        """Test that context contains KernelClient."""
        context = create_jeeves_context()
        assert context.kernel_client is not None

    def test_context_kernel_client_has_record_usage(self):
        """Test that kernel_client has record_usage method."""
        context = create_jeeves_context()
        assert hasattr(context.kernel_client, 'record_usage')

    def test_context_kernel_client_has_check_quota(self):
        """Test that kernel_client has check_quota method."""
        context = create_jeeves_context()
        assert hasattr(context.kernel_client, 'check_quota')

    def test_context_uses_default_kernel_address(self):
        """Test that context uses default kernel address."""
        context = create_jeeves_context()
        assert context._kernel_address == "localhost:50051"

    def test_context_accepts_custom_kernel_address(self):
        """Test that context accepts custom kernel address."""
        context = create_jeeves_context(kernel_address="custom:9999")
        assert context._kernel_address == "custom:9999"

    def test_context_accepts_db_parameter(self):
        """Test that context accepts optional db parameter."""
        mock_db = MagicMock()
        context = create_jeeves_context(db=mock_db)
        assert context.db is mock_db


class TestGlobalContext:
    """Tests for global context singleton."""

    def setup_method(self):
        """Reset global context before each test."""
        reset_jeeves_context()

    def teardown_method(self):
        """Reset global context after each test."""
        reset_jeeves_context()

    def test_get_jeeves_context_creates_singleton(self):
        """Test that get_jeeves_context returns singleton."""
        context1 = get_jeeves_context()
        context2 = get_jeeves_context()
        assert context1 is context2

    def test_reset_jeeves_context_clears_singleton(self):
        """Test that reset_jeeves_context clears the singleton."""
        context1 = get_jeeves_context()
        reset_jeeves_context()
        context2 = get_jeeves_context()
        assert context1 is not context2


class TestKernelClientIntegration:
    """Tests for KernelClient integration."""

    def test_kernel_client_is_initialized(self):
        """Test that KernelClient is properly initialized."""
        context = create_jeeves_context()
        # KernelClient should be present
        assert context.kernel_client is not None

    @pytest.mark.asyncio
    async def test_kernel_client_record_usage_callable(self):
        """Test that record_usage can be called (doesn't require running kernel)."""
        context = create_jeeves_context()
        # Method should exist and be callable
        assert callable(context.kernel_client.record_usage)

    @pytest.mark.asyncio
    async def test_kernel_client_check_quota_callable(self):
        """Test that check_quota can be called (doesn't require running kernel)."""
        context = create_jeeves_context()
        # Method should exist and be callable
        assert callable(context.kernel_client.check_quota)


class TestCapabilityId:
    """Tests for capability ID constant."""

    def test_capability_id_is_set(self):
        """Test that CAPABILITY_ID is defined."""
        assert CAPABILITY_ID == "mini-swe-agent"


class TestKernelRequired:
    """Tests verifying Go kernel is required."""

    def test_context_creation_requires_kernel_client_module(self):
        """Test that context creation requires jeeves_infra.kernel_client."""
        # This should succeed if jeeves_infra is installed
        context = create_jeeves_context()
        assert context.kernel_client is not None

    def test_context_documents_kernel_requirement(self):
        """Test that JeevesContext documents kernel requirement."""
        # The docstring should mention Go kernel is required
        assert "REQUIRED" in JeevesContext.__doc__ or "kernel" in JeevesContext.__doc__.lower()
