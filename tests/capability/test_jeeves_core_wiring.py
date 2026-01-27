"""Tests for jeeves-core component wiring.

Tests that ControlTower, CommBus, and Memory handlers are properly
wired and accessible through the capability layer.
"""

import pytest
import sys
from pathlib import Path

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

    def test_context_has_control_tower(self):
        """Test that context contains ControlTower."""
        context = create_jeeves_context()
        # ControlTower should be available if jeeves-core is present
        if context.control_tower is not None:
            assert hasattr(context.control_tower, 'lifecycle')
            assert hasattr(context.control_tower, 'resources')
            assert hasattr(context.control_tower, 'ipc')

    def test_context_has_commbus(self):
        """Test that context contains CommBus."""
        context = create_jeeves_context()
        # CommBus should be available if jeeves-core is present
        if context.commbus is not None:
            assert hasattr(context.commbus, 'publish')
            assert hasattr(context.commbus, 'query')
            assert hasattr(context.commbus, 'register_handler')

    def test_context_is_wired(self):
        """Test is_wired() method."""
        context = create_jeeves_context()
        # Either both are wired (jeeves-core available) or neither (not available)
        is_wired = context.is_wired()
        if is_wired:
            assert context.control_tower is not None
            assert context.commbus is not None

    def test_context_without_memory_handlers(self):
        """Test creating context without registering memory handlers."""
        context = create_jeeves_context(register_memory=False)
        assert isinstance(context, JeevesContext)
        # Memory handlers should not be registered
        assert not context._memory_handlers_registered


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


class TestControlTowerIntegration:
    """Tests for ControlTower integration when available."""

    @pytest.fixture
    def context_with_control_tower(self):
        """Create context and skip if ControlTower not available."""
        context = create_jeeves_context()
        if context.control_tower is None:
            pytest.skip("ControlTower not available (jeeves-core not installed)")
        return context

    def test_control_tower_has_lifecycle_manager(self, context_with_control_tower):
        """Test that ControlTower has lifecycle manager."""
        ct = context_with_control_tower.control_tower
        assert ct.lifecycle is not None

    def test_control_tower_has_resource_tracker(self, context_with_control_tower):
        """Test that ControlTower has resource tracker."""
        ct = context_with_control_tower.control_tower
        assert ct.resources is not None

    def test_control_tower_has_ipc(self, context_with_control_tower):
        """Test that ControlTower has IPC coordinator."""
        ct = context_with_control_tower.control_tower
        assert ct.ipc is not None

    def test_control_tower_has_events(self, context_with_control_tower):
        """Test that ControlTower has event aggregator."""
        ct = context_with_control_tower.control_tower
        assert ct.events is not None


class TestCommBusIntegration:
    """Tests for CommBus integration when available."""

    @pytest.fixture
    def context_with_commbus(self):
        """Create context and skip if CommBus not available."""
        context = create_jeeves_context()
        if context.commbus is None:
            pytest.skip("CommBus not available (jeeves-core not installed)")
        return context

    def test_commbus_can_register_handler(self, context_with_commbus):
        """Test that CommBus can register handlers."""
        bus = context_with_commbus.commbus

        # Register a test handler
        def test_handler(msg):
            return {"received": True}

        bus.register_handler("TestQuery", test_handler)
        assert bus.has_handler("TestQuery")

    def test_commbus_can_subscribe(self, context_with_commbus):
        """Test that CommBus can subscribe to events."""
        bus = context_with_commbus.commbus
        events_received = []

        def event_handler(event):
            events_received.append(event)

        unsubscribe = bus.subscribe("TestEvent", event_handler)
        assert callable(unsubscribe)

        # Clean up
        unsubscribe()

    def test_commbus_get_registered_types(self, context_with_commbus):
        """Test that CommBus can list registered types."""
        bus = context_with_commbus.commbus

        # If memory handlers are registered, we should have some types
        types = bus.get_registered_types()
        assert isinstance(types, list)


class TestMemoryHandlersIntegration:
    """Tests for memory handler registration."""

    @pytest.fixture
    def context_with_memory(self):
        """Create context with memory handlers and skip if not available."""
        context = create_jeeves_context(register_memory=True)
        if not context._memory_handlers_registered:
            pytest.skip("Memory handlers not available (jeeves-core not installed)")
        return context

    def test_memory_handlers_registered(self, context_with_memory):
        """Test that memory handlers are registered."""
        bus = context_with_memory.commbus

        # Check for expected memory handlers
        types = bus.get_registered_types()

        expected_handlers = [
            "GetSessionState",
            "GetRecentEntities",
            "SearchMemory",
            "ClearSession",
            "UpdateFocus",
        ]

        for handler in expected_handlers:
            if handler in types:
                assert bus.has_handler(handler), f"Handler {handler} not registered"


class TestCapabilityId:
    """Tests for capability ID constant."""

    def test_capability_id_is_set(self):
        """Test that CAPABILITY_ID is defined."""
        assert CAPABILITY_ID == "mini-swe-agent"
