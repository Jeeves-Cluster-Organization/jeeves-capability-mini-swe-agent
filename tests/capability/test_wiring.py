"""Tests for capability registration (wiring.py)."""

import sys
from pathlib import Path

import pytest

# Add jeeves-core to path for protocol imports
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))


class TestCapabilityRegistration:
    """Tests for capability registration with jeeves-core."""

    def test_register_capability_creates_registry_entries(self):
        """Test that register_capability populates the registry."""
        from protocols.capability import (
            get_capability_resource_registry,
            reset_capability_resource_registry,
        )
        from minisweagent.capability.wiring import register_capability, CAPABILITY_ID

        # Reset registry for clean test
        reset_capability_resource_registry()

        # Register capability
        register_capability()

        # Check registry
        registry = get_capability_resource_registry()

        # Mode should be registered
        assert registry.is_mode_registered(CAPABILITY_ID)
        mode_config = registry.get_mode_config(CAPABILITY_ID)
        assert mode_config is not None
        assert mode_config.requires_repo_path is True

        # Service should be registered
        services = registry.get_services(CAPABILITY_ID)
        assert len(services) == 1
        assert services[0].service_id == f"{CAPABILITY_ID}_service"
        assert services[0].is_readonly is False  # SWE agent modifies code

        # Agents should be registered
        agents = registry.get_agents(CAPABILITY_ID)
        assert len(agents) > 0
        agent_names = [a.name for a in agents]
        assert "executor" in agent_names
        assert "verifier" in agent_names

        # Tools should be registered
        tools_config = registry.get_tools(CAPABILITY_ID)
        assert tools_config is not None
        assert "bash_execute" in tools_config.tool_ids

    def test_get_agent_config_returns_correct_config(self):
        """Test that get_agent_config returns the correct LLM config."""
        from minisweagent.capability.wiring import get_agent_config

        config = get_agent_config("planner")
        assert config.agent_name == "planner"
        assert config.temperature == 0.3
        assert config.max_tokens == 4000

    def test_get_agent_config_raises_for_unknown_agent(self):
        """Test that get_agent_config raises KeyError for unknown agent."""
        from minisweagent.capability.wiring import get_agent_config

        with pytest.raises(KeyError):
            get_agent_config("unknown_agent")


class TestAgentLLMConfigs:
    """Tests for agent LLM configurations."""

    def test_all_agents_have_configs(self):
        """Test that all registered agents have LLM configs."""
        from minisweagent.capability.wiring import AGENT_LLM_CONFIGS, AGENT_DEFINITIONS

        agent_names_with_configs = set(AGENT_LLM_CONFIGS.keys())
        agent_names_defined = {a.name for a in AGENT_DEFINITIONS}

        # All defined agents should have LLM configs (or inherit from another)
        for agent_name in agent_names_defined:
            assert (
                agent_name in agent_names_with_configs
                or agent_name == "test_discovery"  # Reuses file_analyzer config
            ), f"Agent {agent_name} missing LLM config"

    def test_config_values_are_reasonable(self):
        """Test that config values are within reasonable ranges."""
        from minisweagent.capability.wiring import AGENT_LLM_CONFIGS

        for agent_name, config in AGENT_LLM_CONFIGS.items():
            assert 0 <= config.temperature <= 1.0, f"{agent_name}: temperature out of range"
            assert config.max_tokens > 0, f"{agent_name}: max_tokens must be positive"
            assert config.context_window > 0, f"{agent_name}: context_window must be positive"
            assert config.timeout_seconds > 0, f"{agent_name}: timeout must be positive"
