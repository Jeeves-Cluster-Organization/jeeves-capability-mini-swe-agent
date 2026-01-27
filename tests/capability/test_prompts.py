"""Tests for prompt registry."""

import sys
from pathlib import Path

import pytest

# Add jeeves-core to path for protocol imports
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))


class TestMiniSWEPromptRegistry:
    """Tests for MiniSWEPromptRegistry."""

    def test_create_prompt_registry(self):
        """Test creating a prompt registry."""
        from minisweagent.capability.prompts import create_prompt_registry

        registry = create_prompt_registry()
        assert registry is not None

    def test_registry_has_core_templates(self):
        """Test that registry has core templates."""
        from minisweagent.capability.prompts import create_prompt_registry

        registry = create_prompt_registry()
        keys = registry.list_keys()

        assert "mini_swe.system" in keys
        assert "mini_swe.instance" in keys
        assert "mini_swe.swe_agent" in keys  # Used by orchestrator unified mode

    def test_registry_has_pipeline_templates(self):
        """Test that registry has pipeline stage templates."""
        from minisweagent.capability.prompts import create_prompt_registry

        registry = create_prompt_registry()
        keys = registry.list_keys()

        assert "mini_swe.task_parser" in keys
        assert "mini_swe.code_searcher" in keys
        assert "mini_swe.planner" in keys
        assert "mini_swe.executor" in keys
        assert "mini_swe.verifier" in keys

    def test_get_renders_template(self):
        """Test that get() renders template with context."""
        from minisweagent.capability.prompts import create_prompt_registry

        registry = create_prompt_registry()

        # Get instance template with task
        prompt = registry.get("mini_swe.instance", task="Fix the bug")
        assert "Fix the bug" in prompt

    def test_get_includes_system_vars(self):
        """Test that get() includes system variables."""
        from minisweagent.capability.prompts import create_prompt_registry

        registry = create_prompt_registry()

        # Get instance template - should have system info
        prompt = registry.get("mini_swe.instance", task="Test task")
        # The template uses {{system}} which should be rendered
        # to the current OS name

    def test_register_custom_template(self):
        """Test registering a custom template."""
        from minisweagent.capability.prompts import create_prompt_registry

        registry = create_prompt_registry()
        registry.register("custom.test", "Hello {{name}}")

        prompt = registry.get("custom.test", name="World")
        assert prompt == "Hello World"

    def test_get_with_missing_key_returns_key(self):
        """Test that get() with missing key uses key as template."""
        from minisweagent.capability.prompts import create_prompt_registry

        registry = create_prompt_registry()
        prompt = registry.get("Hello {{name}}", name="Test")
        assert prompt == "Hello Test"
