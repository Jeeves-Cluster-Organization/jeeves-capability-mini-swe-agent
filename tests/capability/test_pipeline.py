"""Tests for pipeline configuration."""

import sys
from pathlib import Path

import pytest

# Add jeeves-core to path
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))


class TestPipelineMode:
    """Tests for PipelineMode enum."""

    def test_pipeline_mode_values(self):
        """Test PipelineMode has expected values."""
        from minisweagent.capability.config.pipeline import PipelineMode

        assert PipelineMode.SINGLE_AGENT.value == "single_agent"
        assert PipelineMode.SEQUENTIAL.value == "sequential"
        assert PipelineMode.PARALLEL.value == "parallel"


class TestPipelineConfig:
    """Tests for pipeline configuration creation."""

    def test_create_single_agent_config(self):
        """Test creating single-agent pipeline config."""
        from minisweagent.capability.config.pipeline import (
            create_single_agent_config,
        )

        config = create_single_agent_config()

        assert config.name == "mini_swe_single_agent"
        assert len(config.agents) == 1
        assert config.agents[0].name == "swe_agent"
        assert config.agents[0].has_tools is True

    def test_create_sequential_pipeline_config(self):
        """Test creating sequential pipeline config."""
        from minisweagent.capability.config.pipeline import (
            create_swe_pipeline_config,
            PipelineMode,
        )

        config = create_swe_pipeline_config(mode=PipelineMode.SEQUENTIAL)

        assert "sequential" in config.name
        agent_names = [a.name for a in config.agents]

        # Should have all main agents
        assert "task_parser" in agent_names
        assert "code_searcher" in agent_names
        assert "file_analyzer" in agent_names
        assert "planner" in agent_names
        assert "executor" in agent_names
        assert "verifier" in agent_names

        # Should NOT have test_discovery (parallel only)
        assert "test_discovery" not in agent_names

    def test_create_parallel_pipeline_config(self):
        """Test creating parallel pipeline config."""
        from minisweagent.capability.config.pipeline import (
            create_swe_pipeline_config,
            PipelineMode,
        )

        config = create_swe_pipeline_config(mode=PipelineMode.PARALLEL)

        assert "parallel" in config.name
        agent_names = [a.name for a in config.agents]

        # Should have test_discovery in parallel mode
        assert "test_discovery" in agent_names

        # Check parallel dependencies
        code_searcher = next(a for a in config.agents if a.name == "code_searcher")
        file_analyzer = next(a for a in config.agents if a.name == "file_analyzer")
        test_discovery = next(a for a in config.agents if a.name == "test_discovery")

        # All should depend on task_parser (can run in parallel)
        assert "task_parser" in (code_searcher.requires or [])
        assert "task_parser" in (file_analyzer.requires or [])
        assert "task_parser" in (test_discovery.requires or [])

        # Planner should wait for all three (fan-in)
        planner = next(a for a in config.agents if a.name == "planner")
        assert "code_searcher" in (planner.requires or [])
        assert "file_analyzer" in (planner.requires or [])
        assert "test_discovery" in (planner.requires or [])
        assert planner.join_strategy == "all"

    def test_pipeline_config_has_routing_rules(self):
        """Test that verifier has routing rules for loop-back."""
        from minisweagent.capability.config.pipeline import (
            create_swe_pipeline_config,
            PipelineMode,
        )

        config = create_swe_pipeline_config(mode=PipelineMode.PARALLEL)

        verifier = next(a for a in config.agents if a.name == "verifier")
        assert verifier.routing_rules is not None
        assert len(verifier.routing_rules) > 0

        # Should have loop_back rule
        rule_targets = [r.target for r in verifier.routing_rules]
        assert "executor" in rule_targets  # loop_back to executor
        assert "end" in rule_targets  # success -> end

    def test_pipeline_config_respects_limits(self):
        """Test that custom limits are applied."""
        from minisweagent.capability.config.pipeline import (
            create_swe_pipeline_config,
            PipelineMode,
        )

        config = create_swe_pipeline_config(
            mode=PipelineMode.PARALLEL,
            max_iterations=10,
            max_llm_calls=20,
            max_agent_hops=30,
        )

        assert config.max_iterations == 10
        assert config.max_llm_calls == 20
        assert config.max_agent_hops == 30


class TestAgentConfigs:
    """Tests for individual agent configurations."""

    def test_executor_has_write_tools(self):
        """Test that executor has tools for modifying files."""
        from minisweagent.capability.config.pipeline import (
            create_swe_pipeline_config,
            PipelineMode,
        )

        config = create_swe_pipeline_config(mode=PipelineMode.PARALLEL)
        executor = next(a for a in config.agents if a.name == "executor")

        assert "bash_execute" in executor.allowed_tools
        assert "write_file" in executor.allowed_tools
        assert "edit_file" in executor.allowed_tools

    def test_verifier_has_test_tools(self):
        """Test that verifier can run tests."""
        from minisweagent.capability.config.pipeline import (
            create_swe_pipeline_config,
            PipelineMode,
        )

        config = create_swe_pipeline_config(mode=PipelineMode.PARALLEL)
        verifier = next(a for a in config.agents if a.name == "verifier")

        assert "bash_execute" in verifier.allowed_tools
        assert "run_tests" in verifier.allowed_tools

    def test_planner_has_no_tools(self):
        """Test that planner doesn't execute tools directly."""
        from minisweagent.capability.config.pipeline import (
            create_swe_pipeline_config,
            PipelineMode,
        )

        config = create_swe_pipeline_config(mode=PipelineMode.PARALLEL)
        planner = next(a for a in config.agents if a.name == "planner")

        assert planner.has_tools is False
