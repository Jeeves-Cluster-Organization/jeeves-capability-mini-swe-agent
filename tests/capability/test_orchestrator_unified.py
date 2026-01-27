"""Tests for unified orchestrator (single-stage pipeline)."""

import sys
from pathlib import Path

import pytest

# Add jeeves-core to path for protocol imports
_jeeves_core_path = Path(__file__).parent.parent.parent / "jeeves-core"
if _jeeves_core_path.exists() and str(_jeeves_core_path) not in sys.path:
    sys.path.insert(0, str(_jeeves_core_path))


class TestSWEOrchestratorConfig:
    """Tests for orchestrator configuration."""

    def test_default_config_uses_unified_pipeline(self):
        """Test that default config uses unified pipeline mode."""
        from minisweagent.capability.orchestrator import SWEOrchestratorConfig

        config = SWEOrchestratorConfig()
        assert config.pipeline_mode == "unified"

    def test_config_accepts_parallel_mode(self):
        """Test that config accepts parallel pipeline mode."""
        from minisweagent.capability.orchestrator import SWEOrchestratorConfig

        config = SWEOrchestratorConfig(pipeline_mode="parallel")
        assert config.pipeline_mode == "parallel"


class TestOrchestratorMode:
    """Tests for orchestrator mode constants."""

    def test_orchestrator_mode_has_pipeline_constant(self):
        """Test that OrchestratorMode has PIPELINE constant."""
        from minisweagent.capability.orchestrator import OrchestratorMode

        assert OrchestratorMode.PIPELINE == "pipeline"


class TestCreateSWEOrchestrator:
    """Tests for orchestrator factory function."""

    def test_create_orchestrator_returns_swe_orchestrator(self):
        """Test that create_swe_orchestrator returns SWEOrchestrator."""
        from minisweagent.capability.orchestrator import (
            SWEOrchestrator,
            create_swe_orchestrator,
        )

        orchestrator = create_swe_orchestrator()
        assert isinstance(orchestrator, SWEOrchestrator)

    def test_create_orchestrator_with_unified_mode(self):
        """Test creating orchestrator with unified pipeline mode."""
        from minisweagent.capability.orchestrator import create_swe_orchestrator

        orchestrator = create_swe_orchestrator(pipeline_mode="unified")
        assert orchestrator.config.pipeline_mode == "unified"

    def test_create_orchestrator_with_parallel_mode(self):
        """Test creating orchestrator with parallel pipeline mode."""
        from minisweagent.capability.orchestrator import create_swe_orchestrator

        orchestrator = create_swe_orchestrator(pipeline_mode="parallel")
        assert orchestrator.config.pipeline_mode == "parallel"


class TestUnifiedPipelineConfig:
    """Tests for unified (single-stage) pipeline configuration."""

    def test_unified_config_has_single_agent(self):
        """Test that unified pipeline has single swe_agent."""
        from minisweagent.capability.orchestrator import (
            SWEOrchestrator,
            SWEOrchestratorConfig,
        )

        orchestrator = SWEOrchestrator(
            config=SWEOrchestratorConfig(pipeline_mode="unified"),
        )
        config = orchestrator._create_unified_pipeline_config()

        assert len(config.agents) == 1
        assert config.agents[0].name == "swe_agent"

    def test_unified_agent_has_self_routing(self):
        """Test that unified agent loops back to itself."""
        from minisweagent.capability.orchestrator import (
            SWEOrchestrator,
            SWEOrchestratorConfig,
        )

        orchestrator = SWEOrchestrator(
            config=SWEOrchestratorConfig(pipeline_mode="unified"),
        )
        config = orchestrator._create_unified_pipeline_config()
        agent = config.agents[0]

        assert agent.default_next == "swe_agent"

    def test_unified_agent_has_completion_routing(self):
        """Test that unified agent routes to end on completion."""
        from minisweagent.capability.orchestrator import (
            SWEOrchestrator,
            SWEOrchestratorConfig,
        )

        orchestrator = SWEOrchestrator(
            config=SWEOrchestratorConfig(pipeline_mode="unified"),
        )
        config = orchestrator._create_unified_pipeline_config()
        agent = config.agents[0]

        # Check routing rules
        completion_rule = next(
            (r for r in agent.routing_rules if r.condition == "completed"),
            None,
        )
        assert completion_rule is not None
        assert completion_rule.value is True
        assert completion_rule.target == "end"

    def test_unified_agent_has_all_tools(self):
        """Test that unified agent has access to all tools."""
        from minisweagent.capability.orchestrator import (
            SWEOrchestrator,
            SWEOrchestratorConfig,
        )

        orchestrator = SWEOrchestrator(
            config=SWEOrchestratorConfig(pipeline_mode="unified"),
        )
        config = orchestrator._create_unified_pipeline_config()
        agent = config.agents[0]

        expected_tools = {
            "bash_execute", "read_file", "write_file",
            "edit_file", "find_files", "grep_search", "run_tests",
        }
        assert set(agent.allowed_tools) == expected_tools


class TestParallelPipelineConfig:
    """Tests for parallel (multi-stage) pipeline configuration."""

    def test_parallel_config_has_multiple_agents(self):
        """Test that parallel pipeline has multiple agents."""
        from minisweagent.capability.orchestrator import (
            SWEOrchestrator,
            SWEOrchestratorConfig,
        )

        orchestrator = SWEOrchestrator(
            config=SWEOrchestratorConfig(pipeline_mode="parallel"),
        )
        config = orchestrator._create_parallel_pipeline_config()

        assert len(config.agents) > 1
        agent_names = {a.name for a in config.agents}
        assert "task_parser" in agent_names
        assert "code_searcher" in agent_names
        assert "file_analyzer" in agent_names
        assert "planner" in agent_names
        assert "executor" in agent_names
        assert "verifier" in agent_names

    def test_parallel_has_fan_out_fan_in(self):
        """Test that parallel pipeline has fan-out/fan-in structure."""
        from minisweagent.capability.orchestrator import (
            SWEOrchestrator,
            SWEOrchestratorConfig,
        )

        orchestrator = SWEOrchestrator(
            config=SWEOrchestratorConfig(pipeline_mode="parallel"),
        )
        config = orchestrator._create_parallel_pipeline_config()

        # Find agents
        agents_by_name = {a.name: a for a in config.agents}

        # code_searcher and file_analyzer should both require task_parser
        assert agents_by_name["code_searcher"].requires == ["task_parser"]
        assert agents_by_name["file_analyzer"].requires == ["task_parser"]

        # planner should require both (fan-in)
        planner = agents_by_name["planner"]
        assert "code_searcher" in planner.requires
        assert "file_analyzer" in planner.requires
        assert planner.join_strategy == "all"
