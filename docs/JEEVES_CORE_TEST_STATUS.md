# Jeeves-Core Integration Test Status

**Date**: 2026-01-27
**jeeves-core submodule**: `b9bdb2b55086c7384c0759c76ceaca00e2090f19`

---

## Summary

After pulling the jeeves-core submodule, running tests, and cleaning up legacy code:

| Category | Passed | Failed | Skipped |
|----------|--------|--------|---------|
| Capability Tests | 60 | 0 | 0 |
| Environment Tests | 50 | 2* | 0 |
| Model Tests | 59 | 0 | 48 |
| Run Tests | 47 | 0 | 0 |
| **Total** | **216** | **2** | **48** |

**\*** 2 failures due to Docker not being available in test environment (expected infrastructure limitation)

---

## Code Coverage Analysis

### Overall Coverage: 43%

### Capability Layer Coverage: 36%

| File | Stmts | Miss | Cover | Priority |
|------|-------|------|-------|----------|
| **agents/graph_extractor.py** | 66 | 66 | 0% | High |
| **agents/swe_post_processor.py** | 70 | 70 | 0% | High |
| **cli/interactive_runner.py** | 100 | 100 | 0% | High |
| **db/migrator.py** | 90 | 90 | 0% | Medium |
| **tools/confirming_executor.py** | 108 | 108 | 0% | High |
| **interrupts/clarification_handler.py** | 37 | 37 | 0% | Medium |
| observability/metrics.py | 60 | 49 | 18% | Medium |
| services/code_indexer_service.py | 94 | 71 | 24% | Low |
| services/graph_service.py | 102 | 78 | 24% | Low |
| services/tool_health_service.py | 68 | 45 | 34% | Low |
| orchestrator.py | 244 | 158 | 35% | Medium |
| services/checkpoint_service.py | 57 | 36 | 37% | Low |
| services/event_log_service.py | 57 | 36 | 37% | Low |
| services/nli_service.py | 37 | 22 | 41% | Low |
| services/working_memory_service.py | 90 | 48 | 47% | Low |
| interrupts/cli_service.py | 98 | 50 | 49% | Low |
| tools/catalog.py | 193 | 83 | 57% | Low |
| interrupts/confirmation_handler.py | 52 | 19 | 63% | Low |
| services/event_stream_service.py | 44 | 13 | 70% | Done |
| wiring.py | 37 | 8 | 78% | Done |
| interrupts/mode_manager.py | 50 | 8 | 84% | Done |
| prompts/registry.py | 74 | 5 | 93% | Done |
| config/pipeline.py | 36 | 1 | 97% | Done |

---

## Missing Tests (Priority: High)

### 1. ConfirmingToolExecutor (`tools/confirming_executor.py`)
- **Coverage**: 0%
- **Purpose**: Tool executor with confirmation handling and health monitoring
- **Tests needed**:
  - `test_confirming_executor_requires_confirmation_for_high_risk`
  - `test_confirming_executor_skips_confirmation_in_yolo_mode`
  - `test_confirming_executor_records_tool_health`
  - `test_confirming_executor_blocks_quarantined_tools`

### 2. SWEPostProcessor (`agents/swe_post_processor.py`)
- **Coverage**: 0%
- **Purpose**: Completion detection and output processing
- **Tests needed**:
  - `test_post_processor_detects_completion_markers`
  - `test_post_processor_extracts_bash_commands`
  - `test_post_processor_handles_format_errors`
  - `test_post_processor_handles_timeout`

### 3. InteractiveRunner (`cli/interactive_runner.py`)
- **Coverage**: 0%
- **Purpose**: Interactive CLI for pipeline execution
- **Tests needed**:
  - `test_interactive_runner_displays_progress`
  - `test_interactive_runner_handles_user_input`
  - `test_interactive_runner_handles_keyboard_interrupt`

### 4. GraphExtractor (`agents/graph_extractor.py`)
- **Coverage**: 0%
- **Purpose**: AST-based entity extraction for dependency graph
- **Tests needed**:
  - `test_graph_extractor_extracts_functions`
  - `test_graph_extractor_extracts_classes`
  - `test_graph_extractor_extracts_imports`
  - `test_graph_extractor_builds_relationships`

### 5. DBMigrator (`db/migrator.py`)
- **Coverage**: 0%
- **Purpose**: Database migration management
- **Tests needed** (require PostgreSQL fixture):
  - `test_migrator_applies_pending_migrations`
  - `test_migrator_tracks_applied_migrations`
  - `test_migrator_dry_run_mode`

---

## Service Tests (Priority: Medium)

Services have 24-47% coverage and require database fixtures for full testing:

| Service | Coverage | Notes |
|---------|----------|-------|
| CodeIndexerService | 24% | Requires pgvector |
| GraphService | 24% | Requires PostgreSQL |
| ToolHealthService | 34% | Requires PostgreSQL |
| CheckpointService | 37% | Requires PostgreSQL |
| EventLogService | 37% | Requires PostgreSQL |
| NLIService | 41% | Requires sentence-transformers |
| WorkingMemoryService | 47% | Requires PostgreSQL |

**Recommendation**: Create a pytest fixture for PostgreSQL with pgvector to enable database-dependent tests.

---

## Legacy Code Removed

### Removed Source Files (5 files)
- `src/minisweagent/run/hello_world.py`
- `src/minisweagent/run/extra/github_issue.py`
- `src/minisweagent/run/extra/inspector.py`
- `src/minisweagent/run/extra/swebench.py`
- `src/minisweagent/run/extra/swebench_single.py`

### Removed Test Files (9 files)
- `tests/config/test_swebench_template.py`
- `tests/run/test_cli_integration.py`
- `tests/run/test_github_issue.py`
- `tests/run/test_inspector.py`
- `tests/run/test_local.py`
- `tests/run/test_run_hello_world.py`
- `tests/run/test_save.py`
- `tests/run/test_swebench.py`
- `tests/run/test_swebench_single.py`

### Removed Reference Docs (9 files)
- `docs/reference/run/swebench.md`
- `docs/reference/run/hello_world.md`
- `docs/reference/run/mini.md`
- `docs/reference/run/github_issue.md`
- `docs/reference/run/inspector.md`
- `docs/reference/run/swebench_single.md`
- `docs/reference/agents/default.md`
- `docs/reference/agents/interactive.md`
- `docs/reference/agents/textual.md`

---

## Working Integration Points

### jeeves-core protocols (from submodule)

| Component | Import | Tested |
|-----------|--------|--------|
| `CapabilityResourceRegistry` | `get_capability_resource_registry()` | Yes |
| `DomainModeConfig` | Registered via `register_capability()` | Yes |
| `DomainServiceConfig` | Registered via `register_capability()` | Yes |
| `DomainAgentConfig` | Registered via `register_capability()` | Yes |
| `CapabilityToolsConfig` | Registered via `register_capability()` | Yes |
| `CapabilityOrchestratorConfig` | Registered via `register_capability()` | Yes |
| `AgentLLMConfig` | Used in `wiring.py` | Yes |

### Capability Layer Components

| Component | File | Tests |
|-----------|------|-------|
| Tool Catalog | `capability/tools/catalog.py` | 16 tests |
| Orchestrator | `capability/orchestrator.py` | 12 tests |
| Pipeline Config | `capability/config/pipeline.py` | 8 tests |
| Prompt Registry | `capability/prompts/registry.py` | 7 tests |
| Interrupts | `capability/interrupts/` | 12 tests |
| Wiring | `capability/wiring.py` | 5 tests |

---

## Running Tests

```bash
# Install dependencies
pip install -e ".[dev]"
pip install -e jeeves-core/protocols/

# Run all tests
pytest tests/ -v

# Run capability tests only
pytest tests/capability/ -v

# Run with coverage
pytest tests/ --cov=src/minisweagent --cov-report=term-missing

# Run with HTML coverage report
pytest tests/capability/ --cov=src/minisweagent/capability --cov-report=html:coverage_html
```

---

## Estimated Effort for Full Coverage

| Task | Effort |
|------|--------|
| High-priority tests (5 files @ 0%) | 6-8 hours |
| Medium-priority tests (services) | 4-6 hours |
| PostgreSQL test fixtures | 2-3 hours |
| **Total** | **12-17 hours** |

---

**Last Updated**: 2026-01-27
