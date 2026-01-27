# Jeeves-Core Integration Test Status

**Date**: 2026-01-27
**jeeves-core submodule**: `b9bdb2b55086c7384c0759c76ceaca00e2090f19`

---

## Summary

| Category | Passed | Failed | Skipped |
|----------|--------|--------|---------|
| Capability Tests | 163 | 0 | 0 |
| Environment Tests | 50 | 2* | 0 |
| Model Tests | 59 | 0 | 48 |
| Run Tests | 47 | 0 | 0 |
| **Total** | **319** | **2** | **48** |

**\*** 2 failures due to Docker not being available in test environment (expected)

---

## Code Coverage Analysis

### Overall Coverage: 57% (Capability Layer)

### High Coverage (≥75%) ✅

| File | Coverage | Tests |
|------|----------|-------|
| `agents/swe_post_processor.py` | 100% | 23 |
| `interrupts/clarification_handler.py` | 100% | 15 |
| `observability/metrics.py` | 97% | 18 |
| `config/pipeline.py` | 97% | 8 |
| `prompts/registry.py` | 93% | 7 |
| `agents/graph_extractor.py` | 91% | 15 |
| `cli/interactive_runner.py` | 90% | 15 |
| `interrupts/mode_manager.py` | 88% | 12 |
| `tools/confirming_executor.py` | 79% | 17 |
| `wiring.py` | 78% | 5 |

### Medium Coverage (50-74%)

| File | Coverage | Notes |
|------|----------|-------|
| `services/event_stream_service.py` | 70% | Partial DB dependency |
| `interrupts/confirmation_handler.py` | 67% | Edge cases remain |
| `tools/catalog.py` | 57% | Complex tool definitions |
| `interrupts/cli_service.py` | 50% | Interactive prompts |

### Low Coverage (Require PostgreSQL)

| File | Coverage | Dependency |
|------|----------|------------|
| `services/working_memory_service.py` | 47% | PostgreSQL |
| `services/nli_service.py` | 41% | sentence-transformers |
| `services/checkpoint_service.py` | 37% | PostgreSQL |
| `services/event_log_service.py` | 37% | PostgreSQL |
| `orchestrator.py` | 35% | PostgreSQL + complex setup |
| `services/tool_health_service.py` | 34% | PostgreSQL |
| `services/code_indexer_service.py` | 24% | PostgreSQL + pgvector |
| `services/graph_service.py` | 24% | PostgreSQL |
| `db/migrator.py` | 0% | PostgreSQL |

---

## Tests Added This Session

| Test File | Tests | Module Coverage |
|-----------|-------|-----------------|
| `test_swe_post_processor.py` | 23 | 0% → 100% |
| `test_graph_extractor.py` | 15 | 0% → 91% |
| `test_confirming_executor.py` | 17 | 0% → 79% |
| `test_interactive_runner.py` | 15 | 0% → 90% |
| `test_metrics.py` | 18 | 18% → 97% |
| `test_clarification_handler.py` | 15 | 0% → 100% |
| **Total** | **103** | +21% coverage |

---

## Legacy Code Removed

### Source Files (5)
- `src/minisweagent/run/hello_world.py`
- `src/minisweagent/run/extra/github_issue.py`
- `src/minisweagent/run/extra/inspector.py`
- `src/minisweagent/run/extra/swebench.py`
- `src/minisweagent/run/extra/swebench_single.py`

### Test Files (9)
- Tests referencing removed `DefaultAgent`, `InteractiveAgent`, `run.mini` modules

### Reference Docs (9)
- `docs/reference/agents/{default,interactive,textual}.md`
- `docs/reference/run/{mini,hello_world,github_issue,inspector,swebench,swebench_single}.md`

---

## Working Integration Points

### jeeves-core protocols

| Component | Status |
|-----------|--------|
| `CapabilityResourceRegistry` | ✅ Tested |
| `DomainModeConfig` | ✅ Tested |
| `DomainServiceConfig` | ✅ Tested |
| `DomainAgentConfig` | ✅ Tested |
| `CapabilityToolsConfig` | ✅ Tested |
| `CapabilityOrchestratorConfig` | ✅ Tested |
| `AgentLLMConfig` | ✅ Tested |

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
pytest tests/capability/ --cov=src/minisweagent/capability --cov-report=term-missing
```

---

## Remaining Work

| Task | Effort |
|------|--------|
| PostgreSQL test fixtures | 2-3 hours |
| Service tests with fixtures | 4-6 hours |
| **Total** | **6-9 hours** |

---

**Last Updated**: 2026-01-27
