# Jeeves-Core Integration Test Status

**Date**: 2026-01-27
**jeeves-core submodule**: `b9bdb2b55086c7384c0759c76ceaca00e2090f19`

---

## Summary

After pulling the jeeves-core submodule and running tests:

| Category | Passed | Failed | Skipped | Errors |
|----------|--------|--------|---------|--------|
| Capability Tests | 60 | 0 | 0 | 0 |
| Environment Tests | 50 | 2* | 0 | 0 |
| Model Tests | 59 | 0 | 48 | 0 |
| Legacy Tests | 0 | 0 | 0 | 9** |

**\*** 2 failures due to Docker not being available in test environment (expected)
**\*\*** 9 collection errors due to missing legacy modules (removed in v2.0)

**Total Working**: 169 passed, 2 infrastructure failures, 48 skipped
**Total Broken**: 9 test files with import errors

---

## Unwired Capabilities (Legacy Code Not Migrated)

The following legacy modules were removed as part of the v2.0 migration but their tests and dependent source files still reference them:

### Missing Agent Modules

| Module | Class/Function | Status |
|--------|----------------|--------|
| `minisweagent.agents.default` | `DefaultAgent`, `AgentConfig` | Removed |
| `minisweagent.agents.interactive` | `InteractiveAgent` | Removed |
| `minisweagent.agents.interactive_textual` | `_messages_to_steps` | Removed |

**Replacement**: All agent behavior now flows through `minisweagent.capability.orchestrator.SWEOrchestrator` using jeeves-core's `PipelineRunner`.

### Missing Run Modules

| Module | Entry Point | Status |
|--------|-------------|--------|
| `minisweagent.run.mini` | `mini` CLI | Removed |

**Replacement**: `minisweagent.run.mini_jeeves` provides the `mini-jeeves` CLI.

---

## Broken Test Files

These test files fail to import due to missing legacy modules:

| Test File | Missing Import | Action Required |
|-----------|----------------|-----------------|
| `tests/config/test_swebench_template.py` | `agents.default.AgentConfig` | Delete or migrate |
| `tests/run/test_cli_integration.py` | `run.mini` | Delete or migrate |
| `tests/run/test_github_issue.py` | `agents.interactive` | Delete or migrate |
| `tests/run/test_inspector.py` | `agents.interactive_textual` | Delete or migrate |
| `tests/run/test_local.py` | `run.mini` | Delete or migrate |
| `tests/run/test_run_hello_world.py` | `agents.default` | Delete or migrate |
| `tests/run/test_save.py` | `agents.default` | Delete or migrate |
| `tests/run/test_swebench.py` | `agents.default` | Delete or migrate |
| `tests/run/test_swebench_single.py` | `agents.interactive` | Delete or migrate |

---

## Broken Source Files

These source files still import removed legacy modules:

| Source File | Missing Import | Status |
|-------------|----------------|--------|
| `src/minisweagent/run/hello_world.py` | `agents.default.DefaultAgent` | Dead code |
| `src/minisweagent/run/extra/github_issue.py` | `agents.interactive.InteractiveAgent` | Dead code |
| `src/minisweagent/run/extra/inspector.py` | `agents.interactive_textual` | Dead code |
| `src/minisweagent/run/extra/swebench.py` | `agents.default.DefaultAgent` | Dead code |
| `src/minisweagent/run/extra/swebench_single.py` | `agents.interactive.InteractiveAgent` | Dead code |

---

## Working Integration Points

The following jeeves-core integrations are fully wired and tested:

### protocols.capability (from jeeves-core)

| Component | Import | Tests Passing |
|-----------|--------|---------------|
| `CapabilityResourceRegistry` | `get_capability_resource_registry()` | Yes |
| `DomainModeConfig` | Registered via `register_capability()` | Yes |
| `DomainServiceConfig` | Registered via `register_capability()` | Yes |
| `DomainAgentConfig` | Registered via `register_capability()` | Yes |
| `CapabilityToolsConfig` | Registered via `register_capability()` | Yes |
| `CapabilityOrchestratorConfig` | Registered via `register_capability()` | Yes |

### protocols (from jeeves-core)

| Component | Import | Tests Passing |
|-----------|--------|---------------|
| `AgentLLMConfig` | Used in `wiring.py` | Yes |

### Capability Layer

| Component | File | Tests Passing |
|-----------|------|---------------|
| Tool Catalog | `capability/tools/catalog.py` | Yes (9 tests) |
| Orchestrator | `capability/orchestrator.py` | Yes (12 tests) |
| Pipeline Config | `capability/config/pipeline.py` | Yes (8 tests) |
| Prompt Registry | `capability/prompts/registry.py` | Yes (7 tests) |
| Interrupts | `capability/interrupts/` | Yes (12 tests) |
| Wiring | `capability/wiring.py` | Yes (5 tests) |

---

## Recommended Actions

### Immediate (To Fix Test Suite)

1. **Delete broken test files** - These test legacy code that no longer exists
2. **Delete broken source files** - These are dead code that can't execute

### Future (v2.0 Completion)

Per `docs/v2-implementation-final-status.md`:

1. **Phase 3: Legacy Removal** (0% complete)
   - Remove backward compatibility wrappers
   - Remove deprecated code paths
   - Clean up unused imports

2. **Add Tests** (8-12 hours estimated)
   - Unit tests for service wrappers
   - Integration tests for new tools
   - E2E tests for session management

3. **Update Documentation** (4-6 hours estimated)
   - v1 -> v2 migration guide
   - Database setup guide
   - API reference updates

---

## Running Tests

```bash
# Install dependencies
pip install -e ".[dev]"
pip install -e jeeves-core/protocols/

# Run passing tests only
pytest tests/capability/ tests/agents/ tests/config/ tests/environments/ tests/models/ \
    --ignore=tests/config/test_swebench_template.py \
    --ignore=tests/run/ \
    -v

# Full test run (includes broken legacy tests)
pytest tests/ -v
```

---

**Last Updated**: 2026-01-27
