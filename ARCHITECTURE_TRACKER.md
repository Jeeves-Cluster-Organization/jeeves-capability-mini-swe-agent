# Jeeves Architecture Tracker

> **Source of Truth** for jeeves-core architecture. Last updated: 2026-01-29 (Session 18)

---

## Vision: Pure Go Kernel

**jeeves-core = Pure Go binary. No Python.**

| Layer | Language | Owns |
|-------|----------|------|
| **jeeves-core** | Go | Kernel, coreengine, commbus, proto, gRPC servers |
| **jeeves-infra** | Python | LLM adapters, tools, DB, memory, gateway, runtime, utils, **mission_system** |
| **Capabilities** | Python | CLI, prompts, domain logic |

---

## Current State

### Go Kernel Package (COMPLETE - Session 13)

```
coreengine/kernel/
├── types.go        # ProcessState, ResourceQuota, PCB, KernelEvent (~340 LOC)
├── rate_limiter.go # Sliding window rate limiting (~280 LOC)
├── lifecycle.go    # Process scheduler with priority heap (~250 LOC)
├── resources.go    # ResourceTracker - quota enforcement, usage tracking (~300 LOC)
├── interrupts.go   # InterruptService - create/resolve interrupts (~350 LOC)
├── services.go     # ServiceRegistry - dispatch, health tracking (~380 LOC)
├── kernel.go       # Main Kernel struct composing subsystems (~350 LOC)
└── kernel_test.go  # Unit tests for all components (~500 LOC)
```

### Session 17 Completed

1. **Refactored LLMGateway for KernelClient**
   - Removed sync `resource_callback` pattern
   - Added `kernel_client` parameter for direct async integration
   - Added `set_pid()` method for per-request tracking
   - Added `QuotaExceededError` for quota enforcement
   - All `_update_stats()` calls now async

2. **Updated AppContext**
   - Added `kernel_client: Optional[KernelClient]` field
   - Deprecated `control_tower` (kept for migration)
   - Updated `with_request()` to copy kernel_client

3. **Updated bootstrap.py**
   - Removed sync `ControlTower` shim
   - `create_app_context()` returns `kernel_client=None`
   - Kernel connection happens async after event loop starts
   - Kept `ResourceQuota` dataclass for config

### Session 16 Completed

1. **Regenerated Python gRPC stubs with KernelServiceStub**
   - `jeeves-infra/jeeves_infra/protocols/engine_pb2_grpc.py` now has both stubs
   - `KernelClient` updated to use generated stub directly (no fallback)

2. **Moved `mission_system/` from jeeves-core to jeeves-infra**
   - mission_system is Python application layer (LangGraph, FastAPI)
   - Doesn't belong in pure Go jeeves-core
   - Now at `jeeves-infra/mission_system/`

3. **Updated mission_system imports**
   - `from avionics.*` → `from jeeves_infra.*` (partial)
   - Full import migration deferred to Session 18

### Session 14-15 Completed

1. **Deleted `control_tower/` Python** - Fully ported to Go kernel/
2. **Moved `jeeves_core/runtime/` to `jeeves_infra/runtime/`**
   - Agent, PipelineRunner, create_envelope, etc.
   - Python handles LLM calls, tool execution (capability layer)
3. **Moved `jeeves_core/utils/` to `jeeves_infra/utils/`**
   - JSONRepairKit, utc_now, normalize_string_list, truncate_string
4. **Updated all imports** - Clean break, no backwards compatibility
   - `from jeeves_infra.runtime import ...`
   - `from jeeves_infra.utils import ...`
5. **jeeves_core/__init__.py** now only exports types, events, protocols
6. **Created `jeeves_infra/kernel_client.py`** - Python gRPC client for Go kernel

### Package Layout After Session 16

```
jeeves-core/                    # PURE GO - no Python application code
├── coreengine/
│   ├── kernel/                 # Go kernel (process scheduler, resources)
│   ├── runtime/                # Go pipeline runner
│   ├── agents/                 # Go agents
│   ├── envelope/               # Go envelope
│   ├── grpc/                   # gRPC servers (KernelServer, EngineServer)
│   └── proto/                  # Proto definitions (engine.proto)
├── commbus/                    # Go communication bus
├── jeeves_core/                # Python bindings (thin - types only)
│   ├── __init__.py             # Re-exports types, events, protocols
│   ├── types/                  # Python dataclasses mirroring Go
│   └── protocols.py            # Protocol definitions
└── protocols/                  # Python protocols

jeeves-infra/                   # Python infrastructure + application layer
├── jeeves_infra/
│   ├── kernel_client.py        # Python gRPC client for Go kernel
│   ├── protocols/              # Generated gRPC stubs
│   │   ├── engine_pb2.py
│   │   └── engine_pb2_grpc.py  # KernelServiceStub, EngineServiceStub
│   ├── runtime/                # Python agent execution (LLM, tools)
│   ├── utils/                  # Utilities
│   ├── context.py              # AppContext (was avionics.context)
│   ├── logging.py              # Logging (was avionics.logging)
│   ├── settings.py             # Settings (was avionics.settings)
│   ├── gateway/
│   ├── llm/
│   └── ...
├── mission_system/             # Application layer (moved from jeeves-core)
│   ├── bootstrap.py            # Composition root
│   ├── orchestrator/           # LangGraph pipelines
│   ├── services/               # ChatService, WorkerCoordinator
│   ├── api/                    # HTTP API endpoints
│   └── ...
└── tests/
```

### Import Changes (Session 14)

| Old Import | New Import |
|------------|------------|
| `from jeeves_core.runtime import Agent` | `from jeeves_infra.runtime import Agent` |
| `from jeeves_core.runtime import PipelineRunner` | `from jeeves_infra.runtime import PipelineRunner` |
| `from jeeves_core.utils import JSONRepairKit` | `from jeeves_infra.utils import JSONRepairKit` |
| `from jeeves_core.utils import utc_now` | `from jeeves_infra.utils import utc_now` |

---

## Existing Go Code

| Package | LOC | Status |
|---------|-----|--------|
| `commbus/` | 1,579 | DONE |
| `coreengine/runtime/` | 679 | DONE |
| `coreengine/agents/` | 900 | DONE |
| `coreengine/envelope/` | 1,324 | DONE |
| `coreengine/grpc/` | ~1,300 | DONE (includes KernelServer) |
| `coreengine/kernel/` | ~2,750 | DONE |

## Python Code Status

| Package | Status |
|---------|--------|
| `control_tower/` | **DELETED** (Session 14) |
| `jeeves_core/runtime/` | **MOVED** to jeeves_infra (Session 14) |
| `jeeves_core/utils/` | **MOVED** to jeeves_infra (Session 14) |
| `mission_system/` | **MOVED** to jeeves_infra (Session 16) |
| `avionics/` | **INTEGRATED** into jeeves_infra |

---

### Session 18 Completed

1. **Fixed `from avionics.*` imports** (~78 files)
   - All `from avionics.*` → `from jeeves_infra.*`
   - Core modules: logging, settings, wiring, database, gateway, llm, etc.
   - Tests and mission_system updated

2. **Fixed `from control_tower.*` imports** (~9 files)
   - `ControlTower` → `KernelClient`
   - `SchedulingPriority` → string type alias
   - `KernelEvent` → local dataclass (in bridge.py)
   - `InMemoryCommBus` → stub (Go commbus/ is real impl)
   - TYPE_CHECKING imports updated to use KernelClient

3. **Remaining work deferred:**
   - Wire KernelClient to LLMGateway (needs async bootstrap refinement)
   - CommBus Python bindings (Go commbus/ needs gRPC service)

## Next Steps

### Session 19: Harden jeeves-core Go + Increase Test Coverage

Focus: **jeeves-core Go code only** - increase test coverage, harden the kernel.

1. **Increase test coverage for coreengine/kernel/**
   - Current: ~500 LOC tests, target: comprehensive edge cases
   - Test rate limiter edge cases (window boundaries, burst handling)
   - Test lifecycle state machine (invalid transitions, concurrent access)
   - Test resource tracker quota enforcement edge cases

2. **Add integration tests for gRPC servers**
   - `coreengine/grpc/kernel_server_test.go`
   - `coreengine/grpc/engine_server_test.go`
   - Test CreateProcess → RecordUsage → CheckQuota flow

3. **Add CommBus tests**
   - `commbus/bus_test.go` - expand coverage
   - Test pub/sub patterns, error handling

4. **Benchmark tests**
   - Process scheduling throughput
   - Rate limiter performance under load

### After Session 19

- jeeves-core Go code hardened with high test coverage
- Ready for production deployment
- Python integration via KernelClient validated

---

## Quick Reference

### Build Commands

```bash
cd jeeves-core && go build ./...
cd jeeves-core && go test ./coreengine/kernel/... -v
cd jeeves-core && go test ./coreengine/... -v
```

### Regenerate Proto

```bash
cd jeeves-core
protoc --go_out=paths=source_relative:. --go-grpc_out=paths=source_relative:. coreengine/proto/engine.proto
```

### Module Path

```
github.com/jeeves-cluster-organization/codeanalysis
```

Note: Repo is `github.com/Jeeves-Cluster-Organization/jeeves-core` but go.mod uses `codeanalysis`.

---

## Session 18 Prompt

```
Session 18: Complete Import Migration + Wire KernelClient

Session 17 Completed:
- Refactored LLMGateway to use KernelClient directly (no callbacks)
- Updated AppContext: added kernel_client, deprecated control_tower
- Updated bootstrap.py: removed ControlTower shim
- LLMGateway now has set_pid(), set_kernel_client(), QuotaExceededError

Remaining for Session 18:

1. Fix `from avionics.*` imports in jeeves_infra (~70 files)
   - grep -r "from avionics" jeeves-infra/jeeves_infra/
   - Main files: logging/, database/, llm/, gateway/, settings.py, wiring.py
   - Change to jeeves_infra.* or create missing modules

2. Fix `from control_tower.*` imports
   - grep -r "from control_tower" jeeves-infra/
   - Remove control_tower.types, control_tower.kernel imports
   - Use KernelClient or jeeves_core.types

3. Wire KernelClient to LLMGateway in bootstrap.py
   - Update create_avionics_dependencies() to pass kernel_client
   - Add connect_kernel_client() async helper

4. End-to-end test
   - go run ./coreengine/cmd/kernel
   - TEST_WITH_KERNEL=true pytest tests/capability/test_kernel_client.py -v

After Session 18:
- No avionics.* or control_tower.* imports anywhere
- KernelClient fully wired to LLMGateway
- Resource tracking flows: Python -> KernelClient -> Go kernel
```
