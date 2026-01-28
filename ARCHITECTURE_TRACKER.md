# Jeeves Architecture Tracker

> **Source of Truth** for jeeves-core architecture. Last updated: 2026-01-29 (Session 19)

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

### Package Layout After Session 19

```
jeeves-core/                    # PURE GO - no Python at all
├── coreengine/
│   ├── kernel/                 # Go kernel (process scheduler, resources)
│   ├── runtime/                # Go pipeline runner
│   ├── agents/                 # Go agents
│   ├── envelope/               # Go envelope
│   ├── grpc/                   # gRPC servers (KernelServer, EngineServer)
│   ├── proto/                  # Proto definitions (engine.proto)
│   ├── observability/          # Metrics & tracing
│   ├── tools/                  # Tool executor
│   ├── typeutil/               # Safe type utilities
│   ├── config/                 # Pipeline configuration
│   └── testutil/               # Test helpers
├── commbus/                    # Go communication bus
├── docs/                       # Architecture documentation
├── docker/                     # Docker configs
└── systemd/                    # Systemd service files

jeeves-infra/                   # Python infrastructure + application layer
├── jeeves_infra/
│   ├── kernel_client.py        # Python gRPC client for Go kernel
│   ├── protocols/              # All Python protocols (consolidated)
│   │   ├── __init__.py         # Exports all types, protocols
│   │   ├── engine_pb2.py       # Generated protobuf
│   │   ├── engine_pb2_grpc.py  # KernelServiceStub, EngineServiceStub
│   │   ├── interfaces.py       # Protocol interfaces
│   │   ├── types.py            # Dataclass types
│   │   └── capability.py       # Capability registration (moved from jeeves-core)
│   ├── runtime/                # Python agent execution (LLM, tools)
│   ├── utils/                  # Utilities
│   ├── context.py              # AppContext
│   ├── logging/                # Logging adapters
│   ├── settings.py             # Settings
│   ├── gateway/
│   ├── llm/
│   └── ...
├── mission_system/             # Application layer
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
| `jeeves_core/` (Python) | **DELETED** (Session 19) - protocols moved to jeeves_infra |
| `protocols/` (Python) | **DELETED** (Session 19) - moved to jeeves_infra/protocols |

---

### Session 19 Completed

1. **Cleaned jeeves-core - Now 100% Pure Go**
   - Deleted `jeeves_core/` Python package (types, events, protocols)
   - Deleted `protocols/` Python package (capability registry, gRPC client)
   - Deleted `scripts/` directory (Python scripts)
   - Deleted `tests/` Python tests
   - Deleted Python config files (conftest.py, pyproject.toml, pytest.ini)
   - Removed duplicate `github.com/` proto artifact
   - Removed coverage artifacts (cover, coverage.json, etc.)
   - Removed empty `cmd/` directory

2. **Consolidated Python Protocols in jeeves-infra**
   - Moved `capability.py` to `jeeves-infra/jeeves_infra/protocols/`
   - Updated `jeeves_infra/protocols/__init__.py` to export capability registration
   - Migrated all `from jeeves_core import` → `from jeeves_infra.protocols import`
   - Migrated all `from protocols import` → `from jeeves_infra.protocols import`

3. **Go Test Coverage Status**
   - kernel: 54.5% (target: >80%)
   - grpc: 46.6%
   - commbus: 77.9%
   - Other packages: 85-100%

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

### Session 20: Increase Go Test Coverage to >80%

Focus: **jeeves-core Go test coverage** - currently at 54.5% for kernel, need >80%.

1. **Kernel Test Coverage (Priority)**
   - Current: 54.5%, Target: >80%
   - Uncovered functions identified:
     - `rate_limiter.go`: OK, IsEmpty, SetDefaultConfig, SetEndpointLimits, GetUsage, ResetUser, CleanupExpired
     - `lifecycle.go`: IsValidTransition, TransitionState, Cleanup
     - `resources.go`: RecordToolCall, RecordAgentHop, UpdateElapsedTime, AdjustQuota, GetAllUsage
     - `kernel.go`: Lifecycle, RateLimiter, Interrupts, GetNextRunnable, TransitionState, Terminate, CheckQuota
     - `services.go`: UnregisterService, GetService, HasService, HasHandler, GetServiceStats
     - `types.go`: ProcessState methods, PCB methods, NewProcessControlBlock

2. **gRPC Test Coverage**
   - Current: 46.6%, Target: >70%
   - Add kernel_server integration tests

3. **Benchmarks**
   - Process scheduling throughput
   - Rate limiter under concurrent load

### After Session 20

- jeeves-core Go code hardened with >80% test coverage
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

## Session 20 Prompt

```
Session 20: Increase Go Test Coverage to >80%

Session 19 Completed:
- Cleaned jeeves-core to be 100% Pure Go (deleted all Python)
- Moved capability.py to jeeves-infra/jeeves_infra/protocols/
- Migrated all `from jeeves_core` and `from protocols` imports to jeeves_infra.protocols
- Removed duplicate proto directory and coverage artifacts

Current Go Test Coverage:
- kernel: 54.5% (TARGET: >80%)
- grpc: 46.6%
- commbus: 77.9%
- agents: 87.1%
- config: 95.6%
- envelope: 85.0%
- runtime: 91.8%

Tasks:

1. **Add kernel edge case tests** (kernel_test.go)
   - rate_limiter: window boundaries, cleanup, concurrent access
   - lifecycle: invalid state transitions, TransitionState, Cleanup
   - resources: AdjustQuota, UpdateElapsedTime, GetAllUsage
   - types: ProcessState/PCB methods, NewProcessControlBlock

2. **Add kernel benchmarks** (benchmark_test.go)
   - Process scheduling throughput
   - Rate limiter under concurrent load

3. **Verify all tests pass**
   cd jeeves-core
   go test ./... -cover

Build/Test Commands:
cd jeeves-core
go build ./...
go test ./coreengine/kernel/... -v -cover
go test ./coreengine/... -v
go test -bench=. ./coreengine/kernel/...

After Session 20:
- kernel coverage >80%
- Benchmarks established for performance baseline
- jeeves-core production ready
```
