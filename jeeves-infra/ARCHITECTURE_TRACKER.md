# ARCHITECTURE_TRACKER.md

> Last Updated: 2026-01-29 (Session 23)
> Test Coverage: 32% (Go tests pending for CommBusServer)
> Tests: 334 passed, 32 skipped
> Architecture: Agentic OS (Hexagonal Ports & Adapters)
> **Status: P0 Circular Dependencies FIXED** ✅

---

## 0. Vision: Agentic OS

This system is designed as an **Agentic Operating System** - a platform for running autonomous AI agents (capabilities) with proper process isolation, memory management, and inter-agent communication.

### OS Analogy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AGENTIC OS ARCHITECTURE                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L3: USER SPACE (Agentic Apps / Capabilities)                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                           │
│  │mini-swe-    │ │ calendar-   │ │  research-  │  ... more capabilities     │
│  │agent        │ │ agent       │ │  agent      │                           │
│  └─────────────┘ └─────────────┘ └─────────────┘                           │
│                                                                              │
│  L2: SYSTEM SERVICES (Orchestration) ← mission_system                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐      │
│  │ ChatService  │ │ Orchestrator │ │  Governance  │ │ CapabilityMgr │      │
│  │ (routing)    │ │ (scheduling) │ │  (policies)  │ │ (app loading) │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────────┘      │
│  + gateway/ (HTTP API) + prompts/ + events/                                 │
│                                                                              │
│  L1: SYSTEM LIBRARIES (Infrastructure) ← jeeves_infra                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐      │
│  │ protocols/   │ │ memory/      │ │ postgres/    │ │ llm/          │      │
│  │ (interfaces) │ │ (storage)    │ │ (drivers)    │ │ (providers)   │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────────┘      │
│  + kernel_client/ (syscall layer to Go kernel)                              │
│                                                                              │
│  L0: KERNEL (jeeves-core / Go) ← ALWAYS REQUIRED                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐      │
│  │ControlTower  │ │   CommBus    │ │   Memory     │ │  Scheduler    │      │
│  │(process mgmt)│ │   (IPC)      │ │ (primitives) │ │  (execution)  │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Principle: Go Kernel Always Required

The Go kernel (jeeves-core) is **not optional**. Python layers are clients to the kernel:
- No Python fallbacks for kernel functionality
- Single source of truth for process management, IPC, scheduling
- Similar to Docker CLI requiring Docker daemon

---

## 1. jeeves-core: The Go Kernel (L0)

### What is jeeves-core?

**jeeves-core** is the foundational kernel layer written in Go. It provides the low-level primitives that all Python layers depend on. It runs as a **separate process** (daemon/service) that Python connects to via **gRPC**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  jeeves-core (Go Kernel)                                                     │
│  Repository: github.com/Jeeves-Cluster-Organization/jeeves-core             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   ControlTower   │  │     CommBus      │  │     Memory       │          │
│  │                  │  │                  │  │                  │          │
│  │ • Process spawn  │  │ • Pub/Sub        │  │ • KV store       │          │
│  │ • Process kill   │  │ • Request/Reply  │  │ • Session state  │          │
│  │ • Health checks  │  │ • Event routing  │  │ • Checkpoints    │          │
│  │ • Resource limits│  │ • Message queue  │  │ • Working memory │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │    Scheduler     │  │   gRPC Server    │  │    Telemetry     │          │
│  │                  │  │                  │  │                  │          │
│  │ • Task queue     │  │ • Python client  │  │ • Metrics        │          │
│  │ • Priority mgmt  │  │ • Proto defs     │  │ • Tracing        │          │
│  │ • Concurrency    │  │ • Streaming      │  │ • Logging        │          │
│  │ • Backpressure   │  │ • Auth/TLS       │  │ • Health         │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Go for the Kernel?

| Aspect | Go Advantage |
|--------|--------------|
| **Concurrency** | Goroutines handle thousands of concurrent processes efficiently |
| **Process Management** | Native OS process control, signals, resource limits |
| **Performance** | Low latency for IPC, memory operations |
| **Single Binary** | Easy distribution, no runtime dependencies |
| **Stability** | Static typing, no GIL, predictable performance |

### Kernel Components

| Component | Responsibility | Python Access |
|-----------|---------------|---------------|
| **ControlTower** | Process lifecycle (spawn, kill, monitor) | `kernel_client.process.*` |
| **CommBus** | Inter-process communication, pub/sub | `kernel_client.commbus.*` |
| **Memory** | Session state, working memory, checkpoints | `kernel_client.memory.*` |
| **Scheduler** | Task scheduling, priority, backpressure | `kernel_client.scheduler.*` |

### Kernel ↔ Python Communication

```
Python (jeeves_infra)                    Go (jeeves-core)
─────────────────────                    ────────────────

kernel_client/                           gRPC Server
├── client.py      ──── gRPC ────────►  ├── proto/
├── process.py                           ├── control_tower/
├── commbus.py                           ├── commbus/
└── memory.py                            └── memory/

Connection modes:
• Local:  grpc://localhost:50051  (subprocess or sidecar)
• Remote: grpc://kernel.prod:50051 (distributed deployment)
```

### Kernel Versioning

```
jeeves-core follows independent semver:

v1.0.0 - Initial stable release
  └── Python compatibility: jeeves_infra >= 1.0.0

Breaking changes in kernel require:
  1. Major version bump in jeeves-core
  2. Corresponding update in kernel_client/
  3. Compatibility matrix in documentation
```

---

## 2. Usage Modes

### Target Modes (Current Focus)

| Mode | Description | Deployment |
|------|-------------|------------|
| **Mode 3: Bundled** | Python spawns Go kernel as subprocess | `pip install jeeves-platform[kernel]` |
| **Mode 4: Container** | Pre-packaged with everything | `docker run jeeves/platform` |

### Emergent Modes (With Discipline)

If Mode 3/4 are built correctly, these emerge naturally:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Mode 1: SDK** | User imports jeeves_infra, manages own kernel | Custom apps |
| **Mode 2: Framework** | User uses mission_system orchestration | Standard deployment |

### Mode Details

```
MODE 3: BUNDLED (Primary Target)
════════════════════════════════
pip install jeeves-platform[kernel]

from mission_system import Jeeves
jeeves = Jeeves.standalone()  # Auto-spawns bundled Go kernel
jeeves.register_capability(MySWEAgent)
jeeves.run()

MODE 4: CONTAINER (Production Target)
═════════════════════════════════════
docker run -e CAPABILITIES=mini-swe-agent jeeves/platform
# Or: helm install jeeves/jeeves-platform
```

---

## 3. Git Repository Management

### Repository Structure

```
REPOSITORIES:
├── jeeves-core/              # Go kernel (SEPARATE REPO)
│   └── Standalone Go binary
│   └── gRPC server
│   └── Process management, CommBus, Memory
│
└── jeeves-infra/             # Python platform (THIS REPO - MONOREPO)
    ├── jeeves_infra/         # L1: SDK/Library package
    ├── mission_system/       # L2: Framework package
    ├── jeeves_tests/         # Shared test utilities
    └── capabilities/         # (future) Reference capabilities
```

### Why Monorepo for Python

| Aspect | Monorepo (Current) | Multi-repo |
|--------|-------------------|------------|
| Cross-package changes | ✅ Single PR | ❌ Coordinated PRs |
| CI/CD | ✅ Single pipeline | ❌ Multiple pipelines |
| Versioning | ⚠️ Requires discipline | ✅ Independent |
| Package boundaries | ⚠️ Easy to violate | ✅ Enforced |

**Decision**: Keep monorepo, enforce boundaries via:
- CI checks for circular imports
- Separate pyproject.toml per package
- Publish as separate PyPI packages

### Package Publishing

```
PyPI Packages (from monorepo):
├── jeeves-infra              # SDK only
│   └── protocols, memory, llm, postgres, kernel_client
│
├── mission-system            # Framework (depends on jeeves-infra)
│   └── bootstrap, orchestrator, gateway, services
│
└── jeeves-platform           # Meta-package
    └── depends on: jeeves-infra + mission-system
    └── [kernel] extra: bundles Go binary
```

### Version Strategy

```
jeeves-core:       v1.x.x  (Go, independent release cycle)
jeeves-infra:      v1.x.x  (Python SDK, follows semver)
mission-system:    v1.x.x  (Python framework, follows semver)
jeeves-platform:   v1.x.x  (Meta-package, combines above)
```

---

## 4. Architecture Overview

### Current Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Capabilities (User Space)                    │
│         mini-swe-agent │ other-capability │ future-caps         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ depends on
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 mission_system/ (L2: Orchestration)             │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐    │
│  │bootstrap │ │ adapters │ │orchestrator/│ │  services/   │    │
│  └──────────┘ └──────────┘ └─────────────┘ └──────────────┘    │
│  ┌──────────┐ ┌──────────┐                                      │
│  │ gateway/ │ │ prompts/ │  ✅ gateway MOVED HERE (Session 23)  │
│  └──────────┘ └──────────┘                                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ depends on
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                jeeves_infra/ (L1: Infrastructure)               │
│  ┌──────────┐ ┌────────┐ ┌──────────┐ ┌─────┐ ┌─────────────┐  │
│  │protocols/│ │memory/ │ │postgres/ │ │llm/ │ │kernel_client│  │
│  └──────────┘ └────────┘ └──────────┘ └─────┘ └─────────────┘  │
│  ┌──────────┐ ┌────────┐                                        │
│  │ runtime/ │ │ redis/ │  ✅ ZERO imports from mission_system   │
│  └──────────┘ └────────┘                                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ gRPC
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    jeeves-core (Go Kernel)                      │
│  ControlTower │ CommBus │ Memory │ Scheduler │ CommBusService   │
└─────────────────────────────────────────────────────────────────┘
```

### Design Pattern: Hexagonal Architecture

**Ports (Protocols)**: 45 Python Protocol interfaces define contracts
**Adapters**: Concrete implementations injected at runtime
**Core Domain**: Business logic isolated from infrastructure concerns

Key benefits:
- **Testability**: Mock protocols for unit testing
- **Flexibility**: Swap implementations without changing domain code
- **Independence**: Infrastructure changes don't affect business logic

---

## 5. Package Structure (Current)

### jeeves_infra/ (L1: Infrastructure - CLEAN)

```
jeeves_infra/                        # ✅ ZERO imports from mission_system
├── protocols/                       # Port definitions (45 protocols)
│   ├── __init__.py                 # Public exports
│   ├── core.py                     # RequestContext, LoggerProtocol
│   ├── persistence.py              # DatabaseClientProtocol, VectorStorageProtocol
│   ├── llm.py                      # LLMProviderProtocol
│   ├── tools.py                    # ToolProtocol, ToolExecutorProtocol
│   ├── memory.py                   # MemoryServiceProtocol, SessionStateProtocol
│   ├── distributed.py              # DistributedBusProtocol
│   ├── engine.proto                # gRPC proto (shared with jeeves-core)
│   ├── engine_pb2.py               # Generated Python stubs
│   └── engine_pb2_grpc.py          # Generated gRPC stubs
│
├── memory/                          # Memory subsystem
│   ├── manager.py                  # MemoryManager orchestration
│   ├── handlers.py                 # CommBus handler registration
│   ├── messages/                   # Query/Command/Event definitions
│   ├── services/                   # EmbeddingService, NLIService, EventEmitter
│   └── repos/                      # PgVectorRepository, GraphRepository
│
├── postgres/                        # Database adapters
│   ├── client.py                   # PostgreSQL client
│   └── graph.py                    # Graph storage
│
├── llm/                             # LLM integrations
│   └── providers/                  # OpenAI, Anthropic, local models
│
├── runtime/                         # Process/sandbox management
├── redis/                           # Redis connection management
│
├── kernel_client.py                 # Go kernel gRPC client ← ENHANCED (Session 23)
│   ├── KernelClient                # Process lifecycle, resource management
│   ├── CommBusClient               # Pub/sub via Go kernel ← NEW
│   └── get_commbus()               # Helper for CommBus access ← NEW
│
├── [DELETED] gateway/              # ← MOVED to mission_system (Session 23)
├── [DELETED] services/             # ← DELETED (duplicates) (Session 23)
└── [DELETED] utils/formatting/     # ← DELETED (duplicates) (Session 23)
```

### mission_system/ (L2: Orchestration - EXPANDED)

```
mission_system/
├── bootstrap.py                    # Application context factory
├── adapters.py                     # Dependency injection adapters
├── wiring.py                       # DI container configuration
├── orchestrator/                   # Mission orchestration logic
├── services/                       # Business services
│   ├── chat_service.py
│   ├── debug_api.py
│   └── worker_coordinator.py
│
├── gateway/                        # ← MOVED FROM jeeves_infra (Session 23)
│   ├── server.py                  # FastAPI application
│   ├── app.py                     # App factory
│   ├── health.py                  # Health checks
│   ├── chat.py                    # Chat handlers
│   ├── event_bus.py               # Event bus
│   ├── governance.py              # Governance endpoints
│   ├── sse.py                     # Server-sent events
│   ├── websocket.py               # WebSocket handlers
│   ├── websocket_manager.py       # WebSocket connection manager
│   ├── grpc_client.py             # gRPC utilities
│   ├── routers/                   # API route handlers
│   │   ├── chat.py
│   │   ├── health.py
│   │   └── interrupts.py
│   └── proto/                     # Legacy proto (if any)
│
└── tests/                          # Test suite (174+ tests)
```

### Total Codebase

| Package | Lines | Files | Responsibility |
|---------|-------|-------|----------------|
| `jeeves_infra/` | 38,441 | 127 | Infrastructure layer |
| `mission_system/` | 18,941 | 102 | Orchestration layer |
| **Total** | **57,382** | **229** | Full codebase |

---

## 6. Protocol Inventory (45 Protocols)

### Core Protocols

| Protocol | Location | Purpose |
|----------|----------|---------|
| `RequestContext` | protocols/core.py | Per-request state container |
| `LoggerProtocol` | protocols/core.py | Structured logging interface |
| `ConfigProtocol` | protocols/core.py | Configuration access |

### Persistence Protocols

| Protocol | Location | Purpose |
|----------|----------|---------|
| `DatabaseClientProtocol` | protocols/persistence.py | SQL database operations |
| `VectorStorageProtocol` | protocols/persistence.py | Vector embedding storage |
| `CheckpointProtocol` | protocols/persistence.py | State checkpointing |
| `GraphStorageProtocol` | protocols/persistence.py | Graph database operations |

### LLM Protocols

| Protocol | Location | Purpose |
|----------|----------|---------|
| `LLMProviderProtocol` | protocols/llm.py | LLM API abstraction |
| `EmbeddingProviderProtocol` | protocols/llm.py | Embedding generation |

### Tool Protocols

| Protocol | Location | Purpose |
|----------|----------|---------|
| `ToolProtocol` | protocols/tools.py | Tool definition interface |
| `ToolExecutorProtocol` | protocols/tools.py | Tool execution runtime |
| `AgentToolAccessProtocol` | protocols/tools.py | Agent tool permissions |

### Memory Protocols

| Protocol | Location | Purpose |
|----------|----------|---------|
| `MemoryServiceProtocol` | protocols/memory.py | Memory CRUD operations |
| `SessionStateProtocol` | protocols/memory.py | Session state management |
| `WorkingMemoryProtocol` | protocols/working_memory.py | Working memory access |

### Distributed Protocols

| Protocol | Location | Purpose |
|----------|----------|---------|
| `DistributedBusProtocol` | protocols/distributed.py | Cross-service messaging |
| `EventEmitterProtocol` | protocols/distributed.py | Event publication |

---

## 7. External Dependencies

### Required Infrastructure

| Service | Purpose | Protocol |
|---------|---------|----------|
| PostgreSQL | Primary data storage | `DatabaseClientProtocol` |
| pgvector | Vector similarity search | `VectorStorageProtocol` |
| Redis | Caching, pub/sub | `RedisConnectionProtocol` |
| jeeves-core (Go) | Kernel services | gRPC |

### Optional Services

| Service | Purpose | Impact if Missing |
|---------|---------|-------------------|
| OpenAI API | LLM provider | Use alternative provider |
| Anthropic API | LLM provider | Use alternative provider |
| sentence-transformers | Local embeddings | Requires 1.5GB+ download |

### Python Dependencies

```toml
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
structlog>=24.1.0

# Database
asyncpg>=0.29.0
psycopg2-binary>=2.9.0

# ML (optional)
sentence-transformers>=2.2.0
transformers>=4.36.0

# gRPC
grpcio>=1.60.0
protobuf>=4.25.0
```

---

## 8. Test Coverage

### Overall: 32% (Target: 60%)

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| `protocols/` | 78-100% | 45 | ✅ Good |
| `tools/` | 56-94% | 38 | ✅ Good |
| `common/` | 88-100% | 22 | ✅ Good |
| `orchestrator/` | 40-85% | 31 | ⚠️ Medium |
| `runtime/` | 20% | 8 | ❌ Low |
| `gateway/` | 0% | 0 | ❌ Critical |
| `postgres/` | 0% | 0 | ❌ Critical |
| `services/` | 0-22% | 12 | ❌ Critical |
| `memory/repos` | 0% | 0 | ❌ Critical |

### Critical Untested Files

| File | Lines | Impact |
|------|-------|--------|
| `gateway/routers/chat.py` | 33,160 | API endpoints - user-facing |
| `postgres/client.py` | 32,233 | Database layer - data integrity |
| `kernel_client.py` | 25,957 | Go kernel integration |
| `postgres/graph.py` | 21,159 | Graph storage |
| `wiring.py` | 21,853 | DI wiring - startup failures |
| `llm/providers/*` | 18,967+ | LLM integrations |

### Test Markers

| Marker | Count | Purpose |
|--------|-------|---------|
| `@pytest.mark.requires_postgres` | 32 | Needs PostgreSQL |
| `@pytest.mark.slow` | 8 | Long-running tests |
| `@pytest.mark.e2e` | 4 | End-to-end integration |

### Test Suites

```
mission_system/tests/  → 174 tests passing
jeeves_tests/          → 160 tests passing
─────────────────────────────────────────
Total                  → 334 tests passing, 32 skipped
```

---

## 9. Migration Status

### Completed Migrations

| Migration | Status | Session | Notes |
|-----------|--------|---------|-------|
| `avionics` → `jeeves_infra` | ✅ Complete | 18 | All imports migrated |
| `memory_module` → `jeeves_infra.memory` | ✅ Complete | 22 | 23 locations fixed |
| `gateway/` → `mission_system.gateway` | ✅ Complete | 23 | 18 files moved |
| `services/` deletion | ✅ Complete | 23 | Duplicate re-exports removed |
| `utils/formatting/` deletion | ✅ Complete | 23 | Duplicate re-exports removed |
| `control_tower` → `kernel_client` | ✅ Complete | 23 | CommBusClient added |

### Pending Migrations

| Legacy Reference | Count | Severity | Action Required |
|------------------|-------|----------|-----------------|
| `avionics` (docstrings only) | ~56 | Low | Cleanup when touching files |

### control_tower Imports (FIXED ✅)

All `control_tower` imports in `jeeves_infra` have been eliminated:

| Previous Import | Resolution |
|-----------------|------------|
| `gateway/server.py` control_tower refs | gateway moved to mission_system |
| `event_emitter.py` CommBus import | Uses `jeeves_infra.kernel_client.get_commbus()` |
| `services/worker_coordinator.py` | services/ deleted (was duplicate) |

**Pattern**: `control_tower` functionality now accessed via Go kernel's gRPC API through `kernel_client`

---

## 10. Library Readiness Assessment

> ✅ **READY**: jeeves_infra can now be used as a standalone library (Session 23)

### Circular Dependencies: FIXED ✅

**Before Session 23**:
```
mission_system ←──┐
      ↓          │  CIRCULAR!
  depends on     │
      ↓          │
jeeves_infra ────┘
```

**After Session 23**:
```
mission_system ──→ jeeves_infra ──→ jeeves-core (Go)
      ↓                   ↓
  depends on         depends on
                     (gRPC only)
```

### Resolution Applied (Session 23)

| Previous Issue | Resolution |
|----------------|------------|
| `gateway/` imports mission_system (10 imports) | gateway/ moved to mission_system |
| `services/` imports mission_system (3 imports) | services/ deleted (were duplicates) |
| `utils/formatting/` imports mission_system (2 imports) | utils/formatting/ deleted (were duplicates) |
| `event_emitter.py` imports control_tower | Uses `kernel_client.get_commbus()` |

### Library Readiness Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| No circular dependencies | ✅ FIXED | Zero `from mission_system` imports in jeeves_infra |
| No phantom package imports | ✅ FIXED | Zero `from control_tower` imports |
| Optional heavy dependencies | ✅ | ML deps are lazy-loaded |
| Protocol-based public API | ✅ | 45 protocols defined |
| Standalone tests pass | ⚠️ Partial | Need to verify with `pip install jeeves-infra` |
| CI boundary enforcement | ❌ Pending | P1 task |

### Verified Clean

```bash
# Verification commands (Session 23):
$ grep -r "^from mission_system" jeeves_infra/
# No matches (only docstring example)

$ grep -r "^from control_tower" jeeves_infra/
# No matches

$ python -c "import jeeves_infra; print('OK')"
# OK

$ python -c "from jeeves_infra.kernel_client import get_commbus; print('OK')"
# OK
```

---

## 11. Known Issues

### High Priority

| Issue | Location | Impact | Suggested Fix |
|-------|----------|--------|---------------|
| control_tower imports | 5 files | Import errors if control_tower removed | Migrate to kernel_client |
| 0% gateway coverage | gateway/ | API bugs undetected | Add integration tests |
| 0% postgres coverage | postgres/ | Data corruption risk | Add unit tests with mocks |

### Medium Priority

| Issue | Location | Impact | Suggested Fix |
|-------|----------|--------|---------------|
| ML deps not lazy | embedding_service.py | Slow startup | Already lazy, document |
| Large file sizes | chat.py (33K lines) | Hard to maintain | Consider splitting |
| Stale docstrings | 56 files | Confusion | Update when editing |

### Technical Debt

| Item | Effort | Value | Priority |
|------|--------|-------|----------|
| Split large files (chat.py, client.py) | High | High | P2 |
| Add gateway tests | Medium | High | P1 |
| Add postgres tests | Medium | High | P1 |
| Clean avionics docstrings | Low | Low | P3 |
| Document all 45 protocols | Medium | Medium | P2 |

---

## 12. Session History

### Session 23 (2026-01-29)

**Theme**: P0 Circular Dependency Fixes + CommBus gRPC Integration

**Changes**:
- ✅ **DELETED** `jeeves_infra/services/` - duplicate re-exports (4 files)
- ✅ **DELETED** `jeeves_infra/utils/formatting/` - duplicate re-exports (3 files)
- ✅ **MOVED** `jeeves_infra/gateway/` → `mission_system/gateway/` (18 files)
- ✅ Updated all gateway imports across 7 test files
- ✅ Added `CommBusService` to `engine.proto` (4 RPC methods)
- ✅ Implemented Go `CommBusServer` in `jeeves-core/coreengine/grpc/`
- ✅ Added `CommBusClient` wrapper to `kernel_client.py`
- ✅ Updated `event_emitter.py` to use `kernel_client.get_commbus()`
- ✅ Regenerated Go proto stubs

**Files Created/Modified**:
| jeeves-core (Go) | jeeves-infra (Python) |
|------------------|----------------------|
| `coreengine/proto/engine.proto` (CommBusService) | `kernel_client.py` (CommBusClient) |
| `coreengine/grpc/commbus_server.go` (NEW) | `event_emitter.py` (kernel_client import) |
| `coreengine/grpc/server.go` (register CommBus) | `mission_system/gateway/` (MOVED) |
| `coreengine/proto/engine.pb.go` (regen) | 7 test files (import updates) |
| `coreengine/proto/engine_grpc.pb.go` (regen) | |

**Verification**:
```
$ grep -r "^from mission_system" jeeves_infra/ → No matches ✅
$ grep -r "^from control_tower" jeeves_infra/ → No matches ✅
$ python -c "import jeeves_infra" → OK ✅
$ go build ./... (jeeves-core) → OK ✅
```

**Result**: `jeeves_infra` now has **ZERO circular dependencies**

---

### Session 22 (2026-01-29)

**Changes**:
- ✅ Fixed all 23 `memory_module` import references
- ✅ Implemented `set_request_pid`, `get_request_pid`, `clear_request_pid` in bootstrap.py
- ✅ Removed deprecated control_tower assertion in test_bootstrap.py
- ✅ Simplified mission_system dependencies to just `jeeves_infra>=1.0.0`
- ✅ All 174 mission_system tests passing
- ✅ All 160 jeeves_tests passing (32 skipped - require PostgreSQL)
- ✅ Created ARCHITECTURE_TRACKER.md
- ✅ Defined Agentic OS vision and layer model
- ✅ Identified 14 circular dependency violations (jeeves_infra → mission_system)
- ✅ Established usage modes (Mode 3: Bundled, Mode 4: Container as targets)
- ✅ Defined git/package management strategy (monorepo, separate PyPI packages)
- ✅ Prioritized restructuring plan (P0: fix boundaries, P1: CI/publish, P2: tests)

**Architectural Decisions**:
- Go kernel (jeeves-core) is ALWAYS REQUIRED - no Python fallbacks
- jeeves_infra = SDK/Library (L1), mission_system = Framework (L2)
- gateway/ and services/ should move from jeeves_infra to mission_system
- Dependencies flow DOWN only: L3 → L2 → L1 → L0

**Test Results**: 334 passed, 32 skipped, 0 failed

**Commit**: `feat: Session 22 - Complete memory_module migration`

### Session 21 (Previous)

**Changes**:
- Standalone library preparation
- Created `protocols/working_memory.py` (Finding, WorkingMemory, WorkingMemoryProtocol)
- Started memory_module migration (incomplete)

### Session 19

**Changes**:
- Consolidated Python protocols from jeeves-core
- 45 protocols now in jeeves_infra/protocols/

### Session 18

**Changes**:
- Complete import migration from `avionics` → `jeeves_infra`

---

## 13. Quick Reference Commands

### Run Tests

```bash
# All tests
pytest jeeves-infra/ -v

# Skip PostgreSQL tests
pytest jeeves-infra/ -v -m "not requires_postgres"

# With coverage
pytest jeeves-infra/ --cov=jeeves_infra --cov=mission_system --cov-report=term-missing
```

### Check Imports

```bash
# Find legacy imports
grep -r "from memory_module" jeeves-infra/
grep -r "from control_tower" jeeves-infra/
grep -r "from avionics" jeeves-infra/
```

### Package Info

```bash
# Line counts
find jeeves-infra/jeeves_infra -name "*.py" | xargs wc -l | tail -1
find jeeves-infra/mission_system -name "*.py" | xargs wc -l | tail -1
```

---

## 14. Next Steps (Prioritized for Agentic OS Vision)

### P0: Fix Package Boundaries ✅ COMPLETE (Session 23)

| Task | Status | Session |
|------|--------|---------|
| Move `gateway/` from jeeves_infra → mission_system | ✅ Done | 23 |
| Delete `services/` (duplicates) | ✅ Done | 23 |
| Delete `utils/formatting/` (duplicates) | ✅ Done | 23 |
| Migrate `control_tower` imports → `kernel_client` | ✅ Done | 23 |

**Result**: `jeeves_infra` has ZERO imports from `mission_system` ✅

---

### P1: Establish CI/Package Discipline (NEXT PRIORITY)

| Task | Purpose | Status |
|------|---------|--------|
| Add CI check: no jeeves_infra → mission_system imports | Prevent regression | ❌ Pending |
| Regenerate Python proto stubs | CommBusService in Python | ❌ Pending |
| Create separate pyproject.toml for jeeves_infra | Enable independent publish | ❌ Pending |
| Set up PyPI publishing workflow | Enable `pip install jeeves-infra` | ❌ Pending |
| Bundle Go kernel in `[kernel]` extra | Enable Mode 3 deployment | ❌ Pending |

### P2: Test Coverage

| Task | Current | Target | Notes |
|------|---------|--------|-------|
| Add Go tests for `CommBusServer` | 0% | 80% | **NEW - Session 23 gap** |
| Add tests for `mission_system/gateway/` | 0% | 40% | gateway moved from jeeves_infra |
| Add tests for `postgres/client.py` | 0% | 40% | |
| Add tests for `kernel_client.py` | 0% | 60% | Including CommBusClient |

### P3: Cleanup & Documentation

| Task | Priority |
|------|----------|
| Split `mission_system/gateway/routers/chat.py` (33K lines) | P3 |
| Document all 45 protocols with examples | P3 |
| Clean up ~56 stale `avionics` docstrings | P3 |

---

## 15. Remaining Gaps & Concerns

### Critical Gaps (Blocking Production)

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| **Python proto stubs not regenerated** | CommBusClient won't work until `engine_pb2.py` has CommBusService | Low | **P1** |
| **No Go tests for CommBusServer** | 0% coverage on new gRPC service | Medium | **P1** |
| **No CI boundary checks** | Regressions possible | Low | **P1** |

### Architectural Concerns

#### 1. Go Kernel Always Required
**Concern**: The Agentic OS mandates Go kernel for all operations. This means:
- No "lite" Python-only mode
- Docker/container deployment becomes the easiest path
- Local dev requires Go toolchain or pre-built binaries

**Mitigation**: Mode 3 (Bundled) auto-spawns kernel as subprocess, making it transparent to users.

#### 2. gRPC Adds Complexity
**Concern**: Every CommBus operation now crosses gRPC boundary:
- Network latency on local calls (~0.1-1ms per call)
- Proto regeneration required when API changes
- Debugging across Python ↔ Go boundary

**Mitigation**:
- Local gRPC is fast (unix sockets possible)
- Proto files versioned with clear compatibility matrix
- Structured logging on both sides

#### 3. CommBus Not Yet Production-Ready
**Current State**:
```
Go CommBusServer: ✅ Implemented (basic)
Python CommBusClient: ✅ Implemented (basic)
Subscribe (streaming): ⚠️ Implemented but untested
Go tests: ❌ Missing
Python tests: ❌ Missing
Error handling: ⚠️ Basic
Reconnection logic: ❌ Missing
```

**Gap**: The CommBus path is wired but not battle-tested.

#### 4. Test Coverage Debt
**Current State**:
| Component | Coverage | Risk |
|-----------|----------|------|
| `mission_system/gateway/` | 0% | **HIGH** - User-facing API |
| `jeeves-core/grpc/commbus_server.go` | 0% | **HIGH** - New code |
| `kernel_client.py` | 0% | **MEDIUM** - gRPC integration |
| `postgres/client.py` | 0% | **MEDIUM** - Data integrity |

### Operational Concerns

#### 1. Kernel Lifecycle Management
**Question**: Who starts/stops the Go kernel?
- Mode 3 (Bundled): Python spawns kernel subprocess
- Mode 4 (Container): Container orchestrator manages
- Mode 1/2 (SDK/Framework): User manages kernel

**Gap**: No implementation yet for subprocess spawning in Mode 3.

#### 2. Proto Version Compatibility
**Question**: How to handle proto version mismatches?
- Go kernel at v1.1, Python client at v1.0
- Need compatibility matrix and negotiation

**Gap**: No versioning handshake implemented.

#### 3. Observability
**Question**: How to debug cross-language issues?
- Python logs in structlog format
- Go logs in structured format
- Need correlation IDs across boundary

**Gap**: Correlation ID propagation not verified end-to-end.

---

## 16. Summary: Current Architecture (Post Session 23)

```
ACHIEVED (Session 23):

jeeves-infra/
├── jeeves_infra/                    # L1: SDK (pip install jeeves-infra)
│   ├── protocols/                   # 45 interfaces + engine.proto
│   ├── memory/                      # Memory adapters
│   ├── postgres/                    # DB driver
│   ├── llm/                         # LLM adapters
│   ├── redis/                       # Cache driver
│   ├── runtime/                     # Process management
│   ├── kernel_client.py             # gRPC client to Go kernel
│   │   ├── KernelClient             # Process lifecycle
│   │   └── CommBusClient            # Pub/sub (NEW Session 23)
│   └── NO imports from mission_system ✅
│
├── mission_system/                  # L2: Framework (pip install mission-system)
│   ├── gateway/                     # ✅ MOVED FROM jeeves_infra (Session 23)
│   ├── services/                    # Business services
│   ├── bootstrap.py
│   ├── adapters.py
│   ├── orchestrator/
│   ├── prompts/
│   └── depends on: jeeves_infra ✅
│
└── capabilities/                    # L3: Reference apps (future)
    └── mini-swe-agent/

jeeves-core/                         # L0: Go Kernel (separate repo)
├── coreengine/
│   ├── proto/engine.proto           # gRPC API definition
│   ├── grpc/
│   │   ├── server.go               # KernelService, EngineService
│   │   └── commbus_server.go       # CommBusService (NEW Session 23)
│   └── commbus/                    # CommBus implementation
└── Runs as daemon, Python connects via gRPC
```

**Key Invariant**: Dependencies flow DOWN only (L3 → L2 → L1 → L0) ✅

**CommBus Flow** (NEW Session 23):
```
Python EventEmitter
       │
       ↓ (async)
kernel_client.get_commbus()
       │
       ↓ (gRPC)
CommBusClient.publish()
       │
       ↓ (gRPC call)
Go CommBusServer.Publish()
       │
       ↓
InMemoryCommBus (Go)
```
