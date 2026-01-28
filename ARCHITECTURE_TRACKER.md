# Jeeves Architecture Tracker

> **Source of Truth** for jeeves-core architecture. Last updated: 2026-01-29 (Session 21)

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

### Session 21 Completed ✓

**KernelServer wiring FIXED + Python unit tests added.**

| Component | Implemented | Registered with gRPC |
|-----------|-------------|---------------------|
| `KernelServer` | ✅ 11 methods | ✅ **WIRED** (Session 21) |
| `EngineServer` | ✅ 7 methods | ✅ Active |

**Session 21 Changes:**
- Added `kernelServer` field to `EngineServer` struct
- Added `SetKernelServer()` and `SetKernel()` methods
- Updated `Start()`, `StartBackground()`, `NewGracefulServer()` to register both services
- Added 81 Python unit tests:
  - `test_kernel_client.py` (35 tests) - Process lifecycle, resource mgmt, queries
  - `test_protocols.py` (34 tests) - Capability registration, tool catalog
  - `test_llm_gateway.py` (12 tests) - Gateway dataclasses, quota patterns

**Note:** Full LLMGateway integration tests require `shared.logging` dependency (tested dataclasses in isolation)

**Go Server Changes Applied** (in `jeeves-core/coreengine/grpc/server.go`):
- Added `kernelServer *KernelServer` field to EngineServer
- Added `SetKernelServer()` and `SetKernel()` methods
- Updated `Start()`, `StartBackground()`, `NewGracefulServer()` to register both services
- All Go tests pass: `go test ./... -v`

### Go Kernel Package (COMPLETE - Session 20)

```
coreengine/kernel/
├── types.go          # ProcessState, ResourceQuota, PCB, KernelEvent (~340 LOC)
├── rate_limiter.go   # Sliding window rate limiting (~280 LOC)
├── lifecycle.go      # Process scheduler with priority heap (~250 LOC)
├── resources.go      # ResourceTracker - quota enforcement, usage tracking (~300 LOC)
├── interrupts.go     # InterruptService - create/resolve interrupts (~350 LOC)
├── services.go       # ServiceRegistry - dispatch, health tracking (~380 LOC)
├── kernel.go         # Main Kernel struct composing subsystems (~350 LOC)
├── kernel_test.go    # Comprehensive unit tests (~2700 LOC)
└── benchmark_test.go # Performance benchmarks (~250 LOC)
```

**Test Coverage: 92.5%** (target was >80%)

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

### Session 20 Completed

1. **Kernel Test Coverage: 54.5% → 92.5%**
   - Added comprehensive edge case tests for all modules
   - Types: ProcessState, SchedulingPriority, ResourceQuota, ResourceUsage, PCB, ServiceDescriptor, KernelEvent
   - Lifecycle: IsValidTransition, TransitionState, ListProcesses, Cleanup, GetQueueDepth, GetProcessCount
   - RateLimiter: SlidingWindow (IsEmpty, GetCount, TimeUntilSlotAvailable), SetDefaultConfig, GetUsage, ResetUser, CleanupExpired
   - Resources: RecordLLMCall/ToolCall/AgentHop, GetQuota, UpdateElapsedTime, AdjustQuota, GetAllUsage
   - Services: DispatchTarget (CanRetry, IncrementRetry), ServiceInfo (IsHealthy, Clone), UnregisterService, GetService, GetServiceNames, HasService, HasHandler, GetServiceStats, Dispatch edge cases
   - Kernel: Component accessors, GetNextRunnable, TransitionState, Terminate, CheckQuota, GetUsage, GetRemainingBudget, CheckRateLimit, GetRequestStatus, Cleanup, Shutdown
   - Interrupts: All WithInterrupt* options, Create convenience methods, GetInterrupt, GetPendingForSession, CleanupResolved, GetPendingCount, GetStats

2. **Added Kernel Benchmarks** (benchmark_test.go)
   - Process scheduling: Submit, Schedule, GetNextRunnable, FullCycle, PriorityScheduling
   - Rate limiter: CheckRateLimit, MultipleUsers, Concurrent
   - Resources: Allocate, RecordUsage, CheckQuota, GetRemainingBudget, Concurrent
   - Kernel: FullWorkflow, Concurrent_Submit
   - Priority queue: PushPop
   - Concurrent access stress tests

3. **Go Test Coverage Summary**
   - kernel: **92.5%** ✓
   - grpc: 46.6%
   - commbus: 77.9%

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

### Session 21: jeeves-infra Python Tests & Integration

Focus: **Python layer tests** for KernelClient and core infrastructure modules.

#### Why the `jeeves-infra/jeeves_infra/` Path Nesting?

This is standard Python packaging practice (PEP 517/518):
- `jeeves-infra/` = **repository root** (contains pyproject.toml, README, etc.)
- `jeeves_infra/` = **Python package** (importable module)

The nesting allows:
1. Clean `pip install -e .` from repo root
2. No namespace collision with other packages
3. Proper isolation of source from tests/docs/config

**Alternative:** Use `src/` layout (`jeeves-infra/src/jeeves_infra/`) but current layout is fine.

#### Priority Tests for jeeves-infra

1. **kernel_client.py** - Python gRPC client for Go kernel
   - Test connection lifecycle
   - Test RecordLLMCall, RecordToolCall, CheckQuota
   - Test Submit, Schedule, Terminate
   - Mock gRPC for unit tests, integration tests against running kernel

2. **protocols/** - Types and interfaces
   - Test capability registration
   - Test type conversions (proto ↔ Python dataclass)

3. **llm/gateway.py** - LLM gateway with KernelClient integration
   - Test quota enforcement via kernel
   - Test async stats updates
   - Mock LLM provider for fast tests

4. **context.py** - AppContext lifecycle
   - Test with_request() context propagation
   - Test kernel_client field

5. **runtime/agents.py** - Agent execution
   - Test agent lifecycle with kernel tracking

#### Test Patterns to Establish

**gRPC Mocking Pattern** (for kernel_client tests):
```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from jeeves_infra.kernel_client import KernelClient, ProcessInfo, QuotaCheckResult
from jeeves_infra.protocols import engine_pb2 as pb2

@pytest.fixture
def mock_grpc_channel():
    """Mock async gRPC channel."""
    channel = MagicMock()
    channel.close = AsyncMock()
    return channel

@pytest.fixture
def mock_kernel_stub():
    """Mock KernelServiceStub with async methods."""
    stub = MagicMock()
    # Mock process lifecycle
    stub.CreateProcess = AsyncMock()
    stub.GetProcess = AsyncMock()
    stub.ScheduleProcess = AsyncMock()
    stub.TerminateProcess = AsyncMock()
    # Mock resource tracking
    stub.RecordUsage = AsyncMock()
    stub.CheckQuota = AsyncMock()
    stub.CheckRateLimit = AsyncMock()
    return stub

@pytest.fixture
def kernel_client(mock_grpc_channel, mock_kernel_stub):
    """Configured KernelClient with mocked gRPC."""
    return KernelClient(
        channel=mock_grpc_channel,
        kernel_stub=mock_kernel_stub,
    )

def make_pcb(pid="test-1", state=pb2.PROCESS_STATE_RUNNING, priority=pb2.SCHEDULING_PRIORITY_NORMAL):
    """Factory for ProcessControlBlock responses."""
    pcb = pb2.ProcessControlBlock()
    pcb.pid = pid
    pcb.state = state
    pcb.priority = priority
    pcb.usage.llm_calls = 0
    return pcb
```

**LLM Gateway Mocking Pattern**:
```python
@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for gateway tests."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value="test response")
    return provider

@pytest.fixture
def mock_kernel_for_gateway():
    """Mock KernelClient for quota enforcement tests."""
    client = MagicMock()
    client.record_llm_call = AsyncMock(return_value=None)  # None = no quota exceeded
    return client

@pytest.fixture
def gateway_with_kernel(mock_kernel_for_gateway, settings):
    gateway = LLMGateway(settings, kernel_client=mock_kernel_for_gateway)
    gateway.set_pid("test-pid")
    return gateway
```

**Capability Registry Testing Pattern**:
```python
@pytest.fixture(autouse=True)
def reset_registry():
    """Reset global registry before each test."""
    from jeeves_infra.protocols.capability import reset_capability_resource_registry
    reset_capability_resource_registry()
    yield
    reset_capability_resource_registry()
```

**Integration Test Pattern** (requires running Go kernel):
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_kernel_connection():
    """Integration test against real Go kernel."""
    async with KernelClient.connect("localhost:50051") as client:
        # Create process
        proc = await client.create_process(pid="test-1", user_id="user-1")
        assert proc.state == "NEW"

        # Record usage
        usage = await client.record_usage(pid="test-1", llm_calls=1)
        assert usage.llm_calls == 1
```

#### Key Files to Test

| File | Priority | Notes |
|------|----------|-------|
| `kernel_client.py` | **HIGH** | Core integration point |
| `llm/gateway.py` | **HIGH** | Uses kernel for quota |
| `context.py` | MEDIUM | AppContext with kernel |
| `protocols/capability.py` | MEDIUM | Capability registration |
| `runtime/agents.py` | MEDIUM | Agent execution |
| `services/chat_service.py` | LOW | Higher-level service |

#### KernelClient API Reference

```python
class KernelClient:
    """Python gRPC client for Go kernel."""

    # Context manager for connection
    @classmethod
    async def connect(cls, address: str) -> AsyncIterator["KernelClient"]:
        async with KernelClient.connect("localhost:50051") as client:
            ...

    # Process Lifecycle
    async def create_process(pid, *, request_id, user_id, session_id, priority, max_llm_calls, ...) -> ProcessInfo
    async def get_process(pid) -> Optional[ProcessInfo]
    async def schedule_process(pid) -> ProcessInfo  # NEW -> READY
    async def get_next_runnable() -> Optional[ProcessInfo]  # READY -> RUNNING
    async def transition_state(pid, new_state, reason) -> ProcessInfo
    async def terminate_process(pid, reason, force) -> ProcessInfo

    # Resource Management
    async def record_usage(pid, *, llm_calls, tool_calls, agent_hops, tokens_in, tokens_out) -> QuotaCheckResult
    async def check_quota(pid) -> QuotaCheckResult
    async def check_rate_limit(user_id, endpoint, record) -> Dict[str, Any]

    # Queries
    async def list_processes(state, user_id) -> List[ProcessInfo]
    async def get_process_counts() -> Dict[str, int]

    # Convenience (record + check quota)
    async def record_llm_call(pid, tokens_in, tokens_out) -> Optional[str]  # Returns exceeded reason or None
    async def record_tool_call(pid) -> Optional[str]
    async def record_agent_hop(pid) -> Optional[str]

# Global client management
async def get_kernel_client(address) -> KernelClient  # Singleton
async def close_kernel_client() -> None
def reset_kernel_client() -> None  # For testing
```

#### LLMGateway API Reference

```python
class LLMGateway:
    """Unified LLM gateway with quota enforcement."""

    def __init__(settings, fallback_providers=None, logger=None, kernel_client=None, streaming_callback=None)

    async def complete(prompt, system, model, agent_name, tools, temperature, max_tokens) -> LLMResponse
    async def complete_stream(...) -> AsyncIterator[StreamingChunk]
    async def complete_stream_to_response(...) -> LLMResponse

    def set_pid(pid: str) -> None  # Set current process for tracking
    def set_kernel_client(client) -> None
    def get_stats() -> Dict[str, Any]

class QuotaExceededError(Exception):
    reason: str  # Why quota was exceeded
```

#### CapabilityResourceRegistry API Reference

```python
class CapabilityResourceRegistry:
    """Registry for capability resources."""

    # Schema registration
    def register_schema(capability_id, schema_path) -> None
    def get_schemas(capability_id=None) -> List[str]

    # Mode registration
    def register_mode(capability_id, mode_config: DomainModeConfig) -> None
    def get_mode_config(mode_id) -> Optional[DomainModeConfig]
    def is_mode_registered(mode_id) -> bool
    def list_modes() -> List[str]

    # Service registration
    def register_service(capability_id, service_config: DomainServiceConfig) -> None
    def get_services(capability_id=None) -> List[DomainServiceConfig]
    def get_default_service() -> Optional[str]

    # Extended registrations
    def register_orchestrator(capability_id, config: CapabilityOrchestratorConfig) -> None
    def register_tools(capability_id, config: CapabilityToolsConfig) -> None
    def register_prompts(capability_id, prompts: List[CapabilityPromptConfig]) -> None
    def register_agents(capability_id, agents: List[DomainAgentConfig]) -> None
    def register_contracts(capability_id, config: CapabilityContractsConfig) -> None

    def list_capabilities() -> List[str]
    def clear() -> None  # For testing

# Global singleton
def get_capability_resource_registry() -> CapabilityResourceRegistry
def reset_capability_resource_registry() -> None  # For testing
```

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

## Session 21 Prompt

```
Session 21: jeeves-infra Python Tests & KernelClient Integration

Session 20 Completed:
- Kernel test coverage: 54.5% → 92.5% ✓
- Added comprehensive tests for all kernel modules
- Added benchmarks for scheduling, rate limiting, resources
- jeeves-core Go layer production-ready

Current State:
- jeeves-core: 100% Go, kernel 92.5% coverage
- jeeves-infra: Python infrastructure, minimal test coverage
- KernelClient exists but untested

Tasks:

1. **Add KernelClient tests** (tests/test_kernel_client.py)
   - Unit tests with mocked gRPC
   - Connection lifecycle (connect, disconnect, reconnect)
   - RecordLLMCall, RecordToolCall, RecordAgentHop
   - Submit, Schedule, Terminate, CheckQuota

2. **Add LLMGateway tests** (tests/test_llm_gateway.py)
   - Quota enforcement via kernel_client
   - Async stats updates
   - QuotaExceededError handling

3. **Add protocol tests** (tests/test_protocols.py)
   - Capability registration/lookup
   - Type conversions

4. **Establish test patterns**
   - pytest fixtures for mocked kernel
   - Integration test markers (@pytest.mark.integration)

Test Commands:
cd jeeves-infra
pytest tests/ -v
pytest tests/test_kernel_client.py -v

Key Files:
- jeeves_infra/kernel_client.py
- jeeves_infra/llm/gateway.py
- jeeves_infra/protocols/capability.py
- jeeves_infra/context.py
```

---

## Session 22 Prompt (Future)

```
Session 22: Standalone Library & Go Entry Point

Session 21 Completed:
- Unit tests for kernel_client.py (35 tests)
- Unit tests for llm/gateway.py (12 tests - dataclasses only due to shared.* imports)
- Unit tests for protocols/capability.py (34 tests)
- Total: 81 new Python tests, all passing
- KernelServer wired to gRPC in Go server.go
- Docker configs moved from jeeves-core to jeeves-infra

Current Blockers for Standalone Library:
- 41 files import from `shared.*` instead of `jeeves_infra.utils.*`
- pyproject.toml has invalid `jeeves-core>=1.0.0` dependency (Go, not Python)
- No Go entry point to run kernel as standalone binary

Tasks:

1. **Fix shared.* imports** (~41 files)
   - `from shared.logging` → `from jeeves_infra.utils.logging`
   - `from shared.serialization` → `from jeeves_infra.utils.serialization`
   - `from shared.uuid_utils` → `from jeeves_infra.utils.uuid_utils`
   - `from shared.testing` → `from jeeves_infra.utils.testing`
   - `from shared.id_generator` → `from jeeves_infra.utils.id_generator`

2. **Fix pyproject.toml**
   - Remove `jeeves-core>=1.0.0` (Go code, not pip installable)
   - Add `grpcio>=1.60.0` to base dependencies
   - Add `grpcio-tools>=1.60.0` to base dependencies

3. **Create Go kernel entry point** (jeeves-core/cmd/main.go)
   - Standalone binary for kernel + engine servers
   - Enable distributed deployment (localhost or remote)
   - Add gRPC health check service

4. **KernelManager for Python** (jeeves_infra/kernel_manager.py)
   - Subprocess lifecycle for local sidecar mode
   - Connect to remote kernel option
   - Graceful startup/shutdown

5. **Integration tests against real Go kernel**
   - Start Go kernel server before tests
   - Test real gRPC communication
   - Use @pytest.mark.integration marker

Verification:
cd jeeves-infra
pip install -e .  # Should work without errors
pytest tests/unit/ -v  # All tests should pass
```
