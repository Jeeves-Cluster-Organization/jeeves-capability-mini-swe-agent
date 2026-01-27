# Jeeves-Core Integration

This document describes how mini-swe-agent integrates with jeeves-core for enhanced orchestration capabilities.

## Overview

Mini-swe-agent can now operate as a **capability** within the jeeves-core runtime, gaining access to:

- **Parallel Pipeline Execution**: Multi-stage pipelines with fan-out/fan-in patterns
- **Local LLM Support**: Native support for llama-server, Ollama, and other OpenAI-compatible servers
- **Four-Layer Memory**: Session state persistence across queries
- **Event Streaming**: Real-time visibility into agent progress
- **Resource Quotas**: Defense-in-depth bounds enforcement
- **Tool Risk Levels**: Formal approval workflows for high-risk operations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  mini-swe-agent (L5 - Capability Layer)                     │
│  - Domain-specific agents (SWE tasks)                       │
│  - Domain-specific tools (bash_execute, file ops)           │
│  - Pipeline configurations                                   │
└─────────────────────────────────────────────────────────────┘
                              │ imports
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  jeeves-core (L0-L4 - Runtime Layers)                       │
│  - PipelineRunner (orchestration)                           │
│  - LLM providers (OpenAI, Anthropic, llama-server)         │
│  - Control Tower (lifecycle, quotas)                        │
│  - Memory services (L1-L4)                                  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. Clone with submodule:
```bash
git clone --recursive https://github.com/Jeeves-Cluster-Organization/jeeves-capability-mini-swe-agent.git
```

2. Or add submodule to existing clone:
```bash
git submodule add https://github.com/Jeeves-Cluster-Organization/jeeves-core.git jeeves-core
git submodule update --init --recursive
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Basic Usage (Legacy Mode)

The original `mini` command continues to work as before:

```bash
mini -t "Fix the bug in auth.py"
```

### Jeeves Integration Mode

Use `mini-jeeves` for access to jeeves-core features:

```bash
# Single-agent mode (legacy behavior, but with jeeves-core LLM provider)
mini-jeeves -t "Fix the bug" --mode single

# Parallel pipeline mode (multi-stage with parallel analysis)
mini-jeeves -t "Fix the bug" --mode parallel

# With local LLM
mini-jeeves -t "Fix the bug" --llm-url http://localhost:8080/v1 --mode parallel

# With streaming output
mini-jeeves -t "Fix the bug" --mode parallel --streaming
```

### Environment Variables

```bash
# LLM Provider Configuration
export JEEVES_LLM_ADAPTER=openai_http  # or litellm
export JEEVES_LLM_BASE_URL=http://localhost:8080/v1
export JEEVES_LLM_MODEL=qwen2.5-7b-instruct

# Tool Configuration
export MSWEA_TOOL_TIMEOUT=30
export MSWEA_WORKING_DIR=/path/to/repo
```

## Pipeline Modes

### Single-Agent Mode

Matches the original mini-swe-agent behavior: a single agent loop that iterates until completion.

```
[User Task] → [SWE Agent (loop)] → [Result]
```

### Parallel Pipeline Mode

Multi-stage pipeline with parallel analysis for faster execution:

```
                    ┌─> [code_searcher] ──┐
[task_parser] ─────>├─> [file_analyzer] ──├──> [planner] ──> [executor] ──> [verifier]
                    └─> [test_discovery]─┘
```

**Benefits:**
- 3x faster on multi-file tasks
- Different LLM models per stage (big for planning, small for execution)
- Streaming stage outputs for real-time visibility
- Automatic retry with loop-back on verification failure

## Tool Catalog

The capability defines its own tools (per Contract 10):

| Tool | Risk Level | Description |
|------|------------|-------------|
| `bash_execute` | HIGH | Execute bash commands |
| `read_file` | READ_ONLY | Read file contents |
| `write_file` | WRITE | Write content to file |
| `edit_file` | WRITE | Replace text in file |
| `find_files` | READ_ONLY | Find files by pattern |
| `grep_search` | READ_ONLY | Search for pattern |
| `run_tests` | MEDIUM | Run project tests |

## Agent LLM Configurations

Each pipeline stage can use a different LLM model:

| Agent | Default Model | Purpose |
|-------|---------------|---------|
| task_parser | qwen2.5-7b | Fast task analysis |
| code_searcher | qwen2.5-7b | Code search |
| file_analyzer | qwen2.5-14b | Deep file analysis |
| planner | qwen2.5-32b | Change planning |
| executor | qwen2.5-7b | Execute changes |
| verifier | qwen2.5-14b | Verify results |

## Programmatic Usage

```python
import asyncio
from minisweagent.capability import register_capability
from minisweagent.capability.orchestrator import create_swe_orchestrator, OrchestratorMode

# Register capability at startup
register_capability()

# Create orchestrator
orchestrator = create_swe_orchestrator(
    mode=OrchestratorMode.PARALLEL_PIPELINE,
    cost_limit=5.0,
)

# Run task
async def main():
    result = await orchestrator.run("Fix the authentication bug in auth.py")
    print(result)

asyncio.run(main())
```

## Testing

Run the capability tests:

```bash
pytest tests/capability/ -v
```

## Contract Compliance

This integration follows jeeves-core's contracts:

- **Contract 3**: Infrastructure accessed via adapters only
- **Contract 10**: ToolId enum is capability-owned
- **Contract 12**: Go runtime is authoritative for bounds

See [jeeves-core/CONTRACT.md](../jeeves-core/CONTRACT.md) for full contract documentation.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default mode | single_agent | Backward compatibility with existing behavior |
| LLM provider | openai_http | Zero external dependencies, works with local servers |
| Tool implementation | Async functions | Required for jeeves-core integration |
| Pipeline structure | 6-stage parallel | Balance between speed and accuracy |

## Future Enhancements

- [ ] Interrupt handling for clarifications
- [ ] Memory layer integration (findings persistence)
- [ ] gRPC API exposure
- [ ] Docker deployment support
