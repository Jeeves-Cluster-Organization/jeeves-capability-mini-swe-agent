<div align="center">

# jeeves-capability-mini-swe-agent

**A fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with jeeves-core agentic kernel integration**

**v2.0** — Now with session persistence, semantic search, and tool health monitoring

</div>

> [!NOTE]
> This is a **fork** of the excellent [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by the Princeton & Stanford team.
> This fork integrates [jeeves-core](https://github.com/Jeeves-Cluster-Organization/jeeves-core) for enhanced orchestration capabilities.

## What This Fork Adds

This fork extends mini-swe-agent with **jeeves-core agentic kernel** for:

- **Parallel Pipeline Execution**: Multi-stage pipelines with fan-out/fan-in patterns
- **Local LLM Support**: Native support for llama-server, Ollama, and OpenAI-compatible servers
- **Session Persistence**: Four-layer memory with PostgreSQL-backed state
- **Semantic Code Search**: Conceptual queries with pgvector embeddings
- **Dependency Graph**: AST-based entity extraction and relationship queries
- **Event Streaming**: Real-time visibility into agent progress
- **Resource Quotas**: Defense-in-depth bounds enforcement
- **Tool Health Monitoring**: Automatic quarantine of failing tools with metrics
- **Prometheus Metrics**: Full observability stack integration

## About the Original mini-swe-agent

The original [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) is a minimal 100-line AI agent:

- **Minimal**: Just 100 lines of python
- **Performant**: Scores >74% on SWE-bench verified benchmark
- **Deployable**: Supports docker, podman, singularity, apptainer, and more
- **Built by**: Princeton & Stanford team behind [SWE-bench](https://swebench.com) and [SWE-agent](https://swe-agent.com)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  mini-swe-agent (L5 - Capability Layer)                     │
│  - Domain-specific agents (SWE tasks)                       │
│  - Domain-specific tools (bash_execute, file ops)           │
│  - Pipeline configurations                                  │
└─────────────────────────────────────────────────────────────┘
                              │ imports
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  jeeves-core (L0-L4 - Runtime Layers)                       │
│  - PipelineRunner (orchestration)                           │
│  - LLM providers (OpenAI, Anthropic, llama-server)          │
│  - Control Tower (lifecycle, quotas)                        │
│  - Memory services (L1-L4)                                  │
└─────────────────────────────────────────────────────────────┘
                              │ connects
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL 15+ with pgvector                               │
│  - Working memory (L4 session state)                        │
│  - Tool health metrics (L7 monitoring)                      │
│  - Semantic embeddings (L3 code search)                     │
│  - Entity graph (L5 dependencies)                           │
│  - Event log (L2 checkpointing)                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Quick Start

Clone with submodule:

```bash
git clone --recursive https://github.com/Jeeves-Cluster-Organization/jeeves-capability-mini-swe-agent.git
cd jeeves-capability-mini-swe-agent
pip install -e ".[dev]"
```

### Database Setup (for v2.0 features)

```bash
# Set database URL
export MSWEA_DATABASE_URL="postgresql://user:pass@localhost/mswea"

# Run migrations
mini-jeeves db migrate

# Check status
mini-jeeves db status
```

## Usage

### Legacy Mode (Original mini-swe-agent behavior)

The original `mini` command continues to work:

```bash
mini -t "Fix the bug in auth.py"
mini -v  # Visual UI
```

### Jeeves Integration Mode (v2.0)

Use `mini-jeeves` for access to jeeves-core features:

```bash
# Basic run
mini-jeeves run -t "Fix the bug"

# Parallel pipeline mode (multi-stage with parallel analysis)
mini-jeeves run -t "Fix the bug" --mode parallel

# With local LLM
mini-jeeves run -t "Fix the bug" --llm-url http://localhost:8080/v1

# With session persistence
mini-jeeves run -t "Fix the bug" --new-session
mini-jeeves run -t "Continue fixing" --session session_20260127_123456

# With metrics enabled
mini-jeeves run -t "Fix the bug" --enable-metrics --metrics-port 9090

# Resume from checkpoint
mini-jeeves run --resume <checkpoint_id>
```

### Session Management

```bash
# List all sessions
mini-jeeves list-sessions [--limit 20]

# Delete a session
mini-jeeves session-delete <session_id>
```

### Code Indexing & Semantic Search

```bash
# Index your codebase for semantic search
mini-jeeves index . --pattern "**/*.py" [--chunk-size 512]

# Search conceptually
mini-jeeves search "authentication logic" [--limit 5]
```

### Dependency Graph

```bash
# Build dependency graph from AST
mini-jeeves graph-build .

# Query dependencies
mini-jeeves graph-deps src/auth.py --direction depends_on
mini-jeeves graph-deps src/auth.py --direction used_by
```

### Tool Health Monitoring

```bash
# View tool health status
mini-jeeves tool-health

# Reset a quarantined tool
mini-jeeves tool-reset <tool_name>
```

## Pipeline Modes

**Single-Agent Mode** (default): Single agent loop, matches original behavior.

```
[User Task] → [SWE Agent (loop)] → [Result]
```

**Parallel Pipeline Mode**: Multi-stage pipeline with parallel analysis:

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

| Tool | Risk Level | Description |
|------|------------|-------------|
| `bash_execute` | HIGH | Execute bash commands |
| `read_file` | READ_ONLY | Read file contents |
| `write_file` | WRITE | Write content to file |
| `edit_file` | WRITE | Replace text in file |
| `find_files` | READ_ONLY | Find files by pattern |
| `grep_search` | READ_ONLY | Search for pattern |
| `run_tests` | MEDIUM | Run project tests |
| `semantic_search` | READ_ONLY | L3 semantic code search |
| `graph_query` | READ_ONLY | L5 dependency queries |

## Python API

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

## Models

We recommend:
- `anthropic/claude-sonnet-4-5-20250929` for most tasks
- `openai/gpt-5` or `openai/gpt-5-mini` for OpenAI

Check scores at the [SWE-bench (bash-only)](https://swebench.com) leaderboard.

## Environment Variables

```bash
# Database
export MSWEA_DATABASE_URL="postgresql://user:pass@localhost/mswea"

# LLM Provider
export JEEVES_LLM_ADAPTER=openai_http  # or litellm
export JEEVES_LLM_BASE_URL=http://localhost:8080/v1
export JEEVES_LLM_MODEL=qwen2.5-7b-instruct

# Tool Configuration
export MSWEA_TOOL_TIMEOUT=30
export MSWEA_WORKING_DIR=/path/to/repo
```

## Documentation

- [Quick Start](docs/quickstart.md)
- [Jeeves Integration Guide](docs/jeeves_integration.md)
- [v2.0 Implementation Status](docs/v2-implementation-final-status.md)
- [FAQ](docs/faq.md)
- [Original mini-swe-agent docs](https://mini-swe-agent.com/latest/)

## Attribution

This project is a fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by the Princeton & Stanford team.

If you use this work, please cite the original [SWE-agent paper](https://arxiv.org/abs/2405.15793):

```bibtex
@inproceedings{yang2024sweagent,
  title={{SWE}-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={John Yang and Carlos E Jimenez and Alexander Wettig and Kilian Lieret and Shunyu Yao and Karthik R Narasimhan and Ofir Press},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/abs/2405.15793}
}
```

## Related Projects

- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) - The original minimal SWE agent (upstream)
- [jeeves-core](https://github.com/Jeeves-Cluster-Organization/jeeves-core) - Agentic kernel for orchestration
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) - Full-featured SWE agent
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) - Software engineering benchmark
