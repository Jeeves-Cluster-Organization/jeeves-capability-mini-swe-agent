<div align="center">

# jeeves-capability-mini-swe-agent

**A fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with jeeves-core agentic kernel integration**

</div>

> [!NOTE]
> This is a **fork** of the excellent [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by the Princeton & Stanford team.
> This fork integrates [jeeves-core](https://github.com/Jeeves-Cluster-Organization/jeeves-core) for enhanced orchestration capabilities.

## What This Fork Adds

This fork extends mini-swe-agent with **jeeves-core agentic kernel** for:

- **Parallel Pipeline Execution**: Multi-stage pipelines with fan-out/fan-in patterns
- **Local LLM Support**: Native support for llama-server, Ollama, and other OpenAI-compatible servers
- **Session Persistence**: Four-layer memory with PostgreSQL-backed state
- **Event Streaming**: Real-time visibility into agent progress
- **Resource Quotas**: Defense-in-depth bounds enforcement
- **Tool Health Monitoring**: Automatic quarantine of failing tools

## About the Original mini-swe-agent

The original [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) is a minimal 100-line AI agent that scores >74% on SWE-bench verified.

Key features of the original:
- **Minimal**: Just 100 lines of python
- **Performant**: Scores >74% on SWE-bench verified benchmark
- **Deployable**: Supports docker, podman, singularity, apptainer, and more
- **Built by**: Princeton & Stanford team behind [SWE-bench](https://swebench.com) and [SWE-agent](https://swe-agent.com)

## Architecture

```
+-------------------------------------------------------------+
|  jeeves-capability-mini-swe-agent (L5 - Capability Layer)   |
|  - Domain-specific agents (SWE tasks)                       |
|  - Domain-specific tools (bash_execute, file ops)           |
|  - Pipeline configurations                                  |
+-------------------------------------------------------------+
                              | imports
                              v
+-------------------------------------------------------------+
|  jeeves-core (L0-L4 - Runtime Layers)                       |
|  - PipelineRunner (orchestration)                           |
|  - LLM providers (OpenAI, Anthropic, llama-server)          |
|  - Control Tower (lifecycle, quotas)                        |
|  - Memory services (L1-L4)                                  |
+-------------------------------------------------------------+
```

## Installation

Clone with submodule:

```bash
git clone --recursive https://github.com/Jeeves-Cluster-Organization/jeeves-capability-mini-swe-agent.git
cd jeeves-capability-mini-swe-agent
pip install -e ".[dev]"
```

## Usage

### Legacy Mode (Original mini-swe-agent behavior)

The original `mini` command continues to work:

```bash
mini -t "Fix the bug in auth.py"
```

### Jeeves Integration Mode

Use `mini-jeeves` for access to jeeves-core features:

```bash
# Parallel pipeline mode (multi-stage with parallel analysis)
mini-jeeves -t "Fix the bug" --mode parallel

# With local LLM
mini-jeeves -t "Fix the bug" --llm-url http://localhost:8080/v1

# With session persistence
mini-jeeves -t "Continue fixing" --session my-session

# With metrics enabled
mini-jeeves -t "Fix the bug" --enable-metrics --metrics-port 9090
```

### Pipeline Modes

**Unified Mode** (default): Single agent loop, matches original behavior.

**Parallel Mode**: Multi-stage pipeline with parallel analysis:

```
                    +-> [code_searcher] --+
[task_parser] ----->+-> [file_analyzer] --+---> [planner] --> [executor] --> [verifier]
                    +-> [test_discovery] -+
```

### Python API

```python
from minisweagent.capability import register_capability
from minisweagent.capability.orchestrator import create_swe_orchestrator

# Register capability at startup
register_capability()

# Create orchestrator
orchestrator = create_swe_orchestrator(pipeline_mode="parallel")

# Run task
result = await orchestrator.run("Fix the authentication bug in auth.py")
```

## Documentation

- [Jeeves Integration Guide](docs/jeeves_integration.md)
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
