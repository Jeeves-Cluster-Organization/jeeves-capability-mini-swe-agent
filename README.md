# Mini SWE Agent

A minimal AI agent for software engineering tasks, enhanced with the Jeeves micro-kernel architecture.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SWE-bench](https://img.shields.io/badge/SWE--bench-74%25+-brightgreen)](https://swebench.com)

## Overview

Mini SWE Agent is a **capability layer** that provides software engineering automation built on top of the Jeeves micro-kernel architecture. It can:

- Fix bugs and implement features in codebases
- Navigate and understand code structure
- Execute shell commands safely
- Run tests and verify changes

This is a fork of the excellent [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by Princeton & Stanford, extended with the Jeeves orchestration system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  mini-swe-agent (Capability Layer)  ← THIS PACKAGE              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Orchestrator │  │    Tools     │  │   Prompts    │          │
│  │  (Pipelines) │  │  (Catalog)   │  │  (Registry)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │ imports
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  jeeves-infra (Infrastructure Layer)                            │
│  LLM providers, database clients, runtime                       │
└─────────────────────────────────────────────────────────────────┘
                              │ gRPC
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  jeeves-core (Micro-Kernel - Go)                                │
│  Pipeline orchestration, bounds checking, state management      │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/Jeeves-Cluster-Organization/mini-swe-agent.git
cd mini-swe-agent

# Install with dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```bash
# Run with a task
mini-jeeves run -t "Fix the bug in auth.py"

# With local LLM (Ollama, llama-server)
mini-jeeves run -t "Fix the bug" --llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b

# Skip confirmations (YOLO mode)
mini-jeeves run -t "Fix the bug" --yolo
```

### Pipeline Modes

```bash
# Unified mode (default) - single agent loop
mini-jeeves run -t "Fix the bug" --pipeline unified

# Sequential CoT - 4-stage pipeline (Understand → Plan → Execute → Synthesize)
mini-jeeves run -t "Fix the bug" --pipeline sequential

# Parallel mode - multi-stage with parallel analysis
mini-jeeves run -t "Fix the bug" --pipeline parallel
```

### Session Persistence

```bash
# Start new session
mini-jeeves run -t "Fix the bug" --new-session

# Resume session
mini-jeeves run -t "Continue fixing" --session session_20260130_123456

# List sessions
mini-jeeves list-sessions
```

## Pipeline Modes

### Unified (Default)

Single agent loop that mimics the original mini-swe-agent behavior:

```
[Task] → [SWE Agent (loop)] → [Result]
```

### Sequential Chain-of-Thought

4-stage pipeline with **3 LLM calls** (Execute stage is deterministic):

```
[Understand] → [Plan] → [Execute] → [Synthesize]
   (LLM)        (LLM)    (tools)      (LLM)
```

### Parallel

Multi-stage pipeline with parallel analysis:

```
              ┌─> [code_searcher] ──┐
[task_parser] ├─> [file_analyzer] ──├─> [planner] → [executor] → [verifier]
              └─> [test_discovery]─┘
```

## Tools

| Tool | Description |
|------|-------------|
| `bash_execute` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Write content to file |
| `edit_file` | Edit file with search/replace |
| `find_files` | Find files by pattern |
| `grep_search` | Search for patterns in files |
| `run_tests` | Run project test suite |
| `semantic_search` | Conceptual code search (requires database) |
| `graph_query` | Dependency graph queries (requires database) |

## Configuration

### Environment Variables

```bash
# LLM Provider
export JEEVES_LLM_ADAPTER="openai_http"
export JEEVES_LLM_BASE_URL="http://localhost:11434/v1"
export JEEVES_LLM_MODEL="qwen2.5:7b"

# Database (optional, for v2.0 features)
export MSWEA_DATABASE_URL="postgresql://user:pass@localhost/mswea"

# Go Kernel
export KERNEL_GRPC_ADDRESS="localhost:50051"
```

### YAML Configuration

See [mini.yaml](src/minisweagent/config/mini.yaml) for full configuration options.

## Python API

```python
import asyncio
from minisweagent.capability import register_capability
from minisweagent.capability.orchestrator import create_swe_orchestrator

# Register capability
register_capability()

# Create orchestrator
orchestrator = create_swe_orchestrator(
    pipeline_mode="sequential",
    cost_limit=5.0,
)

# Run task
async def main():
    result = await orchestrator.run("Fix the authentication bug")
    print(result)

asyncio.run(main())
```

## Related Projects

- [jeeves-core](https://github.com/Jeeves-Cluster-Organization/jeeves-core) - Go micro-kernel
- [jeeves-infra](https://github.com/Jeeves-Cluster-Organization/jeeves-infra) - Python infrastructure layer
- [Original mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) - Upstream project
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) - Full-featured SWE agent
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) - Software engineering benchmark

## Attribution

This project is a fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by the Princeton & Stanford team.

If you use this work, please cite the original SWE-agent paper:

```bibtex
@inproceedings{yang2024sweagent,
  title={{SWE}-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={John Yang and Carlos E Jimenez and Alexander Wettig and Kilian Lieret and Shunyu Yao and Karthik R Narasimhan and Ofir Press},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://arxiv.org/abs/2405.15793}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2024 Princeton NLP Group (original mini-swe-agent)
Copyright (c) 2024 Jeeves Cluster Organization (Jeeves integration)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
