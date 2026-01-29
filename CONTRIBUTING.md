# Contributing to Mini SWE Agent

Thank you for your interest in contributing to Mini SWE Agent!

## Before You Start

Please read our [CONSTITUTION.md](CONSTITUTION.md) to understand the architectural principles. This is a capability layer that uses the Jeeves infrastructure without modifying it.

## Contribution Guidelines

### What We're Looking For

Mini SWE Agent is a **capability layer**. Contributions should:

1. **Be domain-relevant** - For software engineering tasks
2. **Work within pipelines** - Use declared pipeline execution
3. **Classify tool risks** - All tools need risk levels
4. **Trace evidence** - Output should trace to file reads

### Layer Boundaries

Before contributing, verify your change belongs in this layer:

| Change Type | Belongs In |
|-------------|------------|
| SWE-specific tools | mini-swe-agent (here) |
| SWE prompt templates | mini-swe-agent (here) |
| Pipeline configurations | mini-swe-agent (here) |
| LLM provider adapters | jeeves-infra |
| Runtime/pipeline execution | jeeves-infra |
| Kernel state management | jeeves-core |

## How to Contribute

### Reporting Issues

Please use the following format for issues:

```markdown
## Summary
Brief description of the issue or feature request.

## Type
- [ ] Bug report
- [ ] Feature request
- [ ] Documentation improvement
- [ ] Question

## Current Behavior
What happens now?

## Expected Behavior
What should happen?

## Steps to Reproduce (for bugs)
1. Step one
2. Step two
3. ...

## Environment
- Python version:
- OS:
- LLM provider/model:
- Package version:

## Task/Command Used
```bash
mini-jeeves run -t "your task here" --pipeline unified
```

## Logs/Output
```
Paste relevant logs here
```

## Additional Context
Any other relevant information.
```

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with tests
4. Ensure all tests pass: `pytest`
5. Submit a PR with the following template:

```markdown
## Summary
What does this PR do?

## SWE Relevance
How does this improve software engineering tasks?

## Changes
- List of changes

## Tool Risk Levels (if adding tools)
| Tool | Risk Level | Justification |
|------|------------|---------------|
| ... | ... | ... |

## Testing
- How was this tested?
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Tested with real LLM

## Checklist
- [ ] I've read CONSTITUTION.md
- [ ] Tests pass locally
- [ ] Tools have risk classifications
- [ ] Evidence tracing maintained
- [ ] Documentation updated if needed
```

## Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/Jeeves-Cluster-Organization/mini-swe-agent.git
cd mini-swe-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run the agent (requires Go kernel running)
mini-jeeves run -t "Fix the bug" --llm-url http://localhost:11434/v1
```

## Code Style

- Follow PEP 8
- Use type hints for all public functions
- Run `black` and `ruff` before committing
- Add docstrings for exported functions

## Questions?

Open an issue with the `question` label or start a discussion.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Attribution

This project is a fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by Princeton & Stanford. Please see the original project for their contribution guidelines if your changes affect forked code.
