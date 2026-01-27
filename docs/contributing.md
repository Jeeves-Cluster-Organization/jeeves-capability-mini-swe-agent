# Contributing

We happily accept contributions!

!!! note "Fork Notice"

    This is a fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with jeeves-core integration.
    For contributing to the original project, see [upstream contributing guide](https://mini-swe-agent.com/latest/contributing/).

## Areas of Help

- **Jeeves-core integration**: Improvements to the orchestrator, pipeline configurations, and v2.0 services
- **Documentation**: Examples of jeeves-core features, pipeline usage, local LLM setups
- **Testing**: Integration tests for the capability layer
- Take a look at the [issues](https://github.com/Jeeves-Cluster-Organization/jeeves-capability-mini-swe-agent/issues)

## Design & Architecture

This fork follows the original mini-swe-agent design principles:

- **Minimalistic**: Keep the core simple and hackable
- **Modular**: Components should be self-contained
- **High quality**: Clean, readable code with good test coverage

### Fork-specific Architecture

- `src/minisweagent/capability/` - Jeeves-core capability layer (orchestrator, tools, prompts)
- `src/minisweagent/run/mini_jeeves.py` - CLI for jeeves integration mode
- `jeeves-core/` - Submodule containing the agentic kernel

## Development Setup

```bash
git clone --recursive https://github.com/Jeeves-Cluster-Organization/jeeves-capability-mini-swe-agent.git
cd jeeves-capability-mini-swe-agent
pip install -e '.[dev]'
pip install pre-commit && pre-commit install
```

Run tests with `pytest -n auto`.

{% include-markdown "_footer.md" %}
