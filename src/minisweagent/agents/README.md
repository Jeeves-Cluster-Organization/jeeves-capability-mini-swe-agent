# Agent Implementations

**v2.0 Architecture**

Legacy agent implementations (`default.py`, `interactive.py`, `interactive_textual.py`) have been removed.

All agent behavior now flows through jeeves-core's PipelineRunner:

- `minisweagent.capability.orchestrator` - SWE Orchestrator for pipeline execution
- `minisweagent.capability.agents.swe_post_processor` - Post-processing logic
- `minisweagent.capability.cli.interactive_runner` - Interactive CLI runner

See `minisweagent.capability.wiring` for agent definitions and LLM configurations.