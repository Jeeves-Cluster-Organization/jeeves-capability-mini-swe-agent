"""Mini-SWE-Agent Capability for Jeeves-Core.

This module provides the capability layer integration between mini-swe-agent
and jeeves-core runtime. It registers mini-swe-agent as a capability that
can leverage jeeves-core's orchestration, memory, and infrastructure.

Constitutional Reference:
- Capability owns domain-specific logic (agent behavior, prompts, tools)
- Core provides runtime (LLM providers, orchestration, persistence)
- Integration via protocols and adapters, never direct imports

Usage:
    from minisweagent.capability import register_capability

    # At startup
    register_capability()
"""

from minisweagent.capability.wiring import register_capability

__all__ = ["register_capability"]
