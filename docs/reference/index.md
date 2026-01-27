# API Reference

This section provides detailed documentation for all classes and modules in mini-SWE-agent v2.0.

## Capability Layer (v2.0)

The v2.0 architecture uses jeeves-core's PipelineRunner. Agent behavior is defined in:

- `minisweagent.capability.orchestrator` - SWE Orchestrator for pipeline execution
- `minisweagent.capability.wiring` - Agent definitions and LLM configurations
- `minisweagent.capability.tools.catalog` - Tool definitions and registration

### Key Components

| Component | Module | Description |
|-----------|--------|-------------|
| SWEOrchestrator | `capability.orchestrator` | Pipeline execution and service coordination |
| ToolCatalog | `capability.tools.catalog` | Tool registration and definitions |
| PromptRegistry | `capability.prompts.registry` | Prompt templates for agents |
| InteractiveRunner | `capability.cli.interactive_runner` | CLI execution with interrupt handling |
| ConfirmingExecutor | `capability.tools.confirming_executor` | Tool execution with confirmation |

## Models

- **[LitellmModel](models/litellm.md)** - Wrapper for LiteLLM models (supports most LLM providers)
- **[LitellmResponseAPIModel](models/litellm_response.md)** - Specialized model for OpenAI's Responses API
- **[AnthropicModel](models/anthropic.md)** - Specialized interface for Anthropic models
- **[DeterministicModel](models/test_models.md)** - Deterministic models for testing
- **[Model Utilities](models/utils.md)** - Convenience functions for model selection and configuration

## Environments

- **[LocalEnvironment](environments/local.md)** - Execute commands in the local environment
- **[DockerEnvironment](environments/docker.md)** - Execute commands in Docker containers
- **[SwerexDockerEnvironment](environments/swerex_docker.md)** - Extended Docker environment with SWE-Rex integration
- **[SwerexModalEnvironment](environments/swerex_modal.md)** - Modal cloud environment with SWE-Rex integration

## CLI Commands (v2.0)

The primary CLI is `mini-jeeves`. See [Usage Guide](../usage/mini.md) for details.

```bash
# Basic usage
mini-jeeves run -t "Fix the bug in auth.py"

# With session persistence
mini-jeeves run -t "Task" --new-session
mini-jeeves list-sessions

# Database commands
mini-jeeves db migrate
mini-jeeves db status
```

## Configuration

- **[Config Module](run/config.md)** - Configuration utilities

{% include-markdown "../_footer.md" %}