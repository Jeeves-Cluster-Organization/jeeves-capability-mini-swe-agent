# Mini SWE Agent Constitution

Architectural principles for the software engineering capability.

## Purpose

Mini SWE Agent is a **capability layer** that provides software engineering automation. It uses the Jeeves infrastructure and kernel without modifying them.

## Core Principles

### 1. Capability Isolation

This capability:
- Defines domain-specific agents (understand, plan, execute, synthesize)
- Provides domain-specific tools (bash, file operations, code search)
- Owns prompt templates for SWE tasks
- Registers with the infrastructure via standard protocols

This capability does NOT:
- Modify the kernel or infrastructure
- Define reusable infrastructure components
- Bypass the kernel for state management

### 2. Pipeline-Driven Execution

All execution flows through declared pipelines:

```python
PipelineConfig(
    agents=[
        AgentConfig(name="understand", has_llm=True, has_tools=True),
        AgentConfig(name="plan", has_llm=True, has_tools=False),
        AgentConfig(name="execute", has_llm=False, has_tools=True),
        AgentConfig(name="synthesize", has_llm=True, has_tools=True),
    ]
)
```

Ad-hoc agent invocation is not permitted.

### 3. Tool Safety

Tools are categorized by risk:

| Risk Level | Tools | Confirmation |
|------------|-------|--------------|
| READ_ONLY | `read_file`, `find_files`, `grep_search` | None |
| WRITE | `write_file`, `edit_file` | Optional |
| HIGH | `bash_execute` | Required (unless YOLO mode) |

High-risk tools require explicit user confirmation by default.

### 4. Bounded Execution

All execution respects kernel bounds:
- Max iterations per task
- Max LLM calls per task
- Max agent hops per pipeline

The capability cannot bypass these limits.

### 5. Evidence-Based Output

All code changes must trace to:
1. File reads (what was there before)
2. Analysis (why change is needed)
3. Execution (what was changed)
4. Verification (tests pass)

Hallucinated code without file reads is not acceptable.

## Layer Position

```
Mini SWE Agent ─── THIS LAYER ────────────
     ↑ imports
Infrastructure ───────────────────────────
     ↑ gRPC
Kernel (Go) ──────────────────────────────
```

This capability:
- Imports from infrastructure (LLM, database, runtime)
- Does NOT import from kernel internals
- Does NOT import from other capabilities

## Contribution Criteria

Changes to mini-swe-agent must demonstrate:

1. **Domain relevance** - Is this for software engineering tasks?
2. **Pipeline integration** - Does this work within pipeline execution?
3. **Tool safety** - Are risk levels appropriate?
4. **Evidence tracing** - Can output be traced to file reads?

### Acceptable Changes

- New tools for code manipulation
- Improved prompt templates
- New pipeline modes
- Bug fixes with test coverage

### Requires Discussion

- Changes to tool risk levels
- New pipeline architectures
- Integration with external services

### Not Acceptable

- Bypassing kernel bounds
- Infrastructure modifications
- Tools without risk classification
- Output without evidence tracing

## Testing Requirements

All changes must include:
- Unit tests for new tools
- Integration tests for pipeline changes
- Tests that verify evidence tracing
