# `mini-jeeves`

!!! abstract "Overview"

    * `mini-jeeves` is the v2.0 CLI for mini-SWE-agent, integrated with jeeves-core.
    * It provides session persistence, semantic search, and tool health monitoring.
    * For legacy usage, see the original swe-agent/mini-swe-agent repository.

## Basic Usage

```bash
# Run with a task
mini-jeeves run -t "Fix the bug in auth.py"

# Run with a config file
mini-jeeves run -t "Task" -c mini_v2.yaml

# Run with a specific model
mini-jeeves run -t "Task" -m claude-sonnet-4-20250514
```

## Session Management

```bash
# Start a new session (persists state across runs)
mini-jeeves run -t "Fix auth bug" --new-session

# List existing sessions
mini-jeeves list-sessions

# Resume a session
mini-jeeves run -t "Continue fixing" --session session_20260127_123456

# Delete a session
mini-jeeves session-delete <session_id>
```

## Database Commands

v2.0 features require PostgreSQL with pgvector. Setup:

```bash
# Set database URL
export MSWEA_DATABASE_URL="postgresql://user:pass@localhost/mswea"

# Run migrations
mini-jeeves db migrate

# Check migration status
mini-jeeves db status
```

## Semantic Search

```bash
# Index your codebase
mini-jeeves index . --pattern "**/*.py"

# Search for code
mini-jeeves search "authentication logic"
```

## Dependency Graph

```bash
# Build dependency graph
mini-jeeves graph-build .

# Query dependencies
mini-jeeves graph-deps src/auth.py --direction depends_on
```

## Tool Health

```bash
# View tool health status
mini-jeeves tool-health

# Reset a quarantined tool
mini-jeeves tool-reset bash_execute
```

## Observability

```bash
# Enable Prometheus metrics
mini-jeeves run -t "Task" --enable-metrics --metrics-port 9090

# Then visit http://localhost:9090/metrics
```

## Modes of Operation

`mini-jeeves` provides three execution modes:

- `confirm` (`/c`): The LM proposes an action and the user confirms (Enter) or rejects
- `yolo` (`/y`): Actions are executed immediately without confirmation
- `human` (`/u`): The user takes over to type and execute commands

Start in yolo mode:
```bash
mini-jeeves run -t "Task" -y
```

Switch modes during execution with `/c`, `/y`, or `/u` commands.

## Command Line Options

| Option | Description |
|--------|-------------|
| `-t`, `--task` | Task description |
| `-c`, `--config` | Config file path |
| `-m`, `--model` | Model name |
| `-y`, `--yolo` | Start in yolo mode |
| `--new-session` | Start a new session |
| `--session` | Resume existing session |
| `--enable-metrics` | Enable Prometheus metrics |
| `--metrics-port` | Metrics server port (default: 9090) |

## Implementation

The v2.0 CLI is implemented using:

- `minisweagent.capability.orchestrator` - Pipeline execution
- `minisweagent.capability.cli.interactive_runner` - Interactive CLI
- `minisweagent.capability.wiring` - Agent configuration

{% include-markdown "../_footer.md" %}
