# Inspector

!!! warning "Removed in v2.0"

    The `inspector` tool has been removed in the v2.0 migration.

    For trajectory browsing, see the original [swe-agent/mini-swe-agent](https://github.com/swe-agent/mini-swe-agent) repository.

## v2.0 Alternatives

In v2.0, session and execution history can be accessed via:

### Session Management

```bash
# List all sessions
mini-jeeves list-sessions

# View session details (if database configured)
mini-jeeves session-info <session_id>
```

### Event Logs

If PostgreSQL is configured, execution events are logged to the `event_log` table and can be queried directly:

```sql
SELECT * FROM event_log
WHERE session_id = 'your_session_id'
ORDER BY timestamp;
```

### Prometheus Metrics

For real-time monitoring:

```bash
mini-jeeves run -t "Task" --enable-metrics
# View metrics at http://localhost:9090/metrics
```

{% include-markdown "../_footer.md" %}
