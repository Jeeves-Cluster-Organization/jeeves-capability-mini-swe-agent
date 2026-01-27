"""Prometheus Metrics Exporter."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Export metrics to Prometheus."""

    def __init__(self, port: int = 9090, enabled: bool = True):
        """Initialize metrics exporter.

        Args:
            port: HTTP port for metrics endpoint
            enabled: Whether metrics are enabled
        """
        self.port = port
        self.enabled = enabled
        self._server_started = False
        self._metrics = {}

        if enabled:
            try:
                from prometheus_client import Counter, Histogram, Gauge, start_http_server

                # Pipeline metrics
                self._metrics['pipeline_executions'] = Counter(
                    'mini_swe_pipeline_executions_total',
                    'Total pipeline executions',
                    ['pipeline_mode', 'status']
                )

                self._metrics['pipeline_duration'] = Histogram(
                    'mini_swe_pipeline_duration_seconds',
                    'Pipeline execution duration',
                    ['pipeline_mode'],
                    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
                )

                # Agent metrics
                self._metrics['agent_calls'] = Counter(
                    'mini_swe_agent_calls_total',
                    'Total agent calls',
                    ['agent_name', 'status']
                )

                self._metrics['agent_latency'] = Histogram(
                    'mini_swe_agent_latency_seconds',
                    'Agent latency',
                    ['agent_name'],
                    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
                )

                # LLM metrics
                self._metrics['llm_tokens'] = Counter(
                    'mini_swe_llm_tokens_total',
                    'Total LLM tokens',
                    ['model', 'type']
                )

                self._metrics['llm_cost'] = Counter(
                    'mini_swe_llm_cost_usd_total',
                    'Total LLM cost in USD',
                    ['model']
                )

                # Tool metrics
                self._metrics['tool_executions'] = Counter(
                    'mini_swe_tool_executions_total',
                    'Total tool executions',
                    ['tool_name', 'status']
                )

                self._metrics['tool_latency'] = Histogram(
                    'mini_swe_tool_latency_seconds',
                    'Tool execution latency',
                    ['tool_name'],
                    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30]
                )

                # Active sessions
                self._metrics['active_sessions'] = Gauge(
                    'mini_swe_active_sessions',
                    'Number of active sessions'
                )

                logger.info(f"Metrics configured on port {port}")

            except ImportError:
                logger.warning("prometheus_client not installed. Metrics disabled.")
                self.enabled = False

    def start_server(self):
        """Start Prometheus HTTP server."""
        if not self.enabled or self._server_started:
            return

        try:
            from prometheus_client import start_http_server
            start_http_server(self.port)
            self._server_started = True
            logger.info(f"Prometheus metrics available at http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def record_pipeline_execution(self, pipeline_mode: str, status: str, duration: float):
        """Record pipeline execution.

        Args:
            pipeline_mode: Pipeline mode (unified, parallel)
            status: Execution status (success, error)
            duration: Duration in seconds
        """
        if not self.enabled:
            return

        self._metrics['pipeline_executions'].labels(
            pipeline_mode=pipeline_mode,
            status=status
        ).inc()

        self._metrics['pipeline_duration'].labels(
            pipeline_mode=pipeline_mode
        ).observe(duration)

    def record_agent_call(self, agent_name: str, status: str, latency: float):
        """Record agent call.

        Args:
            agent_name: Agent identifier
            status: Call status (success, error)
            latency: Latency in seconds
        """
        if not self.enabled:
            return

        self._metrics['agent_calls'].labels(
            agent_name=agent_name,
            status=status
        ).inc()

        self._metrics['agent_latency'].labels(
            agent_name=agent_name
        ).observe(latency)

    def record_llm_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float):
        """Record LLM usage.

        Args:
            model: Model identifier
            input_tokens: Input token count
            output_tokens: Output token count
            cost: Cost in USD
        """
        if not self.enabled:
            return

        self._metrics['llm_tokens'].labels(model=model, type='input').inc(input_tokens)
        self._metrics['llm_tokens'].labels(model=model, type='output').inc(output_tokens)
        self._metrics['llm_cost'].labels(model=model).inc(cost)

    def record_tool_execution(self, tool_name: str, status: str, latency: float):
        """Record tool execution.

        Args:
            tool_name: Tool identifier
            status: Execution status (success, error)
            latency: Latency in seconds
        """
        if not self.enabled:
            return

        self._metrics['tool_executions'].labels(
            tool_name=tool_name,
            status=status
        ).inc()

        self._metrics['tool_latency'].labels(
            tool_name=tool_name
        ).observe(latency)

    def set_active_sessions(self, count: int):
        """Set active session count.

        Args:
            count: Number of active sessions
        """
        if not self.enabled:
            return

        self._metrics['active_sessions'].set(count)
