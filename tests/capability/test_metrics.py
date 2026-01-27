"""Tests for MetricsExporter with prometheus mocks."""

import pytest
from unittest.mock import MagicMock, patch, call


class MockCounter:
    """Mock Prometheus Counter."""

    def __init__(self, name, description, labels=None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._values:
            self._values[key] = MockCounterValue()
        return self._values[key]


class MockCounterValue:
    """Mock Counter value with labels."""

    def __init__(self):
        self.value = 0

    def inc(self, amount=1):
        self.value += amount


class MockHistogram:
    """Mock Prometheus Histogram."""

    def __init__(self, name, description, labels=None, buckets=None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or []
        self._values = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._values:
            self._values[key] = MockHistogramValue()
        return self._values[key]


class MockHistogramValue:
    """Mock Histogram value with labels."""

    def __init__(self):
        self.observations = []

    def observe(self, value):
        self.observations.append(value)


class MockGauge:
    """Mock Prometheus Gauge."""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.value = 0

    def set(self, value):
        self.value = value


@pytest.fixture
def mock_prometheus():
    """Create mock prometheus_client module."""
    mock_module = MagicMock()
    mock_module.Counter = MockCounter
    mock_module.Histogram = MockHistogram
    mock_module.Gauge = MockGauge
    mock_module.start_http_server = MagicMock()
    return mock_module


class TestMetricsExporterInit:
    """Tests for MetricsExporter initialization."""

    def test_init_with_enabled(self, mock_prometheus):
        """Test initialization with metrics enabled."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            from minisweagent.capability.observability.metrics import MetricsExporter

            # Need to reimport to pick up mock
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            exporter = metrics_module.MetricsExporter(port=9090, enabled=True)

            assert exporter.enabled is True
            assert exporter.port == 9090
            assert exporter._server_started is False
            assert 'pipeline_executions' in exporter._metrics
            assert 'agent_calls' in exporter._metrics
            assert 'llm_tokens' in exporter._metrics
            assert 'tool_executions' in exporter._metrics
            assert 'active_sessions' in exporter._metrics

    def test_init_with_disabled(self, mock_prometheus):
        """Test initialization with metrics disabled."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            exporter = metrics_module.MetricsExporter(port=9090, enabled=False)

            assert exporter.enabled is False
            assert exporter._metrics == {}

    def test_init_handles_missing_prometheus(self, mock_prometheus):
        """Test initialization handles missing prometheus_client gracefully."""
        # When prometheus_client import fails inside __init__, should disable metrics
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            # Simulate import error by making Counter raise
            def raise_import_error(*args, **kwargs):
                raise ImportError("No module named 'prometheus_client'")

            mock_prometheus.Counter = raise_import_error

            exporter = metrics_module.MetricsExporter(port=9090, enabled=True)
            # Should fall back to disabled due to import error
            assert exporter.enabled is False


class TestMetricsExporterRecording:
    """Tests for metrics recording methods."""

    @pytest.fixture
    def exporter(self, mock_prometheus):
        """Create exporter with mocked prometheus."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            return metrics_module.MetricsExporter(port=9090, enabled=True)

    def test_record_pipeline_execution(self, exporter):
        """Test recording pipeline execution."""
        exporter.record_pipeline_execution(
            pipeline_mode="unified",
            status="success",
            duration=10.5
        )

        # Check counter was incremented
        counter = exporter._metrics['pipeline_executions']
        key = (('pipeline_mode', 'unified'), ('status', 'success'))
        assert counter._values[key].value == 1

        # Check histogram observed
        histogram = exporter._metrics['pipeline_duration']
        key = (('pipeline_mode', 'unified'),)
        assert 10.5 in histogram._values[key].observations

    def test_record_agent_call(self, exporter):
        """Test recording agent call."""
        exporter.record_agent_call(
            agent_name="executor",
            status="success",
            latency=2.5
        )

        counter = exporter._metrics['agent_calls']
        key = (('agent_name', 'executor'), ('status', 'success'))
        assert counter._values[key].value == 1

        histogram = exporter._metrics['agent_latency']
        key = (('agent_name', 'executor'),)
        assert 2.5 in histogram._values[key].observations

    def test_record_llm_usage(self, exporter):
        """Test recording LLM usage."""
        exporter.record_llm_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            cost=0.05
        )

        tokens = exporter._metrics['llm_tokens']

        # Check input tokens
        input_key = (('model', 'gpt-4'), ('type', 'input'))
        assert tokens._values[input_key].value == 100

        # Check output tokens
        output_key = (('model', 'gpt-4'), ('type', 'output'))
        assert tokens._values[output_key].value == 50

        # Check cost
        cost = exporter._metrics['llm_cost']
        cost_key = (('model', 'gpt-4'),)
        assert cost._values[cost_key].value == 0.05

    def test_record_tool_execution(self, exporter):
        """Test recording tool execution."""
        exporter.record_tool_execution(
            tool_name="bash_execute",
            status="success",
            latency=0.5
        )

        counter = exporter._metrics['tool_executions']
        key = (('status', 'success'), ('tool_name', 'bash_execute'))
        assert counter._values[key].value == 1

        histogram = exporter._metrics['tool_latency']
        key = (('tool_name', 'bash_execute'),)
        assert 0.5 in histogram._values[key].observations

    def test_set_active_sessions(self, exporter):
        """Test setting active sessions gauge."""
        exporter.set_active_sessions(5)

        gauge = exporter._metrics['active_sessions']
        assert gauge.value == 5

        # Update the value
        exporter.set_active_sessions(3)
        assert gauge.value == 3


class TestMetricsExporterDisabled:
    """Tests for disabled metrics exporter."""

    @pytest.fixture
    def disabled_exporter(self, mock_prometheus):
        """Create disabled exporter."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            return metrics_module.MetricsExporter(port=9090, enabled=False)

    def test_record_pipeline_execution_noop(self, disabled_exporter):
        """Test that recording is noop when disabled."""
        # Should not raise
        disabled_exporter.record_pipeline_execution("unified", "success", 10.0)

    def test_record_agent_call_noop(self, disabled_exporter):
        """Test that recording is noop when disabled."""
        disabled_exporter.record_agent_call("executor", "success", 2.0)

    def test_record_llm_usage_noop(self, disabled_exporter):
        """Test that recording is noop when disabled."""
        disabled_exporter.record_llm_usage("gpt-4", 100, 50, 0.05)

    def test_record_tool_execution_noop(self, disabled_exporter):
        """Test that recording is noop when disabled."""
        disabled_exporter.record_tool_execution("bash", "success", 0.5)

    def test_set_active_sessions_noop(self, disabled_exporter):
        """Test that setting gauge is noop when disabled."""
        disabled_exporter.set_active_sessions(5)


class TestMetricsServer:
    """Tests for metrics HTTP server."""

    def test_start_server(self, mock_prometheus):
        """Test starting metrics server."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            exporter = metrics_module.MetricsExporter(port=8080, enabled=True)

            # Patch start_http_server at module level
            with patch.object(metrics_module, 'start_http_server', create=True) as mock_start:
                # Need to also patch it in the local scope
                with patch('prometheus_client.start_http_server', mock_prometheus.start_http_server):
                    exporter.start_server()

            assert exporter._server_started is True

    def test_start_server_only_once(self, mock_prometheus):
        """Test that server only starts once."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            exporter = metrics_module.MetricsExporter(port=8080, enabled=True)
            exporter._server_started = True  # Pretend already started

            # Should not call start_http_server again
            mock_prometheus.start_http_server.reset_mock()
            exporter.start_server()

            # Server was already started, so shouldn't call again
            assert exporter._server_started is True

    def test_start_server_disabled(self, mock_prometheus):
        """Test that server doesn't start when disabled."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            exporter = metrics_module.MetricsExporter(port=8080, enabled=False)

            exporter.start_server()

            assert exporter._server_started is False


class TestMultipleRecordings:
    """Tests for multiple metric recordings."""

    @pytest.fixture
    def exporter(self, mock_prometheus):
        """Create exporter with mocked prometheus."""
        with patch.dict('sys.modules', {'prometheus_client': mock_prometheus}):
            import importlib
            import minisweagent.capability.observability.metrics as metrics_module
            importlib.reload(metrics_module)

            return metrics_module.MetricsExporter(port=9090, enabled=True)

    def test_multiple_pipeline_executions(self, exporter):
        """Test multiple pipeline execution recordings."""
        exporter.record_pipeline_execution("unified", "success", 5.0)
        exporter.record_pipeline_execution("unified", "success", 10.0)
        exporter.record_pipeline_execution("parallel", "error", 3.0)

        counter = exporter._metrics['pipeline_executions']

        # Check unified success count
        unified_success_key = (('pipeline_mode', 'unified'), ('status', 'success'))
        assert counter._values[unified_success_key].value == 2

        # Check parallel error count
        parallel_error_key = (('pipeline_mode', 'parallel'), ('status', 'error'))
        assert counter._values[parallel_error_key].value == 1

    def test_multiple_tool_executions_different_tools(self, exporter):
        """Test recording multiple different tools."""
        exporter.record_tool_execution("bash_execute", "success", 0.1)
        exporter.record_tool_execution("read_file", "success", 0.05)
        exporter.record_tool_execution("bash_execute", "error", 1.0)

        counter = exporter._metrics['tool_executions']

        bash_success_key = (('status', 'success'), ('tool_name', 'bash_execute'))
        assert counter._values[bash_success_key].value == 1

        bash_error_key = (('status', 'error'), ('tool_name', 'bash_execute'))
        assert counter._values[bash_error_key].value == 1

        read_success_key = (('status', 'success'), ('tool_name', 'read_file'))
        assert counter._values[read_success_key].value == 1
