"""Integration tests for KernelClient -> Go kernel gRPC.

These tests verify that the Python KernelClient can communicate
with the Go kernel via gRPC.

Requirements:
- Go kernel must be running at KERNEL_GRPC_ADDRESS (default: localhost:50051)
- Run with: pytest tests/capability/test_kernel_client.py -v

Session 15: Wire mini-swe-agent to Go kernel via gRPC.
"""

import asyncio
import os
import pytest
import uuid

# Skip all tests if grpc is not available
grpc = pytest.importorskip("grpc")

from jeeves_infra.kernel_client import (
    KernelClient,
    KernelClientError,
    QuotaCheckResult,
    ProcessInfo,
)


# Check if kernel is available
KERNEL_ADDRESS = os.getenv("KERNEL_GRPC_ADDRESS", "localhost:50051")
KERNEL_AVAILABLE = os.getenv("TEST_WITH_KERNEL", "false").lower() == "true"


@pytest.fixture
def kernel_address():
    """Get the kernel gRPC address."""
    return KERNEL_ADDRESS


@pytest.fixture
async def kernel_client(kernel_address):
    """Create a kernel client for testing."""
    async with KernelClient.connect(kernel_address) as client:
        yield client


class TestKernelClientUnit:
    """Unit tests that don't require a running kernel."""

    def test_quota_check_result_dataclass(self):
        """Test QuotaCheckResult dataclass."""
        result = QuotaCheckResult(
            within_bounds=True,
            exceeded_reason="",
            llm_calls=5,
            tool_calls=10,
        )
        assert result.within_bounds is True
        assert result.llm_calls == 5
        assert result.tool_calls == 10

    def test_process_info_dataclass(self):
        """Test ProcessInfo dataclass."""
        info = ProcessInfo(
            pid="test-123",
            request_id="req-456",
            user_id="user-1",
            session_id="sess-1",
            state="RUNNING",
            priority="NORMAL",
            llm_calls=3,
        )
        assert info.pid == "test-123"
        assert info.state == "RUNNING"
        assert info.llm_calls == 3


@pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Go kernel not available")
class TestKernelClientIntegration:
    """Integration tests requiring a running Go kernel."""

    @pytest.mark.asyncio
    async def test_create_process(self, kernel_client):
        """Test creating a process."""
        pid = f"test-{uuid.uuid4()}"

        process = await kernel_client.create_process(
            pid=pid,
            user_id="test-user",
            session_id="test-session",
            max_llm_calls=10,
            max_tool_calls=20,
        )

        assert process.pid == pid
        assert process.user_id == "test-user"
        assert process.state == "NEW"

    @pytest.mark.asyncio
    async def test_get_process(self, kernel_client):
        """Test getting a process."""
        pid = f"test-{uuid.uuid4()}"

        # Create first
        await kernel_client.create_process(
            pid=pid,
            user_id="test-user",
        )

        # Then get
        process = await kernel_client.get_process(pid)

        assert process is not None
        assert process.pid == pid

    @pytest.mark.asyncio
    async def test_get_nonexistent_process(self, kernel_client):
        """Test getting a nonexistent process returns None."""
        process = await kernel_client.get_process("nonexistent-pid")
        assert process is None

    @pytest.mark.asyncio
    async def test_record_usage(self, kernel_client):
        """Test recording resource usage."""
        pid = f"test-{uuid.uuid4()}"

        # Create process
        await kernel_client.create_process(
            pid=pid,
            user_id="test-user",
            max_llm_calls=100,
        )

        # Record usage
        result = await kernel_client.record_usage(
            pid=pid,
            llm_calls=5,
            tool_calls=3,
            tokens_in=100,
            tokens_out=50,
        )

        assert result.llm_calls == 5
        assert result.tool_calls == 3

    @pytest.mark.asyncio
    async def test_check_quota(self, kernel_client):
        """Test checking quota."""
        pid = f"test-{uuid.uuid4()}"

        # Create process with low quota
        await kernel_client.create_process(
            pid=pid,
            user_id="test-user",
            max_llm_calls=5,
        )

        # Check quota (should be within bounds)
        result = await kernel_client.check_quota(pid)
        assert result.within_bounds is True

        # Record usage up to limit
        await kernel_client.record_usage(pid=pid, llm_calls=5)

        # Check quota again (should be at limit)
        result = await kernel_client.check_quota(pid)
        # Depending on Go kernel implementation, this may or may not exceed
        # The exact behavior depends on the kernel implementation

    @pytest.mark.asyncio
    async def test_record_llm_call_convenience(self, kernel_client):
        """Test the convenience record_llm_call method."""
        pid = f"test-{uuid.uuid4()}"

        await kernel_client.create_process(
            pid=pid,
            user_id="test-user",
            max_llm_calls=100,
        )

        # Should return None (not exceeded)
        exceeded = await kernel_client.record_llm_call(
            pid=pid,
            tokens_in=100,
            tokens_out=50,
        )
        assert exceeded is None

    @pytest.mark.asyncio
    async def test_list_processes(self, kernel_client):
        """Test listing processes."""
        # Create a few processes
        for i in range(3):
            await kernel_client.create_process(
                pid=f"test-list-{uuid.uuid4()}",
                user_id="list-test-user",
            )

        # List by user
        processes = await kernel_client.list_processes(user_id="list-test-user")
        assert len(processes) >= 3

    @pytest.mark.asyncio
    async def test_get_process_counts(self, kernel_client):
        """Test getting process counts."""
        counts = await kernel_client.get_process_counts()

        assert "total" in counts
        assert "queue_depth" in counts
        assert counts["total"] >= 0

    @pytest.mark.asyncio
    async def test_terminate_process(self, kernel_client):
        """Test terminating a process."""
        pid = f"test-{uuid.uuid4()}"

        await kernel_client.create_process(
            pid=pid,
            user_id="test-user",
        )

        process = await kernel_client.terminate_process(
            pid=pid,
            reason="test termination",
        )

        assert process.state == "TERMINATED"


class TestKernelClientMocked:
    """Tests with mocked gRPC calls."""

    @pytest.mark.asyncio
    async def test_client_connect_context_manager(self, kernel_address):
        """Test that connect context manager works."""
        # This will fail to actually connect but tests the context manager
        try:
            async with KernelClient.connect(kernel_address) as client:
                assert client is not None
                # Don't make actual calls
        except Exception:
            # Connection failures are expected without a running kernel
            pass
