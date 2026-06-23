"""Integration tests for SpinClient with real Spin service.

These tests require the Spin tools-sdk container to be running:
    docker run -d -p 3000:3000 ghcr.io/nolabs-ai/deepfabric/tools-sdk:latest

In CI, the container is started as a service in the GitHub Actions workflow.
"""

import asyncio
import os
import uuid

import pytest  # pyright: ignore[reportMissingImports]

from deepfabric.spin.client import SpinClient

# Skip if SPIN_ENDPOINT is not set (container not running)
requires_spin = pytest.mark.skipif(
    not os.getenv("SPIN_ENDPOINT", "").strip(),
    reason="SPIN_ENDPOINT not set; skipping Spin integration test",
)


@pytest.fixture
def spin_endpoint():
    """Get Spin service endpoint from environment."""
    return os.getenv("SPIN_ENDPOINT", "http://localhost:3000")


@pytest.fixture
def spin_client(spin_endpoint):
    """Create a SpinClient instance."""
    return SpinClient(endpoint=spin_endpoint)


@pytest.fixture
def session_id():
    """Generate a unique session ID for test isolation."""
    return f"test-{uuid.uuid4().hex[:8]}"


class TestSpinClientToolExecution:
    """Tests for SpinClient tool execution."""

    @requires_spin
    @pytest.mark.spin
    def test_execute_list_files(self, spin_client, session_id):
        """Test executing the list_files tool."""

        async def run_execute():
            return await spin_client.execute_tool(
                session_id=session_id,
                tool_name="list_files",
                arguments={"directory": "/"},
                component="vfs",
            )

        result = asyncio.run(run_execute())

        assert result is not None
        assert result.success is True
        assert result.error_type is None

    @requires_spin
    @pytest.mark.spin
    def test_execute_write_and_read_file(self, spin_client, session_id):
        """Test writing and reading a file through Spin."""

        async def run_write_read():
            # Write a file
            write_result = await spin_client.execute_tool(
                session_id=session_id,
                tool_name="write_file",
                arguments={
                    "file_path": "/test.txt",
                    "content": "Hello from integration test",
                },
                component="vfs",
            )

            if not write_result.success:
                return write_result, None

            # Read it back
            read_result = await spin_client.execute_tool(
                session_id=session_id,
                tool_name="read_file",
                arguments={"file_path": "/test.txt"},
                component="vfs",
            )

            return write_result, read_result

        write_result, read_result = asyncio.run(run_write_read())

        assert write_result.success is True
        assert read_result is not None
        assert read_result.success is True
        assert "Hello from integration test" in read_result.result

    @requires_spin
    @pytest.mark.spin
    def test_execute_invalid_tool(self, spin_client, session_id):
        """Test that executing an invalid tool returns error."""

        async def run_invalid():
            return await spin_client.execute_tool(
                session_id=session_id,
                tool_name="nonexistent_tool",
                arguments={},
                component="vfs",
            )

        result = asyncio.run(run_invalid())

        assert result.success is False


class TestSpinClientSession:
    """Tests for SpinClient session management."""

    @requires_spin
    @pytest.mark.spin
    def test_session_isolation(self, spin_client):
        """Test that different sessions have isolated state."""
        session1 = f"test-iso-{uuid.uuid4().hex[:8]}"
        session2 = f"test-iso-{uuid.uuid4().hex[:8]}"

        async def run_isolation():
            # Write file in session 1
            await spin_client.execute_tool(
                session_id=session1,
                tool_name="write_file",
                arguments={"file_path": "/isolated.txt", "content": "session1"},
                component="vfs",
            )

            # Try to read in session 2 (should not exist or be different)
            return await spin_client.execute_tool(
                session_id=session2,
                tool_name="read_file",
                arguments={"file_path": "/isolated.txt"},
                component="vfs",
            )

        result = asyncio.run(run_isolation())

        # Session 2 should not see session 1's file
        assert result.success is False
