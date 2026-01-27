"""Tests for SWE post-processor."""

import pytest

from minisweagent.capability.agents.swe_post_processor import (
    SWEPostProcessor,
    create_post_processor,
    COMPLETION_MARKERS,
)


class TestSWEPostProcessor:
    """Tests for SWEPostProcessor class."""

    def test_create_post_processor(self):
        """Test factory function creates processor."""
        processor = create_post_processor()
        assert isinstance(processor, SWEPostProcessor)

    def test_completion_markers_exist(self):
        """Test that completion markers are defined."""
        assert "MINI_SWE_AGENT_FINAL_OUTPUT" in COMPLETION_MARKERS
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in COMPLETION_MARKERS


class TestCompletionDetection:
    """Tests for completion marker detection."""

    def test_detects_completion_marker_in_tool_output(self):
        """Test completion marker detection."""
        processor = SWEPostProcessor()
        output = {
            "tool_results": [
                {
                    "result": {
                        "output": "MINI_SWE_AGENT_FINAL_OUTPUT\nTask completed successfully."
                    }
                }
            ]
        }

        result = processor.process(output)

        assert result["completed"] is True
        assert result["final_message"] == "Task completed successfully."

    def test_detects_alternate_completion_marker(self):
        """Test alternate completion marker detection."""
        processor = SWEPostProcessor()
        output = {
            "tool_results": [
                {
                    "result": {
                        "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nDone with fix."
                    }
                }
            ]
        }

        result = processor.process(output)

        assert result["completed"] is True
        assert result["final_message"] == "Done with fix."

    def test_no_completion_without_marker(self):
        """Test no completion flag without marker."""
        processor = SWEPostProcessor()
        output = {
            "tool_results": [
                {
                    "result": {
                        "output": "Some regular output"
                    }
                }
            ]
        }

        result = processor.process(output)

        assert "completed" not in result

    def test_multiline_final_message(self):
        """Test multiline final message extraction."""
        processor = SWEPostProcessor()
        output = {
            "tool_results": [
                {
                    "result": {
                        "output": "MINI_SWE_AGENT_FINAL_OUTPUT\nLine 1\nLine 2\nLine 3"
                    }
                }
            ]
        }

        result = processor.process(output)

        assert result["completed"] is True
        assert result["final_message"] == "Line 1\nLine 2\nLine 3"


class TestLimitsExceeded:
    """Tests for limits exceeded detection."""

    def test_llm_call_limit_exceeded(self):
        """Test LLM call limit exceeded detection."""
        processor = SWEPostProcessor()
        output = {}

        result = processor.process(output, llm_call_count=10, max_llm_calls=10)

        assert result["limits_exceeded"] is True
        assert "LLM call limit exceeded" in result["error_message"]

    def test_agent_hop_limit_exceeded(self):
        """Test agent hop limit exceeded detection."""
        processor = SWEPostProcessor()
        output = {}

        result = processor.process(output, agent_hop_count=5, max_agent_hops=5)

        assert result["limits_exceeded"] is True
        assert "Agent hop limit exceeded" in result["error_message"]

    def test_no_limit_exceeded_when_under_limit(self):
        """Test no limit exceeded when under limits."""
        processor = SWEPostProcessor()
        output = {}

        result = processor.process(output, llm_call_count=5, max_llm_calls=10)

        assert "limits_exceeded" not in result

    def test_no_limit_check_when_max_is_zero(self):
        """Test no limit check when max is zero (disabled)."""
        processor = SWEPostProcessor()
        output = {}

        result = processor.process(output, llm_call_count=100, max_llm_calls=0)

        assert "limits_exceeded" not in result


class TestFormatErrorDetection:
    """Tests for format error detection."""

    def test_format_error_no_bash_block(self):
        """Test format error when no bash block found."""
        processor = SWEPostProcessor()
        output = {
            "llm_response": "I will fix the bug by changing the code."
        }

        result = processor.process(output)

        assert result["format_error"] is True
        assert result["needs_retry"] is True
        assert "EXACTLY ONE bash command" in result["error_message"]

    def test_format_error_multiple_bash_blocks(self):
        """Test format error when multiple bash blocks found."""
        processor = SWEPostProcessor()
        output = {
            "llm_response": "```bash\necho first\n```\n```bash\necho second\n```"
        }

        result = processor.process(output)

        assert result["format_error"] is True
        assert result["needs_retry"] is True
        assert "Found 2 bash blocks" in result["error_message"]

    def test_no_format_error_with_single_bash_block(self):
        """Test no format error with single bash block."""
        processor = SWEPostProcessor()
        output = {
            "llm_response": "I'll run this:\n```bash\necho hello\n```"
        }

        result = processor.process(output)

        assert "format_error" not in result

    def test_custom_format_error_template(self):
        """Test custom format error template."""
        processor = SWEPostProcessor(
            format_error_template="Custom error: provide one bash block"
        )
        output = {
            "llm_response": "No bash block here"
        }

        result = processor.process(output)

        assert result["format_error"] is True
        assert "Custom error" in result["error_message"]


class TestTimeoutErrorDetection:
    """Tests for timeout error detection."""

    def test_timeout_error_detected(self):
        """Test timeout error detection."""
        processor = SWEPostProcessor()
        output = {
            "tool_results": [
                {
                    "result": {"status": "timeout"},
                    "params": {"command": "sleep 1000"}
                }
            ]
        }

        result = processor.process(output)

        assert result["timeout_error"] is True
        assert result["needs_retry"] is True
        assert "sleep 1000" in result["error_message"]

    def test_no_timeout_on_success(self):
        """Test no timeout flag on successful execution."""
        processor = SWEPostProcessor()
        output = {
            "tool_results": [
                {
                    "result": {"status": "success", "output": "done"},
                    "params": {"command": "echo hello"}
                }
            ]
        }

        result = processor.process(output)

        assert "timeout_error" not in result

    def test_custom_timeout_template(self):
        """Test custom timeout template."""
        processor = SWEPostProcessor(
            timeout_template="TIMEOUT: {command} took too long"
        )
        output = {
            "tool_results": [
                {
                    "result": {"status": "timeout"},
                    "params": {"command": "make build"}
                }
            ]
        }

        result = processor.process(output)

        assert "TIMEOUT: make build took too long" in result["error_message"]


class TestActionExtraction:
    """Tests for action extraction."""

    def test_extract_single_action(self):
        """Test extracting single action from response."""
        processor = SWEPostProcessor()
        llm_response = "Let me run:\n```bash\nls -la\n```"

        action = processor.extract_action(llm_response)

        assert action is not None
        assert action["action"] == "ls -la"

    def test_extract_action_with_multiline_command(self):
        """Test extracting multiline command."""
        processor = SWEPostProcessor()
        llm_response = "```bash\ncd /tmp && \\\nls -la\n```"

        action = processor.extract_action(llm_response)

        assert action is not None
        assert "cd /tmp" in action["action"]

    def test_extract_action_returns_none_for_no_match(self):
        """Test returns None when no bash block."""
        processor = SWEPostProcessor()
        llm_response = "No bash block here"

        action = processor.extract_action(llm_response)

        assert action is None

    def test_extract_action_returns_none_for_multiple_matches(self):
        """Test returns None for multiple bash blocks."""
        processor = SWEPostProcessor()
        llm_response = "```bash\necho 1\n```\n```bash\necho 2\n```"

        action = processor.extract_action(llm_response)

        assert action is None

    def test_extract_action_strips_whitespace(self):
        """Test action is stripped of whitespace."""
        processor = SWEPostProcessor()
        llm_response = "```bash\n  echo hello  \n```"

        action = processor.extract_action(llm_response)

        assert action is not None
        assert action["action"] == "echo hello"

    def test_custom_action_regex(self):
        """Test custom action regex pattern."""
        processor = SWEPostProcessor(
            action_regex=r"<command>(.*?)</command>"
        )
        llm_response = "Run this: <command>echo hello</command>"

        action = processor.extract_action(llm_response)

        assert action is not None
        assert action["action"] == "echo hello"
