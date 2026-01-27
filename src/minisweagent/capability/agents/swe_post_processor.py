"""Post-processor for SWE agent completion detection.

This module provides the post-processing logic that detects when the agent
has completed its task, based on completion markers in tool output.

The original mini-swe-agent used exceptions (Submitted, LimitsExceeded) to
signal completion. In the pipeline model, we instead set output fields that
trigger routing rules.
"""

from typing import Any, Dict, List, Optional
import re


# Completion markers that signal the agent has finished
COMPLETION_MARKERS = frozenset([
    "MINI_SWE_AGENT_FINAL_OUTPUT",
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
])


class SWEPostProcessor:
    """Post-processor for SWE agent outputs.

    Handles:
    - Completion detection via markers in tool output
    - Format error detection and message generation
    - Timeout handling
    - Limits exceeded detection
    """

    def __init__(
        self,
        action_regex: str = r"```bash\s*\n(.*?)\n```",
        format_error_template: Optional[str] = None,
        timeout_template: Optional[str] = None,
    ):
        """Initialize the post-processor.

        Args:
            action_regex: Regex to extract bash commands from LLM output
            format_error_template: Template for format error messages
            timeout_template: Template for timeout error messages
        """
        self._action_regex = re.compile(action_regex, re.DOTALL)
        self._format_error_template = format_error_template or (
            "Please provide EXACTLY ONE bash command in a code block.\n"
            "Example:\n```bash\necho hello\n```"
        )
        self._timeout_template = timeout_template or (
            "Command timed out: {command}\n"
            "Consider breaking into smaller steps or increasing timeout."
        )

    def process(
        self,
        output: Dict[str, Any],
        llm_call_count: int = 0,
        max_llm_calls: int = 0,
        agent_hop_count: int = 0,
        max_agent_hops: int = 0,
    ) -> Dict[str, Any]:
        """Process agent output and detect completion conditions.

        Args:
            output: The agent's output dictionary
            llm_call_count: Current number of LLM calls
            max_llm_calls: Maximum allowed LLM calls
            agent_hop_count: Current number of agent hops
            max_agent_hops: Maximum allowed agent hops

        Returns:
            Updated output dictionary with completion/error flags
        """
        # Check for limits exceeded
        if max_llm_calls > 0 and llm_call_count >= max_llm_calls:
            output["limits_exceeded"] = True
            output["error_message"] = f"LLM call limit exceeded: {llm_call_count}/{max_llm_calls}"
            return output

        if max_agent_hops > 0 and agent_hop_count >= max_agent_hops:
            output["limits_exceeded"] = True
            output["error_message"] = f"Agent hop limit exceeded: {agent_hop_count}/{max_agent_hops}"
            return output

        # Check for completion markers in tool results
        if self._check_completion_markers(output):
            return output

        # Check for format errors in LLM response
        if self._check_format_errors(output):
            return output

        # Check for timeout errors
        if self._check_timeout_errors(output):
            return output

        return output

    def _check_completion_markers(self, output: Dict[str, Any]) -> bool:
        """Check for completion markers in tool output.

        Sets output["completed"] = True and output["final_message"] if found.

        Returns:
            True if completion marker was found
        """
        tool_results = output.get("tool_results", [])

        for result in tool_results:
            tool_output = result.get("result", {}).get("output", "")
            if not tool_output:
                continue

            lines = tool_output.lstrip().splitlines()
            if lines and lines[0].strip() in COMPLETION_MARKERS:
                output["completed"] = True
                output["final_message"] = "\n".join(lines[1:]).strip()
                return True

        return False

    def _check_format_errors(self, output: Dict[str, Any]) -> bool:
        """Check for format errors in LLM response.

        Sets output["format_error"] = True and output["error_message"] if found.

        Returns:
            True if format error was detected
        """
        llm_response = output.get("llm_response", "")
        if not llm_response:
            return False

        # Try to extract bash command
        matches = self._action_regex.findall(llm_response)

        if len(matches) == 0:
            output["format_error"] = True
            output["error_message"] = self._format_error_template
            output["needs_retry"] = True
            return True

        if len(matches) > 1:
            output["format_error"] = True
            output["error_message"] = (
                f"Found {len(matches)} bash blocks, expected exactly 1.\n"
                + self._format_error_template
            )
            output["needs_retry"] = True
            return True

        return False

    def _check_timeout_errors(self, output: Dict[str, Any]) -> bool:
        """Check for timeout errors in tool results.

        Sets output["timeout_error"] = True and output["error_message"] if found.

        Returns:
            True if timeout error was detected
        """
        tool_results = output.get("tool_results", [])

        for result in tool_results:
            if result.get("result", {}).get("status") == "timeout":
                command = result.get("params", {}).get("command", "unknown")
                output["timeout_error"] = True
                output["error_message"] = self._timeout_template.format(command=command)
                output["needs_retry"] = True
                return True

        return False

    def extract_action(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Extract action from LLM response.

        Args:
            llm_response: The raw LLM response text

        Returns:
            Action dict with "action" key, or None if no valid action
        """
        matches = self._action_regex.findall(llm_response)

        if len(matches) != 1:
            return None

        return {"action": matches[0].strip()}


def create_post_processor(
    action_regex: str = r"```bash\s*\n(.*?)\n```",
    format_error_template: Optional[str] = None,
    timeout_template: Optional[str] = None,
) -> SWEPostProcessor:
    """Factory function to create a post-processor.

    Args:
        action_regex: Regex to extract bash commands
        format_error_template: Template for format errors
        timeout_template: Template for timeout errors

    Returns:
        SWEPostProcessor instance
    """
    return SWEPostProcessor(
        action_regex=action_regex,
        format_error_template=format_error_template,
        timeout_template=timeout_template,
    )


__all__ = [
    "SWEPostProcessor",
    "create_post_processor",
    "COMPLETION_MARKERS",
]
