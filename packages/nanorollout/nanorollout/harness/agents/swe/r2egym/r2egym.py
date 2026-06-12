import json
import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..base import AgentConfig, BaseAgent
from .action import _action_to_bashcmd, _parse_xml_action
from .observation import _format_observation
from .prompts import CONTINUE_MSG, FN_CALLING_SYSTEM_PROMPT, INSTANCE_PROMPT, NON_FN_CALLING_SYSTEM_PROMPT
from .tools import EXECUTE_BASH_TOOL, FILE_EDITOR_TOOL, FINISH_TOOL, SEARCH_TOOL

if TYPE_CHECKING:
    from nanorollout.envs.shell_env.base import ShellEnvironment as BaseEnvironment

logger = logging.getLogger(__name__)


# Tool script installation  (adapted from r2egym/agenthub/environment/env.py)
_SCRIPTS_DIR = Path(__file__).resolve().parent / "r2egym_scripts"

# Map of command name -> script file.  The command name (key) is what
# gets installed as ``/usr/local/bin/<name>`` inside the container.
_TOOL_SCRIPTS = {
    "file_editor": _SCRIPTS_DIR / "file_editor.py",
    "execute_bash": _SCRIPTS_DIR / "execute_bash.py",
    "search": _SCRIPTS_DIR / "search.py",
    "finish": _SCRIPTS_DIR / "finish.py",
}


class R2EGymAgent(BaseAgent):
    """
    R2E-Gym SWE agent ported to TinyFlow's harness.

    Supports two interaction modes:
    * **Function-calling** (default) -- tool schemas are sent via the API
      ``tools`` parameter and the model returns structured ``tool_calls``.
    * **Non-function-calling** -- tool documentation is embedded in the system
      prompt and the model returns XML ``<function=...>`` blocks that are
      parsed with regex.

    Tool scripts are installed in the container at ``/usr/local/bin/`` and
    actions are executed as bash commands, exactly as R2E-Gym does.

    Args:
        environment: A TinyFlow ``BaseEnvironment`` instance.
        config: Optional ``AgentConfig``.
        use_fn_calling: Whether to use the function-calling API path.
            Automatically disabled for models that do not support it.
    """

    def __init__(
        self,
        environment: "BaseEnvironment",
        config: Optional[AgentConfig] = None,
        use_fn_calling: bool = True,
    ):
        super().__init__(environment, config)
        self.use_fn_calling: bool = use_fn_calling
        self._step_count: int = 0
        self._max_context_tokens: int = 65536
        self._tools_installed: bool = False

    @property
    def tools(self) -> list[Any]:
        # We bypass TinyFlow's BaseTool abstraction
        return []

    def get_system_prompt(self) -> str:
        if self.use_fn_calling:
            return FN_CALLING_SYSTEM_PROMPT
        return NON_FN_CALLING_SYSTEM_PROMPT

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Return raw R2E-Gym tool schemas (only used in fn-calling mode)."""
        if not self.use_fn_calling:
            return []
        return [SEARCH_TOOL, FILE_EDITOR_TOOL, EXECUTE_BASH_TOOL, FINISH_TOOL]

    def _install_tools(self) -> None:
        """
        Install R2E-Gym tool scripts into the container.
        """
        if self._tools_installed:
            return

        for cmd_name, script_path in _TOOL_SCRIPTS.items():
            if not script_path.exists():
                logger.warning("Tool script not found: %s (expected at %s)", cmd_name, script_path)
                continue

            script_content = script_path.read_text(encoding="utf-8")
            dest = f"/usr/local/bin/{cmd_name}"

            # Write the script into the container
            self.environment.write_file(dest, script_content)
            # Make it executable
            self.environment.execute(f"chmod +x {shlex.quote(dest)}")

            logger.info("Installed tool script: %s -> %s", cmd_name, dest)

        self._tools_installed = True

    def _init_messages(self, task: str) -> None:
        # Install tool scripts before the first turn
        self._install_tools()

        system_prompt = self.get_system_prompt()
        instance_prompt = INSTANCE_PROMPT.format(problem_statement=task)
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
        ]
        self._step_count = 0

    def _needs_continuation(self) -> bool:
        return False

    def _parse_fn_calling_response(self, response: Any) -> tuple[str, dict[str, Any]]:
        """Extract (function_name, parameters) from a tool-call response."""
        message = response.choices[0].message
        try:
            tc = message.tool_calls[0]
            function_name = tc.function.name
            parameters = json.loads(tc.function.arguments)
        except Exception:
            function_name = ""
            parameters = {}

        return function_name, parameters

    def _parse_xml_response(self, response: Any) -> tuple[str, dict[str, str]]:
        """Extract (function_name, parameters) from an XML text response."""
        content = response.choices[0].message.content or ""
        pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
        match_action = pattern_action.search(content)

        if match_action:
            action = match_action.group(1)  # The entire <function=...></function> block
        else:
            action = ""

        if action:
            function_name, parameters = _parse_xml_action(action)
        else:
            function_name, parameters = "", {}

        return function_name, parameters

    _BLOCKED_CMDS = frozenset({"kill", "pkill", "killall", "skill", "xkill"})

    def _execute_action(self, function_name: str, parameters: dict[str, Any]) -> tuple[str, int]:
        """
        Execute an action and return ``(output_text, exit_code)``.
        """
        if function_name in ("finish", "submit"):
            self.finished = True
            self.finish_message = parameters.get("result", "Task completed.")
            return "<<< Finished >>>", 0

        # Block dangerous process-killing commands before they reach the environment
        if function_name == "execute_bash":
            raw_cmd = parameters.get("cmd", "") or parameters.get("command", "") or ""
            if self._BLOCKED_CMDS & set(raw_cmd.split()):
                return (
                    "ERROR: Do not try to kill processes in this environment " "as it will affect other tasks.",
                    1,
                )

        bash_cmd = _action_to_bashcmd(function_name, parameters)
        if not bash_cmd:
            return "", 0

        result = self.environment.execute(bash_cmd)
        return result.output, result.exit_code

    def _step(self) -> bool:
        """
        Execute one agent step (R2E-Gym style):

        1. Append step-count reminder to last message
        2. Call LLM
        3. Parse response (fn-calling or XML)
        4. Execute action in environment
        5. Format observation and update conversation history
        """
        self._step_count += 1

        # --- 1. Step budget message ---
        steps_remaining = self.config.max_iterations - self._step_count
        if steps_remaining > 0:
            step_msg = f"\nSteps Remaining: {steps_remaining}"
        else:
            step_msg = "\nYou have reached the maximum number of steps. Please submit your answer NOW."
        self.messages[-1]["content"] += step_msg

        # --- 2. Call LLM ---
        try:
            response = self._call_llm()
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return False

        # --- 3. Parse response ---
        assistant_message = response.choices[0].message
        if self.use_fn_calling:
            fn_name, params = self._parse_fn_calling_response(response)
        else:
            fn_name, params = self._parse_xml_response(response)

        if hasattr(assistant_message, "reasoning_content"):
            logger.info("REASONING: %s", assistant_message.reasoning_content)
        logger.info("CONTENT: %s", assistant_message.content)
        logger.info("ACTION: %s(%s)", fn_name, params)

        # --- 4. Execute action ---
        try:
            output, exit_code = self._execute_action(fn_name, params)
        except Exception as exc:
            output = str(exc)
            exit_code = 1
            logger.error("Action execution error: %s", exc)

        if self.finished:
            # Still update history so the trajectory is complete
            self._append_history(response, fn_name, params, output, exit_code)
            return False

        # Handle null / missing action
        if not fn_name:
            output = CONTINUE_MSG
            exit_code = 0

        # --- 5. Update conversation history ---
        self._append_history(response, fn_name, params, output, exit_code)

        logger.info("OBSERVATION: %s", output[:500])
        return True

    def _append_history(
        self,
        response: Any,
        fn_name: str,
        params: dict[str, Any],
        output: str,
        exit_code: int,
    ) -> None:
        """Append assistant + observation messages to conversation history."""
        observation_text = _format_observation(fn_name, output, exit_code)

        if self.use_fn_calling:
            # Append the raw assistant message (preserves tool_calls)
            assistant_msg = response.choices[0].message
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.content,
            }
            if hasattr(assistant_msg, "reasoning_content"):
                msg_dict["reasoning_content"] = assistant_msg.reasoning_content
            if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
                # Keep only the first tool call (R2E-Gym convention)
                tc = assistant_msg.tool_calls[0]
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                ]
            self.messages.append(msg_dict)

            # Append tool response
            if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
                tc = assistant_msg.tool_calls[0]
                self.messages.append(
                    {
                        "role": "tool",
                        "content": observation_text,
                        "name": tc.function.name,
                        "tool_call_id": tc.id,
                    }
                )
            else:
                # Fallback: if no tool_calls, send as user message
                self.messages.append({"role": "user", "content": observation_text})
        else:
            assistant_msg = response.choices[0].message
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.content,
            }
            if hasattr(assistant_msg, "reasoning_content"):
                msg_dict["reasoning_content"] = assistant_msg.reasoning_content
            self.messages.append(msg_dict)
            self.messages.append({"role": "user", "content": observation_text})
