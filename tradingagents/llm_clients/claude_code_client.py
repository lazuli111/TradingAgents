"""Claude through your own Claude subscription, via the Claude Code CLI.

Each model call runs the unmodified ``claude`` binary in print mode
(``claude -p``), signed in with your own Claude Pro/Max account (``claude auth
login``). No OAuth token is read or forwarded here: the CLI authenticates
itself, which is the use Anthropic permits for subscription credentials.

Claude Code is run as a plain model: its own tools, MCP servers and slash
commands are turned off, the system prompt is replaced with the agent's, and
nothing is saved to its session history. LangChain tool calling and structured
output are carried over ``--json-schema``: the model answers with a JSON object
that is turned back into ``tool_calls`` or a Pydantic instance. A model that
calls a bound tool directly instead is caught in the CLI's event stream: the
call is taken as the tool call and the CLI is stopped before it answers "No
such tool available", since the tool only exists on the TradingAgents side.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from .base_client import BaseLLMClient
from .validators import validate_model

# Without these the CLI bills an API key instead of the subscription.
_API_AUTH_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

_DEFAULT_SYSTEM = "You are a helpful assistant."

_TOOL_PROTOCOL = """

# Tools
You can call these tools. Their results come back in the next message.
{tools}

Answer with the JSON object the output schema asks for. To call one or more
tools, list them in "tool_calls" with their arguments and leave "content"
empty. When you need no more tools, leave "tool_calls" empty and put your full
answer in "content"."""


class ClaudeCodeError(RuntimeError):
    """The ``claude`` CLI failed or returned an error result."""


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
        )
    return str(content)


_CONTINUE = "\n\nThe user message holds the conversation so far; continue it as the assistant."


def _transcript(messages: Sequence[BaseMessage]) -> tuple[str, list[str]]:
    """The system prompt, and the conversation as the text blocks of one user message.

    One block per turn: user turns as they are, the assistant's and the tools'
    labelled, with tool calls and their results written out as text. A tool
    loop's next request repeats the last one's system prompt and blocks and adds
    its own, so the prompt cache serves the repeated part instead of it being
    paid for again: the cache matches a prefix only at block boundaries.
    """
    system = "\n\n".join(_text(m.content) for m in messages if isinstance(m, SystemMessage))
    turns = [m for m in messages if not isinstance(m, SystemMessage)]

    parts = []
    for message in turns:
        if isinstance(message, AIMessage):
            said = _text(message.content)
            if message.tool_calls:
                calls = "; ".join(
                    f"{c['name']}({json.dumps(c['args'], default=str)})" for c in message.tool_calls
                )
                said = f"{said}\n[Called {calls}]".strip()
            parts.append(f"<assistant>\n{said}\n</assistant>")
        elif isinstance(message, ToolMessage):
            parts.append(f"<tool_result name=\"{message.name or 'tool'}\">\n"
                         f"{_text(message.content)}\n</tool_result>")
        else:
            parts.append(_text(message.content))
    return (system or _DEFAULT_SYSTEM) + _CONTINUE, parts


def _tool_call_schema(tools: list[dict]) -> dict:
    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": [t["function"]["name"] for t in tools]},
                        "args": {"type": "object"},
                    },
                    "required": ["name", "args"],
                },
            },
            "content": {"type": "string"},
        },
        "required": ["tool_calls", "content"],
    }


def _tool_calls(requested: list) -> list[dict]:
    """LangChain tool calls, each with a fresh id, from ``{"name", "args"}`` requests."""
    return [
        {"name": c["name"], "args": c.get("args") or {}, "id": f"call_{uuid.uuid4().hex[:24]}",
         "type": "tool_call"}
        for c in requested
        if isinstance(c, dict) and c.get("name")
    ]


def _json_from_text(text: str) -> Any:
    """The JSON object in a reply, tolerating a Markdown fence around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


class ChatClaudeCode(BaseChatModel):
    """A LangChain chat model that answers through ``claude -p``."""

    model: str = "sonnet"
    cli_path: str = "claude"
    effort: str | None = None
    timeout: float = 900.0
    max_retries: int = 2

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any):
        if include_raw:
            raise NotImplementedError("include_raw is not supported by claude_code")
        is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        json_schema = schema.model_json_schema() if is_model else dict(schema)
        bound = self.bind(json_schema=json_schema)

        def parse(message: AIMessage):
            data = message.additional_kwargs.get("structured_output")
            if data is None:
                data = _json_from_text(_text(message.content))
            return schema.model_validate(data) if is_model else data

        return bound | RunnableLambda(parse)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tools = kwargs.get("tools") or []
        json_schema = kwargs.get("json_schema")
        system, prompt = _transcript(messages)
        if tools:
            listing = "\n".join(
                f"- {t['function']['name']}: {t['function'].get('description', '')}\n"
                f"  parameters: {json.dumps(t['function'].get('parameters', {}))}"
                for t in tools
            )
            system += _TOOL_PROTOCOL.format(tools=listing)
            json_schema = _tool_call_schema(tools)

        result = self._run(system, prompt, json_schema, {t["function"]["name"] for t in tools})
        text = result.get("result") or ""
        structured = result.get("structured_output")
        if json_schema is not None and structured is None:
            try:
                structured = _json_from_text(text)
            except ValueError:
                structured = None

        if result.get("native_tool_calls"):
            message = AIMessage(content="", tool_calls=_tool_calls(result["native_tool_calls"]))
        elif tools and isinstance(structured, dict):
            calls = _tool_calls(structured.get("tool_calls") or [])
            message = AIMessage(content="" if calls else structured.get("content", ""), tool_calls=calls)
        elif tools:
            message = AIMessage(content=text)
        else:
            extra = {"structured_output": structured} if json_schema is not None else {}
            message = AIMessage(content=text if structured is None else json.dumps(structured),
                                additional_kwargs=extra)
        usage = result.get("usage") or {}
        message.response_metadata = {"model_name": self.model, "usage": usage}
        if usage:
            prompt_tokens = sum(usage.get(k) or 0 for k in (
                "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            output_tokens = usage.get("output_tokens") or 0
            message.usage_metadata = {"input_tokens": prompt_tokens, "output_tokens": output_tokens,
                                      "total_tokens": prompt_tokens + output_tokens}
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _command(self, system_file: str, json_schema: dict | None) -> list[str]:
        command = [
            self.cli_path, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose",
            "--model", self.model,
            "--system-prompt-file", system_file,
            "--tools", "",
            "--disallowedTools", "mcp__*",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if self.effort:
            command += ["--effort", self.effort]
        if json_schema is not None:
            command += ["--json-schema", json.dumps(json_schema)]
        return command

    def _run(self, system: str, prompt: list[str], json_schema: dict | None,
             tool_names: set[str] = frozenset()) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in _API_AUTH_ENV}
        last_error: Exception | None = None
        # An empty working directory keeps a project's CLAUDE.md out of the prompt.
        with tempfile.TemporaryDirectory(prefix="tradingagents-claude-") as workdir:
            system_file = os.path.join(workdir, "system.md")
            with open(system_file, "w", encoding="utf-8") as fh:
                fh.write(system)
            for attempt in range(self.max_retries + 1):
                if attempt:
                    time.sleep(min(2 ** attempt, 30))
                try:
                    return self._run_once(self._command(system_file, json_schema), prompt, env, workdir,
                                          tool_names)
                except ClaudeCodeError as exc:
                    last_error = exc
                except subprocess.TimeoutExpired as exc:
                    last_error = ClaudeCodeError(f"claude -p timed out after {exc.timeout:.0f}s")
        raise last_error

    def _run_once(self, command: list[str], prompt: list[str], env: dict, workdir: str,
                  tool_names: set[str] = frozenset()) -> dict:
        """The CLI's result event, or the bound tools the model called directly.

        Calls the model makes to a bound tool are collected from its message, and
        the CLI is stopped as soon as anything else follows that message.
        """
        proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=env, cwd=workdir,
        )
        timed_out = threading.Event()
        timer = threading.Timer(self.timeout, lambda: (timed_out.set(), proc.kill()))
        timer.start()
        stderr: list[str] = []
        threading.Thread(target=lambda: stderr.append(proc.stderr.read()), daemon=True).start()

        def send_prompt():
            try:
                blocks = [{"type": "text", "text": block} for block in prompt]
                proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": blocks}}) + "\n")
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        threading.Thread(target=send_prompt, daemon=True).start()
        result, other = None, []
        calls, call_message, call_usage = [], None, None
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    other.append(line)
                    continue
                message = event.get("message") or {}
                if calls and not (event.get("type") == "assistant" and message.get("id") == call_message):
                    break
                if event.get("type") == "assistant" and tool_names:
                    found = [
                        {"name": block["name"], "args": block.get("input") or {}}
                        for block in message.get("content") or []
                        if isinstance(block, dict) and block.get("type") == "tool_use"
                        and block.get("name") in tool_names
                    ]
                    if found:
                        calls += found
                        call_message, call_usage = message.get("id"), message.get("usage")
                elif event.get("type") == "result":
                    result = event
        finally:
            if calls and proc.poll() is None:
                proc.kill()
            proc.wait()
            timer.cancel()
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(command, self.timeout)
        if calls:
            return {"native_tool_calls": calls, "usage": call_usage}
        if result is None:
            detail = ("".join(stderr) or "".join(other)).strip()[-2000:]
            raise ClaudeCodeError(f"claude -p exited with {proc.returncode}: {detail}")
        if result.get("is_error") or proc.returncode != 0:
            detail = result.get("result") or result.get("subtype") or "".join(stderr).strip()
            raise ClaudeCodeError(f"claude -p failed: {detail}")
        return result


class ClaudeCodeClient(BaseLLMClient):
    """Claude models billed to your Claude subscription through the Claude Code CLI."""

    provider = "claude_code"

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        cli = os.environ.get("CLAUDE_CODE_CLI_PATH") or shutil.which("claude")
        if not cli:
            raise ClaudeCodeError(
                "The claude_code provider needs the Claude Code CLI. Install it "
                "(https://code.claude.com/docs/en/setup), sign in with `claude auth login`, "
                "or point CLAUDE_CODE_CLI_PATH at the binary."
            )
        llm_kwargs: dict[str, Any] = {"model": self.model, "cli_path": cli}
        for key in ("effort", "timeout", "max_retries"):
            if self.kwargs.get(key) is not None:
                llm_kwargs[key] = self.kwargs[key]
        return ChatClaudeCode(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("claude_code", self.model)
