"""The claude_code provider: Claude through the user's subscription via ``claude -p``.

A stand-in ``claude`` script records how it was called and answers with what
the test queued, so these run without the real CLI or an account.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.claude_code_client import (
    ChatClaudeCode,
    ClaudeCodeClient,
    ClaudeCodeError,
)
from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client
from tradingagents.llm_clients.validators import validate_model

FAKE_CLI = textwrap.dedent("""\
    #!{python}
    import json, os, sys
    args = sys.argv[1:]
    system = open(args[args.index("--system-prompt-file") + 1], encoding="utf-8").read()
    sent = json.loads(sys.stdin.readline())["message"]["content"]
    record = {{"args": args, "blocks": [b["text"] for b in sent], "stdin": "\\n\\n".join(b["text"] for b in sent),
              "system": system, "cwd": os.getcwd(),
              "api_key": os.environ.get("ANTHROPIC_API_KEY")}}
    with open(os.environ["FAKE_CLAUDE_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\\n")
    replies = json.load(open(os.environ["FAKE_CLAUDE_REPLIES"], encoding="utf-8"))
    reply = replies.pop(0)
    json.dump(replies, open(os.environ["FAKE_CLAUDE_REPLIES"], "w", encoding="utf-8"))
    for event in reply.pop("_events", []):
        print(json.dumps(event), flush=True)
    if reply.pop("_hang", False):
        import time
        time.sleep(30)
    code = reply.pop("_exit", 0)
    if reply:
        print(json.dumps(reply), flush=True)
    sys.exit(code)
""")


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    cli = tmp_path / "claude"
    cli.write_text(FAKE_CLI.format(python=sys.executable))
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    log, replies = tmp_path / "calls.jsonl", tmp_path / "replies.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_REPLIES", str(replies))

    class Fake:
        path = str(cli)

        def reply(self, *items):
            replies.write_text(json.dumps(list(items)))

        def calls(self):
            return [json.loads(line) for line in log.read_text().splitlines()]

    return Fake()


def ok(result="", **extra):
    return {"type": "result", "subtype": "success", "is_error": False, "result": result, **extra}


def llm(fake, **kwargs):
    return ChatClaudeCode(cli_path=fake.path, max_retries=0, **kwargs)


def test_plain_call_runs_the_cli_as_a_bare_model(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-reach-the-cli")
    fake_cli.reply(ok("Buy.", usage={"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5}))

    reply = llm(fake_cli, model="opus", effort="high").invoke(
        [SystemMessage(content="You are a trader."), HumanMessage(content="NVDA?")]
    )

    assert reply.content == "Buy."
    assert reply.usage_metadata == {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}
    call = fake_cli.calls()[0]
    args = call["args"]
    assert args[:6] == ["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]
    assert args[args.index("--model") + 1] == "opus"
    assert args[args.index("--effort") + 1] == "high"
    assert args[args.index("--tools") + 1] == ""
    assert "--no-session-persistence" in args and "--strict-mcp-config" in args
    assert "--json-schema" not in args
    assert call["system"].startswith("You are a trader.\n\nThe user message holds the conversation")
    assert call["stdin"] == "NVDA?"
    assert call["api_key"] is None  # billed to the subscription, not the key
    assert os.path.basename(call["cwd"]).startswith("tradingagents-claude-")


@tool
def get_stock_data(symbol: str, start_date: str) -> str:
    """Price history for a ticker."""
    return "..."


def test_bound_tools_come_back_as_tool_calls(fake_cli):
    fake_cli.reply(ok("", structured_output={
        "tool_calls": [{"name": "get_stock_data", "args": {"symbol": "NVDA", "start_date": "2026-09-01"}}],
        "content": "",
    }))

    reply = llm(fake_cli).bind_tools([get_stock_data]).invoke([HumanMessage(content="Analyse NVDA")])

    assert [c["name"] for c in reply.tool_calls] == ["get_stock_data"]
    assert reply.tool_calls[0]["args"] == {"symbol": "NVDA", "start_date": "2026-09-01"}
    assert reply.tool_calls[0]["id"]
    call = fake_cli.calls()[0]
    schema = json.loads(call["args"][call["args"].index("--json-schema") + 1])
    assert schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == ["get_stock_data"]
    assert "get_stock_data: Price history for a ticker." in call["system"]


def _assistant(msg_id, *blocks):
    return {"type": "assistant", "message": {"id": msg_id, "content": list(blocks), "usage": {"output_tokens": 7}}}


def _tool_use(name, **args):
    return {"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": args}


def test_a_bound_tool_called_directly_is_taken_and_the_cli_stopped(fake_cli):
    """Called directly, the tool would fail in Claude Code; the call is taken before it can."""
    fake_cli.reply({"_events": [
        _assistant("m1", {"type": "text", "text": "Let me fetch prices."}),
        _assistant("m1", _tool_use("get_stock_data", symbol="NVDA", start_date="2026-09-01")),
        _assistant("m1", _tool_use("get_stock_data", symbol="AMD", start_date="2026-09-01")),
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "No such tool available"}]}},
    ], "_hang": True})

    started = time.monotonic()
    reply = llm(fake_cli).bind_tools([get_stock_data]).invoke("Analyse NVDA and AMD")

    assert time.monotonic() - started < 10  # stopped, not waited out
    assert [c["args"]["symbol"] for c in reply.tool_calls] == ["NVDA", "AMD"]
    assert reply.content == "" and all(c["id"] for c in reply.tool_calls)


def test_the_cli_own_structured_output_tool_is_not_taken_as_a_call(fake_cli):
    fake_cli.reply({"_events": [
        _assistant("m1", _tool_use("StructuredOutput", tool_calls=[], content="Report.")),
    ], **ok("", structured_output={"tool_calls": [], "content": "Report."})})

    reply = llm(fake_cli).bind_tools([get_stock_data]).invoke("Write it up")

    assert reply.tool_calls == [] and reply.content == "Report."


def test_bound_tools_with_no_call_return_the_answer(fake_cli):
    fake_cli.reply(ok("", structured_output={"tool_calls": [], "content": "Final report."}))

    reply = llm(fake_cli).bind_tools([get_stock_data]).invoke("Write it up")

    assert reply.tool_calls == [] and reply.content == "Final report."


def test_tool_history_is_written_out_for_the_next_turn(fake_cli):
    fake_cli.reply(ok("", structured_output={"tool_calls": [], "content": "done"}))
    history = [
        HumanMessage(content="Analyse NVDA"),
        AIMessage(content="", tool_calls=[{"name": "get_stock_data", "args": {"symbol": "NVDA"}, "id": "c1"}]),
        ToolMessage(content="close 181.2", tool_call_id="c1", name="get_stock_data"),
    ]

    llm(fake_cli).bind_tools([get_stock_data]).invoke(history)

    call = fake_cli.calls()[0]
    prompt = call["stdin"]
    # One block per turn, so the next round's request starts with these same blocks.
    assert call["blocks"][0] == "Analyse NVDA" and len(call["blocks"]) == 3
    assert "continue it as the assistant" in call["system"]
    assert prompt.startswith("Analyse NVDA")
    assert '[Called get_stock_data({"symbol": "NVDA"})]' in prompt
    assert '<tool_result name="get_stock_data">\nclose 181.2\n</tool_result>' in prompt


class Decision(BaseModel):
    rating: str
    reason: str


def test_structured_output_parses_into_the_schema(fake_cli):
    fake_cli.reply(ok("", structured_output={"rating": "Buy", "reason": "momentum"}))

    result = llm(fake_cli).with_structured_output(Decision).invoke("Decide")

    assert result == Decision(rating="Buy", reason="momentum")
    args = fake_cli.calls()[0]["args"]
    assert json.loads(args[args.index("--json-schema") + 1])["title"] == "Decision"


def test_structured_output_falls_back_to_json_in_the_text(fake_cli):
    fake_cli.reply(ok('```json\n{"rating": "Hold", "reason": "range-bound"}\n```'))

    assert llm(fake_cli).with_structured_output(Decision).invoke("Decide").rating == "Hold"


def test_cli_errors_are_raised_with_the_reason(fake_cli):
    fake_cli.reply({"type": "result", "subtype": "success", "is_error": True,
                    "result": "Not logged in · Please run /login", "_exit": 1})

    with pytest.raises(ClaudeCodeError, match="Not logged in"):
        llm(fake_cli).invoke("hi")


def test_a_failed_call_is_retried(fake_cli, monkeypatch):
    monkeypatch.setattr("tradingagents.llm_clients.claude_code_client.time.sleep", lambda _: None)
    fake_cli.reply({"is_error": True, "result": "overloaded", "_exit": 1}, ok("second time"))

    assert llm(fake_cli).model_copy(update={"max_retries": 1}).invoke("hi").content == "second time"
    assert len(fake_cli.calls()) == 2


def test_factory_builds_the_client_with_effort(fake_cli, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_CLI_PATH", fake_cli.path)
    kwargs = build_llm_kwargs({"llm_provider": "claude_code", "anthropic_effort": "medium"})
    client = create_llm_client("claude_code", "sonnet", **kwargs)

    assert isinstance(client, ClaudeCodeClient)
    model = client.get_llm()
    assert (model.cli_path, model.model, model.effort) == (fake_cli.path, "sonnet", "medium")


def test_a_missing_cli_says_how_to_install_it(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.setattr("tradingagents.llm_clients.claude_code_client.shutil.which", lambda _: None)

    with pytest.raises(ClaudeCodeError, match="claude auth login"):
        create_llm_client("claude_code", "sonnet").get_llm()


def test_needs_no_api_key_and_takes_any_model_id():
    assert get_api_key_env("claude_code") is None
    assert validate_model("claude_code", "claude-opus-5-5")
    assert validate_model("claude_code", "sonnet")
