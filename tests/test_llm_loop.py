"""The Claude tool-calling loop, driven by a scripted fake client (no API key or network)."""
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from src.agents import agent, llm


def _tool_use(i, name="get_region_info", args=None):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name=name, input=args or {})


def _text(t):
    return SimpleNamespace(type="text", text=t)


class FakeClient:
    """Returns the scripted responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw | {"messages": list(kw["messages"])})  # the loop mutates the list
        return self.responses.pop(0)


@pytest.fixture
def fake(monkeypatch):
    def install(responses):
        fc = FakeClient(responses)
        monkeypatch.setattr(agent, "client", lambda: fc)
        monkeypatch.setattr(agent, "has_llm", lambda: True)
        return fc
    return install


def test_llm_report_runs_tools_then_returns_text(fake):
    fc = fake([SimpleNamespace(stop_reason="tool_use", content=[_tool_use(0)]),
               SimpleNamespace(stop_reason="end_turn", content=[_text("# Report")])])
    r = agent.run_agent("assess downtown")
    assert r["mode"] == "llm_agent" and r["report_markdown"] == "# Report"
    assert [t["tool"] for t in r["tool_log"]] == ["get_region_info"]
    result = fc.requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "tu_0" and result["is_error"] is False


def test_llm_report_caps_tool_calls_not_turns(fake):
    # One turn asks for 12 tools at once: only MAX_TOOL_CALLS run, the rest are refused.
    many = [_tool_use(i) for i in range(12)]
    fc = fake([SimpleNamespace(stop_reason="tool_use", content=many),
               SimpleNamespace(stop_reason="end_turn", content=[_text("done")])])
    r = agent.llm_report("q")
    assert len(r["tool_log"]) == agent.MAX_TOOL_CALLS
    results = fc.requests[1]["messages"][-1]["content"]
    assert len(results) == 12  # every tool_use still gets a tool_result, as the API requires
    assert sum(x["is_error"] for x in results) == 12 - agent.MAX_TOOL_CALLS
    assert r["mode"] == "llm_agent"


def test_llm_report_truncates_if_model_ignores_budget(fake):
    loop = [SimpleNamespace(stop_reason="tool_use", content=[_tool_use(i)]) for i in range(20)]
    fc = fake(loop)
    r = agent.llm_report("q")
    assert r["mode"] == "llm_agent_truncated"
    assert len(r["tool_log"]) == agent.MAX_TOOL_CALLS
    assert len(fc.requests) == agent.MAX_TOOL_CALLS + 2  # 8 executed, 1 refused, then stop


def test_llm_report_refuses_unlisted_tool(fake):
    fc = fake([SimpleNamespace(stop_reason="tool_use", content=[_tool_use(0, name="run_shell")]),
               SimpleNamespace(stop_reason="end_turn", content=[_text("ok")])])
    r = agent.llm_report("q")
    assert r["tool_log"] == []
    assert fc.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_run_agent_falls_back_when_llm_api_fails(monkeypatch):
    class Down:
        @property
        def messages(self):
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))

    monkeypatch.setattr(agent, "client", lambda: Down())
    monkeypatch.setattr(agent, "has_llm", lambda: True)
    r = agent.run_agent("Assess flood risk downtown within 1km")
    assert r["mode"] == "deterministic_fallback_llm_error"
    assert "LLM call failed: APIConnectionError" in r["report_markdown"]
    assert "## Exposed Assets" in r["report_markdown"]


def test_client_has_bounded_timeout(monkeypatch):
    monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    c = llm.client()
    assert c.timeout == llm.LLM_TIMEOUT_S
    assert c.max_retries == 2
