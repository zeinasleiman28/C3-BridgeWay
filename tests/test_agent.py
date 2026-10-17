"""
Offline tests for the Bridgeway assistant.

They use FakeLLM (tests/fake_llm.py) instead of a real model, so they need no API key
and no internet. Run them from the project folder with:

    py -m pytest
"""

import json
import re
from pathlib import Path

import httpx
import openai
import pytest

import agent
import knowledge
import router
import tools
from business_data import DISCOUNTS, PRICES, best_discount_percent, discounted_price
from fake_llm import FakeLLM, tool_call
from reflection import make_reflector

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def simple_prompt(route_hint=None):
    """A stand-in for build_system_prompt() so agent tests don't depend on the documents."""
    return "SYSTEM PROMPT\n" + (route_hint or "")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def test_record_customer_interest_saves_lead(temp_logs):
    result = tools.record_customer_interest(" maya@greensteps.org ", " Maya ", "GreenSteps – volunteer drive in March")
    assert result["status"] == "recorded"

    [entry] = read_jsonl(temp_logs / "leads.jsonl")
    assert entry["id"] == result["id"]
    assert (entry["name"], entry["email"]) == ("Maya", "maya@greensteps.org")
    assert entry["message"] == "GreenSteps – volunteer drive in March"
    assert "T" in entry["timestamp"]  # ISO 8601


@pytest.mark.parametrize("email, name, expected", [
    ("maya-at-greensteps", "Maya", "valid email"),
    ("", "Maya", "Missing email"),
    ("maya@greensteps.org", "  ", "Missing name"),
])
def test_record_customer_interest_rejects_bad_input(temp_logs, email, name, expected):
    result = tools.record_customer_interest(email, name, "Volunteer drive")
    assert result["status"] == "error" and expected in result["reason"]
    assert not (temp_logs / "leads.jsonl").exists()


def test_record_feedback(temp_logs):
    assert tools.record_feedback("Can you write our EU grant proposal?")["status"] == "recorded"
    assert read_jsonl(temp_logs / "feedback.jsonl")[0]["question"] == "Can you write our EU grant proposal?"
    assert tools.record_feedback("   ")["status"] == "error"


@pytest.mark.parametrize("name, arguments, expected", [
    ("book_campaign", "{}", "Unknown tool"),
    ("record_feedback", "not json", "not valid JSON"),
    ("record_feedback", "[1, 2]", "JSON object"),
    ("record_feedback", "{}", "Wrong arguments"),
])
def test_run_tool_errors_never_raise(name, arguments, expected):
    result = tools.run_tool(name, arguments)
    assert result["status"] == "error" and expected in result["reason"]


def test_run_tool_ignores_extra_arguments():
    args = '{"email": "a@b.co", "name": "Ann", "message": "Donor campaign", "phone": "123"}'
    assert tools.run_tool("record_customer_interest", args)["status"] == "recorded"


def test_tool_definitions_match_registry():
    mcp_names = [t["name"] for t in tools.MCP_TOOL_DEFINITIONS]
    openai_names = [t["function"]["name"] for t in tools.OPENAI_TOOLS]
    assert mcp_names == openai_names == list(tools.TOOL_REGISTRY)
    for t in tools.MCP_TOOL_DEFINITIONS:
        assert t["description"] and t["inputSchema"]["type"] == "object"
        assert set(t["inputSchema"]["required"]) <= set(t["inputSchema"]["properties"])


def test_tool_descriptions_say_when_to_call_them():
    descriptions = {t["name"]: t["description"] for t in tools.MCP_TOOL_DEFINITIONS}
    assert "name" in descriptions["record_customer_interest"] and "email" in descriptions["record_customer_interest"]
    assert "does not offer" in descriptions["record_feedback"] and "feedback" in descriptions["record_feedback"]


# ---------------------------------------------------------------------------
# Business data, documents and system prompt
# ---------------------------------------------------------------------------

def all_prices(prices):
    for value in prices.values():
        yield from all_prices(value) if isinstance(value, dict) else [value]


def test_every_price_in_business_data_appears_in_the_pdf():
    profile = knowledge.load_business_profile()
    prices = list(all_prices(PRICES))
    assert len(prices) == 6  # 3 campaigns + grant tracking + 2 add-ons
    for price in prices:
        assert f"${price}" in profile, f"${price} missing from the PDF"


def test_every_price_and_discount_is_in_the_summary():
    summary = knowledge.load_business_summary()
    summary_numbers = set(re.findall(r"\d+", summary))
    for price in all_prices(PRICES):
        assert str(price) in summary_numbers, f"{price} missing from the summary"
    for rule in DISCOUNTS.values():
        assert f"{rule['percent']}% off" in summary


def test_discount_rules_are_in_the_pdf():
    profile = knowledge.load_business_profile()
    assert f"{DISCOUNTS['small_ngo']['percent']}% off for small NGOs" in profile
    assert f"{DISCOUNTS['bundle']['percent']}% off when booking {DISCOUNTS['bundle']['min_services']} or more" in profile
    assert "do not stack" in profile


@pytest.mark.parametrize("small_ngo, services, percent", [
    (False, 1, 0),
    (True, 1, 20),
    (False, 2, 10),
    (True, 3, 20),   # both apply -> no stacking, the bigger one wins
])
def test_best_discount_never_stacks(small_ngo, services, percent):
    assert best_discount_percent(small_ngo, services) == percent


def test_donor_campaign_for_small_ngo_costs_480():
    assert discounted_price(PRICES["campaigns"]["donor_outreach_campaign"], small_ngo=True) == 480


def test_compose_system_prompt_order():
    prompt = knowledge.compose_system_prompt("PERSONA", "INSTRUCTIONS", "SUMMARY", "PROFILE", "HINT")
    positions = [prompt.index(part) for part in ("PERSONA", "INSTRUCTIONS", "SUMMARY", "PROFILE", "HINT")]
    assert positions == sorted(positions)
    assert "Note for this message" not in knowledge.compose_system_prompt("P", "I", "S", "B")


def test_missing_documents_raise_clear_errors(tmp_path):
    with pytest.raises(knowledge.KnowledgeError, match="not found"):
        knowledge.load_business_summary(tmp_path / "nope.txt")
    with pytest.raises(knowledge.KnowledgeError, match="make_pdf"):
        knowledge.load_business_profile(tmp_path / "nope.pdf")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message, intent", [
    ("What are your opening hours?", "general_question"),
    ("How much is a donor campaign for a small NGO?", "general_question"),
    ("Can you write our EU grant proposal?", "general_question"),
    ("We'd like to book a volunteer drive", "lead_collection"),
    ("I'm Maya from GreenSteps, maya@greensteps.org, we need volunteers in March", "lead_collection"),
    ("Sami, sami@ngo.org", "lead_collection"),
    ("Leila did an amazing job on our emails!", "feedback"),
    ("The results report was late, I'm disappointed.", "feedback"),
])
def test_keyword_router(message, intent):
    assert router.classify_intent(message)["intent"] == intent


def test_llm_router_uses_model_answer(temp_logs):
    fake = FakeLLM(intent="feedback")
    result = router.classify_intent("hello", client=fake, model="m")
    assert (result["intent"], result["source"]) == ("feedback", "llm")
    assert fake.calls[0]["temperature"] == 0
    assert read_jsonl(temp_logs / "router.jsonl")[0]["intent"] == "feedback"


@pytest.mark.parametrize("reply", ["not json", '{"intent": "chitchat"}', "", None])
def test_router_falls_back_on_bad_model_output(reply):
    assert router.classify_intent("I want to book", client=_router_stub(reply), model="m")["intent"] == "lead_collection"


def test_parse_router_reply_accepts_code_fences():
    result = router.parse_router_reply('```json\n{"intent": "Feedback", "confidence": 2}\n```')
    assert result["intent"] == "feedback" and result["confidence"] == 1.0


def _router_stub(reply):
    fake = FakeLLM()
    fake.create = lambda **kw: _text_response(reply)
    return fake


def _text_response(text):
    from types import SimpleNamespace
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def test_lead_flow_calls_record_customer_interest(temp_logs):
    fake = FakeLLM("lead_collection", [
        tool_call("record_customer_interest",
                  {"email": "maya@greensteps.org", "name": "Maya", "message": "GreenSteps – volunteer drive for March"}),
        "Thanks Maya! Our team will email you to schedule the discovery call.",
    ])
    history = [{"role": "user", "content": "We need volunteers for a tree-planting day in March."},
               {"role": "assistant", "content": "Happy to help! What's your name and email?"}]
    result = agent.run_agent("I'm Maya from GreenSteps, maya@greensteps.org", history, fake, "m", simple_prompt)

    assert result["route"]["intent"] == "lead_collection"
    assert [c["name"] for c in result["tool_calls"]] == ["record_customer_interest"]
    assert result["reply"] == "Thanks Maya! Our team will email you to schedule the discovery call."
    assert read_jsonl(temp_logs / "leads.jsonl")[0]["email"] == "maya@greensteps.org"

    # The model saw: system prompt with the lead hint, the history, the new message, then the tool result.
    sent = fake.agent_calls()[1]["messages"]
    assert router.ROUTE_HINTS["lead_collection"] in sent[0]["content"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user", "assistant", "tool"]
    assert sent[4]["tool_calls"][0]["id"] == sent[5]["tool_call_id"]
    assert json.loads(sent[5]["content"])["status"] == "recorded"


def test_service_not_offered_is_recorded_as_feedback(temp_logs):
    fake = FakeLLM("general_question", [
        tool_call("record_feedback", {"question": "Can you write our EU grant proposal?"}),
        "Sorry, we don't write full grant proposals. I've passed your question to our team.",
    ])
    result = agent.run_agent("Can you write our EU grant proposal?", [], fake, "m", simple_prompt)
    assert result["tool_calls"][0]["result"]["status"] == "recorded"
    assert read_jsonl(temp_logs / "feedback.jsonl")[0]["question"] == "Can you write our EU grant proposal?"


def test_plain_answer_without_tools():
    fake = FakeLLM("general_question", ["For a small NGO it's 600 minus 20% = 480 USD."])
    result = agent.run_agent("How much is a donor campaign for a small NGO?", [], fake, "m", simple_prompt)
    assert result["reply"] == "For a small NGO it's 600 minus 20% = 480 USD." and result["tool_calls"] == []


def test_tool_loop_is_capped():
    fake = FakeLLM("general_question", [tool_call("record_feedback", {"question": "q"})] * 10)
    result = agent.run_agent("?", [], fake, "m", simple_prompt)
    assert len(result["tool_calls"]) == agent.MAX_TOOL_ROUNDS
    assert fake.agent_calls()[-1]["tool_choice"] == "none"


def test_history_is_trimmed_and_cleaned():
    history = [{"role": "user", "content": f"message {i}"} for i in range(100)]
    history += [{"role": "tool", "content": "ignored"}, {"role": "assistant", "content": [{"type": "text", "text": "parts"}]}]
    fake = FakeLLM(replies=["ok"])
    agent.run_agent("hi", history, fake, "m", simple_prompt)
    sent = fake.agent_calls()[0]["messages"]
    assert len(sent) == agent.MAX_HISTORY_MESSAGES + 2      # + system prompt + new message
    assert sent[-2] == {"role": "assistant", "content": "parts"}
    assert all(m["role"] != "tool" for m in sent)


def _http_error(cls, status):
    request = httpx.Request("POST", "https://example.com")
    return cls("error", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize("error, words", [
    (_http_error(openai.AuthenticationError, 401), "API key"),
    (_http_error(openai.RateLimitError, 429), "busy"),
    (openai.APIConnectionError(request=httpx.Request("POST", "https://example.com")), "can't reach"),
    (RuntimeError("boom"), "went wrong"),
])
def test_api_errors_become_friendly_messages(error, words):
    result = agent.run_agent("hi", [], FakeLLM(error=error), "m", simple_prompt)
    assert words in result["reply"] and result["error"]


def test_missing_client_and_empty_message():
    assert ".env" in agent.run_agent("hi", [], None, None, simple_prompt)["reply"]
    assert "empty" in agent.run_agent("   ", [], FakeLLM(), "m", simple_prompt)["reply"]


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("review, expected_status, uses_revision", [
    ('{"ok": true}', "approved", False),
    ('{"ok": false, "problems": ["promised an amount raised", "wrong price"], '
     '"revised": "The donor outreach campaign is $600. We never promise amounts raised."}',
     "revised", True),
    ("not json", "skipped", False),
    ('{"ok": false, "problems": ["x"]}', "skipped", False),
])
def test_reflection(temp_logs, review, expected_status, uses_revision):
    draft = "We'll raise $20,000 for you! The donor outreach campaign is $500."
    fake = FakeLLM("general_question", [draft], reflection=review)
    reflect = make_reflector(fake, "m", "RULES", "FACTS")
    result = agent.run_agent("How much is a donor campaign?", [], fake, "m", simple_prompt, reflect=reflect)

    assert result["draft"] == draft
    assert result["reply"] == (
        "The donor outreach campaign is $600. We never promise amounts raised." if uses_revision else draft
    )
    [entry] = read_jsonl(temp_logs / "reflection.jsonl")
    assert entry["status"] == expected_status and entry["draft"] == draft and entry["final"] == result["reply"]


# ---------------------------------------------------------------------------
# Notebook deliverable
# ---------------------------------------------------------------------------

def notebook_cells():
    nb = json.loads((ROOT / "business_agent.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def test_notebook_has_labeled_persona_and_instructions_blocks():
    cells = notebook_cells()
    persona_i = next(i for i, c in enumerate(cells) if c.startswith("## 🤝 Persona"))
    instructions_i = next(i for i, c in enumerate(cells) if c.startswith("## 📋 Instructions"))
    assert "PERSONA = " in cells[persona_i + 1]
    assert "INSTRUCTIONS = " in cells[instructions_i + 1]
    assert persona_i < instructions_i


def test_instructions_cover_the_required_rules():
    import app
    text = app.INSTRUCTIONS
    for phrase in ("Answer only from", "Never invent prices", "do not stack", "Never promise",
                   "record_customer_interest", "record_feedback", "Never confirm a booking", "Worked example"):
        assert phrase in text, phrase
    assert "Rima" in app.PERSONA


# ---------------------------------------------------------------------------
# app.py (standalone app / Hugging Face Space)
# ---------------------------------------------------------------------------

def string_constants(source):
    """Top-level NAME = "string" assignments in some Python source."""
    import ast
    return {
        node.targets[0].id: node.value.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }


def test_app_persona_and_instructions_match_notebook():
    notebook = {}
    for cell in notebook_cells():
        notebook.update(string_constants(cell) if cell.lstrip().startswith(("# PERSONA", "# INSTRUCTIONS")) else {})
    app = string_constants((ROOT / "app.py").read_text(encoding="utf-8"))
    assert app["PERSONA"] == notebook["PERSONA"]
    assert app["INSTRUCTIONS"] == notebook["INSTRUCTIONS"]


def test_app_chat_uses_the_agent(monkeypatch):
    import app
    fake = FakeLLM("general_question", ["A volunteer drive is $300."])
    monkeypatch.setattr(app, "client", fake)
    monkeypatch.setattr(app, "MODEL", "fake-model")
    monkeypatch.setattr(app, "USE_REFLECTION", False)

    assert app.chat("How much is a volunteer drive?", []) == "A volunteer drive is $300."
    system_prompt = fake.agent_calls()[0]["messages"][0]["content"]
    assert system_prompt.startswith(app.PERSONA.strip()) and "## Business Profile" in system_prompt
