"""
fake_llm.py
-----------
A fake OpenAI-style client so the agent can be tested offline (no API key, no internet).

It answers client.chat.completions.create(...) like the real SDK, using scripted replies:

    fake = FakeLLM(
        intent="lead_collection",                      # what the router call returns
        replies=[                                      # agent turns, in order
            tool_call("record_customer_interest", {"email": "a@b.co", "name": "Ann", "message": "Volunteer drive"}),
            "Thanks Ann! Our team will email you to schedule the discovery call.",
        ],
        reflection='{"ok": true}',                     # what the reflection call returns
    )

Every request is saved in fake.calls so tests can check what the agent sent.
"""

import json
from types import SimpleNamespace

from reflection import REFLECTION_PROMPT_START
from router import ROUTER_PROMPT


def tool_call(name, arguments, call_id=None):
    """A scripted reply in which the model asks to run one tool."""
    return {"tool_calls": [(name, arguments, call_id)]}


class FakeLLM:
    def __init__(self, intent="general_question", replies=(), reflection='{"ok": true}', error=None):
        self.intent = intent
        self.replies = list(replies)
        self.reflection = reflection
        self.error = error          # an exception to raise on agent calls
        self.calls = []
        self.chat = SimpleNamespace(completions=self)  # so client.chat.completions.create works

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]

        if system == ROUTER_PROMPT:
            return _response(json.dumps({"intent": self.intent, "confidence": 0.9, "reason": "fake"}))
        if system.startswith(REFLECTION_PROMPT_START):
            return _response(self.reflection)

        if self.error is not None:
            raise self.error
        if not self.replies:
            return _response("(no more scripted replies)")
        reply = self.replies.pop(0)
        if isinstance(reply, dict):
            calls = [
                SimpleNamespace(
                    id=call_id if call_id is not None else f"fake_{len(self.calls)}_{i}",
                    type="function",
                    function=SimpleNamespace(
                        name=name, arguments=arguments if isinstance(arguments, str) else json.dumps(arguments)
                    ),
                )
                for i, (name, arguments, call_id) in enumerate(reply["tool_calls"])
            ]
            return _response(None, calls)
        return _response(reply)

    def agent_calls(self):
        """Only the calls made by the agent loop (not the router or reflection)."""
        return [
            c for c in self.calls
            if c["messages"][0]["content"] != ROUTER_PROMPT
            and not c["messages"][0]["content"].startswith(REFLECTION_PROMPT_START)
        ]


def _response(content, tool_calls=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])
