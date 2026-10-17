"""
router.py
---------
The router reads each user message and decides what kind of message it is:

    general_question  -> a question about services, prices, discounts, hours, how it works, ...
    lead_collection   -> the user wants to start a campaign or shares contact details
    feedback          -> praise, a complaint or a suggestion

classify_intent(message, history, client, model) returns
    {"intent": ..., "confidence": 0.0-1.0, "reason": "...", "source": "llm" | "keywords"}

How it works:
1. If we have an LLM client, we ask the model (temperature 0, so it's consistent)
   to answer with JSON only.
2. If there is no client, or the call fails, or the reply isn't valid JSON,
   a simple keyword classifier takes over, so the router always gives an answer.

The agent then uses ROUTE_HINTS[intent] as a short per-message note in the
system prompt, so each kind of message gets the right handling.
"""

import json
import re

from log_utils import append_jsonl, now_iso, safe_print

INTENTS = ("general_question", "lead_collection", "feedback")
ROUTER_LOG_FILE = "router.jsonl"

# One hint per route. It is added at the end of the system prompt for that message.
ROUTE_HINTS = {
    "general_question": (
        "The user is asking a general question. Answer it from the business information, applying "
        "the discount rules exactly as written. If the answer isn't there, or the service isn't offered, "
        "say so politely and call record_feedback with their question."
    ),
    "lead_collection": (
        "The user seems interested in working with us. Help them, ask only for the details still missing "
        "(name, email, NGO name, which service, timing), and call record_customer_interest once you have "
        "a name and a valid email. Never confirm a booking or a date: say our team will email them to "
        "schedule the free discovery call."
    ),
    "feedback": (
        "The user is giving feedback, praise or a complaint. Thank them warmly, call record_feedback "
        "with their feedback, and if it is a complaint, say our team will look into it."
    ),
}

ROUTER_PROMPT = """\
You are a message router for Bridgeway, a fundraising and communications consultancy that helps Lebanese NGOs reach donors, partners and volunteers.
Classify the user's LATEST message into exactly one intent:

- general_question: questions about services, prices, discounts, how it works, hours, location, team, or anything else.
- lead_collection: the user wants to start, book or get a quote for a campaign; asks to work with us or to
  schedule the discovery call; or gives contact details (name, email, phone, NGO name) or project details
  (which service, when) to move forward.
- feedback: praise, complaints, suggestions or reviews about Bridgeway.

Use the recent conversation only to understand short replies (e.g. "maya@greensteps.org" after being asked for an email is lead_collection).

Reply with JSON only, no other text:
{"intent": "<general_question|lead_collection|feedback>", "confidence": <0.0-1.0>, "reason": "<short reason>"}
"""

# How many earlier messages the router sees (enough to understand short replies).
ROUTER_HISTORY_MESSAGES = 4

# --- Keyword fallback ------------------------------------------------------

EMAIL_IN_TEXT = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

LEAD_KEYWORDS = (
    "book", "get started", "start a campaign", "sign up", "schedule", "discovery call",
    "quote", "hire you", "work with you", "my name is", "we need", "we'd like", "we would like",
    "interested in",
)
FEEDBACK_KEYWORDS = (
    "feedback", "complaint", "complain", "review", "suggest", "suggestion",
    "unhappy", "disappointed", "terrible", "awful", "bad experience", "rude", "not happy",
    "loved", "amazing", "great job", "great work", "thank you for", "well done", "wonderful",
)


def keyword_classify(message):
    """Simple rule-based classifier used when the LLM router isn't available."""
    text = message.lower()

    if any(word in text for word in FEEDBACK_KEYWORDS):
        return _result("feedback", 0.6, "Feedback keyword found.", "keywords")
    if EMAIL_IN_TEXT.search(message):
        return _result("lead_collection", 0.7, "Message contains an email address.", "keywords")
    if any(word in text for word in LEAD_KEYWORDS):
        return _result("lead_collection", 0.6, "Lead keyword found.", "keywords")
    return _result("general_question", 0.5, "No lead or feedback keywords.", "keywords")


# --- LLM router -----------------------------------------------------------

def parse_router_reply(text):
    """
    Turn the model's reply into a result dict.
    Accepts plain JSON or JSON wrapped in ```code fences```. Raises ValueError if unusable.
    """
    if not text:
        raise ValueError("empty reply")
    match = re.search(r"\{.*\}", text, re.DOTALL)  # first '{' to last '}'
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:80]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from None

    intent = str(data.get("intent", "")).strip().lower()
    if intent not in INTENTS:
        raise ValueError(f"unknown intent {intent!r}")
    try:
        confidence = min(max(float(data.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.5
    reason = str(data.get("reason", "")).strip() or "(no reason given)"
    return _result(intent, confidence, reason, "llm")


def _router_messages(message, history):
    """Build the short conversation the router model sees."""
    messages = [{"role": "system", "content": ROUTER_PROMPT}]
    recent = [
        m for m in (history or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ][-ROUTER_HISTORY_MESSAGES:]
    if recent:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        messages.append({"role": "user", "content": f"Recent conversation:\n{transcript}\n\nLatest message:\n{message}"})
    else:
        messages.append({"role": "user", "content": f"Latest message:\n{message}"})
    return messages


def classify_intent(message, history=None, client=None, model=None):
    """
    Classify one user message. Always returns a result (never raises).

    message : the user's latest message
    history : earlier messages as [{"role": "user"|"assistant", "content": "..."}]
    client  : an OpenAI-compatible client (None -> keyword fallback only)
    model   : the model name to use with that client
    """
    message = (message or "").strip()
    if not message:
        result = _result("general_question", 0.0, "Empty message.", "keywords")
    elif client is None or not model:
        result = keyword_classify(message)
    else:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=_router_messages(message, history),
                temperature=0,
            )
            result = parse_router_reply(response.choices[0].message.content)
        except Exception as e:  # API error, timeout, bad JSON... fall back instead of crashing
            result = keyword_classify(message)
            result["reason"] += f" (LLM router failed: {type(e).__name__}: {e})"

    safe_print(f"[ROUTER] {result['intent']} ({result['confidence']:.2f}, {result['source']}) - {result['reason']}")
    try:
        append_jsonl(ROUTER_LOG_FILE, {"timestamp": now_iso(), "message": message, **result})
    except OSError:
        pass  # logging must never break the chat
    return result


def _result(intent, confidence, reason, source):
    return {"intent": intent, "confidence": confidence, "reason": reason, "source": source}
