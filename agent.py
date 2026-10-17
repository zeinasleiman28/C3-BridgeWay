"""
agent.py
--------
The agent loop: one user message in, one reply out.

    run_agent(message, history, client, model, build_system_prompt, reflect=None)

Steps for every message:
1. ROUTE   - classify_intent() decides general_question / lead_collection / feedback.
2. PROMPT  - build_system_prompt(route_hint) gives the system prompt for this message.
3. MEMORY  - recent chat history (short-term memory) + the new message.
4. LOOP    - call the model with our tools. If it asks for tools, run them, send the
             results back, and call the model again (at most MAX_TOOL_ROUNDS times).
             When it answers with plain text, that text is the draft reply.
5. REFLECT - optional: check the draft against the instructions and revise it (Step 7).

Errors (no API key, network problems, rate limits...) become a friendly message
instead of a crash.
"""

import json

import openai

from log_utils import safe_print
from router import ROUTE_HINTS, classify_intent
from tools import OPENAI_TOOLS, run_tool

MAX_TOOL_ROUNDS = 5          # stop runaway tool loops
MAX_HISTORY_MESSAGES = 40    # short-term memory: the last ~20 user/assistant turns
TEMPERATURE = 0.3            # a little warmth, but stay close to the facts

FALLBACK_REPLY = (
    "Sorry, I couldn't put an answer together just now. "
    "Please try again, or reach our team at hello@bridgeway.lb."
)


def message_text(content):
    """Gradio may send content as a string or as a list of parts; return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return " ".join(p for p in parts if p)
    return ""


def clean_history(history):
    """Keep only user/assistant text messages, newest MAX_HISTORY_MESSAGES of them."""
    cleaned = []
    for m in history or []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            text = message_text(m.get("content"))
            if text:
                cleaned.append({"role": m["role"], "content": text})
    return cleaned[-MAX_HISTORY_MESSAGES:]


def friendly_error(error):
    """Turn an exception into a message we can show to the user."""
    # Gemini reports a bad key as a 400 BadRequestError ("Please pass a valid API key"), OpenAI as 401.
    if isinstance(error, openai.AuthenticationError) or (
        isinstance(error, openai.BadRequestError) and "api key" in str(error).lower()
    ):
        return "⚠️ The AI service rejected our API key. Please check the key in .env."
    if isinstance(error, openai.RateLimitError):
        return "⚠️ The AI service is busy right now (rate limit). Please wait a moment and try again."
    if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
        return "⚠️ I can't reach the AI service. Please check the internet connection and try again."
    if isinstance(error, openai.APIError):
        return f"⚠️ The AI service returned an error ({type(error).__name__}). Please try again."
    return f"⚠️ Something went wrong ({type(error).__name__}: {error}). Please try again."


def _assistant_tool_message(msg, call_ids):
    """Copy the model's tool-call message into a plain dict we can send back."""
    tool_calls = []
    for call, call_id in zip(msg.tool_calls, call_ids):
        entry = {
            "id": call_id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
        }
        # Gemini 3.x attaches a "thought_signature" under extra_content and rejects the next
        # request if it is not echoed back with the tool call. Other providers have no such field.
        extra_content = getattr(call, "extra_content", None)
        if extra_content:
            entry["extra_content"] = extra_content
        tool_calls.append(entry)
    return {"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls}


def run_agent(message, history, client, model, build_system_prompt, reflect=None):
    """
    Answer one user message. Never raises.

    Returns a dict:
        reply      - the text to show the user
        route      - the router's decision
        tool_calls - [{"name", "arguments", "result"}, ...] in the order they ran
        draft      - the reply before reflection (same as reply if no reflection)
        error      - None, or the error message if something failed
    """
    result = {"reply": "", "route": None, "tool_calls": [], "draft": None, "error": None}
    message = message_text(message).strip()
    if not message:
        result["reply"] = "Your message looks empty. How can I help?"
        return result
    if client is None:
        result["reply"] = result["error"] = (
            "⚠️ The AI service isn't set up yet. Copy .env.example to .env and add your API key."
        )
        return result

    history = clean_history(history)

    # 1-2. Route the message and build this message's system prompt.
    route = classify_intent(message, history, client, model)
    result["route"] = route
    system_prompt = build_system_prompt(ROUTE_HINTS[route["intent"]])

    # 3. System prompt + short-term memory + the new message.
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": message}]

    try:
        # 4. Tool loop.
        draft = None
        for round_number in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=model, messages=messages, tools=OPENAI_TOOLS, temperature=TEMPERATURE
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                draft = msg.content
                break

            # Each tool result must point back to its call id (some providers may leave it empty).
            call_ids = [call.id or f"call_{round_number}_{i}" for i, call in enumerate(msg.tool_calls)]
            messages.append(_assistant_tool_message(msg, call_ids))
            for call, call_id in zip(msg.tool_calls, call_ids):
                tool_result = run_tool(call.function.name, call.function.arguments)
                safe_print(f"[TOOL] {call.function.name}({call.function.arguments}) -> {tool_result}")
                result["tool_calls"].append(
                    {"name": call.function.name, "arguments": call.function.arguments, "result": tool_result}
                )
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(tool_result)})
        else:
            # Still calling tools after MAX_TOOL_ROUNDS: ask for a final answer without tools.
            response = client.chat.completions.create(
                model=model, messages=messages, tools=OPENAI_TOOLS, tool_choice="none", temperature=TEMPERATURE
            )
            draft = response.choices[0].message.content

        draft = (draft or "").strip() or FALLBACK_REPLY
        result["draft"] = draft

        # 5. Optional reflection.
        reply = draft
        if reflect is not None:
            reply = reflect(message, draft, history, result["tool_calls"])
        result["reply"] = reply
    except Exception as e:  # API/network errors, unexpected response shapes, ...
        safe_print(f"[AGENT] error: {type(e).__name__}: {e}")
        result["reply"] = result["error"] = friendly_error(e)

    return result
