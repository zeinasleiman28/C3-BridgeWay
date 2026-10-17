"""
tools.py
--------
The two tools Rima (our assistant) can call:

    record_customer_interest(email, name, message)  -> saves a lead (an NGO interested in a service)
    record_feedback(question)                        -> saves feedback or a question we couldn't answer

Each tool:
1. validates its input and returns {"status": "error", "reason": ...} if something is wrong,
   so the model can ask the user to fix it (instead of the program crashing);
2. prints a short log line (you see it in the notebook / terminal);
3. appends one JSON line to a file in logs/ (leads.jsonl or feedback.jsonl);
4. returns {"status": "recorded", "id": ...} on success.

The tools only *record* things. All business facts and prices stay in business_data.py.

This file lives outside the notebook so the tests in tests/ can import it.
"""

import json
import re
import uuid

# Shared helpers: timestamps, JSONL files in logs/, emoji-safe printing.
# Tests can point the logs somewhere else: log_utils.LOG_DIR = tmp_path
from log_utils import append_jsonl, now_iso, safe_print

LEADS_FILE = "leads.jsonl"
FEEDBACK_FILE = "feedback.jsonl"

# Simple "looks like an email" check: something@something.something (no spaces).
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Upper limits so a runaway model can't write huge entries into our logs.
MAX_NAME_LENGTH = 100
MAX_TEXT_LENGTH = 2000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _new_id(prefix):
    """A short unique id such as 'lead-3f9a1c2b'."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _error(reason):
    """The standard error result every tool returns."""
    return {"status": "error", "reason": reason}


def _clean_text(value):
    """Turn the value into a stripped string ('' for None)."""
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Tool 1: record_customer_interest
# ---------------------------------------------------------------------------

def record_customer_interest(email, name, message=""):
    """Save a potential customer's contact details and what they are interested in."""
    email = _clean_text(email)
    name = _clean_text(name)
    message = _clean_text(message)

    # Validate. Each error says exactly what the model should ask the user to fix.
    if not name:
        return _error("Missing name. Please ask the user for their name.")
    if len(name) > MAX_NAME_LENGTH:
        return _error(f"Name is too long (max {MAX_NAME_LENGTH} characters).")
    if not email:
        return _error("Missing email. Please ask the user for their email address.")
    if not EMAIL_PATTERN.match(email):
        return _error(f"'{email}' does not look like a valid email. Please ask the user to check it.")
    if len(message) > MAX_TEXT_LENGTH:
        return _error(f"Message is too long (max {MAX_TEXT_LENGTH} characters). Please summarize it.")

    record = {
        "id": _new_id("lead"),
        "timestamp": now_iso(),
        "name": name,
        "email": email,
        "message": message,
    }
    try:
        append_jsonl(LEADS_FILE, record)
    except OSError as e:
        return _error(f"Could not save the lead ({e}). Ask the user to email us directly.")

    safe_print(f"📝 [LEAD] {name} <{email}>: {message or '(no details)'}")
    return {"status": "recorded", "id": record["id"]}


# ---------------------------------------------------------------------------
# Tool 2: record_feedback
# ---------------------------------------------------------------------------

def record_feedback(question):
    """Save user feedback, or a question the assistant could not answer."""
    question = _clean_text(question)

    if not question:
        return _error("Missing text. Pass the user's feedback or unanswered question.")
    if len(question) > MAX_TEXT_LENGTH:
        return _error(f"Text is too long (max {MAX_TEXT_LENGTH} characters). Please summarize it.")

    record = {
        "id": _new_id("fb"),
        "timestamp": now_iso(),
        "question": question,
    }
    try:
        append_jsonl(FEEDBACK_FILE, record)
    except OSError as e:
        return _error(f"Could not save the feedback ({e}).")

    safe_print(f"💬 [FEEDBACK] {question}")
    return {"status": "recorded", "id": record["id"]}


# ---------------------------------------------------------------------------
# Tool definitions, MCP-style
# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) describes every tool with three fields:
#   name         - what the model calls
#   description  - when and why to use it (the model reads this!)
#   inputSchema  - a JSON Schema for the arguments
# We write the tools once in this format and convert them for the OpenAI API below.

MCP_TOOL_DEFINITIONS = [
    {
        "name": "record_customer_interest",
        "description": (
            "Record a lead: someone from an NGO who is interested in one of Bridgeway's services "
            "(donor outreach campaign, partnership pitch pack, volunteer drive, grant-deadline tracking) "
            "so the team can follow up by email and schedule the free discovery call. Call it as soon as "
            "you have at least the person's name and a valid-looking email; do not wait for anything else. "
            "Do not call it for general questions without contact details. Put the NGO name, the service "
            "they want and the timing in 'message'. Returns {status: 'recorded', id} or {status: 'error', reason}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The person's email address, e.g. maya@greensteps.org",
                },
                "name": {
                    "type": "string",
                    "description": "The person's name, e.g. Maya",
                },
                "message": {
                    "type": "string",
                    "description": "What they want: NGO name, service(s), timing. "
                                   "E.g. 'GreenSteps – volunteer drive wanted for March'",
                },
            },
            "required": ["email", "name", "message"],
        },
    },
    {
        "name": "record_feedback",
        "description": (
            "Record something the team must review. Call it when (a) the user asks about something "
            "not in the business information, (b) the user asks for a service Bridgeway does not offer "
            "(full grant proposals, buying/selling email lists, paid ads, handling donation money, "
            "outreach for political parties or for-profit companies), or (c) the user gives feedback, "
            "praise or a complaint. Returns {status: 'recorded', id} or {status: 'error', reason}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's unanswered question or their feedback, in their words.",
                },
            },
            "required": ["question"],
        },
    },
]


def to_openai_tools(mcp_definitions):
    """Convert MCP-style definitions into the 'tools' format of the OpenAI chat API."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in mcp_definitions
    ]


# What we pass as tools=... to client.chat.completions.create()
OPENAI_TOOLS = to_openai_tools(MCP_TOOL_DEFINITIONS)

# Tool name (as the model says it) -> the Python function that runs it
TOOL_REGISTRY = {
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
}


# ---------------------------------------------------------------------------
# Running a tool call that came from the model
# ---------------------------------------------------------------------------

def run_tool(name, arguments):
    """
    Run the tool the model asked for and always return a dict (never raise).

    `arguments` is what the model sent: usually a JSON string like '{"question": "..."}',
    but a dict is accepted too. Unknown tools, broken JSON and wrong arguments all become
    {"status": "error", ...} so the model can recover.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return _error(f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY)}.")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return _error(f"Arguments for '{name}' are not valid JSON: {arguments!r}")
    if not isinstance(arguments, dict):
        return _error(f"Arguments for '{name}' must be a JSON object.")

    # Ignore any extra arguments the model invented; pass only the ones the tool accepts.
    schema = next(t["inputSchema"] for t in MCP_TOOL_DEFINITIONS if t["name"] == name)
    allowed = schema["properties"].keys()
    kwargs = {key: value for key, value in arguments.items() if key in allowed}

    try:
        return func(**kwargs)
    except TypeError as e:  # e.g. a required argument is missing
        return _error(f"Wrong arguments for '{name}': {e}")
