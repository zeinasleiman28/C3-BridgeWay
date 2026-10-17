"""
reflection.py
-------------
Bonus: reflection. Before a reply is sent, a second model call reviews the draft
against Rima's instructions and the business information, and rewrites it if it
breaks a rule (wrong price or discount, a promised fundraising result, a confirmed
booking, far too long, ...).

    reflect = make_reflector(client, model, instructions, business_info)
    final_reply = reflect(user_message, draft, history, tool_calls)

The reviewer answers with JSON:
    {"ok": true}                                             -> keep the draft
    {"ok": false, "problems": [...], "revised": "..."}       -> use the revised reply

Draft, problems and final reply are logged to logs/reflection.jsonl.
If anything goes wrong, the original draft is returned, so reflection never breaks the chat.
"""

import json
import re

from log_utils import append_jsonl, now_iso, safe_print

REFLECTION_LOG_FILE = "reflection.jsonl"

# fake_llm.py (tests) recognises reflection calls by this opening line.
REFLECTION_PROMPT_START = "You are a quality reviewer for Rima"

REFLECTION_PROMPT = REFLECTION_PROMPT_START + """, the outreach advisor of Bridgeway – Fundraising & Communications Consultancy for NGOs.
Check Rima's DRAFT reply against Rima's instructions and the business information below.

Check that the draft:
1. only states facts, prices and discounts that appear in the business information (no invented services, prices or discounts), with discounts applied exactly as written (they never stack);
2. never promises or estimates how much money, or how many donors, partners or volunteers, a campaign will bring;
3. does not confirm a booking, a start date or a call time (the team confirms by email);
4. does not claim something was recorded unless the tool calls show status "recorded";
5. answers the user's actual message, in Rima's warm, encouraging, non-salesy voice;
6. stays under about 120 words unless it lists prices.

If the draft passes every check, reply exactly: {"ok": true}
Otherwise reply with JSON only:
{"ok": false, "problems": ["<short problem>", ...], "revised": "<the corrected reply, ready to send>"}
The revised reply must follow all rules and only use facts from the business information.
"""


def parse_reflection_reply(text):
    """Return (ok, problems, revised). Raises ValueError if the reply is unusable."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the reflection reply")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from None
    if data.get("ok") is True:
        return True, [], None
    revised = str(data.get("revised") or "").strip()
    if not revised:
        raise ValueError("the draft was rejected but no revised reply was given")
    problems = [str(p) for p in data.get("problems") or []]
    return False, problems, revised


def make_reflector(client, model, instructions, business_info):
    """Create the reflect(...) function used by the agent loop."""
    system_prompt = (
        REFLECTION_PROMPT
        + "\n\n## Rima's instructions\n" + instructions.strip()
        + "\n\n## Business information\n" + business_info.strip()
    )

    def reflect(user_message, draft, history=None, tool_calls=None):
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-4:]) or "(none)"
        review_request = (
            f"Recent conversation:\n{recent}\n\n"
            f"User's latest message:\n{user_message}\n\n"
            f"Tool calls Rima made for this message:\n{json.dumps(tool_calls or [], ensure_ascii=False)}\n\n"
            f"DRAFT reply:\n{draft}"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": review_request}],
                temperature=0,
            )
            ok, problems, revised = parse_reflection_reply(response.choices[0].message.content)
            final, status = (draft, "approved") if ok else (revised, "revised")
        except Exception as e:  # API error or unusable JSON: keep the draft
            problems, final, status = [f"reflection failed: {type(e).__name__}: {e}"], draft, "skipped"

        safe_print(f"[REFLECTION] {status}" + (f" - {'; '.join(problems)}" if problems else ""))
        try:
            append_jsonl(REFLECTION_LOG_FILE, {
                "timestamp": now_iso(), "user_message": user_message, "status": status,
                "problems": problems, "draft": draft, "final": final,
            })
        except OSError:
            pass  # logging must never break the chat
        return final

    return reflect
