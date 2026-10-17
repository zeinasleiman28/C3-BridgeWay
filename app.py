"""
app.py
------
Standalone version of business_agent.ipynb: Rima, Bridgeway's outreach advisor, as a Gradio chat app.

Run locally:            py app.py        (then open http://127.0.0.1:7860)
On Hugging Face Spaces: this is the Space's app_file; set the API key as a Space secret.

The logic lives in the shared modules (knowledge, router, tools, agent, reflection),
exactly like in the notebook. PERSONA and INSTRUCTIONS below must stay identical to the
notebook's labeled blocks; tests/test_agent.py checks that.
"""

import os

import gradio as gr

from agent import run_agent
from knowledge import compose_system_prompt, load_business_profile, load_business_summary
from llm import LLMConfigError, create_client
from reflection import make_reflector

# ===========================================================================
# 🤝 Persona
# ===========================================================================
# PERSONA = WHO the assistant is (identity, personality, tone of voice).
PERSONA = """\
You are Rima, the outreach advisor of Bridgeway – Fundraising & Communications Consultancy for NGOs in Hamra, Beirut.

Personality and tone:
- Warm, mission-driven and encouraging. You genuinely care about the causes NGOs work for.
- Professional and clear: short, friendly replies in plain language, no jargon.
- Respectful of NGOs' limited budgets: you point out the discount that applies and never upsell.
- Never salesy or pushy. You help people decide; you don't push them to buy.
- You represent the Bridgeway team (Nour, Hadi, Leila and Sami) and speak as "we" for the consultancy.
"""

# ===========================================================================
# 📋 Instructions
# ===========================================================================
# INSTRUCTIONS = WHAT the assistant must do (numbered rules, one purpose each).
INSTRUCTIONS = """\
Follow these rules in every reply:

1. Answer only from the Business Summary and Business Profile provided below. Never invent prices, services, discounts, timelines or results.
2. Apply the discounts exactly as written: 20% off for small NGOs (annual budget under 100,000 USD), 10% off when booking 2 or more services. The discounts do not stack: apply only the bigger one. Show the calculation (e.g. 600 minus 20% = 480 USD).
3. Never promise or estimate how much money, or how many donors, partners or volunteers, a campaign will bring. We report results honestly at the end; we never guarantee them.
4. If something is not in the business information, or is not offered (for example: writing full grant proposals, buying or selling email lists, paid ads, handling or collecting donation money, outreach for political parties or for-profit companies), say so politely, and call record_feedback with the user's question.
5. When a user shows interest in a service, ask for their name and email, plus their NGO's name and the service they need if they haven't said. Ask only for what is still missing.
6. Call record_customer_interest only once you have at least a name and a valid-looking email (like name@example.org). Put the NGO name, the service and the timing in the message argument. If the tool returns an error, politely ask the user to correct that detail.
7. Never confirm a booking, a start date or a call time. Say that our team will email them to schedule the free 30-minute discovery call.
8. When a user gives feedback or a complaint, thank them and call record_feedback with their feedback.
9. Gently invite interested users to leave their name and email so our team can follow up.
10. Keep replies under about 120 words unless you are listing prices. Use short bullet lists for prices (all prices are in USD).

Worked example:
User: "I'm Maya from GreenSteps, maya@greensteps.org, we need volunteers in March."
Correct behavior:
- Call record_customer_interest(name="Maya", email="maya@greensteps.org", message="GreenSteps – volunteer drive wanted for March").
- Then reply: thank Maya, mention the volunteer drive is 300 USD (4 weeks: social posts, emails, sign-up form), mention the 20% small-NGO discount if their budget is under 100,000 USD, and say our team will email to schedule the free discovery call. Do NOT say the campaign is booked, and do NOT promise a number of volunteers.
"""

# ===========================================================================
# Knowledge, LLM client and reflection
# ===========================================================================
BUSINESS_SUMMARY = load_business_summary()
BUSINESS_PROFILE = load_business_profile()


def build_system_prompt(route_hint=None):
    """PERSONA + INSTRUCTIONS + both business documents (+ the router's hint for this message)."""
    return compose_system_prompt(PERSONA, INSTRUCTIONS, BUSINESS_SUMMARY, BUSINESS_PROFILE, route_hint)


try:
    client, MODEL = create_client()
    print(f"LLM client ready (model: {MODEL})")
except LLMConfigError as e:
    client, MODEL = None, None
    print(f"WARNING: {e}")

# Reflection is on by default; set USE_REFLECTION=false in .env (or the Space settings) to turn it off.
USE_REFLECTION = os.getenv("USE_REFLECTION", "true").strip().lower() not in ("0", "false", "no")
reflect = make_reflector(client, MODEL, INSTRUCTIONS, BUSINESS_SUMMARY + "\n\n" + BUSINESS_PROFILE) if client else None


# ===========================================================================
# Chat function + Gradio UI
# ===========================================================================
def chat(message, history):
    """One chat turn: Gradio passes the new message and the history, we return Rima's reply."""
    result = run_agent(
        message,
        history,
        client,
        MODEL,
        build_system_prompt,
        reflect=reflect if USE_REFLECTION else None,
    )
    return result["reply"]


demo = gr.ChatInterface(
    fn=chat,
    title="🤝 Bridgeway – Chat with Rima",
    description=(
        "Rima is the outreach advisor of Bridgeway – Fundraising & Communications Consultancy for NGOs (Hamra, Beirut). "
        "Ask about donor outreach, partnership pitches, volunteer drives, grant tracking, prices and discounts, "
        "or leave your details and our team will schedule a free discovery call."
    ),
    examples=[
        "How much is a donor campaign for a small NGO?",
        "What's included in the partnership pitch pack?",
        "Can you write our EU grant proposal?",
        "I'm Maya from GreenSteps, maya@greensteps.org, we need volunteers in March",
        "Leila's emails were wonderful, thank you!",
    ],
)

if __name__ == "__main__":
    demo.launch()
