"""
business_data.py
----------------
The single source of truth for Bridgeway's facts, prices and discount rules.

Why a separate file?
- make_pdf.py uses it to build the price tables in me/about_business.pdf.
- In C4 (the ReAct agent) a price-estimate tool can simply do
      from business_data import PRICES, best_discount_percent
  instead of copying numbers around (copied numbers drift apart!).

All prices are in USD.
"""

BUSINESS_NAME = "Bridgeway – Fundraising & Communications Consultancy for NGOs"
LOCATION = "Hamra, Beirut, Lebanon"

PRICES = {
    # Campaign packages: one flat price per campaign
    "campaigns": {
        "donor_outreach_campaign": 600,   # research on up to 100 donor prospects + a 3-email sequence
        "partnership_pitch_pack": 450,    # up to 30 target organizations, pitch email + one-page summary
        "volunteer_drive": 300,           # 4 weeks: social posts, emails, sign-up form
    },
    # Subscription: monthly alert of relevant grants and deadlines
    "grant_deadline_tracking_per_month": 80,
    # Add-ons can be added to any campaign
    "add_ons": {
        "translation_arabic_french": 50,
        "extra_follow_up_wave": 100,
    },
}

# Discounts do NOT stack: when both apply, only the bigger one is used.
DISCOUNTS = {
    # 20% off for small NGOs (annual budget under 100,000 USD)
    "small_ngo": {"percent": 20, "max_annual_budget_usd": 100_000},
    # 10% off when booking 2 or more services
    "bundle": {"percent": 10, "min_services": 2},
}

# Human-readable labels for the price keys above (used in tables)
LABELS = {
    "donor_outreach_campaign": "Donor outreach campaign (research on up to 100 donor prospects + a 3-email sequence)",
    "partnership_pitch_pack": "Partnership pitch pack (up to 30 target organizations, pitch email + one-page summary)",
    "volunteer_drive": "Volunteer drive (4 weeks: social posts, emails, sign-up form)",
    "grant_deadline_tracking_per_month": "Grant-deadline tracking (monthly alert of relevant grants and deadlines)",
    "translation_arabic_french": "Arabic/French translation",
    "extra_follow_up_wave": "Extra follow-up wave",
}


def best_discount_percent(small_ngo=False, services_booked=1):
    """
    The one discount that applies (0, 10 or 20).

    small_ngo       : True if the NGO's annual budget is under 100,000 USD
    services_booked : how many services are booked together
    Discounts never stack: the bigger one wins.
    """
    candidates = [0]
    if small_ngo:
        candidates.append(DISCOUNTS["small_ngo"]["percent"])
    if services_booked >= DISCOUNTS["bundle"]["min_services"]:
        candidates.append(DISCOUNTS["bundle"]["percent"])
    return max(candidates)


def discounted_price(price, small_ngo=False, services_booked=1):
    """Price after the single applicable discount, e.g. discounted_price(600, small_ngo=True) -> 480."""
    value = price * (100 - best_discount_percent(small_ngo, services_booked)) / 100
    return int(value) if value == int(value) else round(value, 2)
