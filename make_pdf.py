"""
make_pdf.py
-----------
Generates me/about_business.pdf — the full business profile of Bridgeway
that the agent reads as part of its knowledge.

Run it from the project root:
    python make_pdf.py

It uses `reportlab` to build the PDF from paragraphs and styled tables.
All prices and discount rules come from business_data.py, so the PDF can
never disagree with the numbers used in code.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from business_data import DISCOUNTS, LABELS, PRICES

OUTPUT_PATH = Path(__file__).parent / "me" / "about_business.pdf"

# ---------- Colours and text styles ----------
BRAND = colors.HexColor("#2F5D8A")       # calm blue brand colour
BRAND_LIGHT = colors.HexColor("#EDF3F9")  # light tint for striped table rows

_base = getSampleStyleSheet()
TITLE = ParagraphStyle("TitleBrand", parent=_base["Title"], textColor=BRAND, fontSize=24, spaceAfter=4)
SUBTITLE = ParagraphStyle("Subtitle", parent=_base["Normal"], alignment=TA_CENTER,
                          textColor=colors.grey, fontSize=11, spaceAfter=10)
HEADING = ParagraphStyle("HeadingBrand", parent=_base["Heading2"], textColor=BRAND,
                         spaceBefore=10, spaceAfter=4)
SUBHEADING = ParagraphStyle("Sub", parent=_base["Heading4"], spaceBefore=4, spaceAfter=2)
BODY = ParagraphStyle("Body", parent=_base["BodyText"], fontSize=10.5, leading=14)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=10, leading=13)


# ---------- Small building blocks ----------
def heading(text):
    return Paragraph(text, HEADING)


def para(text):
    return Paragraph(text, BODY)


def bullets(*lines):
    """Simple bullet list, one paragraph per line."""
    return [Paragraph(f"•&nbsp;&nbsp;{line}", BODY) for line in lines]


def cell(text):
    """Table cell that wraps long service descriptions onto several lines."""
    return Paragraph(text, CELL)


def styled_table(rows, col_widths=None):
    """A table with a coloured header row and striped body rows."""
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),   # price column centred
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def usd(amount):
    return f"${amount}"


# ---------- The document, section by section ----------
def build_story():
    story = [
        Paragraph("Bridgeway – Fundraising &amp; Communications Consultancy for NGOs", TITLE),
        Paragraph("Hamra, Beirut, Lebanon · Donor outreach · Partnerships · Volunteers · Grants", SUBTITLE),
        HRFlowable(width="100%", color=BRAND, thickness=1.5, spaceAfter=8),
    ]

    # 1. Overview & Mission
    story += [
        heading("Overview &amp; Mission"),
        para("Bridgeway is a small fundraising and communications consultancy in Hamra, Beirut, working with Lebanese NGOs."),
        para("<b>Mission:</b> Help Lebanese NGOs reach donors, partners and volunteers with honest, "
             "respectful outreach."),
    ]

    # 2. Why Choose Us (unique value proposition)
    story += [heading("Why Choose Us")]
    story += bullets(
        "We know the region's donor landscape: EU grants, Gulf foundations, the Lebanese diaspora",
        "Ethical, non-pushy messaging",
        "We never buy email lists",
        "Trilingual: Arabic, English and French",
        "Transparent reporting",
    )

    # 3. Services & Prices (all from business_data.PRICES)
    c = PRICES["campaigns"]
    story += [
        heading("Services &amp; Prices (USD)"),
        para("All prices are per campaign unless stated otherwise."),
        Spacer(1, 4),
        Paragraph("Campaigns", SUBHEADING),
        styled_table(
            [["Service", "Price"]]
            + [[cell(LABELS[key]), usd(c[key])] for key in c]
            + [[cell(LABELS["grant_deadline_tracking_per_month"]),
                f"{usd(PRICES['grant_deadline_tracking_per_month'])} / month"]],
            col_widths=[12.5 * cm, 3 * cm],
        ),
        Paragraph("Add-ons", SUBHEADING),
        styled_table([["Add-on", "Price"]]
                     + [[cell(LABELS[k]), usd(v)] for k, v in PRICES["add_ons"].items()],
                     col_widths=[12.5 * cm, 3 * cm]),
    ]

    # 4. Discounts (all from business_data.DISCOUNTS)
    small, bundle = DISCOUNTS["small_ngo"], DISCOUNTS["bundle"]
    story += [KeepTogether([
        heading("Discounts"),
        *bullets(
            f"<b>{small['percent']}% off</b> for small NGOs (annual budget under "
            f"{small['max_annual_budget_usd']:,} USD).",
            f"<b>{bundle['percent']}% off</b> when booking {bundle['min_services']} or more services.",
            "Discounts do not stack: when both apply, we apply the bigger one.",
        ),
    ])]

    # 5. How it works
    story += [heading("How It Works")]
    story += bullets(
        "Free 30-minute discovery call",
        "Launch within 2 weeks",
        "Results report at the end of the campaign",
    )

    # 6. Our principles & what we don't do
    story += [heading("Our Principles")]
    story += bullets(
        "We contact only public or opted-in addresses.",
        "We never promise how much money a campaign will raise.",
    )
    story += [heading("What We Do Not Offer")]
    story += bullets(
        "Writing full grant proposals",
        "Buying or selling email lists",
        "Paid ads",
        "Handling or collecting donation money",
        "Outreach for political parties or for-profit companies",
    )

    # 7. Team
    story += [heading("Team")]
    story += bullets(
        "<b>Nour Saleh</b> – Founder (8 years of fundraising at international NGOs)",
        "<b>Hadi Farah</b> – Donor research &amp; data",
        "<b>Leila Mansour</b> – Copywriter (Arabic, English, French)",
        "<b>Sami Khoury</b> – Client care &amp; bookings",
    )

    # 8. Hours & Contact
    story += [
        heading("Hours &amp; Contact"),
        para("<b>Hours:</b> Monday–Friday 9:00–17:00"),
        para("<b>Email:</b> hello@bridgeway.lb · <b>Phone:</b> +961 1 111 111 · "
             "<b>Instagram:</b> @bridgeway.beirut"),
    ]
    return story


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=A4,
        title="Bridgeway – Business Profile", author="Bridgeway",
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )
    doc.build(build_story())
    print(f"PDF created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
