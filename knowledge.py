"""
knowledge.py
------------
Loads Bridgeway's business documents and builds the system prompt from them.

    load_business_summary()  -> text of me/business_summary.txt
    load_business_profile()  -> text extracted from me/about_business.pdf
    compose_system_prompt(persona, instructions, summary, profile, route_hint)
                             -> the full system prompt sent to the model

The model only "knows" what we put in the system prompt, so these documents are
the assistant's knowledge. PERSONA and INSTRUCTIONS stay in the notebook (their
labeled sections) and are passed in as arguments.

This file lives outside the notebook so the tests in tests/ can import it.
"""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

ME_DIR = Path(__file__).parent / "me"
SUMMARY_PATH = ME_DIR / "business_summary.txt"
PDF_PATH = ME_DIR / "about_business.pdf"


class KnowledgeError(Exception):
    """Raised when a business document is missing or can't be read."""


def load_business_summary(path=SUMMARY_PATH):
    """Return the plain-text business summary."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise KnowledgeError(f"Business summary not found at {path}.") from None
    except (OSError, UnicodeDecodeError) as e:
        raise KnowledgeError(f"Could not read the business summary at {path}: {e}") from None
    if not text:
        raise KnowledgeError(f"The business summary at {path} is empty.")
    return text


def load_business_profile(path=PDF_PATH):
    """Return the text of the business PDF, one clean line per item."""
    path = Path(path)
    if not path.exists():
        raise KnowledgeError(f"Business PDF not found at {path}. Run `py make_pdf.py` to build it.")
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, OSError) as e:
        raise KnowledgeError(f"Could not read the business PDF at {path}: {e}") from None

    # The PDF puts every table cell on its own line with stray spaces.
    # Strip each line and drop empty ones so the prompt stays compact.
    lines = [line.strip() for page in pages for line in page.splitlines()]
    text = "\n".join(line for line in lines if line)
    if not text:
        raise KnowledgeError(f"No text could be extracted from {path}.")
    return text


def compose_system_prompt(persona, instructions, summary, profile, route_hint=None):
    """
    Join the building blocks into one system prompt.

    route_hint is a short, per-message note from the router (Step 5), e.g.
    "The user is interested: collect their name and email." It goes last so the
    model sees it right before the conversation.
    """
    sections = [
        persona.strip(),
        instructions.strip(),
        "## Business Summary\n" + summary.strip(),
        "## Business Profile (from about_business.pdf)\n" + profile.strip(),
    ]
    if route_hint:
        sections.append("## Note for this message\n" + route_hint.strip())
    return "\n\n".join(sections)
