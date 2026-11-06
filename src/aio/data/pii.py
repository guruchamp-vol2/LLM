"""Remove personal data: emails, phone numbers, public IP addresses, credit card numbers."""

from __future__ import annotations

import ipaddress
import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Phone numbers need separators or a leading +, so plain numbers in text and math survive.
PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}(?!\w|[.-]\d)"
)
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _ip(m: re.Match) -> str:
    try:
        ip = ipaddress.ip_address(m.group(0))
    except ValueError:
        return m.group(0)  # e.g. 999.1.1.1 or a version number
    if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_unspecified:
        return m.group(0)
    return "[IP]"


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)


def redact_pii(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_replacements)."""
    tags = ("[EMAIL]", "[CARD]", "[IP]", "[PHONE]")
    before = sum(text.count(t) for t in tags)
    text = EMAIL.sub("[EMAIL]", text)
    text = CARD.sub(_card, text)
    text = IPV4.sub(_ip, text)
    text = PHONE.sub("[PHONE]", text)
    return text, sum(text.count(t) for t in tags) - before
