"""Remove personal data: emails, phone numbers, public IP addresses, credit card numbers.

The number rules are strict on purpose: maths, tables and code are full of digit runs,
and a loose rule quietly destroys them (e.g. ~1 in 10 random digit strings passes the
card checksum, and "Example 1.2.3.4" looks like an IP address).
"""

from __future__ import annotations

import ipaddress
import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Phone numbers need separators or a leading +, so plain numbers in text and math survive.
PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}(?!\w|[.-]\d)"
)
# Card numbers as people write them: one run of digits, groups of four (Visa/Mastercard/
# Discover), or 4-6-5 (Amex), with the same separator throughout.
CARD = re.compile(r"(?<![\d.,-])(?:\d{13,19}|\d{4}([ -])\d{4}\1\d{4}\1\d{4}(?:\1\d{1,3})?|\d{4}([ -])\d{6}\2\d{5})"
                  r"(?![\d-]|[.,]\d)")
# Card networks' leading digits and lengths.
_NETWORKS = [(re.compile(r"4"), (13, 16, 19)), (re.compile(r"5[1-5]|2[2-7]"), (16,)), (re.compile(r"3[47]"), (15,)),
             (re.compile(r"6011|65|64[4-9]"), (16, 19))]
_PHONE_WORDS = re.compile(r"(?<![a-z])(?:tel|phone|call|fax|mobile|cell|whatsapp|contact|ph)\b\W{0,6}$", re.I)
_SECTION_WORDS = re.compile(
    r"(?<![a-z])(?:section|sec|example|exercise|theorem|lemma|figure|fig|table|step|chapter|problem|definition|equation|eq|"
    r"version|ver|v|corollary|proposition|remark|question|item|rule|article|clause|part|appendix|§)\W{0,3}$", re.I)


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
    s = m.group(0)
    if _SECTION_WORDS.search(m.string[max(0, m.start() - 14) : m.start()]):
        return s  # "Section 2.1.3.4", "Example 1.2.3.4"
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return s  # e.g. 999.1.1.1 or a version number
    if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_unspecified:
        return s
    if all(int(p) < 32 for p in s.split(".")):
        return s  # section-style numbering (1.2.3.4) far more often than a real address
    return "[IP]"


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    for prefix, lengths in _NETWORKS:
        if prefix.match(digits) and len(digits) in lengths:
            return "[CARD]" if _luhn_ok(digits) else m.group(0)
    return m.group(0)


def _phone(m: re.Match) -> str:
    s = m.group(0)
    if "+" in s or "(" in s or re.search(r"\d[.-]\d", s):
        return "[PHONE]"
    # Only spaces between the groups: lists of numbers look the same, so ask for a
    # phone word just before it ("Tel: 020 7946 0958").
    return "[PHONE]" if _PHONE_WORDS.search(m.string[max(0, m.start() - 20) : m.start()]) else s


def redact_pii(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_replacements)."""
    tags = ("[EMAIL]", "[CARD]", "[IP]", "[PHONE]")
    before = sum(text.count(t) for t in tags)
    text = EMAIL.sub("[EMAIL]", text)
    text = CARD.sub(_card, text)
    text = IPV4.sub(_ip, text)
    text = PHONE.sub(_phone, text)
    return text, sum(text.count(t) for t in tags) - before
