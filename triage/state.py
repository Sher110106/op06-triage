"""Dialogue-state parser: map templated `action`-speaker turns to the action label they record.

The rules were derived by reading the templates in train and checked against train alignment
(see EXPERIMENT_LOG.md, 13 Oct, classical probe E3)."""
from __future__ import annotations

import re

RULES = [
    (r"^account has been pulled up", "pull-up-account"),
    (r"^searching the faq pages", "search-faq"),
    (r"^purchase validation in progress", "validate-purchase"),
    (r"^identity verification in progr?ess", "verify-identity"),
    (r"^querying the system for an answer", "ask-the-oracle"),
    (r"^querying the system for subscription status", "subscription-status"),
    (r"^agent is looking for solutions", "__looking_for_solutions"),  # try-again, log-out-in or instructions
    (r"^system action: search (membership)", "search-membership"),
    (r"^system action: search (pricing)", "search-pricing"),
    (r"^system action: search (policy)", "search-policy"),
    (r"^system action: search (timing)", "search-timing"),
    (r"^system action: search (jeans)", "search-jeans"),
    (r"^system action: search (jacket)", "search-jacket"),
    (r"^system action: search (boots)", "search-boots"),
    (r"^system action: search (shirt)", "search-shirt"),
    (r"^membership level of", "membership"),
    (r"^details of .* (have|has) been entered", "enter-details"),
    (r"^shipping status of", "shipping-status"),
    (r"^the .*has been notified", "notify-team"),
    (r"^a link will be sent", "send-link"),
    (r"^a reason of", "record-reason"),
    (r"^order has been updated", "update-order"),
    (r"^faq answer related to", "select-faq"),
    (r"^a refund has been made", "offer-refund"),
    (r"^account has been updated", "update-account"),
    (r"^a promo code has been created", "promo-code"),
    (r"^a purchase of", "make-purchase"),
    (r"^a password has been generated", "make-password"),
]
_COMPILED = [(re.compile(p), a) for p, a in RULES]


def template_key(text: str) -> str:
    """Action label for an action-log turn, or 'OTHER:<first words>' when no rule matches."""
    s = (text or "").lower().strip()
    s = re.sub(r"\d+", "#", s)
    for rx, act in _COMPILED:
        if rx.search(s):
            return act
    return "OTHER:" + " ".join(s.split()[:4])
