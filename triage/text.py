"""Text cleaning, text views and dialogue-state features. Uses only id, context and turn_index."""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from .state import template_key

ACTIONS30 = ['pull-up-account', 'verify-identity', 'search-faq', 'validate-purchase', 'enter-details',
             'ask-the-oracle', 'select-faq', 'record-reason', 'update-order', 'membership', 'shipping-status',
             'log-out-in', 'offer-refund', 'send-link', 'try-again', 'update-account', 'subscription-status',
             'notify-team', 'promo-code', 'search-membership', 'make-purchase', 'search-pricing', 'instructions',
             'search-policy', 'make-password', 'search-boots', 'search-timing', 'search-jeans', 'search-jacket',
             'search-shirt']
STATE_KEYS = ACTIONS30 + ["__looking_for_solutions"]
SKEY_IDX = {k: i for i, k in enumerate(STATE_KEYS)}
TURN_CAP = 2000                     # characters kept per turn; very_long noise is about 6k chars of filler
_ZW = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_text(s) -> str:
    s = s if isinstance(s, str) else ""
    s = unicodedata.normalize("NFKC", s)
    s = _ZW.sub("", s)
    s = _CTRL.sub(" ", s)
    return s.lower()[:TURN_CAP]


def dedupe_text(s):
    """Drop exact repeated sentences inside one turn (turns of 200+ chars only). Inference time only."""
    if not isinstance(s, str) or len(s) < 200:
        return s
    seen, out = set(), []
    for sent in _SENT_SPLIT.split(s):
        key = " ".join(sent.lower().split())
        if key and key in seen:
            continue
        seen.add(key)
        out.append(sent)
    return " ".join(out)


def turns(row) -> list[tuple[str, str]]:
    out = []
    for t in (row.get("context") or [])[-8:]:
        spk = t.get("speaker") if t.get("speaker") in ("customer", "agent", "action") else "customer"
        out.append((spk, clean_text(t.get("text"))))
    return out


def act_token(text: str) -> str:
    return "act_" + template_key(text).replace("-", "").replace("__", "")


def view_all(row) -> str:
    return " ".join(f"zzact {act_token(t)}" if s == "action" else f"zz{s[:4]} {t}" for s, t in turns(row))


def view_last_k(row, k: int = 3) -> str:
    return " ".join(f"zzact {act_token(t)}" if s == "action" else f"zz{s[:4]} {t}" for s, t in turns(row)[-k:])


def _last(row, speaker: str) -> str:
    for s, t in reversed(turns(row)):
        if s == speaker:
            return t
    return ""


def view_last_agent(row) -> str:
    return _last(row, "agent")


def view_last_customer(row) -> str:
    return _last(row, "customer")


def observed_actions(row) -> list[str]:
    return [template_key(t) for s, t in turns(row) if s == "action"]


def last_obs(row) -> str:
    obs = observed_actions(row)
    return obs[-1] if obs else "NONE"


def state_dense(row) -> np.ndarray:
    """Fixed-width dialogue-state features: last and previous observed action, actions seen and their
    counts, and 12 position/shape features (the first of these, at index -12, is the action-turn count)."""
    tt = turns(row)
    obs = observed_actions(row)
    n = len(STATE_KEYS)
    last, prev = np.zeros(n + 1), np.zeros(n + 1)
    seen, cnt = np.zeros(n), np.zeros(n)
    last[SKEY_IDX.get(obs[-1], n) if obs else n] = 1
    prev[SKEY_IDX.get(obs[-2], n) if len(obs) >= 2 else n] = 1
    for o in obs:
        if o in SKEY_IDX:
            seen[SKEY_IDX[o]] = 1
            cnt[SKEY_IDX[o]] += 1
    ti = row.get("turn_index") or 0
    since = 0
    for s, _ in reversed(tt):
        if s == "action":
            break
        since += 1
    last_spk = tt[-1][0] if tt else "none"
    misc = np.array([
        len(obs), min(ti, 60) / 60.0, ti <= 6, 6 < ti <= 14, ti > 14, ti - len(tt) <= 1,
        since / 8.0, last_spk == "customer", last_spk == "agent", last_spk == "action",
        len(tt) / 8.0, sum(s == "customer" for s, _ in tt) / 8.0,
    ], dtype=float)
    return np.concatenate([last, prev, seen, np.minimum(cnt, 3) / 3.0, misc])
