"""Input gate: checks the request envelope, cleans the turns and sets flags. Reads only id, context and turn_index.

Per turn text, in order: lone surrogates removed; control characters (C0, DEL, C1) except \\t \\n \\r become a
space; runs of combining marks cut to MARK_RUN; format characters (Unicode Cf: zero-width, bidi, BOM, soft hyphen)
removed; NFKC; repeated sentences dropped (`dedupe_text`, turns of 200+ characters); the head kept up to TURN_CAP
characters. Work is bounded by reading at most SCAN_CAP characters of a turn, before and after NFKC (which can grow
text up to 18 times), and by the mark cut (NFKC sorts a run of marks in time that grows with the square of its
length).

The injection check also reads a second, folded copy of each turn that the model never sees: format characters
become spaces, accents and other marks go, look-alike Cyrillic and Greek letters become Latin, and every dash
becomes "-".
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .text import ACTIONS30, dedupe_text

TURN_CAP = 4000                     # characters kept per turn; the classical model keeps its own first 2,000
SCAN_CAP = 4 * TURN_CAP             # characters read per turn; the longest published turn is 5,921
MAX_TURNS = 8
TURN_INDEX_MAX = 10000              # larger values are cut to TURN_INDEX_MAX - 1 (the model's features stop at 60)
MARK_RUN = 8                        # combining marks kept in a row; real text needs 3 or fewer
KNOWN_SHARE = 0.5                   # no_known_text: under this share of the dialogue's letters are Latin
SPEAKERS = ("customer", "agent", "action")
# turn_capped: any cut (SCAN_CAP or TURN_CAP); marks_capped: a run of combining marks cut to MARK_RUN; empty_turns:
# a turn blank after cleaning; bad_turn: a turn that is not an object, skipped; bad_turn_index: turn_index given
# but not a whole number >= 0 (len(context) used) or too large (cut).
FLAGS = frozenset({"truncated_turns", "turn_capped", "marks_capped", "controls_removed", "format_chars_removed",
                   "bidi_override", "surrogates_removed", "bad_speaker", "bad_text_type", "bad_turn", "bad_turn_index",
                   "empty_turns", "injection", "no_dialogue", "no_known_text", "junk"})
ROUTE_FLAGS = frozenset({"injection", "no_dialogue", "no_known_text", "junk", "controls_removed", "bidi_override",
                         "surrogates_removed"})

_SURROGATE = re.compile(r"[\ud800-\udfff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")    # C0, DEL, C1; keeps \t \n \r
_CHARS: dict[str, list[str]] = {"Cf": [], "Mn": [], "Me": [], "Pd": [], "ccc": []}
for _c in map(chr, range(0x110000)):
    _cat = unicodedata.category(_c)
    if _cat in _CHARS:
        _CHARS[_cat].append(_c)
    if _cat in ("Mn", "Me") or unicodedata.combining(_c) or (
            unicodedata.decomposition(_c) and unicodedata.combining(unicodedata.normalize("NFKD", _c)[0])):
        _CHARS["ccc"].append(_c)                     # marks, and the few characters that decompose to marks
_FORMAT = re.compile("[%s]" % "".join(_CHARS["Cf"]))
_MARKS = "".join(map(re.escape, _CHARS["ccc"]))
_MARK_RUN = re.compile("([%s]{%d})[%s]+" % (_MARKS, MARK_RUN, _MARKS))
_MN = re.compile("[%s]" % "".join(map(re.escape, _CHARS["Mn"] + _CHARS["Me"])))
_FILLER = re.compile("[\u115f\u1160\u3164\uffa0\u2800]")    # Hangul fillers and the blank braille cell
_FOLD = str.maketrans("аеорсухіјѕԁԛԝӏһıɡАВЕКМНОРСТХУІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧαιονρτυχκ",
                      "aeopcyxijsdqwlhigABEKMHOPCTXYIJSABEZHIKMNOPTYXaiovptuxk")
_FOLD.update({ord(c): "-" for c in _CHARS["Pd"] + ["\u2212"]})
del _c, _cat, _CHARS
_LETTER = re.compile(r"[^\W\d_]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")
_BIDI = re.compile(r"[\u202a-\u202e\u2066-\u2069]")    # embeddings, overrides, isolates
_SPACE = re.compile(r"\s")
_PLAIN = re.compile(r"[\w!-/:-@\[-`{-~]")         # letters, digits and ASCII punctuation
_ALNUM = re.compile(r"[^\W_]")

# Instruction-like text. Matched case-insensitively on the turn with whitespace runs squeezed to one space, so
# no pattern needs \s+; the only quantifiers are single bounded ones (no nested quantifiers, no blow-up).
_INJECTION = re.compile("|".join([
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.,;!?]{0,30}\b(?:instructions?|rules|prompts?|guidelines|directives)\b",
    r"\b(?:system|assistant|developer) ?:",
    r"</? ?(?:context|system|instructions?|prompt|user|assistant) ?>",
    r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>",
    r"\[/?inst\]|<</?sys>>",
    r"\b(?:new|updated) (?:instructions?|rules) ?:",
    r"\bsystem prompt\b",
    r"\bneeds[-_]?human\b",
    r"\byou are now (?:an |a |the |my )?(?:ai|assistant|chatbot|bot|model|dan|unrestricted|jailbroken|evil)\b",
    r"\b(?:speaking|acting|writing) as (?:the )?(?:system|admin|administrator|developer|operator)\b",
    r"\bas the (?:system|admin|administrator|developer|operator) ?[,:]",
    r"\b(?:developer|admin|god) mode\b",
    r"[\"'](?:intent|action|confidence|needs_human)[\"'] ?:",
]), re.I)
# Hyphenated action labels typed in a customer turn. "promo-code" and "try-again" are left out: customers
# write those as plain words.
_LABELS = [a for a in ACTIONS30 if "-" in a and a not in ("promo-code", "try-again")] + ["membership-privileges"]
_LABEL = re.compile(r"(?<![\w-])(?:%s)(?![\w-])" % "|".join(a.replace("-", "[-_]") for a in _LABELS), re.I)


@dataclass(frozen=True)
class GateResult:
    id: str
    context: list[dict]             # cleaned turns {"speaker", "text"}, at most MAX_TURNS
    turn_index: int                 # given whole number >= 0 (22.0 counts) cut to TURN_INDEX_MAX - 1, else len(context)
    flags: frozenset[str]
    abstain_reason: str | None      # not None -> return the abstention envelope


def envelope_error(obj) -> str | None:
    """None if obj is a JSON object with a non-empty string id (any length, any characters), else a short message
    (HTTP 400). The body limit bounds the id."""
    if not isinstance(obj, dict):
        return "body must be a JSON object"
    if "id" not in obj:
        return "missing id"
    rid = obj["id"]
    if not isinstance(rid, str):
        return "id must be a string"
    if not rid:
        return "id must not be empty"
    return None


def _dedupe(s: str) -> str:
    """dedupe_text, but the text stays as given unless a sentence was dropped (dedupe_text also squeezes spaces)."""
    d = dedupe_text(s)
    return s if d is s or d.split() == s.split() else d


def _cap(s: str, n: int, flags: set[str]) -> str:
    if len(s) <= n:
        return s
    flags.add("turn_capped")
    return s[:n]


def _cap_marks(s: str, flags: set[str]) -> str:
    s, n = _MARK_RUN.subn(r"\1", s)
    if n:
        flags.add("marks_capped")
    return s


def _fold(s: str) -> str:
    """The copy the injection check reads: format characters as spaces, no marks, look-alikes and dashes folded."""
    s = unicodedata.normalize("NFKD", _FORMAT.sub(" ", s))[:SCAN_CAP]
    return _FILLER.sub(" ", _MN.sub("", s)).translate(_FOLD)


def _clean_text(s: str, flags: set[str]) -> tuple[str, str]:
    """(cleaned text for the model, folded copy for the injection check)."""
    s, n = _SURROGATE.subn("", _cap(s, SCAN_CAP, flags))
    if n:
        flags.add("surrogates_removed")
    s, n = _CONTROL.subn(" ", s)
    if n:
        flags.add("controls_removed")
    if _BIDI.search(s):
        flags.add("bidi_override")
    s = _cap_marks(s, flags)
    folded = _fold(s)
    s, n = _FORMAT.subn("", s)
    if n:
        flags.add("format_chars_removed")
        s = _cap_marks(s, flags)                     # removing format characters can join two runs
    s = _cap(unicodedata.normalize("NFKC", s), SCAN_CAP, flags)
    return _cap(_dedupe(s), TURN_CAP, flags), folded


def _junk(s: str) -> bool:
    """8+ non-space characters, under half of them letters, digits or ASCII punctuation (emoji floods, symbols)."""
    ns = _SPACE.sub("", s)
    return len(ns) >= 8 and 2 * len(_PLAIN.sub("", ns)) > len(ns)


def _injection(s: str, speaker: str) -> bool:
    return bool(_INJECTION.search(" ".join(s.split())) or (speaker == "customer" and _LABEL.search(s)))


def _no_known_text(context: list[dict]) -> bool:
    """The customer and agent turns hold letters, but under KNOWN_SHARE of them are Latin (the model's words are)."""
    text = " ".join(t["text"] for t in context if t["speaker"] != "action")
    letters = len(_LETTER.findall(text))
    return letters > 0 and len(_LATIN.findall(text)) < KNOWN_SHARE * letters


def _turn_index(obj: dict, n: int, flags: set[str]) -> int:
    if "turn_index" not in obj:
        return n
    ti = obj["turn_index"]
    if isinstance(ti, float) and ti.is_integer():
        ti = int(ti)
    if not (isinstance(ti, int) and not isinstance(ti, bool) and ti >= 0):
        flags.add("bad_turn_index")
        return n
    if ti >= TURN_INDEX_MAX:
        flags.add("bad_turn_index")
        return TURN_INDEX_MAX - 1
    return ti


def clean(obj: dict) -> GateResult:
    """Clean one request whose envelope passed. Never raises for any JSON value under the envelope."""
    obj = obj if isinstance(obj, dict) else {}
    rid = obj.get("id") if isinstance(obj.get("id"), str) else ""
    flags: set[str] = set()
    raw = obj.get("context")
    turns = [t for t in raw if isinstance(t, dict)] if isinstance(raw, list) else []
    if isinstance(raw, list) and len(turns) < len(raw):
        flags.add("bad_turn")
    if len(turns) > MAX_TURNS:
        flags.add("truncated_turns")
        turns = turns[-MAX_TURNS:]
    context, folded = [], []
    for t in turns:
        spk = t.get("speaker")
        if not (isinstance(spk, str) and spk in SPEAKERS):
            flags.add("bad_speaker")
            spk = "customer"
        text = t.get("text")
        if not isinstance(text, str):
            flags.add("bad_text_type")
            text = ""
        text, fold = _clean_text(text, flags)
        context.append({"speaker": spk, "text": text})
        folded.append(fold)
    for t, fold in zip(context, folded):
        if not t["text"].strip():
            flags.add("empty_turns")
        elif _junk(t["text"]):
            flags.add("junk")
        if _injection(t["text"], t["speaker"]) or (fold != t["text"] and _injection(fold, t["speaker"])):
            flags.add("injection")
    if all(t["speaker"] == "customer" for t in context):
        flags.add("no_dialogue")
    if _no_known_text(context):
        flags.add("no_known_text")
    reason = None
    if not isinstance(raw, list):
        reason = "context is not a list"
    elif not any(_ALNUM.search(t["text"]) for t in context):
        reason = "no readable text in context"
    ti = _turn_index(obj, len(context), flags)
    return GateResult(rid, context, ti, frozenset(flags), reason)
