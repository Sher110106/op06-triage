"""Failure taxonomy for any grader-format prediction file on dev (clean) and dev_noisy.

Port of the 13 Oct probe script scratch/probe_classical/src_remote/e1_taxonomy.py (EXPERIMENT_LOG R1). The
category rules are unchanged; only the data loading moved to triage.data. The rules read the published data
only: the row's context and turn_index, and the gold labels for scoring. Two tables come from train labels, as
in the probe: intent -> flow, and flow -> permitted actions (the published lists, literal, no alias). The
"state ignored" test uses a lookup fitted on all of train: (gold intent, last observed action) -> the most
common next action.

Each wrong row gets exactly one category, by a fixed priority, so the counts add up to the error total.

Intent errors, in priority order:
  numbered_faq_variant_confusion              gold and predicted intent are numbered siblings (policy_2 vs policy_3)
  flow_right_other_sibling                    predicted intent is in the gold flow
  flow_wrong_opening_outside_window           flow wrong; the conversation's first turn is not in the 8-turn window
                                              (turn_index - len(context) > 1)
  flow_wrong_opening_in_window_not_in_last4   flow wrong; first turn in the window but not in the last 4 turns
  flow_wrong_opening_in_last4                 flow wrong; first turn in the last 4 turns

Action errors, in priority order:
  gold_off_policy_naming_alias                gold is `membership` and the flow lists `membership-privileges`
  gold_off_policy_real_deviation              gold action is outside the flow's written list
  intent_wrong_flow_wrong                     predicted intent is wrong and in another flow
  intent_wrong_flow_right                     predicted intent is wrong but in the gold flow
  intent_right_state_ignored                  intent right; the gold-intent + last-action lookup gets the action right
  intent_right_state_insufficient             intent right; that lookup gets it wrong too

  python experiments/taxonomy/taxonomy.py PRED.jsonl [--label NAME] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from functools import cached_property
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from triage.data import load_split  # noqa: E402
from triage.state import template_key  # noqa: E402

INTENT_CATS = ("flow_wrong_opening_outside_window", "flow_right_other_sibling",
               "flow_wrong_opening_in_window_not_in_last4", "flow_wrong_opening_in_last4",
               "numbered_faq_variant_confusion")
ACTION_CATS = ("intent_wrong_flow_wrong", "intent_right_state_ignored", "intent_wrong_flow_right",
               "gold_off_policy_real_deviation", "intent_right_state_insufficient", "gold_off_policy_naming_alias")
NOISE_KINDS = ("code_mixed", "empty", "screenshot", "two_intents", "typo", "very_long")

NUM = re.compile(r"^(.*)_(\d)$")
FILLER = {"ok", "okay", "yes", "no", "sure", "thanks", "thank you", "thx", "k", "great", "alright",
          "yep", "yeah", "cool", "fine", "oh", "ah", "hmm", "", "ty", "perfect", "no problem"}


def family(intent):
    m = NUM.match(intent or "")
    return m.group(1) if m else None


def last_action(ctx) -> str:
    for t in reversed(ctx):
        if t["speaker"] == "action":
            return template_key(t["text"])
    return "NONE"


def last_customer(ctx) -> str:
    for t in reversed(ctx):
        if t["speaker"] == "customer":
            return t["text"]
    return ""


def read_preds(path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            out[str(p["id"])] = p
    return out


class Reference:
    """Published data and the tables the category rules need. Load once, score many files."""

    @cached_property
    def train(self):
        return load_split("train")

    @cached_property
    def dev(self):
        return load_split("dev")

    @cached_property
    def noisy(self):
        return load_split("noisy")

    @cached_property
    def hostile(self):
        return load_split("hostile")

    @cached_property
    def intent_flow(self) -> dict[str, str]:
        return {r["intent"]: r["flow"] for r in self.train}

    @cached_property
    def flow_perm(self) -> dict[str, set]:
        return {r["flow"]: set(r["permitted_actions"]) for r in self.train}

    @cached_property
    def state_rule(self) -> dict:
        """(gold intent, last observed action) -> most common next action, fitted on all of train."""
        rule = defaultdict(Counter)
        for r in self.train:
            rule[(r["intent"], last_action(r["context"]))][r["action"]] += 1
        return {k: c.most_common(1)[0][0] for k, c in rule.items()}

    def labelled(self):
        return [(r, "clean") for r in self.dev] + [(r, "noisy") for r in self.noisy]


def classify_row(ref: Reference, r: dict, p: dict | None) -> dict:
    """One labelled row against one prediction: right/wrong, exclusive category, and the overlapping flags."""
    p = p or {}
    pi, pa = p.get("intent"), p.get("action")
    ctx = r["context"]
    start = (r.get("turn_index") or 0) - len(ctx)
    out = {"id": r["id"], "intent_ok": pi == r["intent"], "action_ok": pa == r["action"],
           "intent_cat": None, "action_cat": None, "intent_flags": [], "action_flags": [],
           "gold": [r["intent"], r["action"]], "pred": [pi, pa],
           "opening_outside_window": start > 1, "gold_numbered_faq": family(r["intent"]) is not None,
           "no_action_turn": last_action(ctx) == "NONE"}
    pflow = ref.intent_flow.get(pi)
    if not out["intent_ok"]:
        lc = last_customer(ctx).strip().lower()
        f = {
            "flow_wrong": pflow != r["flow"],
            "flow_right_sibling": pflow == r["flow"],
            "numbered_faq_variant_confusion": family(pi) is not None and family(pi) == family(r["intent"]),
            "gold_is_numbered_faq": family(r["intent"]) is not None,
            "opening_outside_window": start > 1,
            "short_last_customer_turn": len(lc.split()) <= 3 or lc.strip(" .!?") in FILLER,
            "no_customer_turn": not any(t["speaker"] == "customer" for t in ctx),
            "opening_in_window_but_outside_last4": start <= 1 and len(ctx) > 4,
        }
        out["intent_flags"] = [k for k, v in f.items() if v]
        if f["numbered_faq_variant_confusion"]:
            out["intent_cat"] = "numbered_faq_variant_confusion"
        elif f["flow_right_sibling"]:
            out["intent_cat"] = "flow_right_other_sibling"
        elif f["opening_outside_window"]:
            out["intent_cat"] = "flow_wrong_opening_outside_window"
        elif f["opening_in_window_but_outside_last4"]:
            out["intent_cat"] = "flow_wrong_opening_in_window_not_in_last4"
        else:
            out["intent_cat"] = "flow_wrong_opening_in_last4"
    if not out["action_ok"]:
        la = last_action(ctx)
        perm = ref.flow_perm[r["flow"]]
        last = ctx[-1] if ctx else None
        f = {
            "intent_wrong": not out["intent_ok"],
            "intent_right_action_wrong": out["intent_ok"],
            "state_rule_with_gold_intent_would_fix": ref.state_rule.get((r["intent"], la)) == r["action"],
            "pred_repeats_last_observed_action": pa == la,
            "pred_already_observed_in_context": pa in {template_key(t["text"]) for t in ctx if t["speaker"] == "action"},
            "pred_outside_gold_flow_policy": pa not in perm,
            "gold_outside_written_policy": r["action"] not in perm,
            "gold_off_policy_naming_alias": r["action"] == "membership" and "membership-privileges" in perm,
            "last_turn_is_short_customer_filler": bool(last) and last["speaker"] == "customer"
                and (len(last["text"].split()) <= 3 or last["text"].strip(" .!?").lower() in FILLER),
            "no_action_turn_in_context": la == "NONE",
            "gold_is_ambiguous_troubleshoot_triplet": r["action"] in {"try-again", "log-out-in", "instructions"},
        }
        out["action_flags"] = [k for k, v in f.items() if v]
        if f["gold_off_policy_naming_alias"]:
            out["action_cat"] = "gold_off_policy_naming_alias"
        elif f["gold_outside_written_policy"]:
            out["action_cat"] = "gold_off_policy_real_deviation"
        elif f["intent_wrong"] and pflow != r["flow"]:
            out["action_cat"] = "intent_wrong_flow_wrong"
        elif f["intent_wrong"]:
            out["action_cat"] = "intent_wrong_flow_right"
        elif f["state_rule_with_gold_intent_would_fix"]:
            out["action_cat"] = "intent_right_state_ignored"
        else:
            out["action_cat"] = "intent_right_state_insufficient"
    return out


def classify(ref: Reference, preds: dict[str, dict]) -> list[dict]:
    """Per-row results for dev + dev_noisy, in data order, with the group and noise kind attached."""
    out = []
    for r, group in ref.labelled():
        c = classify_row(ref, r, preds.get(r["id"]))
        c["group"] = group
        c["noise_kind"] = r.get("noise_kind")
        out.append(c)
    return out


def summarise(rows: list[dict]) -> dict:
    """Counts for one group of per-row results."""
    n = len(rows)
    ie = [x for x in rows if not x["intent_ok"]]
    ae = [x for x in rows if not x["action_ok"]]

    def acc(sub, k):
        return round(sum(x[f"{k}_ok"] for x in sub) / len(sub), 4) if sub else None

    outside = [x for x in rows if x["opening_outside_window"]]
    visible = [x for x in rows if not x["opening_outside_window"]]
    context = {   # the probe's context rates (e1_taxonomy.py), share of all rows in the group
        "share_opening_outside_window": round(len(outside) / n, 4) if n else None,
        "intent_acc_opening_visible": acc(visible, "intent"),
        "intent_acc_opening_outside": acc(outside, "intent"),
        "n_gold_numbered_faq": sum(x["gold_numbered_faq"] for x in rows),
        "intent_acc_gold_numbered_faq": acc([x for x in rows if x["gold_numbered_faq"]], "intent"),
        "intent_acc_other": acc([x for x in rows if not x["gold_numbered_faq"]], "intent"),
        "action_acc_no_action_turn": acc([x for x in rows if x["no_action_turn"]], "action"),
        "action_acc_with_action_turn": acc([x for x in rows if not x["no_action_turn"]], "action"),
        "top_intent_confusions": Counter((x["gold"][0], x["pred"][0]) for x in ie).most_common(12),
        "top_action_confusions": Counter((x["gold"][1], x["pred"][1]) for x in ae).most_common(12),
    }
    return {
        "n": n,
        "intent_errors": len(ie), "action_errors": len(ae),
        "intent_acc": round(1 - len(ie) / n, 4) if n else None,
        "action_acc": round(1 - len(ae) / n, 4) if n else None,
        "intent_categories": {k: sum(x["intent_cat"] == k for x in ie) for k in INTENT_CATS},
        "action_categories": {k: sum(x["action_cat"] == k for x in ae) for k in ACTION_CATS},
        "intent_flags": dict(Counter(f for x in ie for f in x["intent_flags"]).most_common()),
        "action_flags": dict(Counter(f for x in ae for f in x["action_flags"]).most_common()),
        "context": context,
    }


def analyse(ref: Reference, preds: dict[str, dict]) -> dict:
    rows = classify(ref, preds)
    clean = [x for x in rows if x["group"] == "clean"]
    noisy = [x for x in rows if x["group"] == "noisy"]
    return {"clean": summarise(clean), "noisy": summarise(noisy),
            "noisy_by_kind": {k: summarise([x for x in noisy if x["noise_kind"] == k]) for k in NOISE_KINDS},
            "_rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred")
    ap.add_argument("--label")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = analyse(Reference(), read_preds(a.pred))
    res.pop("_rows")
    res = {"label": a.label or a.pred, "pred": a.pred, **res}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
