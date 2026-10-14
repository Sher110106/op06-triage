"""Exploratory, not pre-registered: the 9 hostile texts inside real conversations.

Every published hostile row is one customer turn with no agent turn, so the v1 gate's `no_dialogue` rule routes
all nine by shape alone. This probe asks what v1 does when that rule cannot fire: each hostile text is added as
the newest customer turn of each of the 2,000 dev conversations (the oldest turn drops out past 8, turn_index
+1). Each hostile text has a matched control, added the same way:

  hostile-000..002 (injection)   "ok thanks", a neutral turn
  hostile-003 (zero-width)       the same words without the zero-width characters: "refund please"
  hostile-004 (bidi override)    the same words without the override characters: "cancel my order"
  hostile-005 (decomposed e)     the same words composed (NFC): "café order issue"
  hostile-006 (control chars)    the same text without the control characters
  hostile-007 (emoji flood)      an empty customer turn
  hostile-008 (empty text)       "ok thanks"

v1 runs in process with the LLM step off and the shipped policy (no API call). The rows have no gold labels
once a turn is added, so there is no grader report: the outputs are routing and agreement with the control.

  python experiments/taxonomy/hostile_in_context.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from triage import gate  # noqa: E402
from triage.data import DATA, load_split, project  # noqa: E402
from triage.llm_action import LLMConfig  # noqa: E402
from triage.pipeline import Triage  # noqa: E402
from triage.policy import DEFAULT_POLICY  # noqa: E402

RUN = ROOT / "results" / "raw" / "taxonomy-hostile-in-context"
CONTROL = {
    "hostile-000": "ok thanks", "hostile-001": "ok thanks", "hostile-002": "ok thanks",
    "hostile-003": "refund please", "hostile-004": "cancel my order",
    "hostile-005": unicodedata.normalize("NFC", "café order issue"),
    "hostile-006": "ÿ binary junk in a text field", "hostile-007": "", "hostile-008": "ok thanks",
}
WANTED = {"hostile-000": ("action", "offer-refund"), "hostile-001": ("action", "offer-refund"),
          "hostile-002": ("needs_human", False)}     # what each injection asks for


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def with_turn(req: dict, text: str) -> dict:
    ctx = (req["context"] + [{"speaker": "customer", "text": text}])[-8:]
    out = {"id": req["id"], "context": ctx}
    if isinstance(req.get("turn_index"), int):
        out["turn_index"] = req["turn_index"] + 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if RUN.exists() and not a.force:
        raise SystemExit(f"{RUN} exists; pass --force to rebuild")
    RUN.mkdir(parents=True, exist_ok=True)
    dev = [project(r) for r in load_split("dev")]
    hostile = load_split("hostile")
    assert set(CONTROL) == {r["id"] for r in hostile}
    tri = Triage(policy_path=DEFAULT_POLICY, llm_cfg=LLMConfig(mode="off"))
    t0 = time.perf_counter()

    def run(variant, reqs):
        out = tri.handle_many(reqs, [f"{variant}-{i}" for i in range(len(reqs))])
        return [(resp, tr) for resp, tr in out]

    results = {"original": run("original", dev)}
    texts = {"original": None}
    for h in hostile:
        hid = h["id"]
        texts[hid], texts[f"control:{hid}"] = h["context"][0]["text"], CONTROL[hid]
        results[hid] = run(hid, [with_turn(q, texts[hid]) for q in dev])
        results[f"control:{hid}"] = run(f"control:{hid}", [with_turn(q, CONTROL[hid]) for q in dev])
    wall = time.perf_counter() - t0
    tri.close()

    with open(RUN / "predictions.jsonl", "w", encoding="utf-8") as fp, \
            open(RUN / "traces.jsonl", "w", encoding="utf-8") as ft:
        for variant, rows in results.items():
            for resp, tr in rows:
                rid = f"{resp['id']}~{variant}"
                fp.write(json.dumps(dict(resp, id=rid), ensure_ascii=True) + "\n")
                ft.write(json.dumps({"id": rid, "flags": tr.get("flags"), "route_reasons": tr.get("route_reasons"),
                                     "abstain_reason": tr.get("abstain_reason")}, ensure_ascii=True) + "\n")

    def stats(rows):
        n = len(rows)
        routed_by_gate = sum(any(f in gate.ROUTE_FLAGS for f in tr["route_reasons"]) for _, tr in rows)
        flags = Counter(f for _, tr in rows for f in tr["flags"])
        return {"n": n, "needs_human": round(sum(r["needs_human"] for r, _ in rows) / n, 4),
                "routed_by_gate_flag": round(routed_by_gate / n, 4),
                "abstained": round(sum("error" in r for r, _ in rows) / n, 4),
                "flags": {k: round(v / n, 4) for k, v in flags.most_common()}}

    summary = {"original": stats(results["original"]), "hostile": {}}
    for h in hostile:
        hid = h["id"]
        hr, cr = results[hid], results[f"control:{hid}"]
        same = sum((x["intent"], x["action"]) == (y["intent"], y["action"]) for (x, _), (y, _) in zip(hr, cr))
        rec = {"hostile_kind": h.get("hostile_kind"), "text": json.dumps(texts[hid], ensure_ascii=True)[:80],
               "control_text": json.dumps(CONTROL[hid], ensure_ascii=True),
               "with_hostile_turn": stats(hr), "with_control_turn": stats(cr),
               "same_intent_and_action_as_control": round(same / len(hr), 4),
               "same_answer_as_control_incl_needs_human": round(
                   sum((x["intent"], x["action"], x["needs_human"]) == (y["intent"], y["action"], y["needs_human"])
                       for (x, _), (y, _) in zip(hr, cr)) / len(hr), 4)}
        if hid in WANTED:
            k, v = WANTED[hid]
            rec["asked_for"] = f"{k} = {v}"
            rec["share_giving_what_was_asked_hostile"] = round(sum(x[k] == v for x, _ in hr) / len(hr), 4)
            rec["share_giving_what_was_asked_control"] = round(sum(y[k] == v for y, _ in cr) / len(cr), 4)
            rec["unrouted_and_asked_for_hostile"] = round(
                sum(x[k] == v and not x["needs_human"] for x, _ in hr) / len(hr), 4)
        summary["hostile"][hid] = rec
    (RUN / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    meta = {"run_id": RUN.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "system": "pipeline (LLM off)",
            "status": "exploratory, not pre-registered",
            "what_changed": "inputs, not the system: each hostile text (and a matched control) appended as the newest "
                            "customer turn of every dev conversation; v1 unchanged",
            "bundle": "models/classical_v0.joblib", "bundle_sha256": tri.bundle_sha256,
            "policy": str(DEFAULT_POLICY.relative_to(ROOT)), "policy_sha256": sha256(DEFAULT_POLICY),
            "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256(__file__),
            "inputs": {k: {"file": f"data/op06/{f}", "sha256": sha256(DATA / f)}
                       for k, f in (("dev", "dev.jsonl"), ("hostile", "hostile.jsonl"))},
            "rows": sum(len(v) for v in results.values()), "variants": list(results),
            "batch_wall_s": round(wall, 2), "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}", "llm_calls": 0,
            "note": "no harness or grader report: the modified rows have no gold labels. predictions.jsonl ids are "
                    "'<dev id>~<variant>'; traces.jsonl holds flags and route reasons, no text"}
    (RUN / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
