"""Shared parts of the ablation runs: build the shipped pipeline, run dev + dev_noisy + hostile through it, write
results/raw/ablation-<name>/, grade with the official grader (via harness.eval grade), and compare with a reference
run by paired conversation-bootstrap differences (harness.eval.paired). No shipped file is edited; ablations change
behaviour only by subclassing or patching inside these scripts. Traces (no customer text) are gzipped."""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from harness.eval import paired, read_preds  # noqa: E402
from triage import pipeline as P  # noqa: E402
from triage.data import DATA, load_split, project  # noqa: E402
from triage.llm_action import LLMConfig, LLMResult  # noqa: E402

RAW = ROOT / "results" / "raw"
REF_CPU = RAW / "dev-pipeline-cpu"
REF_LLM = RAW / "dev-pipeline-llm-replay"
LLM_DEV = ROOT / "results/raw/llm-tail-luna_V3_none/raw_calls.jsonl"
SHIPPED_GATE = 0.642841
KEYS = ("intent_acc", "macro_f1", "action_acc", "top70_error_ratio", "noisy_gap")


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Replay:
    """Stands in for LLMActionStep: returns the recorded GPT-6 Luna answer for a row id. The same pattern as
    experiments/routing/analyse_routing.py (status rules copied unchanged); no API call is made."""

    def __init__(self, path, gate):
        self.cfg = LLMConfig(mode="on", api_key="replay", gate=gate)
        self.rec = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            p = r.get("pred") or {}
            ok = r.get("status") == "ok" and r.get("parse_status") in ("ok", "lenient_ok") and p
            if not ok:
                status = "invalid"
            elif p.get("intent") == "unknown" or p.get("action") == "none":
                status = "abstained"
            else:
                status = "ok"
            self.rec[r["id"]] = LLMResult(status, intent=p.get("intent"), action=p.get("action"),
                                          latency_s=r.get("latency_s") or 0.0, cost_usd=r.get("cost_list_actual"),
                                          model=r.get("model_returned"), provider=r.get("provider_returned"))

    def decide(self, row):
        return self.rec.get(row["id"], LLMResult("http_error", error="no recorded answer"))

    def close(self):
        pass


def build(cls=P.Triage, llm_gate: float | None = None, **kw) -> P.Triage:
    """The shipped pipeline (shipped bundle, policy and settings) with the LLM step off, or, when llm_gate is
    given, with the recorded Luna answers replayed at that gate."""
    t = cls(llm_cfg=LLMConfig(mode="off"), **kw)
    if llm_gate is not None:
        t.llm = Replay(LLM_DEV, llm_gate)
    return t


def rows():
    return load_split("dev") + load_split("noisy") + load_split("hostile")


def cpu_conf_quantile(q: float) -> float:
    """Quantile of the CPU confidence over the 2,300 labelled dev + dev_noisy rows, from the committed CPU
    reference run. The shipped LLM gate 0.642841 is the 20% quantile rounded to 6 places."""
    c = [d["cpu"]["confidence"] for d in map(json.loads, (REF_CPU / "details.jsonl").read_text().splitlines())
         if not d["id"].startswith("hostile")]
    return float(np.quantile(np.array(c), q))


def execute(name: str, t: P.Triage, ref: Path, changed: str, extra: dict | None = None,
            compare_also: dict | None = None, slices: tuple[str, ...] = ()) -> dict:
    """Run all 2,309 rows through `t`, write results/raw/ablation-<name>/, grade it and compare with `ref`."""
    out_dir = RAW / f"ablation-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = rows()
    t0 = time.perf_counter()
    out = t.handle_many([project(r) for r in data], [f"abl-{i}" for i in range(len(data))])
    wall = time.perf_counter() - t0
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as fh, gzip.open(out_dir / "traces.jsonl.gz", "wt", encoding="utf-8") as ft:
        for r, (resp, tr) in zip(data, out):
            fh.write(json.dumps(resp, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": r["id"], **tr}, ensure_ascii=False) + "\n")
    llm_calls = sum(tr.get("llm") is not None for _, tr in out)
    policy = t.store.current()
    meta = {"run_id": out_dir.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "script": f"experiments/ablations/run.py {name}", "changed": changed, "reference": rel(ref),
            "bundle": rel(t.bundle_path), "bundle_sha256": t.bundle_sha256,
            "policy": {"path": rel(t.store.path), "version": policy.version, "thresholds": policy.thresholds,
                       "alias": policy.alias},
            "settings": {"route_off_policy": t.settings.route_off_policy,
                         "route_second_intent": t.settings.route_second_intent,
                         "flag_conf_cap": t.settings.flag_conf_cap},
            "llm": ({"mode": "replay", "answers": rel(LLM_DEV), "gate": t.llm.cfg.gate, "rows_called": llm_calls}
                    if t.llm is not None else {"mode": "off"}),
            "inputs": {k: {"file": f"data/op06/{f}", "sha256": sha256(DATA / f)}
                       for k, f in (("dev", "dev.jsonl"), ("noisy", "dev_noisy.jsonl"), ("hostile", "hostile.jsonl"))},
            "rows": len(data), "batch_wall_s": round(wall, 2), "python": platform.python_version(),
            "machine": f"{platform.system()} {platform.machine()}", **(extra or {})}
    ref_pred = ref / "predictions.jsonl"
    meta["identical_to_reference"] = pred_path.read_bytes() == ref_pred.read_bytes()
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    subprocess.run([sys.executable, "-m", "harness.eval", "grade", "--pred", str(pred_path)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    res = summarise(out_dir, ref, meta, route_flags=t._route_flags())
    res["extra"] = extra or {}
    for label, other in (compare_also or {}).items():
        res[f"paired_vs_{label}"] = paired(read_preds(other / "predictions.jsonl"), read_preds(pred_path), keys=KEYS)
        res[f"paired_vs_{label}"]["reference"] = rel(other)
    for kind in slices:
        res[f"paired_{kind}_rows"] = paired_slice(ref, pred_path, kind)
    (out_dir / "ablation.json").write_text(json.dumps(res, indent=1))
    show(res)
    return res


def paired_slice(ref: Path, pred_path: Path, noise_kind: str) -> dict:
    """Paired intent and action differences on the dev_noisy rows of one noise kind (50 rows), bootstrap by
    conversation within the slice."""
    sl = [dict(r, _group="clean") for r in load_split("noisy") if r["noise_kind"] == noise_kind]
    res = paired(read_preds(ref / "predictions.jsonl"), read_preds(pred_path), rows=sl,
                 keys=("intent_acc", "action_acc"))
    res["noise_kind"] = noise_kind
    return res


def hostile_view(preds: dict, traces: dict | None = None, route_flags=P.gate.ROUTE_FLAGS) -> dict:
    """The 9 hostile rows: which ones get an answer that is not sent to a person (needs_human false), and which
    would reach the LLM step at the shipped gate under this run's routing flags (LLM answers for hostile rows
    were never recorded, so they cannot be replayed)."""
    hrows = load_split("hostile")
    out = []
    for r in hrows:
        p = preds[r["id"]]
        tr = (traces or {}).get(r["id"]) or {}
        cpu = tr.get("cpu") or {}
        out.append({"id": r["id"], "kind": r["hostile_kind"], "intent": p["intent"], "action": p["action"],
                    "confidence": p["confidence"], "needs_human": p["needs_human"], "abstained": "error" in p,
                    "flags": tr.get("flags"), "route_reasons": tr.get("route_reasons"),
                    "cpu_confidence": cpu.get("confidence"),
                    "would_call_llm_at_shipped_gate": (cpu.get("confidence") is not None
                                                       and cpu["confidence"] < SHIPPED_GATE
                                                       and not set(tr.get("flags") or []) & route_flags)})
    return {"rows": len(out), "needs_human_false": sum(not x["needs_human"] for x in out),
            "abstained": sum(x["abstained"] for x in out), "detail": out}


def summarise(out_dir: Path, ref: Path, meta: dict, route_flags=P.gate.ROUTE_FLAGS) -> dict:
    rep = json.loads((out_dir / "report.json").read_text())
    o, e = rep["official"], rep["extras"]
    preds = read_preds(out_dir / "predictions.jsonl")
    traces = {}
    tpath = out_dir / "traces.jsonl.gz"
    if tpath.exists():
        with gzip.open(tpath, "rt", encoding="utf-8") as fh:
            traces = {d["id"]: d for d in map(json.loads, fh)}
    refp = read_preds(ref / "predictions.jsonl")
    lab = [i for i in refp if not i.startswith("hostile")]
    differ = {k: sum(refp[i].get(k) != preds[i].get(k) for i in lab) for k in ("intent", "action", "needs_human",
                                                                                  "confidence")}
    differ["max_confidence_difference"] = round(max(abs(refp[i]["confidence"] - preds[i]["confidence"]) for i in lab), 6)
    return {"run_id": out_dir.name, "changed": meta.get("changed"), "reference": rel(ref),
            "identical_to_reference": meta.get("identical_to_reference"),
            "llm": meta.get("llm"), "bundle_sha256": meta.get("bundle_sha256"),
            "harness_matches_grader": rep.get("harness_matches_grader"),
            "official": {k: o[k] for k in ("intent_acc", "macro_f1", "action_acc", "action_acc_clean",
                                           "action_acc_noisy", "intent_acc_noisy", "noisy_gap", "top70_error_ratio",
                                           "distinct_confidences", "policy_violation", "gold_policy_violation",
                                           "schema_valid_pct")},
            "bars_pass": all(v for v in rep["bars"].values() if v is not None),
            "policy_violation_literal": o["policy_violation"], "policy_violation_alias": e["policy_violation_alias"],
            "needs_human_rate": e["needs_human_rate"], "action_acc_when_not_routed": e["action_acc_when_not_routed"],
            "action_acc_at_coverage": e["action_acc_at_coverage"], "noise_kinds": e["noise_kinds"],
            "ci95": {k: e["ci95_conversation_bootstrap"].get(k) for k in KEYS},
            "labelled_rows_differing_from_reference": differ,
            "paired_vs_reference": paired(refp, preds, keys=KEYS),
            "hostile": hostile_view(preds, traces, route_flags)}


def show(res: dict) -> None:
    o, pm = res["official"], res["paired_vs_reference"]["metrics"]
    d = {k: (pm[k]["diff"], pm[k]["ci95"]) for k in KEYS if k in pm}
    print(json.dumps({"run": res["run_id"], "identical_to_reference": res["identical_to_reference"],
                      "intent": o["intent_acc"], "f1": o["macro_f1"], "action": o["action_acc"],
                      "gap": o["noisy_gap"], "top70": o["top70_error_ratio"],
                      "viol": [res["policy_violation_literal"], res["policy_violation_alias"]],
                      "needs_human": res["needs_human_rate"], "hostile_unrouted": res["hostile"]["needs_human_false"],
                      "diff_vs_ref": d}))
