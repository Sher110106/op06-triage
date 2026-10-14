"""Leakage probe: the pipeline must give identical answers whether or not gold-only fields are present.
Runs Triage (LLM step off, then a recording stub LLM so the LLM path is exercised) on dev + dev_noisy + hostile:
(a) projected rows, (b) full rows with gold fields, (c) full rows with gold fields set to wrong values."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triage import pipeline as P
from triage.data import load_split, project
from triage.llm_action import LLMConfig, LLMResult

rows = load_split("dev") + load_split("noisy") + load_split("hostile")
GOLD = ("intent", "action", "flow", "permitted_actions", "noise_kind", "hostile_kind")
rng = random.Random(0)
intents = sorted({r["intent"] for r in rows if r.get("intent")}); acts = sorted({r["action"] for r in rows if r.get("action")})
def scramble(r):
    s = dict(r)
    s["intent"] = rng.choice(intents); s["action"] = rng.choice(acts); s["flow"] = "storewide_query"
    s["permitted_actions"] = ["offer-refund"]; s["noise_kind"] = "typo"; s["hostile_kind"] = "injection"
    return s

class Rec:
    """Stub LLM step: records which keys reach it and always answers the same fixed action."""
    def __init__(self):
        self.cfg = LLMConfig(mode="on", api_key="stub", gate=0.642841); self.keys = set(); self.calls = 0
    def decide(self, row):
        self.calls += 1; self.keys |= set(row)
        return LLMResult("ok", intent="unknown_x", action="pull-up-account", latency_s=0.0, cost_usd=0.0)
    def close(self): pass
    def health(self): return {}

out = {}
for mode in ("off", "stub"):
    t = P.Triage(llm_cfg=LLMConfig(mode="off"))
    rec = None
    if mode == "stub":
        rec = Rec(); t.llm = rec
    res = {}
    for name, f in (("projected", project), ("full", lambda r: dict(r)), ("scrambled", scramble)):
        res[name] = [resp for resp, _ in t.handle_many([f(r) for r in rows])]
    diff_full = sum(a != b for a, b in zip(res["projected"], res["full"]))
    diff_scr = sum(a != b for a, b in zip(res["projected"], res["scrambled"]))
    out[mode] = {"rows": len(rows), "differ_projected_vs_full": diff_full, "differ_projected_vs_scrambled": diff_scr,
                 "llm_calls": rec.calls if rec else 0, "keys_seen_by_llm_step": sorted(rec.keys) if rec else None}
    t.close()
print(json.dumps(out, indent=1))
(ROOT / "results/audit/leak_test.json").write_text(json.dumps(out, indent=1) + "\n")
