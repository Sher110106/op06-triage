"""Fold-B confirmation runner: run_llm.py with fold-B rows (rows_B.jsonl) and fold-A-only retrieval (retrieval_B.json).
Rows come from --ids-file (ids_luna.json or ids_haiku.json). Output: results/raw/llm-confirm-<tag>/raw_calls.jsonl,
plus one line per API attempt in the ledger shared with run_llm.py (results/raw/llm-ledger/). The runs of 14 Oct saved
stdout to results/raw/llm-confirm-<tag>/stdout.log.

Original docstring: run one model x variant over dev + dev_noisy at concurrency 8 (ported from the 13 Oct probe runner).
One output jsonl per run under runs/, one ledger line per API attempt (failed ones too). Keys come only from
the environment; nothing here prints or stores them. An interrupted run resumes: rows already answered are skipped.

  python run_llm.py --model claude-haiku-5-5 --variant V3 --rows smoke --tag haiku_V3_low_smoke --effort low
  python run_llm.py --model openai/gpt-6-luna-20260922 --variant V3 --rows all --tag luna_V3_none \
      --reasoning '{"effort":"none"}' --provider-order openai --no-fallbacks
"""
from __future__ import annotations

import argparse
import sys
import concurrent.futures as cf
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.request

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_tail"))
import llm_common as C                     # same prompts, parser, prices; the shared ledger keeps spend cumulative

HERE = Path(__file__).resolve().parent
RUN_PREFIX = "llm-confirm"

CAPS = {"anthropic": 2.90, "openrouter": 3.40}          # cumulative with the probe; stop a little under the $3.00 / $3.50 caps
BLOCKED = {"openai/gpt-5.6-terra"}                       # user decision: never run
OR_FLOOR = 1.20


# ----------------------------------------------------------------------------- subsets

def subsets():
    ids = json.loads((HERE / "ids_luna.json").read_text())
    return {"all": ids, "smoke": ids[:5]}


# ----------------------------------------------------------------------------- spend tracking

class Spend:
    def __init__(self, provider):
        self.provider = provider
        self.total = C.ledger_spend(provider)
        self.lock = threading.Lock()

    def add(self, x):
        with self.lock:
            self.total += x or 0.0

    def ok(self):
        return self.total < CAPS[self.provider]


def or_key_status():
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    with urllib.request.urlopen(req, timeout=30) as resp:
        d = json.loads(resp.read())["data"]
    keep = {k: d.get(k) for k in ("limit", "limit_remaining", "usage", "usage_daily", "is_free_tier")}
    C.KEYCHECK.parent.mkdir(parents=True, exist_ok=True)
    with open(C.KEYCHECK, "a") as fh:
        fh.write(json.dumps(dict(keep, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
    return keep


# ----------------------------------------------------------------------------- calls

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def call_anthropic(client, static, dyn, cfg):
    import anthropic
    kw = dict(model=cfg["model"], max_tokens=cfg.get("max_tokens", 4096),
              system=[{"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}],
              messages=[{"role": "user", "content": dyn}])
    oc = {"format": {"type": "json_schema", "schema": C.output_schema()}}
    if cfg.get("effort"):
        oc["effort"] = cfg["effort"]
    kw["output_config"] = oc
    if cfg.get("thinking") == "disabled":
        kw["thinking"] = {"type": "disabled"}
    t0 = time.perf_counter()
    try:
        r = client.messages.create(**kw)
    except anthropic.APIStatusError as e:
        return {"status": "http_error", "http_status": e.status_code, "error": str(e.message)[:300],
                "latency_s": time.perf_counter() - t0}
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        return {"status": "conn_error", "http_status": None, "error": type(e).__name__, "latency_s": time.perf_counter() - t0}
    lat = time.perf_counter() - t0
    u = r.usage
    text = next((b.text for b in r.content if b.type == "text"), None)
    inp, out = u.input_tokens or 0, u.output_tokens or 0
    cr, cw = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
    p = C.PRICES[cfg["model"]]
    return {"status": "ok", "latency_s": lat, "text": text, "stop_reason": r.stop_reason,
            "model_returned": r.model, "provider_returned": "anthropic", "request_id": r._request_id,
            "usage": {"input_tokens": inp, "output_tokens": out, "cache_read": cr, "cache_write": cw,
                      "input_total": inp + cr + cw,
                      "thinking_blocks": sum(1 for b in r.content if b.type == "thinking")},
            "cost_list_uncached": ((inp + cr + cw) * p["in"] + out * p["out"]) / 1e6,
            "cost_list_actual": (inp * p["in"] + cr * p["cache_read"] + cw * p["cache_write"] + out * p["out"]) / 1e6,
            "cost_reported": None}


def _value_logprob(lp_content, field):
    """Sum of token logprobs over the characters of a JSON string value, plus the first token's logprob."""
    if not lp_content:
        return None
    text, spans = "", []
    for t in lp_content:
        s = len(text); text += t["token"]; spans.append((s, len(text), t["logprob"]))
    m = re.search(r'"%s"\s*:\s*"([^"]*)"' % field, text)
    if not m:
        return None
    a, b = m.span(1)
    toks = [lp for (s, e, lp) in spans if e > a and s < b]
    if not toks:
        return None
    return {"sum": sum(toks), "first": toks[0], "n": len(toks)}


def call_openrouter(client, static, dyn, cfg):
    import openai
    extra = {"provider": {"require_parameters": True}, "usage": {"include": True}}
    if cfg.get("reasoning") is not None:
        extra["reasoning"] = cfg["reasoning"]
    if cfg.get("provider_order"):
        extra["provider"]["order"] = cfg["provider_order"]
    if cfg.get("no_fallbacks"):
        extra["provider"]["allow_fallbacks"] = False
    if cfg.get("provider_ignore"):
        extra["provider"]["ignore"] = cfg["provider_ignore"]
    kw = dict(model=cfg["model"],
              messages=[{"role": "system", "content": static}, {"role": "user", "content": dyn}],
              response_format={"type": "json_schema",
                               "json_schema": {"name": "triage", "strict": True, "schema": C.output_schema()}},
              max_tokens=cfg.get("max_tokens", 2048), extra_body=extra)
    if cfg.get("temperature") is not None:
        kw["temperature"] = cfg["temperature"]
    if cfg.get("logprobs"):
        kw["logprobs"] = True
        kw["top_logprobs"] = 1
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(**kw)
    except openai.APIStatusError as e:
        body = None
        try:
            body = json.dumps(e.body)[:400]
        except Exception:
            pass
        return {"status": "http_error", "http_status": e.status_code, "error": (body or str(e))[:400],
                "latency_s": time.perf_counter() - t0}
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        return {"status": "conn_error", "http_status": None, "error": type(e).__name__, "latency_s": time.perf_counter() - t0}
    lat = time.perf_counter() - t0
    d = r.model_dump()
    if d.get("error"):
        return {"status": "api_error_body", "http_status": None, "error": json.dumps(d["error"])[:400], "latency_s": lat}
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = msg.get("content")
    u = d.get("usage") or {}
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    cwrite = (u.get("prompt_tokens_details") or {}).get("cache_write_tokens") or 0
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    p = C.PRICES[cfg["model"]]
    actual = ((pt - cached - cwrite) * p["in"] + cached * (p["cache_read"] or p["in"])
              + cwrite * (p["cache_write"] or p["in"]) + ct * p["out"]) / 1e6
    lp = None
    if cfg.get("logprobs"):
        content = ((ch.get("logprobs") or {}).get("content")) or []
        lp = {"action": _value_logprob(content, "action"), "intent": _value_logprob(content, "intent"),
              "n_tokens": len(content)}
    return {"status": "ok", "latency_s": lat, "text": text, "stop_reason": ch.get("finish_reason"),
            "model_returned": d.get("model"), "provider_returned": d.get("provider"), "request_id": d.get("id"),
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cached_tokens": cached, "cache_write_tokens": cwrite,
                      "reasoning_tokens": rt, "input_total": pt, "output_tokens": ct},
            "cost_list_uncached": (pt * p["in"] + ct * p["out"]) / 1e6, "cost_list_actual": actual,
            "cost_reported": u.get("cost"), "logprobs": lp,
            "reasoning_present": bool(msg.get("reasoning"))}


# ----------------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True, choices=["V1", "V2", "V3"])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--effort")
    ap.add_argument("--thinking", choices=["adaptive", "disabled"])
    ap.add_argument("--reasoning", help="OpenRouter reasoning object as JSON")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--logprobs", action="store_true")
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--provider-order", help="comma list of OpenRouter provider slugs")
    ap.add_argument("--provider-ignore", help="comma list of OpenRouter provider names to skip")
    ap.add_argument("--no-fallbacks", action="store_true", help="OpenRouter: never route outside --provider-order")
    ap.add_argument("--ids-file", help="json list of row ids (overrides --rows subset lookup)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt-version", default="p1", choices=["p1", "p2"])
    a = ap.parse_args()
    if a.model in BLOCKED:
        sys.exit(f"{a.model} is blocked by user decision")
    provider = "anthropic" if a.model.startswith("claude") else "openrouter"
    cfg = {"model": a.model, "effort": a.effort, "thinking": a.thinking, "temperature": a.temperature,
           "logprobs": a.logprobs, "reasoning": json.loads(a.reasoning) if a.reasoning else None,
           "provider_order": a.provider_order.split(",") if a.provider_order else None,
           "provider_ignore": a.provider_ignore.split(",") if a.provider_ignore else None,
           "no_fallbacks": a.no_fallbacks}
    if a.max_tokens:
        cfg["max_tokens"] = a.max_tokens
    out_path = C.run_file(RUN_PREFIX, a.tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():                                   # resume: keep rows that already have an answer
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add(rec["id"])
        kept = [l for l in out_path.read_text(encoding="utf-8").splitlines() if json.loads(l)["id"] in done]
        out_path.write_text("".join(l + "\n" for l in kept))
        print(f"resuming: {len(done)} rows already answered", flush=True)

    spend = Spend(provider)
    if not spend.ok():
        sys.exit(f"{provider} spend {spend.total:.4f} at cap")
    if provider == "openrouter":
        ks = or_key_status()
        print("key before:", {k: ks[k] for k in ("limit_remaining", "usage")}, flush=True)
        if ks["limit_remaining"] is not None and ks["limit_remaining"] < OR_FLOOR:
            sys.exit("OpenRouter limit_remaining below floor; stopping")
        import openai
        client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                               max_retries=0, timeout=180.0)
        call = call_openrouter
    else:
        import anthropic
        client = anthropic.Anthropic(max_retries=0, timeout=180.0)
        call = call_anthropic

    rows_all = {}
    for line in (HERE / "rows_B.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows_all[r["id"]] = r
    ids = json.loads(open(a.ids_file).read()) if a.ids_file else subsets()[a.rows]
    ids = [i for i in ids if i not in done]
    retr = json.loads((HERE / "retrieval_B.json").read_text())
    trd = {r["id"]: r for r in C.data()["train"]}
    static = C.static_prompt(a.prompt_version)
    cfg["prompt_version"] = a.prompt_version
    (C.WORK / "prompts").mkdir(exist_ok=True)
    (C.WORK / "prompts" / f"static_prompt_{a.prompt_version}.txt").write_text(static)
    wlock = threading.Lock()
    stop = threading.Event()

    def one(rid):
        row = rows_all[rid]
        vis = C.strip_gold(row)
        exs = [trd[t] for t, _ in retr[rid]] if a.variant in ("V2", "V3") else None
        dyn, nonce = C.dynamic_prompt(vis, a.variant, exs)
        t_start = time.perf_counter()
        res = None
        for attempt in range(1, 5):
            if stop.is_set() or not spend.ok():
                res = {"status": "skipped_budget"}
                break
            res = call(client, static, dyn, cfg)
            cost = res.get("cost_reported")
            cost = cost if cost is not None else res.get("cost_list_actual")
            spend.add(cost)
            C.ledger_write({"provider": provider, "model": a.model, "tag": a.tag, "variant": a.variant, "row_id": rid,
                            "attempt": attempt, "status": res["status"], "http_status": res.get("http_status"),
                            "error": res.get("error"), "usage": res.get("usage"),
                            "cost_list_uncached": res.get("cost_list_uncached"),
                            "cost_list_actual": res.get("cost_list_actual"), "cost_reported": res.get("cost_reported"),
                            "latency_s": round(res.get("latency_s", 0), 3),
                            "model_returned": res.get("model_returned"), "provider_returned": res.get("provider_returned")})
            if res["status"] == "ok":
                break
            if res.get("http_status") not in RETRY_STATUS and res["status"] != "conn_error":
                break
            time.sleep(2 ** attempt + random.random())
        pred, pstatus = (C.parse_output(res.get("text")) if res.get("status") == "ok" else (None, "call_failed"))
        rec = {"id": rid, "group": "foldB", "noise_kind": None, "tag": a.tag,
               "model": a.model, "variant": a.variant, "cfg": cfg, "nonce": nonce,
               "status": res.get("status"), "parse_status": pstatus, "pred": pred,
               "latency_s": res.get("latency_s"), "latency_total_s": time.perf_counter() - t_start,
               **{k: res.get(k) for k in ("text", "stop_reason", "model_returned", "provider_returned", "request_id",
                                          "usage", "cost_list_uncached", "cost_list_actual", "cost_reported",
                                          "logprobs", "reasoning_present", "http_status", "error")}}
        with wlock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        recs = list(ex.map(one, ids))
    wall = time.time() - t0
    ok = [r for r in recs if r["status"] == "ok"]
    lat = sorted(r["latency_s"] for r in ok)
    pct = lambda q: round(lat[min(len(lat) - 1, int(math.ceil(q * len(lat))) - 1)], 2) if lat else None
    summ = {"tag": a.tag, "rows": len(ids), "ok": len(ok),
            "parse": dict(__import__("collections").Counter(r["parse_status"] for r in recs)),
            "p50": pct(0.5), "p95": pct(0.95), "wall_s": round(wall, 1),
            "cost_list_uncached": round(sum(r["cost_list_uncached"] or 0 for r in recs), 4),
            "cost_list_actual": round(sum(r["cost_list_actual"] or 0 for r in recs), 4),
            "cost_reported": round(sum(r["cost_reported"] or 0 for r in recs), 4),
            "models_returned": dict(__import__("collections").Counter(r["model_returned"] for r in recs)),
            "providers_returned": dict(__import__("collections").Counter(r["provider_returned"] for r in recs)),
            "provider_spend_total": round(spend.total, 4)}
    print(json.dumps(summ), flush=True)
    with open(C.RUN_SUMMARIES, "a") as fh:            # shared with run_llm.py since the 14 Oct move
        fh.write(json.dumps(summ) + "\n")
    if provider == "openrouter":
        ks = or_key_status()
        print("key after:", {k: ks[k] for k in ("limit_remaining", "usage")}, flush=True)


if __name__ == "__main__":
    main()
