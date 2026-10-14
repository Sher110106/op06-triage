"""The v1 request path: gate, CPU model with the policy prior, optional LLM action step, policy check, routing.

`Triage.handle(obj, request_id)` -> (response, trace) never raises. The trace holds labels, flags, counts and
timings for the logs, never the customer's text. `handle_many` runs the CPU model once over a batch and gives
the same responses as `handle` row by row.

Environment (all optional; the LLM step's own settings are in triage/llm_action.py):
  TRIAGE_BUNDLE               models/classical_v0.joblib
  TRIAGE_POLICY_PATH          policy/policy.toml
  TRIAGE_SECOND_INTENT_P      0.6   flag "second_intent" when the last customer turn alone points to another flow
  TRIAGE_ROUTE_SECOND_INTENT  0     1: "second_intent" routes too
  TRIAGE_ROUTE_OFF_POLICY     1     1: "off_policy" routes
  TRIAGE_FLAG_CONF_CAP        0.5   confidence cap for rows routed by a flag
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from . import gate
from . import text as T
from .classical import DEFAULT_BUNDLE, Predictor, apply_prior
from .data import DATA, ROOT, SPLITS
from .llm_action import LLMActionStep, LLMConfig, LLMResult
from .policy import DEFAULT_POLICY, Policy, PolicyStore

log = logging.getLogger(__name__)
RETRIEVAL_PATH = ROOT / "models" / "retrieval.joblib"
LLM_POOL = 8


def _env_float(env, name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        v = float("nan")
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be a number from 0 to 1, not {raw!r}")
    return v


def _env_bool(env, name: str, default: bool) -> bool:
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be 1 or 0, not {raw!r}")


@dataclass(frozen=True)
class Settings:
    second_intent_p: float = 0.6
    route_second_intent: bool = False
    route_off_policy: bool = True
    flag_conf_cap: float = 0.5

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        return cls(second_intent_p=_env_float(env, "TRIAGE_SECOND_INTENT_P", 0.6),
                   route_second_intent=_env_bool(env, "TRIAGE_ROUTE_SECOND_INTENT", False),
                   route_off_policy=_env_bool(env, "TRIAGE_ROUTE_OFF_POLICY", True),
                   flag_conf_cap=_env_float(env, "TRIAGE_FLAG_CONF_CAP", 0.5))


@dataclass
class _Work:
    """One request on its way through the stages."""
    request_id: str
    policy: Policy
    rid: str = ""
    ms: dict = field(default_factory=lambda: {"gate": 0.0, "cpu": 0.0, "llm": 0.0})
    flags: set = field(default_factory=set)
    gate: dict | None = None                 # counts only
    row: dict | None = None                  # the cleaned model input
    out: tuple[dict, dict] | None = None     # set when the row ends early (abstention or error)
    observed: list = field(default_factory=list)
    masked: frozenset = frozenset()
    intent: str = ""
    flow: str = ""
    action: str = ""
    conf: float = 0.0
    second: list | None = None               # [intent, p] from the last customer turn alone
    llm: LLMResult | None = None


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def _rel(path: Path) -> str:
    p = path.resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def intent_flows(bundle: dict) -> dict[str, str]:
    """intent -> flow. Taken from the bundle when it holds the map, else read once from the train labels (each
    intent has exactly one flow there). Nothing is written to disk."""
    if isinstance(bundle.get("intent_flow"), dict):
        return dict(bundle["intent_flow"])
    out = {}
    with gzip.open(DATA / SPLITS["train"], "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["intent"]] = r["flow"]
    return out


def _gate_counts(obj, g: gate.GateResult) -> dict:
    raw = obj.get("context") if isinstance(obj, dict) else None
    turns = [t for t in raw if isinstance(t, dict)][-gate.MAX_TURNS:] if isinstance(raw, list) else []
    texts = [t.get("text") if isinstance(t.get("text"), str) else "" for t in turns]
    return {"turns_in": len(raw) if isinstance(raw, list) else 0, "turns": len(g.context),
            "turns_changed": sum(a != t["text"] for a, t in zip(texts, g.context)),
            "chars_in": sum(len(s) for s in texts), "chars": sum(len(t["text"]) for t in g.context)}


def _last_customer(row: dict) -> str | None:
    for t in reversed(row["context"]):
        if t["speaker"] == "customer" and t["text"].strip():
            return t["text"]
    return None


def load_llm(cfg: LLMConfig, retrieval_path=RETRIEVAL_PATH) -> tuple[LLMActionStep | None, str | None]:
    """(step, None) when cfg turns the LLM step on; (None, reason) when the retrieval index will not load, so the
    service still answers with the CPU model; (None, None) when the step is off."""
    if not cfg.enabled:
        return None, None
    try:
        from .retrieval import Retriever
        return LLMActionStep(cfg, Retriever.load(retrieval_path)), None
    except Exception as e:
        err = f"LLM step off: cannot load {_rel(Path(retrieval_path))} ({type(e).__name__}); " \
              "run scripts/build_retrieval.py"
        log.warning(err)
        return None, err


def envelope(rid: str, code: str, message: str) -> dict:
    """The abstention response: schema-valid, no intent or action claimed."""
    return {"id": rid, "intent": "unknown", "action": "none", "confidence": 0.0, "needs_human": True,
            "error": {"code": code, "message": message}}


class Triage:
    """The model, the policy store and the optional LLM step, loaded once. Safe to call from many threads."""

    def __init__(self, bundle=DEFAULT_BUNDLE, policy_path=DEFAULT_POLICY, llm_cfg: LLMConfig | None = None,
                 settings: Settings | None = None, retrieval_path=RETRIEVAL_PATH):
        self.started_at = time.time()
        self.bundle_path = Path(bundle)
        self.bundle_sha256 = _sha256(self.bundle_path)
        self.cpu = Predictor(self.bundle_path, dedupe=False)     # the gate already drops repeated sentences
        self.store = PolicyStore(policy_path)
        self.settings = settings or Settings()
        self.intent_flow = intent_flows(self.cpu.b)
        self._aidx = {str(a): j for j, a in enumerate(self.cpu.acts)}
        self._prior_cache: tuple[Policy, np.ndarray] | None = None
        self.llm_cfg = llm_cfg or LLMConfig(mode="off")
        self.llm, self.llm_error = load_llm(self.llm_cfg, retrieval_path)
        self._pool: cf.ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    @classmethod
    def from_env(cls, env=None) -> "Triage":
        env = os.environ if env is None else env
        return cls(bundle=Path(env.get("TRIAGE_BUNDLE") or DEFAULT_BUNDLE),
                   policy_path=Path(env.get("TRIAGE_POLICY_PATH") or DEFAULT_POLICY),
                   llm_cfg=LLMConfig.from_env(env), settings=Settings.from_env(env))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        if self.llm is not None and hasattr(self.llm, "close"):
            self.llm.close()

    # ------------------------------------------------------------------------- public calls

    def handle(self, obj, request_id: str = "") -> tuple[dict, dict]:
        try:
            return self._run([obj], [request_id])[0]
        except Exception as e:
            return self._failed(obj, request_id, e)

    def handle_many(self, objs, request_ids=None) -> list[tuple[dict, dict]]:
        objs = list(objs)
        rids = list(request_ids or [])[:len(objs)]
        rids += [uuid.uuid4().hex for _ in range(len(objs) - len(rids))]
        try:
            return self._run(objs, rids)
        except Exception:                                         # isolate the row that broke the batch
            return [self.handle(o, r) for o, r in zip(objs, rids)]

    def health(self) -> dict:
        policy = self.store.current()
        cfg = self.llm.cfg if self.llm is not None else self.llm_cfg
        try:
            host = urlsplit(cfg.base_url).hostname
        except ValueError:
            host = None
        breaker = getattr(getattr(self.llm, "breaker", None), "state", None)
        return {"status": "ok", "bundle": _rel(self.bundle_path), "bundle_sha256": self.bundle_sha256,
                "policy": {"version": policy.version, "path": _rel(self.store.path),
                           "last_error": self.store.last_error, "loaded_at": _iso(self.store.loaded_at)},
                "llm": {"enabled": self.llm is not None, "mode": cfg.mode, "model": cfg.model, "base_url_host": host,
                        "gate": cfg.gate, "breaker_open": breaker not in (None, "closed"), "breaker": breaker,
                        "error": self.llm_error},
                "started_at": _iso(self.started_at)}

    # ------------------------------------------------------------------------- stages

    def _run(self, objs: list, rids: list) -> list[tuple[dict, dict]]:
        policy = self.store.current()                             # one policy for the whole batch
        work = [self._gate(o, r, policy) for o, r in zip(objs, rids)]
        live = [w for w in work if w.out is None]
        if live:
            self._cpu(live, policy)
        ask = [w for w in live if self._wants_llm(w)]
        if ask:
            self._ask_llm(ask)
        return [w.out if w.out is not None else self._finish(w) for w in work]

    def _gate(self, obj, request_id: str, policy: Policy) -> _Work:
        w = _Work(request_id=request_id, policy=policy)
        t = time.perf_counter()
        try:
            err = gate.envelope_error(obj)
            g = gate.clean(obj)
            w.rid = g.id if err is None else ""
            w.flags, w.gate = set(g.flags), _gate_counts(obj, g)
            w.ms["gate"] = (time.perf_counter() - t) * 1000
            if err or g.abstain_reason:
                resp = envelope(w.rid, "invalid_input", err or g.abstain_reason)
                w.out = resp, self._trace(w, resp, ["abstained"], abstain=err or g.abstain_reason)
            else:
                w.row = {"id": g.id, "turn_index": g.turn_index, "context": g.context}
        except Exception as e:
            w.out = self._failed(obj, request_id, e)
        return w

    def _prior(self, policy: Policy) -> np.ndarray:
        cached = self._prior_cache
        if cached is None or cached[0] is not policy:             # a reloaded policy file gets a new prior
            cached = (policy, policy.prior_matrix(self.cpu.fcls, self.cpu.acts))
            self._prior_cache = cached
        return cached[1]

    def _cpu(self, live: list[_Work], policy: Policy) -> None:
        t = time.perf_counter()
        rows = [w.row for w in live]
        st = self.cpu.state(rows)
        pa, pi, pf = self.cpu.raw_probs(rows, st)
        pa = apply_prior(pa, pf, self._prior(policy))
        for k, w in enumerate(live):
            w.observed = T.observed_actions(w.row)
            w.masked = policy.masked_actions(w.observed)
            idx = [self._aidx[a] for a in w.masked if a in self._aidx]
            if idx:
                p = pa[k].copy()
                p[idx] = 0.0
                s = p.sum()
                if s > 0:
                    pa[k] = p / s
                else:                                             # nothing left: keep the unmasked answer
                    w.flags.add("all_masked")
        conf = self.cpu.confidence(pa, pi, pf, st, [r["turn_index"] for r in rows])
        last = [_last_customer(r) for r in rows]
        has = [k for k, s in enumerate(last) if s is not None]
        p2 = self.cpu.intent_probs([T.view_all({"context": [{"speaker": "customer", "text": last[k]}]})
                                    for k in has]) if has else None
        for k, w in enumerate(live):
            w.intent = str(self.cpu.icls[pi[k].argmax()])
            w.flow = self.intent_flow.get(w.intent) or str(self.cpu.fcls[pf[k].argmax()])
            w.action = str(self.cpu.acts[pa[k].argmax()])
            w.conf = float(min(1.0, max(0.0, round(float(conf[k]), 6))))
        for j, k in enumerate(has):
            w, top = live[k], int(p2[j].argmax())
            second, p = str(self.cpu.icls[top]), float(p2[j][top])
            w.second = [second, round(p, 4)]
            if p >= self.settings.second_intent_p and self.intent_flow.get(second) != w.flow:
                w.flags.add("second_intent")
        share = (time.perf_counter() - t) * 1000 / len(live)
        for w in live:
            w.ms["cpu"] = share

    def _route_flags(self) -> frozenset[str]:
        return gate.ROUTE_FLAGS | ({"second_intent"} if self.settings.route_second_intent else set())

    def _wants_llm(self, w: _Work) -> bool:
        return (w.out is None and self.llm is not None and w.conf < self.llm.cfg.gate
                and not w.flags & self._route_flags())

    def _decide(self, row: dict) -> tuple[LLMResult, float]:
        t = time.perf_counter()
        try:
            res = self.llm.decide(row)
        except Exception as e:                                    # decide should not raise; keep the CPU answer
            res = LLMResult("invalid", error=f"internal: {type(e).__name__}")
        return res, (time.perf_counter() - t) * 1000

    def _ask_llm(self, ask: list[_Work]) -> None:
        if len(ask) == 1:
            results = [self._decide(ask[0].row)]
        else:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = cf.ThreadPoolExecutor(LLM_POOL, thread_name_prefix="triage-llm")
            results = list(self._pool.map(self._decide, [w.row for w in ask]))
        for w, (res, ms) in zip(ask, results):
            w.llm, w.ms["llm"] = res, ms

    def _finish(self, w: _Work) -> tuple[dict, dict]:
        t = time.perf_counter()
        try:
            action, conf, used = w.action, w.conf, False
            r = w.llm
            if r is not None and r.status == "ok" and r.action in self._aidx and r.action not in w.masked:
                agree = 1.0 if r.action == action else 0.0
                action, conf, used = r.action, round((conf + agree) / 2, 6), True
            policy, s = w.policy, self.settings
            w.flags.update(policy.violations(action, w.flow, w.observed))
            reasons = sorted(w.flags & gate.ROUTE_FLAGS)
            if s.route_off_policy and "off_policy" in w.flags:
                reasons.append("off_policy")
            # route rules; a broken mask rule only shows up when every candidate was masked, and routes too
            reasons += sorted(f for f in w.flags if f.startswith("rule:"))
            if s.route_second_intent and "second_intent" in w.flags:
                reasons.append("second_intent")
            flagged = bool(reasons)                               # routed by a flag, not by the threshold alone
            if conf < policy.threshold_for(action):
                reasons.append("low_confidence")
            if flagged:
                conf = min(conf, s.flag_conf_cap)
            resp = {"id": w.rid, "intent": w.intent, "action": action, "confidence": float(conf),
                    "needs_human": bool(reasons)}
            llm = None if r is None else {"status": r.status, "action": r.action, "used": used,
                                          "latency_s": round(r.latency_s, 4), "cost_usd": r.cost_usd,
                                          "model": r.model, "provider": r.provider}
            cpu = {"intent": w.intent, "flow": w.flow, "action": w.action, "confidence": w.conf,
                   "second_intent": w.second, "masked": sorted(w.masked)}
            return resp, self._trace(w, resp, reasons, cpu=cpu, llm=llm, finish_ms=(time.perf_counter() - t) * 1000)
        except Exception as e:
            return self._failed({"id": w.rid}, w.request_id, e)

    # ------------------------------------------------------------------------- envelopes and traces

    def _trace(self, w: _Work, resp: dict, reasons: list, cpu=None, llm=None, abstain=None,
               finish_ms: float = 0.0) -> dict:
        ms = dict(w.ms, total=sum(w.ms.values()) + finish_ms)
        return {"request_id": w.request_id, "policy_version": w.policy.version,
                "timings_ms": {k: round(v, 3) for k, v in ms.items()}, "flags": sorted(w.flags), "gate": w.gate,
                "abstain_reason": abstain, "cpu": cpu, "llm": llm, "route_reasons": reasons,
                "final": {k: resp[k] for k in ("intent", "action", "confidence", "needs_human")}, "error": None}

    def _failed(self, obj, request_id: str, e: Exception) -> tuple[dict, dict]:
        try:
            rid = obj["id"] if gate.envelope_error(obj) is None else ""
        except Exception:
            rid = ""
        resp = envelope(rid, "internal", type(e).__name__)
        return resp, {"request_id": request_id if isinstance(request_id, str) else "",
                      "policy_version": self.store.version, "timings_ms": {}, "flags": [], "gate": None,
                      "abstain_reason": None, "cpu": None, "llm": None, "route_reasons": ["internal_error"],
                      "final": {k: resp[k] for k in ("intent", "action", "confidence", "needs_human")},
                      "error": type(e).__name__}
