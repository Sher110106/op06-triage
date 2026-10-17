"""Train-only, conversation-held-out comparison. See PROTOCOL.md before running.

python experiments/refinement/run_state_blend.py prepare
python experiments/refinement/run_state_blend.py select
python experiments/refinement/run_state_blend.py confirm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triage import classical as C
from triage import text as T
from triage.data import conv_id, load_split, project
from triage.policy import load as load_policy

OUT = ROOT / 'results' / 'refinement-20261017'
MODEL = ROOT / 'models' / 'refinement-control.joblib'
ALPHAS = [0., .15, .30, .50]
GUARDS = [0., .80, .90, .95]


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1) + '\n')


def partitions():
    rows = load_split('train')
    groups = np.array([conv_id(r['id']) for r in rows])
    rem, final = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=20261017).split(rows, groups=groups))
    fit_i, sel_i = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=20261018).split(rem, groups=groups[rem]))
    parts = [[rows[i] for i in idx] for idx in (rem[fit_i], rem[sel_i], final)]
    sets = [set(conv_id(r['id']) for r in p) for p in parts]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(map(len, parts)) == len(rows)
    return parts


class StateTable:
    """Empirical transitions with fixed hierarchical smoothing, marginalized over predicted intents."""

    def __init__(self, fit, acts, icls, strength=5.):
        self.acts, self.icls, self.strength = list(acts), list(icls), strength
        aidx = {str(a): j for j, a in enumerate(acts)}
        n = len(acts)
        self.global_counts = np.zeros(n)
        self.last, self.flow, self.intent = {}, {}, {}
        self.intent_flow = {}
        for r in fit:
            last = T.last_obs(project(r))
            self.intent_flow[r['intent']] = r['flow']
            j = aidx[r['action']]
            self.global_counts[j] += 1
            for table, key in ((self.last, last), (self.flow, (r['flow'], last)), (self.intent, (r['intent'], last))):
                table.setdefault(key, np.zeros(n))[j] += 1

    def _shrink(self, counts, parent):
        if counts is None:
            return parent
        return (counts + self.strength * parent) / (counts.sum() + self.strength)

    def probabilities(self, rows, pi):
        g = self.global_counts / self.global_counts.sum()
        result = []
        for row, probs in zip(rows, pi):
            last = T.last_obs(row)
            lp = self._shrink(self.last.get(last), g)
            conditional = []
            for intent in self.icls:
                fp = self._shrink(self.flow.get((self.intent_flow[intent], last)), lp)
                conditional.append(self._shrink(self.intent.get((intent, last)), fp))
            result.append(probs @ np.asarray(conditional))
        p = np.asarray(result)
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1.)
        return p


def data(pm, rows, table):
    assert list(pm.icls) == table.icls, 'State table and predictor intent class order differ'
    assert list(pm.acts) == table.acts, 'State table and predictor action class order differ'
    x = [pm._sanitise(project(r)) for r in rows]
    st = pm.state(x)
    raw, pi, pf = pm.raw_probs(x, st)
    return dict(rows=rows, x=x, st=st, raw=raw, pi=pi, pf=pf,
                q=table.probabilities(x, pi), y=np.array([r['action'] for r in rows]),
                groups=np.array([conv_id(r['id']) for r in rows]))


def action_probs(pm, d, alpha):
    return C.apply_prior((1-alpha)*d['raw'] + alpha*d['q'], d['pf'], pm.b['perm'])


def correctness(pm, d, pa, oof=False):
    features = C._conf_features(pa, d['pi'], d['pf'], d['st'], [r.get('turn_index', 0) for r in d['rows']])
    y = (pm.acts[pa.argmax(1)] == d['y']).astype(int)
    clf = LogisticRegression(C=1., max_iter=1000)
    if oof:
        scores = cross_val_predict(clf, features, y, groups=d['groups'], cv=GroupKFold(5), method='predict_proba')[:, 1]
    else:
        scores = None
    clf.fit(features, y)
    return clf, scores, features


def threshold(conf, ok, risk):
    best = (float('inf'), 0.)
    for t in np.unique(np.r_[0., conf, 1.01]):
        route = conf < t
        score = float(route.sum() + risk * np.sum(~ok & ~route))
        if score < best[0]:
            best = (score, float(t))
    return best[1]


def facts(pm, d, pa, conf, policy, table):
    predicted = pm.acts[pa.argmax(1)]
    intents = pm.icls[d['pi'].argmax(1)]
    flow = [table.intent_flow[i] for i in intents]
    high = np.array([policy.risk.get(a) == 'high' for a in predicted])
    forced = np.array([not policy._permitted(a, f) for a, f in zip(predicted, flow)])
    legal_gold = np.array([a in (set(r['permitted_actions']) | {policy.alias[x] for x in r['permitted_actions'] if x in policy.alias})
                           for a, r in zip(predicted, d['rows'])])
    support = np.sum(d['pf'] * policy.prior_matrix(pm.fcls, predicted).T, axis=1)
    return dict(action=predicted, intent=intents, ok=predicted == d['y'], conf=conf,
                high=high, forced=forced, legal_gold=legal_gold, support=support)


def summary(f, cuts, guard):
    route = f['forced'] | (f['conf'] < np.where(f['high'], cuts['high'], cuts['low'])) | (f['high'] & (f['support'] < guard))
    wrong_auto = ~f['ok'] & ~route
    forbidden_auto = ~f['legal_gold'] & ~route
    cost = route.astype(float) + wrong_auto*np.where(f['high'], 10., 3.)
    total = cost + forbidden_auto*10.
    overall_error = 1-f['ok'].mean()
    cut = np.sort(f['conf'])[::-1][max(0, int(round(.7*len(route)))-1)]
    top = f['conf'] >= cut
    return dict(rows=len(route), action_acc=float(f['ok'].mean()), review_rate=float(route.mean()),
                auto_accuracy=float(f['ok'][~route].mean()) if (~route).any() else None,
                auto_forbidden_count=int(forbidden_auto.sum()), auto_high_risk_forbidden_count=int((forbidden_auto & f['high']).sum()),
                original_cost_per100=float(cost.mean()*100), policy_cost_per100=float(total.mean()*100),
                top70_error_ratio=float((1-f['ok'][top].mean())/overall_error) if overall_error else 0.,
                _route=route, _cost=cost, _policy_cost=total, _forbidden_auto=forbidden_auto)


def plain(s):
    return {k:v for k,v in s.items() if not k.startswith('_')}


def prepare():
    fit, selection, confirmation = partitions()
    write('partitions.json', {k:dict(rows=len(v), groups=sorted(set(conv_id(r['id']) for r in v)))
                              for k,v in zip(('fit','selection','confirmation'), (fit, selection, confirmation))})
    print('partitions:', [len(v) for v in (fit,selection,confirmation)], flush=True)
    original = C.folds
    C.folds = lambda: (fit, selection)
    try:
        C.train(MODEL, log=lambda s: print(s, flush=True))
    finally:
        C.folds = original


def select():
    fit, selection, _ = partitions()
    pm = C.Predictor(MODEL)
    table = StateTable(fit, pm.acts, pm.icls)
    joblib.dump(table, OUT/'state_table.joblib')
    d = data(pm, selection, table)
    scores = [{ 'alpha':a, 'accuracy':float(np.mean(pm.acts[action_probs(pm,d,a).argmax(1)] == d['y']))} for a in ALPHAS]
    chosen = max(scores, key=lambda s:(s['accuracy'], -s['alpha']))['alpha']
    policy = load_policy()
    results, fitted = {}, {}
    for name, alpha in (('reference',0.), ('candidate',chosen)):
        pa = action_probs(pm,d,alpha)
        clf, oof, _ = correctness(pm,d,pa,oof=True)
        f = facts(pm,d,pa,oof,policy,table)
        cuts = {tier:threshold(oof[(f['high']==is_high)&~f['forced']], f['ok'][(f['high']==is_high)&~f['forced']],risk)
                for tier,is_high,risk in (('high',True,10.),('low',False,3.))}
        guards = [dict(guard=g,**plain(summary(f,cuts,g))) for g in GUARDS]
        guard = min(guards,key=lambda s:(s['policy_cost_per100'],s['guard']))['guard']
        results[name] = dict(alpha=alpha, thresholds=cuts, guard=guard, variants=guards)
        fitted[name] = clf
    joblib.dump(fitted,OUT/'confidence.joblib')
    frozen = dict(seed=20261017, alpha_grid=scores, systems=results,
                  caveat='Selection calibration is grouped OOF, but temperature and alpha choice used selection outcomes. Confirmation remains excluded.')
    write('selection.json',frozen)
    print(json.dumps(frozen,indent=1),flush=True)


def confirm():
    config=json.loads((OUT/'selection.json').read_text())
    _,_,rows=partitions()
    pm=C.Predictor(MODEL)
    table=joblib.load(OUT/'state_table.joblib')
    clfs=joblib.load(OUT/'confidence.joblib')
    d=data(pm,rows,table)
    policy=load_policy()
    arrays, reports = {}, {}
    for name,s in config['systems'].items():
        pa=action_probs(pm,d,s['alpha'])
        features=C._conf_features(pa,d['pi'],d['pf'],d['st'],[r.get('turn_index',0) for r in rows])
        conf=clfs[name].predict_proba(features)[:,1]
        f=facts(pm,d,pa,conf,policy,table)
        reports[name]={}
        for guard_name,guard in (('without_guard',0.),('selected_guard',s['guard'])):
            out=summary(f,s['thresholds'],guard)
            reports[name][guard_name]=plain(out)
            arrays[(name,guard_name)] = dict(ok=f['ok'], **out)
        with (OUT/f'confirmation_{name}.jsonl').open('w') as file:
            routed=arrays[(name,'selected_guard')]['_route']
            for r,it,act,c,nh in zip(rows,f['intent'],f['action'],conf,routed):
                file.write(json.dumps(dict(id=r['id'],intent=str(it),action=str(act),confidence=float(c),needs_human=bool(nh)))+'\n')
    groups=np.unique(d['groups'])
    indices={g:np.flatnonzero(d['groups']==g) for g in groups}
    rng=np.random.default_rng(17)
    names=('action_delta','original_cost_delta_per100','policy_cost_delta_per100','review_delta','forbidden_auto_rate_delta')
    contrasts={
        'combined': (arrays[('reference','without_guard')], arrays[('candidate','selected_guard')]),
        'reference_guard_only': (arrays[('reference','without_guard')], arrays[('reference','selected_guard')]),
        'candidate_guard_only': (arrays[('candidate','without_guard')], arrays[('candidate','selected_guard')]),
    }
    draws={c:{k:[] for k in names} for c in contrasts}
    ref=arrays[('reference','without_guard')]
    new=arrays[('candidate','selected_guard')]
    for _ in range(2000):
        ix=np.concatenate([indices[g] for g in rng.choice(groups,len(groups),replace=True)])
        for label,(b,n) in contrasts.items():
            draws[label]['action_delta'].append(float((n['ok'][ix].astype(float)-b['ok'][ix]).mean()))
            draws[label]['original_cost_delta_per100'].append(float((n['_cost'][ix]-b['_cost'][ix]).mean()*100))
            draws[label]['policy_cost_delta_per100'].append(float((n['_policy_cost'][ix]-b['_policy_cost'][ix]).mean()*100))
            draws[label]['review_delta'].append(float((n['_route'][ix].astype(float)-b['_route'][ix]).mean()))
            draws[label]['forbidden_auto_rate_delta'].append(float((n['_forbidden_auto'][ix].astype(float)-b['_forbidden_auto'][ix]).mean()))
    intervals={label:{k:list(map(float,np.quantile(v,[.025,.975]))) for k,v in values.items()} for label,values in draws.items()}
    ci=intervals['combined']
    guard_ref=arrays[('reference','selected_guard')]
    delta=new['action_acc']-ref['action_acc']
    result=dict(partition='confirmation, excluded from all fitting and current-round selection',
                rows=len(rows),conversations=len(groups),systems=reports,ci95_paired_conversation_bootstrap=ci,
                contrast_ci95=intervals,
                action_change_pass=bool(delta>=.01 and ci['action_delta'][0]>0),
                routing_change_pass=bool(new['auto_forbidden_count']<ref['auto_forbidden_count'] and
                    new['original_cost_per100']-ref['original_cost_per100']<=2 and new['review_rate']-ref['review_rate']<=.05),
                reference_guard_only_pass=bool(guard_ref['auto_forbidden_count']<ref['auto_forbidden_count'] and
                    guard_ref['original_cost_per100']-ref['original_cost_per100']<=2 and
                    guard_ref['review_rate']-ref['review_rate']<=.05))
    write('confirmation.json',result)
    print(json.dumps(result,indent=1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','select','confirm'))
    args=parser.parse_args()
    globals()[args.command]()
