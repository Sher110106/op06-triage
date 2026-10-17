"""Launch a real HTTP service and grade/load-test without overwriting submitted files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default=str(ROOT))
    p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=('cpu','hybrid'),default='cpu')
    p.add_argument('--key-file')
    a=p.parse_args()
    repo=Path(a.repo).resolve()
    out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port=s.getsockname()[1]
    env={k:v for k,v in os.environ.items() if k not in ('OPENROUTER_API_KEY','ANTHROPIC_API_KEY','TRIAGE_LLM_API_KEY')}
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PORT=str(port),TRIAGE_LLM='off')
    if a.mode=='hybrid':
        if not a.key_file:
            raise SystemExit('Hybrid mode requires an external --key-file')
        path=Path(a.key_file).resolve()
        if path.is_relative_to(ROOT):
            raise SystemExit('Credential file must be outside the repository')
        env['TRIAGE_LLM_API_KEY']=path.read_text().strip()
        env['TRIAGE_LLM']='on'
    with (out/'service.log').open('w') as log:
        server=subprocess.Popen([sys.executable,'-m','uvicorn','triage.service:app','--host','127.0.0.1',
                                 '--port',str(port),'--workers','1','--limit-concurrency','256',
                                 '--timeout-keep-alive','5','--no-access-log'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            cmd=[sys.executable,str(ROOT/'scripts/loadtest.py'),'--url',f'http://127.0.0.1:{port}',
                 '--concurrency','8','--warmup','20','--label',out.name,'--out',str(out/'latency.json'),
                 '--pred-out',str(out/'predictions.jsonl')]
            # Client and grader never need the API credential.
            client_env={k:v for k,v in env.items() if k!='TRIAGE_LLM_API_KEY'}
            with (out/'loadtest.log').open('w') as file:
                subprocess.run(cmd,cwd=repo,env=client_env,stdout=file,stderr=subprocess.STDOUT,check=True,timeout=900)
            with (out/'grade.log').open('w') as file:
                subprocess.run([sys.executable,'-m','harness.eval','grade','--pred',str(out/'predictions.jsonl'),
                                '--out',str(out/'report.json')],cwd=ROOT,env=client_env,stdout=file,
                               stderr=subprocess.STDOUT,check=True,timeout=180)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
    report=json.loads((out/'report.json').read_text())
    latency=json.loads((out/'latency.json').read_text())
    rows=[]
    for name in ('dev.jsonl','dev_noisy.jsonl'):
        rows.extend(json.loads(line) for line in (ROOT/'data/op06'/name).read_text().splitlines() if line)
    pred={r['id']:r for r in map(json.loads,(out/'predictions.jsonl').read_text().splitlines())}
    sys.path.insert(0,str(ROOT))
    from triage.policy import load
    policy=load(repo/'policy/policy.toml')
    auto=[r for r in rows if not pred[r['id']]['needs_human']]
    forbidden=[r for r in auto if not policy._permitted(pred[r['id']]['action'],r['flow'])]
    safety=dict(automatic=len(auto),forbidden_automatic=len(forbidden),
                high_risk_forbidden_automatic=sum(policy.risk.get(pred[r['id']]['action'])=='high' for r in forbidden),
                wrong_automatic=sum(pred[r['id']]['action']!=r['action'] for r in auto))
    review_cost=(len(rows)-len(auto)+sum((10 if policy.risk.get(pred[r['id']]['action'])=='high' else 3)
                                       for r in auto if pred[r['id']]['action']!=r['action']))/len(rows)*100
    summary=dict(mode=a.mode,repository=str(repo),official=report['official'],
                 bars=report['bars'],review_rate=report['extras']['needs_human_rate'],safety=safety,
                 action_only_review_cost_units_per100=review_cost,
                 latency_ms=latency['latency_ms'],concurrency=latency['concurrency'],
                 llm_calls=latency['llm_calls'],llm_cost_usd=latency['llm_cost_usd'],
                 llm_cost_uncached_usd=latency['llm_cost_uncached_usd'],
                 llm_statuses=latency['server_metrics_after']['llm']['by_status'])
    (out/'summary.json').write_text(json.dumps(summary,indent=1)+'\n')
    print(json.dumps(summary,indent=1))


if __name__=='__main__':
    main()
