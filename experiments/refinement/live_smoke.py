"""Three bounded real-provider HTTP checks, not an accuracy estimate.

Use --key-file outside the repository. No environment variables or key values
are saved. The service's normal deadline, prompt, model pin and policy apply.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from triage.data import load_split, project
from triage.llm_action import LLMConfig
from triage.pipeline import Triage
from triage.service import create_app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file',required=True)
    args=parser.parse_args()
    path=Path(args.key_file).resolve()
    if path.is_relative_to(ROOT):
        raise SystemExit('Credential file must be outside the repository')
    key=path.read_text().strip()
    request=urllib.request.Request('https://openrouter.ai/api/v1/auth/key',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(request,timeout=20) as response:
        balance=json.load(response)['data'].get('limit_remaining',0)
    if balance<.10:
        raise SystemExit('Less than the conservative smoke allowance remains')
    cpu=Triage(llm_cfg=LLMConfig(mode='off'))
    candidates=[project(r) for r in load_split('dev')]
    scored=cpu.handle_many(candidates)
    cpu.close()
    chosen=[r for r,(p,t) in zip(candidates,scored) if not t['flags'] and p['confidence']<.642841][:3]
    assert len(chosen)==3
    live=Triage(llm_cfg=LLMConfig(mode='on',api_key=key))
    assert live.llm is not None, 'Retrieval/LLM initialization failed; this is not a live smoke'
    results=[]
    with TestClient(create_app(triage=live)) as client:
        for i,row in enumerate(chosen):
            response=client.post('/triage',json=row)
            assert response.status_code==200
            results.append(dict(id=row['id'],http_status=response.status_code,response=response.json()))
            if i==0:
                replay=client.post('/triage',json=row)
                assert replay.status_code==200 and replay.json()==response.json()
        metrics=client.get('/metrics').json()
        health=client.get('/healthz').json()
        assert metrics['replays']==1
        assert metrics['llm']['calls']==3, metrics['llm']
    out=ROOT/'results'/'refinement-20261017'/'live_smoke.json'
    out.write_text(json.dumps(dict(purpose='Compatibility/routing/telemetry smoke only, not an accuracy estimate',
                                   key_balance_before=balance,results=results,metrics=metrics,health=health),indent=1))
    print(json.dumps(dict(rows=len(results),llm=metrics['llm'],replays=metrics['replays']),indent=1))


if __name__=='__main__':
    main()
