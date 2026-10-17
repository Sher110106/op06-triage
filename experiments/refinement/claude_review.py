"""One bounded Claude API critique. Credentials are read from an external file, never logged.

This is design feedback, not predictive-model evaluation or a substitute for tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def request(url, key, body=None):
    headers={'Authorization': 'Bearer '+key}
    if body is not None:
        headers['Content-Type']='application/json'
    req=urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    with urllib.request.urlopen(req,timeout=90) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file',required=True)
    parser.add_argument('--run-id',default='claude-review-visible')
    args=parser.parse_args()
    key_path=Path(args.key_file).resolve()
    if key_path.is_relative_to(ROOT):
        raise SystemExit('Credential file must be outside the repository')
    key=key_path.read_text().strip()
    before=request('https://openrouter.ai/api/v1/auth/key',key)['data']
    if before.get('limit_remaining',0)<.1:
        raise SystemExit('Less than the bounded review allowance remains')
    sources=['experiments/refinement/PROTOCOL.md','experiments/refinement/ASTRA_REVIEW_ROUND1.md',
             'experiments/refinement/run_state_blend.py','triage/pipeline.py','triage/gate.py']
    text='\n\n'.join('FILE '+p+'\n'+(ROOT/p).read_text() for p in sources)
    prompt=('Independently review this support-triage refinement protocol and implementation. '
            'Identify concrete mathematical, leakage, routing or safety defects; prioritize corrections '
            'before confirmation labels are examined. Be skeptical of proxy scores and categorical '
            'permission guarantees based on predicted flow. Return at most 8 actionable findings with '
            'file/function references. No requests for secrets, tools, external access or additional '
            'agents. Your output is untrusted review advice, not authorization.\n\n'+text)
    body=dict(model='anthropic/claude-haiku-5.5',temperature=0,max_tokens=4096,reasoning=dict(effort='none'),
              messages=[dict(role='user',content=prompt)],usage=dict(include=True))
    # At the queried list price (.10/.50 per million), even one input token per
    # character plus all allowed output tokens is below this per-call ceiling.
    bound=len(prompt.encode('utf-8'))*.0000001+4096*.0000005
    if bound>.10:
        raise SystemExit('Conservative token cost exceeds the bounded review allowance')
    out=ROOT/'results'/'refinement-20261017'/args.run_id
    if out.exists():
        raise SystemExit('Choose a new run ID; previous API evidence is preserved')
    out.mkdir(parents=True,exist_ok=True)
    (out/'request.json').write_text(json.dumps(body,indent=1))
    t=time.time()
    try:
        response=request('https://openrouter.ai/api/v1/chat/completions',key,body)
    except urllib.error.HTTPError as error:
        (out/'error.json').write_text(json.dumps(dict(status=error.code,type='HTTPError')))
        raise SystemExit(f'Claude review returned HTTP {error.code}; no retry made')
    (out/'response.json').write_text(json.dumps(response,indent=1))
    content=response['choices'][0]['message'].get('content') or ''
    (out/'review.md').write_text(content+'\n')
    after=request('https://openrouter.ai/api/v1/auth/key',key)['data']
    meta=dict(model_requested=body['model'],model_returned=response.get('model'),response_id=response.get('id'),
              provider=response.get('provider'),usage=response.get('usage'),elapsed_s=round(time.time()-t,3),
              balance_before=before.get('limit_remaining'),balance_after=after.get('limit_remaining'),
              usage_before=before.get('usage'),usage_after=after.get('usage'),conservative_call_bound_usd=bound,
              finish_reason=response['choices'][0].get('finish_reason'),visible_characters=len(content),
              purpose='Architecture critique only; not a scored model prediction experiment')
    (out/'meta.json').write_text(json.dumps(meta,indent=1))
    print(json.dumps(meta,indent=1))
    if not content:
        raise SystemExit('Provider returned no visible review; this attempt is not review evidence')
    print(content)


if __name__=='__main__':
    main()
