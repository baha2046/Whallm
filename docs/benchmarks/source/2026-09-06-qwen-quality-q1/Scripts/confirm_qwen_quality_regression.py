"""Confirm one Q1 regression in reverse order without extending the sample."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from benchmark_research_baseline import digest
from prepare_r0_prompts import token_sha256
from grade_qwen_quality import grade


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    original=json.loads(args.results.read_text())
    assert original['status']=='stopped_on_regression'
    pair=original['pairs'][-1]
    assert pair['regression']
    suite=json.loads((args.results.parent/'suite.json').read_text())
    case=next(c for c in suite['cases'] if c['id']==pair['case'])
    if case.get('source_id')=='HumanEval/50':
        case=dict(case,test='import random; random.seed(20260906)\n'+case['test'])
    selected=[r for r in original['runs'] if r['case']==case['id']]
    assert len(selected)==2
    for name,sha in original['source']['files_sha256'].items():
        assert digest(root/name)==sha, name
    args.output.mkdir(parents=True,exist_ok=False)
    artifact=dict(status='running',original_results_sha256=digest(args.results),case=case['id'],
                  formal_performance_result=False,adoption_passed=False,counts_as_new_task=False,
                  started_at=datetime.now(timezone.utc).isoformat(),runs=[])
    try:
        for previous in reversed(selected):
            variant=previous['variant']; stem=args.output/(case['id']+'-'+variant)
            command=previous['command'].copy()
            assert digest(Path(command[command.index('--prompt-file')+1]))==previous['prompt_text_sha256']
            command[command.index('--metrics-json')+1]=str(stem.with_suffix('.json'))
            if variant=='candidate':
                command[command.index('--record')+1]=str(stem.with_suffix('.record.json'))
            print('START confirmation',case['id'],variant,flush=True)
            with stem.with_suffix('.txt').open('w') as stdout,stem.with_suffix('.stderr').open('w') as stderr:
                subprocess.run(command,cwd=root,env=os.environ|{'PYTHONPATH':str(root/'runtime')},
                               stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,check=True)
            metrics=json.loads(stem.with_suffix('.json').read_text())
            assert metrics['prompt_token_sha256']==previous['metrics']['prompt_token_sha256']
            assert metrics['token_sha256']==token_sha256(metrics['generated_token_ids'])
            assert not metrics['mtp_enabled'] and not metrics['prompt_cache_reused_tokens']
            text=stem.with_suffix('.txt').read_text(); grading=grade(case,text)
            record=json.loads(stem.with_suffix('.record.json').read_text()) if variant=='candidate' else None
            if record:
                assert record['modified_calls']==192 and record['sorting_hint']
            row=dict(variant=variant,command=command,metrics=metrics,text=text,grading=grading,record=record,
                     passed=grading['passed'] and metrics['generated_tokens']<suite['max_tokens'],
                     text_sha256=digest(stem.with_suffix('.txt')),
                     repeats_original_tokens=metrics['token_sha256']==previous['metrics']['token_sha256'])
            artifact['runs'].append(row)
            print('DONE',variant,'passed=',row['passed'],'same_tokens=',row['repeats_original_tokens'],flush=True)
        rows={r['variant']:r for r in artifact['runs']}
        artifact['regression_confirmed']=rows['control']['passed'] and not rows['candidate']['passed']
        artifact['status']='completed'
    except BaseException as error:
        artifact['status']='failed'; artifact['error']=str(error)
        raise
    finally:
        artifact['ended_at']=datetime.now(timezone.utc).isoformat()
        (args.output/'results.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
