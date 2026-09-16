"""Verify Q1 paired evidence and report screening results without adoption claims."""
if __package__:
    from .archived_evidence import archived_path
else:
    from archived_evidence import archived_path
import argparse
import hashlib
import json
import re
from pathlib import Path

from benchmark_research_baseline import digest
from grade_qwen_quality import grade
from prepare_r0_prompts import token_sha256


def analyze(results, suite, confirmation=None):
    if results['status'] not in ('screen_completed', 'stopped_on_regression'):
        raise ValueError('incomplete or failed execution is not screening evidence')
    truncated = {row['case'] for row in results['runs'] if row['truncated']}
    if truncated and results['status'] == 'screen_completed':
        raise ValueError('truncated responses leave the quality comparison unresolved')
    cases = {case['id']: case for case in suite['cases']}
    indexed = {}
    for row in results['runs']:
        key = (row['case'], row['variant'])
        assert key not in indexed and row['case'] in cases
        indexed[key] = row
        assert row['metrics']['token_sha256'] == token_sha256(row['metrics']['generated_token_ids'])
        assert row['text_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()
        assert row['metrics']['prompt_tokens'] == suite['target_tokens']
        assert not row['metrics']['mtp_enabled'] and not row['metrics']['prompt_cache_reused_tokens']
        case = cases[row['case']]
        if case.get('source_id') == 'HumanEval/50':
            case = dict(case, test='import random; random.seed(20260906)\n' + case['test'])
        checked = grade(case, row['text'])
        assert checked['passed'] == row['grading']['passed'], 'regrading changed: ' + row['case']
        assert row['passed'] == (checked['passed'] and not row['truncated'])
        if row['category'] == 'math' and checked['passed']:
            assert re.fullmatch(r'Answer:\s*[-+]?\d[\d,]*(?:\.\d+)?\s*', row['text'].strip().splitlines()[-1]), 'answer was not final'
        if row['variant'] == 'candidate':
            assert row['record']['sorting_hint'] and row['record']['modified_calls'] == 192
    assert len(indexed) == 2 * len(results['pairs'])
    assert len({p['case'] for p in results['pairs']}) == len(results['pairs'])
    for pair in results['pairs']:
        a, b = (indexed[(pair['case'], variant)] for variant in ('control', 'candidate'))
        assert a['metrics']['prompt_token_sha256'] == b['metrics']['prompt_token_sha256']
        assert pair['control_passed'] == a['passed'] and pair['candidate_passed'] == b['passed']
        assert pair['regression'] == (a['passed'] and not b['passed'])
        assert pair['output_parity'] == (a['metrics']['token_sha256'] == b['metrics']['token_sha256'])
    if results['status'] == 'screen_completed':
        assert len(results['pairs']) == len(cases)
    else:
        assert results['pairs'][-1]['regression']
        assert results['pairs'][-1]['case'] not in truncated, 'a truncated pair cannot establish regression'
    categories = {}
    for category in ('code', 'math', 'zh', 'tool'):
        attempted = [p for p in results['pairs'] if p['category'] == category]
        pairs = [p for p in attempted if p['case'] not in truncated]
        categories[category] = dict(planned=8, attempted=len(attempted), valid_pairs=len(pairs),
            truncated_pairs=len(attempted)-len(pairs),
            control_correct=sum(p['control_passed'] for p in pairs),
            candidate_correct=sum(p['candidate_passed'] for p in pairs),
            regressions=sum(p['regression'] for p in pairs),
            gains=sum(not p['control_passed'] and p['candidate_passed'] for p in pairs),
            both_wrong=sum(not p['control_passed'] and not p['candidate_passed'] for p in pairs),
            output_parity=sum(p['output_parity'] for p in pairs))
    confirmed = None
    if confirmation:
        assert confirmation['status'] == 'completed'
        assert confirmation['case'] == results['pairs'][-1]['case']
        checked_rows = {}
        for row in confirmation['runs']:
            assert row['metrics']['token_sha256'] == token_sha256(row['metrics']['generated_token_ids'])
            assert row['text_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()
            case = cases[confirmation['case']]
            if case.get('source_id') == 'HumanEval/50':
                case = dict(case, test='import random; random.seed(20260906)\n' + case['test'])
            checked = grade(case, row['text'])
            assert row['passed'] == (checked['passed'] and row['metrics']['generated_tokens'] < suite['max_tokens'])
            original = indexed[(confirmation['case'], row['variant'])]
            assert row['metrics']['prompt_token_sha256'] == original['metrics']['prompt_token_sha256']
            checked_rows[row['variant']] = row
        assert len(checked_rows) == 2
        confirmed = checked_rows['control']['passed'] and not checked_rows['candidate']['passed']
        assert confirmation['regression_confirmed'] == confirmed
    return dict(adoption_passed=False, formal_performance_result=False,
                decision=('confirmed_regression_stop' if confirmed else
                          'regression_unconfirmed_stop' if results['status'] == 'stopped_on_regression' else
                          'no_observed_regression_continue_q1b'),
                category_summary=categories, attempted_pairs=len(results['pairs']),
                valid_pairs=len(results['pairs'])-len(truncated),
                truncated_cases=sorted(truncated), confirmation_counts_as_new_tasks=False,
                unrun_cases=[c for c in cases if (c, 'control') not in indexed],
                uncertainty='No general task-accuracy bound established; this is screening evidence.',
                hypothetical_zero_loss_upper=(1-(0.05/4)**(1/8) if results['status'] == 'screen_completed' else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    suite = json.loads((args.results.parent / 'suite.json').read_text())
    confirmation = json.loads(args.confirmation.read_text()) if args.confirmation else None
    if confirmation:
        assert confirmation['original_results_sha256'] == digest(args.results)
    report = analyze(results, suite, confirmation)
    report.update(results=results, confirmation=confirmation, raw_results_sha256=digest(args.results),
                  analysis_source_sha256=digest(Path(__file__)),
                  suite_sha256=digest(args.results.parent / 'suite.json'),
                  grading_note_sha256=digest(archived_path('research/QWEN_QUALITY_Q1_GRADING_NOTE_2026-09-06.md')))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('decision', 'category_summary', 'unrun_cases')}, indent=2))


if __name__ == '__main__':
    main()
