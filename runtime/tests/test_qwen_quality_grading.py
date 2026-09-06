import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Scripts'))
from grade_qwen_quality import grade, run_python


class QualityGradingTests(unittest.TestCase):
    def test_function_tests_execute_and_require_completion(self):
        check = 'def check(f): assert f(3) == 4'
        self.assertTrue(run_python('def f(x): return x+1', check, 'f')['passed'])
        self.assertFalse(run_python('def f(x): return x', check, 'f')['passed'])
        self.assertFalse(run_python('import sys; sys.exit(0)', check, 'f')['passed'])

    def test_sandbox_denies_external_io(self):
        canary = str(Path(__file__).resolve())
        source = f'''def f():
    import socket
    denied = 0
    for action in (lambda: open({canary!r}).read(),
                   lambda: open('/tmp/whallm-q1-denied-write', 'w'),
                   lambda: socket.create_connection(('127.0.0.1', 9), timeout=1)):
        try:
            action()
        except PermissionError:
            denied += 1
    return denied
'''
        result = run_python(source, 'def check(f): assert f() == 3', 'f')
        self.assertTrue(result['passed'], result)

    def test_math_requires_one_final_answer(self):
        case = dict(grader='math', expected='120')
        self.assertTrue(grade(case, '20 times 6 is 120.\nAnswer: 120')['passed'])
        for text in ('Answer: 60', '120', 'Answer: 120\nAnswer: 60'):
            self.assertFalse(grade(case, text)['passed'])

    def test_official_reference_solutions(self):
        suite = Path(__file__).resolve().parents[2] / 'docs/benchmarks/prompts/2026-09-06-qwen-quality-q1.json'
        for case in json.loads(suite.read_text())['cases']:
            if case['grader'] == 'python':
                with self.subTest(case=case['id']):
                    result = grade(case, case['reference'])
                    self.assertTrue(result['passed'], result)


if __name__ == '__main__':
    unittest.main()
