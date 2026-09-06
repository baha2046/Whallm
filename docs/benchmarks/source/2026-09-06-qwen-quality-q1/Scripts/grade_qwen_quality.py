"""Q1 graders. Generated Python only runs in a restricted child process."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


def run_python(source, test, entry_point):
    python = Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    # Child receives code through argv, never access to the project or home files.
    profile = ('(version 1)(allow default)(deny network*)(deny file-write*)'
               '(deny process-fork)(deny signal)(deny mach-lookup)(deny mach-register)'
               '(deny file-read* (subpath "/Users") (subpath "/private")'
               ' (subpath "/Volumes") (subpath "/etc") (subpath "/var")'
               ' (subpath "/tmp") (subpath "/opt/homebrew/etc") (subpath "/cores"))')
    wrapper = ('import resource\n'
               'resource.setrlimit(resource.RLIMIT_CPU,(3,3))\n'
               'resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))\n'
               'resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n'
               + source + '\n' + test + f'\ncheck({entry_point})\nprint("WHALLM_TESTS_COMPLETED")\n')
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, str(python),
                                     '-I', '-S', '-B', '-c', wrapper],
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                    env={'PATH': '/usr/bin:/bin'}, cwd='/', timeout=10)
            output.seek(0)
            detail = output.read(1048576).decode(errors='replace')
            passed = result.returncode == 0 and detail.rstrip().endswith('WHALLM_TESTS_COMPLETED')
            return dict(passed=passed, returncode=result.returncode, detail=detail[-4000:])
        except subprocess.TimeoutExpired:
            return dict(passed=False, detail='execution timeout')


def grade(case, text):
    from benchmark_qwen_task_accuracy import score

    if case['grader'] == 'python':
        match = re.fullmatch(r'\s*```(?:python)?\s*\n(.*?)\n```\s*', text, re.S)
        source = match[1] if match else text.strip()
        return run_python(source, case['test'], case['entry_point'])
    if case['grader'] == 'math':
        answers = re.findall(r'^Answer:\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$', text, re.M)
        try:
            passed = len(answers) == 1 and Decimal(answers[0].replace(',', '')) == Decimal(case['expected'])
        except InvalidOperation:
            passed = False
        return dict(passed=passed, extracted_answers=answers)
    return dict(passed=score(text, case['expected']))
