"""Audit the local and extracted bundles without using workspace dependencies."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
APP = ROOT / 'dist/Whallm.app'
ZIP = ROOT / 'dist/Whallm-macOS-arm64.zip'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)


def audit(app, label):
    command(['codesign', '--verify', '--deep', '--strict', str(app)])
    resources = app / 'Contents/Resources'
    env = dict(os.environ, PYTHONHOME=str(app / 'Contents/Frameworks/Python.framework/Versions/Current'),
               PYTHONPATH=f'{resources / "runtime"}:{resources / "python/site-packages"}',
               PYTHONDONTWRITEBYTECODE='1')
    python = str(app / 'Contents/MacOS/python3')
    packages = json.loads(command([python, '-c', 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["mlx","mlx-metal","mlx-lm"]}))'], env=env, cwd=app.parent).stdout)
    assert packages['mlx'] == packages['mlx-metal'] == '0.32.1', packages
    result = command([python, '-m', 'unittest', 'discover', '-s', str(resources / 'runtime/tests'), '-p', 'test_sampling.py'], env=env, cwd=app.parent)
    (OUT / f'{label}-sampling-tests.log').write_text(result.stdout + result.stderr)
    localization = {}
    for lang in ['en', 'zh-Hans', 'zh-Hant']:
        for name in ['Localizable.strings', 'InfoPlist.strings']:
            relative = f'{lang}.lproj/{name}'
            file = resources / relative
            command(['plutil', '-lint', str(file)])
            expected = ROOT / 'Sources/DeepSeekV4SSDApp/Resources' / relative
            assert sha(file) == sha(expected), relative
            localization[relative] = sha(file)
    runtime_files = {}
    for file in (resources / 'runtime').rglob('*.py'):
        relative = file.relative_to(resources / 'runtime')
        assert sha(file) == sha(ROOT / 'runtime' / relative), relative
        runtime_files[str(relative)] = sha(file)
    magic = {bytes.fromhex(x) for x in ['cffaedfe', 'cefaedfe', 'feedfacf', 'feedface', 'cafebabe', 'bebafeca']}
    binaries = []
    for file in (app / 'Contents').rglob('*'):
        if file.is_symlink() or not file.is_file():
            continue
        with file.open('rb') as stream:
            if stream.read(4) in magic:
                binaries.append(file)
    def inspect(file):
        linked = command(['otool', '-L', str(file)]).stdout
        deps = [line.strip().split(' (', 1)[0] for line in linked.splitlines()
                if line.startswith('\t') and '(compatibility version' in line]
        forbidden = [dep for dep in deps if dep.startswith('/') and not dep.startswith(('/usr/lib/', '/System/Library/'))]
        assert not forbidden, (str(file), forbidden)
        return {'file': str(file.relative_to(app)), 'sha256': sha(file), 'dependencies': deps}
    binaries.sort()
    with ThreadPoolExecutor(max_workers=8) as workers:
        native = list(workers.map(inspect, binaries))
    command(['codesign', '--verify', '--deep', '--strict', str(app)])
    plist = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    return {'signature_before_and_after_sampling': 'passed', 'version': plist['CFBundleShortVersionString'],
            'build': plist['CFBundleVersion'], 'packages': packages, 'sampling_tests_passed': 3,
            'localization_sha256': localization, 'runtime_source_sha256': runtime_files,
            'macho': native, 'forbidden_absolute_dependencies': []}


with tempfile.TemporaryDirectory(prefix='whallm-merged-bundle-') as directory:
    command(['ditto', '-x', '-k', str(ZIP), directory])
    report = {'source_commit': command(['git', 'rev-parse', 'HEAD'], cwd=ROOT).stdout.strip(),
              'branch': command(['git', 'branch', '--show-current'], cwd=ROOT).stdout.strip(),
              'original': audit(APP, 'original'),
              'extracted': audit(Path(directory) / 'Whallm.app', 'extracted'),
              'zip_sha256': sha(ZIP), 'zip_bytes': ZIP.stat().st_size}
    assert report['original'] == report['extracted'], 'App and extracted ZIP differ'
    (OUT / 'bundle-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'source_commit': report['source_commit'], 'zip_sha256': report['zip_sha256'],
                      'zip_bytes': report['zip_bytes'], 'native_files_per_bundle': len(report['original']['macho']),
                      'sampler_tests_per_bundle': 3, 'original_equals_extracted': True}))
