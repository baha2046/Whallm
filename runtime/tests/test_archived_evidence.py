import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from Scripts import archived_evidence as evidence


class ArchivedEvidenceTests(unittest.TestCase):
    def test_missing_local_evidence_skips_only_research_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(evidence, 'ROOT', Path(temporary)):
                evidence._archive.cache_clear()
                with self.assertRaises(evidence.MissingLocalArchive):
                    evidence.archived_path('docs/example.txt')
                with self.assertRaises(unittest.SkipTest):
                    evidence.enable_archived_research()
                evidence._archive.cache_clear()

    def test_round_trip_repairs_cache_and_rejects_changed_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.txt'
            source.write_text('historical evidence\n')
            base = root / '.vscode/docs'
            base.mkdir(parents=True)
            archive = base / 'originals.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                tar.add(source, arcname='docs/nested/evidence.txt')
            manifest = {'archive':archive.name, 'archive_sha256':evidence._sha(archive),
                        'files':{'docs/nested/evidence.txt':{'sha256':evidence._sha(source)}}}
            (base/'archive-manifest.json').write_text(json.dumps(manifest))
            with patch.object(evidence, 'ROOT', root):
                evidence._archive.cache_clear()
                path = evidence.archived_path('docs/nested/evidence.txt')
                self.assertEqual(path.read_bytes(), source.read_bytes())
                path.write_text('damaged')
                self.assertEqual(evidence.archived_path('docs/nested').joinpath('evidence.txt').read_bytes(), source.read_bytes())
                self.assertFalse((root/'docs').exists())
                for name in ('../docs', '/docs/evidence.txt', 'docs/../research'):
                    with self.assertRaises(ValueError):evidence.archived_path(name)
                with self.assertRaises(FileNotFoundError):evidence.archived_path('docs/missing')
                archive.write_bytes(b'changed archive')
                evidence._archive.cache_clear()
                with self.assertRaises(ValueError):evidence.archived_path('docs/nested')
            evidence._archive.cache_clear()


if __name__ == '__main__':unittest.main()
