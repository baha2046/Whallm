import unittest

from Scripts.research_qwen_attention import chunk_source
from Scripts.analyze_qsa_research import summarize


class AttentionResearchTest(unittest.TestCase):
    def test_incomplete_measurements_cannot_pass(self):
        with self.assertRaises(ValueError):
            summarize({"status": "running", "runs": []})

    def test_only_chunk_assignment_changes_and_source_drift_stops(self):
        source = "def f():\n    query_chunk = 4\n    unrelated = 4\n    return query_chunk, unrelated\n"
        namespace = {}
        exec(chunk_source(source, 16), namespace)
        self.assertEqual(namespace['f'](), (16, 4))
        for changed in (source.replace('query_chunk = 4', 'query_chunk = 8'),
                        source.replace('query_chunk = 4', 'other = 4')):
            with self.assertRaises(ValueError):
                chunk_source(changed, 16)
