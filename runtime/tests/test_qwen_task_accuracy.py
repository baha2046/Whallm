"""The task screen must not silently accept malformed or wrong answers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Scripts"))
from benchmark_qwen_task_accuracy import score


class TaskAccuracyTests(unittest.TestCase):
    def test_structured_answers(self):
        expected = {"name": "alarm", "arguments": {"hour": 19, "enabled": True}}
        self.assertTrue(score(' {"arguments":{"enabled":true,"hour":19},"name":"alarm"}\n', expected))
        for answer in (
            '{"name":"alarm","arguments":{"hour":19,"enabled":1}}',
            '{"name":"alarm","arguments":{"hour":19.0,"enabled":true}}',
            '{"name":"alarm","arguments":{"hour":19,"enabled":true},"extra":0}',
            '{"name":"other","name":"alarm","arguments":{"hour":19,"enabled":true}}',
            '{"name":"alarm"}',
            '```json\n{}\n```',
            '{} trailing text',
        ):
            with self.subTest(answer=answer):
                self.assertFalse(score(answer, expected))
        self.assertTrue(score('{"outputs":[[1,2],[]]}', {"outputs": [[1, 2], []]}))
        self.assertFalse(score('{"outputs":[[2,1],[]]}', {"outputs": [[1, 2], []]}))


if __name__ == "__main__":
    unittest.main()
