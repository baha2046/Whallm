import unittest
from unittest.mock import patch
from dataclasses import replace
import mlx.core as mx
from runtime.tests.test_v41_prompt_cache import tiny_model
from deepseek_v4_ssd.deepseek_v41.dspark import DSpark
from deepseek_v4_ssd.dspark import generate_tokens, _target_sequence
from deepseek_v4_ssd.model import forward_with_hidden


class V41DSparkTests(unittest.TestCase):
    def test_drafts_verify_and_rejected_branches_preserve_committed_state(self):
        main = tiny_model()
        draft = DSpark(main.args, block_size=3, noise_token_id=15,
                       target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        prompt = [1, 2, 3, 4, 5, 6, 7]
        reference = main.make_cache()
        logits = main(mx.array([prompt]), reference)
        expected = []
        for _ in range(8):
            token = int(mx.argmax(logits[:, -1], axis=-1).item())
            expected.append(token)
            logits = main(mx.array([[token]]), reference)
        rounds = []
        cache = main.make_cache()
        actual = list(generate_tokens(prompt, main, draft, cache, max_tokens=8,
                        prefill_step_size=3, temperature=0, top_p=1,
                        fallback_enabled=False,
                        record_round=lambda *args: rounds.append(args)))
        self.assertEqual([int(value[0]) for value in actual], expected)
        self.assertTrue(rounds)
        # A speculative fork leaves the original position and Engram history intact.
        before = cache[0].persistence_state()
        _target_sequence(main, [1, 2], cache, (1, 2, 3))
        self.assertEqual(cache[0].offset, before['offset'])
        self.assertTrue(mx.array_equal(mx.array(cache[0].cache.engram_ids), before['engram_ids']).item())

    def test_hidden_capture_and_draft_context_snapshot(self):
        main = tiny_model()
        draft = DSpark(main.args, block_size=3, noise_token_id=15,
                       target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        cache = main.make_cache()
        logits, hidden = forward_with_hidden(main, mx.array([[1, 2, 3, 4, 5]]), cache, draft.target_layers)
        self.assertEqual(hidden.shape, (1, 5, main.args.dim * 3))
        draft.prefill_context(hidden, 0)
        saved = draft.cache_state()
        _, hidden = forward_with_hidden(main, mx.array([[6, 7]]), cache, draft.target_layers)
        first = draft.draft(main, 8, hidden, 5, 0, 1, 0)
        draft.restore_cache_state(saved)
        second = draft.draft(main, 8, hidden, 5, 0, 1, 0)
        self.assertEqual(first.tokens, second.tokens)
        self.assertEqual(saved[0][1], 5)
        self.assertEqual(draft.mtp[0].attn.offset, 7)
        with self.assertRaisesRegex(ValueError, 'position'):
            draft.prefill_context(hidden, 2)
