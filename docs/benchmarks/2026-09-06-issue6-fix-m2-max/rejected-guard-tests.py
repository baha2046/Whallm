import unittest

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step

from deepseek_v4_ssd.generation import GenerationOptions, _generation_logits_processors, _no_repeat_generated_ngram


def apply(options, history):
    logits = mx.array([[4.0, -4.0, 2.0, 3.0, 5.0]])
    for processor in _generation_logits_processors(options, is_qwen=True):
        logits = processor(mx.array(history, dtype=mx.int32), logits)
    return np.array(logits)


class SamplingTests(unittest.TestCase):
    def test_ngram_mask_matches_reference_for_short_and_long_sequences(self):
        rng = np.random.default_rng(6)
        for size in (1, 2, 4, 32):
            histories = [[], [2], [0, 1, 2, 0, 1], [0, 1, 2, 0, 1, 3, 0, 1], list(range(8)) * 12,
                         rng.integers(0, 8, size=100).tolist()]
            process = _no_repeat_generated_ngram(size)
            for generated in histories:
                with self.subTest(size=size, length=len(generated)):
                    expected = set()
                    if len(generated) >= size:
                        suffix = generated[-(size - 1):] if size > 1 else []
                        for start in range(len(generated) - size + 1):
                            if generated[start:start + size - 1] == suffix:
                                expected.add(generated[start + size - 1])
                    logits = process(mx.array([7, *generated]), mx.zeros((1, 8)))
                    actual = set(np.flatnonzero(np.isneginf(np.array(logits)[0])))
                    self.assertEqual(actual, expected)

    def test_ngram_guard_does_not_leak_a_discarded_speculative_branch(self):
        process = _no_repeat_generated_ngram(3)
        a = np.array(process(mx.array([4, 0, 1, 2, 0, 1]), mx.zeros((1, 5))))
        b = np.array(process(mx.array([4, 0, 1, 3, 0, 1]), mx.zeros((1, 5))))
        self.assertTrue(np.isneginf(a[0, 2]))
        self.assertTrue(np.isneginf(b[0, 3]))
        self.assertEqual(b[0, 2], 0)

    def test_ngram_guard_breaks_a_greedy_token_cycle(self):
        process = _no_repeat_generated_ngram(4)
        history = [4]
        for _ in range(16):
            logits = process(mx.array(history), mx.array([[4.0, 3.8, 3.6, 3.4, 3.2]]))
            history.append(mx.argmax(logits, axis=-1).item())
        grams = [tuple(history[i:i+4]) for i in range(1, len(history)-3)]
        self.assertEqual(len(grams), len(set(grams)))

    def test_presence_remembers_tokens_older_than_twenty_and_excludes_prompt(self):
        options = GenerationOptions(presence_penalty=1.5)
        # Token 4 is the final prompt token, token 0 is now beyond the old window.
        actual = apply(options, [4, 0] + [2] * 32)
        np.testing.assert_allclose(actual, [[2.5, -4, 0.5, 3, 5]])
        np.testing.assert_allclose(apply(options, [4]), [[4, -4, 2, 3, 5]])

    def test_repetition_is_sign_aware_and_frequency_counts_generated_occurrences(self):
        options = GenerationOptions(repetition_penalty=2, presence_penalty=0.5, frequency_penalty=0.25)
        np.testing.assert_allclose(apply(options, [4, 0, 1, 0]), [[1, -8.75, 2, 3, 5]])

    def test_speculative_branch_rollback_cannot_leak_penalties(self):
        processors = _generation_logits_processors(GenerationOptions(presence_penalty=1.5), is_qwen=True)
        for history in ([4, 0, 1, 2], [4, 0], [4, 0, 3], [4]):
            logits = mx.array([[4.0, -4.0, 2.0, 3.0, 5.0]])
            for processor in processors:
                logits = processor(mx.array(history), logits)
            np.testing.assert_array_equal(np.array(logits), apply(GenerationOptions(presence_penalty=1.5), history))

    def test_neutral_options_preserve_the_model_distribution(self):
        self.assertEqual(_generation_logits_processors(GenerationOptions(), is_qwen=True), [])

    def test_real_decode_history_excludes_prefilled_prompt_and_penalizes_output(self):
        class FixedModel:
            def __call__(self, tokens, cache):
                return mx.broadcast_to(mx.array([4.0, 3.8, 3.6, 3.4, 3.2]), (1, tokens.shape[1], 5))

        for prompt in ([0, 3, 4], [4]):
            with self.subTest(prompt=prompt):
                output = list(generate_step(
                    mx.array(prompt), FixedModel(), prompt_cache=[], prefill_step_size=1,
                    max_tokens=4,
                    logits_processors=_generation_logits_processors(GenerationOptions(presence_penalty=1.5), is_qwen=True),
                ))
                self.assertEqual([token for token, _ in output], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
