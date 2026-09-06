import unittest

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step

from deepseek_v4_ssd.generation import GenerationOptions, _generation_logits_processors


def apply(options, history):
    logits = mx.array([[4.0, -4.0, 2.0, 3.0, 5.0]])
    for processor in _generation_logits_processors(options, is_qwen=True):
        logits = processor(mx.array(history, dtype=mx.int32), logits)
    return np.array(logits)


class SamplingTests(unittest.TestCase):
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
