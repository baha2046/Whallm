"""Regression for Issue #6: server threads must advance the sampler RNG.

Importing sample_utils on the main thread and drawing on a worker reproduced
constant tokens with MLX 0.32.0, even when the sampler was built on the worker.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime


def draw(sampler, seed):
    mx.random.seed(seed)
    logits = mx.linspace(-1.0, 1.0, 128)[None, :]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return [sampler(logprobs).item() for _ in range(64)]


class SamplingTests(unittest.TestCase):
    def test_requests_on_fresh_workers_get_independent_random_state(self):
        # Exercise ModelRuntime's request boundary with the real MLX decode loop.
        # The tiny model avoids loading weights; prompt caching is disabled.
        class FixedModel:
            def __call__(self, tokens, cache):
                return mx.broadcast_to(mx.arange(128) * 0.001, (1, tokens.shape[1], 128))

        def decode(model, tokenizer, prompt, **kwargs):
            for index, (token, _) in enumerate(generate_step(
                mx.array(prompt), model, prompt_cache=[], max_tokens=64,
                sampler=kwargs["sampler"],
            ), 1):
                yield SimpleNamespace(
                    text=str(token), token=token, prompt_tokens=len(prompt),
                    generation_tokens=index, finish_reason="length" if index == 64 else None,
                )

        with (
            patch("deepseek_v4_ssd.generation.load_model", return_value=(
                FixedModel(), SimpleNamespace(close=lambda: None))),
            patch("deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                  return_value=SimpleNamespace(bos_token=None, encode=lambda *_a, **_k: [1])),
            patch("deepseek_v4_ssd.model_support.state.make_prompt_cache", return_value=[]),
            patch("deepseek_v4_ssd.generation.stream_generate", side_effect=decode),
        ):
            runtime = ModelRuntime(
                SimpleNamespace(root=Path("/tmp/seed-test")),
                SimpleNamespace(prefill_step_size=1, layer_major_prefill=False,
                                prompt_cache_entries=0),
            )

            def request(temperature=1.0, **kwargs):
                # A new worker per call matches ThreadingHTTPServer.
                with ThreadPoolExecutor(max_workers=1) as worker:
                    return worker.submit(lambda: [piece.token for piece in runtime.stream(
                        [1], GenerationOptions(max_tokens=64, temperature=temperature, **kwargs)
                    )]).result()

            automatic = [request() for _ in range(3)]
            self.assertEqual(len({tuple(tokens) for tokens in automatic}), 3)
            fixed = request(seed=0)
            self.assertNotEqual(fixed, request(seed=42))
            request()
            self.assertEqual(fixed, request(seed=0))
            for seed in (None, 0, 42):
                self.assertEqual(request(temperature=0, seed=seed), [127] * 64)

    def test_worker_sampling_advances_and_obeys_reseeding(self):
        profiles = [dict(temp=1.0), dict(temp=1.0, top_p=.95, top_k=20),
                    dict(temp=.7, top_p=.8, top_k=20, min_p=.05)]
        with ThreadPoolExecutor(max_workers=1) as worker:
            for profile in profiles:
                with self.subTest(profile=profile):
                    sampler = make_sampler(**profile)
                    expected = draw(sampler, 20260906)
                    actual = worker.submit(draw, sampler, 20260906).result()
                    self.assertGreater(len(set(actual)), 1)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual, worker.submit(draw, sampler, 20260906).result())
                    self.assertNotEqual(actual, worker.submit(draw, sampler, 20260907).result())
                    # ModelRuntime.stream creates its sampler on the request thread.
                    def worker_built():
                        return draw(make_sampler(**profile), 20260906)
                    self.assertEqual(actual, worker.submit(worker_built).result())

    def test_greedy_remains_seed_independent_on_worker(self):
        sampler = make_sampler(temp=0)
        with ThreadPoolExecutor(max_workers=1) as worker:
            for seed in (20260906, 20260907):
                self.assertEqual(worker.submit(draw, sampler, seed).result(), [127] * 64)

    def test_real_decode_loop_samples_on_worker(self):
        class FixedModel:
            def __call__(self, tokens, cache):
                return mx.broadcast_to(mx.zeros((128,)), (1, tokens.shape[1], 128))

        def generate():
            mx.random.seed(20260906)
            return [token for token, _ in generate_step(
                mx.array([0]), FixedModel(), prompt_cache=[], max_tokens=64,
                sampler=make_sampler(temp=1.0, top_p=.95, top_k=20))]

        expected = generate()
        with ThreadPoolExecutor(max_workers=1) as worker:
            actual = worker.submit(generate).result()
        self.assertGreater(len(set(actual)), 1)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
