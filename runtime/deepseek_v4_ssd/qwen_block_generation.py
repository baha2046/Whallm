"""Prompt-lookup proposals with exact target sampling and private verification.

No draft model or extra random draws. A mismatch discards the private state and
replays only the consumed prefix. The resident pool reserves at most 81 MB of
unused payload; forks are limited to 300 MB, leaving workspace headroom.
"""
from contextlib import closing
import time

import mlx.core as mx
from mlx_lm.generate import GenerationResponse
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .cancellation import check_cancelled
from .model import _cache_arrays, _fork_prompt_cache, eval_prompt_cache
from .qwen_short_block import verify_union

MAX_FORK_BYTES = 300_000_000


def cache_bytes(cache):
    arrays = {id(a): a for a in _cache_arrays(cache)}
    def visit(value):
        if isinstance(value, mx.array):
            arrays[id(value)] = value
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    for layer in cache:
        visit(layer.state)
    return sum(a.nbytes for a in arrays.values())


def propose(history, maximum):
    history = history[-2048:]
    for n in range(min(8, len(history) - 1), 2, -1):
        suffix = history[-n:]
        for at in range(len(history) - n - 1, -1, -1):
            if history[at:at+n] == suffix:
                return history[at+n:min(at+n+maximum, len(history))]
    return []


def block_tokens(model, prompt, *, history, prompt_cache, max_tokens, sampler,
                 logits_processors, prefill_step_size, prompt_progress_callback,
                 eos_token_ids, stats):
    """Yield sampled tokens. On normal completion cache includes all non-EOS tokens."""
    prompt = list(prompt)
    history = list(history)
    processed = 0
    while len(prompt) - processed > 1:
        check_cancelled()
        count = min(prefill_step_size, len(prompt) - processed - 1)
        model(mx.array([prompt[processed:processed+count]], mx.int32), cache=prompt_cache)
        eval_prompt_cache(prompt_cache)
        processed += count
        prompt_progress_callback(processed, len(prompt))
        mx.clear_cache()
    anchor = prompt[-1]
    # Match the ordinary generator's processor history after Prefill.
    sampling_tokens = [anchor]
    generated = 0
    cooldown = 0

    def replay(inputs):
        result = None
        for token in inputs:
            check_cancelled()
            result = model(mx.array([[token]], mx.int32), cache=prompt_cache)
            eval_prompt_cache(prompt_cache, result)
        return result

    while generated < max_tokens:
        check_cancelled()
        maximum = min(3, max_tokens - generated - 1)
        draft = propose(history, maximum) if maximum and not cooldown else []
        cooldown = max(0, cooldown - 1)
        if draft and cache_bytes(prompt_cache) > MAX_FORK_BYTES:
            stats['memory_fallbacks'] += 1
            draft = []
        inputs = [anchor, *draft]
        private = None
        if draft:
            private, copied = _fork_prompt_cache(prompt_cache)
            mx.eval(*copied)
            del copied
            logits = verify_union(model, mx.array([inputs], mx.int32), private)
            eval_prompt_cache(private, logits)
            stats['rounds'] += 1
            stats['proposed_tokens'] += len(draft)
        else:
            logits = replay(inputs)
        outputs = []
        accepted = 0
        for index in range(len(inputs)):
            check_cancelled()
            adjusted = logits[:, index]
            for processor in logits_processors:
                adjusted = processor(mx.array(sampling_tokens, mx.int32), adjusted)
            logprobs = adjusted - mx.logsumexp(adjusted, keepdims=True)
            token = int(sampler(logprobs).item())
            outputs.append((token, logprobs.squeeze(0)))
            sampling_tokens.append(token)
            if token in eos_token_ids or generated + len(outputs) == max_tokens:
                break
            if index == len(draft) or token != draft[index]:
                break
            accepted += 1
        if private is not None:
            stats['accepted_tokens'] += accepted
            if len(outputs) == len(inputs):
                prompt_cache[:] = private
            else:
                # Release the rejected fork before allocating replay state.
                del private, logits
                replay(inputs[:len(outputs)])
            if accepted == 0:
                cooldown = 32
        # State now contains the inputs for precisely the emitted output prefix.
        final = outputs[-1][0]
        finished = final in eos_token_ids or generated + len(outputs) == max_tokens
        if finished and final not in eos_token_ids:
            replay([final])
        for token, logprobs in outputs:
            generated += 1
            history.append(token)
            yield token, logprobs
        if finished:
            return
        anchor = final


def stream_block_generate(model, tokenizer, prompt, *, history, stats, **kwargs):
    tokenizer = tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
    detokenizer = tokenizer.detokenizer
    started = time.perf_counter()
    count = 0
    with closing(block_tokens(model, prompt, history=history,
                              eos_token_ids=tokenizer.eos_token_ids, stats=stats, **kwargs)) as tokens:
        for token, logprobs in tokens:
            count += 1
            if count == 1:
                prompt_seconds = max(time.perf_counter() - started, 1e-9)
                decode_started = time.perf_counter()
                kwargs['prompt_progress_callback'](len(prompt), len(prompt))
            stop = token in tokenizer.eos_token_ids
            if not stop:
                detokenizer.add_token(token)
            finished = stop or count == kwargs['max_tokens']
            if finished:
                detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment, token=token, logprobs=logprobs,
                from_draft=False, prompt_tokens=len(prompt),
                prompt_tps=len(prompt)/prompt_seconds, generation_tokens=count,
                generation_tps=count/max(time.perf_counter()-decode_started, 1e-9),
                peak_memory=mx.get_peak_memory()/1e9,
                finish_reason=('stop' if stop else 'length') if finished else None,
            )
            if finished:
                return
