import json
from dataclasses import asdict
import mlx.core as mx
from deepseek_v4_ssd.generation import ModelRuntime
from deepseek_v4_ssd.server import main
original = ModelRuntime.stream
def record(self, *args, **kwargs):
    with mx.stream(self._generation_stream):
        mx.random.seed(20260907)
    with open('/Users/Shared/Project/product/llm-ssd/docs/benchmarks/2026-09-06-issue6-causal-m2-max/current-source-seed-20260907/short-thinking/runtime-request.json', 'w') as request:
        json.dump({'prompt_tokens': self._encode_prompt(args[0]), 'options': asdict(args[1]), 'config': asdict(self.config)}, request)
    iterator = original(self, *args, **kwargs)
    with open('/Users/Shared/Project/product/llm-ssd/docs/benchmarks/2026-09-06-issue6-causal-m2-max/current-source-seed-20260907/short-thinking/tokens.jsonl', 'a') as output:
        try:
            for piece in iterator:
                output.write(json.dumps({'token': int(piece.token), 'index': piece.generation_tokens, 'finish': piece.finish_reason}) + '\n')
                output.flush()
                yield piece
        finally:
            iterator.close()
ModelRuntime.stream = record
main()
