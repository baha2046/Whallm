import json
from deepseek_v4_ssd.generation import ModelRuntime
from deepseek_v4_ssd.server import main
original = ModelRuntime.stream
def record(self, *args, **kwargs):
    iterator = original(self, *args, **kwargs)
    with open('/Users/Shared/Project/product/llm-ssd/docs/benchmarks/2026-09-06-issue6-fix-m2-max/candidate-05-numerics/long-chat/tokens.jsonl', 'a') as output:
        try:
            for piece in iterator:
                output.write(json.dumps({'token': int(piece.token), 'index': piece.generation_tokens, 'finish': piece.finish_reason}) + '\n')
                output.flush()
                yield piece
        finally:
            iterator.close()
ModelRuntime.stream = record
main()
