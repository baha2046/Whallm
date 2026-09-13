from __future__ import annotations

from contextlib import contextmanager

from .catalog import ModelDescriptor


class ModelSupport:
    """Model-specific loading, prefill, state and chat behind one interface.

    One loaded model serves one request at a time. Cache objects belong to the
    model; the caller may only clone or serialize them through these methods.
    """

    def __init__(self, descriptor: ModelDescriptor):
        self.descriptor = descriptor

    def validate_config(self, config) -> None:
        features = self.descriptor.features
        if getattr(config, "qwen_grouped_decode", False) and (
            "groupedDecode" not in features or getattr(config, "mtp_enabled", False)
        ):
            raise ValueError("grouped Decode requires Qwen with MTP disabled")
        for option, feature, label in (
            ("dspark_enabled", "dspark", "DSpark"),
            ("staged_expert_streaming", "stagedExpertStreaming", "staged expert streaming"),
            ("adaptive_expert_prefill_threshold", "adaptiveExpertPrefill", "adaptive expert prefill"),
        ):
            if getattr(config, option, False) and feature not in features:
                raise ValueError(f"{self.option_error_name} does not support {label}")
        if getattr(config, "mtp_enabled", False) and "mtp" not in features:
            raise ValueError("MTP is supported only by Qwen3.8-Flash-Next")

    @property
    def option_error_name(self) -> str:
        return self.descriptor.display_name

    def uses_layer_major_prefill(self, config, token_count: int) -> bool:
        threshold = self.descriptor.prefill_threshold
        if threshold is None:
            threshold = getattr(config, "layer_major_prefill_threshold", 1_024)
        return bool(self.descriptor.supports("layerMajorPrefill")
                    and getattr(config, "layer_major_prefill", True)
                    and token_count >= threshold)

    def load(self, installed, config, raw_config, weights, read_limiter):
        raise NotImplementedError

    def manifest_contract(self, raw):
        raise NotImplementedError

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        raise NotImplementedError

    def open_codec(self, root, tokenizer):
        raise NotImplementedError

    def make_tool_stream_parser(self, thinking_mode):
        raise NotImplementedError

    def reasoning_settings(self, thinking_mode, effort):
        mode = thinking_mode or ("chat" if effort in {None, "none"} else "thinking")
        effort = {"minimal": "low", "low": "low", "medium": "low",
                  "high": "high", "xhigh": "max", "max": "max"}.get(effort, "low")
        return mode, effort

    def sampling_defaults(self, defaults, thinking_mode):
        return {
            "temperature": defaults.temperature, "top_p": defaults.top_p,
            "top_k": defaults.top_k, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0,
        }

    def cli_sampling_defaults(self):
        defaults = self.descriptor.defaults
        return {"temperature": defaults["temperature"], "top_p": defaults["topP"],
                "top_k": defaults["topK"]}

    def default_approximation(self, dspark_enabled=False, mode="exact"):
        return (mode
                if self.descriptor.supports("approximation") and not dspark_enabled else "exact")

    def prefer_tool_first(self, messages, tool_choice, response_tools):
        return False

    def new_cache(self, model):
        from .state import make_cache
        return make_cache(model)

    def evaluate_cache(self, cache):
        from ..model import eval_prompt_cache
        return eval_prompt_cache(cache)

    def clone_cache(self, cache):
        import copy
        if not self.descriptor.supports("promptCache"):
            raise ValueError("model does not support prompt cache restoration")
        return copy.deepcopy(cache)

    def snapshot_cache(self, cache):
        from .state import persistence_cache_state
        if not self.descriptor.supports("promptCache"):
            raise ValueError("model does not support prompt cache persistence")
        return persistence_cache_state(cache)

    def restore_cache(self, cache, state):
        from .state import restore_persistence_cache
        if not self.descriptor.supports("promptCache"):
            raise ValueError("model does not support prompt cache restoration")
        restore_persistence_cache(cache, state)

    @contextmanager
    def approximation(self, model, mode):
        if mode != "exact":
            raise ValueError(f"approximation mode is not supported for {self.option_error_name}")
        yield

    def close(self, model, expert_cache):
        resources = []
        dspark = getattr(model, "dspark", None)
        if dspark is not None:
            dspark.reset_cache()
            resources.append(dspark.expert_cache)
        resources.extend((getattr(model, "mtp_expert_cache", None),
                          getattr(model, "ane_prefill", None), expert_cache))
        error = None
        seen = set()
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            try:
                resource.close()
            except Exception as cause:
                error = error or cause
        if error is not None:
            raise error
