"""Request-local Qwen MTP policy, independent of verification and cache state."""
from dataclasses import dataclass


@dataclass
class MTPDraftPolicy:
    draft_tokens: int = 5
    zero_acceptance_limit: int = 1
    zero_acceptance_streak: int = 0

    def __post_init__(self):
        for name, maximum in (("draft_tokens", 5), ("zero_acceptance_limit", 32)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Qwen MTP {name} must be an integer from 1 through {maximum}")

    def limit(self, remaining):
        return max(0, min(self.draft_tokens, remaining - 1))

    def observe(self, accepted):
        """Return whether this request should permanently fall back to target."""
        self.zero_acceptance_streak = self.zero_acceptance_streak + 1 if accepted == 0 else 0
        return self.zero_acceptance_streak >= self.zero_acceptance_limit
