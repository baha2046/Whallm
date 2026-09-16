"""Convert expert payload budgets to whole slots using the installed model.

These budgets cover expert blobs, not allocator overhead or the process peak.
Absent budgets preserve legacy slot settings exactly.
"""
from dataclasses import replace


BUDGET_FIELDS = ("expert_cache_bytes", "mtp_cache_bytes", "dspark_cache_bytes")


def validate_cache_budgets(config):
    for name in BUDGET_FIELDS:
        value = getattr(config, name, None)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
            or not 0 < value <= 2**40
        ):
            raise ValueError(f"{name} must be a positive integer of at most 2**40 bytes")


def resolve_cache_budgets(installed, config):
    validate_cache_budgets(config)
    changes = {}
    for budget_name, slot_name, enabled in (
        ("expert_cache_bytes", "slots", True),
        ("mtp_cache_bytes", "mtp_slots", getattr(config, "mtp_enabled", False)),
        ("dspark_cache_bytes", "dspark_slots", getattr(config, "dspark_enabled", False)),
    ):
        budget = getattr(config, budget_name, None)
        if budget is None or not enabled:
            continue
        blob = installed.expert_blob_size
        minimum = installed.selected_expert_count
        if slot_name == "dspark_slots":
            # V4.1 uses its own DSpark expert count, checked by its loader too.
            minimum = max(30, minimum * installed.dspark_block_size)
        elif slot_name == "mtp_slots":
            minimum = max(10, minimum)
        slots = budget // blob
        if slots < minimum:
            raise ValueError(
                f"{budget_name} is too small: at least {minimum * blob} bytes "
                f"are required for {minimum} experts"
            )
        changes[slot_name] = slots
    return replace(config, **changes) if changes else config
