"""Validate the built-in catalog: python -m deepseek_v4_ssd.model_support."""

from . import DESCRIPTORS, get_support, support_types


def main():
    kinds = {descriptor.kind for descriptor in DESCRIPTORS}
    if kinds != set(support_types()):
        raise SystemExit("Model descriptors and registered implementations do not match")
    for descriptor in DESCRIPTORS:
        support = get_support(descriptor.kind)
        if support.descriptor != descriptor:
            raise SystemExit(f"Incorrect descriptor for {descriptor.kind}")
        print(f"OK: {descriptor.kind} -> {descriptor.api_model_id}")


if __name__ == "__main__":
    main()
