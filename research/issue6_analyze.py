"""Find sustained verbatim repeats; retain text for human review of other loops."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def analyze(text):
    repeats = []
    for match in re.finditer(r"(.{1,512}?)\1{3,}", text, re.DOTALL):
        if len(match[0]) >= 120 and match[1].strip():
            repeats.append(dict(start_char_zero_based=match.start(), end_char_exclusive=match.end(),
                                unit_chars=len(match[1]), repeats=len(match[0]) // len(match[1]),
                                unit=match[1]))
    paras = Counter(x.strip() for x in text.split("\n") if len(x.strip()) >= 30)
    return dict(chars=len(text), sha256=hashlib.sha256(text.encode()).hexdigest(),
                sustained_verbatim_repeats=repeats,
                duplicate_lines=[dict(text=x, count=n) for x, n in paras.most_common() if n >= 3])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = []
    for path in sorted(args.directory.glob("**/status.json")):
        value = json.loads(path.read_text())
        value["path"] = str(path.parent)
        stopfile = path.parent / "manual-stop.json"
        if stopfile.exists():
            value["manual_stop"] = json.loads(stopfile.read_text())
            value["interpretation"] = "Investigator stopped a confirmed loop; stream incompleteness is intentional, not an independently observed server error."
        for key in ["reasoning", "content"]:
            textfile = path.parent / (key + ".txt")
            if textfile.exists():
                value[key + "_analysis"] = analyze(textfile.read_text())
        results.append(value)
    print(json.dumps(dict(method="exact adjacent repeats: unit 1-512 chars, at least 4 repeats and 120 total chars; nonadjacent lines >=30 chars repeated >=3 times; requires human interpretation", cases=results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
