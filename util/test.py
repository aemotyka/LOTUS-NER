import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT / "data" / "train.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a spaCy NER dataset for label support, duplicates, dense examples, and basic span issues."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Dataset file to audit. Supports a Python train_data module file or raw JSON list.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_examples(path: Path):
    """
    Supports:
      1) .py file containing a variable assignment like:
         train_data = [("text", {"entities": [...]})]
      2) .json file containing the raw list
    """
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        return json.loads(text)

    match = re.search(r"=\s*(\[.*\])\s*$", text, flags=re.S)
    if not match:
        raise ValueError("Could not find top-level list assignment in Python file.")
    return ast.literal_eval(match.group(1))


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def entity_text(text: str, ent):
    start, end, label = ent
    return text[start:end], label


def has_overlap(entities):
    ents = sorted(entities, key=lambda x: (x[0], x[1]))
    overlaps = []
    for i in range(len(ents) - 1):
        s1, e1, _ = ents[i]
        s2, e2, _ = ents[i + 1]
        if s2 < e1:
            overlaps.append((ents[i], ents[i + 1]))
    return overlaps


def validate_examples(examples):
    bad_bounds = []
    overlaps = []
    for idx, ex in enumerate(examples):
        text, ann = ex
        ents = ann.get("entities", [])
        for ent in ents:
            s, e, _ = ent
            if not (0 <= s < e <= len(text)):
                bad_bounds.append((idx, text, ent))
        ov = has_overlap(ents)
        if ov:
            overlaps.append((idx, text, ov))
    return bad_bounds, overlaps


def audit_dataset(examples):
    num_examples = len(examples)
    entity_counts = []
    label_support = Counter()
    label_text_counts = defaultdict(Counter)
    exact_text_counts = Counter()
    normalized_text_counts = Counter()
    dense_examples = []
    long_examples = []

    for idx, (text, ann) in enumerate(examples):
        ents = ann.get("entities", [])
        entity_counts.append(len(ents))
        exact_text_counts[text] += 1
        normalized_text_counts[normalize_text(text)] += 1

        for ent in ents:
            s, e, label = ent
            label_support[label] += 1
            span_text = text[s:e]
            label_text_counts[label][span_text] += 1

        dense_examples.append((idx, len(ents), len(text), text[:200]))
        long_examples.append((idx, len(text), len(ents), text[:200]))

    dense_examples.sort(key=lambda x: (-x[1], -x[2]))
    long_examples.sort(key=lambda x: (-x[1], -x[2]))

    duplicate_texts = [(t, c) for t, c in exact_text_counts.items() if c > 1]
    duplicate_texts.sort(key=lambda x: -x[1])

    near_duplicates = [(t, c) for t, c in normalized_text_counts.items() if c > 1]
    near_duplicates.sort(key=lambda x: -x[1])

    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Examples: {num_examples}")
    print(f"Total entity spans: {sum(entity_counts)}")
    print(f"Avg entities/example: {mean(entity_counts):.2f}")
    print(f"Median entities/example: {median(entity_counts):.2f}")
    print(f"Max entities in one example: {max(entity_counts) if entity_counts else 0}")
    print()

    print("=" * 80)
    print("LABEL SUPPORT")
    print("=" * 80)
    for label, count in label_support.most_common():
        print(f"{label:20s} {count}")
    print()

    print("=" * 80)
    print("TOP REPEATED ENTITY STRINGS BY LABEL")
    print("=" * 80)
    for label in sorted(label_text_counts.keys()):
        print(f"\n[{label}]")
        for text, count in label_text_counts[label].most_common(15):
            print(f"  {count:4d}  {text}")
    print()

    print("=" * 80)
    print("TOP EXACT DUPLICATE TEXTS")
    print("=" * 80)
    if duplicate_texts:
        for text, count in duplicate_texts[:20]:
            print(f"{count:4d}  {text[:200]}")
    else:
        print("No exact duplicate texts found.")
    print()

    print("=" * 80)
    print("TOP NORMALIZED NEAR-DUPLICATE TEXTS")
    print("=" * 80)
    if near_duplicates:
        for text, count in near_duplicates[:20]:
            print(f"{count:4d}  {text[:200]}")
    else:
        print("No normalized near-duplicate texts found.")
    print()

    print("=" * 80)
    print("DENSEST EXAMPLES (MOST ENTITY SPANS)")
    print("=" * 80)
    for idx, ent_count, text_len, preview in dense_examples[:20]:
        print(f"idx={idx:4d} | ents={ent_count:2d} | len={text_len:4d} | {preview}")
    print()

    print("=" * 80)
    print("LONGEST EXAMPLES")
    print("=" * 80)
    for idx, text_len, ent_count, preview in long_examples[:20]:
        print(f"idx={idx:4d} | len={text_len:4d} | ents={ent_count:2d} | {preview}")
    print()

    total_entities = sum(entity_counts)
    sorted_counts = sorted(entity_counts, reverse=True)
    running = 0
    cutoffs = [0.1, 0.25, 0.5]
    cutoff_results = {}
    j = 0
    for i, c in enumerate(sorted_counts, start=1):
        running += c
        while j < len(cutoffs) and total_entities > 0 and running / total_entities >= cutoffs[j]:
            cutoff_results[cutoffs[j]] = i
            j += 1

    print("=" * 80)
    print("ENTITY CONCENTRATION")
    print("=" * 80)
    for frac in cutoffs:
        needed = cutoff_results.get(frac, None)
        if needed is not None:
            print(f"Top {needed} examples account for {int(frac * 100)}% of all entity spans")
    print()


def main():
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    examples = load_examples(input_path)

    bad_bounds, overlaps = validate_examples(examples)

    print(f"Auditing dataset: {input_path}")
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"Examples loaded: {len(examples)}")
    print(f"Bad-boundary examples: {len(bad_bounds)}")
    print(f"Overlap examples: {len(overlaps)}")
    print()

    if bad_bounds:
        print("Sample bad-boundary issues:")
        for idx, text, ent in bad_bounds[:10]:
            print(f"idx={idx} | ent={ent} | text={text[:200]}")
        print()

    if overlaps:
        print("Sample overlap issues:")
        for idx, text, ov in overlaps[:10]:
            print(f"idx={idx} | text={text[:200]}")
            for pair in ov:
                print(f"  {pair}")
        print()

    audit_dataset(examples)


if __name__ == "__main__":
    main()
