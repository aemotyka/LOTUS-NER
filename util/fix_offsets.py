import argparse
from pathlib import Path
import sys

import spacy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.train import train_data
from util.validation import validate_training_data


DATA_FILE = ROOT / "data" / "train.py"


def propose_span_fix(doc, start, end, label):
    clamped_start = max(0, min(start, len(doc.text)))
    clamped_end = max(0, min(end, len(doc.text)))

    if clamped_start >= clamped_end:
        return None

    overlapping_tokens = [
        token for token in doc if token.idx < clamped_end and token.idx + len(token) > clamped_start
    ]
    if not overlapping_tokens:
        return None

    new_start = overlapping_tokens[0].idx
    new_end = overlapping_tokens[-1].idx + len(overlapping_tokens[-1])

    if doc.char_span(new_start, new_end, label=label, alignment_mode="strict") is None:
        return None

    return (new_start, new_end, label)


def build_fixed_dataset(dataset):
    nlp = spacy.blank("en")
    fixed_dataset = []
    changes = []

    for example_index, (text, annotations) in enumerate(dataset, start=1):
        doc = nlp.make_doc(text)
        fixed_entities = []

        for entity_index, (start, end, label) in enumerate(annotations.get("entities", []), start=1):
            if doc.char_span(start, end, label=label, alignment_mode="strict") is not None:
                fixed_entities.append((start, end, label))
                continue

            replacement = propose_span_fix(doc, start, end, label)
            if replacement is None:
                raise ValueError(
                    f'Could not auto-fix example {example_index}, entity {entity_index}: '
                    f'"{text}" ({start}, {end}, "{label}")'
                )

            fixed_entities.append(replacement)
            changes.append({
                "example_index": example_index,
                "entity_index": entity_index,
                "text": text,
                "old": (start, end, label),
                "new": replacement,
            })

        fixed_annotations = dict(annotations)
        fixed_annotations["entities"] = fixed_entities
        fixed_dataset.append((text, fixed_annotations))

    return fixed_dataset, changes


def serialize_train_data(dataset):
    lines = ["train_data = [\n"]
    for example in dataset:
        lines.append(f"    {example!r},\n")
    lines.append("]\n")
    return "".join(lines)


def write_fixed_train_data(path, fixed_dataset):
    path.write_text(serialize_train_data(fixed_dataset))


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fix invalid train_data offsets by snapping spans to overlapping token boundaries."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the fixed train_data back to data/train.py.",
    )
    args = parser.parse_args()

    fixed_dataset, changes = build_fixed_dataset(train_data)

    if not changes:
        print("No invalid offsets found. data/train.py does not need changes.")
        return

    for change in changes:
        print(
            f'Example {change["example_index"]}, entity {change["entity_index"]}: '
            f'{change["old"]} -> {change["new"]} in "{change["text"]}"'
        )

    if not args.write:
        print("\nDry run only. Re-run with --write to patch data/train.py.")
        return

    write_fixed_train_data(DATA_FILE, fixed_dataset)
    validate_training_data(dataset=fixed_dataset)
    print(f"\nPatched {len(changes)} entities in {DATA_FILE.name}.")


if __name__ == "__main__":
    main()
