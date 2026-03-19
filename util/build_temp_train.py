import argparse
import json
from collections import Counter
from pathlib import Path
from pprint import pformat
import sys

import spacy


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.validation import validate_training_data


DEFAULT_INPUT_DIR = ROOT / "temp-training"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "temp_train.py"
DEFAULT_SOURCE_FILES = [
    "clean_enriched.json",
    "clean_keep.json",
    "manually_reviewed_relabelled.json",
]

Entity = tuple[int, int, str]
SpaCyExample = tuple[str, dict[str, list[Entity]]]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine the temp-training JSON files into a validated spaCy training "
            "module under data/."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the temp-training JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output Python module path. Defaults to data/temp_train.py.",
    )
    parser.add_argument(
        "--variable-name",
        default="train_data",
        help="Variable name to assign in the generated Python file.",
    )
    return parser.parse_args()


def resolve_repo_path(path):
    return path if path.is_absolute() else ROOT / path


def serialize_dataset(dataset, variable_name):
    return f"{variable_name} = {pformat(dataset, width=100, sort_dicts=False)}\n"


def normalize_entities(raw_entities, source_name, example_index):
    if not isinstance(raw_entities, list):
        raise ValueError(f"{source_name}[{example_index}] has a non-list entities field")

    entities: list[Entity] = []
    for entity_index, entity in enumerate(raw_entities):
        if not isinstance(entity, (list, tuple)) or len(entity) != 3:
            raise ValueError(
                f"{source_name}[{example_index}] has malformed entity[{entity_index}]: {entity!r}"
            )

        start, end, label = entity
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            raise ValueError(
                f"{source_name}[{example_index}] has invalid entity[{entity_index}] types: {entity!r}"
            )

        entities.append((start, end, label))

    return entities


def load_spacy_examples(path):
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a list in {path}, got {type(raw_data).__name__}")

    examples: list[SpaCyExample] = []
    for example_index, item in enumerate(raw_data):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{path.name}[{example_index}] is not a [text, annotations] pair: {item!r}")

        text, annotations = item
        if not isinstance(text, str) or not isinstance(annotations, dict):
            raise ValueError(
                f"{path.name}[{example_index}] has invalid types for text or annotations: {item!r}"
            )

        entities = normalize_entities(annotations.get("entities", []), path.name, example_index)
        examples.append((text, {"entities": entities}))

    return examples


def merge_examples(input_dir):
    merged_by_text: dict[str, SpaCyExample] = {}
    source_by_text: dict[str, tuple[str, int]] = {}
    label_counts = Counter()
    exact_duplicate_count = 0
    conflict_replacements = []
    source_row_count = 0

    for source_name in DEFAULT_SOURCE_FILES:
        path = input_dir / source_name
        if not path.exists():
            raise FileNotFoundError(f"Missing expected temp-training file: {path}")

        examples = load_spacy_examples(path)
        source_row_count += len(examples)

        for example_index, example in enumerate(examples):
            text, annotations = example
            entities_key = tuple(annotations["entities"])

            if text in merged_by_text:
                existing_text, existing_annotations = merged_by_text[text]
                existing_key = tuple(existing_annotations["entities"])

                if existing_key == entities_key:
                    exact_duplicate_count += 1
                    continue

                previous_source = source_by_text[text]
                conflict_replacements.append({
                    "text": text,
                    "previous_source": previous_source,
                    "replacement_source": (source_name, example_index),
                })
                merged_by_text.pop(text)

            merged_by_text[text] = example
            source_by_text[text] = (source_name, example_index)

    for _, annotations in merged_by_text.values():
        for _, _, label in annotations["entities"]:
            label_counts[label] += 1

    return (
        list(merged_by_text.values()),
        label_counts,
        source_row_count,
        exact_duplicate_count,
        conflict_replacements,
    )


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
    fixed_dataset: list[SpaCyExample] = []
    changes = []

    for example_index, (text, annotations) in enumerate(dataset, start=1):
        doc = nlp.make_doc(text)
        fixed_entities: list[Entity] = []

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

        fixed_dataset.append((text, {"entities": fixed_entities}))

    return fixed_dataset, changes


def write_dataset(path, dataset, variable_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_dataset(dataset, variable_name), encoding="utf-8")


def main():
    args = parse_args()
    input_dir = resolve_repo_path(args.input_dir)
    output_path = resolve_repo_path(args.output)

    combined, label_counts, source_row_count, exact_duplicate_count, conflict_replacements = merge_examples(
        input_dir
    )
    fixed_dataset, offset_changes = build_fixed_dataset(combined)

    issues = validate_training_data(dataset=fixed_dataset, raise_on_error=False)
    if issues:
        raise ValueError(
            "Combined temp dataset still contains invalid entity offsets.\n\n"
            f"{len(issues)} issue(s) found. First issue: {issues[0]}"
        )

    write_dataset(output_path, fixed_dataset, args.variable_name)

    print(f"Read {source_row_count} source rows from {input_dir.relative_to(ROOT)}")
    print(f"Wrote {len(fixed_dataset)} merged examples to {output_path.relative_to(ROOT)}")
    print(f"Labels retained: {', '.join(sorted(label_counts))}")
    print(f"Exact duplicate rows skipped: {exact_duplicate_count}")
    print(f"Conflicting duplicate texts resolved by later-file precedence: {len(conflict_replacements)}")
    print(f"Offset fixes applied: {len(offset_changes)}")

    if conflict_replacements:
        print("Conflict replacements:")
        for replacement in conflict_replacements[:10]:
            print(
                f"  {replacement['previous_source']} -> {replacement['replacement_source']} | "
                f"{replacement['text'][:120]}"
            )


if __name__ == "__main__":
    main()
