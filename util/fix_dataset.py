import argparse
import json
from collections import Counter
from pathlib import Path
from pprint import pformat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.validation import validate_training_data


DEFAULT_INPUT_PATH = ROOT / "consolidated_ner_result_full.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "train.py"

Entity = tuple[int, int, str]
LabeledToken = tuple[str, str]
SpaCyExample = tuple[str, dict[str, list[Entity]]]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the expanded BIO dataset into spaCy training data in the same "
            "shape used by files under data/."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the expanded BIO JSON dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output path for the converted dataset. Use a .py path to emit a "
            "data/train.py-style module, or .json to emit raw JSON."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("python", "json"),
        default=None,
        help="Override the inferred output format.",
    )
    parser.add_argument(
        "--variable-name",
        default="train_data",
        help="Variable name to use when writing a Python dataset module.",
    )
    return parser.parse_args()


def resolve_repo_path(path):
    return path if path.is_absolute() else ROOT / path


def infer_output_format(output_path, requested_format=None):
    if requested_format is not None:
        return requested_format
    return "python" if output_path.suffix == ".py" else "json"


def load_records(input_path):
    with input_path.open("r", encoding="utf-8") as source_file:
        records = json.load(source_file)

    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {input_path}, got {type(records).__name__}")

    return records


def build_text_and_token_offsets(tokens):
    pieces = []
    offsets: list[tuple[int, int]] = []
    position = 0

    for index, token in enumerate(tokens):
        if index > 0:
            pieces.append(" ")
            position += 1

        start = position
        pieces.append(token)
        position += len(token)
        offsets.append((start, position))

    return "".join(pieces), offsets


def get_canonical_labeled_tokens(record):
    labeled_tokens = record.get("labeled_tokens")
    if not isinstance(labeled_tokens, list):
        raise ValueError(f"record_id={record.get('record_id')} is missing a valid labeled_tokens list")

    canonical: list[LabeledToken] = []
    for token_index, item in enumerate(labeled_tokens):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"record_id={record.get('record_id')} has malformed labeled_tokens[{token_index}]: {item!r}"
            )

        token_text, raw_label = item
        if not isinstance(token_text, str) or not isinstance(raw_label, str):
            raise ValueError(
                f"record_id={record.get('record_id')} has non-string token or label at index {token_index}"
            )
        canonical.append((token_text, raw_label))

    return canonical


def bio_to_spans(labeled_tokens):
    tokens = [token_text for token_text, _ in labeled_tokens]
    text, token_offsets = build_text_and_token_offsets(tokens)
    entities: list[Entity] = []

    current_label: str | None = None
    current_start_token: int | None = None
    current_end_token: int | None = None

    def flush_current():
        nonlocal current_label, current_start_token, current_end_token
        if current_label is None:
            return

        if current_start_token is None or current_end_token is None:
            raise ValueError("BIO span state became inconsistent while flushing entities.")

        start_char = token_offsets[current_start_token][0]
        end_char = token_offsets[current_end_token][1]
        entities.append((start_char, end_char, current_label))

        current_label = None
        current_start_token = None
        current_end_token = None

    for token_index, (_, raw_label) in enumerate(labeled_tokens):
        if raw_label == "O":
            flush_current()
            continue

        if "-" not in raw_label:
            raise ValueError(f"Malformed BIO label {raw_label!r} at token index {token_index}")

        prefix, base_label = raw_label.split("-", 1)

        if prefix == "B":
            flush_current()
            current_label = base_label
            current_start_token = token_index
            current_end_token = token_index
        elif prefix == "I":
            if current_label == base_label and current_end_token is not None:
                current_end_token = token_index
            else:
                flush_current()
                current_label = base_label
                current_start_token = token_index
                current_end_token = token_index
        else:
            raise ValueError(f"Unsupported BIO prefix {prefix!r} in label {raw_label!r}")

    flush_current()
    return text, {"entities": entities}


def convert_dataset(records):
    converted: list[SpaCyExample] = []
    label_counts = Counter()
    token_mismatch_count = 0
    length_mismatch_records = []

    for record in records:
        labeled_tokens = get_canonical_labeled_tokens(record)
        source_tokens = record.get("tokens", [])

        if isinstance(source_tokens, list):
            if len(source_tokens) != len(labeled_tokens):
                length_mismatch_records.append(
                    (record.get("record_id"), len(source_tokens), len(labeled_tokens))
                )
            else:
                for source_token, (canonical_token, _) in zip(source_tokens, labeled_tokens):
                    if source_token != canonical_token:
                        token_mismatch_count += 1

        text, annotations = bio_to_spans(labeled_tokens)
        converted.append((text, annotations))

        for _, _, label in annotations["entities"]:
            label_counts[label] += 1

    return converted, label_counts, token_mismatch_count, length_mismatch_records


def write_python_dataset(output_path, dataset, variable_name):
    rendered = f"{variable_name} = {pformat(dataset, width=100, sort_dicts=False)}\n"
    output_path.write_text(rendered, encoding="utf-8")


def write_json_dataset(output_path, dataset):
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(dataset, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def main():
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    output_format = infer_output_format(output_path, args.format)
    records = load_records(input_path)
    converted, label_counts, token_mismatch_count, length_mismatch_records = convert_dataset(records)

    issues = validate_training_data(dataset=converted, raise_on_error=False)
    if issues:
        raise ValueError(
            "Converted dataset contains invalid entity offsets.\n\n"
            f"{len(issues)} issue(s) found. First issue: {issues[0]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "python":
        write_python_dataset(output_path, converted, args.variable_name)
    else:
        write_json_dataset(output_path, converted)

    label_summary = ", ".join(sorted(label_counts))
    print(f"Wrote {len(converted)} examples to {output_path.relative_to(ROOT)}")
    print(f"Labels retained: {label_summary}")
    if token_mismatch_count or length_mismatch_records:
        print(
            "Used labeled_tokens as the canonical token stream "
            f"({token_mismatch_count} token mismatches, "
            f"{len(length_mismatch_records)} length-mismatch records in source tokens)."
        )


if __name__ == "__main__":
    main()
