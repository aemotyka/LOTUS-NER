import argparse
import ast
import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.validation import validate_training_data


DEFAULT_INPUT_PATH = ROOT / "data" / "train.py"
DEFAULT_MAX_TOKENS = 20

LABEL_WEIGHTS = {
    "OBJECT_TYPE": 10,
    "ARTIST": 6,
    "BRAND": 6,
    "MATERIAL_TECHNIQUE": 4,
    "SUBJECT_THEME": 4,
    "ORIGIN": 3,
    "PERIOD": 3,
    "HISTORICAL_CONTEXT": 2,
}

RECOVERABLE_OBJECT_TYPE_PATTERNS = [
    ("painting", re.compile(r"\bpainting\b", flags=re.IGNORECASE)),
    ("watercolor", re.compile(r"\bwatercolor\b", flags=re.IGNORECASE)),
    ("etching", re.compile(r"\betching\b", flags=re.IGNORECASE)),
    ("pastel", re.compile(r"\bpastel\b", flags=re.IGNORECASE)),
    ("drawing", re.compile(r"\bdrawing\b", flags=re.IGNORECASE)),
    ("bust", re.compile(r"\bbust\b", flags=re.IGNORECASE)),
    ("sculpture", re.compile(r"\bsculpture\b", flags=re.IGNORECASE)),
]

SYNTHETIC_OBJECT_TYPE_RULES = [
    ("watercolor", re.compile(r"\bwatercolor on paper\b", flags=re.IGNORECASE)),
    ("etching", re.compile(r"\betching(?: and drypoint)? on paper\b", flags=re.IGNORECASE)),
    ("pastel", re.compile(r"\bpastel on (?:card|paper)\b", flags=re.IGNORECASE)),
    ("drawing", re.compile(r"\bpencil on paper\b", flags=re.IGNORECASE)),
    ("painting", re.compile(r"\boil on (?:canvas|board|panel)\b", flags=re.IGNORECASE)),
]

BRONZE_RULE = re.compile(r"\bbronze(?:/ivory)?\b", flags=re.IGNORECASE)
BRONZE_SCULPTURE_HINTS = [
    re.compile(r"\bsculptor\b", flags=re.IGNORECASE),
    re.compile(r"\bsculptural\b", flags=re.IGNORECASE),
    re.compile(r"\bsculpture\b", flags=re.IGNORECASE),
    re.compile(r"\bbust\b", flags=re.IGNORECASE),
    re.compile(r"\bnapoleon\b", flags=re.IGNORECASE),
    re.compile(r"\bon horseback\b", flags=re.IGNORECASE),
    re.compile(r"\barabian dancer\b", flags=re.IGNORECASE),
    re.compile(r"\beternal spring\b", flags=re.IGNORECASE),
    re.compile(r"\bfarewell kiss\b", flags=re.IGNORECASE),
    re.compile(r"\bnude\b", flags=re.IGNORECASE),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trim long training examples down to a high-value token window, recover some "
            "implicit art object types, and write the result to a new dataset module "
            "without modifying the source dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input Python training dataset file. Defaults to data/train.py.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of whitespace tokens to keep per example.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to data/trimmed_train{max_tokens}.py.",
    )
    parser.add_argument(
        "--dropped-report",
        type=Path,
        default=None,
        help="Optional dropped-report path. Defaults to data/trimmed_train{max_tokens}_dropped.txt.",
    )
    return parser.parse_args()


def default_output_path(max_tokens: int) -> Path:
    return ROOT / "data" / f"trimmed_train{max_tokens}.py"


def default_dropped_report_path(max_tokens: int) -> Path:
    return ROOT / "data" / f"trimmed_train{max_tokens}_dropped.txt"


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_examples(path: Path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"=\s*(\[.*\])\s*$", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not parse training list from Python file: {path}")
    return ast.literal_eval(match.group(1))


def serialize_examples(examples):
    lines = ["train_data = [\n"]
    for text, ann in examples:
        lines.append(f"    ({text!r}, {ann!r}),\n")
    lines.append("]\n")
    return "".join(lines)


def tokenize_with_spans(text: str):
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def token_count(text: str) -> int:
    return len(tokenize_with_spans(text))


def entities_in_window(entities, win_start, win_end):
    kept = []
    partial = []
    for start, end, label in entities:
        if start >= win_start and end <= win_end:
            kept.append((start, end, label))
        elif not (end <= win_start or start >= win_end):
            partial.append((start, end, label))
    return kept, partial


def score_window(text, entities, win_start, win_end):
    kept, partial = entities_in_window(entities, win_start, win_end)

    if not kept:
        return (-10_000, kept, partial)

    score = 0
    labels = [label for _, _, label in kept]

    for _, _, label in kept:
        score += LABEL_WEIGHTS.get(label, 1)

    if "OBJECT_TYPE" in labels:
        score += 15

    if "ARTIST" in labels or "BRAND" in labels:
        score += 6

    score += len(kept) * 2
    score -= len(partial) * 8
    score -= win_start / 100000.0

    return (score, kept, partial)


def choose_best_window(text, entities, max_tokens):
    tokens = tokenize_with_spans(text)

    if len(tokens) <= max_tokens:
        return (0, len(text), entities, [])

    best = None

    for i in range(0, len(tokens) - max_tokens + 1):
        start_char = tokens[i][1]
        end_char = tokens[i + max_tokens - 1][2]
        score, kept, partial = score_window(text, entities, start_char, end_char)
        candidate = (score, start_char, end_char, kept, partial)

        if best is None or candidate[0] > best[0]:
            best = candidate

    _, start_char, end_char, kept, partial = best
    return (start_char, end_char, kept, partial)


def rebase_entities(entities, offset):
    return [(start - offset, end - offset, label) for start, end, label in entities]


def overlapping_window_tokens(tokens, start_char, end_char):
    return [token for token in tokens if token[1] < end_char and token[2] > start_char]


def exclude_partial_entity_fragments(text, start_char, end_char, partial_entities):
    if not partial_entities:
        return start_char, end_char

    tokens = tokenize_with_spans(text)
    adjusted_start = start_char
    adjusted_end = end_char

    for entity_start, entity_end, _ in partial_entities:
        if entity_start < adjusted_start < entity_end:
            overlapping_tokens = overlapping_window_tokens(tokens, adjusted_start, entity_end)
            if overlapping_tokens:
                adjusted_start = max(adjusted_start, overlapping_tokens[-1][2])
            else:
                adjusted_start = max(adjusted_start, entity_end)

        if entity_start < adjusted_end < entity_end:
            overlapping_tokens = overlapping_window_tokens(tokens, entity_start, adjusted_end)
            if overlapping_tokens:
                adjusted_end = min(adjusted_end, overlapping_tokens[0][1])
            else:
                adjusted_end = min(adjusted_end, entity_start)

    return adjusted_start, adjusted_end


def trim_example(example, max_tokens):
    text, ann = example
    entities = sorted(ann.get("entities", []), key=lambda item: (item[0], item[1]))

    start_char, end_char, _, partial_entities = choose_best_window(
        text,
        entities,
        max_tokens=max_tokens,
    )
    start_char, end_char = exclude_partial_entity_fragments(
        text,
        start_char,
        end_char,
        partial_entities,
    )

    if start_char >= end_char:
        return "", {"entities": []}

    window_text = text[start_char:end_char]

    left_trim = len(window_text) - len(window_text.lstrip())
    actual_start = start_char + left_trim
    trimmed_text = text[actual_start:end_char].rstrip()
    actual_end = actual_start + len(trimmed_text)

    final_kept = [
        (start, end, label)
        for start, end, label in entities
        if start >= actual_start and end <= actual_end
    ]

    return trimmed_text, {"entities": rebase_entities(final_kept, actual_start)}


def has_overlap(entities):
    ordered = sorted(entities, key=lambda item: (item[0], item[1]))
    for i in range(len(ordered) - 1):
        start_a, end_a, _ = ordered[i]
        start_b, _, _ = ordered[i + 1]
        if start_b < end_a:
            return True
    return False


def validate_example(text, ann):
    entities = ann.get("entities", [])

    for start, end, label in entities:
        if not (0 <= start < end <= len(text)):
            return False, f"Bad bounds: {(start, end, label)}"

    if has_overlap(entities):
        return False, "Overlapping entities"

    return True, ""


def add_entity(ann, entity):
    entities = list(ann.get("entities", []))
    entities.append(entity)
    entities.sort(key=lambda item: (item[0], item[1], item[2]))
    return {"entities": entities}


def overlaps_existing_entities(start, end, entities):
    for entity_start, entity_end, _ in entities:
        if max(start, entity_start) < min(end, entity_end):
            return True
    return False


def has_label(ann, label):
    return any(entity_label == label for _, _, entity_label in ann.get("entities", []))


def has_art_context(ann):
    art_labels = {"ARTIST", "SUBJECT_THEME", "HISTORICAL_CONTEXT"}
    return any(label in art_labels for _, _, label in ann.get("entities", []))


def find_existing_object_type_entity(text, ann):
    entities = ann.get("entities", [])

    for _, pattern in RECOVERABLE_OBJECT_TYPE_PATTERNS:
        for match in pattern.finditer(text):
            if match.start() > 60:
                continue
            if overlaps_existing_entities(match.start(), match.end(), entities):
                continue
            return (match.start(), match.end(), "OBJECT_TYPE")

    return None


def infer_synthetic_object_type(text, ann):
    if not has_art_context(ann):
        return None

    for object_type, pattern in SYNTHETIC_OBJECT_TYPE_RULES:
        if pattern.search(text):
            return object_type

    if BRONZE_RULE.search(text):
        for hint in BRONZE_SCULPTURE_HINTS:
            if hint.search(text):
                return "sculpture"

    return None


def append_synthetic_object_type(text, ann, object_type):
    separator = " " if text.endswith((",", ";", ":")) else ", "
    start = len(text) + len(separator)
    new_text = f"{text}{separator}{object_type}"
    new_ann = add_entity(ann, (start, start + len(object_type), "OBJECT_TYPE"))
    return new_text, new_ann


def recover_object_type(example, trimmed_text, trimmed_ann, max_tokens):
    if has_label(trimmed_ann, "OBJECT_TYPE"):
        return trimmed_text, trimmed_ann, False

    recovered_entity = find_existing_object_type_entity(trimmed_text, trimmed_ann)
    if recovered_entity is not None:
        return trimmed_text, add_entity(trimmed_ann, recovered_entity), True

    synthetic_object_type = infer_synthetic_object_type(trimmed_text, trimmed_ann)
    if synthetic_object_type is None:
        return trimmed_text, trimmed_ann, False

    required_tokens = token_count(synthetic_object_type)
    if token_count(trimmed_text) + required_tokens > max_tokens:
        reduced_budget = max_tokens - required_tokens
        if reduced_budget < 1:
            return trimmed_text, trimmed_ann, False
        trimmed_text, trimmed_ann = trim_example(example, max_tokens=reduced_budget)

        recovered_entity = find_existing_object_type_entity(trimmed_text, trimmed_ann)
        if recovered_entity is not None:
            return trimmed_text, add_entity(trimmed_ann, recovered_entity), True

        synthetic_object_type = infer_synthetic_object_type(trimmed_text, trimmed_ann)
        if synthetic_object_type is None:
            return trimmed_text, trimmed_ann, False

    return append_synthetic_object_type(trimmed_text, trimmed_ann, synthetic_object_type) + (True,)


def should_drop(text, ann, max_tokens):
    entities = ann.get("entities", [])

    if not entities:
        return True, "No entities after trimming"
    if token_count(text) > max_tokens:
        return True, "Still exceeds token limit"

    return False, ""


def write_dropped_report(path: Path, dropped):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, reason, text, ann in dropped:
            handle.write(f"IDX: {index}\n")
            handle.write(f"REASON: {reason}\n")
            handle.write(f"TEXT: {text}\n")
            handle.write(f"ANN: {ann}\n")
            handle.write("-" * 80 + "\n")


def main():
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output) if args.output else default_output_path(args.max_tokens)
    dropped_report_path = (
        resolve_repo_path(args.dropped_report)
        if args.dropped_report
        else default_dropped_report_path(args.max_tokens)
    )

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1.")

    examples = load_examples(input_path)
    kept_examples = []
    dropped = []
    recovered_object_types = 0

    for index, example in enumerate(examples):
        trimmed_text, trimmed_ann = trim_example(example, max_tokens=args.max_tokens)
        trimmed_text, trimmed_ann, recovered = recover_object_type(
            example,
            trimmed_text,
            trimmed_ann,
            max_tokens=args.max_tokens,
        )
        if recovered:
            recovered_object_types += 1

        ok, reason = validate_example(trimmed_text, trimmed_ann)
        if not ok:
            dropped.append((index, reason, trimmed_text, trimmed_ann))
            continue

        drop, reason = should_drop(trimmed_text, trimmed_ann, max_tokens=args.max_tokens)
        if drop:
            dropped.append((index, reason, trimmed_text, trimmed_ann))
            continue

        kept_examples.append((trimmed_text, trimmed_ann))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_examples(kept_examples), encoding="utf-8")
    write_dropped_report(dropped_report_path, dropped)
    validate_training_data(dataset=kept_examples)

    print(f"Input dataset: {input_path}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Original examples: {len(examples)}")
    print(f"Kept examples: {len(kept_examples)}")
    print(f"Dropped examples: {len(dropped)}")
    print(f"Recovered implicit object types: {recovered_object_types}")
    print(f"Wrote trimmed dataset to: {output_path}")
    print(f"Wrote dropped report to: {dropped_report_path}")


if __name__ == "__main__":
    main()
