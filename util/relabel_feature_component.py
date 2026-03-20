import argparse
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.validation import validate_training_data


DEFAULT_TRAIN_DATA_PATH = ROOT / "data" / "train.py"
SOURCE_LABEL = "FEATURE_COMPONENT"
TARGET_LABEL = "MATERIAL_TECHNIQUE"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace FEATURE_COMPONENT labels in data/train.py with MATERIAL_TECHNIQUE "
            "without changing any entity offsets."
        )
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=DEFAULT_TRAIN_DATA_PATH,
        help="Path to the training data Python file. Defaults to data/train.py.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the relabeled content back to the training data file.",
    )
    return parser.parse_args()


def resolve_repo_path(path):
    return path if path.is_absolute() else ROOT / path


def count_label_occurrences(text):
    return text.count(f"'{SOURCE_LABEL}'") + text.count(f'"{SOURCE_LABEL}"')


def relabel_text(text):
    updated_text = text.replace(f"'{SOURCE_LABEL}'", f"'{TARGET_LABEL}'")
    return updated_text.replace(f'"{SOURCE_LABEL}"', f'"{TARGET_LABEL}"')


def load_train_data_from_path(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training data module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train_data = getattr(module, "train_data", None)
    if train_data is None:
        raise ValueError(f"{path} does not define `train_data`.")
    return train_data


def main():
    args = parse_args()
    train_data_path = resolve_repo_path(args.train_data_path)

    original_text = train_data_path.read_text(encoding="utf-8")
    replacement_count = count_label_occurrences(original_text)

    if replacement_count == 0:
        print(f"No {SOURCE_LABEL} labels found in {train_data_path.name}.")
        return

    if not args.write:
        print(
            f"Would relabel {replacement_count} {SOURCE_LABEL} occurrences to "
            f"{TARGET_LABEL} in {train_data_path}."
        )
        print(f"Dry run only. Re-run with --write to patch {train_data_path.name}.")
        return

    updated_text = relabel_text(original_text)
    train_data_path.write_text(updated_text, encoding="utf-8")

    updated_train_data = load_train_data_from_path(train_data_path)
    validate_training_data(dataset=updated_train_data)
    print(
        f"Relabeled {replacement_count} {SOURCE_LABEL} occurrences to "
        f"{TARGET_LABEL} in {train_data_path}."
    )


if __name__ == "__main__":
    main()
