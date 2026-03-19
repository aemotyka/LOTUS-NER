import argparse
import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.validation import format_validation_issues, validate_training_data


def load_train_data(module_name):
    module = importlib.import_module(module_name)
    train_data = getattr(module, "train_data", None)
    if train_data is None:
        raise ValueError(f"Module {module_name!r} does not define `train_data`.")
    return train_data


def main():
    parser = argparse.ArgumentParser(
        description="Validate that train_data entity offsets align to spaCy token boundaries."
    )
    parser.add_argument(
        "--train-data-module",
        default="data.train",
        help="Python module containing spaCy training data as `train_data`.",
    )
    args = parser.parse_args()

    train_data = load_train_data(args.train_data_module)
    issues = validate_training_data(dataset=train_data, raise_on_error=False)

    if issues:
        print(f"Invalid entity offsets found in {args.train_data_module}:\n")
        print(format_validation_issues(issues))
        raise SystemExit(1)

    print(f"All entity offsets in {args.train_data_module} align with spaCy token boundaries.")


if __name__ == "__main__":
    main()
