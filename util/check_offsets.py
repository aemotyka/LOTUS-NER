from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.validation import format_validation_issues, validate_training_data


def main():
    issues = validate_training_data(raise_on_error=False)

    if issues:
        print("Invalid entity offsets found in train_data:\n")
        print(format_validation_issues(issues))
        raise SystemExit(1)

    print("All train_data entity offsets align with spaCy token boundaries.")


if __name__ == "__main__":
    main()
