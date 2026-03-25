import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT / "all_consolidated_products_run_1.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten a product JSON export into a JSON list of plain inference strings, "
            "where each entry is the product title plus description."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSON file containing a list of product objects.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON file. Defaults to data/<input_stem>_flattened.json relative to the repo root."
        ),
    )
    parser.add_argument(
        "--separator",
        default=" ",
        help="Separator inserted between title and description. Defaults to a single space.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def default_output_path(input_path: Path) -> Path:
    return ROOT / "data" / f"{input_path.stem}_flattened.json"


def normalize_text(value) -> str:
    return " ".join(str(value or "").split())


def flatten_record(record, separator: str) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"Expected each record to be a dict, got {type(record).__name__}.")

    title = normalize_text(record.get("title"))
    description = normalize_text(record.get("description"))
    parts = [part for part in (title, description) if part]
    return separator.join(parts)


def main():
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output) if args.output else default_output_path(input_path)

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected top-level JSON list in {input_path}, got {type(rows).__name__}.")

    flattened = []
    skipped_empty = 0

    for row in rows:
        text = flatten_record(row, args.separator)
        if not text:
            skipped_empty += 1
            continue
        flattened.append(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(flattened, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Loaded {len(rows)} records from {input_path}")
    print(f"Wrote {len(flattened)} flattened entries to {output_path}")
    print(f"Skipped empty records: {skipped_empty}")

    if flattened:
        print("Sample entries:")
        for index, text in enumerate(flattened[:3], start=1):
            print(f"{index}. {text[:300]}")


if __name__ == "__main__":
    main()
