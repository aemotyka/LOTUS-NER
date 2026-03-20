import argparse
import importlib
import json
import math
import random
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from tqdm.auto import tqdm

from data.test import test_data as prediction_queries
from util.validation import validate_training_data

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "consolidated_query_derivation"
DEFAULT_RESULTS_PATH = ROOT / "outputs" / "test-results.json"
DEFAULT_RUN_LOG_PATH = ROOT / "run.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the spaCy NER model and run predictions on the unlabeled queries in data.test."
    )
    parser.add_argument(
        "--train-data-module",
        default="data.train",
        help=(
            "Python module containing spaCy training data as `train_data`, "
            "for example `data.train_old`."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to write the trained spaCy model.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to write unlabeled test predictions as JSON.",
    )
    parser.add_argument(
        "--run-log-path",
        type=Path,
        default=DEFAULT_RUN_LOG_PATH,
        help="Path to write a copy of console output.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout to use during spaCy updates.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of train_data to hold out for validation metrics.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of train_data to hold out for final test metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/validation splits and shuffling.",
    )
    parser.add_argument(
        "--early-stopping-start-epoch",
        type=int,
        default=25,
        help="Do not stop early before this 1-based epoch number.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many epochs without validation F1 improvement once early stopping is active.",
    )
    return parser.parse_args()


def load_train_data(module_name):
    module = importlib.import_module(module_name)
    train_data = getattr(module, "train_data", None)
    if train_data is None:
        raise ValueError(f"Module {module_name!r} does not define `train_data`.")
    return list(train_data)


def split_dataset(dataset, validation_split, test_split, seed):
    if not 0 <= validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1 (exclusive of 1).")
    if not 0 <= test_split < 1:
        raise ValueError("--test-split must be between 0 and 1 (exclusive of 1).")
    if validation_split + test_split >= 1:
        raise ValueError("--validation-split plus --test-split must be less than 1.")

    required_nonempty_splits = 1 + int(validation_split > 0) + int(test_split > 0)
    if len(dataset) < required_nonempty_splits:
        raise ValueError(
            "Dataset is too small for the requested train/validation/test split. "
            f"Got {len(dataset)} examples but need at least {required_nonempty_splits}."
        )

    if validation_split == 0 and test_split == 0:
        return list(dataset), [], []

    shuffled = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    validation_size = int(len(shuffled) * validation_split)
    test_size = int(len(shuffled) * test_split)

    if validation_split > 0 and validation_size == 0:
        validation_size = 1
    if test_split > 0 and test_size == 0:
        test_size = 1

    while validation_size + test_size >= len(shuffled):
        if test_size >= validation_size and test_size > 0:
            test_size -= 1
        elif validation_size > 0:
            validation_size -= 1
        else:
            break

    validation_data = shuffled[:validation_size]
    test_data = shuffled[validation_size:validation_size + test_size]
    training_data = shuffled[validation_size + test_size:]
    return training_data, validation_data, test_data


def validate_early_stopping_args(start_epoch, patience):
    if start_epoch < 1:
        raise ValueError("--early-stopping-start-epoch must be at least 1.")
    if patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1.")


def build_examples(nlp, dataset):
    return [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in dataset]


def evaluate_dataset(nlp, dataset):
    if not dataset:
        return None

    scores = nlp.evaluate(build_examples(nlp, dataset))
    support_by_label = Counter()
    for _, annotations in dataset:
        for _, _, label in annotations.get("entities", []):
            support_by_label[label] += 1

    per_type_scores = {}
    for label, label_metrics in scores.get("ents_per_type", {}).items():
        per_type_scores[label] = {
            **label_metrics,
            "support": support_by_label.get(label, 0),
        }

    return {
        "precision": scores.get("ents_p", 0.0),
        "recall": scores.get("ents_r", 0.0),
        "f1": scores.get("ents_f", 0.0),
        "support": sum(support_by_label.values()),
        "per_type": per_type_scores,
    }


def format_metrics(metrics, include_support=False):
    if metrics is None:
        return "P=n/a R=n/a F1=n/a"
    formatted = (
        f"P={metrics['precision'] * 100:.2f}% "
        f"R={metrics['recall'] * 100:.2f}% "
        f"F1={metrics['f1'] * 100:.2f}%"
    )
    if include_support:
        formatted = f"{formatted} support={metrics.get('support', 0)}"
    return formatted


def print_epoch_summary(epoch, epochs, losses, train_metrics, validation_metrics, test_metrics):
    ner_loss = losses.get("ner", 0.0)
    print(
        f"Epoch {epoch}/{epochs} | "
        f"loss={ner_loss:.4f} | "
        f"train {format_metrics(train_metrics)} | "
        f"validation {format_metrics(validation_metrics)} | "
        f"test {format_metrics(test_metrics)}"
    )


def get_per_type_metrics(metrics):
    if metrics is None:
        return {}
    return metrics.get("per_type", {})


def format_per_type_label_metrics(label_metrics):
    if not label_metrics:
        return "P=n/a R=n/a F1=n/a support=0"
    return (
        f"P={label_metrics['p'] * 100:.2f}% "
        f"R={label_metrics['r'] * 100:.2f}% "
        f"F1={label_metrics['f'] * 100:.2f}% "
        f"support={label_metrics.get('support', 0)}"
    )


def print_per_type_metrics_table(split_name, metrics):
    if metrics is None:
        return

    per_type_metrics = get_per_type_metrics(metrics)
    print(f"Final {split_name} metrics | {format_metrics(metrics, include_support=True)}")
    if not per_type_metrics:
        return

    print(f"{split_name.capitalize()} metrics by label")
    for label in sorted(per_type_metrics):
        print(f"  {label:<20} {format_per_type_label_metrics(per_type_metrics[label])}")


def train_model(
    train_data,
    validation_data,
    test_data,
    model_path,
    epochs,
    batch_size,
    dropout,
    seed,
    early_stopping_start_epoch,
    early_stopping_patience,
):
    validate_training_data(dataset=train_data)
    if validation_data:
        validate_training_data(dataset=validation_data)
        validate_early_stopping_args(early_stopping_start_epoch, early_stopping_patience)
    if test_data:
        validate_training_data(dataset=test_data)

    random.seed(seed)

    nlp = spacy.load("en_core_web_lg")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for dataset_split in (train_data, validation_data, test_data):
        for _, annotations in dataset_split:
            for _, _, label in annotations["entities"]:
                if label not in ner.labels:
                    ner.add_label(label)

    print(
        f"Dataset split | train={len(train_data)} | "
        f"validation={len(validation_data)} | "
        f"test={len(test_data)} | "
        f"unlabeled_test_queries={len(prediction_queries)}"
    )
    if prediction_queries:
        print(
            "Note: data.test remains an unlabeled prediction set. "
            "Train/validation/test metrics below come from the labeled holdout split."
        )
    if validation_data:
        print(
            "Early stopping | "
            f"start_epoch={early_stopping_start_epoch} | "
            f"patience={early_stopping_patience}"
        )

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        best_validation_f1 = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0

        with tempfile.TemporaryDirectory(prefix="spacy_ner_best_") as best_model_dir_str:
            best_model_dir = Path(best_model_dir_str)

            for epoch in range(epochs):
                random.shuffle(train_data)
                losses = {}
                total_batches = max(1, math.ceil(len(train_data) / batch_size))
                batch_iterator = tqdm(
                    minibatch(train_data, size=batch_size),
                    total=total_batches,
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    leave=False,
                    dynamic_ncols=True,
                    file=sys.stderr,
                )

                for batch in batch_iterator:
                    examples = build_examples(nlp, batch)
                    nlp.update(examples, sgd=optimizer, drop=dropout, losses=losses)
                    batch_iterator.set_postfix(loss=f"{losses.get('ner', 0.0):.4f}")

                train_metrics = evaluate_dataset(nlp, train_data)
                validation_metrics = evaluate_dataset(nlp, validation_data)
                test_metrics = evaluate_dataset(nlp, test_data)
                print_epoch_summary(
                    epoch + 1,
                    epochs,
                    losses,
                    train_metrics,
                    validation_metrics,
                    test_metrics,
                )

                if validation_metrics is None:
                    continue

                current_validation_f1 = validation_metrics["f1"]
                if current_validation_f1 > best_validation_f1:
                    best_validation_f1 = current_validation_f1
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    nlp.to_disk(best_model_dir)
                    continue

                if epoch + 1 >= early_stopping_start_epoch:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print(
                            "Early stopping triggered | "
                            f"best_epoch={best_epoch} | "
                            f"best_validation_F1={best_validation_f1 * 100:.2f}%"
                        )
                        break

            if validation_data and best_epoch > 0:
                nlp = spacy.load(best_model_dir)
                print(
                    "Restored best validation checkpoint | "
                    f"epoch={best_epoch} | "
                    f"validation_F1={best_validation_f1 * 100:.2f}%"
                )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(model_path)
    print(f"Saved trained model to {model_path}")

    final_train_metrics = evaluate_dataset(nlp, train_data)
    final_validation_metrics = evaluate_dataset(nlp, validation_data)
    final_test_metrics = evaluate_dataset(nlp, test_data)
    print_per_type_metrics_table("train", final_train_metrics)
    print_per_type_metrics_table("validation", final_validation_metrics)
    print_per_type_metrics_table("test", final_test_metrics)

    return spacy.load(model_path)


def collect_results(trained_nlp):
    results = []

    for text in prediction_queries:
        doc = trained_nlp(text)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        results.append({"text": text, "entities": entities})

    return results


def write_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote structured test predictions for {len(results)} queries to {path}")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def tee_console_output(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        try:
            print(f"Writing console output to {log_path}")
            yield
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout


def main():
    args = parse_args()
    with tee_console_output(args.run_log_path):
        dataset = load_train_data(args.train_data_module)
        train_data, validation_data, test_data = split_dataset(
            dataset,
            args.validation_split,
            args.test_split,
            args.seed,
        )
        trained_nlp = train_model(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dropout=args.dropout,
            seed=args.seed,
            early_stopping_start_epoch=args.early_stopping_start_epoch,
            early_stopping_patience=args.early_stopping_patience,
        )
        results = collect_results(trained_nlp)
        write_results(args.results_path, results)


if __name__ == "__main__":
    main()
