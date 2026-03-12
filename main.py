import argparse
import json
import random
from pathlib import Path

import spacy
from spacy.training.example import Example
from spacy.util import minibatch

from data.test import test_data
from data.train import train_data
from util.validation import validate_training_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train the spaCy NER model and run predictions on test_data.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("outputs/test-results.json"),
        help="Path to write test predictions as JSON.",
    )
    return parser.parse_args()


def train_model():
    validate_training_data()

    nlp = spacy.load("en_core_web_lg")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations["entities"]:
            if ent[2] not in ner.labels:
                ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        epochs = 50
        for epoch in range(epochs):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=2)
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=0.5, losses=losses)
            print(f"Epoch {epoch + 1}, Losses: {losses}")
    model_path = Path("custom_query_derivation")
    nlp.to_disk(model_path)
    return spacy.load(model_path)


def collect_results(trained_nlp):
    results = []

    for text in test_data:
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
    print(f"Wrote structured test results to {path}")


def main():
    args = parse_args()
    trained_nlp = train_model()
    results = collect_results(trained_nlp)
    write_results(args.results_path, results)


if __name__ == "__main__":
    main()
