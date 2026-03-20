# spaCy NER

This repo fine-tunes a spaCy NER model to parse auction search queries into structured pieces used by the search pipeline. Each query is decomposed into:

OBJECT_TYPE: the item or object being searched for

ARTIST / BRAND: artist, maker, or brand when explicitly present

MATERIAL_TECHNIQUE / SUBJECT_THEME / ORIGIN / PERIOD / HISTORICAL_CONTEXT: expanded descriptive labels retained from the new training dataset

## Run

Train the model and generate test results:

```bash
python3 main.py
```

By default, training uses an 80/10/10 train/validation/test split with validation-based early stopping and reports train, validation, and test metrics.

Check training-data offsets before training:

```bash
python3 util/check_offsets.py
```

Validate the legacy dataset instead:

```bash
python3 util/check_offsets.py --train-data-module data.train_old
```

Convert an expanded BIO JSON file from the repo root into `/data` training format:

```bash
python3 util/fix_dataset.py --input consolidated_ner_result_full.json --output data/train.py
```

Auto-fix offset issues in `data/train.py`:

```bash
python3 util/fix_offsets.py --write
```

Relabel all `FEATURE_COMPONENT` entities in `data/train.py` to `MATERIAL_TECHNIQUE`:

```bash
python3 util/relabel_feature_component.py --write
```

For notebook-based inference, open:

`notebooks/inference.ipynb`

## Files

Training data:

`data/train.py`

Test inputs:

`data/test.py`

Training / inference entrypoint:

`main.py`

Validation and data-fix utilities:

`util/`

Notebook for running the trained model on your own inputs:

`notebooks/inference.ipynb`

## Output

After `python3 main.py`, the trained model is written to:

`models/consolidated_query_derivation/`

The test run output is written to:

`outputs/test-results.json`
