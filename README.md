# spaCy NER

This project trains a custom spaCy NER model from:

- `train_data` in `data/train.py`
- `test_data` in `data/test.py`

`data/test.py` also keeps the original labeled tuples in `annotated_test_data` if you want to inspect those later.

## Validate entity offsets

Run the validator before training:

```bash
python3 util/check_offsets.py
```

What it checks:

- Every `(start, end, label)` span stays inside the source text.
- Every span matches spaCy token boundaries exactly.

If the data is valid, `util/check_offsets.py` prints a success message and exits with status `0`.

If the data is invalid, `util/check_offsets.py` prints:

- The example text
- The offending entity tuple
- The text slice captured by those offsets
- The token boundaries spaCy sees for that example

In that case it exits with status `1`, so you can use it as a quick pre-training check.

## Auto-fix invalid offsets

There is a CLI to patch common boundary mistakes in `train_data` automatically by snapping invalid spans to the
overlapping spaCy token boundaries.

Preview the proposed fixes:

```bash
python3 util/fix_offsets.py
```

Apply the fixes to `data/train.py`:

```bash
python3 util/fix_offsets.py --write
```

After writing, run this again to confirm the dataset is clean:

```bash
python3 util/check_offsets.py
```

This is meant for off-by-one and token-boundary issues like trailing spaces or end offsets that run one character
too far. Review the diff after running it.

## Train the model

Run:

```bash
python3 main.py
```

`main.py` runs the same validation first and stops immediately if any training offsets are invalid. Each run
overwrites `outputs/test-results.json` by default.

To save the test-set predictions somewhere else, run:

```bash
python3 main.py --results-path outputs/custom-results.json
```

The JSON contains one object per test example with:

- `text`
- `entities`
- Each entity's `text`, `label`, `start`, and `end`

## Notebook inference

If you want to run the trained model on your own inputs without using terminal Python snippets, open:

`notebooks/inference.ipynb`

The notebook:

- loads `custom_query_derivation`
- lets you edit a single `text` value or a list of `texts`
- returns structured entity output for manual inspection

Train the model with `python3 main.py` first if the saved model folder does not exist yet.
