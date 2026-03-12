# spaCy NER

This project trains a custom spaCy NER model from the examples in `train_data`.

## Validate entity offsets

Run the validator before training:

```bash
python3 test.py
```

What it checks:

- Every `(start, end, label)` span stays inside the source text.
- Every span matches spaCy token boundaries exactly.

If the data is valid, `test.py` prints a success message and exits with status `0`.

If the data is invalid, `test.py` prints:

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
python3 fix_offsets.py
```

Apply the fixes to `data.py`:

```bash
python3 fix_offsets.py --write
```

After writing, run this again to confirm the dataset is clean:

```bash
python3 test.py
```

This is meant for off-by-one and token-boundary issues like trailing spaces or end offsets that run one character
too far. Review the diff after running it.

## Train the model

Run:

```bash
python3 main.py
```

`main.py` runs the same validation first and stops immediately if any training offsets are invalid.

To save the test-set predictions in a structured JSON file for later review, run:

```bash
python3 main.py --results-path outputs/test-results.json
```

The JSON contains one object per test example with:

- `text`
- `entities`
- Each entity's `text`, `label`, `start`, and `end`
