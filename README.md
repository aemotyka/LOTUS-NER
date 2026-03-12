# spaCy NER

This repo fine-tunes a spaCy NER model to parse auction search queries into structured pieces used by the search pipeline. Each query is decomposed into:

TYPE: the object being searched for, including meaningful compound types like coffee table, pocket watch, or machine gun

MAKER_ARTIST: the artist, maker, or brand when one is explicitly present

DESCRIPTOR: other meaningful modifiers, kept as the smallest span that preserves meaning, such as white gold, old mine cut diamonds, calendar complication, or Madonna and Child

## Run

Train the model and generate test results:

```bash
python main.py
```

Check training-data offsets before training:

```bash
python util/check_offsets.py
```

Auto-fix offset issues in `data/train.py`:

```bash
python util/fix_offsets.py --write
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

After `python main.py`, the trained model is written to:

`custom_query_derivation/`

The test run output is written to:

`outputs/test-results.json`
