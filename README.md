# LOTUS NER

LOTUS NER trains a spaCy named entity recognition model to extract structured fields from auction and art search queries.

## Train

Run a benchmark training job like this:

```bash
python3 main.py \
  --train-data-module data.trimmed_train20 \
  --run-log-path outputs/run-trimmed20.txt \
  --results-path outputs/test-results-trimmed20.json
```
