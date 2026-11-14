# virtuos-eval

A draft corpus of cross-lineage wisdom and a forced-choice benchmark for models.

Each entry has four parts: the situation, what the wise person did or said, why it
worked, and the tempting wrong answer. The corpus carries 1,489 publishable
entries, sources and originals where they are held.

This is a draft. Ten entries are approved, no entry has a human label, and no
translation is verified. Treat every score as a test of the pipeline, not a
measure of wisdom.

## Install

No dependencies. Python 3.10 or later.

```
pip install https://github.com/keith448/virtuos-eval/releases/download/v0.1.1/virtuos_eval-0.1.1-py3-none-any.whl
```

The package is not on PyPI yet. The wheel is attached to the v0.1.1 release.

## Use

```
python -m virtuos_eval stats
python -m virtuos_eval list --virtue reciprocity --limit 10
python -m virtuos_eval validate --virtue reciprocity
python -m virtuos_eval test --provider openai --model <model> --n 100 --virtue reciprocity
```

`test` builds forced-choice items: one situation, three responses. The default
`fit` design puts the entry's wise response against two wise responses from other
stories about the same virtue; only one fits the situation. Scoring is exact
match, so no judge model is needed. Each run writes `items.jsonl`,
`results.jsonl`, `scores.csv`, `report.md` and `meta.json` with a Wilson interval.

Providers: `echo`, `random`, `longest`, `shortest`, `openai`, `anthropic`,
`ollama`. The first four need no key. `longest` measures the length cue directly.

`validate` checks a set of items for three cheap cues: answer position, answer
length, and copying the situation's words.

## For an app

Keep the score in memory:

```python
from virtuos_eval import Corpus, build_items, evaluate, get_provider

corpus = Corpus.load()
items = build_items(corpus, virtues=["reciprocity"], n=30, seed=0)
provider = get_provider("openai", model="gpt-5")
summary = evaluate(items, provider)
print(summary["accuracy"], summary["wilson95"], summary["errors"])
```

`summary` carries the rows, accuracy, interval, and tables by virtue, layer and
tradition.

## Retries and resume

Provider calls retry on 429, 5xx and network errors, with backoff and
`Retry-After` respected. A failed call is recorded as an error, and the run
continues. Each result is appended as it arrives. Rerun into the same directory
to skip answered items and retry only the errors; use `--fresh` to start over.

`--json` prints the summary as JSON. `--retries` and `--timeout` set the HTTP
budget.

## Why it exists

The point is to give AI developers a test they can run against a model, and a
dataset a lab can cite. The hope is that a model which handles these situations
makes better decisions when it is deployed. That is the hypothesis, not a
finding. The corpus has no human labels yet.

## Data and licence

The data is CC BY-SA 4.0. The code is Apache-2.0. The bundled file is the
publishable release: entries whose sources are public domain, CC0, CC BY or
CC BY-SA. `research_only` and `unclear` sources are excluded.

The public pages, including the reviewer application, are at
https://keith448.github.io/virtuos-site/.
