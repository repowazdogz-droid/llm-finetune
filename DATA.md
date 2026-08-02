# Dataset: GSM8K

## What it is

8,792 grade-school math word problems, each paired with a natural-language
solution that ends in a single numeric answer. The fine-tuning task is: given
the problem, produce the reasoning and then the answer.

## Provenance

| field | value |
|---|---|
| name | GSM8K (Grade School Math 8K) |
| source repository | https://github.com/openai/grade-school-math |
| files used | `grade_school_math/data/train.jsonl`, `grade_school_math/data/test.jsonl` |
| publisher | OpenAI |
| licence | **MIT License**, Copyright (c) 2021 OpenAI |
| introduced in | Cobbe et al., *Training Verifiers to Solve Math Word Problems*, arXiv:2110.14168 |
| fetched | 2026-08-02 |

Downloaded from the publishing repository directly rather than from a mirror or
a dataset hub, so the provenance chain is one hop and the licence that governs
the files is the one in that repository. A copy of it is written to
`data/raw/LICENSE.gsm8k` by `data/download.sh`, so the terms travel with the
data.

### Integrity

| file | sha256 | records |
|---|---|---:|
| `train.jsonl` | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` | 7,473 |
| `test.jsonl` | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` | 1,319 |

7,473 + 1,319 = 8,792, consistent with the paper's description of the dataset
as ~8.5K problems.

### Citation verification

The reference above was resolved against the arXiv API on 2026-08-02, not taken
from recall:

| check | method | result |
|---|---|---|
| identifier resolves | arXiv API `id_list=2110.14168` | resolves to v2 |
| title | returned record | "Training Verifiers to Solve Math Word Problems" — matches |
| date | returned record | published 2021-10-27 |
| authors | returned record | 12 authors, first listed Karl Cobbe — matches |
| **content** | abstract read | states "we introduce GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems" — supports the specific claim made here |
| licence | fetched `LICENSE` from the repo | MIT, Copyright (c) 2021 OpenAI |

The content check matters separately from the existence check: citing a real
paper for a claim it does not make is the more common failure, and "this paper
introduced GSM8K" is a content claim.

## Why this dataset

**The licence is unambiguous and permissive.** MIT, from the publisher, with the
licence file adjacent to the data. No non-commercial clause to reason about, no
"research use only" ambiguity, no scraped-content provenance question.

**The metric is not a judgement call.** The answer is a number. Exact match after
numeric normalisation is unambiguous, deterministic, and needs no model to
score — which matters because an evaluation harness whose metric requires
another LLM inherits that model's failures and cannot be verified with stubs on
a laptop. ROUGE or an LLM-judge on free-form text would have made Floor B0's
gate untestable.

**The task is hard enough to show movement.** Small models score poorly on GSM8K
before fine-tuning, so there is headroom for a fine-tune to demonstrate
something rather than a ceiling it is already at.

**It is small.** 5 MB. The whole floor runs on a laptop in seconds, and the PC
phase starts from a `git clone` with no dataset work left.

### What is wrong with this choice, stated up front

GSM8K is public, widely mirrored, and predates the pretraining cutoff of
essentially every current base model. **A base model has very likely seen these
exact problems already.** That is a contamination problem this repository cannot
solve and cannot measure from the inside: `eval/memorisation_check.py` compares
generations against *our fine-tuning corpus*, which says nothing about what was
in the base model's pretraining data.

The practical consequence for B1/B2: a GSM8K score after fine-tuning is a
measure of "can this pipeline elicit the behaviour", not "did the model learn to
do arithmetic reasoning". Do not report it as the latter. If a claim about
genuine capability gain is needed later, it requires a held-out set constructed
after the base model's cutoff, which is out of scope here and should be said
plainly rather than papered over.

## Curation

Performed by `data/prepare.py`. The full ledger — every record that entered and
every one that did not — is in [`data/SPLIT_MANIFEST.md`](data/SPLIT_MANIFEST.md).

The honest summary is that **GSM8K arrived clean**. Every drop filter
(structural validity, missing `####` marker, unparseable answer, empty solution,
duplicate question) removed exactly **zero** records from both files. The filters
are kept because they are the guarantee that the data is well-formed — a filter
that fires zero times is still doing work, it is asserting a property — but this
was not a rescue job and describing it as heavy curation would be false.

Two transformations did change the data:

**1. Calculator markup stripped** (7,378 of 7,473 train records). GSM8K solutions
carry inline annotations from the original tooling:

```
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
```

The `<<48/2=24>>` is removed, leaving `Natalia sold 48/2 = 24 clips in May.`
This is a real decision, not tidying. Left in, the model would learn to emit
that markup, and every downstream consumer would then need to strip it — or
worse, the harness would score the markup as part of the answer. Removing it
here means the training target is the text we actually want generated.

**2. Exotic line breaks normalised** (1 train record, 2 characters). GSM8K
contains a U+2028 LINE SEPARATOR inside a question — train record 2230, "Clive
opens a box full of different colored balls". JSON does not escape U+2028, but
Python's `str.splitlines()` treats it as a line break, so writing the record out
with `ensure_ascii=False` and reading it back with `splitlines()` silently
splits one record into two unparseable halves.

This was found the hard way: the memorisation index crashed with
`JSONDecodeError: Unterminated string`. Three fixes, because one was not enough:
the characters are normalised to `\n` during cleaning, output is written with
`ensure_ascii=True` so the files are pure ASCII, and every JSONL reader in the
repo splits on `\n` rather than using `splitlines()`. A test asserts the last of
these.

## Splits

| split | n | purpose |
|---|---:|---|
| `train` | 6,973 | fine-tuning corpus for the PC phase |
| `eval` | 500 | held-out development set — what B1/B2 score against |
| `test_reserved` | 1,319 | GSM8K's official test file. Manifested, then untouched. |

Seed 20260802. The development eval set is carved out of GSM8K's **train** file,
not its test file. This is the load-bearing choice: if `eval` were the official
test set, every B1 decision made against it — hyperparameters, prompt format,
when to stop — would contaminate it, and the final number would be selection
noise wearing the clothes of a result. `test_reserved` exists so the PC phase
has a set no development decision has ever seen, and should be read once, at the
end, exactly as Project A treated the CIFAR-10 test set.

Contamination checks at generation time, all asserted in code:

| check | result |
|---|---:|
| eval items appearing in train | 0 |
| train/test question overlap | 0 |
| eval/test question overlap | 0 |

These are **exact** matches on whitespace- and case-normalised question text.
They catch duplication and copy-paste leakage. They do not catch a paraphrase, a
template instantiated twice with different numbers, or a semantically identical
problem — and GSM8K is human-written from templates in places, so some of that
is likely present. No semantic-duplicate detection is claimed.

## Reproducing

```bash
./data/download.sh                       # ~5 MB, idempotent, checksums printed
.venv/bin/python -m data.prepare         # must reproduce every digest in the manifest
```

Both are deterministic. Regenerating on a clean clone must reproduce every
sha256 in `data/SPLIT_MANIFEST.md`; if it does not, the data underneath changed
and any number measured against the old split is void.
