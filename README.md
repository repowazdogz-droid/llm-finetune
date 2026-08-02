# llm-finetune — Floor B0

The CPU-only preparation floor for a fine-tuning project. It builds the dataset
and the evaluation harness that the GPU phase consumes, and it deliberately
stops there.

**Status: B0 complete. B1/B2 not started.** This repository loads no model
weights, runs no fine-tuning, and has no CUDA dependency — those are the next
floor's job, on different hardware. The constraint is enforced by a test, not
by intention: `tests/test_no_gpu_dependency.py` walks every module's imports and
fails if any of torch, transformers, peft, vllm and friends appear.

## What B0 delivers

| # | Deliverable | Artifact | Gate |
|---|---|---|---|
| 1 | A defensible dataset with recorded provenance | [`DATA.md`](DATA.md) | MIT licence verified from the publishing repo; the introducing paper resolved against the arXiv API, abstract read, not recalled |
| 2 | A cleaned, fixed-seed, non-overlapping split | [`data/SPLIT_MANIFEST.md`](data/SPLIT_MANIFEST.md) | seed recorded; sha256 per split; 0 train/eval overlap, asserted at generation time |
| 3 | A model-agnostic evaluation harness | `eval/harness.py` | scores exactly 100% for the perfect stub and 0% for the strict-mode unformatted stub |
| 4 | A verbatim-regurgitation detector | `eval/memorisation_check.py` | flags 29/30 regurgitated training solutions; falsely flags 0/300 held-out ones |
| 5 | No GPU or model dependency | `tests/test_no_gpu_dependency.py` | AST import scan over the whole tree, with a negative control |

**52 tests, all passing, no GPU.**

## The dataset in one line

GSM8K (OpenAI, MIT licence): 8,792 grade-school math word problems, split
6,973 train / 500 held-out eval / 1,319 reserved test. Full provenance,
licence, curation ledger and the honest case *against* this choice are in
[`DATA.md`](DATA.md) — most importantly that GSM8K predates every current base
model's pretraining cutoff, so a score after fine-tuning measures "can the
pipeline elicit this", not "did the model learn arithmetic reasoning".

## The harness contract

One function. That is the entire coupling to whatever is being evaluated:

```python
def generate(prompts: Sequence[str]) -> Sequence[str]: ...
```

```python
from eval.harness import load_eval_set, run_eval

result = run_eval(
    my_generate_fn,                    # anything satisfying the contract
    load_eval_set(),                   # the 500 held-out problems
    report_dir="runs/baseline",
    run_name="qwen3b_lora_e1",
    model_description="Qwen-3B + LoRA r=16, epoch 1",
)
print(result["accuracy"], result["accuracy_ci95"])
```

The harness never imports an inference library, so it is testable to completion
on a laptop and runs unchanged on the PC. It reports:

- **exact-match accuracy** on the normalised final answer, with a Wilson 95% CI
  (pure Python — no scipy)
- **format compliance**, separately from accuracy, because a model that reasons
  correctly but ignores the output format and one that formats perfectly and
  computes wrong are different failures
- **extraction-strategy breakdown** — the harness falls back to "last number in
  the output" when the `####` marker is missing, and that fallback is generous,
  so the rate at which it fires is reported and the report says plainly that
  accuracy is an upper bound to the extent it is high. `strict=True` disables it
  for the lower bound.
- **accuracy by reasoning depth**, because an aggregate hides that multi-step
  problems are the hard ones

Generation failures are caught per batch and recorded, not raised — a report
saying "37 of 500 generations raised" beats a traceback partway through an
expensive job.

## The memorisation check

Answers one narrow question: *does a generated output contain spans that appear
verbatim in the fine-tuning corpus, and how long are they?*

The primary metric is the **longest matching span**, not n-gram coverage.
Incidental overlap produces many short matches; copying produces one long one.
On GSM8K's short formulaic sentences the two look similar under a coverage
metric and completely different under a longest-span metric.

The threshold is calibrated against this corpus rather than asserted:

```
$ .venv/bin/python -m eval.memorisation_check --calibrate

held-out (true negatives)  p99 =   16 tokens, max = 16 tokens
in-corpus (true positives) p50 =   61 tokens, min = 16 tokens
threshold 25 sits BETWEEN the two — usable
  false positives (held-out flagged) : 0/300  (0.0%)
  false negatives (in-corpus missed) : 14/300  (4.7%)
```

That 4.7% is the honest limit and is why the test asserts 28/30 rather than
30/30: very short training solutions contain no 25-token span, so a model
reproducing one verbatim would not be flagged. Demanding 30/30 would demand the
detector exceed its own measured accuracy.

**What a clean result does and does not mean.** It is evidence about verbatim
copying only. A paraphrase, the same reasoning in different words, or a memorised
answer with freshly generated reasoning would all pass. The report refuses to
phrase a zero result as "no memorisation", and a test asserts that it refuses.

## Reproducing

```bash
git clone https://github.com/repowazdogz-droid/llm-finetune.git
cd llm-finetune

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt   # numpy + pytest. That is all.

./data/download.sh                          # ~5 MB from the OpenAI repo
.venv/bin/python -m data.prepare            # must reproduce the manifest digests
.venv/bin/python -m pytest tests/ -q        # the B0 gate: 52 tests
.venv/bin/python -m eval.memorisation_check --calibrate
```

## Layout

```
data/
  download.sh          fetch GSM8K from the publishing repo, checksummed
  prepare.py           clean, fixed-seed split, contamination checks, manifest
  SPLIT_MANIFEST.md    seed, per-split sha256, full cleaning ledger
eval/
  harness.py           model-agnostic evaluation; the generate() contract
  memorisation_check.py  n-gram verbatim-overlap detector + calibration
  stubs.py             stub generators — shipped, so the PC can smoke-test first
tests/                 the B0 gate
DATA.md                provenance, licence, curation, and the case against GSM8K
```

`eval/stubs.py` is shipped rather than test-only on purpose: the PC phase should
run `run_eval(PerfectGenerator(), ...)` and confirm it scores 100% *before*
spending an hour loading weights.

## Handover to the PC phase

A `git clone` plus `./data/download.sh` plus `python -m data.prepare` leaves zero
dataset work. B1 starts by writing a `generate` function around whatever model it
loads and handing it to `run_eval`.

Three things B1/B2 should carry forward rather than rediscover:

1. **`test_reserved` is read once, at the end.** Every development decision goes
   against `eval`. Project A's `TEST_SET_ACCESS.log` pattern is worth copying.
2. **Report format compliance next to accuracy**, and check the last-number
   fallback rate before quoting a headline number.
3. **The GSM8K contamination caveat in `DATA.md` applies to every number B1/B2
   produces.** It does not go away by being unmentioned.

## Licence

MIT for the code — see [LICENSE](LICENSE). GSM8K is not redistributed here; it is
downloaded from OpenAI's repository under its own MIT licence, as recorded in
[NOTICE](NOTICE) and [DATA.md](DATA.md).
