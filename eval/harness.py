"""Model-agnostic evaluation harness for GSM8K-style tasks.

The contract is one function:

    generate(prompts: list[str]) -> list[str]

That is the entire coupling to whatever is being evaluated. The harness never
imports an inference library, never touches a GPU, and cannot tell a fine-tuned
7B model from the stub in `tests/`. This is what makes it testable to
completion on a laptop and runnable unchanged on the PC.

## What is measured

**Exact match** on the normalised final answer, using the *same* normaliser
that `data/prepare.py` used on the references. Comparing "1,000" against "1000"
and calling it a miss would be measuring the formatter, not the model.

**Format compliance** — the fraction of generations from which an answer could
be extracted at all. Reported separately from accuracy on purpose: a model that
reasons correctly but ignores the output format, and a model that emits perfect
format and wrong arithmetic, are different failures and a single accuracy number
hides which one you have.

**Extraction strategy breakdown** — the harness will fall back to "last number
in the output" when the `####` marker is absent. That fallback is *generous*: it
can credit a model that stumbled onto the right number without following the
requested format, and it can pick up a number from a trailing sentence that was
not the model's answer. It is included because the alternative — scoring all
non-compliant output as wrong — conflates format failure with reasoning failure.
The rate at which it fires is reported so a reader can discount accordingly. If
`strict` is set, the fallback is disabled entirely.

**Accuracy by reasoning depth** — broken out by the reference solution's step
count, because an aggregate hides that multi-step problems are the hard ones and
a model can move the headline number while getting no better at them.

**A 95% confidence interval** on the accuracy, computed with the Wilson score
interval in pure Python. On a 500-item eval set the standard error is roughly
2 points, so differences smaller than about 5 points between two runs are not
distinguishable, and the interval is here to make that visible rather than
leaving it to be inferred.

Generation failures are caught per item and recorded as errors rather than
crashing the run; a report that says "37 of 500 generations raised" is more
useful than a traceback partway through an expensive job.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"

GenerateFn = Callable[[Sequence[str]], Sequence[str]]

DEFAULT_PROMPT = (
    "Solve the following grade school math problem. "
    "Show your reasoning, then give the final numeric answer on its own line "
    "after '#### '.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)

MARKER = "####"
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


# ---------------------------------------------------------------------------
# normalisation and extraction
# ---------------------------------------------------------------------------

def normalise_answer(raw: str) -> str | None:
    """Canonical numeric form. Must stay identical to data/prepare.py's version.

    A test asserts the two agree; if they ever drift, every accuracy number this
    harness has produced becomes uninterpretable, so the duplication is checked
    rather than assumed.
    """
    s = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".").strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    if value == int(value):
        return str(int(value))
    return repr(value)


def extract_answer(text: str, strict: bool = False) -> tuple[str | None, str]:
    """Pull a final answer out of free-form generation.

    Returns (normalised answer or None, strategy) where strategy is one of
    "marker", "last_number", or "none". The strategy is returned rather than
    swallowed because the aggregate rate of each is a reported metric.
    """
    if MARKER in text:
        tail = text.split(MARKER)[-1]
        m = NUMBER_RE.search(tail)
        if m:
            got = normalise_answer(m.group(0))
            if got is not None:
                return got, "marker"

    if strict:
        return None, "none"

    matches = NUMBER_RE.findall(text)
    for candidate in reversed(matches):
        got = normalise_answer(candidate)
        if got is not None:
            return got, "last_number"

    return None, "none"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. No scipy required.

    Wilson rather than the normal approximation because it stays inside [0, 1]
    and behaves sensibly at small n and at proportions near 0 or 1 — which is
    exactly where an early fine-tuning run will be.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Read JSONL, splitting on newlines only.

    Deliberately not `.splitlines()`. That method also breaks on U+2028,
    U+2029, U+0085, \\x0b and \\x0c, none of which JSON escapes — so a record
    containing one gets silently split into two unparseable halves. GSM8K
    contains a U+2028 inside a question, which is how this was found.
    """
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").split("\n")
        if l.strip()
    ]


def load_eval_set(limit: int | None = None) -> list[dict]:
    rows = load_jsonl(PROCESSED / "eval.jsonl")
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------

def run_eval(
    generate: GenerateFn,
    examples: Iterable[dict],
    *,
    prompt_template: str = DEFAULT_PROMPT,
    strict: bool = False,
    batch_size: int = 16,
    report_dir: Path | None = None,
    run_name: str = "eval",
    model_description: str = "(unspecified)",
) -> dict:
    """Score `generate` against `examples`. Returns the full result dict.

    `examples` must carry `question` and `answer`; `n_steps` is used for the
    depth breakdown when present.
    """
    examples = list(examples)
    if not examples:
        raise ValueError("no examples to evaluate")
    for i, ex in enumerate(examples):
        if "question" not in ex or "answer" not in ex:
            raise ValueError(f"example {i} lacks 'question' or 'answer'")

    prompts = [prompt_template.format(question=ex["question"]) for ex in examples]

    outputs: list[str] = []
    errors: list[dict] = []
    t0 = time.time()

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        try:
            produced = list(generate(chunk))
        except Exception as exc:  # a whole batch failed
            errors.append({"batch_start": start, "error": f"{type(exc).__name__}: {exc}"})
            produced = [""] * len(chunk)

        if len(produced) != len(chunk):
            raise ValueError(
                f"generate() returned {len(produced)} outputs for {len(chunk)} prompts; "
                "the contract is one output per prompt, in order"
            )
        outputs.extend(str(o) for o in produced)

    elapsed = time.time() - t0

    # --- score ---
    records = []
    strategies = Counter()
    by_steps_correct = defaultdict(int)
    by_steps_total = defaultdict(int)

    for ex, prompt, out in zip(examples, prompts, outputs):
        pred, strategy = extract_answer(out, strict=strict)
        ref = normalise_answer(ex["answer"])
        correct = pred is not None and pred == ref
        strategies[strategy] += 1

        steps = ex.get("n_steps")
        if steps is not None:
            by_steps_total[steps] += 1
            by_steps_correct[steps] += int(correct)

        records.append(
            {
                "question": ex["question"],
                "reference": ref,
                "prediction": pred,
                "strategy": strategy,
                "correct": bool(correct),
                "n_steps": steps,
                "output": out,
            }
        )

    n = len(records)
    k = sum(r["correct"] for r in records)
    extracted = n - strategies["none"]
    lo, hi = wilson_interval(k, n)

    result = {
        "run_name": run_name,
        "model_description": model_description,
        "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n": n,
        "n_correct": k,
        "accuracy": k / n,
        "accuracy_ci95": [lo, hi],
        "format_compliance": extracted / n,
        "extraction_strategies": dict(strategies),
        "strict_extraction": strict,
        "accuracy_by_steps": {
            str(s): {
                "n": by_steps_total[s],
                "correct": by_steps_correct[s],
                "accuracy": by_steps_correct[s] / by_steps_total[s],
            }
            for s in sorted(by_steps_total)
        },
        "generation_errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "prompt_template": prompt_template,
        "records": records,
    }

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{run_name}.json").write_text(json.dumps(result, indent=2))
        (report_dir / f"{run_name}.md").write_text(format_report(result))

    return result


def format_report(r: dict) -> str:
    lo, hi = r["accuracy_ci95"]
    lines = [
        f"# Evaluation report — {r['run_name']}",
        "",
        f"- model: {r['model_description']}",
        f"- scored: {r['scored_at']}",
        f"- examples: {r['n']}",
        f"- wall clock: {r['elapsed_sec']}s",
        f"- extraction: {'strict (marker only)' if r['strict_extraction'] else 'marker, falling back to last number'}",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| exact-match accuracy | **{r['accuracy'] * 100:.2f}%** ({r['n_correct']}/{r['n']}) |",
        f"| 95% CI (Wilson) | [{lo * 100:.2f}, {hi * 100:.2f}] |",
        f"| format compliance | {r['format_compliance'] * 100:.2f}% |",
        "",
        "## How answers were extracted",
        "",
        "| strategy | count | share |",
        "|---|---:|---:|",
    ]
    for s, c in sorted(r["extraction_strategies"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{s}` | {c} | {c / r['n'] * 100:.1f}% |")

    fallback = r["extraction_strategies"].get("last_number", 0)
    if fallback:
        lines += [
            "",
            f"**{fallback} of {r['n']} answers ({fallback / r['n'] * 100:.1f}%) came from the",
            "last-number fallback rather than the requested `####` marker.** That",
            "fallback is generous — it can credit output that never followed the",
            "format, and it can pick up a number that was not the intended answer.",
            "Accuracy should be read as an upper bound to the extent this rate is",
            "high. Re-run with `strict=True` for the lower bound.",
        ]

    if r["accuracy_by_steps"]:
        lines += [
            "",
            "## Accuracy by reasoning depth",
            "",
            "Steps = lines in the reference solution, a proxy for difficulty.",
            "",
            "| steps | n | correct | accuracy |",
            "|---:|---:|---:|---:|",
        ]
        for s, m in r["accuracy_by_steps"].items():
            lines.append(
                f"| {s} | {m['n']} | {m['correct']} | {m['accuracy'] * 100:.1f}% |"
            )

    if r["generation_errors"]:
        lines += [
            "",
            "## Generation errors",
            "",
            f"**{len(r['generation_errors'])} batches raised.** Their prompts were scored",
            "as empty output, so accuracy is depressed by whatever those would have",
            "scored. This is not a clean run.",
            "",
        ]
        for e in r["generation_errors"][:10]:
            lines.append(f"- batch at index {e['batch_start']}: `{e['error']}`")

    lines += [
        "",
        "## Scope",
        "",
        f"This is exact-match accuracy on {r['n']} held-out GSM8K problems under one",
        "prompt template, with answers extracted by the rule above. It is not a",
        "measure of mathematical ability, of reasoning quality, or of performance",
        "on any other problem distribution. The reasoning text is not scored at",
        "all — only the final number — so a model that reaches the right answer",
        "by invalid reasoning scores identically to one that does not.",
        "",
        "A different prompt template would produce a different number from the",
        "same model. The template is recorded in the JSON alongside the result so",
        "that two reports can be checked for comparability before being compared.",
        "",
    ]
    return "\n".join(lines)
