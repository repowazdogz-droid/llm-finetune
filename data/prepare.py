"""Curate GSM8K and split it into train / held-out eval with a fixed seed.

What this produces, and the reasoning behind the shape:

  train  — the fine-tuning set the PC phase (B1) consumes.
  eval   — a held-out development set, carved out of GSM8K's *train* file.
           This is what `eval/harness.py` scores against during B1/B2.
  test   — GSM8K's official test file, cleaned and manifested but otherwise
           untouched. Nothing in this repo reads it. It exists so that the PC
           phase has a final set that no development decision has ever seen,
           the same discipline Project A used.

Carving the development eval out of GSM8K's train file rather than using the
official test file is the load-bearing choice. If `eval` were the official
test set, then every B1 decision made against it — hyperparameters, prompt
format, when to stop — would contaminate it, and the final number would be
selection noise dressed up as a result.

Cleaning steps, each of which drops or rewrites records and each of which is
counted in the manifest:

  1. Structural validity: both fields present, non-empty after strip.
  2. Final-answer marker: the solution must contain `#### <answer>`.
  3. Parseable answer: the text after `####` must reduce to a number after
     removing thousands separators, currency symbols and trailing punctuation.
  4. Calculator markup removal: GSM8K solutions carry inline annotations like
     `<<48/2=24>>` that were used by the original tooling. They are stripped.
     This is a real decision, not tidying — see DATA.md for why.
  5. Exact deduplication on the normalised question.
  6. Cross-split leakage removal: any eval item whose normalised question also
     appears in train is dropped from eval, not from train.

Usage:  .venv/bin/python -m data.prepare
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 20260802
N_EVAL = 500

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"
MANIFEST = REPO / "data" / "SPLIT_MANIFEST.md"

CALC_RE = re.compile(r"<<[^>]*>>")
MARKER = "####"

# Unicode characters that Python's str.splitlines() treats as line breaks but
# that JSON does not escape. GSM8K contains U+2028 inside at least one question
# (train record 2230, "Clive opens a box full of different colored balls").
# Written raw into JSONL, it silently splits one record into two for any reader
# using splitlines() — which is how this was found: the memorisation index
# crashed with "Unterminated string". They are normalised to "\n" here and the
# count is reported in the manifest.
EXOTIC_BREAKS = {
    " ": "\n",  # LINE SEPARATOR
    " ": "\n",  # PARAGRAPH SEPARATOR
    "": "\n",  # NEXT LINE
    "\x0b": "\n",    # VERTICAL TAB
    "\x0c": "\n",    # FORM FEED
}


def normalise_linebreaks(text: str) -> tuple[str, int]:
    """Replace exotic line-break characters with \\n. Returns (text, n_replaced)."""
    count = sum(text.count(ch) for ch in EXOTIC_BREAKS)
    for ch, repl in EXOTIC_BREAKS.items():
        text = text.replace(ch, repl)
    return text, count


def read_lines(path: Path) -> list[str]:
    """Split a JSONL file on newlines only.

    Deliberately not `.splitlines()`: that also breaks on U+2028 and friends,
    which is exactly the bug above. A JSONL reader must split on the delimiter
    JSON actually uses.
    """
    return path.read_text(encoding="utf-8").split("\n")


# ---------------------------------------------------------------------------
# cleaning primitives
# ---------------------------------------------------------------------------

def strip_calculator_markup(solution: str) -> str:
    """Remove GSM8K's inline `<<expr=value>>` annotations."""
    return CALC_RE.sub("", solution)


def normalise_answer(raw: str) -> str | None:
    """Reduce a final-answer string to a canonical numeric form.

    Returns None if it does not parse as a number, which is a drop signal.
    The same function is used by the evaluation harness to score model output,
    so a prediction and a reference are always compared after identical
    normalisation — scoring "1,000" as wrong against "1000" would be measuring
    the formatter, not the model.
    """
    s = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".").strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    # Integers keep an integer form so "72" and "72.0" compare equal.
    if value == int(value):
        return str(int(value))
    return repr(value)


def normalise_question(q: str) -> str:
    """Whitespace- and case-normalised question, for dedup and leakage checks."""
    return " ".join(q.lower().split())


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_records(records: list[dict]) -> str:
    """Digest of a split's content, order-independent.

    Order-independent on purpose: the digest should identify *which examples*
    are in a split, not the order a particular run happened to emit them, so
    that a reordering does not look like a data change.
    """
    h = hashlib.sha256()
    for digest in sorted(sha256_text(r["question"] + "\x00" + r["answer"]) for r in records):
        h.update(digest.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def load_and_clean(path: Path) -> tuple[list[dict], Counter]:
    counts = Counter()
    cleaned = []
    seen_questions: set[str] = set()

    for line in read_lines(path):
        counts["read"] += 1
        if not line.strip():
            counts["dropped_blank_line"] += 1
            continue
        row = json.loads(line)

        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            counts["dropped_empty_field"] += 1
            continue

        q, nq = normalise_linebreaks(q)
        a, na = normalise_linebreaks(a)
        if nq + na:
            counts["exotic_linebreaks_normalised"] += nq + na
            counts["records_with_exotic_linebreaks"] += 1

        if MARKER not in a:
            counts["dropped_no_marker"] += 1
            continue

        solution, _, final = a.partition(MARKER)
        answer = normalise_answer(final)
        if answer is None:
            counts["dropped_unparseable_answer"] += 1
            continue

        solution = strip_calculator_markup(solution).strip()
        if not solution:
            counts["dropped_empty_solution"] += 1
            continue
        counts["calc_markup_stripped"] += int(bool(CALC_RE.search(a)))

        key = normalise_question(q)
        if key in seen_questions:
            counts["dropped_duplicate_question"] += 1
            continue
        seen_questions.add(key)

        cleaned.append(
            {
                "question": q,
                "solution": solution,
                "answer": answer,
                # `answer` is the scoreable target; `solution` is the reasoning
                # the model is trained to produce before it.
                "n_steps": len([ln for ln in solution.splitlines() if ln.strip()]),
            }
        )
        counts["kept"] += 1

    return cleaned, counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            # ensure_ascii=True escapes every non-ASCII character, including any
            # exotic line break that slipped past normalisation. The written file
            # is then pure ASCII and cannot be mis-split by any reader.
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "(not a git checkout)"


def main() -> dict:
    train_pool, train_counts = load_and_clean(RAW / "train.jsonl")
    test_rows, test_counts = load_and_clean(RAW / "test.jsonl")

    # --- fixed-seed split of the cleaned train pool ---
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_pool))
    eval_idx = set(perm[:N_EVAL].tolist())

    train_rows = [r for i, r in enumerate(train_pool) if i not in eval_idx]
    eval_rows = [r for i, r in enumerate(train_pool) if i in eval_idx]

    # --- leakage removal: no eval question may appear in train ---
    train_keys = {normalise_question(r["question"]) for r in train_rows}
    before = len(eval_rows)
    eval_rows = [r for r in eval_rows if normalise_question(r["question"]) not in train_keys]
    leaked_removed = before - len(eval_rows)

    # --- and the same check against the reserved test file ---
    test_keys = {normalise_question(r["question"]) for r in test_rows}
    train_test_overlap = len(train_keys & test_keys)
    eval_test_overlap = len(
        {normalise_question(r["question"]) for r in eval_rows} & test_keys
    )

    # These must hold or the splits are not splits.
    assert not (train_keys & {normalise_question(r["question"]) for r in eval_rows}), \
        "train/eval question overlap survived the leakage filter"
    assert len(train_rows) + len(eval_rows) + leaked_removed == len(train_pool)

    write_jsonl(PROCESSED / "train.jsonl", train_rows)
    write_jsonl(PROCESSED / "eval.jsonl", eval_rows)
    write_jsonl(PROCESSED / "test_reserved.jsonl", test_rows)

    splits = {
        "train": train_rows,
        "eval": eval_rows,
        "test_reserved": test_rows,
    }
    digests = {name: sha256_records(rows) for name, rows in splits.items()}

    summary = {
        "seed": SEED,
        "n_eval_requested": N_EVAL,
        "counts_train_file": dict(train_counts),
        "counts_test_file": dict(test_counts),
        "sizes": {k: len(v) for k, v in splits.items()},
        "digests": digests,
        "leaked_eval_items_removed": leaked_removed,
        "train_test_question_overlap": train_test_overlap,
        "eval_test_question_overlap": eval_test_overlap,
        "step_distribution": {
            k: dict(Counter(r["n_steps"] for r in v).most_common()) for k, v in splits.items()
        },
    }

    (PROCESSED / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    _write_manifest(summary)

    print(f"seed                    : {SEED}")
    for k, v in summary["sizes"].items():
        print(f"{k:<15} n={v:<6} sha256={digests[k][:16]}…")
    print(f"leaked eval removed     : {leaked_removed}")
    print(f"train/test overlap      : {train_test_overlap}")
    print(f"eval/test overlap       : {eval_test_overlap}")
    print(f"\nwrote {PROCESSED}/ and {MANIFEST}")
    return summary


def _write_manifest(s: dict) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    tc, sc = s["counts_train_file"], s["counts_test_file"]

    lines = [
        "# GSM8K split manifest",
        "",
        "Generated by `data/prepare.py`. Regenerating on a clean clone must",
        "reproduce every digest below. If it does not, the data underneath",
        "changed and any number measured against the old split is void.",
        "",
        f"- generated: {now}",
        f"- git commit: `{git_commit()}`",
        f"- RNG: `numpy.random.default_rng(seed)`, **seed = {s['seed']}**",
        f"- held-out eval size requested: {s['n_eval_requested']}",
        "- provenance and licence: see [DATA.md](../DATA.md)",
        "",
        "## Splits",
        "",
        "| split | n | sha256 (order-independent, over question+answer) |",
        "|---|---:|---|",
    ]
    for name, n in s["sizes"].items():
        lines.append(f"| `{name}` | {n} | `{s['digests'][name]}` |")

    lines += [
        "",
        "`test_reserved` is GSM8K's official test file. It is cleaned and",
        "manifested here so that its identity is pinned, and then left alone —",
        "nothing in this repo reads it, and the PC phase should treat it the way",
        "Project A treated the CIFAR-10 test set: read once, at the end.",
        "",
        "## Cleaning ledger",
        "",
        "Every record that entered and every record that did not, per source file.",
        "",
        "| step | train.jsonl | test.jsonl |",
        "|---|---:|---:|",
        f"| lines read | {tc.get('read', 0)} | {sc.get('read', 0)} |",
        f"| dropped: blank line | {tc.get('dropped_blank_line', 0)} | {sc.get('dropped_blank_line', 0)} |",
        f"| dropped: empty question or answer | {tc.get('dropped_empty_field', 0)} | {sc.get('dropped_empty_field', 0)} |",
        f"| dropped: no `####` marker | {tc.get('dropped_no_marker', 0)} | {sc.get('dropped_no_marker', 0)} |",
        f"| dropped: unparseable final answer | {tc.get('dropped_unparseable_answer', 0)} | {sc.get('dropped_unparseable_answer', 0)} |",
        f"| dropped: empty solution after cleaning | {tc.get('dropped_empty_solution', 0)} | {sc.get('dropped_empty_solution', 0)} |",
        f"| dropped: duplicate question | {tc.get('dropped_duplicate_question', 0)} | {sc.get('dropped_duplicate_question', 0)} |",
        f"| **kept** | **{tc.get('kept', 0)}** | **{sc.get('kept', 0)}** |",
        f"| (of kept) had calculator markup stripped | {tc.get('calc_markup_stripped', 0)} | {sc.get('calc_markup_stripped', 0)} |",
        f"| (of kept) records with exotic line breaks normalised | "
        f"{tc.get('records_with_exotic_linebreaks', 0)} | {sc.get('records_with_exotic_linebreaks', 0)} |",
        f"| (of kept) exotic line-break characters replaced | "
        f"{tc.get('exotic_linebreaks_normalised', 0)} | {sc.get('exotic_linebreaks_normalised', 0)} |",
        "",
        "## Contamination checks",
        "",
        "| check | result |",
        "|---|---:|",
        f"| eval items removed for appearing in train | {s['leaked_eval_items_removed']} |",
        f"| train/test question overlap | {s['train_test_question_overlap']} |",
        f"| eval/test question overlap | {s['eval_test_question_overlap']} |",
        "",
        "Train and eval share **no** normalised question: this is asserted in",
        "`data/prepare.py` at generation time, so a manifest cannot be produced",
        "from overlapping splits.",
        "",
        "## What these checks do not cover",
        "",
        "The overlap checks above are **exact** matches on whitespace- and",
        "case-normalised question text. They catch duplication and copy-paste",
        "leakage. They do not catch a paraphrase, a problem with the same",
        "structure and different numbers, or a template instantiated twice — and",
        "GSM8K is human-written from templates in places, so some of that is",
        "likely present. `eval/memorisation_check.py` covers the n-gram-overlap",
        "direction at generation time; neither is a semantic-duplicate detector,",
        "and no claim of one is made here.",
        "",
        "Nothing here addresses the larger contamination question for the PC",
        "phase: GSM8K is public and predates most current pretraining corpora, so",
        "a base model may well have seen these exact problems before any",
        "fine-tuning happens. That is a property of the base model, not of this",
        "split, and it cannot be measured from inside this repo.",
        "",
    ]
    MANIFEST.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
